from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from src.memory.entities.PromotionRequest import (
    CriticRecommendation,
    PromotionRequest,
    PromotionStatus,
)


# ---------------------------------------------------------------------------
# Dependency protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class AgentProtocol(Protocol):
    agent_id: str
    role: Any


@runtime_checkable
class MemoryStoreProtocol(Protocol):
    """
    Read-only memory access required by PromotionService.

    ACL persistence is owned by MemoryAccessPolicyService.
    """

    def get_by_id(
        self,
        memory_id: str,
    ) -> Any:
        ...


@runtime_checkable
class PromotionRequestStoreProtocol(Protocol):
    def get(
        self,
        request_id: str,
    ) -> PromotionRequest:
        ...

    def save(
        self,
        request: PromotionRequest,
    ) -> PromotionRequest:
        ...


@runtime_checkable
class MemoryAccessPolicyServiceProtocol(
    Protocol
):
    """
    Access-policy operations required by PromotionService.

    PromotionService manages request state, while the access-policy
    service is the only component allowed to modify memory ACLs.
    """

    def grant_read_access(
        self,
        *,
        governance_actor: AgentProtocol,
        memory_id: str,
        agent_ids: list[str],
        reason: str | None = None,
    ) -> Any:
        ...

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PromotionServiceError(Exception):
    """
    Base exception for promotion operations.
    """


class PromotionRequestNotFoundError(
    PromotionServiceError
):
    pass


class MemoryNotFoundError(
    PromotionServiceError
):
    pass


class InvalidPromotionActorError(
    PromotionServiceError
):
    pass


class AccessAlreadyGrantedError(
    PromotionServiceError
):
    pass


class PromotionStateError(
    PromotionServiceError
):
    pass


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class PromotionService:
    """
    Application service for persistent cross-agent memory sharing.

    Lifecycle:

        requester submits request
            -> Critic reviews
            -> Coordinator approves or rejects
            -> approved memory becomes shared
            -> requester is permanently added to readable_by

    The memory owner does not change.
    """

    def __init__(
            self,
            *,
            memory_store: MemoryStoreProtocol,
            request_store: PromotionRequestStoreProtocol,
            access_policy_service: (
                    MemoryAccessPolicyServiceProtocol
            ),
    ) -> None:
        self.memory_store = memory_store
        self.request_store = request_store
        self.access_policy_service = (
            access_policy_service
        )

    # ------------------------------------------------------------------
    # Request submission
    # ------------------------------------------------------------------

    def submit_promotion_request(
        self,
        *,
        requester: AgentProtocol,
        memory_id: str,
        reason: str,
        task_id: str | None = None,
    ) -> PromotionRequest:
        """
        Submit a cross-agent memory access request.
        """
        requester_id = self._agent_id(
            requester
        )
        memory = self._get_memory(memory_id)

        owner_id = self._required_memory_text(
            memory,
            "owner_agent_id",
        )

        if requester_id == owner_id:
            raise PromotionServiceError(
                "The owner already has access to its own "
                "memory and cannot request promotion."
            )

        if self._can_read(
            memory,
            requester_id,
        ):
            raise AccessAlreadyGrantedError(
                f"{requester_id!r} already has access to "
                f"memory {memory_id!r}."
            )

        request = PromotionRequest(
            memory_id=self._required_memory_text(
                memory,
                "memory_id",
            ),
            owner_agent_id=owner_id,
            requester_agent_id=requester_id,
            task_id=task_id,
            reason=reason,
        )

        return self.request_store.save(request)

    # ------------------------------------------------------------------
    # Critic review
    # ------------------------------------------------------------------

    def review_promotion_request(
        self,
        *,
        critic: AgentProtocol,
        request_id: str,
        recommendation: (
            CriticRecommendation | str
        ),
        comment: str | None = None,
    ) -> PromotionRequest:
        """
        Record the Critic's recommendation.

        This method does not modify the memory.
        """
        self._require_role(
            critic,
            expected_role="critic",
        )

        request = self._get_request(request_id)

        try:
            request.record_critic_review(
                critic_agent_id=self._agent_id(
                    critic
                ),
                recommendation=recommendation,
                comment=comment,
            )
        except ValueError as error:
            raise PromotionStateError(
                str(error)
            ) from error

        return self.request_store.save(request)

    # ------------------------------------------------------------------
    # Governed approval
    # ------------------------------------------------------------------

    def approve_promotion_request(
            self,
            *,
            coordinator: AgentProtocol,
            request_id: str,
            comment: str | None = None,
            require_critic_review: bool = True,
    ) -> PromotionRequest:
        """
        Approve an access request and grant the requester read access.

        PromotionService manages the request lifecycle. All memory ACL
        changes are delegated to MemoryAccessPolicyService.
        """
        self._require_role(
            coordinator,
            expected_role="coordinator",
        )

        current_request = self._get_request(
            request_id
        )

        # Preserve the original request so it can be restored if the
        # access-policy operation fails.
        original_request = (
            current_request.model_copy(
                deep=True
            )
        )

        approved_request = (
            current_request.model_copy(
                deep=True
            )
        )

        try:
            approved_request.approve(
                coordinator_agent_id=(
                    self._agent_id(coordinator)
                ),
                comment=comment,
                require_critic_review=(
                    require_critic_review
                ),
            )
        except ValueError as error:
            raise PromotionStateError(
                str(error)
            ) from error

        memory = self._get_memory(
            approved_request.memory_id
        )

        self._validate_request_against_memory(
            request=approved_request,
            memory=memory,
        )

        # Persist the approved request first. If the subsequent ACL
        # operation fails, restore the original request state.
        try:
            saved_request = (
                self.request_store.save(
                    approved_request
                )
            )
        except Exception as error:
            raise PromotionServiceError(
                "Failed to persist the approved "
                "access request."
            ) from error

        try:
            (
                self.access_policy_service
                .grant_read_access(
                    governance_actor=coordinator,
                    memory_id=(
                        saved_request.memory_id
                    ),
                    agent_ids=[
                        saved_request
                        .requester_agent_id
                    ],
                    reason=(
                            comment
                            or (
                                "Access granted through "
                                f"request "
                                f"{saved_request.request_id}."
                            )
                    ),
                )
            )
        except Exception as error:
            try:
                self.request_store.save(
                    original_request
                )
            except Exception as rollback_error:
                raise PromotionServiceError(
                    "Granting memory access failed, "
                    "and restoring the original "
                    "request state also failed: "
                    f"{rollback_error}"
                ) from error

            raise PromotionServiceError(
                "Granting memory access failed. "
                "The access request was restored "
                "to its previous state."
            ) from error

        return saved_request

    def approve_ungoverned_request(
        self,
        *,
        coordinator: AgentProtocol,
        request_id: str,
        comment: str | None = None,
    ) -> PromotionRequest:
        """
        Directly approve a request without Critic review.

        This is intended only for the ungoverned_shared baseline.
        """
        return self.approve_promotion_request(
            coordinator=coordinator,
            request_id=request_id,
            comment=comment,
            require_critic_review=False,
        )

    # ------------------------------------------------------------------
    # Rejection
    # ------------------------------------------------------------------

    def reject_promotion_request(
        self,
        *,
        coordinator: AgentProtocol,
        request_id: str,
        comment: str | None = None,
        require_critic_review: bool = True,
    ) -> PromotionRequest:
        """
        Reject a request.

        Rejection does not modify the memory or its ACL.
        """
        self._require_role(
            coordinator,
            expected_role="coordinator",
        )

        request = self._get_request(request_id)

        try:
            request.reject(
                coordinator_agent_id=self._agent_id(
                    coordinator
                ),
                comment=comment,
                require_critic_review=(
                    require_critic_review
                ),
            )
        except ValueError as error:
            raise PromotionStateError(
                str(error)
            ) from error

        return self.request_store.save(request)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_request(
        self,
        request_id: str,
    ) -> PromotionRequest:
        return self._get_request(request_id)

    def has_access(
        self,
        *,
        memory_id: str,
        agent_id: str,
    ) -> bool:
        memory = self._get_memory(memory_id)

        return self._can_read(
            memory,
            str(agent_id).strip(),
        )

    # ------------------------------------------------------------------
    # Memory update
    # ------------------------------------------------------------------

    def _validate_request_against_memory(
        self,
        *,
        request: PromotionRequest,
        memory: Any,
    ) -> None:
        memory_id = self._required_memory_text(
            memory,
            "memory_id",
        )
        owner_id = self._required_memory_text(
            memory,
            "owner_agent_id",
        )

        if memory_id != request.memory_id:
            raise PromotionServiceError(
                "The request references a different "
                "memory than the loaded memory."
            )

        if owner_id != request.owner_agent_id:
            raise PromotionServiceError(
                "The memory owner changed after the "
                "promotion request was created."
            )

    # ------------------------------------------------------------------
    # Store access
    # ------------------------------------------------------------------

    def _get_memory(
        self,
        memory_id: str,
    ) -> Any:
        clean_id = str(memory_id).strip()

        if not clean_id:
            raise ValueError(
                "memory_id cannot be empty."
            )

        try:
            memory = self.memory_store.get_by_id(
                clean_id
            )
        except Exception as error:
            raise MemoryNotFoundError(
                f"Failed to load memory {clean_id!r}: "
                f"{error}"
            ) from error

        if memory is None:
            raise MemoryNotFoundError(
                f"Memory {clean_id!r} was not found."
            )

        return memory

    def _get_request(
        self,
        request_id: str,
    ) -> PromotionRequest:
        clean_id = str(request_id).strip()

        if not clean_id:
            raise ValueError(
                "request_id cannot be empty."
            )

        try:
            request = self.request_store.get(
                clean_id
            )
        except Exception as error:
            raise PromotionRequestNotFoundError(
                f"Failed to load promotion request "
                f"{clean_id!r}."
            ) from error

        if request is None:
            raise PromotionRequestNotFoundError(
                f"Promotion request {clean_id!r} "
                "was not found."
            )

        return request

    # ------------------------------------------------------------------
    # Actor validation
    # ------------------------------------------------------------------

    @classmethod
    def _require_role(
        cls,
        agent: AgentProtocol,
        *,
        expected_role: str,
    ) -> None:
        actual_role = cls._role_value(
            getattr(agent, "role", None)
        )

        if actual_role != expected_role:
            raise InvalidPromotionActorError(
                f"Operation requires role "
                f"{expected_role!r}, but received "
                f"{actual_role!r}."
            )

    @staticmethod
    def _agent_id(
        agent: AgentProtocol,
    ) -> str:
        agent_id = str(
            getattr(agent, "agent_id", "")
        ).strip()

        if not agent_id:
            raise InvalidPromotionActorError(
                "Agent must have a non-empty agent_id."
            )

        return agent_id

    @staticmethod
    def _role_value(
        role: Any,
    ) -> str:
        if role is None:
            return ""

        value = getattr(
            role,
            "value",
            role,
        )

        return str(value).strip().lower()

    # ------------------------------------------------------------------
    # Generic memory helpers
    # ------------------------------------------------------------------

    @classmethod
    def _can_read(
        cls,
        memory: Any,
        agent_id: str,
    ) -> bool:
        clean_agent_id = str(
            agent_id
        ).strip()

        if not clean_agent_id:
            return False

        can_be_read_by = getattr(
            memory,
            "can_be_read_by",
            None,
        )

        if callable(can_be_read_by):
            try:
                return bool(
                    can_be_read_by(
                        clean_agent_id
                    )
                )
            except Exception:
                pass

        owner_id = cls._required_memory_text(
            memory,
            "owner_agent_id",
        )

        if clean_agent_id == owner_id:
            return True

        readers = cls._clean_string_list(
            cls._read_memory_field(
                memory,
                "readable_by",
                [],
            )
        )

        return clean_agent_id in readers

    @staticmethod
    def _read_memory_field(
        memory: Any,
        field_name: str,
        default: Any = None,
    ) -> Any:
        """
        Read either a top-level MemoryItem field or a governance field stored
        inside ``memory.metadata``.
        """
        if isinstance(memory, Mapping):
            if field_name in memory:
                return memory.get(
                    field_name,
                    default,
                )

            metadata = memory.get(
                "metadata"
            )

            if isinstance(metadata, Mapping):
                return metadata.get(
                    field_name,
                    default,
                )

            if metadata is not None:
                return getattr(
                    metadata,
                    field_name,
                    default,
                )

            return default

        if hasattr(memory, field_name):
            return getattr(
                memory,
                field_name,
                default,
            )

        metadata = getattr(
            memory,
            "metadata",
            None,
        )

        if metadata is None:
            return default

        if isinstance(metadata, Mapping):
            return metadata.get(
                field_name,
                default,
            )

        return getattr(
            metadata,
            field_name,
            default,
        )


    @classmethod
    def _required_memory_text(
        cls,
        memory: Any,
        field_name: str,
    ) -> str:
        value = cls._read_memory_field(
            memory,
            field_name,
        )

        cleaned = str(
            value or ""
        ).strip()

        if not cleaned:
            raise PromotionServiceError(
                f"Memory field {field_name!r} "
                "cannot be empty."
            )

        return cleaned

    @staticmethod
    def _clean_string_list(
        values: Any,
    ) -> list[str]:
        if values is None:
            return []

        if isinstance(values, str):
            values = [values]

        result: list[str] = []

        for value in values:
            cleaned = str(value).strip()

            if cleaned and cleaned not in result:
                result.append(cleaned)

        return result

