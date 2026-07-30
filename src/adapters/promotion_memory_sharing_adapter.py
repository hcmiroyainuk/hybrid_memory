from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from src.memory.entities import Agent
from src.memory.services.promotion_service import (
    AccessAlreadyGrantedError,
    InvalidPromotionActorError,
    MemoryNotFoundError,
    PromotionRequestNotFoundError,
    PromotionService,
    PromotionServiceError,
    PromotionStateError,
)
from .memory_sharing_exceptions import (
    MemorySharingConflictError,
    MemorySharingError,
    MemorySharingNotFoundError,
    MemorySharingPermissionError,
)
from .memory_sharing_gateway import MemorySharingGateway
from .memory_sharing_models import (
    MemoryAccessDecision,
    MemoryAccessDecisionResult,
    MemoryAccessRequestResult,
    MemoryAccessStatus,
)


T = TypeVar("T")


class PromotionMemorySharingAdapter(MemorySharingGateway):
    """
    Adapt PromotionService to the workflow-facing MemorySharingGateway.

    Responsibilities:
    - resolve agent IDs to Agent objects;
    - translate gateway calls into PromotionService calls;
    - convert PromotionRequest entities into stable workflow DTOs;
    - translate service-specific exceptions into gateway exceptions.

    It does not implement governance rules and does not directly modify
    MemoryStore, PromotionRequestStore, or memory ACL fields.
    """

    def __init__(
        self,
        *,
        promotion_service: PromotionService,
        agents: Mapping[str, Agent],
    ) -> None:
        self._promotion_service = promotion_service
        self._agents = dict(agents)

        for mapping_id, agent in self._agents.items():
            clean_mapping_id = str(mapping_id).strip()
            actual_agent_id = str(
                getattr(agent, "agent_id", "")
            ).strip()

            if not clean_mapping_id:
                raise ValueError("Agent mapping keys cannot be empty.")

            if not actual_agent_id:
                raise ValueError(
                    "Every registered Agent must have a non-empty agent_id."
                )

            if clean_mapping_id != actual_agent_id:
                raise ValueError(
                    "Agent mapping key does not match Agent.agent_id: "
                    f"{mapping_id!r} != {actual_agent_id!r}."
                )

    # ------------------------------------------------------------------
    # Gateway operations
    # ------------------------------------------------------------------

    def has_access(
        self,
        *,
        agent_id: str,
        memory_id: str,
    ) -> bool:
        clean_agent_id = self._required_text(agent_id, "agent_id")
        clean_memory_id = self._required_text(memory_id, "memory_id")

        # Keep unknown-agent behaviour consistent across all gateway methods.
        self._get_agent(clean_agent_id)

        return self._call_service(
            lambda: self._promotion_service.has_access(
                memory_id=clean_memory_id,
                agent_id=clean_agent_id,
            )
        )

    def request_access(
        self,
        *,
        requester_agent_id: str,
        memory_id: str,
        reason: str,
        task_id: str | None = None,
    ) -> MemoryAccessRequestResult:
        requester = self._get_agent(requester_agent_id)

        request = self._call_service(
            lambda: self._promotion_service.submit_promotion_request(
                requester=requester,
                memory_id=self._required_text(memory_id, "memory_id"),
                reason=self._required_text(reason, "reason"),
                task_id=self._optional_text(task_id),
            )
        )

        return self._to_request_result(request)

    def review_request(
        self,
        *,
        critic_agent_id: str,
        request_id: str,
        recommendation: str,
        comment: str | None = None,
    ) -> MemoryAccessRequestResult:
        critic = self._get_agent(critic_agent_id)

        request = self._call_service(
            lambda: self._promotion_service.review_promotion_request(
                critic=critic,
                request_id=self._required_text(request_id, "request_id"),
                recommendation=self._required_text(
                    recommendation,
                    "recommendation",
                ),
                comment=self._optional_text(comment),
            )
        )

        return self._to_request_result(request)

    def approve_request(
        self,
        *,
        coordinator_agent_id: str,
        request_id: str,
        comment: str | None = None,
    ) -> MemoryAccessDecisionResult:
        coordinator = self._get_agent(coordinator_agent_id)

        request = self._call_service(
            lambda: self._promotion_service.approve_promotion_request(
                coordinator=coordinator,
                request_id=self._required_text(request_id, "request_id"),
                comment=self._optional_text(comment),
                require_critic_review=True,
            )
        )

        return self._to_decision_result(request)

    def approve_direct_request(
        self,
        *,
        coordinator_agent_id: str,
        request_id: str,
        comment: str | None = None,
    ) -> MemoryAccessDecisionResult:
        coordinator = self._get_agent(coordinator_agent_id)

        request = self._call_service(
            lambda: self._promotion_service.approve_ungoverned_request(
                coordinator=coordinator,
                request_id=self._required_text(request_id, "request_id"),
                comment=self._optional_text(comment),
            )
        )

        return self._to_decision_result(request)

    def reject_request(
        self,
        *,
        coordinator_agent_id: str,
        request_id: str,
        comment: str | None = None,
        require_critic_review: bool = True,
    ) -> MemoryAccessDecisionResult:
        coordinator = self._get_agent(coordinator_agent_id)

        request = self._call_service(
            lambda: self._promotion_service.reject_promotion_request(
                coordinator=coordinator,
                request_id=self._required_text(request_id, "request_id"),
                comment=self._optional_text(comment),
                require_critic_review=require_critic_review,
            )
        )

        return self._to_decision_result(request)

    def get_request(
        self,
        *,
        request_id: str,
    ) -> MemoryAccessRequestResult:
        request = self._call_service(
            lambda: self._promotion_service.get_request(
                self._required_text(request_id, "request_id")
            )
        )

        return self._to_request_result(request)

    # ------------------------------------------------------------------
    # Agent resolution
    # ------------------------------------------------------------------

    def _get_agent(self, agent_id: str) -> Agent:
        clean_agent_id = self._required_text(agent_id, "agent_id")
        agent = self._agents.get(clean_agent_id)

        if agent is None:
            raise MemorySharingNotFoundError(
                f"Agent {clean_agent_id!r} is not registered."
            )

        return agent

    # ------------------------------------------------------------------
    # DTO conversion
    # ------------------------------------------------------------------

    @classmethod
    def _to_request_result(
        cls,
        request: Any,
    ) -> MemoryAccessRequestResult:
        return MemoryAccessRequestResult(
            request_id=cls._required_attribute_text(request, "request_id"),
            memory_id=cls._required_attribute_text(request, "memory_id"),
            owner_agent_id=cls._required_attribute_text(
                request,
                "owner_agent_id",
            ),
            requester_agent_id=cls._required_attribute_text(
                request,
                "requester_agent_id",
            ),
            task_id=cls._optional_text(getattr(request, "task_id", None)),
            status=cls._normalise_status(getattr(request, "status", None)),
            critic_recommendation=cls._normalise_optional_enum(
                getattr(request, "critic_recommendation", None)
            ),
        )

    @classmethod
    def _to_decision_result(
        cls,
        request: Any,
    ) -> MemoryAccessDecisionResult:
        status = cls._normalise_status(getattr(request, "status", None))

        if status == MemoryAccessStatus.APPROVED:
            decision = MemoryAccessDecision.APPROVED
            access_granted = True
        elif status == MemoryAccessStatus.REJECTED:
            decision = MemoryAccessDecision.REJECTED
            access_granted = False
        else:
            raise MemorySharingConflictError(
                "A final decision result requires an approved or rejected "
                f"request, but received {status.value!r}."
            )

        return MemoryAccessDecisionResult(
            request_id=cls._required_attribute_text(request, "request_id"),
            memory_id=cls._required_attribute_text(request, "memory_id"),
            requester_agent_id=cls._required_attribute_text(
                request,
                "requester_agent_id",
            ),
            status=status,
            decision=decision,
            access_granted=access_granted,
        )

    @staticmethod
    def _normalise_status(status: Any) -> MemoryAccessStatus:
        value = getattr(status, "value", status)

        try:
            return MemoryAccessStatus(str(value).strip().lower())
        except ValueError as error:
            raise MemorySharingError(
                "PromotionService returned an unsupported request status: "
                f"{value!r}."
            ) from error

    @staticmethod
    def _normalise_optional_enum(value: Any) -> str | None:
        if value is None:
            return None

        raw_value = getattr(value, "value", value)
        return str(raw_value).strip() or None

    # ------------------------------------------------------------------
    # Exception translation
    # ------------------------------------------------------------------

    @staticmethod
    def _call_service(operation: Callable[[], T]) -> T:
        try:
            return operation()
        except (
            PromotionRequestNotFoundError,
            MemoryNotFoundError,
        ) as error:
            raise MemorySharingNotFoundError(str(error)) from error
        except (
            AccessAlreadyGrantedError,
            PromotionStateError,
        ) as error:
            raise MemorySharingConflictError(str(error)) from error
        except InvalidPromotionActorError as error:
            raise MemorySharingPermissionError(str(error)) from error
        except PromotionServiceError as error:
            raise MemorySharingError(str(error)) from error

    # ------------------------------------------------------------------
    # Value validation
    # ------------------------------------------------------------------

    @staticmethod
    def _required_text(value: Any, field_name: str) -> str:
        cleaned = str(value or "").strip()

        if not cleaned:
            raise ValueError(f"{field_name} cannot be empty.")

        return cleaned

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        if value is None:
            return None

        cleaned = str(value).strip()
        return cleaned or None

    @classmethod
    def _required_attribute_text(
        cls,
        instance: Any,
        field_name: str,
    ) -> str:
        if not hasattr(instance, field_name):
            raise MemorySharingError(
                "PromotionService returned an object without required field "
                f"{field_name!r}."
            )

        return cls._required_text(
            getattr(instance, field_name),
            field_name,
        )


__all__ = ["PromotionMemorySharingAdapter"]