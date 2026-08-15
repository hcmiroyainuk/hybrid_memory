from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

from ..memory.entities import Agent
from ..memory.services.memory_access_policy_service import (
    AccessPolicyPersistenceError,
    InvalidAccessPolicyError,
    MemoryAccessPolicyError,
    MemoryAccessPolicyService,
)
from ..memory.services.permission_service import (
    PermissionDeniedError,
)
from .agent_registry import (
    AgentNotRegisteredError,
    AgentRegistry,
    InvalidAgentRegistryError,
)
from .memory_access_policy_exceptions import (
    MemoryAccessPolicyConflictError,
    MemoryAccessPolicyGatewayError,
    MemoryAccessPolicyNotFoundError,
    MemoryAccessPolicyPermissionError,
    MemoryAccessPolicyPersistenceError,
    MemoryAccessPolicyValidationError,
)
from .memory_access_policy_gateway import (
    MemoryAccessPolicyGateway,
)
from .memory_access_policy_models import (
    MemoryAccessPolicyResult,
)


T = TypeVar("T")


class MemoryAccessPolicyServiceAdapter(
    MemoryAccessPolicyGateway
):
    """
    Adapt MemoryAccessPolicyService to MemoryAccessPolicyGateway.

    Responsibilities:
    - resolve governance-actor IDs to Agent objects;
    - validate and normalise workflow-facing input values;
    - delegate all ACL mutations to MemoryAccessPolicyService;
    - convert persisted MemoryItem objects into stable workflow DTOs;
    - expose whether an operation actually changed the ACL;
    - translate service-specific exceptions into gateway exceptions.

    This adapter does not:
    - decide which policy should be applied;
    - call an LLM;
    - mutate MemoryItem or MemoryMetadata directly;
    - persist access policies independently of the service.
    """

    _CONFLICT_MESSAGE_MARKERS = (
        "deprecated memory",
        "cannot be changed for deprecated",
        "specific readers cannot be revoked from a global",
        "global '*' acl",
        "global \"*\" acl",
        "lifecycle",
        "current memory state",
        "memory state changed",
    )

    def __init__(
        self,
        *,
        access_policy_service: (
            MemoryAccessPolicyService
        ),
        agents: Mapping[str, Agent] | AgentRegistry,
    ) -> None:
        if access_policy_service is None:
            raise ValueError(
                "access_policy_service cannot be None."
            )

        self._access_policy_service = (
            access_policy_service
        )

        if isinstance(agents, AgentRegistry):
            self._agent_registry = agents
        else:
            self._agent_registry = AgentRegistry(
                agents
            )

    # ------------------------------------------------------------------
    # Gateway operations
    # ------------------------------------------------------------------

    def apply_policy(
        self,
        *,
        governance_actor_id: str,
        memory_id: str,
        target_scope: str,
        allowed_agent_ids: Sequence[str] = (),
        reason: str | None = None,
    ) -> MemoryAccessPolicyResult:
        """
        Replace the complete read-access policy of a memory.
        """
        actor = self._resolve_actor(
            governance_actor_id
        )
        clean_memory_id = self._required_text(
            memory_id,
            "memory_id",
        )
        clean_scope = self._normalise_scope(
            target_scope
        )
        clean_agent_ids = self._clean_agent_ids(
            allowed_agent_ids
        )
        clean_reason = self._optional_text(
            reason
        )

        return self._execute_policy_operation(
            memory_id=clean_memory_id,
            operation=lambda: (
                self._access_policy_service
                .apply_access_policy(
                    governance_actor=actor,
                    memory_id=clean_memory_id,
                    target_scope=clean_scope,
                    allowed_agent_ids=(
                        clean_agent_ids
                    ),
                    reason=clean_reason,
                )
            ),
        )

    def grant_read_access(
        self,
        *,
        governance_actor_id: str,
        memory_id: str,
        agent_ids: Sequence[str],
        reason: str | None = None,
    ) -> MemoryAccessPolicyResult:
        """
        Add one or more Agents to a memory's read ACL.
        """
        actor = self._resolve_actor(
            governance_actor_id
        )
        clean_memory_id = self._required_text(
            memory_id,
            "memory_id",
        )
        clean_agent_ids = self._require_agent_ids(
            agent_ids,
            field_name="agent_ids",
        )
        clean_reason = self._optional_text(
            reason
        )

        return self._execute_policy_operation(
            memory_id=clean_memory_id,
            operation=lambda: (
                self._access_policy_service
                .grant_read_access(
                    governance_actor=actor,
                    memory_id=clean_memory_id,
                    agent_ids=clean_agent_ids,
                    reason=clean_reason,
                )
            ),
        )

    def revoke_read_access(
        self,
        *,
        governance_actor_id: str,
        memory_id: str,
        agent_ids: Sequence[str],
        reason: str | None = None,
    ) -> MemoryAccessPolicyResult:
        """
        Remove one or more Agents from a memory's read ACL.
        """
        actor = self._resolve_actor(
            governance_actor_id
        )
        clean_memory_id = self._required_text(
            memory_id,
            "memory_id",
        )
        clean_agent_ids = self._require_agent_ids(
            agent_ids,
            field_name="agent_ids",
        )
        clean_reason = self._optional_text(
            reason
        )

        return self._execute_policy_operation(
            memory_id=clean_memory_id,
            operation=lambda: (
                self._access_policy_service
                .revoke_read_access(
                    governance_actor=actor,
                    memory_id=clean_memory_id,
                    agent_ids=clean_agent_ids,
                    reason=clean_reason,
                )
            ),
        )

    def make_private(
        self,
        *,
        governance_actor_id: str,
        memory_id: str,
        reason: str | None = None,
    ) -> MemoryAccessPolicyResult:
        """
        Restore a memory to strict owner-only access.
        """
        actor = self._resolve_actor(
            governance_actor_id
        )
        clean_memory_id = self._required_text(
            memory_id,
            "memory_id",
        )
        clean_reason = self._optional_text(
            reason
        )

        return self._execute_policy_operation(
            memory_id=clean_memory_id,
            operation=lambda: (
                self._access_policy_service
                .make_private(
                    governance_actor=actor,
                    memory_id=clean_memory_id,
                    reason=clean_reason,
                )
            ),
        )

    def make_globally_shared(
        self,
        *,
        governance_actor_id: str,
        memory_id: str,
        reason: str | None = None,
    ) -> MemoryAccessPolicyResult:
        """
        Make a memory globally readable through the '*' ACL.
        """
        actor = self._resolve_actor(
            governance_actor_id
        )
        clean_memory_id = self._required_text(
            memory_id,
            "memory_id",
        )
        clean_reason = self._optional_text(
            reason
        )

        return self._execute_policy_operation(
            memory_id=clean_memory_id,
            operation=lambda: (
                self._access_policy_service
                .make_globally_shared(
                    governance_actor=actor,
                    memory_id=clean_memory_id,
                    reason=clean_reason,
                )
            ),
        )

    # ------------------------------------------------------------------
    # Operation execution and DTO conversion
    # ------------------------------------------------------------------

    def _execute_policy_operation(
        self,
        *,
        memory_id: str,
        operation: Callable[[], Any],
    ) -> MemoryAccessPolicyResult:
        """
        Execute one service operation and convert its persisted result.

        The pre-operation snapshot is used only to determine whether the ACL
        changed. Memory mutation remains exclusively inside
        MemoryAccessPolicyService.
        """
        before_memory = self._load_memory(
            memory_id
        )
        before_policy = self._policy_snapshot(
            before_memory
        )

        persisted_memory = self._call_service(
            operation
        )
        after_policy = self._policy_snapshot(
            persisted_memory
        )

        changed = before_policy != after_policy

        return self._to_policy_result(
            persisted_memory,
            changed=changed,
        )

    def _load_memory(
        self,
        memory_id: str,
    ) -> Any:
        """
        Load the current memory only for change detection.

        The adapter never modifies or persists this object.
        """
        try:
            memory = (
                self._access_policy_service
                .memory_store
                .get_by_id(memory_id)
            )
        except KeyError as error:
            raise MemoryAccessPolicyNotFoundError(
                f"Memory {memory_id!r} was not found."
            ) from error
        except Exception as error:
            raise MemoryAccessPolicyGatewayError(
                "Failed to load memory "
                f"{memory_id!r} before applying "
                "the access policy."
            ) from error

        if memory is None:
            raise MemoryAccessPolicyNotFoundError(
                f"Memory {memory_id!r} was not found."
            )

        return memory

    @classmethod
    def _to_policy_result(
        cls,
        memory: Any,
        *,
        changed: bool,
    ) -> MemoryAccessPolicyResult:
        metadata = getattr(
            memory,
            "metadata",
            None,
        )

        if metadata is None:
            raise MemoryAccessPolicyGatewayError(
                "MemoryAccessPolicyService returned a "
                "memory without metadata."
            )

        memory_id = cls._required_result_text(
            getattr(memory, "memory_id", None),
            "memory.memory_id",
        )
        owner_agent_id = cls._required_result_text(
            getattr(
                metadata,
                "owner_agent_id",
                None,
            ),
            "memory.metadata.owner_agent_id",
        )

        scope_value = getattr(
            getattr(metadata, "scope", None),
            "value",
            getattr(metadata, "scope", None),
        )

        operation_id: str | None = None

        if changed:
            operation_id = cls._optional_text(
                getattr(
                    metadata,
                    "last_operation_id",
                    None,
                )
            )

        try:
            return MemoryAccessPolicyResult(
                memory_id=memory_id,
                owner_agent_id=owner_agent_id,
                scope=scope_value,
                readable_by=tuple(
                    cls._clean_agent_ids(
                        getattr(
                            metadata,
                            "readable_by",
                            (),
                        )
                    )
                ),
                writable_by=tuple(
                    cls._clean_agent_ids(
                        getattr(
                            metadata,
                            "writable_by",
                            (),
                        )
                    )
                ),
                changed=changed,
                operation_id=operation_id,
            )
        except Exception as error:
            raise MemoryAccessPolicyGatewayError(
                "MemoryAccessPolicyService returned "
                "an invalid persisted policy result."
            ) from error

    @classmethod
    def _policy_snapshot(
        cls,
        memory: Any,
    ) -> tuple[
        str,
        tuple[str, ...],
        tuple[str, ...],
    ]:
        metadata = getattr(
            memory,
            "metadata",
            None,
        )

        if metadata is None:
            raise MemoryAccessPolicyGatewayError(
                "Cannot inspect access policy because "
                "the memory has no metadata."
            )

        raw_scope = getattr(
            metadata,
            "scope",
            None,
        )
        scope = str(
            getattr(
                raw_scope,
                "value",
                raw_scope,
            )
            or ""
        ).strip().lower()

        return (
            scope,
            tuple(
                cls._clean_agent_ids(
                    getattr(
                        metadata,
                        "readable_by",
                        (),
                    )
                )
            ),
            tuple(
                cls._clean_agent_ids(
                    getattr(
                        metadata,
                        "writable_by",
                        (),
                    )
                )
            ),
        )

    # ------------------------------------------------------------------
    # Agent resolution
    # ------------------------------------------------------------------

    def _resolve_actor(
        self,
        governance_actor_id: str,
    ) -> Agent:
        try:
            return self._agent_registry.require(
                governance_actor_id
            )
        except AgentNotRegisteredError as error:
            raise MemoryAccessPolicyNotFoundError(
                str(error)
            ) from error
        except InvalidAgentRegistryError as error:
            raise MemoryAccessPolicyValidationError(
                str(error)
            ) from error

    # ------------------------------------------------------------------
    # Exception translation
    # ------------------------------------------------------------------

    @classmethod
    def _call_service(
        cls,
        operation: Callable[[], T],
    ) -> T:
        try:
            return operation()
        except AccessPolicyPersistenceError as error:
            raise MemoryAccessPolicyPersistenceError(
                str(error)
            ) from error
        except PermissionDeniedError as error:
            raise MemoryAccessPolicyPermissionError(
                str(error)
            ) from error
        except InvalidAccessPolicyError as error:
            if cls._is_conflict_error(error):
                raise MemoryAccessPolicyConflictError(
                    str(error)
                ) from error

            raise MemoryAccessPolicyValidationError(
                str(error)
            ) from error
        except KeyError as error:
            raise MemoryAccessPolicyNotFoundError(
                str(error)
            ) from error
        except MemoryAccessPolicyError as error:
            raise MemoryAccessPolicyGatewayError(
                str(error)
            ) from error

    @classmethod
    def _is_conflict_error(
        cls,
        error: Exception,
    ) -> bool:
        message = str(error).strip().lower()

        return any(
            marker in message
            for marker in (
                cls._CONFLICT_MESSAGE_MARKERS
            )
        )

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    @staticmethod
    def _required_text(
        value: Any,
        field_name: str,
    ) -> str:
        cleaned = str(
            value or ""
        ).strip()

        if not cleaned:
            raise MemoryAccessPolicyValidationError(
                f"{field_name} cannot be empty."
            )

        return cleaned

    @staticmethod
    def _required_result_text(
        value: Any,
        field_name: str,
    ) -> str:
        cleaned = str(
            value or ""
        ).strip()

        if not cleaned:
            raise MemoryAccessPolicyGatewayError(
                f"{field_name} cannot be empty."
            )

        return cleaned

    @staticmethod
    def _optional_text(
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        cleaned = str(value).strip()
        return cleaned or None

    @classmethod
    def _normalise_scope(
        cls,
        value: Any,
    ) -> str:
        raw_value = getattr(
            value,
            "value",
            value,
        )
        scope = cls._required_text(
            raw_value,
            "target_scope",
        ).lower()

        if scope not in {
            "private",
            "shared",
        }:
            raise MemoryAccessPolicyValidationError(
                "target_scope must be either "
                "'private' or 'shared'."
            )

        return scope

    @classmethod
    def _require_agent_ids(
        cls,
        values: Sequence[str],
        *,
        field_name: str,
    ) -> list[str]:
        cleaned = cls._clean_agent_ids(
            values
        )

        if not cleaned:
            raise MemoryAccessPolicyValidationError(
                f"{field_name} must contain at least "
                "one Agent ID."
            )

        return cleaned

    @staticmethod
    def _clean_agent_ids(
        values: Any,
    ) -> list[str]:
        if values is None:
            return []

        if isinstance(values, str):
            values = [values]

        try:
            iterator = iter(values)
        except TypeError as error:
            raise MemoryAccessPolicyValidationError(
                "Agent IDs must be supplied as "
                "a sequence of strings."
            ) from error

        result: list[str] = []

        for value in iterator:
            cleaned = str(
                value or ""
            ).strip()

            if (
                cleaned
                and cleaned not in result
            ):
                result.append(cleaned)

        return result


__all__ = [
    "MemoryAccessPolicyServiceAdapter",
]