from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

from ..entities import (
    Agent,
    MemoryItem,
    MemoryOperationRecord,
    MemoryOperationType,
    MemoryScope,
    MemoryStatus,
)
from ..manager import MemoryStore, OperationLogStore
from .permission_service import PermissionService


class MemoryAccessPolicyError(Exception):
    """Base exception for memory access-policy operations."""


class InvalidAccessPolicyError(MemoryAccessPolicyError):
    """Raised when a requested access policy is invalid."""


class AccessPolicyPersistenceError(MemoryAccessPolicyError):
    """Raised when an access-policy change cannot be persisted safely."""


class MemoryAccessPolicyService:
    """
    Application service for managing memory visibility and read ACLs.

    Responsibilities:
    - apply private or shared access policies to existing memories;
    - grant and revoke cross-agent read access;
    - ensure the memory owner always retains access;
    - ensure only an authorised Coordinator can change access policy;
    - persist an auditable SCOPE_CHANGE operation when logging is configured.

    This service does not:
    - call an LLM;
    - decide which policy should be selected;
    - modify memory content;
    - process promotion requests.

    The orchestration layer should convert a structured access decision into
    arguments for ``apply_access_policy``.
    """

    def __init__(
        self,
        *,
        memory_store: MemoryStore,
        permission_service: Optional[PermissionService] = None,
        operation_log_store: Optional[OperationLogStore] = None,
        known_agent_ids: Optional[Iterable[str]] = None,
    ) -> None:
        self.memory_store = memory_store
        self.permission_service = (
            permission_service or PermissionService()
        )
        self.operation_log_store = operation_log_store
        self.known_agent_ids = (
            None
            if known_agent_ids is None
            else set(self._clean_ids(known_agent_ids))
        )

    # ------------------------------------------------------------------
    # Main policy operation
    # ------------------------------------------------------------------

    def apply_access_policy(
        self,
        *,
        governance_actor: Agent,
        memory_id: str,
        target_scope: MemoryScope | str,
        allowed_agent_ids: Optional[Iterable[str]] = None,
        reason: Optional[str] = None,
    ) -> MemoryItem:
        """
        Apply a complete read-access policy to an existing memory.

        ``allowed_agent_ids`` contains additional authorised readers rather
        than the owner. The owner is added automatically for targeted sharing.

        Rules:
        - private memory is strictly owner-only;
        - shared memory requires at least one non-owner reader or ``"*"``
          for global read access;
        - ``"*"`` cannot be combined with specific Agent IDs;
        - the governance actor does not automatically receive read access.
        """
        clean_memory_id = self._required_text(
            memory_id,
            "memory_id",
        )
        clean_reason = self._optional_text(reason)

        # Reuse the current final-governance permission rule. A dedicated
        # assert_can_manage_access_policy() method can replace this later.
        self.permission_service.assert_can_approve_promotion(
            governance_actor
        )

        memory = self.memory_store.get_by_id(
            clean_memory_id
        )

        if memory.metadata.status == MemoryStatus.DEPRECATED:
            raise InvalidAccessPolicyError(
                "Access policy cannot be changed for deprecated memory "
                f"{clean_memory_id!r}."
            )

        try:
            scope = MemoryScope(target_scope)
        except ValueError as error:
            raise InvalidAccessPolicyError(
                f"Unsupported target_scope: {target_scope!r}."
            ) from error

        owner_agent_id = self._required_text(
            memory.metadata.owner_agent_id,
            "memory.metadata.owner_agent_id",
        )
        requested_readers = self._clean_ids(
            allowed_agent_ids or []
        )

        final_readers = self._build_read_acl(
            owner_agent_id=owner_agent_id,
            target_scope=scope,
            requested_readers=requested_readers,
        )
        self._validate_known_agents(
            final_readers,
            owner_agent_id=owner_agent_id,
        )

        before_state = self.snapshot_access_policy(
            memory
        )

        updated_memory = memory.model_copy(
            deep=True
        )
        updated_memory.metadata.scope = scope
        updated_memory.metadata.readable_by = (
            final_readers
        )

        if scope == MemoryScope.PRIVATE:
            # Private memory remains entirely owner-managed.
            updated_memory.metadata.writable_by = [
                owner_agent_id
            ]
        else:
            # Read-policy changes must not silently remove existing writers.
            updated_memory.metadata.writable_by = (
                self._owner_first(
                    owner_agent_id,
                    self._clean_ids(
                        updated_memory.metadata.writable_by
                    ),
                )
            )

        after_state = self.snapshot_access_policy(
            updated_memory
        )

        if before_state == after_state:
            return memory

        operation_record = self._build_operation_record(
            memory=updated_memory,
            governance_actor=governance_actor,
            before_state=before_state,
            after_state=after_state,
            reason=clean_reason,
        )

        if operation_record is not None:
            updated_memory.metadata.last_operation_id = (
                operation_record.record_id
            )
            updated_memory.metadata.related_operation_ids = (
                self._clean_ids(
                    [
                        *updated_memory.metadata.related_operation_ids,
                        operation_record.record_id,
                    ]
                )
            )

        persisted_memory = self.memory_store.replace(
            updated_memory
        )

        if operation_record is not None:
            self._append_operation_with_rollback(
                record=operation_record,
                original_memory=memory,
                memory_id=clean_memory_id,
            )

        return persisted_memory

    # ------------------------------------------------------------------
    # Convenience operations
    # ------------------------------------------------------------------

    def grant_read_access(
        self,
        *,
        governance_actor: Agent,
        memory_id: str,
        agent_ids: Iterable[str],
        reason: Optional[str] = None,
    ) -> MemoryItem:
        """
        Grant one or more Agents read access.

        A private memory becomes shared. Existing targeted readers are retained.
        Global shared memory is returned unchanged because every Agent already
        has read access.
        """
        memory = self.memory_store.get_by_id(
            self._required_text(
                memory_id,
                "memory_id",
            )
        )

        current_readers = self._clean_ids(
            memory.metadata.readable_by
        )

        if "*" in current_readers:
            return memory

        owner_agent_id = self._required_text(
            memory.metadata.owner_agent_id,
            "memory.metadata.owner_agent_id",
        )
        additional_readers = self._clean_ids(
            agent_ids
        )

        requested_readers = self._clean_ids(
            [
                *[
                    reader
                    for reader in current_readers
                    if reader != owner_agent_id
                ],
                *additional_readers,
            ]
        )

        return self.apply_access_policy(
            governance_actor=governance_actor,
            memory_id=memory.memory_id,
            target_scope=MemoryScope.SHARED,
            allowed_agent_ids=requested_readers,
            reason=reason or "Read access granted.",
        )

    def revoke_read_access(
        self,
        *,
        governance_actor: Agent,
        memory_id: str,
        agent_ids: Iterable[str],
        reason: Optional[str] = None,
        make_private_when_empty: bool = True,
    ) -> MemoryItem:
        """
        Revoke read access from specific Agents.

        The owner cannot be revoked. A global ``["*"]`` ACL cannot be partially
        revoked without first replacing it with an explicit reader list.

        When no non-owner reader remains, the memory becomes private by default.
        """
        memory = self.memory_store.get_by_id(
            self._required_text(
                memory_id,
                "memory_id",
            )
        )
        owner_agent_id = self._required_text(
            memory.metadata.owner_agent_id,
            "memory.metadata.owner_agent_id",
        )
        revoked_ids = set(
            self._clean_ids(agent_ids)
        )

        if owner_agent_id in revoked_ids:
            raise InvalidAccessPolicyError(
                "The memory owner cannot have read access revoked."
            )

        current_readers = self._clean_ids(
            memory.metadata.readable_by
        )

        if "*" in current_readers:
            raise InvalidAccessPolicyError(
                "Specific readers cannot be revoked from a global '*' ACL. "
                "Apply an explicit targeted policy first."
            )

        remaining_readers = [
            reader
            for reader in current_readers
            if (
                reader != owner_agent_id
                and reader not in revoked_ids
            )
        ]

        if not remaining_readers:
            if make_private_when_empty:
                return self.make_private(
                    governance_actor=governance_actor,
                    memory_id=memory.memory_id,
                    reason=(
                        reason
                        or "Last non-owner reader removed; memory made private."
                    ),
                )

            raise InvalidAccessPolicyError(
                "A shared memory must retain at least one non-owner reader."
            )

        return self.apply_access_policy(
            governance_actor=governance_actor,
            memory_id=memory.memory_id,
            target_scope=MemoryScope.SHARED,
            allowed_agent_ids=remaining_readers,
            reason=reason or "Read access revoked.",
        )

    def make_private(
        self,
        *,
        governance_actor: Agent,
        memory_id: str,
        reason: Optional[str] = None,
    ) -> MemoryItem:
        """Reset a memory to strict owner-only access."""
        return self.apply_access_policy(
            governance_actor=governance_actor,
            memory_id=memory_id,
            target_scope=MemoryScope.PRIVATE,
            allowed_agent_ids=[],
            reason=reason or "Memory access reset to owner-only.",
        )

    def make_globally_shared(
        self,
        *,
        governance_actor: Agent,
        memory_id: str,
        reason: Optional[str] = None,
    ) -> MemoryItem:
        """Make a memory readable by every Agent."""
        return self.apply_access_policy(
            governance_actor=governance_actor,
            memory_id=memory_id,
            target_scope=MemoryScope.SHARED,
            allowed_agent_ids=["*"],
            reason=reason or "Memory made globally readable.",
        )

    # ------------------------------------------------------------------
    # Validation and ACL construction
    # ------------------------------------------------------------------

    def _build_read_acl(
        self,
        *,
        owner_agent_id: str,
        target_scope: MemoryScope,
        requested_readers: list[str],
    ) -> list[str]:
        if target_scope == MemoryScope.PRIVATE:
            non_owner_readers = [
                reader
                for reader in requested_readers
                if reader != owner_agent_id
            ]

            if non_owner_readers:
                raise InvalidAccessPolicyError(
                    "A private memory cannot authorise non-owner readers."
                )

            return [owner_agent_id]

        if "*" in requested_readers:
            if len(requested_readers) != 1:
                raise InvalidAccessPolicyError(
                    "The '*' wildcard cannot be combined with "
                    "specific Agent IDs."
                )
            return ["*"]

        non_owner_readers = [
            reader
            for reader in requested_readers
            if reader != owner_agent_id
        ]

        if not non_owner_readers:
            raise InvalidAccessPolicyError(
                "A shared memory requires at least one non-owner "
                "reader or the '*' wildcard."
            )

        return self._owner_first(
            owner_agent_id,
            non_owner_readers,
        )

    def _validate_known_agents(
        self,
        reader_ids: Iterable[str],
        *,
        owner_agent_id: str,
    ) -> None:
        if self.known_agent_ids is None:
            return

        unknown_agent_ids = {
            agent_id
            for agent_id in reader_ids
            if (
                agent_id not in {
                    "*",
                    owner_agent_id,
                }
                and agent_id not in self.known_agent_ids
            )
        }

        if unknown_agent_ids:
            raise InvalidAccessPolicyError(
                "Access policy contains unknown Agent IDs: "
                f"{sorted(unknown_agent_ids)}."
            )

    # ------------------------------------------------------------------
    # Persistence and audit logging
    # ------------------------------------------------------------------

    def _build_operation_record(
        self,
        *,
        memory: MemoryItem,
        governance_actor: Agent,
        before_state: dict[str, object],
        after_state: dict[str, object],
        reason: Optional[str],
    ) -> Optional[MemoryOperationRecord]:
        if self.operation_log_store is None:
            return None

        return MemoryOperationRecord(
            operation_type=MemoryOperationType.SCOPE_CHANGE,
            target_memory_ids=[memory.memory_id],
            actor_agent_id=governance_actor.agent_id,
            reviewer_agent_id=governance_actor.agent_id,
            reason=reason or "Memory access policy changed.",
            description=(
                "Memory scope or read-access control list changed."
            ),
            before_state=before_state,
            after_state=after_state,
        )

    def _append_operation_with_rollback(
        self,
        *,
        record: MemoryOperationRecord,
        original_memory: MemoryItem,
        memory_id: str,
    ) -> None:
        assert self.operation_log_store is not None

        try:
            self.operation_log_store.append(
                record
            )
        except Exception as log_error:
            try:
                self.memory_store.replace(
                    original_memory
                )
            except Exception as rollback_error:
                raise AccessPolicyPersistenceError(
                    "Access policy was changed, audit logging failed, and "
                    f"rollback also failed for memory {memory_id!r}: "
                    f"{rollback_error}"
                ) from log_error

            raise AccessPolicyPersistenceError(
                "Access policy change was rolled back because audit "
                f"logging failed for memory {memory_id!r}."
            ) from log_error

    @staticmethod
    def snapshot_access_policy(
        memory: MemoryItem,
    ) -> dict[str, object]:
        """Return a compact snapshot of access-relevant metadata."""
        return {
            "memory_id": memory.memory_id,
            "owner_agent_id": (
                memory.metadata.owner_agent_id
            ),
            "scope": memory.metadata.scope.value,
            "readable_by": list(
                memory.metadata.readable_by
            ),
            "writable_by": list(
                memory.metadata.writable_by
            ),
            "status": memory.metadata.status.value,
        }

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _required_text(
        value: object,
        field_name: str,
    ) -> str:
        cleaned = str(value).strip()

        if not cleaned:
            raise InvalidAccessPolicyError(
                f"{field_name} cannot be empty."
            )

        return cleaned

    @staticmethod
    def _optional_text(
        value: object | None,
    ) -> Optional[str]:
        if value is None:
            return None

        cleaned = str(value).strip()
        return cleaned or None

    @staticmethod
    def _clean_ids(
        values: Iterable[object],
    ) -> list[str]:
        cleaned: list[str] = []

        for value in values:
            item = str(value).strip()

            if item and item not in cleaned:
                cleaned.append(item)

        return cleaned

    @classmethod
    def _owner_first(
        cls,
        owner_agent_id: str,
        values: Iterable[object],
    ) -> list[str]:
        return cls._clean_ids(
            [
                owner_agent_id,
                *values,
            ]
        )


__all__ = [
    "MemoryAccessPolicyError",
    "InvalidAccessPolicyError",
    "AccessPolicyPersistenceError",
    "MemoryAccessPolicyService",
]