from __future__ import annotations

from typing import Any, Optional

from ..entities import (
    Agent,
    MemoryItem,
    PromotionRequest,
    MemoryOperationRecord,
    MemoryOperationType,
    ConflictType,
    ResolutionStrategy,
)
from ..manager import OperationLogStore


class OperationLogService:
    """
    Business services for creating and storing memory operation logs.

    OperationLogStore only persists records.
    OperationLogService knows what kind of operation record should be created
    for each memory governance event.

    This services does not:
    - check permissions
    - modify memories
    - approve promotion requests
    - resolve conflicts

    It only creates and appends MemoryOperationRecord objects.
    """

    def __init__(self, operation_log_store: OperationLogStore) -> None:
        self.operation_log_store = operation_log_store

    # ------------------------------------------------------------------
    # Memory operation logs
    # ------------------------------------------------------------------

    def record_memory_created(
        self,
        memory: MemoryItem,
        actor_agent: Agent,
        reason: Optional[str] = None,
    ) -> MemoryOperationRecord:
        """
        Record memory creation.
        """

        record = MemoryOperationRecord.create_record(
            memory_id=memory.memory_id,
            actor_agent_id=actor_agent.agent_id,
            after_state=self.snapshot_memory_basic(memory),
            reason=reason or "Memory created.",
        )

        return self.operation_log_store.append(record)

    def record_memory_updated(
        self,
        memory: MemoryItem,
        actor_agent: Agent,
        before_state: dict[str, Any],
        after_state: dict[str, Any],
        reason: Optional[str] = None,
    ) -> MemoryOperationRecord:
        """
        Record memory update.

        before_state and after_state should be compact snapshots of changed fields.
        """

        record = MemoryOperationRecord.update_record(
            memory_id=memory.memory_id,
            actor_agent_id=actor_agent.agent_id,
            before_state=before_state,
            after_state=after_state,
            reason=reason or "Memory updated.",
        )

        return self.operation_log_store.append(record)

    def record_memory_deprecated(
        self,
        memory: MemoryItem,
        actor_agent: Agent,
        before_state: dict[str, Any],
        after_state: dict[str, Any],
        reason: Optional[str] = None,
    ) -> MemoryOperationRecord:
        """
        Record memory deprecation.

        Deprecation is the formal soft-delete operation in this system.
        """

        record = MemoryOperationRecord(
            operation_type=MemoryOperationType.DEPRECATE,
            target_memory_ids=[memory.memory_id],
            actor_agent_id=actor_agent.agent_id,
            reason=reason or "Memory deprecated.",
            before_state=before_state,
            after_state=after_state,
        )

        return self.operation_log_store.append(record)

    # ------------------------------------------------------------------
    # Promotion workflow logs
    # ------------------------------------------------------------------

    def record_promotion_submitted(
        self,
        request: PromotionRequest,
        memory: MemoryItem,
        actor_agent: Agent,
        reason: Optional[str] = None,
    ) -> MemoryOperationRecord:
        """
        Record that a promotion request has been submitted.

        This is not a memory scope change yet.
        It only records the workflow event.
        """

        record = MemoryOperationRecord(
            operation_type=MemoryOperationType.PROMOTE,
            target_memory_ids=[memory.memory_id],
            actor_agent_id=actor_agent.agent_id,
            related_request_id=request.request_id,
            reason=reason or request.reason,
            description="Promotion request submitted.",
            before_state={
                "request_id": request.request_id,
                "request_status": request.status.value,
                "memory_scope": memory.metadata.scope.value,
                "memory_status": memory.metadata.status.value,
                "owner_agent_id": memory.metadata.owner_agent_id,
            },
            after_state={
                "request_id": request.request_id,
                "request_status": request.status.value,
            },
        )

        return self.operation_log_store.append(record)

    def record_promotion_reviewed(
        self,
        request: PromotionRequest,
        memory: MemoryItem,
        reviewer_agent: Agent,
        comment: Optional[str] = None,
    ) -> MemoryOperationRecord:
        """
        Record critic or coordinator review of a promotion request.

        This records recommendation/review, not final approval.
        """

        record = MemoryOperationRecord(
            operation_type=MemoryOperationType.UPDATE,
            target_memory_ids=[memory.memory_id],
            actor_agent_id=reviewer_agent.agent_id,
            reviewer_agent_id=reviewer_agent.agent_id,
            related_request_id=request.request_id,
            reason=comment or request.review_comment or "Promotion request reviewed.",
            description="Promotion request reviewed.",
            before_state={
                "request_id": request.request_id,
                "memory_id": memory.memory_id,
            },
            after_state={
                "request_id": request.request_id,
                "request_status": request.status.value,
                "reviewed_by_agent_id": request.reviewed_by_agent_id,
                "review_comment": request.review_comment,
                "reviewed_at": (
                    request.reviewed_at.isoformat()
                    if request.reviewed_at is not None
                    else None
                ),
            },
        )

        return self.operation_log_store.append(record)

    def record_promotion_approved(
        self,
        request: PromotionRequest,
        memory: MemoryItem,
        actor_agent: Agent,
        comment: Optional[str] = None,
    ) -> MemoryOperationRecord:
        """
        Record that a promotion request has been approved.

        This records the approval decision.
        The actual memory scope change should be recorded by record_memory_promoted().
        """

        record = MemoryOperationRecord(
            operation_type=MemoryOperationType.PROMOTE,
            target_memory_ids=[memory.memory_id],
            actor_agent_id=actor_agent.agent_id,
            reviewer_agent_id=actor_agent.agent_id,
            related_request_id=request.request_id,
            reason=comment or request.review_comment or "Promotion request approved.",
            description="Promotion request approved.",
            before_state={
                "request_id": request.request_id,
                "memory_scope": memory.metadata.scope.value,
                "request_status": "pending",
            },
            after_state={
                "request_id": request.request_id,
                "memory_scope": memory.metadata.scope.value,
                "request_status": request.status.value,
                "reviewed_by_agent_id": request.reviewed_by_agent_id,
                "review_comment": request.review_comment,
            },
        )

        return self.operation_log_store.append(record)

    def record_promotion_rejected(
        self,
        request: PromotionRequest,
        memory: MemoryItem,
        actor_agent: Agent,
        comment: Optional[str] = None,
    ) -> MemoryOperationRecord:
        """
        Record that a promotion request has been rejected.
        """

        record = MemoryOperationRecord(
            operation_type=MemoryOperationType.REJECT_PROMOTION,
            target_memory_ids=[memory.memory_id],
            actor_agent_id=actor_agent.agent_id,
            reviewer_agent_id=actor_agent.agent_id,
            related_request_id=request.request_id,
            reason=comment or request.review_comment or "Promotion request rejected.",
            description="Promotion request rejected.",
            before_state={
                "request_id": request.request_id,
                "memory_scope": memory.metadata.scope.value,
                "request_status": "pending",
            },
            after_state={
                "request_id": request.request_id,
                "memory_scope": memory.metadata.scope.value,
                "request_status": request.status.value,
                "reviewed_by_agent_id": request.reviewed_by_agent_id,
                "review_comment": request.review_comment,
            },
        )

        return self.operation_log_store.append(record)

    def record_memory_promoted(
        self,
        memory: MemoryItem,
        actor_agent: Agent,
        request: PromotionRequest,
        before_state: dict[str, Any],
        after_state: dict[str, Any],
        reason: Optional[str] = None,
        reviewer_agent: Optional[Agent] = None,
    ) -> MemoryOperationRecord:
        """
        Record the actual memory scope change from private to shared.
        """

        record = MemoryOperationRecord.promote_record(
            memory_id=memory.memory_id,
            actor_agent_id=actor_agent.agent_id,
            reviewer_agent_id=(
                reviewer_agent.agent_id
                if reviewer_agent is not None
                else request.reviewed_by_agent_id
            ),
            related_request_id=request.request_id,
            before_state=before_state,
            after_state=after_state,
            reason=reason or "Memory promoted from private to shared.",
        )

        return self.operation_log_store.append(record)

    # ------------------------------------------------------------------
    # Conflict logs
    # ------------------------------------------------------------------

    def record_conflict_detected(
        self,
        memory_a: MemoryItem,
        memory_b: MemoryItem,
        actor_agent: Agent,
        conflict_type: ConflictType,
        description: str,
    ) -> MemoryOperationRecord:
        """
        Record detected conflict between two memories.
        """

        record = MemoryOperationRecord.detect_conflict_record(
            memory_a_id=memory_a.memory_id,
            memory_b_id=memory_b.memory_id,
            actor_agent_id=actor_agent.agent_id,
            conflict_type=conflict_type,
            description=description,
        )

        return self.operation_log_store.append(record)

    def record_conflict_resolved(
        self,
        memory_ids: list[str],
        actor_agent: Agent,
        conflict_type: ConflictType,
        resolution_strategy: ResolutionStrategy,
        before_state: dict[str, Any],
        after_state: dict[str, Any],
        reason: Optional[str] = None,
        reviewer_agent: Optional[Agent] = None,
    ) -> MemoryOperationRecord:
        """
        Record conflict resolution.
        """

        record = MemoryOperationRecord.resolve_conflict_record(
            memory_ids=memory_ids,
            actor_agent_id=actor_agent.agent_id,
            conflict_type=conflict_type,
            resolution_strategy=resolution_strategy,
            before_state=before_state,
            after_state=after_state,
            reason=reason or "Memory conflict resolved.",
            reviewer_agent_id=(
                reviewer_agent.agent_id
                if reviewer_agent is not None
                else None
            ),
        )

        return self.operation_log_store.append(record)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_operation(self, record_id: str) -> MemoryOperationRecord:
        """
        Get operation record by ID.
        """

        return self.operation_log_store.get_by_id(record_id)

    def list_all_operations(self) -> list[MemoryOperationRecord]:
        """
        List all operation records.
        """

        return self.operation_log_store.list_all()

    def list_memory_history(self, memory_id: str) -> list[MemoryOperationRecord]:
        """
        List all operation records related to a memory.
        """

        return self.operation_log_store.list_by_memory_id(memory_id)

    def list_agent_operations(self, agent_id: str) -> list[MemoryOperationRecord]:
        """
        List operation records related to an agent.
        """

        return self.operation_log_store.list_by_agent_id(agent_id)

    def list_operations_by_type(
        self,
        operation_type: MemoryOperationType | str,
    ) -> list[MemoryOperationRecord]:
        """
        List operation records by operation type.
        """

        return self.operation_log_store.list_by_operation_type(operation_type)

    def list_request_operations(
        self,
        request_id: str,
    ) -> list[MemoryOperationRecord]:
        """
        List operation records related to a promotion request.
        """

        return self.operation_log_store.list_by_request_id(request_id)

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------

    @staticmethod
    def snapshot_memory_basic(memory: MemoryItem) -> dict[str, Any]:
        """
        Create a compact snapshot of the current memory state.
        """

        return {
            "memory_id": memory.memory_id,
            "content": memory.content,
            "summary": memory.summary,
            "scope": memory.metadata.scope.value,
            "owner_agent_id": memory.metadata.owner_agent_id,
            "created_by_agent_id": memory.metadata.created_by_agent_id,
            "status": memory.metadata.status.value,
            "memory_type": memory.metadata.memory_type.value,
            "tags": list(memory.metadata.tags),
            "importance": memory.metadata.importance,
            "confidence": memory.metadata.confidence,
        }

    @staticmethod
    def snapshot_memory_content(memory: MemoryItem) -> dict[str, Any]:
        """
        Snapshot content-related fields.
        """

        return {
            "memory_id": memory.memory_id,
            "content": memory.content,
            "summary": memory.summary,
        }

    @staticmethod
    def snapshot_memory_metadata(memory: MemoryItem) -> dict[str, Any]:
        """
        Snapshot non-access metadata fields.
        """

        return {
            "memory_id": memory.memory_id,
            "memory_type": memory.metadata.memory_type.value,
            "tags": list(memory.metadata.tags),
            "importance": memory.metadata.importance,
            "confidence": memory.metadata.confidence,
            "source_task_id": memory.metadata.source_task_id,
            "source_message_ids": list(memory.metadata.source_message_ids),
            "source_type": memory.metadata.source_type.value,
        }

    @staticmethod
    def snapshot_memory_access(memory: MemoryItem) -> dict[str, Any]:
        """
        Snapshot access-related fields.
        """

        return {
            "memory_id": memory.memory_id,
            "scope": memory.metadata.scope.value,
            "owner_agent_id": memory.metadata.owner_agent_id,
            "readable_by": list(memory.metadata.readable_by),
            "writable_by": list(memory.metadata.writable_by),
        }

    @staticmethod
    def snapshot_memory_status(memory: MemoryItem) -> dict[str, Any]:
        """
        Snapshot lifecycle status fields.
        """

        return {
            "memory_id": memory.memory_id,
            "status": memory.metadata.status.value,
            "last_operation_id": memory.metadata.last_operation_id,
            "related_operation_ids": list(memory.metadata.related_operation_ids),
        }