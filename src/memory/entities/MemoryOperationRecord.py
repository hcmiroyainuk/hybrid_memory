from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryOperationType(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    PROMOTE = "promote"
    REJECT_PROMOTION = "reject_promotion"
    DETECT_CONFLICT = "detect_conflict"
    RESOLVE_CONFLICT = "resolve_conflict"
    MERGE = "merge"
    DEPRECATE = "deprecate"
    SCOPE_CHANGE = "scope_change"


class ConflictType(str, Enum):
    CONTRADICTION = "contradiction"
    DUPLICATION = "duplication"
    OUTDATED = "outdated"
    SCOPE_VIOLATION = "scope_violation"


class ResolutionStrategy(str, Enum):
    KEEP_EXISTING = "keep_existing"
    REPLACE_WITH_NEW = "replace_with_new"
    MERGE = "merge"
    MARK_DEPRECATED = "mark_deprecated"
    IGNORE = "ignore"
    MANUAL_REVIEW = "manual_review"


class OperationStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    IGNORED = "ignored"


class MemoryOperationRecord(BaseModel):
    """
    Unified audit record for memory governance operations.

    This record is used for updates, promotion results, conflict detection,
    conflict resolution, merging, deprecation, and scope changes.
    """

    record_id: str = Field(
        default_factory=lambda: f"op_{uuid4().hex[:12]}",
        description="Unique identifier of the operation record."
    )

    operation_type: MemoryOperationType = Field(
        ...,
        description="Type of memory operation."
    )

    target_memory_ids: list[str] = Field(
        ...,
        min_length=1,
        description="Memory IDs affected by this operation."
    )

    actor_agent_id: str = Field(
        ...,
        description="The agent that performed or initiated this operation."
    )

    reviewer_agent_id: Optional[str] = Field(
        default=None,
        description="The agent that reviewed or approved this operation."
    )

    reason: Optional[str] = Field(
        default=None,
        description="Reason for performing the operation."
    )

    description: Optional[str] = Field(
        default=None,
        description="Detailed description of the operation."
    )

    before_state: Optional[dict[str, Any]] = Field(
        default=None,
        description="Snapshot of relevant memory fields before the operation."
    )

    after_state: Optional[dict[str, Any]] = Field(
        default=None,
        description="Snapshot of relevant memory fields after the operation."
    )

    related_request_id: Optional[str] = Field(
        default=None,
        description="Related promotion request ID, if applicable."
    )

    conflict_type: Optional[ConflictType] = Field(
        default=None,
        description="Conflict type if the operation is conflict-related."
    )

    resolution_strategy: Optional[ResolutionStrategy] = Field(
        default=None,
        description="Resolution strategy if the operation resolves a conflict."
    )

    status: OperationStatus = Field(
        default=OperationStatus.COMPLETED,
        description="Execution status of this operation."
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp."
    )

    @classmethod
    def create_record(
        cls,
        memory_id: str,
        actor_agent_id: str,
        after_state: Optional[dict[str, Any]] = None,
        reason: Optional[str] = None,
    ) -> "MemoryOperationRecord":
        return cls(
            operation_type=MemoryOperationType.CREATE,
            target_memory_ids=[memory_id],
            actor_agent_id=actor_agent_id,
            reason=reason,
            after_state=after_state,
        )

    @classmethod
    def update_record(
        cls,
        memory_id: str,
        actor_agent_id: str,
        before_state: dict[str, Any],
        after_state: dict[str, Any],
        reason: Optional[str] = None,
    ) -> "MemoryOperationRecord":
        return cls(
            operation_type=MemoryOperationType.UPDATE,
            target_memory_ids=[memory_id],
            actor_agent_id=actor_agent_id,
            reason=reason,
            before_state=before_state,
            after_state=after_state,
        )

    @classmethod
    def promote_record(
        cls,
        memory_id: str,
        actor_agent_id: str,
        reviewer_agent_id: Optional[str],
        related_request_id: str,
        before_state: dict[str, Any],
        after_state: dict[str, Any],
        reason: Optional[str] = None,
    ) -> "MemoryOperationRecord":
        return cls(
            operation_type=MemoryOperationType.PROMOTE,
            target_memory_ids=[memory_id],
            actor_agent_id=actor_agent_id,
            reviewer_agent_id=reviewer_agent_id,
            related_request_id=related_request_id,
            reason=reason,
            before_state=before_state,
            after_state=after_state,
        )

    @classmethod
    def detect_conflict_record(
        cls,
        memory_a_id: str,
        memory_b_id: str,
        actor_agent_id: str,
        conflict_type: ConflictType,
        description: str,
    ) -> "MemoryOperationRecord":
        return cls(
            operation_type=MemoryOperationType.DETECT_CONFLICT,
            target_memory_ids=[memory_a_id, memory_b_id],
            actor_agent_id=actor_agent_id,
            conflict_type=conflict_type,
            description=description,
        )

    @classmethod
    def resolve_conflict_record(
        cls,
        memory_ids: list[str],
        actor_agent_id: str,
        conflict_type: ConflictType,
        resolution_strategy: ResolutionStrategy,
        before_state: dict[str, Any],
        after_state: dict[str, Any],
        reason: Optional[str] = None,
        reviewer_agent_id: Optional[str] = None,
    ) -> "MemoryOperationRecord":
        return cls(
            operation_type=MemoryOperationType.RESOLVE_CONFLICT,
            target_memory_ids=memory_ids,
            actor_agent_id=actor_agent_id,
            reviewer_agent_id=reviewer_agent_id,
            conflict_type=conflict_type,
            resolution_strategy=resolution_strategy,
            reason=reason,
            before_state=before_state,
            after_state=after_state,
        )