from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class MemoryScope(str, Enum):
    PRIVATE = "private"
    SHARED = "shared"


class SourceType(str, Enum):
    MESSAGE = "message"
    TOOL_RESULT = "tool_result"
    AGENT_OUTPUT = "agent_output"
    MANUAL = "manual"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    PENDING_REVIEW = "pending_review"
    CONFLICTING = "conflicting"
    DEPRECATED = "deprecated"
    RESOLVED = "resolved"


class MemoryType(str, Enum):
    FACT = "fact"
    EVENT = "event"
    RULE = "rule"
    PREFERENCE = "preference"
    NOTE = "note"


class MemoryMetadata(BaseModel):
    """
    Governance metadata for a memory item.

    This metadata supports ownership tracking, access control, provenance,
    lifecycle management, retrieval filtering, and operation history.
    """

    # Ownership
    owner_agent_id: str = Field(
        ...,
        description="The agent that currently owns or manages this memory."
    )
    created_by_agent_id: str = Field(
        ...,
        description="The agent that originally created this memory."
    )

    # Access control
    scope: MemoryScope = Field(
        ...,
        description="Whether the memory is private or shared."
    )
    readable_by: list[str] = Field(
        default_factory=list,
        description="Agent IDs allowed to read this memory. Use ['*'] for all agents."
    )
    writable_by: list[str] = Field(
        default_factory=list,
        description="Agent IDs allowed to update this memory."
    )

    # Provenance
    source_task_id: Optional[str] = Field(
        default=None,
        description="Task ID from which this memory was generated."
    )
    source_message_ids: list[str] = Field(
        default_factory=list,
        description="Message IDs used as evidence for this memory."
    )
    source_type: SourceType = Field(
        default=SourceType.AGENT_OUTPUT,
        description="The type of source from which the memory was extracted."
    )

    # Lifecycle
    status: MemoryStatus = Field(
        default=MemoryStatus.ACTIVE,
        description="Current lifecycle state of the memory."
    )

    # Retrieval
    memory_type: MemoryType = Field(
        default=MemoryType.NOTE,
        description="Lightweight type annotation for retrieval and filtering."
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags used for filtering and retrieval."
    )

    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Importance score used for retrieval ranking."
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the memory content."
    )

    # Time
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp."
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Last update timestamp."
    )

    # Governance history
    last_operation_id: Optional[str] = Field(
        default=None,
        description="Most recent memory operation related to this memory."
    )
    related_operation_ids: list[str] = Field(
        default_factory=list,
        description="All related operation records."
    )

    @field_validator("readable_by", "writable_by", "tags", mode="before")
    @classmethod
    def remove_duplicates(cls, value):
        if value is None:
            return []
        return list(dict.fromkeys(value))

    @classmethod
    def private(
        cls,
        owner_agent_id: str,
        created_by_agent_id: str,
        **kwargs
    ) -> "MemoryMetadata":
        """
        Create default metadata for private memory.
        """
        return cls(
            owner_agent_id=owner_agent_id,
            created_by_agent_id=created_by_agent_id,
            scope=MemoryScope.PRIVATE,
            readable_by=[owner_agent_id],
            writable_by=[owner_agent_id],
            **kwargs
        )

    @classmethod
    def shared(
        cls,
        owner_agent_id: str,
        created_by_agent_id: str,
        writable_by: Optional[list[str]] = None,
        **kwargs
    ) -> "MemoryMetadata":
        """
        Create default metadata for shared memory.
        """
        return cls(
            owner_agent_id=owner_agent_id,
            created_by_agent_id=created_by_agent_id,
            scope=MemoryScope.SHARED,
            readable_by=["*"],
            writable_by=writable_by or [owner_agent_id],
            **kwargs
        )