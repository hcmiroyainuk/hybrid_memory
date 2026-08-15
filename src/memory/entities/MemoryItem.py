from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from .MemoryMetadata import MemoryMetadata


class MemoryItem(BaseModel):
    """
    Core memory object.

    MemoryItem = content + summary + metadata

    The content stores the actual memory.
    The metadata stores governance information such as ownership,
    scope, access control, provenance, lifecycle status, and retrieval hints.
    """

    memory_id: str = Field(
        default_factory=lambda: f"mem_{uuid4().hex[:12]}",
        description="Unique identifier of the memory."
    )

    content: str = Field(
        ...,
        min_length=1,
        description="The actual memory content."
    )

    summary: Optional[str] = Field(
        default=None,
        description="Optional short summary of the memory."
    )

    metadata: MemoryMetadata = Field(
        ...,
        description="Governance metadata for the memory."
    )

    def is_private(self) -> bool:
        return self.metadata.scope == "private"

    def is_shared(self) -> bool:
        return self.metadata.scope == "shared"

    def is_active(self) -> bool:
        return self.metadata.status == "active"

    def can_be_read_by(self, agent_id: str) -> bool:
        return "*" in self.metadata.readable_by or agent_id in self.metadata.readable_by

    def can_be_written_by(self, agent_id: str) -> bool:
        return agent_id in self.metadata.writable_by

    def mark_deprecated(self, operation_id: Optional[str] = None) -> None:
        # self.metadata.status = "deprecated"
        self.metadata.status = MemoryMetadata.status.DEPRECATED
        if operation_id:
            self.metadata.last_operation_id = operation_id
            self.metadata.related_operation_ids.append(operation_id)

    def mark_conflicting(self, operation_id: Optional[str] = None) -> None:
        # self.metadata.status = "conflicting"
        self.metadata.status = MemoryMetadata.status.CONFLICTING
        if operation_id:
            self.metadata.last_operation_id = operation_id
            self.metadata.related_operation_ids.append(operation_id)