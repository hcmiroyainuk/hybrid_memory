from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MemoryAccessStatus(str, Enum):
    """Stable request statuses exposed to workflow code."""

    PENDING = "pending"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    REJECTED = "rejected"


class MemoryAccessDecision(str, Enum):
    """Final access decisions exposed to workflow code."""

    APPROVED = "approved"
    REJECTED = "rejected"


class MemoryAccessRequestResult(BaseModel):
    """
    Workflow-facing view of a cross-agent memory access request.

    Only fields required for workflow state, routing, and evaluation are
    exposed. The complete PromotionRequest entity remains internal to the
    memory-governance subsystem.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(min_length=1)
    memory_id: str = Field(min_length=1)
    owner_agent_id: str = Field(min_length=1)
    requester_agent_id: str = Field(min_length=1)
    task_id: str | None = None
    status: MemoryAccessStatus
    critic_recommendation: str | None = None


class MemoryAccessDecisionResult(BaseModel):
    """Workflow-facing result of a final access decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(min_length=1)
    memory_id: str = Field(min_length=1)
    requester_agent_id: str = Field(min_length=1)
    status: MemoryAccessStatus
    decision: MemoryAccessDecision
    access_granted: bool

    @model_validator(mode="after")
    def validate_decision_consistency(self) -> "MemoryAccessDecisionResult":
        if (
            self.decision == MemoryAccessDecision.APPROVED
            and not self.access_granted
        ):
            raise ValueError("An approved decision must grant access.")

        if (
            self.decision == MemoryAccessDecision.REJECTED
            and self.access_granted
        ):
            raise ValueError("A rejected decision cannot grant access.")

        return self


__all__ = [
    "MemoryAccessStatus",
    "MemoryAccessDecision",
    "MemoryAccessRequestResult",
    "MemoryAccessDecisionResult",
]