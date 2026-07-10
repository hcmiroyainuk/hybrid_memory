from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class PromotionStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class PromotionRequest(BaseModel):
    """
    A request to promote a private memory into shared memory.

    Workers can propose promotion.
    Critics or coordinators can review it.
    Coordinators can approve the final promotion.
    """

    request_id: str = Field(
        default_factory=lambda: f"req_{uuid4().hex[:12]}",
        description="Unique identifier of the promotion request."
    )

    memory_id: str = Field(
        ...,
        description="The private memory being proposed for sharing."
    )

    proposed_by_agent_id: str = Field(
        ...,
        description="The agent that proposed this memory for promotion."
    )

    reason: str = Field(
        ...,
        min_length=1,
        description="Reason why this memory should be shared."
    )

    status: PromotionStatus = Field(
        default=PromotionStatus.PENDING,
        description="Current review status of the request."
    )

    reviewed_by_agent_id: Optional[str] = Field(
        default=None,
        description="The agent that reviewed this request."
    )

    review_comment: Optional[str] = Field(
        default=None,
        description="Review comment explaining approval or rejection."
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp."
    )

    reviewed_at: Optional[datetime] = Field(
        default=None,
        description="Review timestamp."
    )

    def approve(self, reviewer_agent_id: str, comment: Optional[str] = None) -> None:
        self.status = PromotionStatus.APPROVED
        self.reviewed_by_agent_id = reviewer_agent_id
        self.review_comment = comment
        self.reviewed_at = datetime.now(timezone.utc)

    def reject(self, reviewer_agent_id: str, comment: Optional[str] = None) -> None:
        self.status = PromotionStatus.REJECTED
        self.reviewed_by_agent_id = reviewer_agent_id
        self.review_comment = comment
        self.reviewed_at = datetime.now(timezone.utc)

    def is_pending(self) -> bool:
        return self.status == PromotionStatus.PENDING

    def is_approved(self) -> bool:
        return self.status == PromotionStatus.APPROVED

    def is_rejected(self) -> bool:
        return self.status == PromotionStatus.REJECTED