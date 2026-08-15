from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PromotionStatus(str, Enum):
    """
    Lifecycle state of a promotion request.
    """

    PENDING = "pending"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    REJECTED = "rejected"


class CriticRecommendation(str, Enum):
    """
    Recommendation produced by the Critic.

    The recommendation is advisory. Only the Coordinator can make the
    final approval or rejection decision.
    """

    APPROVE = "approve"
    REJECT = "reject"
    KEEP_PRIVATE = "keep_private"
    MERGE = "merge"
    SUPERSEDE = "supersede"
    UNCERTAIN = "uncertain"


class PromotionRequest(BaseModel):
    """
    A request for cross-agent access to an existing memory.

    The memory is initially owned by ``owner_agent_id``. Another worker,
    identified by ``requester_agent_id``, requests access to it.

    When the request is approved:

    1. the memory is promoted to shared scope if it is still private;
    2. the requester is permanently added to the memory's readable ACL;
    3. the original owner retains ownership and read access.

    This object records the request and governance lifecycle. It does not
    modify the memory itself.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
    )

    request_id: str = Field(
        default_factory=lambda: (
            f"promotion_{uuid4().hex[:16]}"
        ),
        description="Unique identifier of the request.",
    )

    memory_id: str = Field(
        description="Memory for which access is requested.",
    )

    owner_agent_id: str = Field(
        description="Agent that owns the requested memory.",
    )

    requester_agent_id: str = Field(
        description=(
            "Agent requesting cross-agent read access."
        ),
    )

    task_id: str | None = Field(
        default=None,
        description=(
            "Optional task that triggered the access request."
        ),
    )

    reason: str = Field(
        min_length=1,
        description=(
            "Why the requester needs access to the memory."
        ),
    )

    status: PromotionStatus = Field(
        default=PromotionStatus.PENDING,
        description="Current lifecycle state.",
    )

    # ------------------------------------------------------------------
    # Critic review
    # ------------------------------------------------------------------

    critic_agent_id: str | None = Field(
        default=None,
        description="Critic that reviewed the request.",
    )

    critic_recommendation: (
        CriticRecommendation | None
    ) = Field(
        default=None,
        description="Critic's advisory recommendation.",
    )

    critic_comment: str | None = Field(
        default=None,
        description="Critic's explanation.",
    )

    critic_reviewed_at: datetime | None = Field(
        default=None,
        description="Time at which the Critic reviewed the request.",
    )

    # ------------------------------------------------------------------
    # Coordinator decision
    # ------------------------------------------------------------------

    decided_by_agent_id: str | None = Field(
        default=None,
        description=(
            "Coordinator that made the final decision."
        ),
    )

    decision_comment: str | None = Field(
        default=None,
        description="Coordinator's final explanation.",
    )

    decided_at: datetime | None = Field(
        default=None,
        description="Time of the final decision.",
    )

    created_at: datetime = Field(
        default_factory=utc_now,
        description="Request creation time.",
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @field_validator(
        "request_id",
        "memory_id",
        "owner_agent_id",
        "requester_agent_id",
        "reason",
    )
    @classmethod
    def required_text_must_not_be_empty(
        cls,
        value: str,
    ) -> str:
        cleaned = str(value).strip()

        if not cleaned:
            raise ValueError(
                "Required text fields cannot be empty."
            )

        return cleaned

    @field_validator(
        "task_id",
        "critic_agent_id",
        "critic_comment",
        "decided_by_agent_id",
        "decision_comment",
    )
    @classmethod
    def clean_optional_text(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        cleaned = str(value).strip()
        return cleaned or None

    @model_validator(mode="after")
    def validate_request_consistency(
        self,
    ) -> "PromotionRequest":
        if (
            self.owner_agent_id
            == self.requester_agent_id
        ):
            raise ValueError(
                "The memory owner does not need a "
                "cross-agent promotion request."
            )

        if self.status == PromotionStatus.REVIEWED:
            if not self.critic_agent_id:
                raise ValueError(
                    "A reviewed request requires "
                    "critic_agent_id."
                )

            if self.critic_recommendation is None:
                raise ValueError(
                    "A reviewed request requires "
                    "critic_recommendation."
                )

            if self.critic_reviewed_at is None:
                raise ValueError(
                    "A reviewed request requires "
                    "critic_reviewed_at."
                )

        if self.status in {
            PromotionStatus.APPROVED,
            PromotionStatus.REJECTED,
        }:
            if not self.decided_by_agent_id:
                raise ValueError(
                    "A completed request requires "
                    "decided_by_agent_id."
                )

            if self.decided_at is None:
                raise ValueError(
                    "A completed request requires decided_at."
                )

        return self

    # ------------------------------------------------------------------
    # Lifecycle operations
    # ------------------------------------------------------------------

    def record_critic_review(
        self,
        *,
        critic_agent_id: str,
        recommendation: CriticRecommendation | str,
        comment: str | None = None,
    ) -> None:
        """
        Record the Critic's advisory review.
        """
        if self.status != PromotionStatus.PENDING:
            raise ValueError(
                "Only a pending request can be reviewed."
            )

        critic_id = str(critic_agent_id).strip()

        if not critic_id:
            raise ValueError(
                "critic_agent_id cannot be empty."
            )

        self.critic_agent_id = critic_id
        self.critic_recommendation = (
            CriticRecommendation(recommendation)
        )
        self.critic_comment = (
            str(comment).strip()
            if comment is not None
            else None
        )
        self.critic_reviewed_at = utc_now()
        self.status = PromotionStatus.REVIEWED

    def approve(
        self,
        *,
        coordinator_agent_id: str,
        comment: str | None = None,
        require_critic_review: bool = True,
    ) -> None:
        """
        Record final approval by the Coordinator.

        ``require_critic_review=False`` is used by the ungoverned
        baseline, where the request is approved without Critic review.
        """
        self._ensure_not_completed()

        if (
            require_critic_review
            and self.status != PromotionStatus.REVIEWED
        ):
            raise ValueError(
                "This request must be reviewed by the Critic "
                "before approval."
            )

        coordinator_id = str(
            coordinator_agent_id
        ).strip()

        if not coordinator_id:
            raise ValueError(
                "coordinator_agent_id cannot be empty."
            )

        # ``validate_assignment=True`` validates the whole model after each
        # assignment. Complete the decision metadata first and change the
        # lifecycle status last, so the completed-state validator sees a
        # consistent object.
        self.decided_by_agent_id = coordinator_id
        self.decision_comment = (
            str(comment).strip()
            if comment is not None
            else None
        )
        self.decided_at = utc_now()
        self.status = PromotionStatus.APPROVED

    def reject(
        self,
        *,
        coordinator_agent_id: str,
        comment: str | None = None,
        require_critic_review: bool = True,
    ) -> None:
        """
        Record final rejection by the Coordinator.
        """
        self._ensure_not_completed()

        if (
            require_critic_review
            and self.status != PromotionStatus.REVIEWED
        ):
            raise ValueError(
                "This request must be reviewed by the Critic "
                "before rejection."
            )

        coordinator_id = str(
            coordinator_agent_id
        ).strip()

        if not coordinator_id:
            raise ValueError(
                "coordinator_agent_id cannot be empty."
            )

        # Set all fields required by a completed request before assigning
        # the terminal status. This keeps assignment-time validation valid.
        self.decided_by_agent_id = coordinator_id
        self.decision_comment = (
            str(comment).strip()
            if comment is not None
            else None
        )
        self.decided_at = utc_now()
        self.status = PromotionStatus.REJECTED

    def _ensure_not_completed(self) -> None:
        if self.status in {
            PromotionStatus.APPROVED,
            PromotionStatus.REJECTED,
        }:
            raise ValueError(
                "A completed promotion request cannot be "
                "modified."
            )

    def is_pending(self) -> bool:
        return self.status == PromotionStatus.PENDING

    def is_reviewed(self) -> bool:
        return self.status == PromotionStatus.REVIEWED

    def is_approved(self) -> bool:
        return self.status == PromotionStatus.APPROVED

    def is_rejected(self) -> bool:
        return self.status == PromotionStatus.REJECTED