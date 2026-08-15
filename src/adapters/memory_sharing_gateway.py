from __future__ import annotations

from typing import Protocol, runtime_checkable

from .memory_sharing_models import (
    MemoryAccessDecisionResult,
    MemoryAccessRequestResult,
)


@runtime_checkable
class MemorySharingGateway(Protocol):
    """
    Workflow-facing contract for cross-agent memory sharing.

    Workflow modules should depend on this Protocol rather than importing
    PromotionService, PromotionRequest, or persistence-layer classes.
    """

    def has_access(
        self,
        *,
        agent_id: str,
        memory_id: str,
    ) -> bool:
        """Return whether an Agent can currently read a Memory."""
        ...

    def request_access(
        self,
        *,
        requester_agent_id: str,
        memory_id: str,
        reason: str,
        task_id: str | None = None,
    ) -> MemoryAccessRequestResult:
        """Submit a cross-agent memory access request."""
        ...

    def review_request(
        self,
        *,
        critic_agent_id: str,
        request_id: str,
        recommendation: str,
        comment: str | None = None,
    ) -> MemoryAccessRequestResult:
        """Record the Critic's advisory review."""
        ...

    def approve_request(
            self,
            *,
            coordinator_agent_id: str,
            request_id: str,
            comment: str | None = None,
            require_critic_review: bool = True,
    ) -> MemoryAccessDecisionResult:
        """
        Approve an access request.

        When require_critic_review is True, the request must have
        completed Critic review before Coordinator approval.

        When False, the Coordinator may approve the request directly.
        """
        ...

    def approve_direct_request(
            self,
            *,
            coordinator_agent_id: str,
            request_id: str,
            comment: str | None = None,
    ) -> MemoryAccessDecisionResult:
        """
        Approve an access request without requiring prior Critic review.

        This method is retained as a compatibility shortcut. New workflow
        code may call approve_request(..., require_critic_review=False)
        directly.
        """
        ...

    def reject_request(
        self,
        *,
        coordinator_agent_id: str,
        request_id: str,
        comment: str | None = None,
        require_critic_review: bool = True,
    ) -> MemoryAccessDecisionResult:
        """Reject a request without granting access."""
        ...

    def get_request(
        self,
        *,
        request_id: str,
    ) -> MemoryAccessRequestResult:
        """Return the workflow-facing state of a request."""
        ...


__all__ = ["MemorySharingGateway"]