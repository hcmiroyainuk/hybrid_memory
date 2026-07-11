from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Protocol

from ..entities import (
    Agent,
    PromotionRequest,
    PromotionStatus,
    MemoryItem,
    MemoryScope,
    MemoryStatus,
)
from ..manager import MemoryStore
from .permission_service import PermissionService
from .operation_log_service import OperationLogService


class InvalidPromotionStateError(Exception):
    """Raised when a promotion request is in an invalid state for the operation."""


class PromotionRequestStoreProtocol(Protocol):
    """
    Minimal interface expected by PromotionService.

    Your concrete PromotionRequestStore should implement these methods.
    """

    def create(self, request: PromotionRequest) -> PromotionRequest:
        ...

    def get_by_id(self, request_id: str) -> PromotionRequest:
        ...

    def replace(self, request: PromotionRequest) -> PromotionRequest:
        ...

    def list_all(self) -> list[PromotionRequest]:
        ...

    def list_pending(self) -> list[PromotionRequest]:
        ...


class PromotionService:
    """
    Business services for private-to-shared memory promotion.

    Workflow:
    1. Worker proposes a private memory for sharing.
    2. Critic reviews the promotion request and gives a recommendation.
    3. Coordinator approves or rejects the request.
    4. If approved, the private memory is promoted to shared memory.
    5. All key operations are recorded by OperationLogService.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        request_store: PromotionRequestStoreProtocol,
        operation_log_service: OperationLogService,
        permission_service: Optional[PermissionService] = None,
    ) -> None:
        self.memory_store = memory_store
        self.request_store = request_store
        self.operation_log_service = operation_log_service
        self.permission_service = permission_service or PermissionService()

    # ------------------------------------------------------------------
    # Submit promotion request
    # ------------------------------------------------------------------

    def submit_promotion_request(
        self,
        agent: Agent,
        memory_id: str,
        reason: str,
    ) -> PromotionRequest:
        """
        Submit a request to promote a private memory into shared memory.

        Baseline rule:
        - The memory must be private.
        - The memory must be active.
        - The requester must be allowed to read the memory.
        - Normally this means the requester is the memory owner.
        """

        memory = self.memory_store.get_by_id(memory_id)

        self.permission_service.assert_can_propose_promotion(
            agent=agent,
            memory=memory,
        )

        request = PromotionRequest(
            memory_id=memory.memory_id,
            proposed_by_agent_id=agent.agent_id,
            reason=reason,
        )

        created_request = self.request_store.create(request)

        self.operation_log_service.record_promotion_submitted(
            request=created_request,
            memory=memory,
            actor_agent=agent,
            reason=reason,
        )

        return created_request

    # ------------------------------------------------------------------
    # Critic / coordinator review
    # ------------------------------------------------------------------

    def review_promotion_request(
        self,
        agent: Agent,
        request_id: str,
        comment: str,
    ) -> PromotionRequest:
        """
        Review a promotion request.

        This is mainly for critic recommendation.
        It does not approve or reject the request.
        The request remains pending.
        """

        self.permission_service.assert_can_review_promotion(agent)

        request = self.request_store.get_by_id(request_id)
        self._assert_request_pending(request)

        memory = self.memory_store.get_by_id(request.memory_id)

        request.reviewed_by_agent_id = agent.agent_id
        request.review_comment = comment
        request.reviewed_at = datetime.now(timezone.utc)

        updated_request = self.request_store.replace(request)

        self.operation_log_service.record_promotion_reviewed(
            request=updated_request,
            memory=memory,
            reviewer_agent=agent,
            comment=comment,
        )

        return updated_request

    # ------------------------------------------------------------------
    # Coordinator approval
    # ------------------------------------------------------------------

    def approve_promotion_request(
        self,
        agent: Agent,
        request_id: str,
        comment: Optional[str] = None,
    ) -> MemoryItem:
        """
        Approve a promotion request and promote the target memory to shared memory.

        Baseline rule:
        - Only coordinator can approve.
        - The target memory must still be private and active.
        - The memory becomes shared.
        - The coordinator becomes the owner/manager of the shared memory.
        - All agents can read it.
        - Only the coordinator can write it.
        """

        self.permission_service.assert_can_approve_promotion(agent)

        request = self.request_store.get_by_id(request_id)
        self._assert_request_pending(request)

        memory = self.memory_store.get_by_id(request.memory_id)
        self._assert_memory_promotable(memory)

        before_state = self.operation_log_service.snapshot_memory_access(memory)

        request.approve(
            reviewer_agent_id=agent.agent_id,
            comment=comment or "Promotion approved by coordinator.",
        )

        updated_request = self.request_store.replace(request)

        self.operation_log_service.record_promotion_approved(
            request=updated_request,
            memory=memory,
            actor_agent=agent,
            comment=comment,
        )

        promoted_memory = self.memory_store.update_scope(
            memory_id=memory.memory_id,
            scope=MemoryScope.SHARED,
            owner_agent_id=agent.agent_id,
            readable_by=["*"],
            writable_by=[agent.agent_id],
        )

        after_state = self.operation_log_service.snapshot_memory_access(promoted_memory)

        promote_record = self.operation_log_service.record_memory_promoted(
            memory=promoted_memory,
            actor_agent=agent,
            request=updated_request,
            before_state=before_state,
            after_state=after_state,
            reason=comment or "Memory promoted from private to shared.",
            reviewer_agent=agent,
        )

        promoted_memory.metadata.last_operation_id = promote_record.record_id

        if promote_record.record_id not in promoted_memory.metadata.related_operation_ids:
            promoted_memory.metadata.related_operation_ids.append(promote_record.record_id)

        promoted_memory = self.memory_store.replace(promoted_memory)

        return promoted_memory

    # ------------------------------------------------------------------
    # Coordinator rejection
    # ------------------------------------------------------------------

    def reject_promotion_request(
        self,
        agent: Agent,
        request_id: str,
        comment: Optional[str] = None,
    ) -> PromotionRequest:
        """
        Reject a promotion request.

        Baseline rule:
        - Only coordinator can make the final rejection decision.
        - The target memory remains private.
        """

        self.permission_service.assert_can_reject_promotion(agent)

        request = self.request_store.get_by_id(request_id)
        self._assert_request_pending(request)

        memory = self.memory_store.get_by_id(request.memory_id)

        request.reject(
            reviewer_agent_id=agent.agent_id,
            comment=comment or "Promotion rejected by coordinator.",
        )

        updated_request = self.request_store.replace(request)

        self.operation_log_service.record_promotion_rejected(
            request=updated_request,
            memory=memory,
            actor_agent=agent,
            comment=comment,
        )

        return updated_request

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_request(self, request_id: str) -> PromotionRequest:
        """
        Get a promotion request by ID.
        """

        return self.request_store.get_by_id(request_id)

    def list_all_requests(self) -> list[PromotionRequest]:
        """
        List all promotion requests.
        """

        return self.request_store.list_all()

    def list_pending_requests(self) -> list[PromotionRequest]:
        """
        List pending promotion requests.
        """

        return self.request_store.list_pending()

    def list_requests_by_memory_id(
        self,
        memory_id: str,
    ) -> list[PromotionRequest]:
        """
        List all promotion requests related to a memory.
        """

        return [
            request
            for request in self.request_store.list_all()
            if request.memory_id == memory_id
        ]

    def list_requests_by_agent_id(
        self,
        agent_id: str,
    ) -> list[PromotionRequest]:
        """
        List all promotion requests proposed by an agent.
        """

        return [
            request
            for request in self.request_store.list_all()
            if request.proposed_by_agent_id == agent_id
        ]

    # ------------------------------------------------------------------
    # Internal validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_request_pending(request: PromotionRequest) -> None:
        """
        Ensure the promotion request is pending.
        """

        if request.status != PromotionStatus.PENDING:
            raise InvalidPromotionStateError(
                f"Promotion request '{request.request_id}' is not pending. "
                f"Current status: {request.status}."
            )

    @staticmethod
    def _assert_memory_promotable(memory: MemoryItem) -> None:
        """
        Ensure the memory can be promoted.
        """

        if memory.metadata.scope != MemoryScope.PRIVATE:
            raise InvalidPromotionStateError(
                f"Memory '{memory.memory_id}' cannot be promoted because it is not private."
            )

        if memory.metadata.status != MemoryStatus.ACTIVE:
            raise InvalidPromotionStateError(
                f"Memory '{memory.memory_id}' cannot be promoted because it is not active. "
                f"Current status: {memory.metadata.status}."
            )