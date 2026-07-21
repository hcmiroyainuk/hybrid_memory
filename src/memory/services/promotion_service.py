from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Protocol

from ..entities import (
    Agent,
    MemoryItem,
    MemoryScope,
    MemoryStatus,
    PromotionRequest,
    PromotionStatus,
)
from ..manager import MemoryStore
from .operation_log_service import OperationLogService
from .permission_service import PermissionService


class InvalidPromotionStateError(Exception):
    """Raised when a promotion request is in an invalid workflow state."""


class PromotionRequestStoreProtocol(Protocol):
    """Persistence interface required by ``PromotionService``."""

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
    Govern private-to-shared memory promotion.

    Required workflow:
        Worker owner submits an active private memory
        -> Critic reviews and provides a recommendation
        -> Coordinator approves or rejects
        -> approved memory becomes shared with an explicit read ACL

    Role and object permissions are delegated to ``PermissionService``.
    Promotion requests are persisted through ``PromotionRequestStore`` and all
    governance actions are recorded through ``OperationLogService``.
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
    # Worker submission
    # ------------------------------------------------------------------

    def submit_promotion_request(
        self,
        agent: Agent,
        memory_id: str,
        reason: str,
    ) -> PromotionRequest:
        """
        Submit an active private memory for possible sharing.

        Only the Worker that owns the private memory may submit it. A memory
        may have at most one pending promotion request at a time.
        """

        normalized_reason = reason.strip()
        if not normalized_reason:
            raise ValueError("Promotion reason cannot be empty.")

        memory = self.memory_store.get_by_id(memory_id)

        self.permission_service.assert_can_propose_promotion(
            agent=agent,
            memory=memory,
        )

        self._assert_no_pending_request_for_memory(memory.memory_id)

        request = PromotionRequest(
            memory_id=memory.memory_id,
            proposed_by_agent_id=agent.agent_id,
            reason=normalized_reason,
        )

        created_request = self.request_store.create(request)

        self.operation_log_service.record_promotion_submitted(
            request=created_request,
            memory=memory,
            actor_agent=agent,
            reason=normalized_reason,
        )

        return created_request

    # ------------------------------------------------------------------
    # Critic recommendation
    # ------------------------------------------------------------------

    def review_promotion_request(
        self,
        agent: Agent,
        request_id: str,
        comment: str,
    ) -> PromotionRequest:
        """
        Record the Critic's recommendation.

        This operation does not approve or reject the request. Its status
        remains ``PENDING`` until the Coordinator makes the final decision.
        """

        normalized_comment = comment.strip()
        if not normalized_comment:
            raise ValueError("Critic review comment cannot be empty.")

        self.permission_service.assert_can_review_promotion(agent)

        request = self.request_store.get_by_id(request_id)
        self._assert_request_pending(request)
        self._assert_request_not_reviewed(request)

        memory = self.memory_store.get_by_id(request.memory_id)
        self._assert_memory_promotable(memory)

        request.reviewed_by_agent_id = agent.agent_id
        request.review_comment = normalized_comment
        request.reviewed_at = datetime.now(timezone.utc)

        updated_request = self.request_store.replace(request)

        self.operation_log_service.record_promotion_reviewed(
            request=updated_request,
            memory=memory,
            reviewer_agent=agent,
            comment=normalized_comment,
        )

        return updated_request

    # ------------------------------------------------------------------
    # Coordinator approval
    # ------------------------------------------------------------------

    def approve_promotion_request(
        self,
        agent: Agent,
        request_id: str,
        readable_by: list[str],
        comment: Optional[str] = None,
    ) -> MemoryItem:
        """
        Approve a Critic-reviewed request and promote the memory to shared.

        ``readable_by`` is the Coordinator's explicit access decision. It may
        contain one or more Agent IDs, or exactly ``["*"]`` for global shared
        access. Only the Coordinator is added to ``writable_by``.

        The current ``PromotionRequest`` schema stores the Critic's review in
        ``reviewed_by_agent_id`` / ``review_comment``. Those fields are kept
        unchanged after the Coordinator decision; the Coordinator decision is
        represented by ``status`` and by the operation log.
        """

        self.permission_service.assert_can_approve_promotion(agent)

        request = self.request_store.get_by_id(request_id)
        self._assert_request_pending(request)
        self._assert_request_reviewed(request)

        memory = self.memory_store.get_by_id(request.memory_id)
        self._assert_memory_promotable(memory)

        normalized_readers = self._normalize_readable_by(readable_by)
        decision_comment = (
            comment.strip()
            if comment is not None and comment.strip()
            else "Promotion approved by Coordinator."
        )

        before_state = self.operation_log_service.snapshot_memory_access(
            memory
        )

        # Preserve the Critic review fields. The Coordinator decision is
        # represented by status plus the approval audit record.
        request.status = PromotionStatus.APPROVED
        updated_request = self.request_store.replace(request)

        self.operation_log_service.record_promotion_approved(
            request=updated_request,
            memory=memory,
            actor_agent=agent,
            comment=decision_comment,
        )

        promoted_memory = self.memory_store.update_scope(
            memory_id=memory.memory_id,
            scope=MemoryScope.SHARED,
            owner_agent_id=agent.agent_id,
            readable_by=normalized_readers,
            writable_by=[agent.agent_id],
        )

        after_state = self.operation_log_service.snapshot_memory_access(
            promoted_memory
        )

        promote_record = self.operation_log_service.record_memory_promoted(
            memory=promoted_memory,
            actor_agent=agent,
            request=updated_request,
            before_state=before_state,
            after_state=after_state,
            reason=decision_comment,
            reviewer_agent=agent,
        )

        promoted_memory.metadata.last_operation_id = promote_record.record_id

        if (
            promote_record.record_id
            not in promoted_memory.metadata.related_operation_ids
        ):
            promoted_memory.metadata.related_operation_ids.append(
                promote_record.record_id
            )

        return self.memory_store.replace(promoted_memory)

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
        Reject a Critic-reviewed request.

        The target memory remains private. Critic review fields are preserved;
        the final Coordinator decision is represented by request status and the
        rejection operation log.
        """

        self.permission_service.assert_can_reject_promotion(agent)

        request = self.request_store.get_by_id(request_id)
        self._assert_request_pending(request)
        self._assert_request_reviewed(request)

        memory = self.memory_store.get_by_id(request.memory_id)
        self._assert_memory_still_private(memory)

        decision_comment = (
            comment.strip()
            if comment is not None and comment.strip()
            else "Promotion rejected by Coordinator."
        )

        request.status = PromotionStatus.REJECTED
        updated_request = self.request_store.replace(request)

        self.operation_log_service.record_promotion_rejected(
            request=updated_request,
            memory=memory,
            actor_agent=agent,
            comment=decision_comment,
        )

        return updated_request

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_request(self, request_id: str) -> PromotionRequest:
        """Return a promotion request by ID."""

        return self.request_store.get_by_id(request_id)

    def list_all_requests(self) -> list[PromotionRequest]:
        """Return all promotion requests."""

        return self.request_store.list_all()

    def list_pending_requests(self) -> list[PromotionRequest]:
        """Return all requests awaiting a Coordinator decision."""

        return self.request_store.list_pending()

    def list_requests_by_memory_id(
        self,
        memory_id: str,
    ) -> list[PromotionRequest]:
        """Return every promotion request associated with one memory."""

        return [
            request
            for request in self.request_store.list_all()
            if request.memory_id == memory_id
        ]

    def list_requests_by_agent_id(
        self,
        agent_id: str,
    ) -> list[PromotionRequest]:
        """Return requests submitted by one Worker."""

        return [
            request
            for request in self.request_store.list_all()
            if request.proposed_by_agent_id == agent_id
        ]

    # ------------------------------------------------------------------
    # Internal validation helpers
    # ------------------------------------------------------------------

    def _assert_no_pending_request_for_memory(
        self,
        memory_id: str,
    ) -> None:
        pending_request = next(
            (
                request
                for request in self.request_store.list_pending()
                if request.memory_id == memory_id
            ),
            None,
        )

        if pending_request is not None:
            raise InvalidPromotionStateError(
                f"Memory '{memory_id}' already has pending promotion "
                f"request '{pending_request.request_id}'."
            )

    @staticmethod
    def _assert_request_pending(request: PromotionRequest) -> None:
        if request.status != PromotionStatus.PENDING:
            raise InvalidPromotionStateError(
                f"Promotion request '{request.request_id}' is not pending. "
                f"Current status: {request.status}."
            )

    @staticmethod
    def _assert_request_not_reviewed(request: PromotionRequest) -> None:
        if (
            request.reviewed_by_agent_id is not None
            or request.reviewed_at is not None
            or request.review_comment is not None
        ):
            raise InvalidPromotionStateError(
                f"Promotion request '{request.request_id}' has already "
                "received a Critic review."
            )

    @staticmethod
    def _assert_request_reviewed(request: PromotionRequest) -> None:
        if request.reviewed_by_agent_id is None:
            raise InvalidPromotionStateError(
                f"Promotion request '{request.request_id}' must be reviewed "
                "by the Critic before the Coordinator decides."
            )

        if request.reviewed_at is None:
            raise InvalidPromotionStateError(
                f"Promotion request '{request.request_id}' has no Critic "
                "review timestamp."
            )

        if not request.review_comment or not request.review_comment.strip():
            raise InvalidPromotionStateError(
                f"Promotion request '{request.request_id}' has no Critic "
                "recommendation comment."
            )

    @staticmethod
    def _assert_memory_promotable(memory: MemoryItem) -> None:
        if memory.metadata.scope != MemoryScope.PRIVATE:
            raise InvalidPromotionStateError(
                f"Memory '{memory.memory_id}' cannot be promoted because "
                "it is not private."
            )

        if memory.metadata.status != MemoryStatus.ACTIVE:
            raise InvalidPromotionStateError(
                f"Memory '{memory.memory_id}' cannot be promoted because "
                f"it is not active. Current status: "
                f"{memory.metadata.status}."
            )

    @staticmethod
    def _assert_memory_still_private(memory: MemoryItem) -> None:
        if memory.metadata.scope != MemoryScope.PRIVATE:
            raise InvalidPromotionStateError(
                f"Promotion request cannot be rejected because memory "
                f"'{memory.memory_id}' is no longer private."
            )

    @staticmethod
    def _normalize_readable_by(readable_by: list[str]) -> list[str]:
        if not isinstance(readable_by, list):
            raise TypeError("readable_by must be a list of Agent IDs.")

        normalized = list(
            dict.fromkeys(
                value.strip()
                for value in readable_by
                if isinstance(value, str) and value.strip()
            )
        )

        if not normalized:
            raise ValueError(
                "readable_by must contain at least one Agent ID or '*'."
            )

        if "*" in normalized and normalized != ["*"]:
            raise ValueError(
                "The wildcard '*' must be used alone in readable_by."
            )

        return normalized