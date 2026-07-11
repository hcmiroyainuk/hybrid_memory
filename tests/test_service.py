from __future__ import annotations

import pytest

from src.memory.entities import (
    Agent,
    MemoryScope,
    MemoryStatus,
    MemoryType,
    PromotionRequest,
    PromotionStatus,
)

from src.memory.manager import (
    MemoryStore,
    OperationLogStore,
)

from src.memory.services import (
    PermissionService,
    PermissionDeniedError,
    OperationLogService,
    MemoryService,
    PromotionService,
)


class InMemoryPromotionRequestStore:
    """
    Temporary request store for integration testing.

    Later, you can replace this with a JSON-backed PromotionRequestStore.
    """

    def __init__(self) -> None:
        self._requests: dict[str, PromotionRequest] = {}

    def create(self, request: PromotionRequest) -> PromotionRequest:
        if request.request_id in self._requests:
            raise ValueError(
                f"Promotion request '{request.request_id}' already exists."
            )

        self._requests[request.request_id] = request
        return request

    def get_by_id(self, request_id: str) -> PromotionRequest:
        if request_id not in self._requests:
            raise ValueError(
                f"Promotion request '{request_id}' was not found."
            )

        return self._requests[request_id]

    def replace(self, request: PromotionRequest) -> PromotionRequest:
        if request.request_id not in self._requests:
            raise ValueError(
                f"Promotion request '{request.request_id}' was not found."
            )

        self._requests[request.request_id] = request
        return request

    def list_all(self) -> list[PromotionRequest]:
        return list(self._requests.values())

    def list_pending(self) -> list[PromotionRequest]:
        return [
            request
            for request in self._requests.values()
            if request.status == PromotionStatus.PENDING
        ]

    def clear(self) -> None:
        self._requests.clear()


def test_private_to_shared_memory_governance_workflow() -> None:
    """
    Integration test for:

    - PermissionService
    - OperationLogService
    - MemoryService
    - PromotionService
    """

    # ------------------------------------------------------------------
    # 1. Initialize stores
    # ------------------------------------------------------------------

    memory_store = MemoryStore(file_path="data/test_memories.json")
    operation_log_store = OperationLogStore(
        file_path="data/test_operation_logs.json"
    )
    request_store = InMemoryPromotionRequestStore()

    memory_store.clear()
    operation_log_store.clear()
    request_store.clear()

    # ------------------------------------------------------------------
    # 2. Initialize services
    # ------------------------------------------------------------------

    permission_service = PermissionService()
    operation_log_service = OperationLogService(
        operation_log_store=operation_log_store
    )

    memory_service = MemoryService(
        memory_store=memory_store,
        operation_log_store=operation_log_store,
        memory_retriever=None,
        permission_service=permission_service,
    )

    promotion_service = PromotionService(
        memory_store=memory_store,
        request_store=request_store,
        operation_log_service=operation_log_service,
        permission_service=permission_service,
    )

    # ------------------------------------------------------------------
    # 3. Initialize agents
    # ------------------------------------------------------------------

    worker_a = Agent.worker(
        agent_id="agent_worker_a",
        name="Worker A",
    )

    worker_b = Agent.worker(
        agent_id="agent_worker_b",
        name="Worker B",
    )

    critic = Agent.critic(
        agent_id="agent_critic",
    )

    coordinator = Agent.coordinator(
        agent_id="agent_coordinator",
    )

    # ------------------------------------------------------------------
    # 4. Worker A creates private memory
    # ------------------------------------------------------------------

    private_memory = memory_service.create_private_memory(
        agent=worker_a,
        content=(
            "Workers cannot directly write shared memory. "
            "They must submit a promotion request first."
        ),
        summary="Workers need approval before writing shared memory.",
        memory_type=MemoryType.RULE,
        tags=["permission", "shared_memory", "promotion"],
        importance=0.9,
        confidence=1.0,
        reason="Worker A created a private access-control rule.",
    )

    assert private_memory.metadata.scope == MemoryScope.PRIVATE
    assert private_memory.metadata.status == MemoryStatus.ACTIVE
    assert private_memory.metadata.owner_agent_id == worker_a.agent_id

    # ------------------------------------------------------------------
    # 5. PermissionService: Worker A can read, Worker B cannot read
    # ------------------------------------------------------------------

    assert permission_service.can_read_memory(worker_a, private_memory) is True
    assert permission_service.can_read_memory(worker_b, private_memory) is False

    readable_memory = memory_service.get_memory(
        agent=worker_a,
        memory_id=private_memory.memory_id,
    )

    assert readable_memory.memory_id == private_memory.memory_id

    with pytest.raises(PermissionDeniedError):
        memory_service.get_memory(
            agent=worker_b,
            memory_id=private_memory.memory_id,
        )

    # ------------------------------------------------------------------
    # 6. Worker A submits promotion request
    # ------------------------------------------------------------------

    request = promotion_service.submit_promotion_request(
        agent=worker_a,
        memory_id=private_memory.memory_id,
        reason="This rule should be shared because it affects all workers.",
    )

    assert request.memory_id == private_memory.memory_id
    assert request.proposed_by_agent_id == worker_a.agent_id
    assert request.status == PromotionStatus.PENDING

    # ------------------------------------------------------------------
    # 7. Critic reviews promotion request
    # ------------------------------------------------------------------

    reviewed_request = promotion_service.review_promotion_request(
        agent=critic,
        request_id=request.request_id,
        comment="This memory is general and should be promoted to shared memory.",
    )

    assert reviewed_request.status == PromotionStatus.PENDING
    assert reviewed_request.reviewed_by_agent_id == critic.agent_id
    assert reviewed_request.review_comment is not None

    # Critic can review but cannot approve final promotion.
    assert permission_service.can_review_promotion(critic) is True
    assert permission_service.can_approve_promotion(critic) is False

    # ------------------------------------------------------------------
    # 8. Coordinator approves promotion
    # ------------------------------------------------------------------

    promoted_memory = promotion_service.approve_promotion_request(
        agent=coordinator,
        request_id=request.request_id,
        comment="Approved. This is a global access-control rule.",
    )

    assert promoted_memory.memory_id == private_memory.memory_id
    assert promoted_memory.metadata.scope == MemoryScope.SHARED
    assert promoted_memory.metadata.status == MemoryStatus.ACTIVE
    assert promoted_memory.metadata.owner_agent_id == coordinator.agent_id
    assert "*" in promoted_memory.metadata.readable_by
    assert coordinator.agent_id in promoted_memory.metadata.writable_by

    approved_request = promotion_service.get_request(request.request_id)

    assert approved_request.status == PromotionStatus.APPROVED
    assert approved_request.reviewed_by_agent_id == coordinator.agent_id

    # ------------------------------------------------------------------
    # 9. Worker B can now read promoted shared memory
    # ------------------------------------------------------------------

    assert permission_service.can_read_memory(worker_b, promoted_memory) is True

    shared_memory_for_worker_b = memory_service.get_memory(
        agent=worker_b,
        memory_id=promoted_memory.memory_id,
    )

    assert shared_memory_for_worker_b.memory_id == promoted_memory.memory_id
    assert shared_memory_for_worker_b.metadata.scope == MemoryScope.SHARED

    # Worker B still cannot directly write shared memory.
    assert permission_service.can_write_memory(worker_b, promoted_memory) is False

    with pytest.raises(PermissionDeniedError):
        memory_service.update_memory_content(
            agent=worker_b,
            memory_id=promoted_memory.memory_id,
            content="Worker B tries to edit shared memory directly.",
            reason="This should be blocked.",
        )

    # Coordinator can update shared memory.
    updated_shared_memory = memory_service.update_memory_content(
        agent=coordinator,
        memory_id=promoted_memory.memory_id,
        content=(
            "Workers cannot directly write shared memory. "
            "They must submit a promotion request and receive coordinator approval."
        ),
        summary="Coordinator approval is required before shared memory writes.",
        reason="Coordinator clarified the shared-memory access-control rule.",
    )

    assert updated_shared_memory.metadata.scope == MemoryScope.SHARED
    assert "coordinator approval" in updated_shared_memory.content.lower()

    # ------------------------------------------------------------------
    # 10. OperationLogService: logs should exist
    # ------------------------------------------------------------------

    all_logs = operation_log_service.list_all_operations()

    assert len(all_logs) >= 5

    memory_history = operation_log_service.list_memory_history(
        promoted_memory.memory_id
    )

    assert len(memory_history) >= 5

    request_logs = operation_log_service.list_request_operations(
        request.request_id
    )

    assert len(request_logs) >= 3

    # ------------------------------------------------------------------
    # 11. Final stored memory state should be shared and active
    # ------------------------------------------------------------------

    final_memory = memory_store.get_by_id(promoted_memory.memory_id)

    assert final_memory.metadata.scope == MemoryScope.SHARED
    assert final_memory.metadata.status == MemoryStatus.ACTIVE
    assert final_memory.metadata.owner_agent_id == coordinator.agent_id