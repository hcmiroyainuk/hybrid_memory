from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from src.memory.entities import (
    Agent,
    MemoryItem,
    MemoryMetadata,
    MemoryOperationType,
    MemoryScope,
    MemoryType,
    PromotionRequest,
    PromotionStatus,
    SourceType,
)
from src.memory.lifecycle import ConflictDetector
from src.memory.lifecycle.candidate_memory_builder import (
    CandidateMemoryBuilder,
)
from src.memory.lifecycle.conflict_schema import (
    ConflictCheckResult,
)
from src.memory.lifecycle.governed_memory_lifecycle import (
    GovernedMemoryLifecycle,
)
from src.memory.lifecycle.memory_write_controller import (
    MemoryWriteController,
)
from src.memory.manager.memory_store import MemoryStore
from src.memory.manager.operation_log_store import (
    OperationLogStore,
)
from src.memory.services import OperationLogService
from src.memory.services.memory_service import MemoryService

from src.memory.services.permission_service import (
    PermissionService,
)
from src.memory.services.promotion_service import (
    PromotionService,
)


# ----------------------------------------------------------------------
# Test-only PromotionRequestStore
# ----------------------------------------------------------------------

class InMemoryPromotionRequestStore:
    """
    Minimal in-memory implementation of PromotionRequestStoreProtocol.

    It is used only by this integration test so that promotion requests do
    not need another JSON file.
    """

    def __init__(self) -> None:
        self._requests: dict[str, PromotionRequest] = {}

    def create(
        self,
        request: PromotionRequest,
    ) -> PromotionRequest:
        if request.request_id in self._requests:
            raise ValueError(
                f"Promotion request {request.request_id!r} already exists."
            )

        self._requests[request.request_id] = deepcopy(request)
        return deepcopy(request)

    def get_by_id(
        self,
        request_id: str,
    ) -> PromotionRequest:
        if request_id not in self._requests:
            raise KeyError(
                f"Promotion request {request_id!r} was not found."
            )

        return deepcopy(self._requests[request_id])

    def replace(
        self,
        request: PromotionRequest,
    ) -> PromotionRequest:
        if request.request_id not in self._requests:
            raise KeyError(
                f"Promotion request {request.request_id!r} was not found."
            )

        self._requests[request.request_id] = deepcopy(request)
        return deepcopy(request)

    def list_all(self) -> list[PromotionRequest]:
        return [
            deepcopy(request)
            for request in self._requests.values()
        ]

    def list_pending(self) -> list[PromotionRequest]:
        return [
            deepcopy(request)
            for request in self._requests.values()
            if request.status == PromotionStatus.PENDING
        ]


# ----------------------------------------------------------------------
# Deterministic conflict detector
# ----------------------------------------------------------------------

class NoConflictDetector(ConflictDetector):
    """
    Deterministic conflict detector used in lifecycle tests.
    """

    def check_conflict(
        self,
        candidate_memory: MemoryItem,
        existing_memories: list[MemoryItem],
    ) -> ConflictCheckResult:
        return ConflictCheckResult.no_conflict(
            candidate_memory_id=candidate_memory.memory_id,
            message="No conflict detected by lifecycle integration test.",
        )


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def worker_b() -> Agent:
    return Agent.worker(
        agent_id="worker_b",
    )


@pytest.fixture
def critic() -> Agent:
    return Agent.critic(
        agent_id="critic",
    )


@pytest.fixture
def coordinator() -> Agent:
    return Agent.coordinator(
        agent_id="coordinator",
    )


@pytest.fixture
def invalid_coordinator() -> Agent:
    """
    A worker that is intentionally passed into the coordinator position.
    """

    return Agent.worker(
        agent_id="not_a_coordinator",
    )


@pytest.fixture
def candidate_memory() -> MemoryItem:
    """
    Temporary candidate produced before controlled persistence.

    MemoryWriteController will pass its semantic fields to MemoryService,
    which creates a new formally persisted MemoryItem with a new ID.
    """

    metadata = MemoryMetadata.private(
        owner_agent_id="worker_b",
        created_by_agent_id="worker_b",
        memory_type=MemoryType.FACT,
        tags=[
            "test",
            "qa",
            "candidate",
            "governed_rag_qa",
        ],
        importance=0.7,
        confidence=0.9,
        source_task_id="test_task_001",
        source_message_ids=[],
        source_type=SourceType.AGENT_OUTPUT,
    )

    return MemoryItem(
        memory_id="candidate_lifecycle_001",
        content="\n".join(
            [
                "Memory source: governed_rag_qa_workflow",
                "",
                "Question: What is the capital of France?",
                "Final answer: Paris",
                "Worker A answer: Paris",
                "Worker B answer: Paris",
                "Critic recommendation: use_worker_b",
            ]
        ),
        summary="France's capital is Paris.",
        metadata=metadata,
    )


@pytest.fixture
def lifecycle_environment(tmp_path):
    # 1. 创建底层 Store
    memory_store = MemoryStore(
        tmp_path / "memories.json"
    )

    operation_log_store = OperationLogStore(
        tmp_path / "operation_logs.json"
    )

    request_store = InMemoryPromotionRequestStore()

    # 2. 创建共享的权限服务
    permission_service = PermissionService()

    # 3. 创建日志业务服务
    operation_log_service = OperationLogService(
        operation_log_store=operation_log_store,
    )

    # 4. 创建 MemoryService
    memory_service = MemoryService(
        memory_store=memory_store,
        operation_log_store=operation_log_store,
        memory_retriever=None,
        permission_service=permission_service,
    )

    # 5. 创建 PromotionService
    promotion_service = PromotionService(
        memory_store=memory_store,
        request_store=request_store,
        operation_log_service=operation_log_service,
        permission_service=permission_service,
    )

    # 6. 创建写入控制器
    write_controller = MemoryWriteController(
        memory_service=memory_service,
    )

    # 7. 创建最终 Lifecycle
    lifecycle = GovernedMemoryLifecycle(
        memory_service=memory_service,
        conflict_detector=NoConflictDetector(),
        memory_write_controller=write_controller,
        operation_log_service=operation_log_service,
        promotion_service=promotion_service,
    )

    # 8. 把测试需要的对象返回
    return {
        "memory_store": memory_store,
        "operation_log_store": operation_log_store,
        "operation_log_service": operation_log_service,
        "permission_service": permission_service,
        "memory_service": memory_service,
        "request_store": request_store,
        "promotion_service": promotion_service,
        "write_controller": write_controller,
        "lifecycle": lifecycle,
    }

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def _patch_candidate_builder(
    monkeypatch: pytest.MonkeyPatch,
    candidate: MemoryItem,
) -> None:
    """
    Replace CandidateMemoryBuilder only for lifecycle orchestration tests.

    CandidateMemoryBuilder should have its own unit tests. Returning a fixed
    candidate keeps this test independent from workflow final-state schemas.
    """

    def fake_build_from_final_state(
        final_state: dict[str, Any],
        **kwargs: Any,
    ) -> MemoryItem:
        return candidate.model_copy(deep=True)

    monkeypatch.setattr(
        CandidateMemoryBuilder,
        "build_from_final_state",
        staticmethod(fake_build_from_final_state),
    )


def _enum_value(value: Any) -> str:
    if hasattr(value, "value"):
        return str(value.value)

    return str(value)


# ----------------------------------------------------------------------
# Complete promotion lifecycle
# ----------------------------------------------------------------------

def test_governed_lifecycle_completes_real_promotion_flow(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_environment: dict[str, Any],
    candidate_memory: MemoryItem,
    worker_b: Agent,
    critic: Agent,
    coordinator: Agent,
) -> None:
    """
    No conflict and auto promotion enabled:

    candidate
        -> private write
        -> promotion submitted by Worker B
        -> reviewed by Critic
        -> approved by Coordinator
        -> memory becomes shared
    """

    _patch_candidate_builder(
        monkeypatch,
        candidate_memory,
    )

    lifecycle = lifecycle_environment["lifecycle"]
    memory_store = lifecycle_environment["memory_store"]
    operation_log_store = lifecycle_environment[
        "operation_log_store"
    ]
    request_store = lifecycle_environment["request_store"]

    result = lifecycle.process_qa_result(
        final_state={},
        worker_agent=worker_b,
        critic_agent=critic,
        coordinator_agent=coordinator,
        auto_promote=True,
    )

    print("\n========== GOVERNED LIFECYCLE RESULT ==========")
    print(result)

    assert result.success is True

    assert (
        result.candidate_memory_id
        == candidate_memory.memory_id
    )

    assert result.private_memory_written is True
    assert result.private_memory_id is not None

    assert result.promotion_request_id is not None
    assert result.promotion_status == "approved"

    # Promotion modifies the same persisted memory.
    assert result.shared_memory_id == result.private_memory_id

    # One CREATE plus four promotion-related logs.
    assert result.operation_log_count == 5

    # Only one formal memory should exist.
    assert memory_store.count() == 1

    promoted_memory = memory_store.get_by_id(
        result.shared_memory_id
    )

    assert (
        _enum_value(promoted_memory.metadata.scope)
        == MemoryScope.SHARED.value
    )

    assert (
        promoted_memory.metadata.owner_agent_id
        == coordinator.agent_id
    )

    # Original creator remains Worker B.
    assert (
        promoted_memory.metadata.created_by_agent_id
        == worker_b.agent_id
    )

    assert promoted_memory.metadata.readable_by == ["*"]

    assert promoted_memory.metadata.writable_by == [
        coordinator.agent_id
    ]

    assert promoted_memory.content == candidate_memory.content
    assert promoted_memory.summary == candidate_memory.summary

    # --------------------------------------------------------------
    # Verify final promotion request
    # --------------------------------------------------------------

    requests = request_store.list_all()

    assert len(requests) == 1

    approved_request = requests[0]

    assert (
        approved_request.request_id
        == result.promotion_request_id
    )

    assert (
        approved_request.memory_id
        == result.private_memory_id
    )

    assert (
        approved_request.proposed_by_agent_id
        == worker_b.agent_id
    )

    assert approved_request.status == PromotionStatus.APPROVED

    # Coordinator approval overwrites the final reviewer field.
    assert (
        approved_request.reviewed_by_agent_id
        == coordinator.agent_id
    )

    assert request_store.list_pending() == []

    # --------------------------------------------------------------
    # Verify operation logs
    # --------------------------------------------------------------

    logs = operation_log_store.list_all()

    assert len(logs) == 5

    operation_types = [
        log.operation_type
        for log in logs
    ]

    assert operation_types.count(
        MemoryOperationType.CREATE
    ) == 1

    assert operation_types.count(
        MemoryOperationType.UPDATE
    ) == 1

    assert operation_types.count(
        MemoryOperationType.PROMOTE
    ) == 3

    # Expected append order:
    # 1. private memory creation
    # 2. promotion submitted
    # 3. promotion reviewed
    # 4. promotion approved
    # 5. actual private -> shared scope change
    assert logs[0].actor_agent_id == worker_b.agent_id
    assert logs[1].actor_agent_id == worker_b.agent_id
    assert logs[2].actor_agent_id == critic.agent_id
    assert logs[3].actor_agent_id == coordinator.agent_id
    assert logs[4].actor_agent_id == coordinator.agent_id

    assert logs[1].description == "Promotion request submitted."
    assert logs[2].description == "Promotion request reviewed."
    assert logs[3].description == "Promotion request approved."

    promotion_logs = logs[1:]

    assert all(
        log.related_request_id
        == result.promotion_request_id
        for log in promotion_logs
    )

    assert (
        promoted_memory.metadata.last_operation_id
        == logs[-1].record_id
    )

    assert (
        logs[-1].record_id
        in promoted_memory.metadata.related_operation_ids
    )


# ----------------------------------------------------------------------
# Auto-promotion disabled
# ----------------------------------------------------------------------

def test_governed_lifecycle_writes_private_memory_without_auto_promotion(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_environment: dict[str, Any],
    candidate_memory: MemoryItem,
    worker_b: Agent,
    critic: Agent,
    coordinator: Agent,
) -> None:
    """
    A safe memory should remain private when auto_promote=False.
    """

    _patch_candidate_builder(
        monkeypatch,
        candidate_memory,
    )

    lifecycle = lifecycle_environment["lifecycle"]
    memory_store = lifecycle_environment["memory_store"]
    operation_log_store = lifecycle_environment[
        "operation_log_store"
    ]
    request_store = lifecycle_environment["request_store"]

    result = lifecycle.process_qa_result(
        final_state={},
        worker_agent=worker_b,
        critic_agent=critic,
        coordinator_agent=coordinator,
        auto_promote=False,
    )

    print("\n========== AUTO-PROMOTION DISABLED ==========")
    print(result)

    assert result.success is True
    assert result.private_memory_written is True
    assert result.private_memory_id is not None

    assert result.promotion_request_id is None
    assert (
        result.promotion_status
        == "eligible_not_submitted"
    )
    assert result.shared_memory_id is None

    # Only private memory creation is logged.
    assert result.operation_log_count == 1
    assert operation_log_store.count() == 1

    assert request_store.list_all() == []

    stored_memory = memory_store.get_by_id(
        result.private_memory_id
    )

    assert (
        _enum_value(stored_memory.metadata.scope)
        == MemoryScope.PRIVATE.value
    )

    assert (
        stored_memory.metadata.owner_agent_id
        == worker_b.agent_id
    )

    assert stored_memory.metadata.readable_by == [
        worker_b.agent_id
    ]

    assert stored_memory.metadata.writable_by == [
        worker_b.agent_id
    ]


# ----------------------------------------------------------------------
# PromotionService not configured
# ----------------------------------------------------------------------

def test_governed_lifecycle_handles_missing_promotion_service(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_environment: dict[str, Any],
    candidate_memory: MemoryItem,
    worker_b: Agent,
    critic: Agent,
    coordinator: Agent,
) -> None:
    """
    The lifecycle should still preserve a safe candidate privately when
    PromotionService has not been configured.
    """

    _patch_candidate_builder(
        monkeypatch,
        candidate_memory,
    )

    memory_service = lifecycle_environment["memory_service"]
    write_controller = lifecycle_environment["write_controller"]
    memory_store = lifecycle_environment["memory_store"]
    operation_log_store = lifecycle_environment[
        "operation_log_store"
    ]
    operation_log_service = lifecycle_environment["operation_log_service"]

    lifecycle_without_promotion = GovernedMemoryLifecycle(
        memory_service=memory_service,
        conflict_detector=NoConflictDetector(),
        memory_write_controller=write_controller,
        operation_log_service=operation_log_service,
        promotion_service=None,
    )

    result = lifecycle_without_promotion.process_qa_result(
        final_state={},
        worker_agent=worker_b,
        critic_agent=critic,
        coordinator_agent=coordinator,
        auto_promote=True,
    )

    print("\n========== NO PROMOTION SERVICE ==========")
    print(result)

    assert result.success is True
    assert result.private_memory_written is True
    assert result.private_memory_id is not None

    assert result.promotion_request_id is None

    assert (
        result.promotion_status
        == "promotion_service_not_configured"
    )

    assert result.shared_memory_id is None

    assert result.operation_log_count == 1
    assert operation_log_store.count() == 1

    stored_memory = memory_store.get_by_id(
        result.private_memory_id
    )

    assert (
        _enum_value(stored_memory.metadata.scope)
        == MemoryScope.PRIVATE.value
    )


# ----------------------------------------------------------------------
# Coordinator permission failure
# ----------------------------------------------------------------------

def test_governed_lifecycle_returns_failure_when_approver_is_not_coordinator(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_environment: dict[str, Any],
    candidate_memory: MemoryItem,
    worker_b: Agent,
    critic: Agent,
    invalid_coordinator: Agent,
) -> None:
    """
    Real PromotionService permission checks should reject approval by a worker.

    Expected completed steps:
    - private memory created
    - promotion submitted
    - critic review recorded

    Expected blocked step:
    - coordinator approval
    """

    _patch_candidate_builder(
        monkeypatch,
        candidate_memory,
    )

    lifecycle = lifecycle_environment["lifecycle"]
    memory_store = lifecycle_environment["memory_store"]
    operation_log_store = lifecycle_environment[
        "operation_log_store"
    ]
    request_store = lifecycle_environment["request_store"]

    result = lifecycle.process_qa_result(
        final_state={},
        worker_agent=worker_b,
        critic_agent=critic,
        coordinator_agent=invalid_coordinator,
        auto_promote=True,
    )

    print("\n========== INVALID COORDINATOR RESULT ==========")
    print(result)

    assert result.success is False
    assert result.private_memory_written is True
    assert result.private_memory_id is not None

    assert result.promotion_status == "failed"
    assert result.shared_memory_id is None

    assert "Governed memory lifecycle failed" in result.message

    # CREATE + SUBMIT + REVIEW.
    # Approval fails before approval/promotion logs are added.
    assert result.operation_log_count == 3
    assert operation_log_store.count() == 3

    stored_memory = memory_store.get_by_id(
        result.private_memory_id
    )

    # Failed approval must not expose the memory as shared.
    assert (
        _enum_value(stored_memory.metadata.scope)
        == MemoryScope.PRIVATE.value
    )

    assert (
        stored_memory.metadata.owner_agent_id
        == worker_b.agent_id
    )

    requests = request_store.list_all()

    assert len(requests) == 1

    pending_request = requests[0]

    assert pending_request.status == PromotionStatus.PENDING

    assert (
        pending_request.reviewed_by_agent_id
        == critic.agent_id
    )

    assert len(request_store.list_pending()) == 1