from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from src.memory.entities import (
    Agent,
    ConflictType as OperationConflictType,
    MemoryItem,
    MemoryMetadata,
    MemoryOperationType,
    MemoryScope,
    MemoryType,
    PromotionStatus,
)
from src.memory.lifecycle.candidate_memory_builder import (
    CandidateMemoryBuilder,
)
from src.memory.lifecycle.conflict_detector import (
    ConflictDetector,
)
from src.memory.lifecycle.conflict_schema import (
    ConflictType as LifecycleConflictType,
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
from src.memory.manager.promotion_request_store import (
    PromotionRequestStore,
)
from src.memory.services.memory_service import MemoryService
from src.memory.services.operation_log_service import (
    OperationLogService,
)
from src.memory.services.permission_service import (
    PermissionService,
)
from src.memory.services.promotion_service import (
    PromotionService,
)


# ----------------------------------------------------------------------
# Test helpers
# ----------------------------------------------------------------------


def _qa_content(
    question: str,
    answer: str,
) -> str:
    """
    Build content in the structured format expected by ConflictDetector.
    """

    return (
        f"Question: {question.strip()}\n"
        f"Final answer: {answer.strip()}"
    )


def _build_candidate_memory(
    *,
    worker_agent: Agent,
    question: str,
    answer: str,
) -> MemoryItem:
    """
    Build one temporary candidate memory.

    This candidate has not yet been persisted in MemoryStore.
    """

    metadata = MemoryMetadata.private(
        owner_agent_id=worker_agent.agent_id,
        created_by_agent_id=worker_agent.agent_id,
        memory_type=MemoryType.NOTE,
        tags=["integration_test", "qa_memory"],
    )

    return MemoryItem(
        content=_qa_content(
            question=question,
            answer=answer,
        ),
        summary=f"{question} -> {answer}",
        metadata=metadata,
    )


def _patch_candidate_builder(
    monkeypatch: pytest.MonkeyPatch,
    candidate_memory: MemoryItem,
) -> None:
    """
    Force the lifecycle to receive a deterministic candidate memory.

    A deep copy is returned so lifecycle changes cannot mutate the fixture
    object shared by the test.
    """

    def fake_build_from_final_state(
        final_state: dict[str, Any],
        *,
        owner_agent_id: str,
        created_by_agent_id: str,
        tags: list[str] | None = None,
    ) -> MemoryItem:
        del final_state
        del owner_agent_id
        del created_by_agent_id
        del tags

        return deepcopy(candidate_memory)

    monkeypatch.setattr(
        CandidateMemoryBuilder,
        "build_from_final_state",
        staticmethod(fake_build_from_final_state),
    )


def _create_existing_private_memory(
    *,
    memory_service: MemoryService,
    worker_agent: Agent,
    question: str,
    answer: str,
) -> MemoryItem:
    """
    Create one existing persisted private memory.

    This is normally called before process_qa_result(), so its CREATE log
    is historical and should not be included in the lifecycle result's
    operation_log_count.
    """

    return memory_service.create_private_memory(
        agent=worker_agent,
        content=_qa_content(
            question=question,
            answer=answer,
        ),
        summary=f"{question} -> {answer}",
        memory_type=MemoryType.NOTE,
        tags=["integration_test", "existing_memory"],
        reason="Seed existing memory for lifecycle integration test.",
    )


def _build_lifecycle(
    *,
    memory_service: MemoryService,
    operation_log_service: OperationLogService,
    conflict_detector: ConflictDetector,
    write_controller: MemoryWriteController,
    promotion_service: PromotionService | None,
) -> GovernedMemoryLifecycle:
    """
    Build one lifecycle using the real project services.
    """

    return GovernedMemoryLifecycle(
        memory_service=memory_service,
        conflict_detector=conflict_detector,
        memory_write_controller=write_controller,
        operation_log_service=operation_log_service,
        promotion_service=promotion_service,
    )


# ----------------------------------------------------------------------
# Shared fixtures
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
def lifecycle_environment(
    tmp_path,
) -> dict[str, Any]:
    """
    Create a complete isolated lifecycle environment.

    Every store writes to pytest's temporary directory, so tests do not
    modify the project's real data files.
    """

    memory_store = MemoryStore(
        tmp_path / "memories.json"
    )

    operation_log_store = OperationLogStore(
        tmp_path / "operation_logs.json"
    )

    promotion_request_store = PromotionRequestStore(
        tmp_path / "promotion_requests.json"
    )

    permission_service = PermissionService()

    operation_log_service = OperationLogService(
        operation_log_store=operation_log_store,
    )

    memory_service = MemoryService(
        memory_store=memory_store,
        operation_log_store=operation_log_store,
        memory_retriever=None,
        permission_service=permission_service,
    )

    promotion_service = PromotionService(
        memory_store=memory_store,
        request_store=promotion_request_store,
        operation_log_service=operation_log_service,
        permission_service=permission_service,
    )

    conflict_detector = ConflictDetector()

    write_controller = MemoryWriteController(
        memory_service=memory_service,
    )

    lifecycle = _build_lifecycle(
        memory_service=memory_service,
        operation_log_service=operation_log_service,
        conflict_detector=conflict_detector,
        write_controller=write_controller,
        promotion_service=promotion_service,
    )

    return {
        "memory_store": memory_store,
        "operation_log_store": operation_log_store,
        "promotion_request_store": promotion_request_store,
        "permission_service": permission_service,
        "operation_log_service": operation_log_service,
        "memory_service": memory_service,
        "promotion_service": promotion_service,
        "conflict_detector": conflict_detector,
        "write_controller": write_controller,
        "lifecycle": lifecycle,
    }


# ----------------------------------------------------------------------
# 1. Safe candidate: complete private -> shared promotion
# ----------------------------------------------------------------------


def test_safe_candidate_completes_full_promotion_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_environment: dict[str, Any],
    worker_b: Agent,
    critic: Agent,
    coordinator: Agent,
) -> None:
    candidate = _build_candidate_memory(
        worker_agent=worker_b,
        question="What is the capital of France?",
        answer="Paris",
    )

    _patch_candidate_builder(
        monkeypatch,
        candidate,
    )

    lifecycle = lifecycle_environment["lifecycle"]
    memory_store = lifecycle_environment["memory_store"]
    operation_log_store = lifecycle_environment[
        "operation_log_store"
    ]
    request_store = lifecycle_environment[
        "promotion_request_store"
    ]

    result = lifecycle.process_qa_result(
        final_state={},
        worker_agent=worker_b,
        critic_agent=critic,
        coordinator_agent=coordinator,
        auto_promote=True,
    )

    assert result.success is True
    assert result.candidate_memory_id == candidate.memory_id
    assert result.private_memory_written is True
    assert result.private_memory_id is not None

    assert result.promotion_request_id is not None
    assert result.promotion_status == PromotionStatus.APPROVED.value

    assert result.shared_memory_id is not None
    assert result.shared_memory_id == result.private_memory_id

    # Current lifecycle execution:
    # 1 CREATE private memory
    # 2 promotion submitted
    # 3 Critic reviewed
    # 4 Coordinator approved
    # 5 memory scope changed to shared
    assert result.operation_log_count == 5

    assert memory_store.count() == 1
    assert request_store.count() == 1
    assert operation_log_store.count() == 5

    promoted_memory = memory_store.get_by_id(
        result.shared_memory_id
    )

    assert promoted_memory.metadata.scope == MemoryScope.SHARED
    assert (
        promoted_memory.metadata.owner_agent_id
        == coordinator.agent_id
    )
    assert promoted_memory.metadata.readable_by == ["*"]
    assert promoted_memory.metadata.writable_by == [
        coordinator.agent_id
    ]

    promotion_request = request_store.get_by_id(
        result.promotion_request_id
    )

    assert promotion_request.status == PromotionStatus.APPROVED
    assert (
        promotion_request.memory_id
        == result.shared_memory_id
    )
    assert (
        promotion_request.reviewed_by_agent_id
        == coordinator.agent_id
    )

    operations = operation_log_store.list_all()

    assert operations[0].operation_type == MemoryOperationType.CREATE
    assert operations[-1].operation_type == MemoryOperationType.PROMOTE

    related_request_operations = [
        operation
        for operation in operations
        if (
            operation.related_request_id
            == result.promotion_request_id
        )
    ]

    assert len(related_request_operations) == 4


# ----------------------------------------------------------------------
# 2. Safe candidate: auto promotion disabled
# ----------------------------------------------------------------------


def test_safe_candidate_stays_private_when_auto_promote_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_environment: dict[str, Any],
    worker_b: Agent,
    critic: Agent,
    coordinator: Agent,
) -> None:
    candidate = _build_candidate_memory(
        worker_agent=worker_b,
        question="What is the capital of Germany?",
        answer="Berlin",
    )

    _patch_candidate_builder(
        monkeypatch,
        candidate,
    )

    lifecycle = lifecycle_environment["lifecycle"]
    memory_store = lifecycle_environment["memory_store"]
    operation_log_store = lifecycle_environment[
        "operation_log_store"
    ]
    request_store = lifecycle_environment[
        "promotion_request_store"
    ]

    result = lifecycle.process_qa_result(
        final_state={},
        worker_agent=worker_b,
        critic_agent=critic,
        coordinator_agent=coordinator,
        auto_promote=False,
    )

    assert result.success is True
    assert result.private_memory_written is True
    assert result.private_memory_id is not None

    assert result.promotion_status == "eligible_not_submitted"
    assert result.promotion_request_id is None
    assert result.shared_memory_id is None

    # Only private-memory CREATE was added.
    assert result.operation_log_count == 1

    stored_memory = memory_store.get_by_id(
        result.private_memory_id
    )

    assert stored_memory.metadata.scope == MemoryScope.PRIVATE
    assert (
        stored_memory.metadata.owner_agent_id
        == worker_b.agent_id
    )

    assert memory_store.count() == 1
    assert request_store.count() == 0
    assert operation_log_store.count() == 1


# ----------------------------------------------------------------------
# 3. Safe candidate: PromotionService not configured
# ----------------------------------------------------------------------


def test_safe_candidate_stays_private_when_promotion_service_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_environment: dict[str, Any],
    worker_b: Agent,
    critic: Agent,
    coordinator: Agent,
) -> None:
    candidate = _build_candidate_memory(
        worker_agent=worker_b,
        question="What is the capital of Italy?",
        answer="Rome",
    )

    _patch_candidate_builder(
        monkeypatch,
        candidate,
    )

    lifecycle_without_promotion = _build_lifecycle(
        memory_service=lifecycle_environment["memory_service"],
        operation_log_service=lifecycle_environment[
            "operation_log_service"
        ],
        conflict_detector=lifecycle_environment[
            "conflict_detector"
        ],
        write_controller=lifecycle_environment[
            "write_controller"
        ],
        promotion_service=None,
    )

    memory_store = lifecycle_environment["memory_store"]
    operation_log_store = lifecycle_environment[
        "operation_log_store"
    ]
    request_store = lifecycle_environment[
        "promotion_request_store"
    ]

    result = lifecycle_without_promotion.process_qa_result(
        final_state={},
        worker_agent=worker_b,
        critic_agent=critic,
        coordinator_agent=coordinator,
        auto_promote=True,
    )

    assert result.success is True
    assert result.private_memory_written is True
    assert result.private_memory_id is not None

    assert (
        result.promotion_status
        == "promotion_service_not_configured"
    )
    assert result.promotion_request_id is None
    assert result.shared_memory_id is None

    assert result.operation_log_count == 1

    stored_memory = memory_store.get_by_id(
        result.private_memory_id
    )

    assert stored_memory.metadata.scope == MemoryScope.PRIVATE
    assert request_store.count() == 0
    assert operation_log_store.count() == 1


# ----------------------------------------------------------------------
# 4. Duplicate: block write and record conflict
# ----------------------------------------------------------------------


def test_duplicate_candidate_is_blocked_and_conflict_is_logged(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_environment: dict[str, Any],
    worker_b: Agent,
    critic: Agent,
    coordinator: Agent,
) -> None:
    memory_service = lifecycle_environment["memory_service"]
    lifecycle = lifecycle_environment["lifecycle"]
    memory_store = lifecycle_environment["memory_store"]
    operation_log_store = lifecycle_environment[
        "operation_log_store"
    ]
    request_store = lifecycle_environment[
        "promotion_request_store"
    ]

    existing_memory = _create_existing_private_memory(
        memory_service=memory_service,
        worker_agent=worker_b,
        question="What is the capital of Spain?",
        answer="Madrid",
    )

    historical_log_count = operation_log_store.count()
    assert historical_log_count == 1

    candidate = _build_candidate_memory(
        worker_agent=worker_b,
        question="What is the capital of Spain?",
        answer="Madrid",
    )

    _patch_candidate_builder(
        monkeypatch,
        candidate,
    )

    result = lifecycle.process_qa_result(
        final_state={},
        worker_agent=worker_b,
        critic_agent=critic,
        coordinator_agent=coordinator,
        auto_promote=True,
    )

    assert result.success is True
    assert result.private_memory_written is False
    assert result.private_memory_id is None

    assert result.promotion_status == "blocked_duplicate"
    assert result.promotion_request_id is None
    assert result.shared_memory_id is None

    assert (
        result.conflict_result.conflict_type
        == LifecycleConflictType.DUPLICATE
    )

    # Only one DETECT_CONFLICT log was added during this lifecycle.
    assert result.operation_log_count == 1

    # Duplicate candidate was not persisted.
    assert memory_store.count() == 1
    assert request_store.count() == 0

    # Historical CREATE + current DETECT_CONFLICT.
    assert operation_log_store.count() == 2

    operations = operation_log_store.list_all()
    conflict_operation = operations[-1]

    assert (
        conflict_operation.operation_type
        == MemoryOperationType.DETECT_CONFLICT
    )
    assert (
        conflict_operation.conflict_type
        == OperationConflictType.DUPLICATION
    )

    assert set(conflict_operation.target_memory_ids) == {
        candidate.memory_id,
        existing_memory.memory_id,
    }

    assert conflict_operation.actor_agent_id == critic.agent_id


# ----------------------------------------------------------------------
# 5. Conflicting answer: write private, block promotion, record conflict
# ----------------------------------------------------------------------


def test_conflicting_answer_is_kept_private_and_conflict_is_logged(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_environment: dict[str, Any],
    worker_b: Agent,
    critic: Agent,
    coordinator: Agent,
) -> None:
    memory_service = lifecycle_environment["memory_service"]
    lifecycle = lifecycle_environment["lifecycle"]
    memory_store = lifecycle_environment["memory_store"]
    operation_log_store = lifecycle_environment[
        "operation_log_store"
    ]
    request_store = lifecycle_environment[
        "promotion_request_store"
    ]

    existing_memory = _create_existing_private_memory(
        memory_service=memory_service,
        worker_agent=worker_b,
        question="What is the capital of Canada?",
        answer="Ottawa",
    )

    assert operation_log_store.count() == 1

    candidate = _build_candidate_memory(
        worker_agent=worker_b,
        question="What is the capital of Canada?",
        answer="Toronto",
    )

    _patch_candidate_builder(
        monkeypatch,
        candidate,
    )

    result = lifecycle.process_qa_result(
        final_state={},
        worker_agent=worker_b,
        critic_agent=critic,
        coordinator_agent=coordinator,
        auto_promote=True,
    )

    assert result.success is True
    assert result.private_memory_written is True
    assert result.private_memory_id is not None

    assert result.promotion_status == "requires_review"
    assert result.promotion_request_id is None
    assert result.shared_memory_id is None

    assert (
        result.conflict_result.conflict_type
        == LifecycleConflictType.CONFLICTING_ANSWER
    )

    # 1 CREATE private memory + 1 DETECT_CONFLICT.
    assert result.operation_log_count == 2

    assert memory_store.count() == 2
    assert request_store.count() == 0

    persisted_candidate = memory_store.get_by_id(
        result.private_memory_id
    )

    assert persisted_candidate.metadata.scope == MemoryScope.PRIVATE
    assert (
        persisted_candidate.metadata.owner_agent_id
        == worker_b.agent_id
    )

    # Existing CREATE + new CREATE + DETECT_CONFLICT.
    assert operation_log_store.count() == 3

    new_operations = operation_log_store.list_all()[-2:]

    assert (
        new_operations[0].operation_type
        == MemoryOperationType.CREATE
    )
    assert (
        new_operations[1].operation_type
        == MemoryOperationType.DETECT_CONFLICT
    )
    assert (
        new_operations[1].conflict_type
        == OperationConflictType.CONTRADICTION
    )

    assert set(new_operations[1].target_memory_ids) == {
        result.private_memory_id,
        existing_memory.memory_id,
    }

    assert new_operations[1].actor_agent_id == critic.agent_id


# ----------------------------------------------------------------------
# 6. operation_log_count only contains logs from the current execution
# ----------------------------------------------------------------------


def test_operation_log_count_excludes_historical_logs(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_environment: dict[str, Any],
    worker_b: Agent,
    critic: Agent,
    coordinator: Agent,
) -> None:
    lifecycle = lifecycle_environment["lifecycle"]
    operation_log_store = lifecycle_environment[
        "operation_log_store"
    ]

    first_candidate = _build_candidate_memory(
        worker_agent=worker_b,
        question="What is the capital of Japan?",
        answer="Tokyo",
    )

    _patch_candidate_builder(
        monkeypatch,
        first_candidate,
    )

    first_result = lifecycle.process_qa_result(
        final_state={},
        worker_agent=worker_b,
        critic_agent=critic,
        coordinator_agent=coordinator,
        auto_promote=False,
    )

    assert first_result.success is True
    assert first_result.operation_log_count == 1
    assert operation_log_store.count() == 1

    second_candidate = _build_candidate_memory(
        worker_agent=worker_b,
        question="What is the capital of Portugal?",
        answer="Lisbon",
    )

    _patch_candidate_builder(
        monkeypatch,
        second_candidate,
    )

    second_result = lifecycle.process_qa_result(
        final_state={},
        worker_agent=worker_b,
        critic_agent=critic,
        coordinator_agent=coordinator,
        auto_promote=False,
    )

    assert second_result.success is True

    # The second result reports only the second execution's CREATE log.
    assert second_result.operation_log_count == 1

    # The persistent store contains both executions.
    assert operation_log_store.count() == 2


# ----------------------------------------------------------------------
# 7. Coordinator approval failure: preserve partial state and log delta
# ----------------------------------------------------------------------


def test_invalid_coordinator_returns_failure_after_partial_promotion_flow(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_environment: dict[str, Any],
    worker_b: Agent,
    critic: Agent,
) -> None:
    candidate = _build_candidate_memory(
        worker_agent=worker_b,
        question="What is the capital of Australia?",
        answer="Canberra",
    )

    _patch_candidate_builder(
        monkeypatch,
        candidate,
    )

    invalid_coordinator = Agent.worker(
        agent_id="invalid_coordinator",
    )

    lifecycle = lifecycle_environment["lifecycle"]
    memory_store = lifecycle_environment["memory_store"]
    operation_log_store = lifecycle_environment[
        "operation_log_store"
    ]
    request_store = lifecycle_environment[
        "promotion_request_store"
    ]

    result = lifecycle.process_qa_result(
        final_state={},
        worker_agent=worker_b,
        critic_agent=critic,
        coordinator_agent=invalid_coordinator,
        auto_promote=True,
    )

    assert result.success is False
    assert result.private_memory_written is True
    assert result.private_memory_id is not None

    assert result.promotion_status == "failed"
    assert result.shared_memory_id is None

    # Before Coordinator approval fails:
    # 1 CREATE private memory
    # 2 promotion submitted
    # 3 Critic reviewed
    assert result.operation_log_count == 3
    assert operation_log_store.count() == 3

    stored_memory = memory_store.get_by_id(
        result.private_memory_id
    )

    assert stored_memory.metadata.scope == MemoryScope.PRIVATE

    requests = request_store.list_all()

    assert len(requests) == 1
    assert requests[0].status == PromotionStatus.PENDING
    assert requests[0].reviewed_by_agent_id == critic.agent_id


# ----------------------------------------------------------------------
# 8. Promotion request survives a new store instance
# ----------------------------------------------------------------------


def test_approved_promotion_request_is_persisted(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_environment: dict[str, Any],
    worker_b: Agent,
    critic: Agent,
    coordinator: Agent,
) -> None:
    candidate = _build_candidate_memory(
        worker_agent=worker_b,
        question="What is the capital of Norway?",
        answer="Oslo",
    )

    _patch_candidate_builder(
        monkeypatch,
        candidate,
    )

    lifecycle = lifecycle_environment["lifecycle"]
    request_store = lifecycle_environment[
        "promotion_request_store"
    ]

    result = lifecycle.process_qa_result(
        final_state={},
        worker_agent=worker_b,
        critic_agent=critic,
        coordinator_agent=coordinator,
        auto_promote=True,
    )

    assert result.success is True
    assert result.promotion_request_id is not None

    reloaded_store = PromotionRequestStore(
        request_store.file_path
    )

    reloaded_request = reloaded_store.get_by_id(
        result.promotion_request_id
    )

    assert (
        reloaded_request.request_id
        == result.promotion_request_id
    )
    assert reloaded_request.status == PromotionStatus.APPROVED
    assert reloaded_request.memory_id == result.shared_memory_id
