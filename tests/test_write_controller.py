from __future__ import annotations

import pytest

from src.memory.entities import (
    Agent,
    MemoryItem,
    MemoryMetadata,
)
from src.memory.lifecycle import (
    ConflictCheckResult,
    ConflictType,
    MemoryConflictRecord,
    MemoryWriteController,
    MemoryWriteResult,
)
from src.memory.manager.memory_store import MemoryStore
from src.memory.services.memory_service import MemoryService
from src.memory.services.permission_service import PermissionService


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def worker_b() -> Agent:
    """
    Worker B owns and writes QA candidate memories.
    """

    return Agent.worker(
        agent_id="worker_b",
    )


@pytest.fixture
def worker_a() -> Agent:
    """
    Another worker used to test ownership enforcement.
    """

    return Agent.worker(
        agent_id="worker_a",
    )


@pytest.fixture
def memory_service(tmp_path) -> MemoryService:
    """
    Create an isolated JSON-backed MemoryService.

    tmp_path prevents tests from modifying the project's real memories.json.
    """

    memory_file = tmp_path / "memories.json"

    memory_store = MemoryStore(memory_file)
    permission_service = PermissionService()

    return MemoryService(
        memory_store=memory_store,
        operation_log_store=None,
        memory_retriever=None,
        permission_service=permission_service,
    )


@pytest.fixture
def write_controller(
    memory_service: MemoryService,
) -> MemoryWriteController:
    return MemoryWriteController(
        memory_service=memory_service,
    )


# ----------------------------------------------------------------------
# Test data helpers
# ----------------------------------------------------------------------

def _create_candidate_memory(
    *,
    memory_id: str,
    question: str = "When did Beyonce start becoming popular?",
    answer: str = "in the late 1990s",
    owner_agent_id: str = "worker_b",
    scope: str = "private",
) -> MemoryItem:
    """
    Create a structured QA candidate memory.

    This object represents the temporary candidate before MemoryService
    creates the formally persisted memory.
    """

    content = "\n".join(
        [
            "Memory source: governed_rag_qa_workflow",
            "",
            f"Question: {question}",
            f"Final answer: {answer}",
        ]
    )

    metadata = MemoryMetadata(
        memory_id=memory_id,
        scope=scope,
        owner_agent_id=owner_agent_id,
        created_by_agent_id=owner_agent_id,
        status="active",
        memory_type="fact",
        tags=[
            "test",
            "qa",
            "candidate",
        ],
        readable_by=(
            [owner_agent_id]
            if scope == "private"
            else ["*"]
        ),
        writable_by=[owner_agent_id],
    )

    return MemoryItem(
        memory_id=memory_id,
        content=content,
        summary=f"QA memory for: {question}",
        metadata=metadata,
    )


def _create_matched_record(
    *,
    memory_id: str = "existing_memory_001",
    question: str = "When did Beyonce start becoming popular?",
    answer: str = "in the late 1990s",
) -> MemoryConflictRecord:
    """
    Create a compact record representing an existing related memory.
    """

    content = "\n".join(
        [
            f"Question: {question}",
            f"Final answer: {answer}",
        ]
    )

    return MemoryConflictRecord(
        matched_memory_id=memory_id,
        matched_memory_content=content,
        matched_question=question,
        matched_answer=answer,
        similarity_score=1.0,
    )


def _enum_value(value) -> str:
    """
    Normalize enum or plain string for assertions.
    """

    if hasattr(value, "value"):
        return str(value.value)

    return str(value)


# ----------------------------------------------------------------------
# No-conflict write tests
# ----------------------------------------------------------------------

def test_write_controller_writes_no_conflict_candidate(
    write_controller: MemoryWriteController,
    memory_service: MemoryService,
    worker_b: Agent,
) -> None:
    """
    No conflict:
    - write candidate as private memory
    - allow later promotion
    - do not require review
    """

    candidate = _create_candidate_memory(
        memory_id="candidate_no_conflict",
    )

    conflict_result = ConflictCheckResult.no_conflict(
        candidate_memory_id=candidate.memory_id,
        candidate_question=(
            "When did Beyonce start becoming popular?"
        ),
        candidate_answer="in the late 1990s",
    )

    result = write_controller.write_candidate_memory(
        candidate_memory=candidate,
        requesting_agent=worker_b,
        conflict_result=conflict_result,
    )

    print("\n========== NO-CONFLICT WRITE RESULT ==========")
    print(result)

    assert isinstance(result, MemoryWriteResult)
    assert result.success is True
    assert result.written is True

    assert result.candidate_memory_id == "candidate_no_conflict"
    assert result.stored_memory_id is not None

    assert result.should_promote is True
    assert result.requires_review is False
    assert result.conflict_type == ConflictType.NONE.value

    # MemoryService generates a new persistent memory ID.
    assert result.stored_memory_id != result.candidate_memory_id

    assert memory_service.memory_store.exists(
        result.stored_memory_id
    ) is True

    stored_memory = memory_service.memory_store.get_by_id(
        result.stored_memory_id
    )

    assert stored_memory.content == candidate.content
    assert stored_memory.summary == candidate.summary

    assert stored_memory.metadata.owner_agent_id == "worker_b"
    assert stored_memory.metadata.created_by_agent_id == "worker_b"
    assert _enum_value(stored_memory.metadata.scope) == "private"
    assert _enum_value(stored_memory.metadata.memory_type) == "fact"

    assert "qa" in stored_memory.metadata.tags
    assert "candidate" in stored_memory.metadata.tags


def test_write_without_conflict_convenience_method(
    write_controller: MemoryWriteController,
    memory_service: MemoryService,
    worker_b: Agent,
) -> None:
    """
    Test write_without_conflict(), which internally creates a no-conflict result.
    """

    candidate = _create_candidate_memory(
        memory_id="candidate_direct_write",
        question="What is the capital of France?",
        answer="Paris",
    )

    result = write_controller.write_without_conflict(
        candidate_memory=candidate,
        requesting_agent=worker_b,
    )

    print("\n========== DIRECT WRITE RESULT ==========")
    print(result)

    assert result.success is True
    assert result.written is True

    assert result.candidate_memory_id == "candidate_direct_write"
    assert result.stored_memory_id is not None

    assert result.should_promote is True
    assert result.requires_review is False

    stored_memory = memory_service.memory_store.get_by_id(
        result.stored_memory_id
    )

    assert "Question: What is the capital of France?" in stored_memory.content
    assert "Final answer: Paris" in stored_memory.content


# ----------------------------------------------------------------------
# Duplicate policy tests
# ----------------------------------------------------------------------

def test_write_controller_blocks_duplicate_candidate(
    write_controller: MemoryWriteController,
    memory_service: MemoryService,
    worker_b: Agent,
) -> None:
    """
    Duplicate:
    - do not write candidate
    - do not allow promotion
    - do not require review
    """

    candidate = _create_candidate_memory(
        memory_id="candidate_duplicate",
    )

    conflict_result = ConflictCheckResult.duplicate(
        candidate_memory_id=candidate.memory_id,
        matched_records=[
            _create_matched_record(),
        ],
        candidate_question=(
            "When did Beyonce start becoming popular?"
        ),
        candidate_answer="in the late 1990s",
    )

    result = write_controller.write_candidate_memory(
        candidate_memory=candidate,
        requesting_agent=worker_b,
        conflict_result=conflict_result,
    )

    print("\n========== DUPLICATE WRITE RESULT ==========")
    print(result)

    assert result.success is True
    assert result.written is False

    assert result.candidate_memory_id == "candidate_duplicate"
    assert result.stored_memory_id is None

    assert result.should_promote is False
    assert result.requires_review is False
    assert result.conflict_type == ConflictType.DUPLICATE.value

    assert memory_service.memory_store.count() == 0
    assert memory_service.memory_store.exists(
        "candidate_duplicate"
    ) is False


# ----------------------------------------------------------------------
# Conflicting-answer policy tests
# ----------------------------------------------------------------------

def test_write_controller_preserves_conflicting_candidate_for_review(
    write_controller: MemoryWriteController,
    memory_service: MemoryService,
    worker_b: Agent,
) -> None:
    """
    Conflicting answer:
    - preserve as private memory
    - block automatic promotion
    - require review
    """

    candidate = _create_candidate_memory(
        memory_id="candidate_conflicting",
        answer="early 2000s",
    )

    conflict_result = ConflictCheckResult.conflicting_answer(
        candidate_memory_id=candidate.memory_id,
        matched_records=[
            _create_matched_record(
                memory_id="existing_memory_late_1990s",
                answer="in the late 1990s",
            ),
        ],
        candidate_question=(
            "When did Beyonce start becoming popular?"
        ),
        candidate_answer="early 2000s",
    )

    result = write_controller.write_candidate_memory(
        candidate_memory=candidate,
        requesting_agent=worker_b,
        conflict_result=conflict_result,
    )

    print("\n========== CONFLICTING WRITE RESULT ==========")
    print(result)

    assert result.success is True
    assert result.written is True

    assert result.candidate_memory_id == "candidate_conflicting"
    assert result.stored_memory_id is not None

    assert result.should_promote is False
    assert result.requires_review is True
    assert (
        result.conflict_type
        == ConflictType.CONFLICTING_ANSWER.value
    )

    stored_memory = memory_service.memory_store.get_by_id(
        result.stored_memory_id
    )

    assert "Final answer: early 2000s" in stored_memory.content
    assert _enum_value(stored_memory.metadata.scope) == "private"
    assert stored_memory.metadata.owner_agent_id == "worker_b"


# ----------------------------------------------------------------------
# Governance policy validation
# ----------------------------------------------------------------------

def test_write_controller_rejects_direct_shared_candidate(
    write_controller: MemoryWriteController,
    memory_service: MemoryService,
    worker_b: Agent,
) -> None:
    """
    A worker must not write a candidate directly into shared memory.
    """

    candidate = _create_candidate_memory(
        memory_id="candidate_illegal_shared",
        scope="shared",
    )

    conflict_result = ConflictCheckResult.no_conflict(
        candidate_memory_id=candidate.memory_id,
    )

    with pytest.raises(
        ValueError,
        match="Candidate memory must be private",
    ):
        write_controller.write_candidate_memory(
            candidate_memory=candidate,
            requesting_agent=worker_b,
            conflict_result=conflict_result,
        )

    assert memory_service.memory_store.count() == 0


def test_write_controller_rejects_candidate_owned_by_another_agent(
    write_controller: MemoryWriteController,
    memory_service: MemoryService,
    worker_a: Agent,
) -> None:
    """
    Worker A must not write a candidate owned by Worker B.
    """

    candidate = _create_candidate_memory(
        memory_id="candidate_owned_by_worker_b",
        owner_agent_id="worker_b",
    )

    conflict_result = ConflictCheckResult.no_conflict(
        candidate_memory_id=candidate.memory_id,
    )

    with pytest.raises(
        ValueError,
        match="Requesting agent does not own",
    ):
        write_controller.write_candidate_memory(
            candidate_memory=candidate,
            requesting_agent=worker_a,
            conflict_result=conflict_result,
        )

    assert memory_service.memory_store.count() == 0


def test_write_controller_rejects_mismatched_conflict_result(
    write_controller: MemoryWriteController,
    memory_service: MemoryService,
    worker_b: Agent,
) -> None:
    """
    Conflict result must belong to the same candidate being written.
    """

    candidate = _create_candidate_memory(
        memory_id="candidate_actual",
    )

    conflict_result = ConflictCheckResult.no_conflict(
        candidate_memory_id="different_candidate_id",
    )

    with pytest.raises(
        ValueError,
        match="Conflict result candidate ID does not match",
    ):
        write_controller.write_candidate_memory(
            candidate_memory=candidate,
            requesting_agent=worker_b,
            conflict_result=conflict_result,
        )

    assert memory_service.memory_store.count() == 0


def test_duplicate_is_blocked_before_scope_and_owner_validation(
    write_controller: MemoryWriteController,
    memory_service: MemoryService,
    worker_a: Agent,
) -> None:
    """
    Duplicate policy blocks persistence immediately.

    Even if the duplicate candidate has an invalid scope or owner,
    no write attempt should occur.
    """

    candidate = _create_candidate_memory(
        memory_id="duplicate_invalid_candidate",
        owner_agent_id="worker_b",
        scope="shared",
    )

    conflict_result = ConflictCheckResult.duplicate(
        candidate_memory_id=candidate.memory_id,
        matched_records=[
            _create_matched_record(),
        ],
    )

    result = write_controller.write_candidate_memory(
        candidate_memory=candidate,
        requesting_agent=worker_a,
        conflict_result=conflict_result,
    )

    assert result.success is True
    assert result.written is False
    assert result.stored_memory_id is None
    assert result.should_promote is False

    assert memory_service.memory_store.count() == 0