from __future__ import annotations

from src.memory.entities import MemoryItem, MemoryMetadata
from src.memory.lifecycle import (
    ConflictDetector,
    ConflictType,
    ConflictSeverity,
    ConflictDecision,
)


def _create_memory(
    *,
    memory_id: str,
    question: str,
    answer: str,
    owner_agent_id: str = "worker_b",
) -> MemoryItem:
    """
    Create a structured QA memory for conflict detector tests.
    """

    content = "\n".join(
        [
            "Memory source: test",
            "",
            f"Question: {question}",
            f"Final answer: {answer}",
        ]
    )

    metadata = MemoryMetadata(
        memory_id=memory_id,
        scope="private",
        owner_agent_id=owner_agent_id,
        created_by_agent_id=owner_agent_id,
        status="active",
        memory_type="fact",
        tags=["test", "qa"],
        readable_by=[owner_agent_id],
        writable_by=[owner_agent_id],
    )

    return MemoryItem(
        memory_id=memory_id,
        content=content,
        metadata=metadata,
    )


def test_conflict_detector_detects_duplicate_memory() -> None:
    """
    Same question + equivalent answer -> duplicate.
    """

    detector = ConflictDetector(
        question_similarity_threshold=0.88,
    )

    existing_memory = _create_memory(
        memory_id="existing_memory_001",
        question="When did Beyonce start becoming popular?",
        answer="in the late 1990s",
    )

    candidate_memory = _create_memory(
        memory_id="candidate_memory_001",
        question="When did Beyonce start becoming popular?",
        answer="in the late 1990s",
    )

    result = detector.check_conflict(
        candidate_memory=candidate_memory,
        existing_memories=[existing_memory],
    )

    print("\n========== DUPLICATE DETECTION RESULT ==========")
    print(result)

    assert result.conflict_type == ConflictType.DUPLICATE
    assert result.severity == ConflictSeverity.LOW
    assert result.decision == ConflictDecision.BLOCK_WRITE
    assert result.should_write is False
    assert result.should_promote is False
    assert result.requires_review is False

    assert result.candidate_memory_id == "candidate_memory_001"
    assert result.candidate_question == "When did Beyonce start becoming popular?"
    assert result.candidate_answer == "in the late 1990s"

    assert len(result.matched_records) == 1
    assert result.matched_records[0].matched_memory_id == "existing_memory_001"
    assert result.matched_records[0].matched_question == (
        "When did Beyonce start becoming popular?"
    )
    assert result.matched_records[0].matched_answer == "in the late 1990s"


def test_conflict_detector_detects_conflicting_answer() -> None:
    """
    Same question + different answer -> conflicting_answer.
    """

    detector = ConflictDetector(
        question_similarity_threshold=0.88,
    )

    existing_memory = _create_memory(
        memory_id="existing_memory_001",
        question="When did Beyonce start becoming popular?",
        answer="in the late 1990s",
    )

    candidate_memory = _create_memory(
        memory_id="candidate_memory_002",
        question="When did Beyonce start becoming popular?",
        answer="early 2000s",
    )

    result = detector.check_conflict(
        candidate_memory=candidate_memory,
        existing_memories=[existing_memory],
    )

    print("\n========== CONFLICTING ANSWER DETECTION RESULT ==========")
    print(result)

    assert result.conflict_type == ConflictType.CONFLICTING_ANSWER
    assert result.severity == ConflictSeverity.HIGH
    assert result.decision == ConflictDecision.REQUIRE_REVIEW
    assert result.should_write is True
    assert result.should_promote is False
    assert result.requires_review is True

    assert result.candidate_memory_id == "candidate_memory_002"
    assert result.candidate_question == "When did Beyonce start becoming popular?"
    assert result.candidate_answer == "early 2000s"

    assert len(result.matched_records) == 1
    assert result.matched_records[0].matched_memory_id == "existing_memory_001"


def test_conflict_detector_returns_no_conflict_for_different_question() -> None:
    """
    Different question -> no conflict.
    """

    detector = ConflictDetector(
        question_similarity_threshold=0.88,
    )

    existing_memory = _create_memory(
        memory_id="existing_memory_001",
        question="When did Beyonce start becoming popular?",
        answer="in the late 1990s",
    )

    candidate_memory = _create_memory(
        memory_id="candidate_memory_003",
        question="What is the capital of France?",
        answer="Paris",
    )

    result = detector.check_conflict(
        candidate_memory=candidate_memory,
        existing_memories=[existing_memory],
    )

    print("\n========== NO CONFLICT DETECTION RESULT ==========")
    print(result)

    assert result.conflict_type == ConflictType.NONE
    assert result.severity == ConflictSeverity.NONE
    assert result.decision == ConflictDecision.ALLOW_WRITE_AND_PROMOTE
    assert result.should_write is True
    assert result.should_promote is True
    assert result.requires_review is False
    assert result.matched_records == []

    assert result.candidate_memory_id == "candidate_memory_003"
    assert result.candidate_question == "What is the capital of France?"
    assert result.candidate_answer == "Paris"


def test_conflict_detector_detects_similar_question_duplicate() -> None:
    """
    Similar question + same answer -> duplicate.

    This tests the question similarity threshold.
    """

    detector = ConflictDetector(
        question_similarity_threshold=0.75,
    )

    existing_memory = _create_memory(
        memory_id="existing_memory_001",
        question="When did Beyonce start becoming popular?",
        answer="in the late 1990s",
    )

    candidate_memory = _create_memory(
        memory_id="candidate_memory_004",
        question="When did Beyonce become popular?",
        answer="in the late 1990s",
    )

    result = detector.check_conflict(
        candidate_memory=candidate_memory,
        existing_memories=[existing_memory],
    )

    print("\n========== SIMILAR QUESTION DUPLICATE RESULT ==========")
    print(result)

    assert result.conflict_type == ConflictType.DUPLICATE
    assert result.should_write is False
    assert result.should_promote is False
    assert len(result.matched_records) == 1
    assert result.matched_records[0].similarity_score is not None
    assert result.matched_records[0].similarity_score >= 0.75


def test_conflict_detector_prioritizes_conflict_over_duplicate() -> None:
    """
    If both duplicate and conflicting memories are found,
    conflicting_answer should be prioritized.
    """

    detector = ConflictDetector(
        question_similarity_threshold=0.88,
    )

    duplicate_existing = _create_memory(
        memory_id="existing_duplicate",
        question="When did Beyonce start becoming popular?",
        answer="in the late 1990s",
    )

    conflicting_existing = _create_memory(
        memory_id="existing_conflict",
        question="When did Beyonce start becoming popular?",
        answer="early 2000s",
    )

    candidate_memory = _create_memory(
        memory_id="candidate_memory_005",
        question="When did Beyonce start becoming popular?",
        answer="in the late 1990s",
    )

    result = detector.check_conflict(
        candidate_memory=candidate_memory,
        existing_memories=[
            duplicate_existing,
            conflicting_existing,
        ],
    )

    print("\n========== CONFLICT PRIORITY RESULT ==========")
    print(result)

    assert result.conflict_type == ConflictType.CONFLICTING_ANSWER
    assert result.severity == ConflictSeverity.HIGH
    assert result.should_write is True
    assert result.should_promote is False
    assert result.requires_review is True

    matched_ids = [
        record.matched_memory_id
        for record in result.matched_records
    ]

    assert "existing_conflict" in matched_ids


def test_conflict_detector_ignores_self_comparison() -> None:
    """
    Candidate memory should not be compared with itself.
    """

    detector = ConflictDetector()

    candidate_memory = _create_memory(
        memory_id="same_memory_id",
        question="When did Beyonce start becoming popular?",
        answer="in the late 1990s",
    )

    result = detector.check_conflict(
        candidate_memory=candidate_memory,
        existing_memories=[candidate_memory],
    )

    print("\n========== SELF COMPARISON RESULT ==========")
    print(result)

    assert result.conflict_type == ConflictType.NONE
    assert result.matched_records == []


def test_conflict_detector_handles_unstructured_candidate_memory() -> None:
    """
    Candidate without Question / Final answer fields should not crash.
    """

    detector = ConflictDetector()

    metadata = MemoryMetadata(
        memory_id="candidate_unstructured",
        scope="private",
        owner_agent_id="worker_b",
        created_by_agent_id="worker_b",
        status="active",
        memory_type="fact",
        tags=["test"],
        readable_by=["worker_b"],
        writable_by=["worker_b"],
    )

    candidate_memory = MemoryItem(
        memory_id="candidate_unstructured",
        content="This is an unstructured memory without explicit QA fields.",
        metadata=metadata,
    )

    existing_memory = _create_memory(
        memory_id="existing_memory_001",
        question="When did Beyonce start becoming popular?",
        answer="in the late 1990s",
    )

    result = detector.check_conflict(
        candidate_memory=candidate_memory,
        existing_memories=[existing_memory],
    )

    print("\n========== UNSTRUCTURED CANDIDATE RESULT ==========")
    print(result)

    assert result.conflict_type == ConflictType.NONE
    assert result.should_write is True
    assert result.should_promote is True
    assert result.matched_records == []
    assert result.candidate_question is None
    assert result.candidate_answer is None


def test_conflict_detector_extracts_question_and_answer() -> None:
    content = "\n".join(
        [
            "Memory source: test",
            "Question: When did Beyonce start becoming popular?",
            "Final answer: in the late 1990s",
        ]
    )

    question = ConflictDetector.extract_question(content)
    answer = ConflictDetector.extract_answer(content)

    assert question == "When did Beyonce start becoming popular?"
    assert answer == "in the late 1990s"


def test_conflict_detector_supports_short_q_and_a_prefixes() -> None:
    content = "\n".join(
        [
            "Q: What is the capital of France?",
            "A: Paris",
        ]
    )

    question = ConflictDetector.extract_question(content)
    answer = ConflictDetector.extract_answer(content)

    assert question == "What is the capital of France?"
    assert answer == "Paris"


def test_conflict_detector_question_similarity() -> None:
    detector = ConflictDetector()

    similarity_same = detector.question_similarity(
        "When did Beyonce start becoming popular?",
        "When did Beyonce start becoming popular?",
    )

    similarity_similar = detector.question_similarity(
        "When did Beyonce start becoming popular?",
        "When did Beyonce become popular?",
    )

    similarity_different = detector.question_similarity(
        "When did Beyonce start becoming popular?",
        "What is the capital of France?",
    )

    print("\n========== QUESTION SIMILARITY ==========")
    print("same:", similarity_same)
    print("similar:", similarity_similar)
    print("different:", similarity_different)

    assert similarity_same == 1.0
    assert similarity_similar > similarity_different
    assert similarity_different < 0.5


def test_conflict_detector_answer_equivalence() -> None:
    assert ConflictDetector.answers_equivalent(
        "Beyoncé.",
        "Beyoncé",
    ) is True

    assert ConflictDetector.answers_equivalent(
        "The United States",
        "United States",
    ) is True

    assert ConflictDetector.answers_equivalent(
        "late 1990s",
        "early 2000s",
    ) is False