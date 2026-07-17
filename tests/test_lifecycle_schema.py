from __future__ import annotations

import pytest

from src.memory.lifecycle import (
    CandidateMemoryBuilder,
    ConflictCheckResult,
    ConflictDecision,
    ConflictSeverity,
    ConflictType,
    MemoryConflictRecord,
)
from src.llm import AgentAnswer, CriticOutput, CoordinatorOutput


def _create_mock_final_state() -> dict:
    """
    Create a mock GovernedRAGQAWorkflow final_state.

    This avoids calling LangGraph, Chroma, or OpenAI.
    """

    worker_a_output = AgentAnswer(
        answer="the late 1990s",
        reasoning="Beyonce became popular as a member of Destiny's Child.",
        confidence=0.8,
        evidence_chunk_ids=[],
    )

    worker_b_output = AgentAnswer(
        answer="in the late 1990s",
        reasoning="The retrieved evidence states that Beyonce rose to fame in the late 1990s.",
        confidence=0.95,
        evidence_chunk_ids=["chunk_beyonce_001"],
    )

    critic_output = CriticOutput(
        recommended_answer="in the late 1990s",
        preferred_worker="worker_b",
        comment="Worker B's answer is directly supported by the retrieved evidence.",
        confidence=0.95,
    )

    coordinator_output = CoordinatorOutput(
        final_answer="in the late 1990s",
        reasoning="The evidence-supported answer is selected as the final answer.",
        confidence=0.95,
    )

    return {
        "sample_id": "test_beyonce_001",
        "title": "Beyoncé",
        "question": "When did Beyonce start becoming popular?",
        "gold_answers": ["in the late 1990s"],
        "gold_context_hash": "mock_context_hash",

        "retrieval_top_k": 3,
        "retrieved_chunks": [],
        "retrieved_chunk_ids": ["chunk_beyonce_001", "chunk_beyonce_002"],
        "retrieved_context_hashes": ["mock_context_hash"],
        "retrieval_hit": True,

        "worker_a_output": worker_a_output,
        "worker_b_output": worker_b_output,
        "critic_output": critic_output,
        "coordinator_output": coordinator_output,

        "prediction": "in the late 1990s",

        "current_step": "coordinator_finalized",
        "success": True,
        "error_message": None,
    }


# ----------------------------------------------------------------------
# Conflict schema tests
# ----------------------------------------------------------------------

def test_conflict_check_result_no_conflict() -> None:
    result = ConflictCheckResult.no_conflict(
        candidate_memory_id="candidate_001",
        candidate_question="When did Beyonce start becoming popular?",
        candidate_answer="in the late 1990s",
    )

    print("\n========== NO CONFLICT RESULT ==========")
    print(result)

    assert result.candidate_memory_id == "candidate_001"
    assert result.conflict_type == ConflictType.NONE
    assert result.severity == ConflictSeverity.NONE
    assert result.decision == ConflictDecision.ALLOW_WRITE_AND_PROMOTE
    assert result.should_write is True
    assert result.should_promote is True
    assert result.requires_review is False
    assert result.matched_records == []
    assert result.candidate_question == "When did Beyonce start becoming popular?"
    assert result.candidate_answer == "in the late 1990s"


def test_conflict_check_result_duplicate() -> None:
    matched_record = MemoryConflictRecord(
        matched_memory_id="memory_existing_001",
        matched_memory_content=(
            "Question: When did Beyonce start becoming popular?\n"
            "Final answer: in the late 1990s"
        ),
        matched_question="When did Beyonce start becoming popular?",
        matched_answer="in the late 1990s",
        similarity_score=1.0,
    )

    result = ConflictCheckResult.duplicate(
        candidate_memory_id="candidate_001",
        matched_records=[matched_record],
        candidate_question="When did Beyonce start becoming popular?",
        candidate_answer="in the late 1990s",
    )

    print("\n========== DUPLICATE RESULT ==========")
    print(result)

    assert result.conflict_type == ConflictType.DUPLICATE
    assert result.severity == ConflictSeverity.LOW
    assert result.decision == ConflictDecision.BLOCK_WRITE
    assert result.should_write is False
    assert result.should_promote is False
    assert result.requires_review is False
    assert len(result.matched_records) == 1
    assert result.matched_records[0].matched_memory_id == "memory_existing_001"


def test_conflict_check_result_conflicting_answer() -> None:
    matched_record = MemoryConflictRecord(
        matched_memory_id="memory_existing_001",
        matched_memory_content=(
            "Question: When did Beyonce start becoming popular?\n"
            "Final answer: in the late 1990s"
        ),
        matched_question="When did Beyonce start becoming popular?",
        matched_answer="in the late 1990s",
        similarity_score=1.0,
    )

    result = ConflictCheckResult.conflicting_answer(
        candidate_memory_id="candidate_002",
        matched_records=[matched_record],
        candidate_question="When did Beyonce start becoming popular?",
        candidate_answer="early 2000s",
    )

    print("\n========== CONFLICTING ANSWER RESULT ==========")
    print(result)

    assert result.conflict_type == ConflictType.CONFLICTING_ANSWER
    assert result.severity == ConflictSeverity.HIGH
    assert result.decision == ConflictDecision.REQUIRE_REVIEW
    assert result.should_write is True
    assert result.should_promote is False
    assert result.requires_review is True
    assert len(result.matched_records) == 1


def test_memory_conflict_record_requires_memory_id() -> None:
    with pytest.raises(ValueError):
        MemoryConflictRecord(
            matched_memory_id="",
            matched_memory_content="Some content",
        )


def test_conflict_check_result_requires_candidate_memory_id() -> None:
    with pytest.raises(ValueError):
        ConflictCheckResult.no_conflict(
            candidate_memory_id="",
        )


# ----------------------------------------------------------------------
# Candidate memory builder tests
# ----------------------------------------------------------------------

def test_candidate_memory_builder_builds_private_memory_from_final_state() -> None:
    final_state = _create_mock_final_state()

    candidate_memory = CandidateMemoryBuilder.build_from_final_state(
        final_state,
        owner_agent_id="worker_b",
        created_by_agent_id="worker_b",
        memory_id="candidate_memory_test_001",
    )

    print("\n========== CANDIDATE MEMORY ==========")
    print(candidate_memory)

    assert candidate_memory.memory_id == "candidate_memory_test_001"
    assert candidate_memory.content

    assert "Question: When did Beyonce start becoming popular?" in candidate_memory.content
    assert "Final answer: in the late 1990s" in candidate_memory.content
    assert "Worker A answer: the late 1990s" in candidate_memory.content
    assert "Worker B answer: in the late 1990s" in candidate_memory.content
    assert "Critic recommendation: in the late 1990s" in candidate_memory.content
    assert "chunk_beyonce_001" in candidate_memory.content
    assert "Retrieval hit: True" in candidate_memory.content

    metadata = candidate_memory.metadata

    assert metadata.memory_id == "candidate_memory_test_001"
    assert metadata.scope == "private"
    assert metadata.owner_agent_id == "worker_b"
    assert metadata.created_by_agent_id == "worker_b"
    assert metadata.status == "active"
    assert metadata.memory_type == "fact"
    assert "qa" in metadata.tags
    assert "squad" in metadata.tags
    assert "candidate" in metadata.tags
    assert "governed_rag_qa" in metadata.tags
    assert metadata.readable_by == ["worker_b"]
    assert metadata.writable_by == ["worker_b"]


def test_candidate_memory_builder_generates_memory_id_when_missing() -> None:
    final_state = _create_mock_final_state()

    candidate_memory = CandidateMemoryBuilder.build_from_final_state(
        final_state,
        owner_agent_id="worker_b",
        created_by_agent_id="worker_b",
    )

    print("\n========== GENERATED MEMORY ID ==========")
    print(candidate_memory.memory_id)

    assert candidate_memory.memory_id.startswith("candidate_memory_")
    assert candidate_memory.metadata.memory_id == candidate_memory.memory_id


def test_candidate_memory_builder_accepts_extra_tags() -> None:
    final_state = _create_mock_final_state()

    candidate_memory = CandidateMemoryBuilder.build_from_final_state(
        final_state,
        owner_agent_id="worker_b",
        created_by_agent_id="worker_b",
        memory_id="candidate_memory_test_002",
        tags=["experiment", "qa", "experiment"],
    )

    metadata = candidate_memory.metadata

    print("\n========== CANDIDATE TAGS ==========")
    print(metadata.tags)

    assert "experiment" in metadata.tags
    assert metadata.tags.count("experiment") == 1
    assert metadata.tags.count("qa") == 1


def test_candidate_memory_builder_uses_coordinator_output_when_prediction_missing() -> None:
    final_state = _create_mock_final_state()
    final_state["prediction"] = None

    candidate_memory = CandidateMemoryBuilder.build_from_final_state(
        final_state,
        owner_agent_id="worker_b",
        created_by_agent_id="worker_b",
        memory_id="candidate_memory_test_003",
    )

    print("\n========== CONTENT WITHOUT PREDICTION ==========")
    print(candidate_memory.content)

    assert "Final answer: in the late 1990s" in candidate_memory.content


def test_candidate_memory_builder_requires_question() -> None:
    final_state = _create_mock_final_state()
    final_state["question"] = ""

    with pytest.raises(ValueError):
        CandidateMemoryBuilder.build_from_final_state(
            final_state,
            owner_agent_id="worker_b",
            created_by_agent_id="worker_b",
        )


def test_candidate_memory_builder_requires_prediction_or_coordinator_answer() -> None:
    final_state = _create_mock_final_state()
    final_state["prediction"] = None
    final_state["coordinator_output"] = None

    with pytest.raises(ValueError):
        CandidateMemoryBuilder.build_from_final_state(
            final_state,
            owner_agent_id="worker_b",
            created_by_agent_id="worker_b",
        )


def test_candidate_memory_content_has_structured_question_and_answer_lines() -> None:
    final_state = _create_mock_final_state()

    content = CandidateMemoryBuilder.build_memory_content(final_state)

    print("\n========== STRUCTURED MEMORY CONTENT ==========")
    print(content)

    lines = content.splitlines()

    assert any(line.startswith("Question: ") for line in lines)
    assert any(line.startswith("Final answer: ") for line in lines)
    assert any(line.startswith("Worker A answer: ") for line in lines)
    assert any(line.startswith("Worker B answer: ") for line in lines)
    assert any(line.startswith("Critic recommendation: ") for line in lines)