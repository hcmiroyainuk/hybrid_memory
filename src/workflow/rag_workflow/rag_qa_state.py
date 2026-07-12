from __future__ import annotations

from typing import Optional, TypedDict

from src.memory.external_knowledge import RetrievedKnowledge
from src.llm import (
    AgentAnswer,
    CriticOutput,
    CoordinatorOutput,
)


class RAGQAState(TypedDict):
    """
    Shared state for the LLM-RAG multi-agent QA workflow.

    This state is passed between LangGraph nodes.

    Workflow:
        question
          -> Worker A direct answer
          -> external knowledge retrieval
          -> Worker B retrieval-grounded answer
          -> Critic evaluation
          -> Coordinator final answer

    This workflow does not include governed memory yet.
    Memory permission control will be added in a later governed workflow.
    """

    # ------------------------------------------------------------------
    # Input sample information
    # ------------------------------------------------------------------

    sample_id: str
    title: str
    question: str
    gold_answers: list[str]
    gold_context_hash: Optional[str]

    # ------------------------------------------------------------------
    # Retrieval configuration and result
    # ------------------------------------------------------------------

    retrieval_top_k: int
    retrieved_chunks: list[RetrievedKnowledge]
    retrieved_chunk_ids: list[str]
    retrieved_context_hashes: list[str]
    retrieval_hit: Optional[bool]

    # ------------------------------------------------------------------
    # Agent outputs
    # ------------------------------------------------------------------

    worker_a_output: Optional[AgentAnswer]
    worker_b_output: Optional[AgentAnswer]
    critic_output: Optional[CriticOutput]
    coordinator_output: Optional[CoordinatorOutput]

    # ------------------------------------------------------------------
    # Final prediction
    # ------------------------------------------------------------------

    prediction: Optional[str]

    # ------------------------------------------------------------------
    # Workflow status
    # ------------------------------------------------------------------

    current_step: str
    success: bool
    error_message: Optional[str]


def create_initial_rag_qa_state(
    *,
    sample_id: str,
    title: str,
    question: str,
    gold_answers: Optional[list[str]] = None,
    gold_context_hash: Optional[str] = None,
    retrieval_top_k: int = 3,
) -> RAGQAState:
    """
    Create the initial state for one RAG QA workflow run.

    Args:
        sample_id:
            ID of the SQuAD sample.
        title:
            Article title or dataset topic.
        question:
            User question / SQuAD question.
        gold_answers:
            Ground-truth answers. Used later by evaluator.
        gold_context_hash:
            Hash of the gold context. Used for retrieval hit analysis.
        retrieval_top_k:
            Number of chunks to retrieve from external knowledge base.

    Returns:
        Initial RAGQAState.
    """

    if not question or not question.strip():
        raise ValueError("question cannot be empty.")

    if retrieval_top_k <= 0:
        raise ValueError("retrieval_top_k must be greater than 0.")

    return {
        # Input sample information
        "sample_id": sample_id,
        "title": title,
        "question": question,
        "gold_answers": gold_answers or [],
        "gold_context_hash": gold_context_hash,

        # Retrieval
        "retrieval_top_k": retrieval_top_k,
        "retrieved_chunks": [],
        "retrieved_chunk_ids": [],
        "retrieved_context_hashes": [],
        "retrieval_hit": None,

        # Agent outputs
        "worker_a_output": None,
        "worker_b_output": None,
        "critic_output": None,
        "coordinator_output": None,

        # Final prediction
        "prediction": None,

        # Workflow status
        "current_step": "initialized",
        "success": False,
        "error_message": None,
    }


def create_initial_rag_qa_state_from_sample(
    sample,
    retrieval_top_k: int = 3,
) -> RAGQAState:
    """
    Convenience helper for creating initial state from a SquadSample.

    Expected sample fields:
        sample_id
        title
        question
        answers
        context_hash

    This keeps the workflow runner cleaner.
    """

    return create_initial_rag_qa_state(
        sample_id=sample.sample_id,
        title=sample.title,
        question=sample.question,
        gold_answers=sample.answers,
        gold_context_hash=sample.context_hash,
        retrieval_top_k=retrieval_top_k,
    )