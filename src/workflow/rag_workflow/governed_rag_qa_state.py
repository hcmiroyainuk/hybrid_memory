from __future__ import annotations

from typing import Optional, TypedDict

from src.memory.context import GovernedContextBuilder, GovernedMemoryContext
from src.memory.entities import Agent
from src.memory.external_knowledge import RetrievedKnowledge
from src.llm import (
    AgentAnswer,
    CriticOutput,
    CoordinatorOutput,
)


class GovernedRAGQAState(TypedDict):
    """
    Shared state for the governed LLM-RAG multi-agent QA workflow.

    This state extends the standard RAG QA workflow with read-governed memory
    retrieval.

    Workflow idea:
        question
          -> retrieve Worker A accessible memory
          -> Worker A answer
          -> retrieve Worker B accessible memory
          -> external knowledge retrieval
          -> Worker B answer
          -> retrieve Critic accessible memory
          -> Critic evaluation
          -> retrieve Coordinator accessible memory
          -> Coordinator final answer

    This version only includes read governance.
    Write governance and promotion workflow should be added later.
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
    # Agent identities
    # ------------------------------------------------------------------

    worker_a_agent: Agent
    worker_b_agent: Agent
    critic_agent: Agent
    coordinator_agent: Agent

    # ------------------------------------------------------------------
    # Memory retrieval configuration
    # ------------------------------------------------------------------

    memory_top_k: int

    # ------------------------------------------------------------------
    # Permission-filtered memory contexts
    # ------------------------------------------------------------------

    worker_a_memory_context: Optional[GovernedMemoryContext]
    worker_b_memory_context: Optional[GovernedMemoryContext]
    critic_memory_context: Optional[GovernedMemoryContext]
    coordinator_memory_context: Optional[GovernedMemoryContext]

    # ------------------------------------------------------------------
    # Convenience memory context text for prompt input
    # ------------------------------------------------------------------

    worker_a_memory_text: Optional[str]
    worker_b_memory_text: Optional[str]
    critic_memory_text: Optional[str]
    coordinator_memory_text: Optional[str]

    # ------------------------------------------------------------------
    # Convenience memory ids for inspection / testing
    # ------------------------------------------------------------------

    worker_a_memory_ids: list[str]
    worker_b_memory_ids: list[str]
    critic_memory_ids: list[str]
    coordinator_memory_ids: list[str]

    # ------------------------------------------------------------------
    # External retrieval configuration and result
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


def create_initial_governed_rag_qa_state(
    *,
    sample_id: str,
    title: str,
    question: str,
    worker_a_agent: Agent,
    worker_b_agent: Agent,
    critic_agent: Agent,
    coordinator_agent: Agent,
    gold_answers: Optional[list[str]] = None,
    gold_context_hash: Optional[str] = None,
    retrieval_top_k: int = 3,
    memory_top_k: int = 3,
) -> GovernedRAGQAState:
    """
    Create the initial state for one governed RAG QA workflow run.

    Args:
        sample_id:
            ID of the SQuAD sample.
        title:
            Article title or dataset topic.
        question:
            User question / SQuAD question.
        worker_a_agent:
            Worker A identity.
        worker_b_agent:
            Worker B identity.
        critic_agent:
            Critic identity.
        coordinator_agent:
            Coordinator identity.
        gold_answers:
            Ground-truth answers. Used later by evaluator.
        gold_context_hash:
            Hash of the gold context. Used for retrieval hit analysis.
        retrieval_top_k:
            Number of external knowledge chunks to retrieve.
        memory_top_k:
            Number of governed memory items to retrieve for each agent.

    Returns:
        Initial GovernedRAGQAState.
    """

    if not question or not question.strip():
        raise ValueError("question cannot be empty.")

    if retrieval_top_k <= 0:
        raise ValueError("retrieval_top_k must be greater than 0.")

    if memory_top_k <= 0:
        raise ValueError("memory_top_k must be greater than 0.")

    return {
        # Input sample information
        "sample_id": sample_id,
        "title": title,
        "question": question,
        "gold_answers": gold_answers or [],
        "gold_context_hash": gold_context_hash,

        # Agent identities
        "worker_a_agent": worker_a_agent,
        "worker_b_agent": worker_b_agent,
        "critic_agent": critic_agent,
        "coordinator_agent": coordinator_agent,

        # Memory retrieval configuration
        "memory_top_k": memory_top_k,

        # Permission-filtered memory contexts
        "worker_a_memory_context": None,
        "worker_b_memory_context": None,
        "critic_memory_context": None,
        "coordinator_memory_context": None,

        # Prompt-ready memory text
        "worker_a_memory_text": None,
        "worker_b_memory_text": None,
        "critic_memory_text": None,
        "coordinator_memory_text": None,

        # Memory ids
        "worker_a_memory_ids": [],
        "worker_b_memory_ids": [],
        "critic_memory_ids": [],
        "coordinator_memory_ids": [],

        # External retrieval
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


def create_initial_governed_rag_qa_state_from_sample(
    sample,
    *,
    worker_a_agent: Agent,
    worker_b_agent: Agent,
    critic_agent: Agent,
    coordinator_agent: Agent,
    retrieval_top_k: int = 3,
    memory_top_k: int = 3,
) -> GovernedRAGQAState:
    """
    Convenience helper for creating initial governed state from a SquadSample.

    Expected sample fields:
        sample_id
        title
        question
        answers
        context_hash
    """

    return create_initial_governed_rag_qa_state(
        sample_id=sample.sample_id,
        title=sample.title,
        question=sample.question,
        gold_answers=sample.answers,
        gold_context_hash=sample.context_hash,
        worker_a_agent=worker_a_agent,
        worker_b_agent=worker_b_agent,
        critic_agent=critic_agent,
        coordinator_agent=coordinator_agent,
        retrieval_top_k=retrieval_top_k,
        memory_top_k=memory_top_k,
    )