from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from src.memory.external_knowledge import ExternalKnowledgeRetriever
from src.llm import (
    AgentAnswer,
    CriticOutput,
    CoordinatorOutput,
    LLMClient,
)
from src.workflow.rag_workflow import (
    RAGQAWorkflow,
    create_initial_rag_qa_state,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY is not set.",
)
def test_rag_qa_workflow_single_question() -> None:
    """
    Test the full LangGraph RAG QA workflow on one question.

    This test checks:
    - Worker A can generate a direct answer
    - External retriever can retrieve evidence
    - Worker B can generate a retrieval-grounded answer
    - Critic can compare Worker A and Worker B
    - Coordinator can generate the final answer
    """

    external_retriever = ExternalKnowledgeRetriever(
        persist_directory="data/test_external_knowledge/squad_chroma",
        collection_name="test_squad_external_knowledge",
        default_top_k=3,
    )

    llm_client = LLMClient(
        model_name="gpt-4o-mini",
        temperature=0.0,
        max_tokens=500,
        env_path=PROJECT_ROOT / ".env",
    )

    workflow = RAGQAWorkflow(
        llm_client=llm_client,
        external_retriever=external_retriever,
    )

    initial_state = create_initial_rag_qa_state(
        sample_id="test_beyonce_001",
        title="Beyoncé",
        question="When did Beyonce start becoming popular?",
        gold_answers=["in the late 1990s"],
        gold_context_hash=None,
        retrieval_top_k=3,
    )

    final_state = workflow.run(initial_state)

    print("\n========== RAG QA WORKFLOW RESULT ==========")
    print("Current step:", final_state["current_step"])
    print("Success:", final_state["success"])
    print("Error message:", final_state["error_message"])

    print("\nQuestion:")
    print(final_state["question"])

    print("\nRetrieved chunk ids:")
    print(final_state["retrieved_chunk_ids"])

    print("\nWorker A:")
    print(final_state["worker_a_output"])

    print("\nWorker B:")
    print(final_state["worker_b_output"])

    print("\nCritic:")
    print(final_state["critic_output"])

    print("\nCoordinator:")
    print(final_state["coordinator_output"])

    print("\nPrediction:")
    print(final_state["prediction"])
    print("===========================================\n")

    assert final_state["success"] is True
    assert final_state["error_message"] is None
    assert final_state["current_step"] == "coordinator_finalized"

    assert final_state["worker_a_output"] is not None
    assert isinstance(final_state["worker_a_output"], AgentAnswer)
    assert final_state["worker_a_output"].answer

    assert final_state["retrieved_chunks"]
    assert final_state["retrieved_chunk_ids"]

    assert final_state["worker_b_output"] is not None
    assert isinstance(final_state["worker_b_output"], AgentAnswer)
    assert final_state["worker_b_output"].answer

    assert final_state["critic_output"] is not None
    assert isinstance(final_state["critic_output"], CriticOutput)
    assert final_state["critic_output"].recommended_answer

    assert final_state["coordinator_output"] is not None
    assert isinstance(final_state["coordinator_output"], CoordinatorOutput)
    assert final_state["coordinator_output"].final_answer

    assert final_state["prediction"] == final_state["coordinator_output"].final_answer