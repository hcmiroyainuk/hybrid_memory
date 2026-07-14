from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from src.memory.context import GovernedContextBuilder
from src.memory.entities import Agent, MemoryItem, MemoryMetadata
from src.memory.external_knowledge import ExternalKnowledgeRetriever
from src.llm import (
    AgentAnswer,
    CriticOutput,
    CoordinatorOutput,
    LLMClient,
)
from src.memory.manager.memory_store import MemoryStore
from src.workflow.rag_workflow import (
    GovernedRAGQAWorkflow,
    create_initial_governed_rag_qa_state,
)
from src.memory.services.memory_service import MemoryService
from src.memory.services.permission_service import PermissionService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

TEST_DATA_DIR = Path("data/test_governed_rag_qa_workflow")
TEST_MEMORY_FILE = TEST_DATA_DIR / "memories.json"


def _clear_test_file() -> None:
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if TEST_MEMORY_FILE.exists():
        TEST_MEMORY_FILE.unlink()


def _create_test_agents():
    """
    Create test agents.

    If your Agent factory method signatures are different,
    adjust this helper only.
    """

    worker_a = Agent.worker(
        agent_id="worker_a",
    )

    worker_b = Agent.worker(
        agent_id="worker_b",
    )

    critic = Agent.critic(
        agent_id="critic",
    )

    coordinator = Agent.coordinator(
        agent_id="coordinator",
    )

    return worker_a, worker_b, critic, coordinator


def _create_private_memory(
    *,
    memory_id: str,
    content: str,
    owner_agent_id: str,
    creator_agent_id: str,
) -> MemoryItem:
    """
    Create private memory for testing read governance.
    """

    metadata = MemoryMetadata(
        memory_id=memory_id,
        scope="private",
        owner_agent_id=owner_agent_id,
        created_by_agent_id=creator_agent_id,
        status="active",
        memory_type="fact",
        tags=["test", owner_agent_id],
        readable_by=[owner_agent_id],
        writable_by=[owner_agent_id],
    )

    return MemoryItem(
        memory_id=memory_id,
        content=content,
        metadata=metadata,
    )


def _create_shared_memory(
    *,
    memory_id: str,
    content: str,
    creator_agent_id: str,
) -> MemoryItem:
    """
    Create shared memory for testing read governance.
    """

    metadata = MemoryMetadata(
        memory_id=memory_id,
        scope="shared",
        owner_agent_id="shared",
        created_by_agent_id=creator_agent_id,
        status="active",
        memory_type="fact",
        tags=["test", "shared"],
        readable_by=["*"],
        writable_by=["coordinator"],
    )

    return MemoryItem(
        memory_id=memory_id,
        content=content,
        metadata=metadata,
    )


def _setup_memory_service_with_test_memories() -> MemoryService:
    """
    Create MemoryService and seed test memories.

    This setup creates:
    - Worker A private memory
    - Worker B private memory
    - shared memory
    """

    _clear_test_file()

    # Use positional argument to avoid mismatch with MemoryStore parameter name.
    memory_store = MemoryStore(TEST_MEMORY_FILE)

    permission_service = PermissionService()

    memory_service = MemoryService(
        memory_store=memory_store,
        permission_service=permission_service,
        operation_log_store=None,
        memory_retriever=None,
    )

    worker_a_private = _create_private_memory(
        memory_id="memory_worker_a_private",
        content=(
            "Worker A private memory: Beyonce became popular in the late 1990s."
        ),
        owner_agent_id="worker_a",
        creator_agent_id="worker_a",
    )

    worker_b_private = _create_private_memory(
        memory_id="memory_worker_b_private",
        content=(
            "Worker B private memory: Beyonce's popularity started "
            "in the late 1990s."
        ),
        owner_agent_id="worker_b",
        creator_agent_id="worker_b",
    )

    shared_memory = _create_shared_memory(
        memory_id="memory_shared_beyonce",
        content=(
            "Shared memory: Beyonce is an American singer and performer."
        ),
        creator_agent_id="coordinator",
    )

    memory_service.memory_store.create(worker_a_private)
    memory_service.memory_store.create(worker_b_private)
    memory_service.memory_store.create(shared_memory)

    return memory_service


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY is not set.",
)
def test_governed_rag_qa_workflow_single_question() -> None:
    """
    Test the full governed LangGraph RAG QA workflow.

    This test checks:
    - Worker A receives only Worker A accessible memory.
    - Worker B receives only Worker B accessible memory.
    - Shared memory is visible to both workers.
    - External RAG retrieval still works.
    - Worker A, Worker B, Critic, and Coordinator all produce structured outputs.
    """

    worker_a, worker_b, critic, coordinator = _create_test_agents()

    memory_service = _setup_memory_service_with_test_memories()

    governed_context_builder = GovernedContextBuilder(
        memory_service=memory_service,
        memory_retriever=None,
        default_top_k=10,
        max_chars_per_memory=800,
    )

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

    workflow = GovernedRAGQAWorkflow(
        llm_client=llm_client,
        external_retriever=external_retriever,
        governed_context_builder=governed_context_builder,
    )

    initial_state = create_initial_governed_rag_qa_state(
        sample_id="test_beyonce_governed_001",
        title="Beyoncé",
        question="When did Beyonce start becoming popular?",
        gold_answers=["in the late 1990s"],
        gold_context_hash=None,
        worker_a_agent=worker_a,
        worker_b_agent=worker_b,
        critic_agent=critic,
        coordinator_agent=coordinator,
        retrieval_top_k=3,
        memory_top_k=10,
    )

    final_state = workflow.run(initial_state)

    from tests.utils import format_governed_rag_qa_state_summary

    print(format_governed_rag_qa_state_summary(final_state))

    # Workflow status
    assert final_state["success"] is True
    assert final_state["error_message"] is None
    assert final_state["current_step"] == "coordinator_finalized"

    # Worker A memory access
    assert "memory_worker_a_private" in final_state["worker_a_memory_ids"]
    assert "memory_worker_b_private" not in final_state["worker_a_memory_ids"]
    assert "memory_shared_beyonce" in final_state["worker_a_memory_ids"]

    assert final_state["worker_a_memory_text"] is not None
    assert "Worker A private memory" in final_state["worker_a_memory_text"]
    assert "Worker B private memory" not in final_state["worker_a_memory_text"]

    # Worker B memory access
    assert "memory_worker_b_private" in final_state["worker_b_memory_ids"]
    assert "memory_worker_a_private" not in final_state["worker_b_memory_ids"]
    assert "memory_shared_beyonce" in final_state["worker_b_memory_ids"]

    assert final_state["worker_b_memory_text"] is not None
    assert "Worker B private memory" in final_state["worker_b_memory_text"]
    assert "Worker A private memory" not in final_state["worker_b_memory_text"]

    # External retrieval
    assert final_state["retrieved_chunks"]
    assert final_state["retrieved_chunk_ids"]

    # Agent outputs
    assert final_state["worker_a_output"] is not None
    assert isinstance(final_state["worker_a_output"], AgentAnswer)
    assert final_state["worker_a_output"].answer

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