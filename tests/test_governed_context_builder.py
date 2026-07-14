from __future__ import annotations

from pathlib import Path

from src.memory.context import GovernedContextBuilder
from src.memory.entities import Agent, MemoryItem, MemoryMetadata
from src.memory.manager.memory_store import MemoryStore
from src.memory.services.memory_service import MemoryService
from src.memory.services.permission_service import PermissionService


TEST_DATA_DIR = Path("data/test_governed_context_builder")
TEST_MEMORY_FILE = TEST_DATA_DIR / "memories.json"


def _clear_test_file() -> None:
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if TEST_MEMORY_FILE.exists():
        TEST_MEMORY_FILE.unlink()


def _create_test_agents():
    """
    Create test agents.

    If your Agent factory methods have different signatures,
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
    Create a private MemoryItem for tests.

    If your MemoryItem / MemoryMetadata constructors are different,
    adjust this helper only.
    """

    metadata = MemoryMetadata(
        memory_id=memory_id,
        scope="private",
        owner_agent_id=owner_agent_id,
        creator_agent_id=creator_agent_id,
        status="active",
        memory_type="qa_memory",
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
    Create a shared MemoryItem for tests.

    Shared memory should be readable by all agents.
    """

    metadata = MemoryMetadata(
        memory_id=memory_id,
        scope="shared",
        owner_agent_id="shared",
        creator_agent_id=creator_agent_id,
        status="active",
        memory_type="qa_memory",
        tags=["test", "shared"],
        readable_by=["*"],
        writable_by=["coordinator"],
    )

    return MemoryItem(
        memory_id=memory_id,
        content=content,
        metadata=metadata,
    )


def _setup_memory_service() -> MemoryService:
    """
    Create MemoryService with JSON-backed test MemoryStore.
    """

    _clear_test_file()

    memory_store = MemoryStore(
        file_path=TEST_MEMORY_FILE,
    )

    permission_service = PermissionService()

    memory_service = MemoryService(
        memory_store=memory_store,
        permission_service=permission_service,
        operation_log_store=None,
        memory_retriever=None,
    )

    return memory_service


def test_governed_context_builder_private_and_shared_memory_access() -> None:
    """
    Test read governance behavior.

    Expected:
    - Worker A can retrieve Worker A private memory.
    - Worker A cannot retrieve Worker B private memory.
    - Worker B can retrieve Worker B private memory.
    - Worker B cannot retrieve Worker A private memory.
    - Both workers can retrieve shared memory.
    """

    worker_a, worker_b, critic, coordinator = _create_test_agents()

    memory_service = _setup_memory_service()

    worker_a_private = _create_private_memory(
        memory_id="memory_worker_a_private",
        content="Worker A private memory says Beyonce became popular in the late 1990s.",
        owner_agent_id="worker_a",
        creator_agent_id="worker_a",
    )

    worker_b_private = _create_private_memory(
        memory_id="memory_worker_b_private",
        content="Worker B private memory says Paris is the capital of France.",
        owner_agent_id="worker_b",
        creator_agent_id="worker_b",
    )

    shared_memory = _create_shared_memory(
        memory_id="memory_shared",
        content="Shared memory says Beyonce is an American singer.",
        creator_agent_id="coordinator",
    )

    memory_service.memory_store.create(worker_a_private)
    memory_service.memory_store.create(worker_b_private)
    memory_service.memory_store.create(shared_memory)

    context_builder = GovernedContextBuilder(
        memory_service=memory_service,
        memory_retriever=None,
        default_top_k=10,
    )

    worker_a_context = context_builder.build_memory_context(
        agent=worker_a,
        query="When did Beyonce become popular?",
        top_k=10,
    )

    worker_b_context = context_builder.build_memory_context(
        agent=worker_b,
        query="When did Beyonce become popular?",
        top_k=10,
    )

    print("\n========== WORKER A MEMORY CONTEXT ==========")
    print(worker_a_context.formatted_context)
    print("Worker A memory ids:", worker_a_context.memory_ids)

    print("\n========== WORKER B MEMORY CONTEXT ==========")
    print(worker_b_context.formatted_context)
    print("Worker B memory ids:", worker_b_context.memory_ids)

    assert "memory_worker_a_private" in worker_a_context.memory_ids
    assert "memory_worker_b_private" not in worker_a_context.memory_ids
    assert "memory_shared" in worker_a_context.memory_ids

    assert "memory_worker_b_private" in worker_b_context.memory_ids
    assert "memory_worker_a_private" not in worker_b_context.memory_ids
    assert "memory_shared" in worker_b_context.memory_ids

    assert "Worker A private memory" in worker_a_context.formatted_context
    assert "Worker B private memory" not in worker_a_context.formatted_context

    assert "Worker B private memory" in worker_b_context.formatted_context
    assert "Worker A private memory" not in worker_b_context.formatted_context


def test_governed_context_builder_returns_no_accessible_memory_when_empty() -> None:
    """
    Test empty memory store behavior.
    """

    worker_a, _, _, _ = _create_test_agents()

    memory_service = _setup_memory_service()

    context_builder = GovernedContextBuilder(
        memory_service=memory_service,
        memory_retriever=None,
        default_top_k=5,
    )

    context = context_builder.build_memory_context(
        agent=worker_a,
        query="What is Beyonce known for?",
        top_k=5,
    )

    print("\n========== EMPTY MEMORY CONTEXT ==========")
    print(context.formatted_context)

    assert context.memory_ids == []
    assert context.memory_items == []
    assert context.formatted_context == "No accessible memory."