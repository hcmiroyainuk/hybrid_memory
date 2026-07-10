from src.memory.entities import (
    Agent,
    MemoryItem,
    MemoryMetadata,
    MemoryType,
)
from src.memory.manager import MemoryStore
from src.memory.retrieval import MemoryRetriever


store = MemoryStore()

worker_a = Agent.worker(agent_id="agent_worker_a", name="Worker A")
worker_b = Agent.worker(agent_id="agent_worker_b", name="Worker B")
coordinator = Agent.coordinator()

# Shared memory: all agents should be able to retrieve this.
shared_metadata = MemoryMetadata.shared(
    owner_agent_id=coordinator.agent_id,
    created_by_agent_id=coordinator.agent_id,
    memory_type=MemoryType.RULE,
    tags=["shared_memory", "access_control", "promotion"],
    importance=0.95,
    confidence=1.0,
)

shared_memory = MemoryItem(
    content="Workers cannot directly write shared memory. They must submit a promotion request first.",
    summary="Workers need promotion approval before shared memory write.",
    metadata=shared_metadata,
)

store.create(shared_memory)

# Worker A private memory.
worker_a_private_metadata = MemoryMetadata.private(
    owner_agent_id=worker_a.agent_id,
    created_by_agent_id=worker_a.agent_id,
    memory_type=MemoryType.NOTE,
    tags=["worker_a", "schema_design"],
    importance=0.8,
    confidence=0.9,
)

worker_a_private_memory = MemoryItem(
    content="Worker A prefers a simple unified MemoryItem schema instead of many memory types.",
    summary="Worker A prefers simple memory schema.",
    metadata=worker_a_private_metadata,
)

store.create(worker_a_private_memory)

# Worker B private memory.
worker_b_private_metadata = MemoryMetadata.private(
    owner_agent_id=worker_b.agent_id,
    created_by_agent_id=worker_b.agent_id,
    memory_type=MemoryType.NOTE,
    tags=["worker_b", "private_strategy"],
    importance=0.8,
    confidence=0.9,
)

worker_b_private_memory = MemoryItem(
    content="Worker B's private memory says to use a complex graph-based memory schema.",
    summary="Worker B prefers graph-based schema.",
    metadata=worker_b_private_metadata,
)

store.create(worker_b_private_memory)

retriever = MemoryRetriever(
    memory_store=store,
    embedding_model="text-embedding-3-small",
)

results = retriever.retrieve(
    agent=worker_a,
    query="How should workers write shared memory?",
    top_k=5,
)

print("Retrieved memories for Worker A:")
for memory in results:
    print("----")
    print("memory_id:", memory.memory_id)
    print("scope:", memory.metadata.scope)
    print("owner:", memory.metadata.owner_agent_id)
    print("content:", memory.content)

result_ids = {memory.memory_id for memory in results}

assert shared_memory.memory_id in result_ids
assert worker_b_private_memory.memory_id not in result_ids

print("Test passed: Worker A cannot retrieve Worker B's private memory.")