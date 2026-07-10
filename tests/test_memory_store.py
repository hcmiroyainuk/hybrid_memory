from src.memory.entities import MemoryItem, MemoryMetadata, MemoryType
from src.memory.manager import MemoryStore
from src.memory.entities import Agent


store = MemoryStore()

worker = Agent.worker(agent_id="agent_worker_a", name="Worker A")

metadata = MemoryMetadata.private(
    owner_agent_id=worker.agent_id,
    created_by_agent_id=worker.agent_id,
    memory_type=MemoryType.RULE,
    tags=["shared_memory", "access_control"],
    importance=0.9,
    confidence=1.0,
)

memory = MemoryItem(
    content="Workers cannot directly write to shared memory.",
    summary="Workers need approval before writing shared memory.",
    metadata=metadata,
)

created = store.create(memory)
print("Created:", created.memory_id)

found = store.get_by_id(created.memory_id)
print("Found:", found.content)

active_memories = store.list_active()
print("Active count:", len(active_memories))

tagged = store.list_by_tags(["shared_memory"], active_only=True)
print("Tagged count:", len(tagged))

updated = store.update_content(
    created.memory_id,
    content="Workers must submit a promotion request before writing to shared memory.",
    summary="Workers need promotion approval before shared memory write.",
)
print("Updated:", updated.content)

deprecated = store.deprecate(created.memory_id)
print("Status after deprecate:", deprecated.metadata.status)

active_after_delete = store.list_active()
print("Active count after deprecate:", len(active_after_delete))

all_memories = store.list_all(include_deprecated=True)
print("All count:", len(all_memories))