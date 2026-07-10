from src.memory.entities import (
    MemoryOperationRecord,
    MemoryOperationType,
    ConflictType,
)

from src.memory.manager import OperationLogStore


store = OperationLogStore(file_path="data/test_operation_logs.json")
store.clear()

create_record = MemoryOperationRecord.create_record(
    memory_id="mem_001",
    actor_agent_id="agent_worker_a",
    after_state={
        "content": "Workers cannot directly write shared memory.",
        "scope": "private",
        "status": "active",
    },
    reason="Initial private memory creation.",
)

store.append(create_record)

print("Created record id:", create_record.record_id)
print("Count:", store.count())

found = store.get_by_id(create_record.record_id)
print("Found operation type:", found.operation_type)

by_memory = store.list_by_memory_id("mem_001")
print("Logs for mem_001:", len(by_memory))

by_agent = store.list_by_agent_id("agent_worker_a")
print("Logs by agent_worker_a:", len(by_agent))

by_type = store.list_by_operation_type(MemoryOperationType.CREATE)
print("Create logs:", len(by_type))


conflict_record = MemoryOperationRecord.detect_conflict_record(
    memory_a_id="mem_001",
    memory_b_id="mem_002",
    actor_agent_id="agent_critic",
    conflict_type=ConflictType.CONTRADICTION,
    description="mem_001 says workers cannot write shared memory, but mem_002 says workers can.",
)

store.append(conflict_record)

print("Count after conflict:", store.count())

conflict_logs = store.list_by_operation_type(MemoryOperationType.DETECT_CONFLICT)
print("Conflict logs:", len(conflict_logs))

print("Exported JSON-like records:")
for item in store.export_as_dicts():
    print(item)