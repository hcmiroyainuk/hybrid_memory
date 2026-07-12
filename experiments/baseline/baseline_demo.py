from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from src.memory.entities import (
    Agent,
    PromotionRequest,
    PromotionStatus,
)

from src.memory.manager import (
    MemoryStore,
    OperationLogStore,
)

from src.memory.retrieval import MemoryRetriever

from src.memory.services import (
    PermissionService,
    OperationLogService,
    MemoryService,
    PromotionService,
)

from experiments.baseline.baseline_workflow import build_baseline_workflow
from experiments.baseline.baseline_workflow import create_initial_baseline_state



class InMemoryPromotionRequestStore:
    """
    Temporary in-memory promotion request store for baseline demo.

    Later, replace this with a JSON-backed PromotionRequestStore.
    """

    def __init__(self) -> None:
        self._requests: dict[str, PromotionRequest] = {}

    def create(self, request: PromotionRequest) -> PromotionRequest:
        if request.request_id in self._requests:
            raise ValueError(
                f"Promotion request '{request.request_id}' already exists."
            )

        self._requests[request.request_id] = request
        return request

    def get_by_id(self, request_id: str) -> PromotionRequest:
        if request_id not in self._requests:
            raise ValueError(
                f"Promotion request '{request_id}' was not found."
            )

        return self._requests[request_id]

    def replace(self, request: PromotionRequest) -> PromotionRequest:
        if request.request_id not in self._requests:
            raise ValueError(
                f"Promotion request '{request.request_id}' was not found."
            )

        self._requests[request.request_id] = request
        return request

    def list_all(self) -> list[PromotionRequest]:
        return list(self._requests.values())

    def list_pending(self) -> list[PromotionRequest]:
        return [
            request
            for request in self._requests.values()
            if request.status == PromotionStatus.PENDING
        ]

    def clear(self) -> None:
        self._requests.clear()


def build_services():
    """
    Build stores, retriever, and services for the baseline demo.
    """

    memory_store = MemoryStore(
        file_path="data/demo_memories.json"
    )

    operation_log_store = OperationLogStore(
        file_path="data/demo_operation_logs.json"
    )

    request_store = InMemoryPromotionRequestStore()

    memory_store.clear()
    operation_log_store.clear()
    request_store.clear()

    permission_service = PermissionService()

    operation_log_service = OperationLogService(
        operation_log_store=operation_log_store
    )

    memory_retriever = MemoryRetriever(
        memory_store=memory_store,
        embedding_model="text-embedding-3-small",
    )

    memory_service = MemoryService(
        memory_store=memory_store,
        operation_log_store=operation_log_store,
        memory_retriever=memory_retriever,
        permission_service=permission_service,
    )

    promotion_service = PromotionService(
        memory_store=memory_store,
        request_store=request_store,
        operation_log_service=operation_log_service,
        permission_service=permission_service,
    )

    return {
        "memory_store": memory_store,
        "operation_log_store": operation_log_store,
        "request_store": request_store,
        "permission_service": permission_service,
        "operation_log_service": operation_log_service,
        "memory_service": memory_service,
        "promotion_service": promotion_service,
    }


def create_agents():
    """
    Create baseline agents.
    """

    worker_a = Agent.worker(
        agent_id="agent_worker_a",
        name="Worker A",
    )

    worker_b = Agent.worker(
        agent_id="agent_worker_b",
        name="Worker B",
    )

    # If your Agent.critic() does not support name=,
    # remove the name argument.
    critic = Agent.critic(
        agent_id="agent_critic",
        name="Critic",
    )

    # If your Agent.coordinator() does not support name=,
    # remove the name argument.
    coordinator = Agent.coordinator(
        agent_id="agent_coordinator",
        name="Coordinator",
    )

    return worker_a, worker_b, critic, coordinator


def print_final_state(final_state: dict) -> None:
    """
    Print final workflow state in a readable way.
    """

    print("\n" + "=" * 80)
    print("BASELINE WORKFLOW RESULT")
    print("=" * 80)

    print(f"Success: {final_state.get('success')}")
    print(f"Current step: {final_state.get('current_step')}")
    print(f"Error message: {final_state.get('error_message')}")
    print()

    print("Memory IDs")
    print("-" * 80)
    print(f"Private memory ID: {final_state.get('private_memory_id')}")
    print(f"Promotion request ID: {final_state.get('promotion_request_id')}")
    print(f"Promoted memory ID: {final_state.get('promoted_memory_id')}")
    print()

    print("Access verification")
    print("-" * 80)
    print(
        "Worker B can read before promotion:",
        final_state.get("worker_b_can_read_before_promotion"),
    )
    print(
        "Worker B can read after promotion:",
        final_state.get("worker_b_can_read_after_promotion"),
    )
    print(
        "Before promotion retrieved IDs:",
        final_state.get("before_retrieved_memory_ids"),
    )
    print(
        "After promotion retrieved IDs:",
        final_state.get("after_retrieved_memory_ids"),
    )
    print()

    print("Operation logs")
    print("-" * 80)
    print(f"Operation log count: {final_state.get('operation_log_count')}")
    print(f"Memory history count: {final_state.get('memory_history_count')}")
    print(f"Request log count: {final_state.get('request_log_count')}")
    print()

    print("Summary")
    print("-" * 80)
    print(final_state.get("result_summary"))
    print("=" * 80)


def print_stored_memories(memory_store: MemoryStore) -> None:
    """
    Print stored memories after workflow execution.
    """

    print("\nStored memories")
    print("-" * 80)

    memories = memory_store.list_all(include_deprecated=True)

    for memory in memories:
        print(f"Memory ID: {memory.memory_id}")
        print(f"Content: {memory.content}")
        print(f"Scope: {memory.metadata.scope}")
        print(f"Status: {memory.metadata.status}")
        print(f"Owner: {memory.metadata.owner_agent_id}")
        print(f"Readable by: {memory.metadata.readable_by}")
        print(f"Writable by: {memory.metadata.writable_by}")
        print()


def print_operation_logs(operation_log_store: OperationLogStore) -> None:
    """
    Print operation logs after workflow execution.
    """

    print("\nOperation logs")
    print("-" * 80)

    logs = operation_log_store.list_all()

    for log in logs:
        print(f"Record ID: {log.record_id}")
        print(f"Operation type: {log.operation_type}")
        print(f"Target memories: {log.target_memory_ids}")
        print(f"Actor: {log.actor_agent_id}")
        print(f"Reviewer: {log.reviewer_agent_id}")
        print(f"Related request: {log.related_request_id}")
        print(f"Reason: {log.reason}")
        print()


def main() -> None:
    """
    Run the LangGraph baseline workflow.
    """

    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. "
            "Set it in your .env file or PyCharm Run Configuration."
        )

    services = build_services()

    worker_a, worker_b, critic, coordinator = create_agents()

    graph = build_baseline_workflow(
        memory_service=services["memory_service"],
        promotion_service=services["promotion_service"],
        operation_log_service=services["operation_log_service"],
    )

    initial_state = create_initial_baseline_state(
        worker_a=worker_a,
        worker_b=worker_b,
        critic=critic,
        coordinator=coordinator,
    )

    final_state = graph.invoke(initial_state)

    print_final_state(final_state)
    print_stored_memories(services["memory_store"])
    print_operation_logs(services["operation_log_store"])

    print("\nDemo data written to:")
    print(f"- {Path('data/demo_memories.json').resolve()}")
    print(f"- {Path('data/demo_operation_logs.json').resolve()}")


if __name__ == "__main__":
    main()