from __future__ import annotations

from typing import Any, Optional

from langgraph.graph import StateGraph, START, END

from src.memory.entities import (
    Agent,
    MemoryType,
)
from src.memory.manager import (
    MemoryStore,
    OperationLogStore,
)
from src.memory.retrieval import MemoryRetriever
from src.memory.services import (
    MemoryService,
    OperationLogService,
    PermissionDeniedError,
    PermissionService,
    PromotionService,
)
from .baseline_state import BaselineState


class BaselineWorkflow:
    """
    LangGraph baseline workflow for private/shared memory governance.

    Workflow:
    1. Worker A creates a private memory.
    2. Worker B verifies that the private memory is inaccessible.
    3. Worker A submits a promotion request.
    4. Critic reviews the request.
    5. Coordinator approves the request.
    6. Worker B verifies that the promoted memory is now shared and retrievable.
    7. Operation logs are inspected.

    This workflow is deterministic.
    LangGraph is used as an orchestration layer, while business logic is handled
    by the service layer.
    """

    def __init__(
        self,
        memory_service: MemoryService,
        promotion_service: PromotionService,
        operation_log_service: OperationLogService,
    ) -> None:
        self.memory_service = memory_service
        self.promotion_service = promotion_service
        self.operation_log_service = operation_log_service

    # ------------------------------------------------------------------
    # Node 1: Worker A creates private memory
    # ------------------------------------------------------------------

    def create_private_memory(
        self,
        state: BaselineState,
    ) -> dict[str, Any]:
        worker_a = state["worker_a"]

        memory = self.memory_service.create_private_memory(
            agent=worker_a,
            content=(
                "Workers cannot directly write shared memory. "
                "They must submit a promotion request first."
            ),
            summary="Workers need approval before writing shared memory.",
            memory_type=MemoryType.RULE,
            tags=["permission", "shared_memory", "promotion"],
            importance=0.9,
            confidence=1.0,
            reason="Worker A created a private access-control rule.",
        )

        return {
            "private_memory_id": memory.memory_id,
            "current_step": "private_memory_created",
        }

    # ------------------------------------------------------------------
    # Node 2: Worker B verifies private memory isolation
    # ------------------------------------------------------------------

    def verify_private_isolation(
        self,
        state: BaselineState,
    ) -> dict[str, Any]:
        worker_b = state["worker_b"]
        private_memory_id = self._require_value(
            state.get("private_memory_id"),
            "private_memory_id",
        )

        worker_b_can_read = True

        try:
            self.memory_service.get_memory(
                agent=worker_b,
                memory_id=private_memory_id,
            )
        except PermissionDeniedError:
            worker_b_can_read = False

        retrieved_ids: list[str] = []

        if self.memory_service.memory_retriever is not None:
            retrieved_memories = self.memory_service.retrieve_memories(
                agent=worker_b,
                query="Who can write shared memory?",
                top_k=5,
            )
            retrieved_ids = [
                memory.memory_id
                for memory in retrieved_memories
            ]

        if private_memory_id in retrieved_ids:
            raise AssertionError(
                "Private memory leaked into Worker B retrieval results before promotion."
            )

        if worker_b_can_read:
            raise AssertionError(
                "Worker B should not be able to read Worker A's private memory before promotion."
            )

        return {
            "worker_b_can_read_before_promotion": worker_b_can_read,
            "before_retrieved_memory_ids": retrieved_ids,
            "current_step": "private_isolation_verified",
        }

    # ------------------------------------------------------------------
    # Node 3: Worker A submits promotion request
    # ------------------------------------------------------------------

    def submit_promotion_request(
        self,
        state: BaselineState,
    ) -> dict[str, Any]:
        worker_a = state["worker_a"]
        private_memory_id = self._require_value(
            state.get("private_memory_id"),
            "private_memory_id",
        )

        request = self.promotion_service.submit_promotion_request(
            agent=worker_a,
            memory_id=private_memory_id,
            reason="This memory is a general access-control rule and should be shared.",
        )

        return {
            "promotion_request_id": request.request_id,
            "current_step": "promotion_request_submitted",
        }

    # ------------------------------------------------------------------
    # Node 4: Critic reviews promotion request
    # ------------------------------------------------------------------

    def critic_review(
        self,
        state: BaselineState,
    ) -> dict[str, Any]:
        critic = state["critic"]
        request_id = self._require_value(
            state.get("promotion_request_id"),
            "promotion_request_id",
        )

        self.promotion_service.review_promotion_request(
            agent=critic,
            request_id=request_id,
            comment=(
                "The memory describes a general permission rule. "
                "It is suitable for shared memory."
            ),
        )

        return {
            "current_step": "critic_reviewed_promotion_request",
        }

    # ------------------------------------------------------------------
    # Node 5: Coordinator approves promotion request
    # ------------------------------------------------------------------

    def coordinator_approve(
        self,
        state: BaselineState,
    ) -> dict[str, Any]:
        coordinator = state["coordinator"]
        request_id = self._require_value(
            state.get("promotion_request_id"),
            "promotion_request_id",
        )

        promoted_memory = self.promotion_service.approve_promotion_request(
            agent=coordinator,
            request_id=request_id,
            comment="Approved. This rule should be available to all workers.",
        )

        return {
            "promoted_memory_id": promoted_memory.memory_id,
            "current_step": "promotion_approved",
        }

    # ------------------------------------------------------------------
    # Node 6: Worker B verifies shared memory retrieval
    # ------------------------------------------------------------------

    def verify_shared_retrieval(
        self,
        state: BaselineState,
    ) -> dict[str, Any]:
        worker_b = state["worker_b"]
        promoted_memory_id = self._require_value(
            state.get("promoted_memory_id"),
            "promoted_memory_id",
        )

        worker_b_can_read = True

        try:
            self.memory_service.get_memory(
                agent=worker_b,
                memory_id=promoted_memory_id,
            )
        except PermissionDeniedError:
            worker_b_can_read = False

        if not worker_b_can_read:
            raise AssertionError(
                "Worker B should be able to read the memory after promotion."
            )

        retrieved_ids: list[str] = []

        if self.memory_service.memory_retriever is not None:
            retrieved_memories = self.memory_service.retrieve_memories(
                agent=worker_b,
                query="Who can write shared memory?",
                top_k=5,
            )
            retrieved_ids = [
                memory.memory_id
                for memory in retrieved_memories
            ]

            if promoted_memory_id not in retrieved_ids:
                raise AssertionError(
                    "Promoted shared memory was not retrieved by Worker B."
                )

        return {
            "worker_b_can_read_after_promotion": worker_b_can_read,
            "after_retrieved_memory_ids": retrieved_ids,
            "current_step": "shared_retrieval_verified",
        }

    # ------------------------------------------------------------------
    # Node 7: Inspect operation logs
    # ------------------------------------------------------------------

    def inspect_operation_logs(
        self,
        state: BaselineState,
    ) -> dict[str, Any]:
        promoted_memory_id = self._require_value(
            state.get("promoted_memory_id"),
            "promoted_memory_id",
        )
        request_id = self._require_value(
            state.get("promotion_request_id"),
            "promotion_request_id",
        )

        all_logs = self.operation_log_service.list_all_operations()
        memory_history = self.operation_log_service.list_memory_history(
            promoted_memory_id
        )
        request_logs = self.operation_log_service.list_request_operations(
            request_id
        )

        if len(all_logs) == 0:
            raise AssertionError("No operation logs were recorded.")

        if len(memory_history) == 0:
            raise AssertionError("No memory history was recorded.")

        if len(request_logs) == 0:
            raise AssertionError("No request-related operation logs were recorded.")

        result_summary = (
            "Baseline completed: Worker A created a private memory; "
            "Worker B could not access it before promotion; "
            "Worker A submitted a promotion request; "
            "Critic reviewed it; Coordinator approved it; "
            "Worker B could access the promoted shared memory; "
            "operation logs were recorded."
        )

        return {
            "operation_log_count": len(all_logs),
            "memory_history_count": len(memory_history),
            "request_log_count": len(request_logs),
            "success": True,
            "error_message": None,
            "result_summary": result_summary,
            "current_step": "baseline_completed",
        }

    # ------------------------------------------------------------------
    # Build LangGraph workflow
    # ------------------------------------------------------------------

    def build(self):
        """
        Build and compile the baseline StateGraph.

        LangGraph nodes are plain Python functions that read and update state.
        Edges define the fixed workflow order.
        """

        graph = StateGraph(BaselineState)

        graph.add_node("create_private_memory", self.create_private_memory)
        graph.add_node("verify_private_isolation", self.verify_private_isolation)
        graph.add_node("submit_promotion_request", self.submit_promotion_request)
        graph.add_node("critic_review", self.critic_review)
        graph.add_node("coordinator_approve", self.coordinator_approve)
        graph.add_node("verify_shared_retrieval", self.verify_shared_retrieval)
        graph.add_node("inspect_operation_logs", self.inspect_operation_logs)

        graph.add_edge(START, "create_private_memory")
        graph.add_edge("create_private_memory", "verify_private_isolation")
        graph.add_edge("verify_private_isolation", "submit_promotion_request")
        graph.add_edge("submit_promotion_request", "critic_review")
        graph.add_edge("critic_review", "coordinator_approve")
        graph.add_edge("coordinator_approve", "verify_shared_retrieval")
        graph.add_edge("verify_shared_retrieval", "inspect_operation_logs")
        graph.add_edge("inspect_operation_logs", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_value(value: Optional[str], field_name: str) -> str:
        if value is None:
            raise ValueError(f"State field '{field_name}' is required but missing.")

        return value


def build_baseline_workflow(
    memory_service: MemoryService,
    promotion_service: PromotionService,
    operation_log_service: OperationLogService,
):
    """
    Functional helper for building the baseline workflow.
    """

    workflow = BaselineWorkflow(
        memory_service=memory_service,
        promotion_service=promotion_service,
        operation_log_service=operation_log_service,
    )

    return workflow.build()


def create_initial_baseline_state(
    worker_a: Agent,
    worker_b: Agent,
    critic: Agent,
    coordinator: Agent,
) -> BaselineState:
    """
    Create the initial state for the baseline workflow.
    """

    return {
        "worker_a": worker_a,
        "worker_b": worker_b,
        "critic": critic,
        "coordinator": coordinator,
        "private_memory_id": None,
        "promotion_request_id": None,
        "promoted_memory_id": None,
        "worker_b_can_read_before_promotion": None,
        "worker_b_can_read_after_promotion": None,
        "before_retrieved_memory_ids": [],
        "after_retrieved_memory_ids": [],
        "operation_log_count": 0,
        "memory_history_count": 0,
        "request_log_count": 0,
        "current_step": "initialized",
        "success": False,
        "error_message": None,
        "result_summary": None,
    }