from __future__ import annotations

from typing import Optional, TypedDict

from src.memory.entities import Agent


class BaselineState(TypedDict):
    """
    State shared across the baseline LangGraph workflow.

    This state only stores cross-node runtime information.
    Persistent memory data is stored in MemoryStore.
    Promotion requests are stored in PromotionRequestStore.
    Operation logs are stored in OperationLogStore.
    """

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------

    worker_a: Agent
    worker_b: Agent
    critic: Agent
    coordinator: Agent

    # ------------------------------------------------------------------
    # Memory / request IDs
    # ------------------------------------------------------------------

    private_memory_id: Optional[str]
    promotion_request_id: Optional[str]
    promoted_memory_id: Optional[str]

    # ------------------------------------------------------------------
    # Verification results
    # ------------------------------------------------------------------

    worker_b_can_read_before_promotion: Optional[bool]
    worker_b_can_read_after_promotion: Optional[bool]

    before_retrieved_memory_ids: list[str]
    after_retrieved_memory_ids: list[str]

    # ------------------------------------------------------------------
    # Operation log inspection
    # ------------------------------------------------------------------

    operation_log_count: int
    memory_history_count: int
    request_log_count: int

    # ------------------------------------------------------------------
    # Workflow status
    # ------------------------------------------------------------------

    current_step: str
    success: bool
    error_message: Optional[str]
    result_summary: Optional[str]