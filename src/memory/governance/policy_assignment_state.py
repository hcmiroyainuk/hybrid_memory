from __future__ import annotations

"""
LangGraph state for the generic initial memory-policy assignment subgraph.

The state contains only JSON-serialisable workflow data. Long-lived runtime
objects such as Coordinators, Critics, Gateways, Services, Stores, and LLM
clients belong in ``PolicyAssignmentDependencies`` rather than in this state.
"""

from collections.abc import Mapping, Sequence
from operator import add
from typing import (
    Annotated,
    Any,
    Final,
    Literal,
    TypedDict,
)


PolicyAssignmentStatus = Literal[
    "initialised",
    "validated",
    "policy_proposed",
    "review_pending",
    "reviewed",
    "finalised",
    "applied",
    "failed",
]

TERMINAL_POLICY_ASSIGNMENT_STATUSES: Final[
    frozenset[str]
] = frozenset(
    {
        "applied",
        "failed",
    }
)


class PolicyAssignmentState(
    TypedDict,
    total=False,
):
    """
    Shared state carried by the initial access-policy assignment subgraph.

    Pydantic decision objects are stored as ``model_dump(mode="json")``
    dictionaries so the state remains compatible with LangGraph
    checkpointing and JSON serialisation.
    """

    # ------------------------------------------------------------------
    # Invocation identity
    # ------------------------------------------------------------------

    run_id: str
    status: PolicyAssignmentStatus

    # ------------------------------------------------------------------
    # Required workflow input
    # ------------------------------------------------------------------

    memory_id: str
    owner_agent_id: str

    # Agent authorised to execute the final ACL change, normally the
    # Coordinator. This ID is passed to MemoryAccessPolicyGateway.
    governance_actor_id: str

    # Agents that may legally appear in a targeted shared ACL.
    available_agent_ids: list[str]

    # ------------------------------------------------------------------
    # Generic decision context
    # ------------------------------------------------------------------

    # Safe representation of the memory supplied to the decision adapters.
    # The exact shape is experiment-specific but must remain serialisable.
    memory_context: dict[str, Any]

    # Active governance rules and constraints.
    policy_context: dict[str, Any]

    # Current task or workflow context.
    task_context: dict[str, Any]

    # Risk labels supplied before Coordinator/Critic evaluation.
    risk_labels: list[str]

    # ------------------------------------------------------------------
    # Coordinator initial decision
    # ------------------------------------------------------------------

    # Serialised MemoryAccessDecisionOutput.
    initial_decision: dict[str, Any] | None

    # ------------------------------------------------------------------
    # Deterministic review-gate result
    # ------------------------------------------------------------------

    review_required: bool
    review_reason_codes: list[str]
    review_reasons: list[str]
    triggered_risk_labels: list[str]

    # ------------------------------------------------------------------
    # Critic review and Coordinator finalisation
    # ------------------------------------------------------------------

    # Serialised MemoryAccessReviewOutput.
    critic_review: dict[str, Any] | None

    review_completed: bool

    # Serialised final MemoryAccessDecisionOutput. On the no-review path,
    # this is a copy of ``initial_decision``.
    final_decision: dict[str, Any] | None

    # ------------------------------------------------------------------
    # Gateway execution result
    # ------------------------------------------------------------------

    # Serialised MemoryAccessPolicyResult.
    policy_result: dict[str, Any] | None

    # True when the Gateway call completed successfully, including a
    # successful idempotent no-op.
    policy_applied: bool

    # Mirrors MemoryAccessPolicyResult.changed when available.
    policy_changed: bool

    # ------------------------------------------------------------------
    # Observability and failure handling
    # ------------------------------------------------------------------

    # ``add`` preserves the actual execution order, including retries.
    node_trace: Annotated[
        list[str],
        add,
    ]

    warnings: Annotated[
        list[str],
        add,
    ]

    errors: Annotated[
        list[str],
        add,
    ]

    failed_node: str | None
    error_type: str | None
    error_message: str | None


def build_initial_policy_assignment_state(
    *,
    memory_id: str,
    owner_agent_id: str,
    governance_actor_id: str,
    available_agent_ids: Sequence[str],
    memory_context: Mapping[str, Any] | None = None,
    policy_context: Mapping[str, Any] | None = None,
    task_context: Mapping[str, Any] | None = None,
    risk_labels: Sequence[str] = (),
    run_id: str | None = None,
) -> PolicyAssignmentState:
    """
    Build a validated, serialisable initial state for one policy assignment.

    The owner and governance actor must be registered in
    ``available_agent_ids``. The wildcard ``"*"`` is not a valid registered
    Agent ID and is therefore removed from this list.
    """
    clean_memory_id = _required_text(
        memory_id,
        "memory_id",
    )
    clean_owner_id = _required_text(
        owner_agent_id,
        "owner_agent_id",
    )
    clean_governance_actor_id = _required_text(
        governance_actor_id,
        "governance_actor_id",
    )

    clean_available_agent_ids = [
        agent_id
        for agent_id in _clean_string_list(
            available_agent_ids
        )
        if agent_id != "*"
    ]

    if not clean_available_agent_ids:
        raise ValueError(
            "available_agent_ids must contain at least one Agent ID."
        )

    if clean_owner_id not in clean_available_agent_ids:
        raise ValueError(
            "owner_agent_id must appear in available_agent_ids."
        )

    if (
        clean_governance_actor_id
        not in clean_available_agent_ids
    ):
        raise ValueError(
            "governance_actor_id must appear in "
            "available_agent_ids."
        )

    clean_run_id = _optional_text(run_id)

    return PolicyAssignmentState(
        run_id=(
            clean_run_id
            or f"policy:{clean_memory_id}"
        ),
        status="initialised",
        memory_id=clean_memory_id,
        owner_agent_id=clean_owner_id,
        governance_actor_id=(
            clean_governance_actor_id
        ),
        available_agent_ids=(
            clean_available_agent_ids
        ),
        memory_context=_clean_context(
            memory_context,
            "memory_context",
        ),
        policy_context=_clean_context(
            policy_context,
            "policy_context",
        ),
        task_context=_clean_context(
            task_context,
            "task_context",
        ),
        risk_labels=_clean_labels(
            risk_labels
        ),
        initial_decision=None,
        review_required=False,
        review_reason_codes=[],
        review_reasons=[],
        triggered_risk_labels=[],
        critic_review=None,
        review_completed=False,
        final_decision=None,
        policy_result=None,
        policy_applied=False,
        policy_changed=False,
        node_trace=[],
        warnings=[],
        errors=[],
        failed_node=None,
        error_type=None,
        error_message=None,
    )


def is_terminal_policy_assignment_status(
    status: str | None,
) -> bool:
    """
    Return whether a workflow status is terminal.
    """
    if status is None:
        return False

    return (
        str(status).strip().lower()
        in TERMINAL_POLICY_ASSIGNMENT_STATUSES
    )


def validate_policy_assignment_identity(
    state: Mapping[str, Any],
) -> None:
    """
    Validate the identity fields required by every policy-assignment node.

    This helper intentionally validates only stable invocation identity.
    Node-specific fields such as ``initial_decision`` or ``critic_review``
    should be validated at the node that consumes them.
    """
    memory_id = _required_text(
        state.get("memory_id"),
        "memory_id",
    )
    owner_agent_id = _required_text(
        state.get("owner_agent_id"),
        "owner_agent_id",
    )
    governance_actor_id = _required_text(
        state.get("governance_actor_id"),
        "governance_actor_id",
    )

    available_agent_ids = [
        agent_id
        for agent_id in _clean_string_list(
            state.get(
                "available_agent_ids",
                [],
            )
        )
        if agent_id != "*"
    ]

    if not available_agent_ids:
        raise ValueError(
            "available_agent_ids must contain at least one Agent ID."
        )

    if owner_agent_id not in available_agent_ids:
        raise ValueError(
            "owner_agent_id must appear in available_agent_ids."
        )

    if (
        governance_actor_id
        not in available_agent_ids
    ):
        raise ValueError(
            "governance_actor_id must appear in "
            "available_agent_ids."
        )

    # Force evaluation so malformed values are detected even though the
    # normalised local variable is not returned.
    _ = memory_id


def _required_text(
    value: Any,
    field_name: str,
) -> str:
    cleaned = str(
        value or ""
    ).strip()

    if not cleaned:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    return cleaned


def _optional_text(
    value: Any,
) -> str | None:
    if value is None:
        return None

    cleaned = str(value).strip()
    return cleaned or None


def _clean_string_list(
    values: Sequence[Any] | Any,
) -> list[str]:
    if values is None:
        return []

    if isinstance(values, str):
        values = [values]

    try:
        iterator = iter(values)
    except TypeError as error:
        raise ValueError(
            "Expected a sequence of string values."
        ) from error

    result: list[str] = []

    for value in iterator:
        cleaned = str(
            value or ""
        ).strip()

        if (
            cleaned
            and cleaned not in result
        ):
            result.append(cleaned)

    return result


def _clean_labels(
    values: Sequence[Any] | Any,
) -> list[str]:
    return [
        value.lower()
        for value in _clean_string_list(values)
    ]


def _clean_context(
    value: Mapping[str, Any] | None,
    field_name: str,
) -> dict[str, Any]:
    if value is None:
        return {}

    if not isinstance(value, Mapping):
        raise ValueError(
            f"{field_name} must be a mapping."
        )

    return dict(value)


__all__ = [
    "PolicyAssignmentStatus",
    "TERMINAL_POLICY_ASSIGNMENT_STATUSES",
    "PolicyAssignmentState",
    "build_initial_policy_assignment_state",
    "is_terminal_policy_assignment_status",
    "validate_policy_assignment_identity",
]