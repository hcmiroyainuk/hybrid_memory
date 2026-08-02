from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import pytest

from src.llm.output_schemas import (
    MemoryAccessDecisionOutput,
    MemoryAccessReviewOutput,
)
from src.adapters.memory_access_policy_models import (
    MemoryAccessPolicyResult,
)
from src.memory.governance.config import (
    GovernanceConfig,
)
from src.memory.governance.dependencies import (
    PolicyAssignmentDependencies,
)
from src.memory.governance.policy_assignment_state import (
    build_initial_policy_assignment_state,
)
from src.memory.governance.policy_assignment_workflow import (
    POLICY_ASSIGNMENT_NODE_NAMES,
    PolicyAssignmentWorkflow,
    PolicyAssignmentWorkflowError,
    build_policy_assignment_graph,
    build_policy_assignment_workflow,
    create_policy_assignment_workflow,
)
from src.memory.governance.review_gate import (
    ReviewGateResult,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class FakeCoordinator:
    initial_decision: Any
    final_decision: Any | None = None
    initial_error: Exception | None = None
    final_error: Exception | None = None
    initial_calls: list[dict[str, Any]] = field(
        default_factory=list
    )
    final_calls: list[dict[str, Any]] = field(
        default_factory=list
    )

    def assign_initial_policy(
        self,
        **kwargs: Any,
    ) -> Any:
        self.initial_calls.append(
            deepcopy(kwargs)
        )

        if self.initial_error is not None:
            raise self.initial_error

        return self.initial_decision

    def finalise_initial_policy(
        self,
        **kwargs: Any,
    ) -> Any:
        self.final_calls.append(
            deepcopy(kwargs)
        )

        if self.final_error is not None:
            raise self.final_error

        return (
            self.final_decision
            if self.final_decision is not None
            else self.initial_decision
        )


@dataclass
class FakeCritic:
    review: Any
    error: Exception | None = None
    calls: list[dict[str, Any]] = field(
        default_factory=list
    )

    def review_initial_policy(
        self,
        **kwargs: Any,
    ) -> Any:
        self.calls.append(
            deepcopy(kwargs)
        )

        if self.error is not None:
            raise self.error

        return self.review


@dataclass
class FakeReviewGate:
    result: Any
    error: Exception | None = None
    calls: list[dict[str, Any]] = field(
        default_factory=list
    )

    def requires_policy_review(
        self,
        **kwargs: Any,
    ) -> Any:
        self.calls.append(
            deepcopy(kwargs)
        )

        if self.error is not None:
            raise self.error

        return self.result


@dataclass
class FakePolicyGateway:
    result: Any
    error: Exception | None = None
    calls: list[dict[str, Any]] = field(
        default_factory=list
    )

    def apply_policy(
        self,
        **kwargs: Any,
    ) -> Any:
        self.calls.append(
            deepcopy(kwargs)
        )

        if self.error is not None:
            raise self.error

        return self.result


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _decision(
    *,
    review_required: bool = False,
    confidence: float = 0.95,
    target_scope: str = "shared",
    allowed_agent_ids: list[str] | None = None,
    reason: str = "Bob needs this memory.",
    memory_id: str = "mem_001",
) -> MemoryAccessDecisionOutput:
    return MemoryAccessDecisionOutput(
        memory_id=memory_id,
        target_scope=target_scope,
        allowed_agent_ids=(
            ["bob_agent"]
            if allowed_agent_ids is None
            else allowed_agent_ids
        ),
        review_required=review_required,
        reason=reason,
        confidence=confidence,
    )


def _review() -> MemoryAccessReviewOutput:
    return MemoryAccessReviewOutput(
        memory_id="mem_001",
        recommendation="approve",
        policy_compliant=True,
        least_privilege_satisfied=True,
        risk_labels=[],
        suggested_scope="shared",
        suggested_agent_ids=[
            "bob_agent",
        ],
        reason="The targeted ACL is acceptable.",
        confidence=0.9,
    )


def _policy_result(
    *,
    changed: bool = True,
) -> MemoryAccessPolicyResult:
    return MemoryAccessPolicyResult(
        memory_id="mem_001",
        owner_agent_id="alice_agent",
        scope="shared",
        readable_by=(
            "alice_agent",
            "bob_agent",
        ),
        writable_by=(
            "alice_agent",
        ),
        changed=changed,
        operation_id=(
            "op_001"
            if changed
            else None
        ),
    )


def _state(
    **overrides: Any,
) -> dict[str, Any]:
    state = (
        build_initial_policy_assignment_state(
            run_id="run_001",
            memory_id="mem_001",
            owner_agent_id="alice_agent",
            governance_actor_id=(
                "coordinator_agent"
            ),
            available_agent_ids=[
                "alice_agent",
                "bob_agent",
                "critic_agent",
                "coordinator_agent",
            ],
            memory_context={
                "summary": "A reusable fact.",
            },
            policy_context={
                "global_sharing_allowed": (
                    False
                ),
            },
            task_context={
                "task_id": "task_001",
            },
            risk_labels=[],
        )
    )
    state.update(overrides)
    return state


def _workflow(
    *,
    gate_result: ReviewGateResult | None = None,
    initial_decision: Any | None = None,
    final_decision: Any | None = None,
    gateway_result: Any | None = None,
    config: GovernanceConfig | None = None,
) -> tuple[
    PolicyAssignmentWorkflow,
    FakeCoordinator,
    FakeCritic,
    FakeReviewGate,
    FakePolicyGateway,
    PolicyAssignmentDependencies,
]:
    coordinator = FakeCoordinator(
        initial_decision=(
            initial_decision
            if initial_decision is not None
            else _decision()
        ),
        final_decision=final_decision,
    )
    critic = FakeCritic(
        review=_review()
    )
    gate = FakeReviewGate(
        result=(
            gate_result
            if gate_result is not None
            else ReviewGateResult.no_review()
        )
    )
    gateway = FakePolicyGateway(
        result=(
            gateway_result
            if gateway_result is not None
            else _policy_result()
        )
    )

    dependencies = (
        PolicyAssignmentDependencies.create(
            coordinator=coordinator,
            critic=critic,
            access_policy_gateway=gateway,
            review_gate=gate,
            config=(
                config or GovernanceConfig()
            ),
        )
    )

    workflow = (
        build_policy_assignment_workflow(
            dependencies=dependencies
        )
    )

    return (
        workflow,
        coordinator,
        critic,
        gate,
        gateway,
        dependencies,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_build_graph_returns_builder_and_bound_nodes() -> None:
    (
        _,
        _,
        _,
        _,
        _,
        dependencies,
    ) = _workflow()

    builder, nodes = (
        build_policy_assignment_graph(
            dependencies=dependencies
        )
    )

    assert builder is not None
    assert nodes.dependencies is dependencies
    assert POLICY_ASSIGNMENT_NODE_NAMES == (
        "validate_policy_assignment_input",
        "assign_initial_policy",
        "evaluate_policy_review",
        "review_initial_policy",
        "accept_unreviewed_policy",
        "finalise_reviewed_policy",
        "apply_access_policy",
    )


def test_build_graph_rejects_wrong_dependency_type() -> None:
    with pytest.raises(
        TypeError,
        match="PolicyAssignmentDependencies",
    ):
        build_policy_assignment_graph(
            dependencies=object(),  # type: ignore[arg-type]
        )


def test_create_workflow_alias_builds_workflow() -> None:
    (
        _,
        _,
        _,
        _,
        _,
        dependencies,
    ) = _workflow()

    workflow = (
        create_policy_assignment_workflow(
            dependencies=dependencies
        )
    )

    assert isinstance(
        workflow,
        PolicyAssignmentWorkflow,
    )
    assert workflow.nodes.dependencies is (
        dependencies
    )


# ---------------------------------------------------------------------------
# Complete graph paths
# ---------------------------------------------------------------------------


def test_unreviewed_path_applies_policy_and_skips_critic() -> None:
    (
        workflow,
        coordinator,
        critic,
        gate,
        gateway,
        _,
    ) = _workflow(
        gate_result=(
            ReviewGateResult.no_review()
        )
    )

    result = workflow.invoke(
        _state()
    )

    assert result["status"] == "applied"
    assert result[
        "policy_applied"
    ] is True
    assert result[
        "policy_changed"
    ] is True
    assert result[
        "review_required"
    ] is False
    assert result[
        "review_completed"
    ] is False

    assert len(
        coordinator.initial_calls
    ) == 1
    assert coordinator.final_calls == []
    assert critic.calls == []
    assert len(gate.calls) == 1
    assert len(gateway.calls) == 1

    assert result["node_trace"] == [
        "validate_policy_assignment_input",
        "assign_initial_policy",
        "evaluate_policy_review",
        "accept_unreviewed_policy",
        "apply_access_policy",
    ]


def test_reviewed_path_calls_critic_and_final_coordinator() -> None:
    initial = _decision(
        review_required=True,
        confidence=0.6,
    )
    final = _decision(
        review_required=False,
        confidence=0.95,
        reason="Approved after review.",
    )
    gate_result = ReviewGateResult(
        review_required=True,
        reason_codes=(
            "policy_low_confidence",
        ),
        reasons=(
            "Low confidence requires review.",
        ),
        triggered_risk_labels=(),
    )

    (
        workflow,
        coordinator,
        critic,
        gate,
        gateway,
        _,
    ) = _workflow(
        initial_decision=initial,
        final_decision=final,
        gate_result=gate_result,
    )

    result = workflow.invoke(
        _state()
    )

    assert result["status"] == "applied"
    assert result[
        "review_required"
    ] is True
    assert result[
        "review_completed"
    ] is True
    assert result[
        "review_reason_codes"
    ] == [
        "policy_low_confidence",
    ]
    assert result[
        "final_decision"
    ]["reason"] == (
        "Approved after review."
    )

    assert len(
        coordinator.initial_calls
    ) == 1
    assert len(
        coordinator.final_calls
    ) == 1
    assert len(critic.calls) == 1
    assert len(gate.calls) == 1
    assert len(gateway.calls) == 1

    assert result["node_trace"] == [
        "validate_policy_assignment_input",
        "assign_initial_policy",
        "evaluate_policy_review",
        "review_initial_policy",
        "finalise_reviewed_policy",
        "apply_access_policy",
    ]


def test_async_invoke_executes_complete_path() -> None:
    workflow, *_ = _workflow()

    result = asyncio.run(
        workflow.ainvoke(
            _state()
        )
    )

    assert result["status"] == "applied"
    assert result[
        "policy_applied"
    ] is True


# ---------------------------------------------------------------------------
# Fail-closed routing
# ---------------------------------------------------------------------------


def test_validation_failure_ends_before_coordinator() -> None:
    (
        workflow,
        coordinator,
        critic,
        gate,
        gateway,
        _,
    ) = _workflow()

    result = workflow.invoke(
        _state(
            owner_agent_id=(
                "unknown_owner"
            )
        )
    )

    assert result["status"] == "failed"
    assert result["failed_node"] == (
        "validate_policy_assignment_input"
    )
    assert coordinator.initial_calls == []
    assert critic.calls == []
    assert gate.calls == []
    assert gateway.calls == []


def test_coordinator_failure_ends_before_review_and_gateway() -> None:
    (
        workflow,
        coordinator,
        critic,
        gate,
        gateway,
        _,
    ) = _workflow()

    coordinator.initial_error = (
        RuntimeError("LLM unavailable")
    )

    result = workflow.invoke(
        _state()
    )

    assert result["status"] == "failed"
    assert result["failed_node"] == (
        "assign_initial_policy"
    )
    assert len(
        coordinator.initial_calls
    ) == 1
    assert critic.calls == []
    assert gate.calls == []
    assert gateway.calls == []


def test_review_failure_ends_before_finalisation_and_gateway() -> None:
    gate_result = ReviewGateResult(
        review_required=True,
        reason_codes=(
            "sensitive_memory",
        ),
        reasons=(
            "Sensitive memory requires review.",
        ),
        triggered_risk_labels=(
            "sensitive_content",
        ),
    )
    (
        workflow,
        coordinator,
        critic,
        _,
        gateway,
        _,
    ) = _workflow(
        initial_decision=_decision(
            review_required=True
        ),
        gate_result=gate_result,
    )
    critic.error = RuntimeError(
        "Critic unavailable"
    )

    result = workflow.invoke(
        _state()
    )

    assert result["status"] == "failed"
    assert result["failed_node"] == (
        "review_initial_policy"
    )
    assert coordinator.final_calls == []
    assert len(critic.calls) == 1
    assert gateway.calls == []


def test_gateway_failure_returns_terminal_failed_state() -> None:
    (
        workflow,
        _,
        _,
        _,
        gateway,
        _,
    ) = _workflow()
    gateway.error = RuntimeError(
        "Persistence unavailable"
    )

    result = workflow.invoke(
        _state()
    )

    assert result["status"] == "failed"
    assert result["failed_node"] == (
        "apply_access_policy"
    )
    assert result[
        "policy_applied"
    ] is False
    assert "Persistence unavailable" in (
        result["error_message"]
    )


def test_fail_closed_false_is_wrapped_by_workflow() -> None:
    config = GovernanceConfig(
        fail_closed=False
    )
    (
        workflow,
        coordinator,
        _,
        _,
        _,
        _,
    ) = _workflow(
        config=config
    )
    coordinator.initial_error = RuntimeError(
        "LLM unavailable"
    )

    with pytest.raises(
        PolicyAssignmentWorkflowError,
        match="graph execution failed",
    ) as exc_info:
        workflow.invoke(
            _state()
        )

    assert isinstance(
        exc_info.value.__cause__,
        Exception,
    )


# ---------------------------------------------------------------------------
# Wrapper validation and output properties
# ---------------------------------------------------------------------------


def test_invoke_rejects_non_mapping_state() -> None:
    workflow, *_ = _workflow()

    with pytest.raises(
        PolicyAssignmentWorkflowError,
        match="state must be a mapping",
    ):
        workflow.invoke(
            ["not", "a", "mapping"]  # type: ignore[arg-type]
        )


def test_invoke_does_not_mutate_caller_input() -> None:
    workflow, *_ = _workflow()
    initial_state = _state()
    original = deepcopy(
        initial_state
    )

    result = workflow.invoke(
        initial_state
    )

    assert initial_state == original
    assert result is not initial_state
    assert result["status"] == "applied"


def test_terminal_result_is_json_serialisable() -> None:
    workflow, *_ = _workflow()

    result = workflow.invoke(
        _state()
    )

    encoded = json.dumps(
        result,
        ensure_ascii=False,
    )

    assert '"status": "applied"' in (
        encoded
    )


def test_successful_idempotent_gateway_no_op_is_still_applied() -> None:
    (
        workflow,
        _,
        _,
        _,
        gateway,
        _,
    ) = _workflow(
        gateway_result=_policy_result(
            changed=False
        )
    )

    result = workflow.invoke(
        _state()
    )

    assert len(gateway.calls) == 1
    assert result["status"] == "applied"
    assert result[
        "policy_applied"
    ] is True
    assert result[
        "policy_changed"
    ] is False