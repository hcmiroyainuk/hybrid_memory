from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import pytest

from src.llm.output_schemas import (
    MemoryAccessRequestDecisionOutput,
    MemoryAccessRequestReviewOutput,
)
from src.adapters.memory_sharing_models import (
    MemoryAccessDecisionResult,
    MemoryAccessRequestResult,
)
from src.memory.governance.access_request_state import (
    build_initial_access_request_state,
)
from src.memory.governance.access_request_workflow import (
    ACCESS_REQUEST_NODE_NAMES,
    AccessRequestWorkflow,
    AccessRequestWorkflowError,
    build_access_request_graph,
    build_access_request_workflow,
    create_access_request_workflow,
)
from src.memory.governance.config import (
    GovernanceConfig,
)
from src.memory.governance.dependencies import (
    AccessRequestDependencies,
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

    def evaluate_access_request(
        self,
        **kwargs: Any,
    ) -> Any:
        self.initial_calls.append(
            deepcopy(kwargs)
        )

        if self.initial_error is not None:
            raise self.initial_error

        return self.initial_decision

    def finalise_access_request(
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

    def review_access_request(
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

    def requires_access_request_review(
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
class FakeSharingGateway:
    has_access_values: list[bool] = field(
        default_factory=lambda: [
            False,
            True,
        ]
    )

    request_result: Any = None
    review_result: Any = None
    approve_result: Any = None
    direct_approve_result: Any = None
    reject_result: Any = None

    has_access_error: Exception | None = None
    request_error: Exception | None = None
    review_error: Exception | None = None
    approve_error: Exception | None = None
    direct_approve_error: Exception | None = None
    reject_error: Exception | None = None

    has_access_calls: list[dict[str, Any]] = field(
        default_factory=list
    )
    request_calls: list[dict[str, Any]] = field(
        default_factory=list
    )
    review_calls: list[dict[str, Any]] = field(
        default_factory=list
    )
    approve_calls: list[dict[str, Any]] = field(
        default_factory=list
    )
    direct_approve_calls: list[
        dict[str, Any]
    ] = field(
        default_factory=list
    )
    reject_calls: list[dict[str, Any]] = field(
        default_factory=list
    )
    get_calls: list[dict[str, Any]] = field(
        default_factory=list
    )

    def has_access(
        self,
        **kwargs: Any,
    ) -> bool:
        self.has_access_calls.append(
            deepcopy(kwargs)
        )

        if self.has_access_error is not None:
            raise self.has_access_error

        if not self.has_access_values:
            return False

        return self.has_access_values.pop(0)

    def request_access(
        self,
        **kwargs: Any,
    ) -> Any:
        self.request_calls.append(
            deepcopy(kwargs)
        )

        if self.request_error is not None:
            raise self.request_error

        return self.request_result

    def review_request(
        self,
        **kwargs: Any,
    ) -> Any:
        self.review_calls.append(
            deepcopy(kwargs)
        )

        if self.review_error is not None:
            raise self.review_error

        return self.review_result

    def approve_request(
        self,
        **kwargs: Any,
    ) -> Any:
        self.approve_calls.append(
            deepcopy(kwargs)
        )

        if self.approve_error is not None:
            raise self.approve_error

        return self.approve_result

    def approve_direct_request(
        self,
        **kwargs: Any,
    ) -> Any:
        self.direct_approve_calls.append(
            deepcopy(kwargs)
        )

        if (
            self.direct_approve_error
            is not None
        ):
            raise self.direct_approve_error

        return self.direct_approve_result

    def reject_request(
        self,
        **kwargs: Any,
    ) -> Any:
        self.reject_calls.append(
            deepcopy(kwargs)
        )

        if self.reject_error is not None:
            raise self.reject_error

        return self.reject_result

    def get_request(
        self,
        **kwargs: Any,
    ) -> Any:
        self.get_calls.append(
            deepcopy(kwargs)
        )
        return self.request_result


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _decision(
    *,
    approved: bool = True,
    review_required: bool = False,
    confidence: float = 0.95,
    reason: str = "The request is justified.",
) -> MemoryAccessRequestDecisionOutput:
    return MemoryAccessRequestDecisionOutput(
        request_id="req_001",
        memory_id="mem_001",
        approved=approved,
        review_required=review_required,
        reason=reason,
        confidence=confidence,
    )


def _review(
    *,
    recommendation: str = "approve",
) -> MemoryAccessRequestReviewOutput:
    approved = (
        recommendation == "approve"
    )

    return MemoryAccessRequestReviewOutput(
        request_id="req_001",
        memory_id="mem_001",
        recommendation=recommendation,
        request_justified=approved,
        policy_compliant=approved,
        least_privilege_satisfied=(
            approved
        ),
        risk_labels=[],
        reason=(
            "The request is policy compliant."
            if approved
            else "The request should be rejected."
        ),
        confidence=0.9,
    )


def _request_result(
    *,
    status: str = "pending",
    critic_recommendation: (
        str | None
    ) = None,
) -> MemoryAccessRequestResult:
    return MemoryAccessRequestResult(
        request_id="req_001",
        memory_id="mem_001",
        owner_agent_id="alice_agent",
        requester_agent_id="bob_agent",
        task_id="task_001",
        status=status,
        critic_recommendation=(
            critic_recommendation
        ),
    )


def _decision_result(
    *,
    approved: bool,
) -> MemoryAccessDecisionResult:
    return MemoryAccessDecisionResult(
        request_id="req_001",
        memory_id="mem_001",
        requester_agent_id="bob_agent",
        status=(
            "approved"
            if approved
            else "rejected"
        ),
        decision=(
            "approved"
            if approved
            else "rejected"
        ),
        access_granted=approved,
    )


def _state(
    **overrides: Any,
) -> dict[str, Any]:
    state = (
        build_initial_access_request_state(
            run_id="run_001",
            memory_id="mem_001",
            owner_agent_id="alice_agent",
            requester_agent_id="bob_agent",
            coordinator_agent_id=(
                "coordinator_agent"
            ),
            critic_agent_id="critic_agent",
            available_agent_ids=[
                "alice_agent",
                "bob_agent",
                "critic_agent",
                "coordinator_agent",
            ],
            request_reason=(
                "Bob requires this memory "
                "for the current task."
            ),
            task_id="task_001",
            memory_context={
                "summary": "Requested memory.",
            },
            policy_context={
                "task_scoped_access": True,
            },
            task_context={
                "question": "Current task",
            },
            risk_labels=[],
        )
    )
    state.update(overrides)
    return state


def _workflow(
    *,
    initial_decision: Any | None = None,
    final_decision: Any | None = None,
    critic_review: Any | None = None,
    gate_result: ReviewGateResult | None = None,
    has_access_values: list[bool] | None = None,
    config: GovernanceConfig | None = None,
) -> tuple[
    AccessRequestWorkflow,
    FakeCoordinator,
    FakeCritic,
    FakeReviewGate,
    FakeSharingGateway,
    AccessRequestDependencies,
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
        review=(
            critic_review
            if critic_review is not None
            else _review()
        )
    )
    gate = FakeReviewGate(
        result=(
            gate_result
            if gate_result is not None
            else ReviewGateResult.no_review()
        )
    )
    gateway = FakeSharingGateway(
        has_access_values=(
            list(has_access_values)
            if has_access_values is not None
            else [False, True]
        ),
        request_result=(
            _request_result()
        ),
        review_result=(
            _request_result(
                status="reviewed",
                critic_recommendation=(
                    "approve"
                ),
            )
        ),
        approve_result=(
            _decision_result(
                approved=True
            )
        ),
        direct_approve_result=(
            _decision_result(
                approved=True
            )
        ),
        reject_result=(
            _decision_result(
                approved=False
            )
        ),
    )

    dependencies = (
        AccessRequestDependencies.create(
            coordinator=coordinator,
            critic=critic,
            sharing_gateway=gateway,
            review_gate=gate,
            config=(
                config or GovernanceConfig()
            ),
        )
    )

    workflow = (
        build_access_request_workflow(
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


def test_build_graph_returns_builder_and_nodes() -> None:
    *_, dependencies = _workflow()

    builder, nodes = (
        build_access_request_graph(
            dependencies=dependencies
        )
    )

    assert builder is not None
    assert nodes.dependencies is dependencies
    assert ACCESS_REQUEST_NODE_NAMES == (
        "validate_access_request_input",
        "check_existing_access",
        "submit_access_request",
        "evaluate_access_request",
        "evaluate_request_review",
        "review_access_request",
        "record_critic_review",
        "finalise_reviewed_request",
        "accept_unreviewed_decision",
        "execute_access_decision",
    )


def test_build_graph_rejects_wrong_dependencies() -> None:
    with pytest.raises(
        TypeError,
        match="AccessRequestDependencies",
    ):
        build_access_request_graph(
            dependencies=object(),  # type: ignore[arg-type]
        )


def test_create_alias_builds_workflow() -> None:
    *_, dependencies = _workflow()

    workflow = (
        create_access_request_workflow(
            dependencies=dependencies
        )
    )

    assert isinstance(
        workflow,
        AccessRequestWorkflow,
    )
    assert workflow.nodes.dependencies is (
        dependencies
    )


# ---------------------------------------------------------------------------
# Complete workflow paths
# ---------------------------------------------------------------------------


def test_already_authorised_path_ends_without_request() -> None:
    (
        workflow,
        coordinator,
        critic,
        gate,
        gateway,
        _,
    ) = _workflow(
        has_access_values=[True]
    )

    result = workflow.invoke(
        _state()
    )

    assert result["status"] == (
        "already_authorised"
    )
    assert result[
        "request_skipped"
    ] is True
    assert result[
        "access_granted"
    ] is True

    assert coordinator.initial_calls == []
    assert coordinator.final_calls == []
    assert critic.calls == []
    assert gate.calls == []
    assert gateway.request_calls == []
    assert gateway.review_calls == []
    assert gateway.approve_calls == []
    assert (
        gateway.direct_approve_calls
        == []
    )
    assert gateway.reject_calls == []

    assert result["node_trace"] == [
        "validate_access_request_input",
        "check_existing_access",
    ]


def test_unreviewed_approval_uses_direct_approval() -> None:
    (
        workflow,
        coordinator,
        critic,
        gate,
        gateway,
        _,
    ) = _workflow(
        initial_decision=_decision(
            approved=True,
            review_required=False,
        ),
        gate_result=(
            ReviewGateResult.no_review()
        ),
        has_access_values=[
            False,
            True,
        ],
    )

    result = workflow.invoke(
        _state()
    )

    assert result["status"] == "executed"
    assert result[
        "decision_executed"
    ] is True
    assert result[
        "access_granted"
    ] is True
    assert result[
        "final_access_confirmed"
    ] is True

    assert len(
        coordinator.initial_calls
    ) == 1
    assert coordinator.final_calls == []
    assert critic.calls == []
    assert len(gate.calls) == 1
    assert gateway.review_calls == []
    assert gateway.approve_calls == []
    assert len(
        gateway.direct_approve_calls
    ) == 1
    assert gateway.reject_calls == []

    assert result["node_trace"] == [
        "validate_access_request_input",
        "check_existing_access",
        "submit_access_request",
        "evaluate_access_request",
        "evaluate_request_review",
        "accept_unreviewed_decision",
        "execute_access_decision",
    ]


def test_reviewed_approval_uses_governed_approval() -> None:
    gate_result = ReviewGateResult(
        review_required=True,
        reason_codes=(
            "request_low_confidence",
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
        initial_decision=_decision(
            approved=True,
            review_required=True,
            confidence=0.6,
        ),
        final_decision=_decision(
            approved=True,
            review_required=False,
            confidence=0.95,
            reason="Approved after review.",
        ),
        gate_result=gate_result,
        has_access_values=[
            False,
            True,
        ],
    )

    result = workflow.invoke(
        _state()
    )

    assert result["status"] == "executed"
    assert result[
        "review_required"
    ] is True
    assert result[
        "review_completed"
    ] is True
    assert result[
        "review_recorded"
    ] is True
    assert result[
        "access_granted"
    ] is True

    assert len(
        coordinator.initial_calls
    ) == 1
    assert len(
        coordinator.final_calls
    ) == 1
    assert len(critic.calls) == 1
    assert len(gate.calls) == 1
    assert len(gateway.review_calls) == 1
    assert len(gateway.approve_calls) == 1
    assert (
        gateway.direct_approve_calls
        == []
    )
    assert gateway.reject_calls == []

    assert result["node_trace"] == [
        "validate_access_request_input",
        "check_existing_access",
        "submit_access_request",
        "evaluate_access_request",
        "evaluate_request_review",
        "review_access_request",
        "record_critic_review",
        "finalise_reviewed_request",
        "execute_access_decision",
    ]


@pytest.mark.parametrize(
    (
        "reviewed",
        "expected_require_review",
    ),
    [
        (False, False),
        (True, True),
    ],
)
def test_rejection_path_executes_without_granting_access(
    reviewed: bool,
    expected_require_review: bool,
) -> None:
    gate_result = (
        ReviewGateResult(
            review_required=True,
            reason_codes=(
                "request_policy_violation",
            ),
            reasons=(
                "Policy review required.",
            ),
            triggered_risk_labels=(
                "policy_violation",
            ),
        )
        if reviewed
        else ReviewGateResult.no_review()
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
            approved=False,
            review_required=reviewed,
        ),
        final_decision=(
            _decision(
                approved=False,
                review_required=False,
                reason="Rejected after review.",
            )
            if reviewed
            else None
        ),
        critic_review=(
            _review(
                recommendation="reject"
            )
            if reviewed
            else None
        ),
        gate_result=gate_result,
        has_access_values=[
            False,
            False,
        ],
    )

    if reviewed:
        gateway.review_result = (
            _request_result(
                status="reviewed",
                critic_recommendation=(
                    "reject"
                ),
            )
        )

    result = workflow.invoke(
        _state()
    )

    assert result["status"] == "executed"
    assert result[
        "access_granted"
    ] is False
    assert result[
        "final_access_confirmed"
    ] is False

    assert len(gateway.reject_calls) == 1
    assert gateway.reject_calls[0][
        "require_critic_review"
    ] is expected_require_review

    if reviewed:
        assert len(
            coordinator.final_calls
        ) == 1
        assert len(critic.calls) == 1
        assert len(
            gateway.review_calls
        ) == 1
    else:
        assert coordinator.final_calls == []
        assert critic.calls == []
        assert gateway.review_calls == []


def test_async_invoke_executes_workflow() -> None:
    workflow, *_ = _workflow(
        has_access_values=[
            False,
            True,
        ]
    )

    result = asyncio.run(
        workflow.ainvoke(
            _state()
        )
    )

    assert result["status"] == "executed"
    assert result[
        "decision_executed"
    ] is True


# ---------------------------------------------------------------------------
# Fail-closed routing
# ---------------------------------------------------------------------------


def test_validation_failure_ends_before_gateway() -> None:
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
            requester_agent_id=(
                "unknown_agent"
            )
        )
    )

    assert result["status"] == "failed"
    assert result["failed_node"] == (
        "validate_access_request_input"
    )
    assert coordinator.initial_calls == []
    assert critic.calls == []
    assert gate.calls == []
    assert gateway.has_access_calls == []
    assert gateway.request_calls == []


def test_existing_access_check_failure_ends_before_submission() -> None:
    (
        workflow,
        coordinator,
        critic,
        gate,
        gateway,
        _,
    ) = _workflow()
    gateway.has_access_error = (
        RuntimeError(
            "permission store unavailable"
        )
    )

    result = workflow.invoke(
        _state()
    )

    assert result["status"] == "failed"
    assert result["failed_node"] == (
        "check_existing_access"
    )
    assert coordinator.initial_calls == []
    assert critic.calls == []
    assert gate.calls == []
    assert gateway.request_calls == []


def test_coordinator_failure_ends_before_review() -> None:
    (
        workflow,
        coordinator,
        critic,
        gate,
        gateway,
        _,
    ) = _workflow(
        has_access_values=[False]
    )
    coordinator.initial_error = (
        RuntimeError("LLM unavailable")
    )

    result = workflow.invoke(
        _state()
    )

    assert result["status"] == "failed"
    assert result["failed_node"] == (
        "evaluate_access_request"
    )
    assert len(
        coordinator.initial_calls
    ) == 1
    assert critic.calls == []
    assert gate.calls == []
    assert gateway.review_calls == []
    assert gateway.approve_calls == []
    assert gateway.reject_calls == []


def test_critic_failure_ends_before_review_recording() -> None:
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
        has_access_values=[False],
    )
    critic.error = RuntimeError(
        "Critic unavailable"
    )

    result = workflow.invoke(
        _state()
    )

    assert result["status"] == "failed"
    assert result["failed_node"] == (
        "review_access_request"
    )
    assert coordinator.final_calls == []
    assert len(critic.calls) == 1
    assert gateway.review_calls == []
    assert gateway.approve_calls == []
    assert gateway.reject_calls == []


def test_execution_failure_returns_failed_state() -> None:
    (
        workflow,
        _,
        _,
        _,
        gateway,
        _,
    ) = _workflow(
        has_access_values=[False]
    )
    gateway.direct_approve_error = (
        RuntimeError(
            "Persistence unavailable"
        )
    )

    result = workflow.invoke(
        _state()
    )

    assert result["status"] == "failed"
    assert result["failed_node"] == (
        "execute_access_decision"
    )
    assert result[
        "decision_executed"
    ] is False
    assert result[
        "access_granted"
    ] is False
    assert "Persistence unavailable" in (
        result["error_message"]
    )


def test_fail_closed_false_is_wrapped_by_workflow() -> None:
    (
        workflow,
        _,
        _,
        _,
        gateway,
        _,
    ) = _workflow(
        config=GovernanceConfig(
            fail_closed=False
        )
    )
    gateway.has_access_error = (
        RuntimeError(
            "permission store unavailable"
        )
    )

    with pytest.raises(
        AccessRequestWorkflowError,
        match="graph execution failed",
    ):
        workflow.invoke(
            _state()
        )


# ---------------------------------------------------------------------------
# Wrapper behaviour
# ---------------------------------------------------------------------------


def test_invoke_rejects_non_mapping_state() -> None:
    workflow, *_ = _workflow()

    with pytest.raises(
        AccessRequestWorkflowError,
        match="state must be a mapping",
    ):
        workflow.invoke(
            ["not", "a", "mapping"]  # type: ignore[arg-type]
        )


def test_invoke_does_not_mutate_input() -> None:
    workflow, *_ = _workflow(
        has_access_values=[
            False,
            True,
        ]
    )
    state = _state()
    original = deepcopy(state)

    result = workflow.invoke(
        state
    )

    assert state == original
    assert result is not state
    assert result["status"] == "executed"


def test_terminal_result_is_json_serialisable() -> None:
    workflow, *_ = _workflow(
        has_access_values=[
            False,
            True,
        ]
    )

    result = workflow.invoke(
        _state()
    )

    encoded = json.dumps(
        result,
        ensure_ascii=False,
    )

    assert '"status": "executed"' in (
        encoded
    )