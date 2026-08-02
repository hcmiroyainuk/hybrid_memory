from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import pytest

from src.llm.output_schemas import (
    MemoryAccessDecisionOutput,
    MemoryAccessReviewOutput,
)
from src.adapters.memory_access_policy_models import (
    AccessPolicyScope,
    MemoryAccessPolicyResult,
)
from src.memory.governance.config import (
    GovernanceConfig,
)
from src.memory.governance.dependencies import (
    PolicyAssignmentDependencies,
)
from src.memory.governance.policy_assignment_nodes import (
    PolicyAssignmentNodeError,
    PolicyAssignmentNodes,
    build_policy_assignment_nodes,
)
from src.memory.governance.policy_assignment_state import (
    PolicyAssignmentState,
    build_initial_policy_assignment_state,
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

        if self.final_decision is None:
            return self.initial_decision

        return self.final_decision


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
class FakePolicyReviewGate:
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
class FakeAccessPolicyGateway:
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
    memory_id: str = "mem_001",
    target_scope: str = "shared",
    allowed_agent_ids: list[str] | None = None,
    review_required: bool = False,
    reason: str = "Bob needs this memory.",
    confidence: float = 0.95,
) -> MemoryAccessDecisionOutput:
    if allowed_agent_ids is None:
        allowed_agent_ids = ["bob_agent"]

    return MemoryAccessDecisionOutput(
        memory_id=memory_id,
        target_scope=target_scope,
        allowed_agent_ids=(
            allowed_agent_ids
        ),
        review_required=review_required,
        reason=reason,
        confidence=confidence,
    )


def _review(
    *,
    memory_id: str = "mem_001",
    recommendation: str = "approve",
    suggested_scope: str = "shared",
    suggested_agent_ids: list[str] | None = None,
) -> MemoryAccessReviewOutput:
    if suggested_agent_ids is None:
        suggested_agent_ids = [
            "bob_agent"
        ]

    return MemoryAccessReviewOutput(
        memory_id=memory_id,
        recommendation=recommendation,
        policy_compliant=True,
        least_privilege_satisfied=True,
        risk_labels=[],
        suggested_scope=suggested_scope,
        suggested_agent_ids=(
            suggested_agent_ids
        ),
        reason="The proposed ACL is acceptable.",
        confidence=0.9,
    )


def _policy_result(
    *,
    scope: str = "shared",
    readable_by: tuple[str, ...] = (
        "alice_agent",
        "bob_agent",
    ),
    writable_by: tuple[str, ...] = (
        "alice_agent",
    ),
    changed: bool = True,
    memory_id: str = "mem_001",
    owner_agent_id: str = "alice_agent",
) -> MemoryAccessPolicyResult:
    return MemoryAccessPolicyResult(
        memory_id=memory_id,
        owner_agent_id=owner_agent_id,
        scope=scope,
        readable_by=readable_by,
        writable_by=writable_by,
        changed=changed,
        operation_id=(
            "op_001"
            if changed
            else None
        ),
    )


def _initial_state(
    **overrides: Any,
) -> PolicyAssignmentState:
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


def _merge_state(
    state: PolicyAssignmentState,
    update: dict[str, Any],
) -> PolicyAssignmentState:
    merged = deepcopy(state)

    for key, value in update.items():
        if (
            key
            in {
                "node_trace",
                "warnings",
                "errors",
            }
            and isinstance(value, list)
        ):
            merged[key] = [
                *merged.get(key, []),
                *value,
            ]
        else:
            merged[key] = deepcopy(value)

    return merged


def _nodes(
    *,
    initial_decision: Any | None = None,
    final_decision: Any | None = None,
    review: Any | None = None,
    gate_result: Any | None = None,
    gateway_result: Any | None = None,
    config: GovernanceConfig | None = None,
) -> tuple[
    PolicyAssignmentNodes,
    FakeCoordinator,
    FakeCritic,
    FakePolicyReviewGate,
    FakeAccessPolicyGateway,
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
            review
            if review is not None
            else _review()
        )
    )
    gate = FakePolicyReviewGate(
        result=(
            gate_result
            if gate_result is not None
            else ReviewGateResult.no_review()
        )
    )
    gateway = FakeAccessPolicyGateway(
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

    return (
        build_policy_assignment_nodes(
            dependencies
        ),
        coordinator,
        critic,
        gate,
        gateway,
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_validate_input_normalises_serialisable_state() -> None:
    nodes, *_ = _nodes()
    state = _initial_state(
        run_id=" ",
        memory_id=" mem_001 ",
        owner_agent_id=" alice_agent ",
        governance_actor_id=(
            " coordinator_agent "
        ),
        available_agent_ids=[
            " alice_agent ",
            "bob_agent",
            "bob_agent",
            "*",
            "critic_agent",
            "coordinator_agent",
        ],
        risk_labels=[
            " Sensitive_Content ",
            "sensitive_content",
        ],
    )

    update = (
        nodes
        .validate_policy_assignment_input(
            state
        )
    )

    assert update["status"] == "validated"
    assert update["run_id"] == (
        "policy:mem_001"
    )
    assert update["memory_id"] == "mem_001"
    assert update["owner_agent_id"] == (
        "alice_agent"
    )
    assert update[
        "governance_actor_id"
    ] == "coordinator_agent"
    assert update[
        "available_agent_ids"
    ] == [
        "alice_agent",
        "bob_agent",
        "critic_agent",
        "coordinator_agent",
    ]
    assert update["risk_labels"] == [
        "sensitive_content"
    ]
    assert update["node_trace"] == [
        "validate_policy_assignment_input"
    ]


def test_validate_input_fails_closed_for_non_serialisable_context() -> None:
    nodes, *_ = _nodes()
    state = _initial_state(
        memory_context={
            "not_json": object(),
        }
    )

    update = (
        nodes
        .validate_policy_assignment_input(
            state
        )
    )

    assert update["status"] == "failed"
    assert update["policy_applied"] is False
    assert update["failed_node"] == (
        "validate_policy_assignment_input"
    )
    assert update["error_type"] == (
        "ValueError"
    )
    assert "JSON serialisable" in (
        update["error_message"]
    )


def test_validate_input_fails_when_owner_is_not_registered() -> None:
    nodes, *_ = _nodes()
    state = _initial_state(
        owner_agent_id="unknown_owner",
    )

    update = (
        nodes
        .validate_policy_assignment_input(
            state
        )
    )

    assert update["status"] == "failed"
    assert "owner_agent_id must appear" in (
        update["error_message"]
    )


# ---------------------------------------------------------------------------
# Coordinator proposal
# ---------------------------------------------------------------------------


def test_assign_initial_policy_forwards_context_and_serialises_output() -> None:
    proposed = _decision(
        allowed_agent_ids=[
            "bob_agent",
        ]
    )
    nodes, coordinator, *_ = _nodes(
        initial_decision=proposed
    )
    state = _initial_state(
        status="validated"
    )

    update = nodes.assign_initial_policy(
        state
    )

    assert update["status"] == (
        "policy_proposed"
    )
    assert update["initial_decision"] == (
        proposed.model_dump(
            mode="json"
        )
    )
    assert update["node_trace"] == [
        "assign_initial_policy"
    ]

    assert len(
        coordinator.initial_calls
    ) == 1
    call = coordinator.initial_calls[0]
    assert call["memory_id"] == "mem_001"
    assert call["owner_agent_id"] == (
        "alice_agent"
    )
    assert call[
        "available_agent_ids"
    ] == state["available_agent_ids"]
    assert call["memory_context"] == (
        state["memory_context"]
    )
    assert call["policy_context"] == (
        state["policy_context"]
    )
    assert call["task_context"] == (
        state["task_context"]
    )


def test_assign_initial_policy_accepts_mapping_output() -> None:
    proposed = _decision()
    nodes, *_ = _nodes(
        initial_decision=(
            proposed.model_dump(
                mode="json"
            )
        )
    )

    update = nodes.assign_initial_policy(
        _initial_state(
            status="validated"
        )
    )

    assert update["status"] == (
        "policy_proposed"
    )
    assert update[
        "initial_decision"
    ]["memory_id"] == "mem_001"


def test_assign_initial_policy_fails_for_wrong_memory_id() -> None:
    nodes, coordinator, *_ = _nodes(
        initial_decision=_decision(
            memory_id="wrong_memory"
        )
    )

    update = nodes.assign_initial_policy(
        _initial_state(
            status="validated"
        )
    )

    assert len(
        coordinator.initial_calls
    ) == 1
    assert update["status"] == "failed"
    assert update["failed_node"] == (
        "assign_initial_policy"
    )
    assert "wrong memory" in (
        update["error_message"]
    )


def test_assign_initial_policy_fails_from_wrong_status() -> None:
    nodes, coordinator, *_ = _nodes()

    update = nodes.assign_initial_policy(
        _initial_state(
            status="initialised"
        )
    )

    assert coordinator.initial_calls == []
    assert update["status"] == "failed"
    assert "cannot run from status" in (
        update["error_message"]
    )


# ---------------------------------------------------------------------------
# Deterministic review gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    (
        "gate_result",
        "expected_status",
        "expected_required",
    ),
    [
        (
            ReviewGateResult.no_review(),
            "policy_proposed",
            False,
        ),
        (
            ReviewGateResult(
                review_required=True,
                reason_codes=(
                    "global_sharing",
                ),
                reasons=(
                    "Global sharing requires review.",
                ),
                triggered_risk_labels=(),
            ),
            "review_pending",
            True,
        ),
    ],
)
def test_evaluate_policy_review_routes_from_gate_result(
    gate_result: ReviewGateResult,
    expected_status: str,
    expected_required: bool,
) -> None:
    nodes, _, _, gate, _ = _nodes(
        gate_result=gate_result
    )
    decision = _decision()
    state = _initial_state(
        status="policy_proposed",
        initial_decision=(
            decision.model_dump(
                mode="json"
            )
        ),
        risk_labels=[
            "sensitive_content"
        ],
    )

    update = (
        nodes.evaluate_policy_review(
            state
        )
    )

    assert update["status"] == (
        expected_status
    )
    assert update[
        "review_required"
    ] is expected_required
    assert gate.calls[0][
        "decision"
    ] == decision
    assert gate.calls[0][
        "risk_labels"
    ] == [
        "sensitive_content"
    ]


def test_evaluate_policy_review_rejects_invalid_gate_result() -> None:
    nodes, *_ = _nodes(
        gate_result=True
    )
    state = _initial_state(
        status="policy_proposed",
        initial_decision=(
            _decision().model_dump(
                mode="json"
            )
        ),
    )

    update = (
        nodes.evaluate_policy_review(
            state
        )
    )

    assert update["status"] == "failed"
    assert "ReviewGateResult" in (
        update["error_message"]
    )


# ---------------------------------------------------------------------------
# Critic review
# ---------------------------------------------------------------------------


def test_review_initial_policy_calls_critic_and_serialises_review() -> None:
    review = _review()
    nodes, _, critic, *_ = _nodes(
        review=review
    )
    initial_decision = _decision(
        review_required=True
    )
    state = _initial_state(
        status="review_pending",
        review_required=True,
        initial_decision=(
            initial_decision.model_dump(
                mode="json"
            )
        ),
    )

    update = (
        nodes.review_initial_policy(
            state
        )
    )

    assert update["status"] == "reviewed"
    assert update[
        "review_completed"
    ] is True
    assert update["critic_review"] == (
        review.model_dump(
            mode="json"
        )
    )
    assert critic.calls[0][
        "proposed_decision"
    ] == initial_decision


def test_review_initial_policy_fails_for_wrong_memory_id() -> None:
    nodes, *_ = _nodes(
        review=_review(
            memory_id="wrong_memory"
        )
    )
    state = _initial_state(
        status="review_pending",
        review_required=True,
        initial_decision=(
            _decision(
                review_required=True
            ).model_dump(
                mode="json"
            )
        ),
    )

    update = (
        nodes.review_initial_policy(
            state
        )
    )

    assert update["status"] == "failed"
    assert "wrong memory" in (
        update["error_message"]
    )


def test_review_initial_policy_cannot_run_when_review_not_required() -> None:
    nodes, _, critic, *_ = _nodes()
    state = _initial_state(
        status="review_pending",
        review_required=False,
        initial_decision=(
            _decision().model_dump(
                mode="json"
            )
        ),
    )

    update = (
        nodes.review_initial_policy(
            state
        )
    )

    assert critic.calls == []
    assert update["status"] == "failed"
    assert "review_required=False" in (
        update["error_message"]
    )


# ---------------------------------------------------------------------------
# Finalisation
# ---------------------------------------------------------------------------


def test_finalise_reviewed_policy_calls_coordinator() -> None:
    final = _decision(
        allowed_agent_ids=[
            "bob_agent",
        ],
        review_required=False,
        reason="Approved after review.",
    )
    nodes, coordinator, *_ = _nodes(
        final_decision=final
    )
    initial = _decision(
        review_required=True
    )
    review = _review()
    state = _initial_state(
        status="reviewed",
        review_required=True,
        review_completed=True,
        initial_decision=(
            initial.model_dump(
                mode="json"
            )
        ),
        critic_review=(
            review.model_dump(
                mode="json"
            )
        ),
    )

    update = (
        nodes.finalise_reviewed_policy(
            state
        )
    )

    assert update["status"] == "finalised"
    assert update["final_decision"] == (
        final.model_dump(
            mode="json"
        )
    )
    assert coordinator.final_calls[0][
        "initial_decision"
    ] == initial
    assert coordinator.final_calls[0][
        "critic_review"
    ] == review


def test_finalise_reviewed_policy_clears_redundant_review_flag() -> None:
    final = _decision(
        review_required=True,
    )
    nodes, *_ = _nodes(
        final_decision=final
    )
    state = _initial_state(
        status="reviewed",
        review_completed=True,
        initial_decision=(
            _decision(
                review_required=True
            ).model_dump(
                mode="json"
            )
        ),
        critic_review=(
            _review().model_dump(
                mode="json"
            )
        ),
    )

    update = (
        nodes.finalise_reviewed_policy(
            state
        )
    )

    assert update["status"] == "finalised"
    assert update[
        "final_decision"
    ]["review_required"] is False
    assert len(update["warnings"]) == 1
    assert "already completed" in (
        update["warnings"][0]
    )


def test_finalise_reviewed_policy_fails_for_unknown_reader() -> None:
    final = _decision(
        allowed_agent_ids=[
            "unknown_agent"
        ]
    )
    nodes, *_ = _nodes(
        final_decision=final
    )
    state = _initial_state(
        status="reviewed",
        review_completed=True,
        initial_decision=(
            _decision(
                review_required=True
            ).model_dump(
                mode="json"
            )
        ),
        critic_review=(
            _review().model_dump(
                mode="json"
            )
        ),
    )

    update = (
        nodes.finalise_reviewed_policy(
            state
        )
    )

    assert update["status"] == "failed"
    assert "unknown Agent IDs" in (
        update["error_message"]
    )


def test_accept_unreviewed_policy_copies_initial_decision() -> None:
    initial = _decision(
        review_required=False
    )
    nodes, *_ = _nodes()
    state = _initial_state(
        status="policy_proposed",
        review_required=False,
        initial_decision=(
            initial.model_dump(
                mode="json"
            )
        ),
    )

    update = (
        nodes.accept_unreviewed_policy(
            state
        )
    )

    assert update["status"] == "finalised"
    assert update["final_decision"] == (
        initial.model_dump(
            mode="json"
        )
    )


def test_accept_unreviewed_policy_rejects_review_required_path() -> None:
    nodes, *_ = _nodes()
    state = _initial_state(
        status="policy_proposed",
        review_required=True,
        initial_decision=(
            _decision(
                review_required=True
            ).model_dump(
                mode="json"
            )
        ),
    )

    update = (
        nodes.accept_unreviewed_policy(
            state
        )
    )

    assert update["status"] == "failed"
    assert "requiring review" in (
        update["error_message"]
    )


# ---------------------------------------------------------------------------
# Gateway execution
# ---------------------------------------------------------------------------


def test_apply_targeted_shared_policy_removes_owner_from_gateway_readers() -> None:
    final = _decision(
        allowed_agent_ids=[
            "alice_agent",
            "bob_agent",
        ]
    )
    gateway_result = _policy_result()
    nodes, _, _, _, gateway = _nodes(
        gateway_result=gateway_result
    )
    state = _initial_state(
        status="finalised",
        final_decision=(
            final.model_dump(
                mode="json"
            )
        ),
    )

    update = nodes.apply_access_policy(
        state
    )

    assert update["status"] == "applied"
    assert update[
        "policy_applied"
    ] is True
    assert update[
        "policy_changed"
    ] is True
    assert update["policy_result"] == (
        gateway_result.model_dump(
            mode="json"
        )
    )

    call = gateway.calls[0]
    assert call[
        "governance_actor_id"
    ] == "coordinator_agent"
    assert call["memory_id"] == "mem_001"
    assert call["target_scope"] == (
        "shared"
    )
    assert call[
        "allowed_agent_ids"
    ] == ["bob_agent"]


def test_apply_private_policy_sends_empty_reader_list() -> None:
    final = _decision(
        target_scope="private",
        allowed_agent_ids=[],
        reason="Keep private.",
    )
    result = _policy_result(
        scope="private",
        readable_by=("alice_agent",),
        writable_by=("alice_agent",),
    )
    nodes, _, _, _, gateway = _nodes(
        gateway_result=result
    )
    state = _initial_state(
        status="finalised",
        final_decision=(
            final.model_dump(
                mode="json"
            )
        ),
    )

    update = nodes.apply_access_policy(
        state
    )

    assert update["status"] == "applied"
    assert gateway.calls[0][
        "allowed_agent_ids"
    ] == []


def test_apply_global_policy_preserves_wildcard() -> None:
    final = _decision(
        allowed_agent_ids=["*"],
    )
    result = _policy_result(
        readable_by=("*",),
    )
    nodes, _, _, _, gateway = _nodes(
        gateway_result=result
    )
    state = _initial_state(
        status="finalised",
        final_decision=(
            final.model_dump(
                mode="json"
            )
        ),
    )

    update = nodes.apply_access_policy(
        state
    )

    assert update["status"] == "applied"
    assert gateway.calls[0][
        "allowed_agent_ids"
    ] == ["*"]


def test_apply_policy_accepts_successful_no_op() -> None:
    final = _decision()
    result = _policy_result(
        changed=False
    )
    nodes, *_ = _nodes(
        gateway_result=result
    )
    state = _initial_state(
        status="finalised",
        final_decision=(
            final.model_dump(
                mode="json"
            )
        ),
    )

    update = nodes.apply_access_policy(
        state
    )

    assert update[
        "policy_applied"
    ] is True
    assert update[
        "policy_changed"
    ] is False


def test_apply_policy_fails_closed_for_mismatched_gateway_readers() -> None:
    final = _decision(
        allowed_agent_ids=[
            "bob_agent",
        ]
    )
    wrong_result = _policy_result(
        readable_by=(
            "alice_agent",
            "critic_agent",
        )
    )
    nodes, _, _, _, gateway = _nodes(
        gateway_result=wrong_result
    )
    state = _initial_state(
        status="finalised",
        final_decision=(
            final.model_dump(
                mode="json"
            )
        ),
    )

    update = nodes.apply_access_policy(
        state
    )

    assert len(gateway.calls) == 1
    assert update["status"] == "failed"
    assert update[
        "policy_applied"
    ] is False
    assert "readers do not match" in (
        update["error_message"]
    )


def test_apply_policy_fails_closed_when_gateway_raises() -> None:
    nodes, _, _, _, gateway = _nodes()
    gateway.error = RuntimeError(
        "persistence unavailable"
    )
    state = _initial_state(
        status="finalised",
        final_decision=(
            _decision().model_dump(
                mode="json"
            )
        ),
    )

    update = nodes.apply_access_policy(
        state
    )

    assert update["status"] == "failed"
    assert update["failed_node"] == (
        "apply_access_policy"
    )
    assert update[
        "policy_applied"
    ] is False
    assert "persistence unavailable" in (
        update["error_message"]
    )


# ---------------------------------------------------------------------------
# Error mode and route helpers
# ---------------------------------------------------------------------------


def test_fail_closed_false_raises_node_error() -> None:
    config = GovernanceConfig(
        fail_closed=False
    )
    nodes, coordinator, *_ = _nodes(
        config=config
    )
    coordinator.initial_error = RuntimeError(
        "LLM unavailable"
    )

    with pytest.raises(
        PolicyAssignmentNodeError,
        match="assign_initial_policy",
    ) as exc_info:
        nodes.assign_initial_policy(
            _initial_state(
                status="validated"
            )
        )

    assert isinstance(
        exc_info.value.cause,
        RuntimeError,
    )


@pytest.mark.parametrize(
    (
        "router_name",
        "state",
        "expected",
    ),
    [
        (
            "route_after_validation",
            {"status": "validated"},
            "assign",
        ),
        (
            "route_after_policy_proposal",
            {
                "status": (
                    "policy_proposed"
                )
            },
            "evaluate_review",
        ),
        (
            "route_after_review_evaluation",
            {
                "status": (
                    "review_pending"
                ),
                "review_required": True,
            },
            "review",
        ),
        (
            "route_after_review_evaluation",
            {
                "status": (
                    "policy_proposed"
                ),
                "review_required": False,
            },
            "accept",
        ),
        (
            "route_after_review",
            {"status": "reviewed"},
            "finalise",
        ),
        (
            "route_after_finalisation",
            {"status": "finalised"},
            "apply",
        ),
        (
            "route_after_finalisation",
            {"status": "failed"},
            "end",
        ),
    ],
)
def test_route_helpers(
    router_name: str,
    state: dict[str, Any],
    expected: str,
) -> None:
    nodes, *_ = _nodes()
    router = getattr(
        nodes,
        router_name,
    )

    assert router(state) == expected


def test_complete_unreviewed_node_sequence() -> None:
    nodes, coordinator, critic, gate, gateway = (
        _nodes(
            gate_result=(
                ReviewGateResult.no_review()
            )
        )
    )
    state = _initial_state()

    for node in (
        nodes.validate_policy_assignment_input,
        nodes.assign_initial_policy,
        nodes.evaluate_policy_review,
        nodes.accept_unreviewed_policy,
        nodes.apply_access_policy,
    ):
        state = _merge_state(
            state,
            node(state),
        )

    assert state["status"] == "applied"
    assert state[
        "policy_applied"
    ] is True
    assert len(
        coordinator.initial_calls
    ) == 1
    assert coordinator.final_calls == []
    assert critic.calls == []
    assert len(gate.calls) == 1
    assert len(gateway.calls) == 1
    assert state["node_trace"] == [
        "validate_policy_assignment_input",
        "assign_initial_policy",
        "evaluate_policy_review",
        "accept_unreviewed_policy",
        "apply_access_policy",
    ]


def test_complete_reviewed_node_sequence() -> None:
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
    initial = _decision(
        review_required=True,
        confidence=0.6,
    )
    final = _decision(
        review_required=False,
        confidence=0.95,
        reason="Approved after Critic review.",
    )
    nodes, coordinator, critic, gate, gateway = (
        _nodes(
            initial_decision=initial,
            final_decision=final,
            gate_result=gate_result,
        )
    )
    state = _initial_state()

    for node in (
        nodes.validate_policy_assignment_input,
        nodes.assign_initial_policy,
        nodes.evaluate_policy_review,
        nodes.review_initial_policy,
        nodes.finalise_reviewed_policy,
        nodes.apply_access_policy,
    ):
        state = _merge_state(
            state,
            node(state),
        )

    assert state["status"] == "applied"
    assert state[
        "review_required"
    ] is True
    assert state[
        "review_completed"
    ] is True
    assert len(
        coordinator.initial_calls
    ) == 1
    assert len(
        coordinator.final_calls
    ) == 1
    assert len(critic.calls) == 1
    assert len(gate.calls) == 1
    assert len(gateway.calls) == 1
    assert state["node_trace"] == [
        "validate_policy_assignment_input",
        "assign_initial_policy",
        "evaluate_policy_review",
        "review_initial_policy",
        "finalise_reviewed_policy",
        "apply_access_policy",
    ]