from __future__ import annotations

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
from src.memory.governance.access_request_nodes import (
    AccessRequestNodeError,
    AccessRequestNodes,
    build_access_request_nodes,
)
from src.memory.governance.access_request_state import (
    AccessRequestState,
    build_initial_access_request_state,
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
    submitted_request: Any = None
    reviewed_request: Any = None
    approved_result: Any = None
    direct_approved_result: Any = None
    rejected_result: Any = None

    has_access_error: Exception | None = None
    submit_error: Exception | None = None
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

        if self.submit_error is not None:
            raise self.submit_error

        return self.submitted_request

    def review_request(
        self,
        **kwargs: Any,
    ) -> Any:
        self.review_calls.append(
            deepcopy(kwargs)
        )

        if self.review_error is not None:
            raise self.review_error

        return self.reviewed_request

    def approve_request(
        self,
        **kwargs: Any,
    ) -> Any:
        self.approve_calls.append(
            deepcopy(kwargs)
        )

        if self.approve_error is not None:
            raise self.approve_error

        return self.approved_result

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

        return self.direct_approved_result

    def reject_request(
        self,
        **kwargs: Any,
    ) -> Any:
        self.reject_calls.append(
            deepcopy(kwargs)
        )

        if self.reject_error is not None:
            raise self.reject_error

        return self.rejected_result

    def get_request(
        self,
        **kwargs: Any,
    ) -> Any:
        self.get_calls.append(
            deepcopy(kwargs)
        )
        return self.submitted_request


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _decision(
    *,
    approved: bool = True,
    review_required: bool = False,
    confidence: float = 0.95,
    request_id: str = "req_001",
    memory_id: str = "mem_001",
    reason: str = "The request is justified.",
) -> MemoryAccessRequestDecisionOutput:
    return MemoryAccessRequestDecisionOutput(
        request_id=request_id,
        memory_id=memory_id,
        approved=approved,
        review_required=review_required,
        reason=reason,
        confidence=confidence,
    )


def _review(
    *,
    recommendation: str = "approve",
    request_id: str = "req_001",
    memory_id: str = "mem_001",
) -> MemoryAccessRequestReviewOutput:
    approved = (
        recommendation == "approve"
    )

    return MemoryAccessRequestReviewOutput(
        request_id=request_id,
        memory_id=memory_id,
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
    request_id: str = "req_001",
    memory_id: str = "mem_001",
    owner_agent_id: str = "alice_agent",
    requester_agent_id: str = "bob_agent",
    task_id: str | None = "task_001",
) -> MemoryAccessRequestResult:
    return MemoryAccessRequestResult(
        request_id=request_id,
        memory_id=memory_id,
        owner_agent_id=owner_agent_id,
        requester_agent_id=(
            requester_agent_id
        ),
        task_id=task_id,
        status=status,
        critic_recommendation=(
            critic_recommendation
        ),
    )


def _decision_result(
    *,
    approved: bool = True,
    request_id: str = "req_001",
    memory_id: str = "mem_001",
    requester_agent_id: str = "bob_agent",
) -> MemoryAccessDecisionResult:
    return MemoryAccessDecisionResult(
        request_id=request_id,
        memory_id=memory_id,
        requester_agent_id=(
            requester_agent_id
        ),
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


def _initial_state(
    **overrides: Any,
) -> AccessRequestState:
    state = (
        build_initial_access_request_state(
            run_id="run_001",
            memory_id="mem_001",
            owner_agent_id="alice_agent",
            requester_agent_id="bob_agent",
            coordinator_agent_id=(
                "coordinator_agent"
            ),
            critic_agent_id=(
                "critic_agent"
            ),
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


def _merge_state(
    state: AccessRequestState,
    update: dict[str, Any],
) -> AccessRequestState:
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
    config: GovernanceConfig | None = None,
) -> tuple[
    AccessRequestNodes,
    FakeCoordinator,
    FakeCritic,
    FakeReviewGate,
    FakeSharingGateway,
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
    gate = FakeReviewGate(
        result=(
            gate_result
            if gate_result is not None
            else ReviewGateResult.no_review()
        )
    )
    gateway = FakeSharingGateway(
        submitted_request=(
            _request_result()
        ),
        reviewed_request=(
            _request_result(
                status="reviewed",
                critic_recommendation=(
                    "approve"
                ),
            )
        ),
        approved_result=(
            _decision_result(
                approved=True
            )
        ),
        direct_approved_result=(
            _decision_result(
                approved=True
            )
        ),
        rejected_result=(
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

    return (
        build_access_request_nodes(
            dependencies
        ),
        coordinator,
        critic,
        gate,
        gateway,
    )


# ---------------------------------------------------------------------------
# Input validation and access check
# ---------------------------------------------------------------------------


def test_validate_input_normalises_state() -> None:
    nodes, *_ = _nodes()
    state = _initial_state(
        run_id=" ",
        memory_id=" mem_001 ",
        owner_agent_id=" alice_agent ",
        requester_agent_id=" bob_agent ",
        coordinator_agent_id=(
            " coordinator_agent "
        ),
        critic_agent_id=(
            " critic_agent "
        ),
        available_agent_ids=[
            "alice_agent",
            " bob_agent ",
            "bob_agent",
            "critic_agent",
            "coordinator_agent",
            "*",
        ],
        risk_labels=[
            " Sensitive_Content ",
            "sensitive_content",
        ],
    )

    update = (
        nodes
        .validate_access_request_input(
            state
        )
    )

    assert update["status"] == "validated"
    assert update["run_id"] == (
        "access:mem_001:bob_agent"
    )
    assert update["memory_id"] == "mem_001"
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
        "validate_access_request_input"
    ]


def test_validate_input_fails_for_non_serialisable_context() -> None:
    nodes, *_ = _nodes()

    update = (
        nodes
        .validate_access_request_input(
            _initial_state(
                memory_context={
                    "bad": object(),
                }
            )
        )
    )

    assert update["status"] == "failed"
    assert update["failed_node"] == (
        "validate_access_request_input"
    )
    assert "JSON serialisable" in (
        update["error_message"]
    )


def test_check_existing_access_ends_when_already_authorised() -> None:
    nodes, _, _, _, gateway = _nodes()
    gateway.has_access_values = [True]

    update = (
        nodes.check_existing_access(
            _initial_state(
                status="validated"
            )
        )
    )

    assert update["status"] == (
        "already_authorised"
    )
    assert update[
        "had_access_before"
    ] is True
    assert update[
        "request_skipped"
    ] is True
    assert update[
        "access_granted"
    ] is True


def test_check_existing_access_continues_when_access_absent() -> None:
    nodes, _, _, _, gateway = _nodes()
    gateway.has_access_values = [False]

    update = (
        nodes.check_existing_access(
            _initial_state(
                status="validated"
            )
        )
    )

    assert update["status"] == (
        "access_checked"
    )
    assert update[
        "had_access_before"
    ] is False
    assert update[
        "request_skipped"
    ] is False


# ---------------------------------------------------------------------------
# Request submission
# ---------------------------------------------------------------------------


def test_submit_request_calls_gateway_and_stores_dto() -> None:
    nodes, _, _, _, gateway = _nodes()
    state = _initial_state(
        status="access_checked",
        had_access_before=False,
    )

    update = (
        nodes.submit_access_request(
            state
        )
    )

    assert update["status"] == (
        "request_submitted"
    )
    assert update["request_id"] == (
        "req_001"
    )
    assert update[
        "request_submitted"
    ] is True
    assert update[
        "submitted_request"
    ]["status"] == "pending"

    assert gateway.request_calls == [
        {
            "requester_agent_id": (
                "bob_agent"
            ),
            "memory_id": "mem_001",
            "reason": (
                state["request_reason"]
            ),
            "task_id": "task_001",
        }
    ]


def test_submit_request_rejects_wrong_gateway_identity() -> None:
    nodes, _, _, _, gateway = _nodes()
    gateway.submitted_request = (
        _request_result(
            memory_id="wrong_memory"
        )
    )

    update = (
        nodes.submit_access_request(
            _initial_state(
                status="access_checked",
                had_access_before=False,
            )
        )
    )

    assert update["status"] == "failed"
    assert "wrong memory" in (
        update["error_message"]
    )


def test_submit_request_requires_absent_access_check() -> None:
    nodes, _, _, _, gateway = _nodes()

    update = (
        nodes.submit_access_request(
            _initial_state(
                status="access_checked",
                had_access_before=None,
            )
        )
    )

    assert gateway.request_calls == []
    assert update["status"] == "failed"
    assert "confirming that access is absent" in (
        update["error_message"]
    )


# ---------------------------------------------------------------------------
# Coordinator initial decision and review gate
# ---------------------------------------------------------------------------


def test_evaluate_request_forwards_context() -> None:
    decision = _decision()
    nodes, coordinator, *_ = _nodes(
        initial_decision=decision
    )
    state = _initial_state(
        status="request_submitted",
        request_id="req_001",
        request_submitted=True,
        submitted_request=(
            _request_result().model_dump(
                mode="json"
            )
        ),
    )

    update = (
        nodes.evaluate_access_request(
            state
        )
    )

    assert update["status"] == (
        "decision_proposed"
    )
    assert update[
        "initial_decision"
    ] == decision.model_dump(
        mode="json"
    )

    call = coordinator.initial_calls[0]
    assert call["request_id"] == (
        "req_001"
    )
    assert call["memory_id"] == (
        "mem_001"
    )
    assert call[
        "requester_agent_id"
    ] == "bob_agent"
    assert call["memory_context"] == (
        state["memory_context"]
    )


@pytest.mark.parametrize(
    (
        "gate_result",
        "expected_status",
        "expected_required",
    ),
    [
        (
            ReviewGateResult.no_review(),
            "decision_proposed",
            False,
        ),
        (
            ReviewGateResult(
                review_required=True,
                reason_codes=(
                    "request_low_confidence",
                ),
                reasons=(
                    "Low confidence.",
                ),
                triggered_risk_labels=(),
            ),
            "review_pending",
            True,
        ),
    ],
)
def test_evaluate_review_uses_gate_result(
    gate_result: ReviewGateResult,
    expected_status: str,
    expected_required: bool,
) -> None:
    nodes, _, _, gate, _ = _nodes(
        gate_result=gate_result
    )
    decision = _decision()
    state = _initial_state(
        status="decision_proposed",
        request_id="req_001",
        initial_decision=(
            decision.model_dump(
                mode="json"
            )
        ),
    )

    update = (
        nodes.evaluate_request_review(
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


# ---------------------------------------------------------------------------
# Critic review and persistence
# ---------------------------------------------------------------------------


def test_review_access_request_calls_critic() -> None:
    review = _review()
    nodes, _, critic, *_ = _nodes(
        review=review
    )
    decision = _decision(
        review_required=True
    )
    state = _initial_state(
        status="review_pending",
        request_id="req_001",
        review_required=True,
        initial_decision=(
            decision.model_dump(
                mode="json"
            )
        ),
    )

    update = (
        nodes.review_access_request(
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
    ] == decision


def test_review_access_request_rejects_wrong_request_id() -> None:
    nodes, *_ = _nodes(
        review=_review(
            request_id="wrong_request"
        )
    )

    update = (
        nodes.review_access_request(
            _initial_state(
                status="review_pending",
                request_id="req_001",
                review_required=True,
                initial_decision=(
                    _decision(
                        review_required=True
                    ).model_dump(
                        mode="json"
                    )
                ),
            )
        )
    )

    assert update["status"] == "failed"
    assert "wrong request" in (
        update["error_message"]
    )


def test_record_review_calls_gateway() -> None:
    review = _review()
    nodes, _, _, _, gateway = _nodes(
        review=review
    )
    state = _initial_state(
        status="reviewed",
        request_id="req_001",
        review_required=True,
        review_completed=True,
        critic_review=(
            review.model_dump(
                mode="json"
            )
        ),
    )

    update = (
        nodes.record_critic_review(
            state
        )
    )

    assert update["status"] == "reviewed"
    assert update[
        "review_recorded"
    ] is True
    assert update[
        "reviewed_request"
    ]["status"] == "reviewed"
    assert gateway.review_calls == [
        {
            "critic_agent_id": (
                "critic_agent"
            ),
            "request_id": "req_001",
            "recommendation": "approve",
            "comment": review.reason,
        }
    ]


def test_record_review_rejects_changed_recommendation() -> None:
    nodes, _, _, _, gateway = _nodes()
    gateway.reviewed_request = (
        _request_result(
            status="reviewed",
            critic_recommendation="reject",
        )
    )
    review = _review(
        recommendation="approve"
    )

    update = (
        nodes.record_critic_review(
            _initial_state(
                status="reviewed",
                request_id="req_001",
                review_required=True,
                review_completed=True,
                critic_review=(
                    review.model_dump(
                        mode="json"
                    )
                ),
            )
        )
    )

    assert update["status"] == "failed"
    assert "does not preserve" in (
        update["error_message"]
    )


# ---------------------------------------------------------------------------
# Finalisation
# ---------------------------------------------------------------------------


def test_finalise_reviewed_request_calls_coordinator() -> None:
    initial = _decision(
        review_required=True
    )
    review = _review()
    final = _decision(
        approved=True,
        review_required=False,
        reason="Approved after review.",
    )
    nodes, coordinator, *_ = _nodes(
        initial_decision=initial,
        final_decision=final,
        review=review,
    )
    state = _initial_state(
        status="reviewed",
        request_id="req_001",
        review_required=True,
        review_completed=True,
        review_recorded=True,
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
        nodes.finalise_reviewed_request(
            state
        )
    )

    assert update["status"] == "finalised"
    assert update[
        "final_decision"
    ] == final.model_dump(
        mode="json"
    )
    assert coordinator.final_calls[0][
        "critic_review"
    ] == review


def test_finalise_requires_recorded_review() -> None:
    nodes, coordinator, *_ = _nodes()

    update = (
        nodes.finalise_reviewed_request(
            _initial_state(
                status="reviewed",
                request_id="req_001",
                review_completed=True,
                review_recorded=False,
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
        )
    )

    assert coordinator.final_calls == []
    assert update["status"] == "failed"
    assert "must be recorded" in (
        update["error_message"]
    )


def test_finalise_clears_redundant_review_flag() -> None:
    final = _decision(
        review_required=True
    )
    nodes, *_ = _nodes(
        final_decision=final
    )

    update = (
        nodes.finalise_reviewed_request(
            _initial_state(
                status="reviewed",
                request_id="req_001",
                review_completed=True,
                review_recorded=True,
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
        )
    )

    assert update["status"] == "finalised"
    assert update[
        "final_decision"
    ]["review_required"] is False
    assert len(update["warnings"]) == 1


def test_accept_unreviewed_decision_copies_initial() -> None:
    decision = _decision(
        review_required=False
    )
    nodes, *_ = _nodes()

    update = (
        nodes.accept_unreviewed_decision(
            _initial_state(
                status="decision_proposed",
                request_id="req_001",
                review_required=False,
                initial_decision=(
                    decision.model_dump(
                        mode="json"
                    )
                ),
            )
        )
    )

    assert update["status"] == "finalised"
    assert update[
        "final_decision"
    ] == decision.model_dump(
        mode="json"
    )


def test_accept_unreviewed_rejects_review_flag() -> None:
    nodes, *_ = _nodes()

    update = (
        nodes.accept_unreviewed_decision(
            _initial_state(
                status="decision_proposed",
                request_id="req_001",
                review_required=True,
                initial_decision=(
                    _decision(
                        review_required=True
                    ).model_dump(
                        mode="json"
                    )
                ),
            )
        )
    )

    assert update["status"] == "failed"
    assert "requiring review" in (
        update["error_message"]
    )


# ---------------------------------------------------------------------------
# Execute final decision
# ---------------------------------------------------------------------------


def test_execute_reviewed_approval_uses_governed_approve() -> None:
    decision = _decision(
        approved=True
    )
    nodes, _, _, _, gateway = _nodes()
    gateway.has_access_values = [True]
    state = _initial_state(
        status="finalised",
        request_id="req_001",
        review_completed=True,
        review_recorded=True,
        final_decision=(
            decision.model_dump(
                mode="json"
            )
        ),
    )

    update = (
        nodes.execute_access_decision(
            state
        )
    )

    assert update["status"] == "executed"
    assert update[
        "decision_executed"
    ] is True
    assert update[
        "access_granted"
    ] is True
    assert len(gateway.approve_calls) == 1
    assert (
        gateway.direct_approve_calls
        == []
    )
    assert gateway.reject_calls == []


def test_execute_unreviewed_approval_uses_direct_approve() -> None:
    decision = _decision(
        approved=True
    )
    nodes, _, _, _, gateway = _nodes()
    gateway.has_access_values = [True]
    state = _initial_state(
        status="finalised",
        request_id="req_001",
        review_completed=False,
        review_recorded=False,
        final_decision=(
            decision.model_dump(
                mode="json"
            )
        ),
    )

    update = (
        nodes.execute_access_decision(
            state
        )
    )

    assert update["status"] == "executed"
    assert len(
        gateway.direct_approve_calls
    ) == 1
    assert gateway.approve_calls == []


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
def test_execute_rejection_uses_correct_review_requirement(
    reviewed: bool,
    expected_require_review: bool,
) -> None:
    decision = _decision(
        approved=False
    )
    nodes, _, _, _, gateway = _nodes()
    gateway.has_access_values = [False]
    state = _initial_state(
        status="finalised",
        request_id="req_001",
        review_completed=reviewed,
        review_recorded=reviewed,
        final_decision=(
            decision.model_dump(
                mode="json"
            )
        ),
    )

    update = (
        nodes.execute_access_decision(
            state
        )
    )

    assert update["status"] == "executed"
    assert update[
        "access_granted"
    ] is False
    assert gateway.reject_calls[0][
        "require_critic_review"
    ] is expected_require_review


def test_execute_fails_for_post_decision_access_mismatch() -> None:
    decision = _decision(
        approved=True
    )
    nodes, _, _, _, gateway = _nodes()
    gateway.has_access_values = [False]

    update = (
        nodes.execute_access_decision(
            _initial_state(
                status="finalised",
                request_id="req_001",
                final_decision=(
                    decision.model_dump(
                        mode="json"
                    )
                ),
            )
        )
    )

    assert update["status"] == "failed"
    assert update[
        "decision_executed"
    ] is False
    assert "verification does not match" in (
        update["error_message"]
    )


def test_execute_fails_for_wrong_gateway_decision() -> None:
    decision = _decision(
        approved=True
    )
    nodes, _, _, _, gateway = _nodes()
    gateway.direct_approved_result = (
        _decision_result(
            approved=False
        )
    )
    gateway.has_access_values = [False]

    update = (
        nodes.execute_access_decision(
            _initial_state(
                status="finalised",
                request_id="req_001",
                final_decision=(
                    decision.model_dump(
                        mode="json"
                    )
                ),
            )
        )
    )

    assert update["status"] == "failed"
    assert "does not match" in (
        update["error_message"]
    )


# ---------------------------------------------------------------------------
# Fail-closed, routing, and complete paths
# ---------------------------------------------------------------------------


def test_fail_closed_false_raises_node_error() -> None:
    config = GovernanceConfig(
        fail_closed=False
    )
    nodes, _, _, _, gateway = _nodes(
        config=config
    )
    gateway.has_access_error = (
        RuntimeError(
            "permission store unavailable"
        )
    )

    with pytest.raises(
        AccessRequestNodeError,
        match="check_existing_access",
    ) as exc_info:
        nodes.check_existing_access(
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
            "check_access",
        ),
        (
            "route_after_access_check",
            {
                "status": (
                    "already_authorised"
                )
            },
            "end",
        ),
        (
            "route_after_access_check",
            {
                "status": (
                    "access_checked"
                )
            },
            "submit",
        ),
        (
            "route_after_submission",
            {
                "status": (
                    "request_submitted"
                )
            },
            "decide",
        ),
        (
            "route_after_initial_decision",
            {
                "status": (
                    "decision_proposed"
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
                    "decision_proposed"
                ),
                "review_required": False,
            },
            "accept",
        ),
        (
            "route_after_critic_review",
            {"status": "reviewed"},
            "record_review",
        ),
        (
            "route_after_review_recording",
            {"status": "reviewed"},
            "finalise",
        ),
        (
            "route_after_finalisation",
            {"status": "finalised"},
            "execute",
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


def test_complete_already_authorised_path() -> None:
    nodes, coordinator, critic, gate, gateway = (
        _nodes()
    )
    gateway.has_access_values = [True]
    state = _initial_state()

    for node in (
        nodes.validate_access_request_input,
        nodes.check_existing_access,
    ):
        state = _merge_state(
            state,
            node(state),
        )

    assert state["status"] == (
        "already_authorised"
    )
    assert state[
        "request_skipped"
    ] is True
    assert coordinator.initial_calls == []
    assert critic.calls == []
    assert gate.calls == []
    assert gateway.request_calls == []


def test_complete_unreviewed_approval_path() -> None:
    nodes, coordinator, critic, gate, gateway = (
        _nodes(
            gate_result=(
                ReviewGateResult.no_review()
            )
        )
    )
    gateway.has_access_values = [
        False,
        True,
    ]
    state = _initial_state()

    for node in (
        nodes.validate_access_request_input,
        nodes.check_existing_access,
        nodes.submit_access_request,
        nodes.evaluate_access_request,
        nodes.evaluate_request_review,
        nodes.accept_unreviewed_decision,
        nodes.execute_access_decision,
    ):
        state = _merge_state(
            state,
            node(state),
        )

    assert state["status"] == "executed"
    assert state[
        "access_granted"
    ] is True
    assert len(
        coordinator.initial_calls
    ) == 1
    assert coordinator.final_calls == []
    assert critic.calls == []
    assert len(gate.calls) == 1
    assert len(
        gateway.direct_approve_calls
    ) == 1
    assert state["node_trace"] == [
        "validate_access_request_input",
        "check_existing_access",
        "submit_access_request",
        "evaluate_access_request",
        "evaluate_request_review",
        "accept_unreviewed_decision",
        "execute_access_decision",
    ]


def test_complete_reviewed_approval_path() -> None:
    initial = _decision(
        approved=True,
        review_required=True,
        confidence=0.6,
    )
    final = _decision(
        approved=True,
        review_required=False,
        confidence=0.95,
    )
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
    nodes, coordinator, critic, gate, gateway = (
        _nodes(
            initial_decision=initial,
            final_decision=final,
            gate_result=gate_result,
        )
    )
    gateway.has_access_values = [
        False,
        True,
    ]
    state = _initial_state()

    for node in (
        nodes.validate_access_request_input,
        nodes.check_existing_access,
        nodes.submit_access_request,
        nodes.evaluate_access_request,
        nodes.evaluate_request_review,
        nodes.review_access_request,
        nodes.record_critic_review,
        nodes.finalise_reviewed_request,
        nodes.execute_access_decision,
    ):
        state = _merge_state(
            state,
            node(state),
        )

    assert state["status"] == "executed"
    assert state[
        "review_completed"
    ] is True
    assert state[
        "review_recorded"
    ] is True
    assert state[
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
    assert state["node_trace"] == [
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