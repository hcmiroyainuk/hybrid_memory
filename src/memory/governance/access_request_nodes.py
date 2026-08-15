from __future__ import annotations

"""
LangGraph nodes for the generic runtime memory-access request subgraph.

The nodes coordinate access checking, request submission, Coordinator and
Critic decisions, and final Gateway execution. They do not import concrete
PromotionService, persistence entities, experiment prompts, or an LLM client.

Every public node returns a partial ``AccessRequestState`` update.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ValidationError

from src.llm.output_schemas import (
    MemoryAccessRequestDecisionOutput,
    MemoryAccessRequestReviewOutput,
)
from src.adapters.memory_sharing_models import (
    MemoryAccessDecision,
    MemoryAccessDecisionResult,
    MemoryAccessRequestResult,
    MemoryAccessStatus,
)

from .access_request_state import (
    AccessRequestState,
    validate_access_request_identity,
)
from .dependencies import AccessRequestDependencies
from .review_gate import ReviewGateResult


AccessRequestRoute = Literal[
    "check_access",
    "submit",
    "decide",
    "evaluate_review",
    "review",
    "record_review",
    "finalise",
    "accept",
    "execute",
    "end",
]

T = TypeVar("T")


class AccessRequestNodeError(RuntimeError):
    """
    Raised when an access-request node cannot complete safely.
    """

    def __init__(
        self,
        node_name: str,
        message: str,
        *,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(
            f"{node_name}: {message}"
        )
        self.node_name = node_name
        self.cause = cause


@dataclass(
    frozen=True,
    slots=True,
)
class AccessRequestNodes:
    """
    Node collection for the runtime cross-Agent access-request subgraph.

    A single instance is created with ``AccessRequestDependencies`` and its
    bound methods are registered as LangGraph nodes.
    """

    dependencies: AccessRequestDependencies

    # ------------------------------------------------------------------
    # 1. Input validation
    # ------------------------------------------------------------------

    def validate_access_request_input(
        self,
        state: AccessRequestState,
    ) -> dict[str, Any]:
        """
        Validate and normalise stable request identity and context.
        """
        node_name = (
            "validate_access_request_input"
        )

        def operation() -> dict[str, Any]:
            validate_access_request_identity(
                state
            )

            memory_id = self._required_text(
                state.get("memory_id"),
                "memory_id",
            )
            owner_agent_id = (
                self._required_text(
                    state.get(
                        "owner_agent_id"
                    ),
                    "owner_agent_id",
                )
            )
            requester_agent_id = (
                self._required_text(
                    state.get(
                        "requester_agent_id"
                    ),
                    "requester_agent_id",
                )
            )
            coordinator_agent_id = (
                self._required_text(
                    state.get(
                        "coordinator_agent_id"
                    ),
                    "coordinator_agent_id",
                )
            )
            critic_agent_id = (
                self._required_text(
                    state.get(
                        "critic_agent_id"
                    ),
                    "critic_agent_id",
                )
            )
            request_reason = (
                self._required_text(
                    state.get(
                        "request_reason"
                    ),
                    "request_reason",
                )
            )

            available_agent_ids = [
                agent_id
                for agent_id in (
                    self._clean_string_list(
                        state.get(
                            "available_agent_ids",
                            [],
                        )
                    )
                )
                if agent_id != "*"
            ]

            memory_context = (
                self._mapping_copy(
                    state.get(
                        "memory_context",
                        {},
                    ),
                    "memory_context",
                )
            )
            policy_context = (
                self._mapping_copy(
                    state.get(
                        "policy_context",
                        {},
                    ),
                    "policy_context",
                )
            )
            task_context = (
                self._mapping_copy(
                    state.get(
                        "task_context",
                        {},
                    ),
                    "task_context",
                )
            )

            risk_labels: list[str] = []

            for label in (
                self._clean_string_list(
                    state.get(
                        "risk_labels",
                        [],
                    )
                )
            ):
                normalised_label = (
                    label.lower()
                )

                if (
                    normalised_label
                    not in risk_labels
                ):
                    risk_labels.append(
                        normalised_label
                    )

            self._assert_json_serialisable(
                {
                    "memory_context": (
                        memory_context
                    ),
                    "policy_context": (
                        policy_context
                    ),
                    "task_context": (
                        task_context
                    ),
                    "risk_labels": risk_labels,
                }
            )

            task_id = self._optional_text(
                state.get("task_id")
            )
            run_id = (
                self._optional_text(
                    state.get("run_id")
                )
                or (
                    f"access:{memory_id}:"
                    f"{requester_agent_id}"
                )
            )

            return {
                "run_id": run_id,
                "memory_id": memory_id,
                "owner_agent_id": (
                    owner_agent_id
                ),
                "requester_agent_id": (
                    requester_agent_id
                ),
                "coordinator_agent_id": (
                    coordinator_agent_id
                ),
                "critic_agent_id": (
                    critic_agent_id
                ),
                "available_agent_ids": (
                    available_agent_ids
                ),
                "request_reason": (
                    request_reason
                ),
                "task_id": task_id,
                "memory_context": (
                    memory_context
                ),
                "policy_context": (
                    policy_context
                ),
                "task_context": (
                    task_context
                ),
                "risk_labels": risk_labels,
                "status": "validated",
            }

        return self._run_node(
            node_name,
            operation,
        )

    # ------------------------------------------------------------------
    # 2. Existing-access check
    # ------------------------------------------------------------------

    def check_existing_access(
        self,
        state: AccessRequestState,
    ) -> dict[str, Any]:
        """
        Avoid creating a redundant request when access already exists.
        """
        node_name = "check_existing_access"

        def operation() -> dict[str, Any]:
            self._require_status(
                state,
                {
                    "validated",
                },
                node_name,
            )

            has_access = bool(
                self.dependencies
                .sharing_gateway
                .has_access(
                    agent_id=(
                        self._state_text(
                            state,
                            "requester_agent_id",
                        )
                    ),
                    memory_id=(
                        self._state_text(
                            state,
                            "memory_id",
                        )
                    ),
                )
            )

            if has_access:
                return {
                    "had_access_before": True,
                    "request_skipped": True,
                    "access_granted": True,
                    "final_access_confirmed": (
                        True
                    ),
                    "status": (
                        "already_authorised"
                    ),
                }

            return {
                "had_access_before": False,
                "request_skipped": False,
                "status": "access_checked",
            }

        return self._run_node(
            node_name,
            operation,
        )

    # ------------------------------------------------------------------
    # 3. Submit the access request
    # ------------------------------------------------------------------

    def submit_access_request(
        self,
        state: AccessRequestState,
    ) -> dict[str, Any]:
        """
        Submit one pending request through MemorySharingGateway.
        """
        node_name = "submit_access_request"

        def operation() -> dict[str, Any]:
            self._require_status(
                state,
                {
                    "access_checked",
                },
                node_name,
            )

            if (
                state.get(
                    "had_access_before"
                )
                is not False
            ):
                raise ValueError(
                    "A request may be submitted "
                    "only after confirming that "
                    "access is absent."
                )

            raw_request = (
                self.dependencies
                .sharing_gateway
                .request_access(
                    requester_agent_id=(
                        self._state_text(
                            state,
                            "requester_agent_id",
                        )
                    ),
                    memory_id=(
                        self._state_text(
                            state,
                            "memory_id",
                        )
                    ),
                    reason=(
                        self._state_text(
                            state,
                            "request_reason",
                        )
                    ),
                    task_id=(
                        self._optional_text(
                            state.get(
                                "task_id"
                            )
                        )
                    ),
                )
            )

            request = self._model_validate(
                MemoryAccessRequestResult,
                raw_request,
                "Submitted access request",
            )

            self._validate_request_result(
                request,
                state=state,
                expected_status=(
                    MemoryAccessStatus.PENDING
                ),
            )

            return {
                "request_id": (
                    request.request_id
                ),
                "submitted_request": (
                    request.model_dump(
                        mode="json"
                    )
                ),
                "request_submitted": True,
                "status": "request_submitted",
            }

        return self._run_node(
            node_name,
            operation,
        )

    # ------------------------------------------------------------------
    # 4. Coordinator initial decision
    # ------------------------------------------------------------------

    def evaluate_access_request(
        self,
        state: AccessRequestState,
    ) -> dict[str, Any]:
        """
        Ask the Coordinator for an initial approve/reject decision.
        """
        node_name = "evaluate_access_request"

        def operation() -> dict[str, Any]:
            self._require_status(
                state,
                {
                    "request_submitted",
                },
                node_name,
            )

            if not bool(
                state.get(
                    "request_submitted"
                )
            ):
                raise ValueError(
                    "The access request has not "
                    "been submitted."
                )

            raw_decision = (
                self.dependencies
                .coordinator
                .evaluate_access_request(
                    request_id=(
                        self._state_text(
                            state,
                            "request_id",
                        )
                    ),
                    memory_id=(
                        self._state_text(
                            state,
                            "memory_id",
                        )
                    ),
                    owner_agent_id=(
                        self._state_text(
                            state,
                            "owner_agent_id",
                        )
                    ),
                    requester_agent_id=(
                        self._state_text(
                            state,
                            "requester_agent_id",
                        )
                    ),
                    request_reason=(
                        self._state_text(
                            state,
                            "request_reason",
                        )
                    ),
                    task_id=(
                        self._optional_text(
                            state.get(
                                "task_id"
                            )
                        )
                    ),
                    available_agent_ids=(
                        self._state_ids(
                            state,
                            "available_agent_ids",
                        )
                    ),
                    memory_context=(
                        self._state_mapping(
                            state,
                            "memory_context",
                        )
                    ),
                    policy_context=(
                        self._state_mapping(
                            state,
                            "policy_context",
                        )
                    ),
                    task_context=(
                        self._state_mapping(
                            state,
                            "task_context",
                        )
                    ),
                )
            )

            decision = (
                self._validate_decision(
                    raw_decision,
                    state=state,
                    final=False,
                )
            )

            return {
                "initial_decision": (
                    decision.model_dump(
                        mode="json"
                    )
                ),
                "status": (
                    "decision_proposed"
                ),
            }

        return self._run_node(
            node_name,
            operation,
        )

    # ------------------------------------------------------------------
    # 5. Deterministic review gate
    # ------------------------------------------------------------------

    def evaluate_request_review(
        self,
        state: AccessRequestState,
    ) -> dict[str, Any]:
        """
        Decide deterministically whether Critic review is mandatory.
        """
        node_name = "evaluate_request_review"

        def operation() -> dict[str, Any]:
            self._require_status(
                state,
                {
                    "decision_proposed",
                },
                node_name,
            )

            decision = self._state_decision(
                state,
                "initial_decision",
            )

            raw_result = (
                self.dependencies
                .review_gate
                .requires_access_request_review(
                    decision=decision,
                    requester_agent_id=(
                        self._state_text(
                            state,
                            "requester_agent_id",
                        )
                    ),
                    owner_agent_id=(
                        self._state_text(
                            state,
                            "owner_agent_id",
                        )
                    ),
                    available_agent_ids=(
                        self._state_ids(
                            state,
                            "available_agent_ids",
                        )
                    ),
                    request_reason=(
                        self._state_text(
                            state,
                            "request_reason",
                        )
                    ),
                    risk_labels=(
                        self._state_ids(
                            state,
                            "risk_labels",
                            allow_empty=True,
                        )
                    ),
                )
            )

            result = (
                self._validate_gate_result(
                    raw_result
                )
            )

            update = (
                result.as_state_update()
            )
            update["status"] = (
                "review_pending"
                if result.review_required
                else "decision_proposed"
            )

            return update

        return self._run_node(
            node_name,
            operation,
        )

    # ------------------------------------------------------------------
    # 6A. Critic review path
    # ------------------------------------------------------------------

    def review_access_request(
        self,
        state: AccessRequestState,
    ) -> dict[str, Any]:
        """
        Ask the Critic for an advisory request review.
        """
        node_name = "review_access_request"

        def operation() -> dict[str, Any]:
            self._require_status(
                state,
                {
                    "review_pending",
                },
                node_name,
            )

            if not bool(
                state.get("review_required")
            ):
                raise ValueError(
                    "review_access_request cannot "
                    "run when review_required=False."
                )

            initial_decision = (
                self._state_decision(
                    state,
                    "initial_decision",
                )
            )

            raw_review = (
                self.dependencies.critic
                .review_access_request(
                    proposed_decision=(
                        initial_decision
                    ),
                    request_id=(
                        self._state_text(
                            state,
                            "request_id",
                        )
                    ),
                    memory_id=(
                        self._state_text(
                            state,
                            "memory_id",
                        )
                    ),
                    owner_agent_id=(
                        self._state_text(
                            state,
                            "owner_agent_id",
                        )
                    ),
                    requester_agent_id=(
                        self._state_text(
                            state,
                            "requester_agent_id",
                        )
                    ),
                    request_reason=(
                        self._state_text(
                            state,
                            "request_reason",
                        )
                    ),
                    task_id=(
                        self._optional_text(
                            state.get(
                                "task_id"
                            )
                        )
                    ),
                    available_agent_ids=(
                        self._state_ids(
                            state,
                            "available_agent_ids",
                        )
                    ),
                    memory_context=(
                        self._state_mapping(
                            state,
                            "memory_context",
                        )
                    ),
                    policy_context=(
                        self._state_mapping(
                            state,
                            "policy_context",
                        )
                    ),
                    task_context=(
                        self._state_mapping(
                            state,
                            "task_context",
                        )
                    ),
                )
            )

            review = self._validate_review(
                raw_review,
                state=state,
            )

            return {
                "critic_review": (
                    review.model_dump(
                        mode="json"
                    )
                ),
                "review_completed": True,
                "status": "reviewed",
            }

        return self._run_node(
            node_name,
            operation,
        )

    def record_critic_review(
        self,
        state: AccessRequestState,
    ) -> dict[str, Any]:
        """
        Persist the Critic recommendation through MemorySharingGateway.
        """
        node_name = "record_critic_review"

        def operation() -> dict[str, Any]:
            self._require_status(
                state,
                {
                    "reviewed",
                },
                node_name,
            )

            if not bool(
                state.get(
                    "review_completed"
                )
            ):
                raise ValueError(
                    "A completed Critic review "
                    "is required."
                )

            review = self._state_review(
                state,
                "critic_review",
            )

            raw_request = (
                self.dependencies
                .sharing_gateway
                .review_request(
                    critic_agent_id=(
                        self._state_text(
                            state,
                            "critic_agent_id",
                        )
                    ),
                    request_id=(
                        self._state_text(
                            state,
                            "request_id",
                        )
                    ),
                    recommendation=(
                        review.recommendation
                    ),
                    comment=review.reason,
                )
            )

            request = self._model_validate(
                MemoryAccessRequestResult,
                raw_request,
                "Reviewed access request",
            )

            self._validate_request_result(
                request,
                state=state,
                expected_status=(
                    MemoryAccessStatus.REVIEWED
                ),
            )

            if (
                request.critic_recommendation
                != review.recommendation
            ):
                raise ValueError(
                    "Gateway review result does "
                    "not preserve the Critic "
                    "recommendation."
                )

            return {
                "reviewed_request": (
                    request.model_dump(
                        mode="json"
                    )
                ),
                "review_recorded": True,
                "status": "reviewed",
            }

        return self._run_node(
            node_name,
            operation,
        )

    def finalise_reviewed_request(
        self,
        state: AccessRequestState,
    ) -> dict[str, Any]:
        """
        Ask the Coordinator for the final decision after recorded review.
        """
        node_name = (
            "finalise_reviewed_request"
        )

        def operation() -> dict[str, Any]:
            self._require_status(
                state,
                {
                    "reviewed",
                },
                node_name,
            )

            if not bool(
                state.get(
                    "review_completed"
                )
            ):
                raise ValueError(
                    "A completed Critic review "
                    "is required."
                )

            if not bool(
                state.get(
                    "review_recorded"
                )
            ):
                raise ValueError(
                    "The Critic review must be "
                    "recorded before finalisation."
                )

            initial_decision = (
                self._state_decision(
                    state,
                    "initial_decision",
                )
            )
            critic_review = (
                self._state_review(
                    state,
                    "critic_review",
                )
            )

            raw_final_decision = (
                self.dependencies
                .coordinator
                .finalise_access_request(
                    initial_decision=(
                        initial_decision
                    ),
                    critic_review=(
                        critic_review
                    ),
                    request_id=(
                        self._state_text(
                            state,
                            "request_id",
                        )
                    ),
                    memory_id=(
                        self._state_text(
                            state,
                            "memory_id",
                        )
                    ),
                    owner_agent_id=(
                        self._state_text(
                            state,
                            "owner_agent_id",
                        )
                    ),
                    requester_agent_id=(
                        self._state_text(
                            state,
                            "requester_agent_id",
                        )
                    ),
                    request_reason=(
                        self._state_text(
                            state,
                            "request_reason",
                        )
                    ),
                    task_id=(
                        self._optional_text(
                            state.get(
                                "task_id"
                            )
                        )
                    ),
                    available_agent_ids=(
                        self._state_ids(
                            state,
                            "available_agent_ids",
                        )
                    ),
                    memory_context=(
                        self._state_mapping(
                            state,
                            "memory_context",
                        )
                    ),
                    policy_context=(
                        self._state_mapping(
                            state,
                            "policy_context",
                        )
                    ),
                    task_context=(
                        self._state_mapping(
                            state,
                            "task_context",
                        )
                    ),
                )
            )

            decision = (
                self._validate_decision(
                    raw_final_decision,
                    state=state,
                    final=True,
                )
            )

            warnings: list[str] = []

            if decision.review_required:
                decision = decision.model_copy(
                    update={
                        "review_required": False,
                    }
                )
                warnings.append(
                    "The final Coordinator decision "
                    "still requested review; the flag "
                    "was cleared because review has "
                    "already completed."
                )

            return self._with_warnings(
                {
                    "final_decision": (
                        decision.model_dump(
                            mode="json"
                        )
                    ),
                    "status": "finalised",
                },
                warnings,
            )

        return self._run_node(
            node_name,
            operation,
        )

    # ------------------------------------------------------------------
    # 6B. No-review path
    # ------------------------------------------------------------------

    def accept_unreviewed_decision(
        self,
        state: AccessRequestState,
    ) -> dict[str, Any]:
        """
        Promote the initial decision to final when review is unnecessary.
        """
        node_name = (
            "accept_unreviewed_decision"
        )

        def operation() -> dict[str, Any]:
            self._require_status(
                state,
                {
                    "decision_proposed",
                },
                node_name,
            )

            if bool(
                state.get("review_required")
            ):
                raise ValueError(
                    "A request requiring review "
                    "cannot use the unreviewed path."
                )

            decision = (
                self._validate_decision(
                    self._state_decision(
                        state,
                        "initial_decision",
                    ),
                    state=state,
                    final=True,
                )
            )

            if decision.review_required:
                raise ValueError(
                    "The initial decision still "
                    "requires Critic review."
                )

            return {
                "final_decision": (
                    decision.model_dump(
                        mode="json"
                    )
                ),
                "status": "finalised",
            }

        return self._run_node(
            node_name,
            operation,
        )

    # ------------------------------------------------------------------
    # 7. Execute final approve/reject decision
    # ------------------------------------------------------------------

    def execute_access_decision(
        self,
        state: AccessRequestState,
    ) -> dict[str, Any]:
        """
        Execute the final decision and verify the resulting permission.
        """
        node_name = (
            "execute_access_decision"
        )

        def operation() -> dict[str, Any]:
            self._require_status(
                state,
                {
                    "finalised",
                },
                node_name,
            )

            decision = (
                self._validate_decision(
                    self._state_decision(
                        state,
                        "final_decision",
                    ),
                    state=state,
                    final=True,
                )
            )
            request_id = self._state_text(
                state,
                "request_id",
            )
            coordinator_agent_id = (
                self._state_text(
                    state,
                    "coordinator_agent_id",
                )
            )
            reviewed = (
                bool(
                    state.get(
                        "review_completed"
                    )
                )
                and bool(
                    state.get(
                        "review_recorded"
                    )
                )
            )

            if decision.approved:
                if reviewed:
                    raw_result = (
                        self.dependencies
                        .sharing_gateway
                        .approve_request(
                            coordinator_agent_id=(
                                coordinator_agent_id
                            ),
                            request_id=(
                                request_id
                            ),
                            comment=(
                                decision.reason
                            ),
                        )
                    )
                else:
                    approve_direct = getattr(
                        self.dependencies
                        .sharing_gateway,
                        "approve_direct_request",
                        None,
                    )

                    if not callable(
                        approve_direct
                    ):
                        raise TypeError(
                            "sharing_gateway must "
                            "provide "
                            "approve_direct_request() "
                            "for an unreviewed "
                            "approval."
                        )

                    raw_result = (
                        approve_direct(
                            coordinator_agent_id=(
                                coordinator_agent_id
                            ),
                            request_id=(
                                request_id
                            ),
                            comment=(
                                decision.reason
                            ),
                        )
                    )
            else:
                raw_result = (
                    self.dependencies
                    .sharing_gateway
                    .reject_request(
                        coordinator_agent_id=(
                            coordinator_agent_id
                        ),
                        request_id=request_id,
                        comment=decision.reason,
                        require_critic_review=(
                            reviewed
                        ),
                    )
                )

            result = self._model_validate(
                MemoryAccessDecisionResult,
                raw_result,
                "Access decision result",
            )

            self._validate_decision_result(
                result,
                decision=decision,
                state=state,
            )

            confirmed_access = bool(
                self.dependencies
                .sharing_gateway
                .has_access(
                    agent_id=(
                        self._state_text(
                            state,
                            "requester_agent_id",
                        )
                    ),
                    memory_id=(
                        self._state_text(
                            state,
                            "memory_id",
                        )
                    ),
                )
            )

            if (
                confirmed_access
                != decision.approved
            ):
                raise ValueError(
                    "Post-decision permission "
                    "verification does not match "
                    "the final Coordinator "
                    "decision."
                )

            return {
                "decision_result": (
                    result.model_dump(
                        mode="json"
                    )
                ),
                "decision_executed": True,
                "access_granted": (
                    result.access_granted
                ),
                "final_access_confirmed": (
                    confirmed_access
                ),
                "final_request_status": (
                    result.status.value
                ),
                "final_request_decision": (
                    result.decision.value
                ),
                "status": "executed",
            }

        return self._run_node(
            node_name,
            operation,
        )

    # ------------------------------------------------------------------
    # Routing helpers for access_request_workflow.py
    # ------------------------------------------------------------------

    @staticmethod
    def route_after_validation(
        state: AccessRequestState,
    ) -> AccessRequestRoute:
        return (
            "end"
            if state.get("status") == "failed"
            else "check_access"
        )

    @staticmethod
    def route_after_access_check(
        state: AccessRequestState,
    ) -> AccessRequestRoute:
        if state.get("status") in {
            "failed",
            "already_authorised",
        }:
            return "end"

        return "submit"

    @staticmethod
    def route_after_submission(
        state: AccessRequestState,
    ) -> AccessRequestRoute:
        return (
            "end"
            if state.get("status") == "failed"
            else "decide"
        )

    @staticmethod
    def route_after_initial_decision(
        state: AccessRequestState,
    ) -> AccessRequestRoute:
        return (
            "end"
            if state.get("status") == "failed"
            else "evaluate_review"
        )

    @staticmethod
    def route_after_review_evaluation(
        state: AccessRequestState,
    ) -> AccessRequestRoute:
        if state.get("status") == "failed":
            return "end"

        return (
            "review"
            if bool(
                state.get("review_required")
            )
            else "accept"
        )

    @staticmethod
    def route_after_critic_review(
        state: AccessRequestState,
    ) -> AccessRequestRoute:
        return (
            "end"
            if state.get("status") == "failed"
            else "record_review"
        )

    @staticmethod
    def route_after_review_recording(
        state: AccessRequestState,
    ) -> AccessRequestRoute:
        return (
            "end"
            if state.get("status") == "failed"
            else "finalise"
        )

    @staticmethod
    def route_after_finalisation(
        state: AccessRequestState,
    ) -> AccessRequestRoute:
        return (
            "end"
            if state.get("status") == "failed"
            else "execute"
        )

    # ------------------------------------------------------------------
    # Common node execution and failure handling
    # ------------------------------------------------------------------

    def _run_node(
        self,
        node_name: str,
        operation: Callable[
            [],
            dict[str, Any],
        ],
    ) -> dict[str, Any]:
        try:
            update = dict(operation())
            update.update(
                self._trace_update(
                    node_name
                )
            )
            return update
        except AccessRequestNodeError:
            raise
        except Exception as error:
            wrapped = (
                AccessRequestNodeError(
                    node_name,
                    str(error)
                    or error.__class__.__name__,
                    cause=error,
                )
            )

            if not (
                self.dependencies
                .config
                .fail_closed
            ):
                raise wrapped from error

            update: dict[str, Any] = {
                "status": "failed",
                "decision_executed": False,
                "access_granted": False,
                "final_access_confirmed": None,
                "failed_node": node_name,
                "error_type": (
                    error.__class__.__name__
                ),
                "error_message": str(error),
                "errors": [
                    str(wrapped)
                ],
            }

            if (
                self.dependencies
                .config
                .save_node_trace
            ):
                update["node_trace"] = [
                    node_name
                ]

            return update

    def _trace_update(
        self,
        node_name: str,
    ) -> dict[str, Any]:
        if not (
            self.dependencies
            .config
            .save_node_trace
        ):
            return {}

        return {
            "node_trace": [node_name],
        }

    def _with_warnings(
        self,
        update: dict[str, Any],
        warnings: Sequence[str],
    ) -> dict[str, Any]:
        clean_warnings = (
            self._clean_string_list(
                warnings
            )
        )

        if (
            clean_warnings
            and self.dependencies
            .config
            .save_warnings
        ):
            update["warnings"] = (
                clean_warnings
            )

        return update

    # ------------------------------------------------------------------
    # Structured-output and Gateway-result validation
    # ------------------------------------------------------------------

    def _validate_decision(
        self,
        value: Any,
        *,
        state: AccessRequestState,
        final: bool,
    ) -> MemoryAccessRequestDecisionOutput:
        decision = self._model_validate(
            MemoryAccessRequestDecisionOutput,
            value,
            (
                "Final access-request decision"
                if final
                else (
                    "Initial access-request "
                    "decision"
                )
            ),
        )

        expected_request_id = (
            self._state_text(
                state,
                "request_id",
            )
        )
        expected_memory_id = (
            self._state_text(
                state,
                "memory_id",
            )
        )

        if (
            decision.request_id
            != expected_request_id
        ):
            raise ValueError(
                "Coordinator returned a "
                "decision for the wrong "
                "request: "
                f"{decision.request_id!r} != "
                f"{expected_request_id!r}."
            )

        if (
            decision.memory_id
            != expected_memory_id
        ):
            raise ValueError(
                "Coordinator returned a "
                "decision for the wrong "
                "memory: "
                f"{decision.memory_id!r} != "
                f"{expected_memory_id!r}."
            )

        return decision

    def _validate_review(
        self,
        value: Any,
        *,
        state: AccessRequestState,
    ) -> MemoryAccessRequestReviewOutput:
        review = self._model_validate(
            MemoryAccessRequestReviewOutput,
            value,
            "Critic access-request review",
        )

        expected_request_id = (
            self._state_text(
                state,
                "request_id",
            )
        )
        expected_memory_id = (
            self._state_text(
                state,
                "memory_id",
            )
        )

        if (
            review.request_id
            != expected_request_id
        ):
            raise ValueError(
                "Critic returned a review "
                "for the wrong request."
            )

        if (
            review.memory_id
            != expected_memory_id
        ):
            raise ValueError(
                "Critic returned a review "
                "for the wrong memory."
            )

        return review

    @staticmethod
    def _validate_request_result(
        request: MemoryAccessRequestResult,
        *,
        state: AccessRequestState,
        expected_status: MemoryAccessStatus,
    ) -> None:
        if (
            request.memory_id
            != str(state.get("memory_id"))
        ):
            raise ValueError(
                "Gateway request result refers "
                "to the wrong memory."
            )

        if (
            request.owner_agent_id
            != str(
                state.get(
                    "owner_agent_id"
                )
            )
        ):
            raise ValueError(
                "Gateway request result refers "
                "to the wrong owner."
            )

        if (
            request.requester_agent_id
            != str(
                state.get(
                    "requester_agent_id"
                )
            )
        ):
            raise ValueError(
                "Gateway request result refers "
                "to the wrong requester."
            )

        expected_task_id = (
            AccessRequestNodes
            ._optional_text(
                state.get("task_id")
            )
        )

        if request.task_id != expected_task_id:
            raise ValueError(
                "Gateway request result refers "
                "to the wrong task."
            )

        state_request_id = (
            AccessRequestNodes
            ._optional_text(
                state.get("request_id")
            )
        )

        if (
            state_request_id is not None
            and request.request_id
            != state_request_id
        ):
            raise ValueError(
                "Gateway returned a different "
                "request_id."
            )

        if request.status != expected_status:
            raise ValueError(
                "Gateway request status "
                f"{request.status.value!r} "
                "does not match expected "
                f"{expected_status.value!r}."
            )

    @staticmethod
    def _validate_decision_result(
        result: MemoryAccessDecisionResult,
        *,
        decision: (
            MemoryAccessRequestDecisionOutput
        ),
        state: AccessRequestState,
    ) -> None:
        expected_request_id = str(
            state.get("request_id")
        )
        expected_memory_id = str(
            state.get("memory_id")
        )
        expected_requester_id = str(
            state.get(
                "requester_agent_id"
            )
        )

        if (
            result.request_id
            != expected_request_id
        ):
            raise ValueError(
                "Gateway decision result refers "
                "to the wrong request."
            )

        if (
            result.memory_id
            != expected_memory_id
        ):
            raise ValueError(
                "Gateway decision result refers "
                "to the wrong memory."
            )

        if (
            result.requester_agent_id
            != expected_requester_id
        ):
            raise ValueError(
                "Gateway decision result refers "
                "to the wrong requester."
            )

        expected_decision = (
            MemoryAccessDecision.APPROVED
            if decision.approved
            else MemoryAccessDecision.REJECTED
        )
        expected_status = (
            MemoryAccessStatus.APPROVED
            if decision.approved
            else MemoryAccessStatus.REJECTED
        )

        if (
            result.decision
            != expected_decision
        ):
            raise ValueError(
                "Gateway decision does not "
                "match the final Coordinator "
                "decision."
            )

        if result.status != expected_status:
            raise ValueError(
                "Gateway final request status "
                "does not match the final "
                "Coordinator decision."
            )

        if (
            result.access_granted
            != decision.approved
        ):
            raise ValueError(
                "Gateway access_granted value "
                "does not match the final "
                "Coordinator decision."
            )

    @staticmethod
    def _validate_gate_result(
        value: Any,
    ) -> ReviewGateResult:
        if not isinstance(
            value,
            ReviewGateResult,
        ):
            raise TypeError(
                "Review gate must return "
                "ReviewGateResult."
            )

        return value

    @staticmethod
    def _model_validate(
        model_type: type[T],
        value: Any,
        label: str,
    ) -> T:
        try:
            if isinstance(
                value,
                BaseModel,
            ):
                payload: Any = (
                    value.model_dump(
                        mode="python"
                    )
                )
            else:
                payload = value

            return model_type.model_validate(
                payload
            )
        except ValidationError as error:
            raise ValueError(
                f"{label} is invalid: {error}"
            ) from error

    # ------------------------------------------------------------------
    # State access helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_status(
        state: Mapping[str, Any],
        allowed_statuses: set[str],
        node_name: str,
    ) -> None:
        status = str(
            state.get("status") or ""
        ).strip()

        if status not in allowed_statuses:
            raise ValueError(
                f"{node_name} cannot run from "
                f"status {status!r}; expected "
                f"one of "
                f"{sorted(allowed_statuses)}."
            )

    @classmethod
    def _state_text(
        cls,
        state: Mapping[str, Any],
        field_name: str,
    ) -> str:
        return cls._required_text(
            state.get(field_name),
            field_name,
        )

    @classmethod
    def _state_ids(
        cls,
        state: Mapping[str, Any],
        field_name: str,
        *,
        allow_empty: bool = False,
    ) -> list[str]:
        values = cls._clean_string_list(
            state.get(
                field_name,
                [],
            )
        )

        if (
            not values
            and not allow_empty
        ):
            raise ValueError(
                f"{field_name} cannot be empty."
            )

        return values

    @classmethod
    def _state_mapping(
        cls,
        state: Mapping[str, Any],
        field_name: str,
    ) -> dict[str, Any]:
        return cls._mapping_copy(
            state.get(
                field_name,
                {},
            ),
            field_name,
        )

    @classmethod
    def _state_decision(
        cls,
        state: Mapping[str, Any],
        field_name: str,
    ) -> MemoryAccessRequestDecisionOutput:
        value = state.get(field_name)

        if value is None:
            raise ValueError(
                f"{field_name} is missing."
            )

        return cls._model_validate(
            MemoryAccessRequestDecisionOutput,
            value,
            field_name,
        )

    @classmethod
    def _state_review(
        cls,
        state: Mapping[str, Any],
        field_name: str,
    ) -> MemoryAccessRequestReviewOutput:
        value = state.get(field_name)

        if value is None:
            raise ValueError(
                f"{field_name} is missing."
            )

        return cls._model_validate(
            MemoryAccessRequestReviewOutput,
            value,
            field_name,
        )

    # ------------------------------------------------------------------
    # Primitive validation helpers
    # ------------------------------------------------------------------

    @staticmethod
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

    @staticmethod
    def _optional_text(
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        cleaned = str(value).strip()
        return cleaned or None

    @staticmethod
    def _clean_string_list(
        values: Any,
    ) -> list[str]:
        if values is None:
            return []

        if isinstance(values, str):
            values = [values]

        try:
            iterator = iter(values)
        except TypeError as error:
            raise ValueError(
                "Expected a sequence of "
                "string values."
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

    @staticmethod
    def _mapping_copy(
        value: Any,
        field_name: str,
    ) -> dict[str, Any]:
        if not isinstance(
            value,
            Mapping,
        ):
            raise ValueError(
                f"{field_name} must be a mapping."
            )

        return dict(value)

    @staticmethod
    def _assert_json_serialisable(
        value: Any,
    ) -> None:
        try:
            json.dumps(
                value,
                ensure_ascii=False,
            )
        except (
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                "Access-request context must "
                "be JSON serialisable."
            ) from error


def build_access_request_nodes(
    dependencies: AccessRequestDependencies,
) -> AccessRequestNodes:
    """
    Construct the node collection used by the LangGraph builder.
    """
    return AccessRequestNodes(
        dependencies=dependencies
    )


__all__ = [
    "AccessRequestRoute",
    "AccessRequestNodeError",
    "AccessRequestNodes",
    "build_access_request_nodes",
]