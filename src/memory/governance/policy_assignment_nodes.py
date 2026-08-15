from __future__ import annotations

"""
LangGraph nodes for the generic initial memory-policy assignment subgraph.

The nodes coordinate decision-making and policy execution, while delegating:

- policy proposals and final decisions to the Coordinator;
- advisory review to the Critic;
- deterministic review routing to GovernanceReviewGate;
- persisted ACL mutation to MemoryAccessPolicyGateway.

Every public node returns a partial ``PolicyAssignmentState`` update.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ValidationError

from src.llm.output_schemas import (
    MemoryAccessDecisionOutput,
    MemoryAccessReviewOutput,
)
from src.adapters.memory_access_policy_models import (
    AccessPolicyScope,
    MemoryAccessPolicyResult,
)

from .dependencies import PolicyAssignmentDependencies
from .policy_assignment_state import (
    PolicyAssignmentState,
    validate_policy_assignment_identity,
)
from .review_gate import ReviewGateResult


PolicyAssignmentRoute = Literal[
    "assign",
    "evaluate_review",
    "review",
    "accept",
    "finalise",
    "apply",
    "end",
]

T = TypeVar("T")


class PolicyAssignmentNodeError(RuntimeError):
    """
    Raised when a policy-assignment node cannot complete safely.
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
class PolicyAssignmentNodes:
    """
    Node collection for the initial access-policy assignment subgraph.

    The object is created once with ``PolicyAssignmentDependencies`` and its
    bound methods are registered as LangGraph nodes.
    """

    dependencies: PolicyAssignmentDependencies

    # ------------------------------------------------------------------
    # 1. Input validation
    # ------------------------------------------------------------------

    def validate_policy_assignment_input(
        self,
        state: PolicyAssignmentState,
    ) -> dict[str, Any]:
        """
        Validate and normalise the stable workflow input.
        """
        node_name = (
            "validate_policy_assignment_input"
        )

        def operation() -> dict[str, Any]:
            validate_policy_assignment_identity(
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
            governance_actor_id = (
                self._required_text(
                    state.get(
                        "governance_actor_id"
                    ),
                    "governance_actor_id",
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

            risk_labels = list(
                dict.fromkeys(
                    label.lower()
                    for label in self._clean_string_list(
                        state.get(
                            "risk_labels",
                            [],
                        )
                    )
                )
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

            run_id = (
                self._optional_text(
                    state.get("run_id")
                )
                or f"policy:{memory_id}"
            )

            return {
                "run_id": run_id,
                "memory_id": memory_id,
                "owner_agent_id": (
                    owner_agent_id
                ),
                "governance_actor_id": (
                    governance_actor_id
                ),
                "available_agent_ids": (
                    available_agent_ids
                ),
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
    # 2. Coordinator initial proposal
    # ------------------------------------------------------------------

    def assign_initial_policy(
        self,
        state: PolicyAssignmentState,
    ) -> dict[str, Any]:
        """
        Ask the Coordinator to propose an initial ACL.
        """
        node_name = "assign_initial_policy"

        def operation() -> dict[str, Any]:
            self._require_status(
                state,
                {
                    "validated",
                },
                node_name,
            )

            raw_decision = (
                self.dependencies.coordinator
                .assign_initial_policy(
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

            decision = self._validate_decision(
                raw_decision,
                state=state,
                final=False,
            )

            return {
                "initial_decision": (
                    decision.model_dump(
                        mode="json"
                    )
                ),
                "status": "policy_proposed",
            }

        return self._run_node(
            node_name,
            operation,
        )

    # ------------------------------------------------------------------
    # 3. Deterministic review gate
    # ------------------------------------------------------------------

    def evaluate_policy_review(
        self,
        state: PolicyAssignmentState,
    ) -> dict[str, Any]:
        """
        Decide deterministically whether Critic review is required.
        """
        node_name = "evaluate_policy_review"

        def operation() -> dict[str, Any]:
            self._require_status(
                state,
                {
                    "policy_proposed",
                },
                node_name,
            )

            decision = self._state_decision(
                state,
                "initial_decision",
            )

            raw_result = (
                self.dependencies.review_gate
                .requires_policy_review(
                    decision=decision,
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
                else "policy_proposed"
            )

            return update

        return self._run_node(
            node_name,
            operation,
        )

    # ------------------------------------------------------------------
    # 4A. Critic review path
    # ------------------------------------------------------------------

    def review_initial_policy(
        self,
        state: PolicyAssignmentState,
    ) -> dict[str, Any]:
        """
        Ask the Critic to review the proposed policy.
        """
        node_name = "review_initial_policy"

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
                    "review_initial_policy cannot "
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
                .review_initial_policy(
                    proposed_decision=(
                        initial_decision
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

            review = self._model_validate(
                MemoryAccessReviewOutput,
                raw_review,
                "Critic review",
            )

            expected_memory_id = (
                self._state_text(
                    state,
                    "memory_id",
                )
            )

            if (
                review.memory_id
                != expected_memory_id
            ):
                raise ValueError(
                    "Critic returned a review "
                    "for the wrong memory: "
                    f"{review.memory_id!r} != "
                    f"{expected_memory_id!r}."
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

    def finalise_reviewed_policy(
        self,
        state: PolicyAssignmentState,
    ) -> dict[str, Any]:
        """
        Ask the Coordinator to make the final decision after review.
        """
        node_name = (
            "finalise_reviewed_policy"
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
                state.get("review_completed")
            ):
                raise ValueError(
                    "A completed Critic review "
                    "is required."
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
                self.dependencies.coordinator
                .finalise_initial_policy(
                    initial_decision=(
                        initial_decision
                    ),
                    critic_review=(
                        critic_review
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

            decision = self._validate_decision(
                raw_final_decision,
                state=state,
                final=True,
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
                    "was cleared because Critic review "
                    "has already completed."
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
    # 4B. No-review path
    # ------------------------------------------------------------------

    def accept_unreviewed_policy(
        self,
        state: PolicyAssignmentState,
    ) -> dict[str, Any]:
        """
        Promote the initial decision to final when review is unnecessary.
        """
        node_name = (
            "accept_unreviewed_policy"
        )

        def operation() -> dict[str, Any]:
            self._require_status(
                state,
                {
                    "policy_proposed",
                },
                node_name,
            )

            if bool(
                state.get("review_required")
            ):
                raise ValueError(
                    "A policy requiring review "
                    "cannot use the unreviewed path."
                )

            decision = self._validate_decision(
                self._state_decision(
                    state,
                    "initial_decision",
                ),
                state=state,
                final=True,
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
    # 5. Persist final ACL through the Gateway
    # ------------------------------------------------------------------

    def apply_access_policy(
        self,
        state: PolicyAssignmentState,
    ) -> dict[str, Any]:
        """
        Apply the final policy through MemoryAccessPolicyGateway.
        """
        node_name = "apply_access_policy"

        def operation() -> dict[str, Any]:
            self._require_status(
                state,
                {
                    "finalised",
                },
                node_name,
            )

            decision = self._validate_decision(
                self._state_decision(
                    state,
                    "final_decision",
                ),
                state=state,
                final=True,
            )

            owner_agent_id = (
                self._state_text(
                    state,
                    "owner_agent_id",
                )
            )
            gateway_reader_ids = (
                self._gateway_reader_ids(
                    decision=decision,
                    owner_agent_id=(
                        owner_agent_id
                    ),
                )
            )

            raw_result = (
                self.dependencies
                .access_policy_gateway
                .apply_policy(
                    governance_actor_id=(
                        self._state_text(
                            state,
                            "governance_actor_id",
                        )
                    ),
                    memory_id=(
                        decision.memory_id
                    ),
                    target_scope=(
                        decision.target_scope
                    ),
                    allowed_agent_ids=(
                        gateway_reader_ids
                    ),
                    reason=decision.reason,
                )
            )

            result = self._model_validate(
                MemoryAccessPolicyResult,
                raw_result,
                "Memory access-policy result",
            )

            self._validate_policy_result(
                result,
                decision=decision,
                owner_agent_id=(
                    owner_agent_id
                ),
                expected_reader_ids=(
                    gateway_reader_ids
                ),
            )

            return {
                "policy_result": (
                    result.model_dump(
                        mode="json"
                    )
                ),
                "policy_applied": True,
                "policy_changed": (
                    result.changed
                ),
                "status": "applied",
            }

        return self._run_node(
            node_name,
            operation,
        )

    # ------------------------------------------------------------------
    # Routing helpers for policy_assignment_workflow.py
    # ------------------------------------------------------------------

    @staticmethod
    def route_after_validation(
        state: PolicyAssignmentState,
    ) -> PolicyAssignmentRoute:
        return (
            "end"
            if state.get("status") == "failed"
            else "assign"
        )

    @staticmethod
    def route_after_policy_proposal(
        state: PolicyAssignmentState,
    ) -> PolicyAssignmentRoute:
        return (
            "end"
            if state.get("status") == "failed"
            else "evaluate_review"
        )

    @staticmethod
    def route_after_review_evaluation(
        state: PolicyAssignmentState,
    ) -> PolicyAssignmentRoute:
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
    def route_after_review(
        state: PolicyAssignmentState,
    ) -> PolicyAssignmentRoute:
        return (
            "end"
            if state.get("status") == "failed"
            else "finalise"
        )

    @staticmethod
    def route_after_finalisation(
        state: PolicyAssignmentState,
    ) -> PolicyAssignmentRoute:
        return (
            "end"
            if state.get("status") == "failed"
            else "apply"
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
        except PolicyAssignmentNodeError:
            raise
        except Exception as error:
            wrapped = (
                PolicyAssignmentNodeError(
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
                "policy_applied": False,
                "policy_changed": False,
                "failed_node": node_name,
                "error_type": (
                    error.__class__.__name__
                ),
                "error_message": str(error),
            }

            if (
                self.dependencies
                .config
                .save_node_trace
            ):
                update["node_trace"] = [
                    node_name
                ]

            update["errors"] = [
                str(wrapped)
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
    # Decision and result validation
    # ------------------------------------------------------------------

    def _validate_decision(
        self,
        value: Any,
        *,
        state: PolicyAssignmentState,
        final: bool,
    ) -> MemoryAccessDecisionOutput:
        decision = self._model_validate(
            MemoryAccessDecisionOutput,
            value,
            (
                "Final policy decision"
                if final
                else "Initial policy decision"
            ),
        )

        expected_memory_id = (
            self._state_text(
                state,
                "memory_id",
            )
        )

        if (
            decision.memory_id
            != expected_memory_id
        ):
            raise ValueError(
                "Coordinator returned a decision "
                "for the wrong memory: "
                f"{decision.memory_id!r} != "
                f"{expected_memory_id!r}."
            )

        if final:
            self._validate_final_reader_ids(
                decision,
                state=state,
            )

        return decision

    def _validate_final_reader_ids(
        self,
        decision: MemoryAccessDecisionOutput,
        *,
        state: PolicyAssignmentState,
    ) -> None:
        if (
            decision.target_scope
            == "private"
        ):
            return

        reader_ids = (
            self._clean_string_list(
                decision.allowed_agent_ids
            )
        )

        if reader_ids == ["*"]:
            return

        known_agent_ids = set(
            self._state_ids(
                state,
                "available_agent_ids",
            )
        )
        owner_agent_id = (
            self._state_text(
                state,
                "owner_agent_id",
            )
        )

        unknown_ids = sorted(
            {
                reader_id
                for reader_id in reader_ids
                if reader_id
                not in known_agent_ids
            }
        )

        if unknown_ids:
            raise ValueError(
                "The final policy contains "
                "unknown Agent IDs: "
                f"{unknown_ids}."
            )

        non_owner_reader_ids = [
            reader_id
            for reader_id in reader_ids
            if reader_id
            != owner_agent_id
        ]

        if not non_owner_reader_ids:
            raise ValueError(
                "A targeted shared final policy "
                "must contain at least one "
                "non-owner reader."
            )

    @staticmethod
    def _gateway_reader_ids(
        *,
        decision: MemoryAccessDecisionOutput,
        owner_agent_id: str,
    ) -> list[str]:
        if (
            decision.target_scope
            == "private"
        ):
            return []

        reader_ids = [
            str(reader_id).strip()
            for reader_id in (
                decision.allowed_agent_ids
            )
            if str(reader_id).strip()
        ]

        if reader_ids == ["*"]:
            return ["*"]

        return [
            reader_id
            for reader_id in reader_ids
            if reader_id
            != owner_agent_id
        ]

    @staticmethod
    def _validate_policy_result(
        result: MemoryAccessPolicyResult,
        *,
        decision: MemoryAccessDecisionOutput,
        owner_agent_id: str,
        expected_reader_ids: Sequence[str],
    ) -> None:
        if (
            result.memory_id
            != decision.memory_id
        ):
            raise ValueError(
                "Gateway returned a result for "
                "the wrong memory."
            )

        if (
            result.owner_agent_id
            != owner_agent_id
        ):
            raise ValueError(
                "Gateway returned an unexpected "
                "memory owner."
            )

        expected_scope = (
            AccessPolicyScope(
                decision.target_scope
            )
        )

        if result.scope != expected_scope:
            raise ValueError(
                "Gateway result scope does not "
                "match the final decision."
            )

        if (
            expected_scope
            == AccessPolicyScope.PRIVATE
        ):
            if result.readable_by != (
                owner_agent_id,
            ):
                raise ValueError(
                    "Private policy result must be "
                    "owner-only."
                )
            return

        if list(expected_reader_ids) == ["*"]:
            if result.readable_by != ("*",):
                raise ValueError(
                    "Global policy result must use "
                    "the '*' read ACL."
                )
            return

        expected_readers = {
            owner_agent_id,
            *expected_reader_ids,
        }

        if (
            set(result.readable_by)
            != expected_readers
        ):
            raise ValueError(
                "Gateway result readers do not "
                "match the final targeted policy."
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
                f"status {status!r}; expected one "
                f"of {sorted(allowed_statuses)}."
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

        if not values and not allow_empty:
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
    ) -> MemoryAccessDecisionOutput:
        value = state.get(field_name)

        if value is None:
            raise ValueError(
                f"{field_name} is missing."
            )

        return cls._model_validate(
            MemoryAccessDecisionOutput,
            value,
            field_name,
        )

    @classmethod
    def _state_review(
        cls,
        state: Mapping[str, Any],
        field_name: str,
    ) -> MemoryAccessReviewOutput:
        value = state.get(field_name)

        if value is None:
            raise ValueError(
                f"{field_name} is missing."
            )

        return cls._model_validate(
            MemoryAccessReviewOutput,
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
                "Policy-assignment context "
                "must be JSON serialisable."
            ) from error


def build_policy_assignment_nodes(
    dependencies: PolicyAssignmentDependencies,
) -> PolicyAssignmentNodes:
    """
    Construct the node collection used by the LangGraph builder.
    """
    return PolicyAssignmentNodes(
        dependencies=dependencies
    )


__all__ = [
    "PolicyAssignmentRoute",
    "PolicyAssignmentNodeError",
    "PolicyAssignmentNodes",
    "build_policy_assignment_nodes",
]