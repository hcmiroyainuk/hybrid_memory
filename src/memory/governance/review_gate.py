from __future__ import annotations

"""
Deterministic Critic-review routing for memory governance.

GovernanceReviewGate decides whether a Coordinator decision must be reviewed
by the Critic. It does not call an LLM, mutate workflow state, approve a
request, or modify a memory ACL.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from src.llm.output_schemas import (
    MemoryAccessDecisionOutput,
    MemoryAccessRequestDecisionOutput,
)

from .config import (
    AccessRequestConfig,
    GovernanceConfig,
    PolicyAssignmentConfig,
)


_POLICY_ANOMALY_LABELS = frozenset(
    {
        "duplicate",
        "conflict",
        "outdated",
        "uncertain",
    }
)

_POLICY_LEAST_PRIVILEGE_LABELS = frozenset(
    {
        "least_privilege_violation",
        "over_sharing",
    }
)

_ACCESS_REQUEST_POLICY_LABELS = frozenset(
    {
        "policy_violation",
    }
)

_ACCESS_REQUEST_LEAST_PRIVILEGE_LABELS = frozenset(
    {
        "least_privilege_violation",
    }
)


@dataclass(
    frozen=True,
    slots=True,
)
class ReviewGateResult:
    """
    Explainable result returned by GovernanceReviewGate.

    ``reason_codes`` are stable machine-facing identifiers, while ``reasons``
    are human-readable messages suitable for workflow traces and evaluation.
    """

    review_required: bool

    reason_codes: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    triggered_risk_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            len(self.reason_codes)
            != len(self.reasons)
        ):
            raise ValueError(
                "reason_codes and reasons must "
                "have the same length."
            )

        if not self.review_required:
            if (
                self.reason_codes
                or self.reasons
                or self.triggered_risk_labels
            ):
                raise ValueError(
                    "A no-review result cannot contain "
                    "review reasons or triggered risks."
                )

    @classmethod
    def no_review(
        cls,
    ) -> "ReviewGateResult":
        return cls(
            review_required=False,
        )

    def as_state_update(
        self,
    ) -> dict[str, object]:
        """
        Convert the result into JSON-serialisable workflow-state fields.
        """
        return {
            "review_required": (
                self.review_required
            ),
            "review_reason_codes": list(
                self.reason_codes
            ),
            "review_reasons": list(
                self.reasons
            ),
            "triggered_risk_labels": list(
                self.triggered_risk_labels
            ),
        }


class GovernanceReviewGate:
    """
    Apply deterministic rules before routing a decision to the Critic.

    The gate supplements, rather than replaces, the Coordinator's own
    ``review_required`` signal. High-impact or suspicious decisions can still
    be forced through review even when the Coordinator did not request it.
    """

    def __init__(
        self,
        config: GovernanceConfig | None = None,
    ) -> None:
        self.config = (
            config or GovernanceConfig()
        )

    # ------------------------------------------------------------------
    # Initial access-policy assignment
    # ------------------------------------------------------------------

    def requires_policy_review(
        self,
        *,
        decision: MemoryAccessDecisionOutput,
        owner_agent_id: str,
        available_agent_ids: Sequence[str],
        risk_labels: Sequence[str] = (),
    ) -> ReviewGateResult:
        """
        Decide whether an initial access-policy proposal requires review.
        """
        config = self.config.policy_assignment

        owner_id = self._required_text(
            owner_agent_id,
            "owner_agent_id",
        )
        available_ids = set(
            self._clean_values(
                available_agent_ids
            )
        )
        supplied_risks = set(
            self._clean_labels(
                risk_labels
            )
        )

        reason_codes: list[str] = []
        reasons: list[str] = []
        triggered_risks: list[str] = []

        def add(
            code: str,
            reason: str,
            *,
            risks: Sequence[str] = (),
        ) -> None:
            self._add_reason(
                reason_codes=reason_codes,
                reasons=reasons,
                triggered_risks=triggered_risks,
                code=code,
                reason=reason,
                risks=risks,
            )

        if config.always_review:
            add(
                "policy_always_review",
                "Policy-assignment configuration "
                "requires Critic review.",
            )

        if (
            config.honour_coordinator_review_request
            and decision.review_required
        ):
            add(
                "coordinator_requested_review",
                "The Coordinator explicitly "
                "requested Critic review.",
            )

        if (
            decision.confidence
            < config.confidence_threshold
        ):
            add(
                "policy_low_confidence",
                "The policy decision confidence "
                f"{decision.confidence:.3f} is below "
                "the configured threshold "
                f"{config.confidence_threshold:.3f}.",
            )

        if (
            config.review_global_sharing
            and decision.target_scope == "shared"
            and "*" in decision.allowed_agent_ids
        ):
            add(
                "global_sharing",
                "Global '*' read access requires "
                "Critic review.",
            )

        if (
            config.review_sensitive_memories
            and "sensitive_content" in supplied_risks
        ):
            add(
                "sensitive_memory",
                "The memory is marked as containing "
                "sensitive content.",
                risks=("sensitive_content",),
            )

        anomaly_risks = sorted(
            supplied_risks
            & _POLICY_ANOMALY_LABELS
        )

        if (
            config.review_anomalous_memories
            and anomaly_risks
        ):
            add(
                "memory_anomaly",
                "The memory has governance anomalies: "
                f"{anomaly_risks}.",
                risks=anomaly_risks,
            )

        proposed_reader_ids = {
            reader
            for reader in self._clean_values(
                decision.allowed_agent_ids
            )
            if reader not in {
                "*",
                owner_id,
            }
        }

        unknown_reader_ids = sorted(
            proposed_reader_ids - available_ids
        )

        if (
            config.review_unknown_readers
            and unknown_reader_ids
        ):
            add(
                "unknown_reader",
                "The proposed ACL contains unknown "
                f"Agent IDs: {unknown_reader_ids}.",
                risks=("unknown_reader",),
            )

        least_privilege_risks = sorted(
            supplied_risks
            & _POLICY_LEAST_PRIVILEGE_LABELS
        )

        if (
            config.review_least_privilege_violations
            and least_privilege_risks
        ):
            add(
                "least_privilege_violation",
                "The proposed policy may violate "
                "least-privilege access.",
                risks=least_privilege_risks,
            )

        self._add_unhandled_configured_risks(
            configured_labels=(
                config.risk_labels_requiring_review
            ),
            supplied_risks=supplied_risks,
            already_triggered=triggered_risks,
            reason_codes=reason_codes,
            reasons=reasons,
            triggered_risks=triggered_risks,
            code_prefix="policy",
        )

        return self._build_result(
            reason_codes=reason_codes,
            reasons=reasons,
            triggered_risks=triggered_risks,
        )

    # ------------------------------------------------------------------
    # Runtime access request
    # ------------------------------------------------------------------

    def requires_access_request_review(
        self,
        *,
        decision: MemoryAccessRequestDecisionOutput,
        requester_agent_id: str,
        owner_agent_id: str,
        available_agent_ids: Sequence[str],
        request_reason: str | None = None,
        risk_labels: Sequence[str] = (),
    ) -> ReviewGateResult:
        """
        Decide whether a runtime access-request decision requires review.
        """
        config = self.config.access_request

        requester_id = self._required_text(
            requester_agent_id,
            "requester_agent_id",
        )
        owner_id = self._required_text(
            owner_agent_id,
            "owner_agent_id",
        )
        available_ids = set(
            self._clean_values(
                available_agent_ids
            )
        )
        supplied_risks = set(
            self._clean_labels(
                risk_labels
            )
        )
        clean_request_reason = (
            self._optional_text(request_reason)
        )

        reason_codes: list[str] = []
        reasons: list[str] = []
        triggered_risks: list[str] = []

        def add(
            code: str,
            reason: str,
            *,
            risks: Sequence[str] = (),
        ) -> None:
            self._add_reason(
                reason_codes=reason_codes,
                reasons=reasons,
                triggered_risks=triggered_risks,
                code=code,
                reason=reason,
                risks=risks,
            )

        if config.always_review:
            add(
                "request_always_review",
                "Access-request configuration "
                "requires Critic review.",
            )

        if (
            config.honour_coordinator_review_request
            and decision.review_required
        ):
            add(
                "coordinator_requested_review",
                "The Coordinator explicitly "
                "requested Critic review.",
            )

        if (
            decision.confidence
            < config.confidence_threshold
        ):
            add(
                "request_low_confidence",
                "The access-request decision confidence "
                f"{decision.confidence:.3f} is below "
                "the configured threshold "
                f"{config.confidence_threshold:.3f}.",
            )

        if requester_id not in available_ids:
            add(
                "unknown_requester",
                "The requester is not present in the "
                f"available Agent registry: {requester_id!r}.",
                risks=("unknown_requester",),
            )

        if requester_id == owner_id:
            add(
                "owner_access_request",
                "The memory owner submitted an access "
                "request for its own memory.",
            )

        if (
            config.review_sensitive_memories
            and "sensitive_content" in supplied_risks
        ):
            add(
                "sensitive_memory",
                "The requested memory is marked as "
                "containing sensitive content.",
                risks=("sensitive_content",),
            )

        insufficient_reason = (
            "insufficient_reason" in supplied_risks
            or clean_request_reason is None
        )

        if (
            config.review_insufficient_reason
            and insufficient_reason
        ):
            add(
                "insufficient_request_reason",
                "The access request has no sufficiently "
                "supported justification.",
                risks=("insufficient_reason",),
            )

        if (
            config.review_role_mismatch
            and "role_mismatch" in supplied_risks
        ):
            add(
                "role_mismatch",
                "The requested access may be inconsistent "
                "with the requester's role.",
                risks=("role_mismatch",),
            )

        request_policy_risks = sorted(
            supplied_risks
            & _ACCESS_REQUEST_POLICY_LABELS
        )

        if (
            config.review_policy_violations
            and request_policy_risks
        ):
            add(
                "request_policy_violation",
                "Granting the request may violate the "
                "active governance policy.",
                risks=request_policy_risks,
            )

        request_least_privilege_risks = sorted(
            supplied_risks
            & _ACCESS_REQUEST_LEAST_PRIVILEGE_LABELS
        )

        if (
            config.review_least_privilege_violations
            and request_least_privilege_risks
        ):
            add(
                "request_least_privilege_violation",
                "Granting the request may violate "
                "least-privilege access.",
                risks=request_least_privilege_risks,
            )

        self._add_unhandled_configured_risks(
            configured_labels=(
                config.risk_labels_requiring_review
            ),
            supplied_risks=supplied_risks,
            already_triggered=triggered_risks,
            reason_codes=reason_codes,
            reasons=reasons,
            triggered_risks=triggered_risks,
            code_prefix="request",
        )

        return self._build_result(
            reason_codes=reason_codes,
            reasons=reasons,
            triggered_risks=triggered_risks,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _add_reason(
        *,
        reason_codes: list[str],
        reasons: list[str],
        triggered_risks: list[str],
        code: str,
        reason: str,
        risks: Sequence[str] = (),
    ) -> None:
        if code not in reason_codes:
            reason_codes.append(code)
            reasons.append(reason)

        for risk in risks:
            clean_risk = str(risk).strip().lower()

            if (
                clean_risk
                and clean_risk
                not in triggered_risks
            ):
                triggered_risks.append(
                    clean_risk
                )

    @classmethod
    def _add_unhandled_configured_risks(
        cls,
        *,
        configured_labels: Sequence[str],
        supplied_risks: set[str],
        already_triggered: Sequence[str],
        reason_codes: list[str],
        reasons: list[str],
        triggered_risks: list[str],
        code_prefix: str,
    ) -> None:
        remaining = sorted(
            supplied_risks
            & set(
                cls._clean_labels(
                    configured_labels
                )
            )
            - set(already_triggered)
        )

        for risk in remaining:
            cls._add_reason(
                reason_codes=reason_codes,
                reasons=reasons,
                triggered_risks=triggered_risks,
                code=(
                    f"{code_prefix}_risk_"
                    f"{risk}"
                ),
                reason=(
                    "Configured governance risk "
                    f"requires review: {risk!r}."
                ),
                risks=(risk,),
            )

    @staticmethod
    def _build_result(
        *,
        reason_codes: list[str],
        reasons: list[str],
        triggered_risks: list[str],
    ) -> ReviewGateResult:
        if not reason_codes:
            return ReviewGateResult.no_review()

        return ReviewGateResult(
            review_required=True,
            reason_codes=tuple(reason_codes),
            reasons=tuple(reasons),
            triggered_risk_labels=tuple(
                triggered_risks
            ),
        )

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
    def _clean_values(
        values: Sequence[Any],
    ) -> tuple[str, ...]:
        if values is None:
            return ()

        if isinstance(values, str):
            values = [values]

        result: list[str] = []

        for value in values:
            cleaned = str(
                value or ""
            ).strip()

            if (
                cleaned
                and cleaned not in result
            ):
                result.append(cleaned)

        return tuple(result)

    @classmethod
    def _clean_labels(
        cls,
        values: Sequence[Any],
    ) -> tuple[str, ...]:
        return tuple(
            value.lower()
            for value in cls._clean_values(values)
        )


__all__ = [
    "ReviewGateResult",
    "GovernanceReviewGate",
]