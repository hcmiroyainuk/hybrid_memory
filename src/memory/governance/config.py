from __future__ import annotations

"""
Configuration for the generic memory-governance subgraphs.

This module contains deterministic workflow and review-gate settings only.
It does not contain experiment prompts, LLM clients, Agents, Gateways,
Services, Stores, or LangGraph state.
"""

from dataclasses import dataclass, field
from typing import Final


DEFAULT_POLICY_RISK_LABELS: Final[tuple[str, ...]] = (
    "sensitive_content",
    "over_sharing",
    "unknown_reader",
    "policy_violation",
    "least_privilege_violation",
    "conflict",
    "outdated",
    "duplicate",
    "uncertain",
)

DEFAULT_ACCESS_REQUEST_RISK_LABELS: Final[
    tuple[str, ...]
] = (
    "sensitive_content",
    "insufficient_reason",
    "role_mismatch",
    "policy_violation",
    "least_privilege_violation",
    "unknown_requester",
    "conflict",
    "uncertain",
)


def _validate_confidence_threshold(
    value: float,
    field_name: str,
) -> float:
    """
    Validate and normalise a confidence threshold.
    """
    try:
        threshold = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field_name} must be a number between 0.0 and 1.0."
        ) from error

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"{field_name} must be between 0.0 and 1.0."
        )

    return threshold


def _normalise_risk_labels(
    values: tuple[str, ...] | list[str],
    field_name: str,
) -> tuple[str, ...]:
    """
    Strip, lower-case, and deduplicate configured risk labels.
    """
    if values is None:
        raise ValueError(
            f"{field_name} cannot be None."
        )

    result: list[str] = []

    for value in values:
        label = str(value).strip().lower()

        if label and label not in result:
            result.append(label)

    return tuple(result)


@dataclass(
    frozen=True,
    slots=True,
)
class PolicyAssignmentConfig:
    """
    Settings for initial memory access-policy assignment.

    These values are consumed mainly by GovernanceReviewGate and by the
    policy-assignment workflow. They do not decide the final ACL themselves.
    """

    confidence_threshold: float = 0.75

    # Force every initial policy proposal through Critic review.
    always_review: bool = False

    # Global "*" access is high-impact and should normally be reviewed.
    review_global_sharing: bool = True

    # Review decisions involving sensitive memory content.
    review_sensitive_memories: bool = True

    # Review duplicate, conflict, outdated, or otherwise anomalous memories.
    review_anomalous_memories: bool = True

    # Review proposals that contain readers outside the supplied Agent set.
    review_unknown_readers: bool = True

    # Review proposals that appear to violate least-privilege access.
    review_least_privilege_violations: bool = True

    # Coordinator may explicitly request review in its structured output.
    honour_coordinator_review_request: bool = True

    risk_labels_requiring_review: tuple[
        str,
        ...,
    ] = field(
        default_factory=lambda: (
            DEFAULT_POLICY_RISK_LABELS
        )
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "confidence_threshold",
            _validate_confidence_threshold(
                self.confidence_threshold,
                "PolicyAssignmentConfig."
                "confidence_threshold",
            ),
        )

        object.__setattr__(
            self,
            "risk_labels_requiring_review",
            _normalise_risk_labels(
                self.risk_labels_requiring_review,
                "PolicyAssignmentConfig."
                "risk_labels_requiring_review",
            ),
        )


@dataclass(
    frozen=True,
    slots=True,
)
class AccessRequestConfig:
    """
    Settings for runtime cross-agent access-request decisions.
    """

    confidence_threshold: float = 0.75

    # Force every runtime access request through Critic review.
    always_review: bool = False

    # Review requests involving sensitive memories.
    review_sensitive_memories: bool = True

    # Review requests with missing, vague, or insufficient justification.
    review_insufficient_reason: bool = True

    # Review requests that appear inconsistent with Agent role or task.
    review_role_mismatch: bool = True

    # Review requests that appear to violate policy or least privilege.
    review_policy_violations: bool = True
    review_least_privilege_violations: bool = True

    # Coordinator may explicitly request review in its structured output.
    honour_coordinator_review_request: bool = True

    risk_labels_requiring_review: tuple[
        str,
        ...,
    ] = field(
        default_factory=lambda: (
            DEFAULT_ACCESS_REQUEST_RISK_LABELS
        )
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "confidence_threshold",
            _validate_confidence_threshold(
                self.confidence_threshold,
                "AccessRequestConfig."
                "confidence_threshold",
            ),
        )

        object.__setattr__(
            self,
            "risk_labels_requiring_review",
            _normalise_risk_labels(
                self.risk_labels_requiring_review,
                "AccessRequestConfig."
                "risk_labels_requiring_review",
            ),
        )


@dataclass(
    frozen=True,
    slots=True,
)
class GovernanceConfig:
    """
    Top-level configuration shared by the governance subgraphs.

    The nested configuration objects keep policy-assignment settings separate
    from runtime access-request settings while exposing one dependency object
    to the governance runtime.
    """

    policy_assignment: PolicyAssignmentConfig = field(
        default_factory=PolicyAssignmentConfig
    )

    access_request: AccessRequestConfig = field(
        default_factory=AccessRequestConfig
    )

    # Fail closed means a failed governance decision must not broaden access.
    fail_closed: bool = True

    # Preserve compact node names in workflow state for debugging/evaluation.
    save_node_trace: bool = True

    # Preserve non-fatal review and normalisation warnings in workflow state.
    save_warnings: bool = True


DEFAULT_POLICY_ASSIGNMENT_CONFIG: Final[
    PolicyAssignmentConfig
] = PolicyAssignmentConfig()

DEFAULT_ACCESS_REQUEST_CONFIG: Final[
    AccessRequestConfig
] = AccessRequestConfig()

DEFAULT_GOVERNANCE_CONFIG: Final[
    GovernanceConfig
] = GovernanceConfig()


__all__ = [
    "DEFAULT_POLICY_RISK_LABELS",
    "DEFAULT_ACCESS_REQUEST_RISK_LABELS",
    "PolicyAssignmentConfig",
    "AccessRequestConfig",
    "GovernanceConfig",
    "DEFAULT_POLICY_ASSIGNMENT_CONFIG",
    "DEFAULT_ACCESS_REQUEST_CONFIG",
    "DEFAULT_GOVERNANCE_CONFIG",
]