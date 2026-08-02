from __future__ import annotations

"""
Dependency containers for the generic memory-governance subgraphs.

The classes in this module collect the external capabilities required by
LangGraph nodes. They do not execute workflow steps themselves.

Nodes depend on these abstractions instead of importing concrete LLM clients,
experiment prompts, application services, stores, or persistence entities.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.llm.output_schemas import (
    MemoryAccessDecisionOutput,
    MemoryAccessRequestDecisionOutput,
)
from src.adapters.memory_access_policy_gateway import (
    MemoryAccessPolicyGateway,
)
from src.adapters.memory_sharing_gateway import (
    MemorySharingGateway,
)

from .config import (
    AccessRequestConfig,
    GovernanceConfig,
    PolicyAssignmentConfig,
)
from .decision_protocols import (
    AccessRequestCoordinatorProtocol,
    AccessRequestCriticProtocol,
    InitialPolicyCoordinatorProtocol,
    InitialPolicyCriticProtocol,
)
from .review_gate import (
    GovernanceReviewGate,
    ReviewGateResult,
)


# ---------------------------------------------------------------------------
# Review-gate contracts
# ---------------------------------------------------------------------------


@runtime_checkable
class PolicyReviewGateProtocol(Protocol):
    """
    Review-gate capability required by the policy-assignment subgraph.
    """

    def requires_policy_review(
        self,
        *,
        decision: MemoryAccessDecisionOutput,
        owner_agent_id: str,
        available_agent_ids: Sequence[str],
        risk_labels: Sequence[str] = (),
    ) -> ReviewGateResult:
        ...


@runtime_checkable
class AccessRequestReviewGateProtocol(Protocol):
    """
    Review-gate capability required by the access-request subgraph.
    """

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
        ...


# ---------------------------------------------------------------------------
# Policy-assignment dependencies
# ---------------------------------------------------------------------------


@dataclass(
    frozen=True,
    slots=True,
)
class PolicyAssignmentDependencies:
    """
    External capabilities used by the initial policy-assignment subgraph.

    The policy-assignment nodes use:

    - ``coordinator`` to propose and finalise the ACL decision;
    - ``critic`` to review a risky proposal;
    - ``access_policy_gateway`` to execute the final ACL change;
    - ``review_gate`` to determine whether Critic review is mandatory;
    - ``config`` for deterministic workflow behaviour.

    This object contains dependencies only. It does not hold per-run workflow
    state such as memory IDs, decisions, reviews, or execution results.
    """

    coordinator: InitialPolicyCoordinatorProtocol
    critic: InitialPolicyCriticProtocol
    access_policy_gateway: MemoryAccessPolicyGateway
    review_gate: PolicyReviewGateProtocol
    config: GovernanceConfig

    def __post_init__(self) -> None:
        self._require_dependency(
            self.coordinator,
            "coordinator",
            (
                "assign_initial_policy",
                "finalise_initial_policy",
            ),
        )
        self._require_dependency(
            self.critic,
            "critic",
            ("review_initial_policy",),
        )
        self._require_dependency(
            self.access_policy_gateway,
            "access_policy_gateway",
            ("apply_policy",),
        )
        self._require_dependency(
            self.review_gate,
            "review_gate",
            ("requires_policy_review",),
        )
        self._validate_config_alignment()

    @classmethod
    def create(
        cls,
        *,
        coordinator: InitialPolicyCoordinatorProtocol,
        critic: InitialPolicyCriticProtocol,
        access_policy_gateway: MemoryAccessPolicyGateway,
        config: GovernanceConfig | None = None,
        review_gate: PolicyReviewGateProtocol | None = None,
    ) -> "PolicyAssignmentDependencies":
        """
        Build dependencies with a default GovernanceReviewGate.

        Supplying ``review_gate`` is useful for unit tests or custom routing
        policies. When omitted, the gate is constructed from the same
        GovernanceConfig stored in this dependency container.
        """
        resolved_config = (
            config or GovernanceConfig()
        )
        resolved_gate = (
            review_gate
            or GovernanceReviewGate(
                resolved_config
            )
        )

        return cls(
            coordinator=coordinator,
            critic=critic,
            access_policy_gateway=(
                access_policy_gateway
            ),
            review_gate=resolved_gate,
            config=resolved_config,
        )

    @property
    def policy_config(
        self,
    ) -> PolicyAssignmentConfig:
        """
        Return the policy-assignment section of the shared configuration.
        """
        return self.config.policy_assignment

    def _validate_config_alignment(self) -> None:
        if not isinstance(
            self.config,
            GovernanceConfig,
        ):
            raise TypeError(
                "config must be a GovernanceConfig instance."
            )

        if (
            isinstance(
                self.review_gate,
                GovernanceReviewGate,
            )
            and self.review_gate.config
            != self.config
        ):
            raise ValueError(
                "review_gate and dependencies must use "
                "the same GovernanceConfig."
            )

    @staticmethod
    def _require_dependency(
        dependency: object,
        field_name: str,
        required_methods: Sequence[str],
    ) -> None:
        if dependency is None:
            raise ValueError(
                f"{field_name} cannot be None."
            )

        missing_methods = [
            method_name
            for method_name in required_methods
            if not callable(
                getattr(
                    dependency,
                    method_name,
                    None,
                )
            )
        ]

        if missing_methods:
            raise TypeError(
                f"{field_name} is missing required "
                f"methods: {missing_methods}."
            )


# ---------------------------------------------------------------------------
# Access-request dependencies
# ---------------------------------------------------------------------------


@dataclass(
    frozen=True,
    slots=True,
)
class AccessRequestDependencies:
    """
    External capabilities used by the runtime access-request subgraph.

    The access-request nodes use:

    - ``coordinator`` to evaluate and finalise a request decision;
    - ``critic`` to review a risky request;
    - ``sharing_gateway`` to submit, review, approve, or reject the request;
    - ``review_gate`` to determine whether Critic review is mandatory;
    - ``config`` for deterministic workflow behaviour.

    This object contains no request-specific or memory-specific workflow state.
    """

    coordinator: AccessRequestCoordinatorProtocol
    critic: AccessRequestCriticProtocol
    sharing_gateway: MemorySharingGateway
    review_gate: AccessRequestReviewGateProtocol
    config: GovernanceConfig

    def __post_init__(self) -> None:
        self._require_dependency(
            self.coordinator,
            "coordinator",
            (
                "evaluate_access_request",
                "finalise_access_request",
            ),
        )
        self._require_dependency(
            self.critic,
            "critic",
            ("review_access_request",),
        )
        self._require_dependency(
            self.sharing_gateway,
            "sharing_gateway",
            (
                "has_access",
                "request_access",
                "review_request",
                "approve_request",
                "reject_request",
                "get_request",
            ),
        )
        self._require_dependency(
            self.review_gate,
            "review_gate",
            (
                "requires_access_request_review",
            ),
        )
        self._validate_config_alignment()

    @classmethod
    def create(
        cls,
        *,
        coordinator: AccessRequestCoordinatorProtocol,
        critic: AccessRequestCriticProtocol,
        sharing_gateway: MemorySharingGateway,
        config: GovernanceConfig | None = None,
        review_gate: (
            AccessRequestReviewGateProtocol
            | None
        ) = None,
    ) -> "AccessRequestDependencies":
        """
        Build dependencies with a default GovernanceReviewGate.
        """
        resolved_config = (
            config or GovernanceConfig()
        )
        resolved_gate = (
            review_gate
            or GovernanceReviewGate(
                resolved_config
            )
        )

        return cls(
            coordinator=coordinator,
            critic=critic,
            sharing_gateway=sharing_gateway,
            review_gate=resolved_gate,
            config=resolved_config,
        )

    @property
    def access_request_config(
        self,
    ) -> AccessRequestConfig:
        """
        Return the access-request section of the shared configuration.
        """
        return self.config.access_request

    def _validate_config_alignment(self) -> None:
        if not isinstance(
            self.config,
            GovernanceConfig,
        ):
            raise TypeError(
                "config must be a GovernanceConfig instance."
            )

        if (
            isinstance(
                self.review_gate,
                GovernanceReviewGate,
            )
            and self.review_gate.config
            != self.config
        ):
            raise ValueError(
                "review_gate and dependencies must use "
                "the same GovernanceConfig."
            )

    @staticmethod
    def _require_dependency(
        dependency: object,
        field_name: str,
        required_methods: Sequence[str],
    ) -> None:
        if dependency is None:
            raise ValueError(
                f"{field_name} cannot be None."
            )

        missing_methods = [
            method_name
            for method_name in required_methods
            if not callable(
                getattr(
                    dependency,
                    method_name,
                    None,
                )
            )
        ]

        if missing_methods:
            raise TypeError(
                f"{field_name} is missing required "
                f"methods: {missing_methods}."
            )


__all__ = [
    "PolicyReviewGateProtocol",
    "AccessRequestReviewGateProtocol",
    "PolicyAssignmentDependencies",
    "AccessRequestDependencies",
]