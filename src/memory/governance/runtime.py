from __future__ import annotations

"""
Composition root for the generic memory-governance subsystem.

This module assembles concrete decision adapters, Gateway implementations,
configuration, deterministic review routing, dependency containers, and the
two compiled LangGraph subgraphs.

It does not contain prompt logic, governance decisions, ACL mutation logic,
persistence access, or experiment-specific workflow state.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import (
    Any,
    Protocol,
    runtime_checkable,
)

from src.adapters.memory_access_policy_gateway import (
    MemoryAccessPolicyGateway,
)
from src.adapters.memory_sharing_gateway import (
    MemorySharingGateway,
)

from .access_request_state import (
    AccessRequestState,
)
from .access_request_workflow import (
    AccessRequestWorkflow,
    build_access_request_workflow,
)
from .config import GovernanceConfig
from .decision_protocols import (
    MemoryGovernanceCoordinatorProtocol,
    MemoryGovernanceCriticProtocol,
)
from .dependencies import (
    AccessRequestDependencies,
    AccessRequestReviewGateProtocol,
    PolicyAssignmentDependencies,
    PolicyReviewGateProtocol,
)
from .policy_assignment_state import (
    PolicyAssignmentState,
)
from .policy_assignment_workflow import (
    PolicyAssignmentWorkflow,
    build_policy_assignment_workflow,
)
from .review_gate import GovernanceReviewGate


@runtime_checkable
class MemoryGovernanceReviewGateProtocol(
    PolicyReviewGateProtocol,
    AccessRequestReviewGateProtocol,
    Protocol,
):
    """
    Review-gate contract required by both governance subgraphs.
    """


@dataclass(
    frozen=True,
    slots=True,
)
class WorkflowCompileOptions:
    """
    LangGraph compilation options for one governance subgraph.

    Separate option objects allow policy assignment and access requests to use
    different checkpointers or interruption points.
    """

    checkpointer: Any | None = None
    interrupt_before: tuple[str, ...] = ()
    interrupt_after: tuple[str, ...] = ()
    debug: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "interrupt_before",
            self._normalise_node_names(
                self.interrupt_before,
                "interrupt_before",
            ),
        )
        object.__setattr__(
            self,
            "interrupt_after",
            self._normalise_node_names(
                self.interrupt_after,
                "interrupt_after",
            ),
        )
        object.__setattr__(
            self,
            "debug",
            bool(self.debug),
        )

    @classmethod
    def create(
        cls,
        *,
        checkpointer: Any | None = None,
        interrupt_before: Sequence[str] | None = None,
        interrupt_after: Sequence[str] | None = None,
        debug: bool = False,
    ) -> "WorkflowCompileOptions":
        """
        Build options from ordinary sequences.
        """
        return cls(
            checkpointer=checkpointer,
            interrupt_before=tuple(
                interrupt_before or ()
            ),
            interrupt_after=tuple(
                interrupt_after or ()
            ),
            debug=debug,
        )

    @staticmethod
    def _normalise_node_names(
        values: Sequence[str],
        field_name: str,
    ) -> tuple[str, ...]:
        if isinstance(values, str):
            values = (values,)

        result: list[str] = []

        try:
            iterator = iter(values)
        except TypeError as error:
            raise TypeError(
                f"{field_name} must be a "
                "sequence of node names."
            ) from error

        for value in iterator:
            node_name = str(
                value or ""
            ).strip()

            if not node_name:
                raise ValueError(
                    f"{field_name} cannot "
                    "contain an empty node name."
                )

            if node_name not in result:
                result.append(node_name)

        return tuple(result)


@dataclass(
    frozen=True,
    slots=True,
)
class MemoryGovernanceRuntime:
    """
    Fully assembled runtime for generic memory governance.

    The runtime exposes the two compiled subgraphs and retains their
    dependencies for inspection, testing, and parent-workflow integration.
    """

    config: GovernanceConfig
    review_gate: (
        MemoryGovernanceReviewGateProtocol
    )

    policy_assignment_dependencies: (
        PolicyAssignmentDependencies
    )
    access_request_dependencies: (
        AccessRequestDependencies
    )

    policy_assignment_workflow: (
        PolicyAssignmentWorkflow
    )
    access_request_workflow: (
        AccessRequestWorkflow
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.config,
            GovernanceConfig,
        ):
            raise TypeError(
                "config must be a "
                "GovernanceConfig instance."
            )

        if (
            self.policy_assignment_dependencies
            .config
            != self.config
        ):
            raise ValueError(
                "Policy-assignment dependencies "
                "must use the runtime config."
            )

        if (
            self.access_request_dependencies
            .config
            != self.config
        ):
            raise ValueError(
                "Access-request dependencies "
                "must use the runtime config."
            )

        if (
            self.policy_assignment_dependencies
            .review_gate
            is not self.review_gate
        ):
            raise ValueError(
                "Policy-assignment dependencies "
                "must use the runtime review gate."
            )

        if (
            self.access_request_dependencies
            .review_gate
            is not self.review_gate
        ):
            raise ValueError(
                "Access-request dependencies "
                "must use the runtime review gate."
            )

        if (
            self.policy_assignment_workflow
            .nodes
            .dependencies
            is not
            self.policy_assignment_dependencies
        ):
            raise ValueError(
                "Policy-assignment workflow is "
                "not bound to the runtime "
                "dependencies."
            )

        if (
            self.access_request_workflow
            .nodes
            .dependencies
            is not
            self.access_request_dependencies
        ):
            raise ValueError(
                "Access-request workflow is "
                "not bound to the runtime "
                "dependencies."
            )

    def assign_initial_policy(
        self,
        state: Mapping[str, Any],
        *,
        config: Mapping[str, Any] | None = None,
    ) -> PolicyAssignmentState:
        """
        Execute the initial policy-assignment subgraph.
        """
        return (
            self.policy_assignment_workflow
            .invoke(
                state,
                config=config,
            )
        )

    async def aassign_initial_policy(
        self,
        state: Mapping[str, Any],
        *,
        config: Mapping[str, Any] | None = None,
    ) -> PolicyAssignmentState:
        """
        Execute the initial policy-assignment subgraph asynchronously.
        """
        return await (
            self.policy_assignment_workflow
            .ainvoke(
                state,
                config=config,
            )
        )

    def request_memory_access(
        self,
        state: Mapping[str, Any],
        *,
        config: Mapping[str, Any] | None = None,
    ) -> AccessRequestState:
        """
        Execute the runtime memory-access request subgraph.
        """
        return (
            self.access_request_workflow
            .invoke(
                state,
                config=config,
            )
        )

    async def arequest_memory_access(
        self,
        state: Mapping[str, Any],
        *,
        config: Mapping[str, Any] | None = None,
    ) -> AccessRequestState:
        """
        Execute the runtime access-request subgraph asynchronously.
        """
        return await (
            self.access_request_workflow
            .ainvoke(
                state,
                config=config,
            )
        )


def build_memory_governance_runtime(
    *,
    coordinator: (
        MemoryGovernanceCoordinatorProtocol
    ),
    critic: MemoryGovernanceCriticProtocol,
    access_policy_gateway: (
        MemoryAccessPolicyGateway
    ),
    sharing_gateway: MemorySharingGateway,
    config: GovernanceConfig | None = None,
    review_gate: (
        MemoryGovernanceReviewGateProtocol
        | None
    ) = None,
    policy_assignment_compile: (
        WorkflowCompileOptions | None
    ) = None,
    access_request_compile: (
        WorkflowCompileOptions | None
    ) = None,
) -> MemoryGovernanceRuntime:
    """
    Assemble and compile the complete memory-governance runtime.

    The supplied Coordinator and Critic must implement both the initial policy
    assignment and runtime access-request decision protocols.
    """
    _require_methods(
        coordinator,
        "coordinator",
        (
            "assign_initial_policy",
            "finalise_initial_policy",
            "evaluate_access_request",
            "finalise_access_request",
        ),
    )
    _require_methods(
        critic,
        "critic",
        (
            "review_initial_policy",
            "review_access_request",
        ),
    )
    _require_methods(
        access_policy_gateway,
        "access_policy_gateway",
        (
            "apply_policy",
            "grant_read_access",
            "revoke_read_access",
            "make_private",
            "make_globally_shared",
        ),
    )
    _require_methods(
        sharing_gateway,
        "sharing_gateway",
        (
            "has_access",
            "request_access",
            "review_request",
            "approve_request",
            "approve_direct_request",
            "reject_request",
            "get_request",
        ),
    )

    resolved_config = (
        config or GovernanceConfig()
    )

    if not isinstance(
        resolved_config,
        GovernanceConfig,
    ):
        raise TypeError(
            "config must be a "
            "GovernanceConfig instance."
        )

    resolved_review_gate = (
        review_gate
        or GovernanceReviewGate(
            resolved_config
        )
    )

    _require_methods(
        resolved_review_gate,
        "review_gate",
        (
            "requires_policy_review",
            "requires_access_request_review",
        ),
    )

    if (
        isinstance(
            resolved_review_gate,
            GovernanceReviewGate,
        )
        and resolved_review_gate.config
        != resolved_config
    ):
        raise ValueError(
            "review_gate and runtime must use "
            "the same GovernanceConfig."
        )

    policy_dependencies = (
        PolicyAssignmentDependencies.create(
            coordinator=coordinator,
            critic=critic,
            access_policy_gateway=(
                access_policy_gateway
            ),
            config=resolved_config,
            review_gate=(
                resolved_review_gate
            ),
        )
    )

    access_dependencies = (
        AccessRequestDependencies.create(
            coordinator=coordinator,
            critic=critic,
            sharing_gateway=sharing_gateway,
            config=resolved_config,
            review_gate=(
                resolved_review_gate
            ),
        )
    )

    policy_options = (
        policy_assignment_compile
        or WorkflowCompileOptions()
    )
    access_options = (
        access_request_compile
        or WorkflowCompileOptions()
    )

    if not isinstance(
        policy_options,
        WorkflowCompileOptions,
    ):
        raise TypeError(
            "policy_assignment_compile must "
            "be a WorkflowCompileOptions "
            "instance."
        )

    if not isinstance(
        access_options,
        WorkflowCompileOptions,
    ):
        raise TypeError(
            "access_request_compile must be "
            "a WorkflowCompileOptions "
            "instance."
        )

    policy_workflow = (
        build_policy_assignment_workflow(
            dependencies=policy_dependencies,
            checkpointer=(
                policy_options.checkpointer
            ),
            interrupt_before=(
                policy_options.interrupt_before
                or None
            ),
            interrupt_after=(
                policy_options.interrupt_after
                or None
            ),
            debug=policy_options.debug,
        )
    )

    access_workflow = (
        build_access_request_workflow(
            dependencies=access_dependencies,
            checkpointer=(
                access_options.checkpointer
            ),
            interrupt_before=(
                access_options.interrupt_before
                or None
            ),
            interrupt_after=(
                access_options.interrupt_after
                or None
            ),
            debug=access_options.debug,
        )
    )

    return MemoryGovernanceRuntime(
        config=resolved_config,
        review_gate=resolved_review_gate,
        policy_assignment_dependencies=(
            policy_dependencies
        ),
        access_request_dependencies=(
            access_dependencies
        ),
        policy_assignment_workflow=(
            policy_workflow
        ),
        access_request_workflow=(
            access_workflow
        ),
    )


def create_memory_governance_runtime(
    *,
    coordinator: (
        MemoryGovernanceCoordinatorProtocol
    ),
    critic: MemoryGovernanceCriticProtocol,
    access_policy_gateway: (
        MemoryAccessPolicyGateway
    ),
    sharing_gateway: MemorySharingGateway,
    config: GovernanceConfig | None = None,
    review_gate: (
        MemoryGovernanceReviewGateProtocol
        | None
    ) = None,
    policy_assignment_compile: (
        WorkflowCompileOptions | None
    ) = None,
    access_request_compile: (
        WorkflowCompileOptions | None
    ) = None,
) -> MemoryGovernanceRuntime:
    """
    Compatibility alias for build_memory_governance_runtime().
    """
    return build_memory_governance_runtime(
        coordinator=coordinator,
        critic=critic,
        access_policy_gateway=(
            access_policy_gateway
        ),
        sharing_gateway=sharing_gateway,
        config=config,
        review_gate=review_gate,
        policy_assignment_compile=(
            policy_assignment_compile
        ),
        access_request_compile=(
            access_request_compile
        ),
    )


def _require_methods(
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
            f"{field_name} is missing "
            f"required methods: "
            f"{missing_methods}."
        )


__all__ = [
    "MemoryGovernanceReviewGateProtocol",
    "WorkflowCompileOptions",
    "MemoryGovernanceRuntime",
    "build_memory_governance_runtime",
    "create_memory_governance_runtime",
]