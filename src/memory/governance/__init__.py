"""
Public API for the generic memory-governance package.

The package exposes configuration, decision contracts, dependency containers,
state builders, node collections, workflow builders, and the assembled runtime.
Concrete Service adapters and persistence implementations remain under
``src.memory.adapters`` and ``src.memory.services``.
"""

from .access_request_nodes import (
    AccessRequestNodeError,
    AccessRequestNodes,
    AccessRequestRoute,
    build_access_request_nodes,
)
from .access_request_state import (
    AccessRequestState,
    AccessRequestWorkflowStatus,
    TERMINAL_ACCESS_REQUEST_STATUSES,
    build_initial_access_request_state,
    is_terminal_access_request_status,
    validate_access_request_identity,
)
from .access_request_workflow import (
    ACCEPT_UNREVIEWED_DECISION_NODE,
    ACCESS_REQUEST_NODE_NAMES,
    CHECK_EXISTING_ACCESS_NODE,
    EVALUATE_ACCESS_REQUEST_NODE,
    EVALUATE_REQUEST_REVIEW_NODE,
    EXECUTE_ACCESS_DECISION_NODE,
    FINALISE_REVIEWED_REQUEST_NODE,
    RECORD_CRITIC_REVIEW_NODE,
    REVIEW_ACCESS_REQUEST_NODE,
    SUBMIT_ACCESS_REQUEST_NODE,
    VALIDATE_INPUT_NODE as ACCESS_VALIDATE_INPUT_NODE,
    AccessRequestWorkflow,
    AccessRequestWorkflowError,
    build_access_request_graph,
    build_access_request_workflow,
    create_access_request_workflow,
)
from .config import (
    DEFAULT_ACCESS_REQUEST_CONFIG,
    DEFAULT_ACCESS_REQUEST_RISK_LABELS,
    DEFAULT_GOVERNANCE_CONFIG,
    DEFAULT_POLICY_ASSIGNMENT_CONFIG,
    DEFAULT_POLICY_RISK_LABELS,
    AccessRequestConfig,
    GovernanceConfig,
    PolicyAssignmentConfig,
)
from .decision_protocols import (
    AgentIdSequence,
    DecisionContext,
    AccessRequestCoordinatorProtocol,
    AccessRequestCriticProtocol,
    InitialPolicyCoordinatorProtocol,
    InitialPolicyCriticProtocol,
    MemoryGovernanceCoordinatorProtocol,
    MemoryGovernanceCriticProtocol,
)
from .dependencies import (
    AccessRequestDependencies,
    AccessRequestReviewGateProtocol,
    PolicyAssignmentDependencies,
    PolicyReviewGateProtocol,
)
from .policy_assignment_nodes import (
    PolicyAssignmentNodeError,
    PolicyAssignmentNodes,
    PolicyAssignmentRoute,
    build_policy_assignment_nodes,
)
from .policy_assignment_state import (
    PolicyAssignmentState,
    PolicyAssignmentStatus,
    TERMINAL_POLICY_ASSIGNMENT_STATUSES,
    build_initial_policy_assignment_state,
    is_terminal_policy_assignment_status,
    validate_policy_assignment_identity,
)
from .policy_assignment_workflow import (
    ACCEPT_UNREVIEWED_POLICY_NODE,
    APPLY_ACCESS_POLICY_NODE,
    ASSIGN_INITIAL_POLICY_NODE,
    EVALUATE_REVIEW_NODE,
    FINALISE_REVIEWED_POLICY_NODE,
    POLICY_ASSIGNMENT_NODE_NAMES,
    REVIEW_INITIAL_POLICY_NODE,
    VALIDATE_INPUT_NODE as POLICY_VALIDATE_INPUT_NODE,
    PolicyAssignmentWorkflow,
    PolicyAssignmentWorkflowError,
    build_policy_assignment_graph,
    build_policy_assignment_workflow,
    create_policy_assignment_workflow,
)
from .review_gate import (
    GovernanceReviewGate,
    ReviewGateResult,
)
from .runtime import (
    MemoryGovernanceReviewGateProtocol,
    MemoryGovernanceRuntime,
    WorkflowCompileOptions,
    build_memory_governance_runtime,
    create_memory_governance_runtime,
)


__all__ = [
    # Configuration
    "DEFAULT_POLICY_RISK_LABELS",
    "DEFAULT_ACCESS_REQUEST_RISK_LABELS",
    "PolicyAssignmentConfig",
    "AccessRequestConfig",
    "GovernanceConfig",
    "DEFAULT_POLICY_ASSIGNMENT_CONFIG",
    "DEFAULT_ACCESS_REQUEST_CONFIG",
    "DEFAULT_GOVERNANCE_CONFIG",

    # Decision protocols
    "DecisionContext",
    "AgentIdSequence",
    "InitialPolicyCoordinatorProtocol",
    "InitialPolicyCriticProtocol",
    "AccessRequestCoordinatorProtocol",
    "AccessRequestCriticProtocol",
    "MemoryGovernanceCoordinatorProtocol",
    "MemoryGovernanceCriticProtocol",

    # Dependency contracts
    "PolicyReviewGateProtocol",
    "AccessRequestReviewGateProtocol",
    "PolicyAssignmentDependencies",
    "AccessRequestDependencies",

    # Review routing
    "ReviewGateResult",
    "GovernanceReviewGate",

    # Policy-assignment state
    "PolicyAssignmentStatus",
    "TERMINAL_POLICY_ASSIGNMENT_STATUSES",
    "PolicyAssignmentState",
    "build_initial_policy_assignment_state",
    "is_terminal_policy_assignment_status",
    "validate_policy_assignment_identity",

    # Policy-assignment nodes
    "PolicyAssignmentRoute",
    "PolicyAssignmentNodeError",
    "PolicyAssignmentNodes",
    "build_policy_assignment_nodes",

    # Policy-assignment workflow
    "POLICY_VALIDATE_INPUT_NODE",
    "ASSIGN_INITIAL_POLICY_NODE",
    "EVALUATE_REVIEW_NODE",
    "REVIEW_INITIAL_POLICY_NODE",
    "ACCEPT_UNREVIEWED_POLICY_NODE",
    "FINALISE_REVIEWED_POLICY_NODE",
    "APPLY_ACCESS_POLICY_NODE",
    "POLICY_ASSIGNMENT_NODE_NAMES",
    "PolicyAssignmentWorkflowError",
    "PolicyAssignmentWorkflow",
    "build_policy_assignment_graph",
    "build_policy_assignment_workflow",
    "create_policy_assignment_workflow",

    # Access-request state
    "AccessRequestWorkflowStatus",
    "TERMINAL_ACCESS_REQUEST_STATUSES",
    "AccessRequestState",
    "build_initial_access_request_state",
    "is_terminal_access_request_status",
    "validate_access_request_identity",

    # Access-request nodes
    "AccessRequestRoute",
    "AccessRequestNodeError",
    "AccessRequestNodes",
    "build_access_request_nodes",

    # Access-request workflow
    "ACCESS_VALIDATE_INPUT_NODE",
    "CHECK_EXISTING_ACCESS_NODE",
    "SUBMIT_ACCESS_REQUEST_NODE",
    "EVALUATE_ACCESS_REQUEST_NODE",
    "EVALUATE_REQUEST_REVIEW_NODE",
    "REVIEW_ACCESS_REQUEST_NODE",
    "RECORD_CRITIC_REVIEW_NODE",
    "FINALISE_REVIEWED_REQUEST_NODE",
    "ACCEPT_UNREVIEWED_DECISION_NODE",
    "EXECUTE_ACCESS_DECISION_NODE",
    "ACCESS_REQUEST_NODE_NAMES",
    "AccessRequestWorkflowError",
    "AccessRequestWorkflow",
    "build_access_request_graph",
    "build_access_request_workflow",
    "create_access_request_workflow",

    # Runtime
    "MemoryGovernanceReviewGateProtocol",
    "WorkflowCompileOptions",
    "MemoryGovernanceRuntime",
    "build_memory_governance_runtime",
    "create_memory_governance_runtime",
]