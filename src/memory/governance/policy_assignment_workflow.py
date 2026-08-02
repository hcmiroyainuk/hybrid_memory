from __future__ import annotations

"""
LangGraph construction for the generic initial memory-policy assignment flow.

The workflow coordinates the node methods defined in
``policy_assignment_nodes.py``. It contains no prompt logic, no LLM calls,
and no direct Service or Store access.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from langgraph.graph import END, START, StateGraph

from .dependencies import PolicyAssignmentDependencies
from .policy_assignment_nodes import (
    PolicyAssignmentNodes,
    build_policy_assignment_nodes,
)
from .policy_assignment_state import (
    PolicyAssignmentState,
)


VALIDATE_INPUT_NODE: Final[str] = (
    "validate_policy_assignment_input"
)
ASSIGN_INITIAL_POLICY_NODE: Final[str] = (
    "assign_initial_policy"
)
EVALUATE_REVIEW_NODE: Final[str] = (
    "evaluate_policy_review"
)
REVIEW_INITIAL_POLICY_NODE: Final[str] = (
    "review_initial_policy"
)
ACCEPT_UNREVIEWED_POLICY_NODE: Final[str] = (
    "accept_unreviewed_policy"
)
FINALISE_REVIEWED_POLICY_NODE: Final[str] = (
    "finalise_reviewed_policy"
)
APPLY_ACCESS_POLICY_NODE: Final[str] = (
    "apply_access_policy"
)

POLICY_ASSIGNMENT_NODE_NAMES: Final[
    tuple[str, ...]
] = (
    VALIDATE_INPUT_NODE,
    ASSIGN_INITIAL_POLICY_NODE,
    EVALUATE_REVIEW_NODE,
    REVIEW_INITIAL_POLICY_NODE,
    ACCEPT_UNREVIEWED_POLICY_NODE,
    FINALISE_REVIEWED_POLICY_NODE,
    APPLY_ACCESS_POLICY_NODE,
)


class PolicyAssignmentWorkflowError(
    RuntimeError
):
    """
    Raised when the workflow wrapper receives invalid invocation data.
    """


@dataclass(
    frozen=True,
    slots=True,
)
class PolicyAssignmentWorkflow:
    """
    Compiled initial-policy assignment subgraph.

    ``graph`` is the compiled LangGraph runnable. ``nodes`` is retained for
    testing, inspection, and parent-graph integration.
    """

    graph: Any
    nodes: PolicyAssignmentNodes

    def invoke(
        self,
        state: Mapping[str, Any],
        *,
        config: Mapping[str, Any] | None = None,
    ) -> PolicyAssignmentState:
        """
        Execute the subgraph synchronously and return its terminal state.
        """
        input_state = self._copy_input_state(
            state
        )

        try:
            result = self.graph.invoke(
                input_state,
                config=config,
            )
        except Exception as error:
            raise PolicyAssignmentWorkflowError(
                "Policy-assignment graph "
                f"execution failed: {error}"
            ) from error

        return self._normalise_result(
            result
        )

    async def ainvoke(
        self,
        state: Mapping[str, Any],
        *,
        config: Mapping[str, Any] | None = None,
    ) -> PolicyAssignmentState:
        """
        Execute the subgraph asynchronously and return its terminal state.
        """
        input_state = self._copy_input_state(
            state
        )

        try:
            result = await self.graph.ainvoke(
                input_state,
                config=config,
            )
        except Exception as error:
            raise PolicyAssignmentWorkflowError(
                "Policy-assignment graph "
                f"execution failed: {error}"
            ) from error

        return self._normalise_result(
            result
        )

    @staticmethod
    def _copy_input_state(
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(
            state,
            Mapping,
        ):
            raise PolicyAssignmentWorkflowError(
                "state must be a mapping."
            )

        return dict(state)

    @staticmethod
    def _normalise_result(
        result: Any,
    ) -> PolicyAssignmentState:
        if not isinstance(
            result,
            Mapping,
        ):
            raise PolicyAssignmentWorkflowError(
                "The compiled graph returned "
                "a non-mapping result."
            )

        return dict(result)


def build_policy_assignment_graph(
    *,
    dependencies: PolicyAssignmentDependencies,
) -> tuple[
    StateGraph,
    PolicyAssignmentNodes,
]:
    """
    Construct the uncompiled policy-assignment StateGraph.

    Returning the node collection alongside the builder makes the topology
    testable without duplicating dependency construction.
    """
    if not isinstance(
        dependencies,
        PolicyAssignmentDependencies,
    ):
        raise TypeError(
            "dependencies must be a "
            "PolicyAssignmentDependencies instance."
        )

    nodes = build_policy_assignment_nodes(
        dependencies
    )
    builder = StateGraph(
        PolicyAssignmentState
    )

    builder.add_node(
        VALIDATE_INPUT_NODE,
        nodes.validate_policy_assignment_input,
    )
    builder.add_node(
        ASSIGN_INITIAL_POLICY_NODE,
        nodes.assign_initial_policy,
    )
    builder.add_node(
        EVALUATE_REVIEW_NODE,
        nodes.evaluate_policy_review,
    )
    builder.add_node(
        REVIEW_INITIAL_POLICY_NODE,
        nodes.review_initial_policy,
    )
    builder.add_node(
        ACCEPT_UNREVIEWED_POLICY_NODE,
        nodes.accept_unreviewed_policy,
    )
    builder.add_node(
        FINALISE_REVIEWED_POLICY_NODE,
        nodes.finalise_reviewed_policy,
    )
    builder.add_node(
        APPLY_ACCESS_POLICY_NODE,
        nodes.apply_access_policy,
    )

    builder.add_edge(
        START,
        VALIDATE_INPUT_NODE,
    )

    builder.add_conditional_edges(
        VALIDATE_INPUT_NODE,
        nodes.route_after_validation,
        {
            "assign": (
                ASSIGN_INITIAL_POLICY_NODE
            ),
            "end": END,
        },
    )

    builder.add_conditional_edges(
        ASSIGN_INITIAL_POLICY_NODE,
        nodes.route_after_policy_proposal,
        {
            "evaluate_review": (
                EVALUATE_REVIEW_NODE
            ),
            "end": END,
        },
    )

    builder.add_conditional_edges(
        EVALUATE_REVIEW_NODE,
        nodes.route_after_review_evaluation,
        {
            "review": (
                REVIEW_INITIAL_POLICY_NODE
            ),
            "accept": (
                ACCEPT_UNREVIEWED_POLICY_NODE
            ),
            "end": END,
        },
    )

    builder.add_conditional_edges(
        REVIEW_INITIAL_POLICY_NODE,
        nodes.route_after_review,
        {
            "finalise": (
                FINALISE_REVIEWED_POLICY_NODE
            ),
            "end": END,
        },
    )

    builder.add_conditional_edges(
        ACCEPT_UNREVIEWED_POLICY_NODE,
        nodes.route_after_finalisation,
        {
            "apply": APPLY_ACCESS_POLICY_NODE,
            "end": END,
        },
    )

    builder.add_conditional_edges(
        FINALISE_REVIEWED_POLICY_NODE,
        nodes.route_after_finalisation,
        {
            "apply": APPLY_ACCESS_POLICY_NODE,
            "end": END,
        },
    )

    builder.add_edge(
        APPLY_ACCESS_POLICY_NODE,
        END,
    )

    return builder, nodes


def build_policy_assignment_workflow(
    *,
    dependencies: PolicyAssignmentDependencies,
    checkpointer: Any | None = None,
    interrupt_before: Sequence[str] | None = None,
    interrupt_after: Sequence[str] | None = None,
    debug: bool = False,
) -> PolicyAssignmentWorkflow:
    """
    Build and compile the initial memory-policy assignment subgraph.

    ``checkpointer`` and interrupt options are passed directly to LangGraph.
    They are optional because short-lived experiment runs may not require
    durable execution.
    """
    builder, nodes = (
        build_policy_assignment_graph(
            dependencies=dependencies
        )
    )

    compile_kwargs: dict[str, Any] = {
        "debug": bool(debug),
    }

    if checkpointer is not None:
        compile_kwargs["checkpointer"] = (
            checkpointer
        )

    if interrupt_before is not None:
        compile_kwargs["interrupt_before"] = (
            list(interrupt_before)
        )

    if interrupt_after is not None:
        compile_kwargs["interrupt_after"] = (
            list(interrupt_after)
        )

    compiled = builder.compile(
        **compile_kwargs
    )

    return PolicyAssignmentWorkflow(
        graph=compiled,
        nodes=nodes,
    )


def create_policy_assignment_workflow(
    *,
    dependencies: PolicyAssignmentDependencies,
    checkpointer: Any | None = None,
    interrupt_before: Sequence[str] | None = None,
    interrupt_after: Sequence[str] | None = None,
    debug: bool = False,
) -> PolicyAssignmentWorkflow:
    """
    Compatibility alias for build_policy_assignment_workflow().
    """
    return build_policy_assignment_workflow(
        dependencies=dependencies,
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
    )


__all__ = [
    "VALIDATE_INPUT_NODE",
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
]