from __future__ import annotations

"""
LangGraph construction for the generic runtime memory-access request flow.

The workflow coordinates the node methods defined in
``access_request_nodes.py``. It contains no prompt logic, no LLM calls,
and no direct Service, Store, or persistence-entity access.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from langgraph.graph import END, START, StateGraph

from .access_request_nodes import (
    AccessRequestNodes,
    build_access_request_nodes,
)
from .access_request_state import (
    AccessRequestState,
)
from .dependencies import (
    AccessRequestDependencies,
)


VALIDATE_INPUT_NODE: Final[str] = (
    "validate_access_request_input"
)
CHECK_EXISTING_ACCESS_NODE: Final[str] = (
    "check_existing_access"
)
SUBMIT_ACCESS_REQUEST_NODE: Final[str] = (
    "submit_access_request"
)
EVALUATE_ACCESS_REQUEST_NODE: Final[str] = (
    "evaluate_access_request"
)
EVALUATE_REQUEST_REVIEW_NODE: Final[str] = (
    "evaluate_request_review"
)
REVIEW_ACCESS_REQUEST_NODE: Final[str] = (
    "review_access_request"
)
RECORD_CRITIC_REVIEW_NODE: Final[str] = (
    "record_critic_review"
)
FINALISE_REVIEWED_REQUEST_NODE: Final[str] = (
    "finalise_reviewed_request"
)
ACCEPT_UNREVIEWED_DECISION_NODE: Final[str] = (
    "accept_unreviewed_decision"
)
EXECUTE_ACCESS_DECISION_NODE: Final[str] = (
    "execute_access_decision"
)

ACCESS_REQUEST_NODE_NAMES: Final[
    tuple[str, ...]
] = (
    VALIDATE_INPUT_NODE,
    CHECK_EXISTING_ACCESS_NODE,
    SUBMIT_ACCESS_REQUEST_NODE,
    EVALUATE_ACCESS_REQUEST_NODE,
    EVALUATE_REQUEST_REVIEW_NODE,
    REVIEW_ACCESS_REQUEST_NODE,
    RECORD_CRITIC_REVIEW_NODE,
    FINALISE_REVIEWED_REQUEST_NODE,
    ACCEPT_UNREVIEWED_DECISION_NODE,
    EXECUTE_ACCESS_DECISION_NODE,
)


class AccessRequestWorkflowError(
    RuntimeError
):
    """
    Raised when the workflow wrapper receives invalid data or graph execution
    raises an exception.
    """


@dataclass(
    frozen=True,
    slots=True,
)
class AccessRequestWorkflow:
    """
    Compiled runtime access-request subgraph.

    ``graph`` is the compiled LangGraph runnable. ``nodes`` is retained for
    testing, inspection, and parent-graph integration.
    """

    graph: Any
    nodes: AccessRequestNodes

    def invoke(
        self,
        state: Mapping[str, Any],
        *,
        config: Mapping[str, Any] | None = None,
    ) -> AccessRequestState:
        """
        Execute the access-request subgraph synchronously.
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
            raise AccessRequestWorkflowError(
                "Access-request graph "
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
    ) -> AccessRequestState:
        """
        Execute the access-request subgraph asynchronously.
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
            raise AccessRequestWorkflowError(
                "Access-request graph "
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
            raise AccessRequestWorkflowError(
                "state must be a mapping."
            )

        return dict(state)

    @staticmethod
    def _normalise_result(
        result: Any,
    ) -> AccessRequestState:
        if not isinstance(
            result,
            Mapping,
        ):
            raise AccessRequestWorkflowError(
                "The compiled graph returned "
                "a non-mapping result."
            )

        return dict(result)


def build_access_request_graph(
    *,
    dependencies: AccessRequestDependencies,
) -> tuple[
    StateGraph,
    AccessRequestNodes,
]:
    """
    Construct the uncompiled access-request StateGraph.

    The returned builder can be embedded into a parent workflow or compiled
    directly by ``build_access_request_workflow``.
    """
    if not isinstance(
        dependencies,
        AccessRequestDependencies,
    ):
        raise TypeError(
            "dependencies must be an "
            "AccessRequestDependencies instance."
        )

    nodes = build_access_request_nodes(
        dependencies
    )
    builder = StateGraph(
        AccessRequestState
    )

    builder.add_node(
        VALIDATE_INPUT_NODE,
        nodes.validate_access_request_input,
    )
    builder.add_node(
        CHECK_EXISTING_ACCESS_NODE,
        nodes.check_existing_access,
    )
    builder.add_node(
        SUBMIT_ACCESS_REQUEST_NODE,
        nodes.submit_access_request,
    )
    builder.add_node(
        EVALUATE_ACCESS_REQUEST_NODE,
        nodes.evaluate_access_request,
    )
    builder.add_node(
        EVALUATE_REQUEST_REVIEW_NODE,
        nodes.evaluate_request_review,
    )
    builder.add_node(
        REVIEW_ACCESS_REQUEST_NODE,
        nodes.review_access_request,
    )
    builder.add_node(
        RECORD_CRITIC_REVIEW_NODE,
        nodes.record_critic_review,
    )
    builder.add_node(
        FINALISE_REVIEWED_REQUEST_NODE,
        nodes.finalise_reviewed_request,
    )
    builder.add_node(
        ACCEPT_UNREVIEWED_DECISION_NODE,
        nodes.accept_unreviewed_decision,
    )
    builder.add_node(
        EXECUTE_ACCESS_DECISION_NODE,
        nodes.execute_access_decision,
    )

    builder.add_edge(
        START,
        VALIDATE_INPUT_NODE,
    )

    builder.add_conditional_edges(
        VALIDATE_INPUT_NODE,
        nodes.route_after_validation,
        {
            "check_access": (
                CHECK_EXISTING_ACCESS_NODE
            ),
            "end": END,
        },
    )

    builder.add_conditional_edges(
        CHECK_EXISTING_ACCESS_NODE,
        nodes.route_after_access_check,
        {
            "submit": (
                SUBMIT_ACCESS_REQUEST_NODE
            ),
            "end": END,
        },
    )

    builder.add_conditional_edges(
        SUBMIT_ACCESS_REQUEST_NODE,
        nodes.route_after_submission,
        {
            "decide": (
                EVALUATE_ACCESS_REQUEST_NODE
            ),
            "end": END,
        },
    )

    builder.add_conditional_edges(
        EVALUATE_ACCESS_REQUEST_NODE,
        nodes.route_after_initial_decision,
        {
            "evaluate_review": (
                EVALUATE_REQUEST_REVIEW_NODE
            ),
            "end": END,
        },
    )

    builder.add_conditional_edges(
        EVALUATE_REQUEST_REVIEW_NODE,
        nodes.route_after_review_evaluation,
        {
            "review": (
                REVIEW_ACCESS_REQUEST_NODE
            ),
            "accept": (
                ACCEPT_UNREVIEWED_DECISION_NODE
            ),
            "end": END,
        },
    )

    builder.add_conditional_edges(
        REVIEW_ACCESS_REQUEST_NODE,
        nodes.route_after_critic_review,
        {
            "record_review": (
                RECORD_CRITIC_REVIEW_NODE
            ),
            "end": END,
        },
    )

    builder.add_conditional_edges(
        RECORD_CRITIC_REVIEW_NODE,
        nodes.route_after_review_recording,
        {
            "finalise": (
                FINALISE_REVIEWED_REQUEST_NODE
            ),
            "end": END,
        },
    )

    builder.add_conditional_edges(
        FINALISE_REVIEWED_REQUEST_NODE,
        nodes.route_after_finalisation,
        {
            "execute": (
                EXECUTE_ACCESS_DECISION_NODE
            ),
            "end": END,
        },
    )

    builder.add_conditional_edges(
        ACCEPT_UNREVIEWED_DECISION_NODE,
        nodes.route_after_finalisation,
        {
            "execute": (
                EXECUTE_ACCESS_DECISION_NODE
            ),
            "end": END,
        },
    )

    builder.add_edge(
        EXECUTE_ACCESS_DECISION_NODE,
        END,
    )

    return builder, nodes


def build_access_request_workflow(
    *,
    dependencies: AccessRequestDependencies,
    checkpointer: Any | None = None,
    interrupt_before: Sequence[str] | None = None,
    interrupt_after: Sequence[str] | None = None,
    debug: bool = False,
) -> AccessRequestWorkflow:
    """
    Build and compile the runtime access-request subgraph.

    ``checkpointer`` and interrupt options are passed directly to LangGraph.
    They are optional for short-lived isolated experiment runs.
    """
    builder, nodes = (
        build_access_request_graph(
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

    return AccessRequestWorkflow(
        graph=compiled,
        nodes=nodes,
    )


def create_access_request_workflow(
    *,
    dependencies: AccessRequestDependencies,
    checkpointer: Any | None = None,
    interrupt_before: Sequence[str] | None = None,
    interrupt_after: Sequence[str] | None = None,
    debug: bool = False,
) -> AccessRequestWorkflow:
    """
    Compatibility alias for build_access_request_workflow().
    """
    return build_access_request_workflow(
        dependencies=dependencies,
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
    )


__all__ = [
    "VALIDATE_INPUT_NODE",
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
]