from __future__ import annotations

from collections.abc import Mapping
from typing import (
    Any,
    cast,
)

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from ..config.needle_persona_config import (
    NeedlePersonaWorkflowConfig,
    validate_experiment_mode,
)
from ..retrieval.needle_persona_retrieval import (
    NeedlePersonaRetrievalService,
)
from .dependencies.needle_persona_dependencies import (
    NeedlePersonaNodeDependencies,
)
from .nodes.needle_persona_nodes import (
    NeedlePersonaNodeError,
    NeedlePersonaNodes,
)
from .state.needle_persona_state import (
    NeedlePersonaWorkflowState,
    is_terminal_workflow_status,
)


class NeedlePersonaWorkflowError(Exception):
    """
    Raised when workflow construction, execution, or validation fails.
    """

    def __init__(
        self,
        message: str,
        *,
        node_name: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        prefix = (
            f"[{node_name}] "
            if node_name
            else ""
        )
        super().__init__(
            prefix + message
        )
        self.node_name = node_name
        self.cause = cause


class NeedlePersonaWorkflow:
    """
    Thin runtime wrapper around the compiled Needle Persona StateGraph.
    """

    def __init__(
        self,
        *,
        graph: Any,
        config: NeedlePersonaWorkflowConfig,
        nodes: NeedlePersonaNodes,
    ) -> None:
        self.graph = graph
        self.config = config
        self.nodes = nodes

    def invoke(
        self,
        initial_state: NeedlePersonaWorkflowState,
        *,
        config: Mapping[str, Any] | None = None,
    ) -> NeedlePersonaWorkflowState:
        """
        Execute one ``sample × mode`` workflow run.
        """
        if self.config.validate_initial_state:
            self._validate_initial_state(
                initial_state
            )

        try:
            result = self.graph.invoke(
                dict(initial_state),
                config=dict(
                    config or {}
                ),
            )
        except NeedlePersonaNodeError as error:
            raise NeedlePersonaWorkflowError(
                str(error),
                node_name=error.node_name,
                cause=error,
            ) from error
        except Exception as error:
            raise NeedlePersonaWorkflowError(
                f"Workflow execution failed: {error}",
                cause=error,
            ) from error

        final_state = cast(
            NeedlePersonaWorkflowState,
            dict(result),
        )

        if self.config.validate_final_state:
            self._validate_final_state(
                final_state
            )

        return final_state

    @staticmethod
    def _validate_initial_state(
        state: Mapping[str, Any],
    ) -> None:
        required_text_fields = (
            "run_id",
            "sample_id",
            "question",
        )

        for field_name in required_text_fields:
            value = str(
                state.get(
                    field_name,
                    "",
                )
            ).strip()

            if not value:
                raise NeedlePersonaWorkflowError(
                    "Initial state field "
                    f"{field_name!r} cannot be empty."
                )

        validate_experiment_mode(
            str(
                state.get(
                    "experiment_mode",
                    "",
                )
            )
        )

        if (
            state.get(
                "workflow_status"
            )
            != "initialized"
        ):
            raise NeedlePersonaWorkflowError(
                "Initial workflow_status must be "
                "'initialized'."
            )

        private_memory_ids = state.get(
            "private_memory_ids"
        )

        if not isinstance(
            private_memory_ids,
            Mapping,
        ):
            raise NeedlePersonaWorkflowError(
                "Initial private_memory_ids must "
                "be a mapping."
            )

        forbidden_fields = {
            "gold_answer",
            "alternative_answers",
            "persona_sources",
            "needle_detail",
        }

        leaked_fields = (
            forbidden_fields
            & set(state)
        )

        if leaked_fields:
            raise NeedlePersonaWorkflowError(
                "Held-out evaluation data leaked "
                "into workflow state: "
                f"{sorted(leaked_fields)}."
            )

    @staticmethod
    def _validate_final_state(
        state: Mapping[str, Any],
    ) -> None:
        status = str(
            state.get(
                "workflow_status",
                "",
            )
        ).strip()

        if not is_terminal_workflow_status(
            status
        ):
            raise NeedlePersonaWorkflowError(
                "Workflow finished without a "
                "terminal workflow_status; received "
                f"{status!r}."
            )

        final_answer = state.get(
            "final_answer"
        )

        if final_answer is None:
            raise NeedlePersonaWorkflowError(
                "Workflow finished without "
                "final_answer."
            )

        if final_answer.status != status:
            raise NeedlePersonaWorkflowError(
                "final_answer.status does not match "
                "workflow_status: "
                f"{final_answer.status!r} != "
                f"{status!r}."
            )

        responder_agent_id = str(
            state.get(
                "responder_agent_id",
                "",
            )
        ).strip()

        if not responder_agent_id:
            raise NeedlePersonaWorkflowError(
                "Workflow finished without a "
                "responder_agent_id."
            )

        if responder_agent_id not in state.get(
            "selected_agent_ids",
            [],
        ):
            raise NeedlePersonaWorkflowError(
                "The responder must appear in "
                "selected_agent_ids."
            )

        if (
            state.get(
                "experiment_mode"
            )
            == "private_only"
        ):
            if state.get(
                "access_requests"
            ):
                raise NeedlePersonaWorkflowError(
                    "private_only produced access "
                    "requests."
                )

            if state.get(
                "approved_memory_ids"
            ):
                raise NeedlePersonaWorkflowError(
                    "private_only approved shared "
                    "memory access."
                )


def build_needle_persona_workflow(
    *,
    dependencies: NeedlePersonaNodeDependencies,
    workflow_config: (
        NeedlePersonaWorkflowConfig | None
    ) = None,
    retrieval_service: (
        NeedlePersonaRetrievalService | None
    ) = None,
) -> NeedlePersonaWorkflow:
    """
    Construct and compile the Needle Persona LangGraph workflow.

    Graph:

        START
          -> route_task
          -> retrieve_initial_memory
          -> assess_initial_evidence
          -> finish
             or discover -> submit
                -> direct approval
                or critic review -> coordinator decision
          -> retrieve_final_memory
          -> generate_final_answer
          -> END
    """
    config = (
        workflow_config
        or NeedlePersonaWorkflowConfig()
    )

    nodes = NeedlePersonaNodes(
        dependencies=dependencies,
        config=config.node_config,
        retrieval_service=(
            retrieval_service
        ),
    )

    builder = StateGraph(
        NeedlePersonaWorkflowState
    )

    builder.add_node(
        "route_task",
        nodes.route_task,
    )
    builder.add_node(
        "retrieve_initial_memory",
        nodes.retrieve_initial_memory,
    )
    builder.add_node(
        "assess_initial_evidence",
        nodes.assess_initial_evidence,
    )
    builder.add_node(
        "discover_memory_candidates",
        nodes.discover_memory_candidates,
    )
    builder.add_node(
        "submit_access_requests",
        nodes.submit_access_requests,
    )
    builder.add_node(
        "review_access_requests",
        nodes.review_access_requests,
    )
    builder.add_node(
        "decide_access_requests",
        nodes.decide_access_requests,
    )
    builder.add_node(
        "approve_direct_requests",
        nodes.approve_direct_requests,
    )
    builder.add_node(
        "retrieve_final_memory",
        nodes.retrieve_final_memory,
    )
    builder.add_node(
        "generate_final_answer",
        nodes.generate_final_answer,
    )
    builder.add_node(
        "finalise_initial_answer",
        nodes.finalise_initial_answer,
    )

    builder.add_edge(
        START,
        "route_task",
    )
    builder.add_edge(
        "route_task",
        "retrieve_initial_memory",
    )
    builder.add_edge(
        "retrieve_initial_memory",
        "assess_initial_evidence",
    )

    builder.add_conditional_edges(
        "assess_initial_evidence",
        nodes.route_after_initial_assessment,
        {
            "finish": (
                "finalise_initial_answer"
            ),
            "discover": (
                "discover_memory_candidates"
            ),
        },
    )

    builder.add_conditional_edges(
        "discover_memory_candidates",
        nodes.route_after_candidate_discovery,
        {
            "finish": (
                "finalise_initial_answer"
            ),
            "submit": (
                "submit_access_requests"
            ),
        },
    )

    builder.add_conditional_edges(
        "submit_access_requests",
        nodes.route_after_request_submission,
        {
            "finish": (
                "finalise_initial_answer"
            ),
            "retrieve": (
                "retrieve_final_memory"
            ),
            "ungoverned": (
                "approve_direct_requests"
            ),
            "governed": (
                "review_access_requests"
            ),
        },
    )

    builder.add_edge(
        "review_access_requests",
        "decide_access_requests",
    )

    builder.add_conditional_edges(
        "decide_access_requests",
        nodes.route_after_access_decision,
        {
            "finish": (
                "finalise_initial_answer"
            ),
            "retrieve": (
                "retrieve_final_memory"
            ),
        },
    )

    builder.add_conditional_edges(
        "approve_direct_requests",
        nodes.route_after_access_decision,
        {
            "finish": (
                "finalise_initial_answer"
            ),
            "retrieve": (
                "retrieve_final_memory"
            ),
        },
    )

    builder.add_edge(
        "retrieve_final_memory",
        "generate_final_answer",
    )
    builder.add_edge(
        "generate_final_answer",
        END,
    )
    builder.add_edge(
        "finalise_initial_answer",
        END,
    )

    if config.checkpointer is None:
        compiled = builder.compile()
    else:
        compiled = builder.compile(
            checkpointer=config.checkpointer
        )

    return NeedlePersonaWorkflow(
        graph=compiled,
        config=config,
        nodes=nodes,
    )


def create_needle_persona_workflow(
    *,
    dependencies: NeedlePersonaNodeDependencies,
    workflow_config: (
        NeedlePersonaWorkflowConfig | None
    ) = None,
    retrieval_service: (
        NeedlePersonaRetrievalService | None
    ) = None,
) -> NeedlePersonaWorkflow:
    """
    Compatibility alias for build_needle_persona_workflow().
    """
    return build_needle_persona_workflow(
        dependencies=dependencies,
        workflow_config=workflow_config,
        retrieval_service=(
            retrieval_service
        ),
    )


__all__ = [
    "NeedlePersonaWorkflowError",
    "NeedlePersonaWorkflow",
    "build_needle_persona_workflow",
    "create_needle_persona_workflow",
]