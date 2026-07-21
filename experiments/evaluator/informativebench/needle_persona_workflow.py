from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, cast

from langgraph.graph import END, START, StateGraph

from .needle_persona_models import AGENT_PERSONA_MAP
from .needle_persona_nodes import (
    NeedlePersonaNodeConfig,
    NeedlePersonaNodeDependencies,
    NeedlePersonaNodeError,
    NeedlePersonaNodes,
)
from .needle_persona_state import (
    ExperimentMode,
    NeedlePersonaWorkflowState,
)


class NeedlePersonaWorkflowError(Exception):
    """
    Raised when graph construction, input validation, node execution, or final
    state validation fails.
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

        super().__init__(f"{prefix}{message}")
        self.node_name = node_name
        self.cause = cause


@dataclass(frozen=True)
class NeedlePersonaWorkflowConfig:
    """
    Workflow-level configuration.

    Node behaviour such as top-k values and routing strategy belongs in
    NeedlePersonaNodeConfig. This object controls graph compilation and input
    validation only.
    """

    validate_initial_state: bool = True
    validate_final_state: bool = True

    # A checkpointer is optional because the benchmark currently runs one
    # short-lived task at a time. Supply one when durable execution or resume
    # behaviour is required.
    checkpointer: Any | None = None


class NeedlePersonaWorkflow:
    """
    Compiled LangGraph workflow for one Needle in the Persona task.

    Graph structure:

        START
          |
        route_task
          |
          +-- private_only ----------------------+
          |                                      |
          +-- sharing --> collect_candidates     |
                              |                   |
                    +---------+---------+         |
                    |                   |         |
             ungoverned_shared    governed_shared|
                    |                   |         |
               direct_share      submit_requests |
                    |                   |         |
                    |             critic_review   |
                    |                   |         |
                    |          coordinator_decide |
                    |                   |         |
                    |           apply_governance  |
                    +---------+---------+---------+
                              |
                    retrieve_for_responder
                              |
                       generate_answer
                              |
                             END

    The workflow intentionally contains no evaluator node. Reference answers
    are added only after graph completion.
    """

    _FORBIDDEN_STATE_KEYS = {
        "gold_answer",
        "alternative_answers",
        "accepted_answers",
        "persona_sources",
        "raw_conversations",
        "raw_dialogues",
        "chat_bob_charlie",
        "needle_detail",
    }

    def __init__(
        self,
        *,
        nodes: NeedlePersonaNodes,
        config: NeedlePersonaWorkflowConfig | None = None,
    ) -> None:
        self.nodes = nodes
        self.config = (
            config
            or NeedlePersonaWorkflowConfig()
        )
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        builder = StateGraph(
            NeedlePersonaWorkflowState
        )

        builder.add_node(
            "route_task",
            self.nodes.route_task,
        )
        builder.add_node(
            "collect_candidate_memories",
            self.nodes.collect_candidate_memories,
        )
        builder.add_node(
            "direct_share",
            self.nodes.direct_share,
        )
        builder.add_node(
            "submit_promotion_requests",
            self.nodes.submit_promotion_requests,
        )
        builder.add_node(
            "critic_review",
            self.nodes.critic_review,
        )
        builder.add_node(
            "coordinator_decide",
            self.nodes.coordinator_decide,
        )
        builder.add_node(
            "apply_governance",
            self.nodes.apply_governance,
        )
        builder.add_node(
            "retrieve_for_responder",
            self.nodes.retrieve_for_responder,
        )
        builder.add_node(
            "generate_answer",
            self.nodes.generate_answer,
        )

        builder.add_edge(
            START,
            "route_task",
        )

        # Private-only skips contributor retrieval entirely. This preserves a
        # clean baseline in which only the responder accesses memory.
        builder.add_conditional_edges(
            "route_task",
            self._route_after_task_routing,
            {
                "private_only": (
                    "retrieve_for_responder"
                ),
                "sharing": (
                    "collect_candidate_memories"
                ),
            },
        )

        builder.add_conditional_edges(
            "collect_candidate_memories",
            self._route_sharing_mode,
            {
                "ungoverned_shared": "direct_share",
                "governed_shared": (
                    "submit_promotion_requests"
                ),
            },
        )

        builder.add_edge(
            "direct_share",
            "retrieve_for_responder",
        )

        builder.add_edge(
            "submit_promotion_requests",
            "critic_review",
        )
        builder.add_edge(
            "critic_review",
            "coordinator_decide",
        )
        builder.add_edge(
            "coordinator_decide",
            "apply_governance",
        )
        builder.add_edge(
            "apply_governance",
            "retrieve_for_responder",
        )

        builder.add_edge(
            "retrieve_for_responder",
            "generate_answer",
        )
        builder.add_edge(
            "generate_answer",
            END,
        )

        try:
            if self.config.checkpointer is None:
                return builder.compile()

            return builder.compile(
                checkpointer=self.config.checkpointer
            )
        except Exception as error:
            raise NeedlePersonaWorkflowError(
                f"Failed to compile workflow: {error}",
                cause=error,
            ) from error

    @staticmethod
    def _route_after_task_routing(
        state: NeedlePersonaWorkflowState,
    ) -> str:
        mode = NeedlePersonaNodes.select_experiment_branch(
            state
        )

        if mode == "private_only":
            return "private_only"

        return "sharing"

    @staticmethod
    def _route_sharing_mode(
        state: NeedlePersonaWorkflowState,
    ) -> ExperimentMode:
        mode = NeedlePersonaNodes.select_experiment_branch(
            state
        )

        if mode == "private_only":
            raise ValueError(
                "private_only must not enter the sharing branch."
            )

        return mode

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------

    def invoke(
        self,
        initial_state: NeedlePersonaWorkflowState,
        *,
        config: Mapping[str, Any] | None = None,
    ) -> NeedlePersonaWorkflowState:
        """
        Execute the graph once and return the completed state.
        """
        state = dict(initial_state)

        if self.config.validate_initial_state:
            self._validate_initial_state(state)

        try:
            result = self.graph.invoke(
                state,
                config=dict(config or {}),
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
            result,
        )

        if self.config.validate_final_state:
            self._validate_final_state(final_state)

        return final_state

    def stream(
        self,
        initial_state: NeedlePersonaWorkflowState,
        *,
        config: Mapping[str, Any] | None = None,
        stream_mode: str = "updates",
    ) -> Iterator[Any]:
        """
        Stream graph execution events.

        ``stream_mode="updates"`` is useful while debugging because each item
        contains only the state update returned by the node that just ran.
        """
        state = dict(initial_state)

        if self.config.validate_initial_state:
            self._validate_initial_state(state)

        try:
            yield from self.graph.stream(
                state,
                config=dict(config or {}),
                stream_mode=stream_mode,
            )
        except NeedlePersonaNodeError as error:
            raise NeedlePersonaWorkflowError(
                str(error),
                node_name=error.node_name,
                cause=error,
            ) from error
        except Exception as error:
            raise NeedlePersonaWorkflowError(
                f"Workflow streaming failed: {error}",
                cause=error,
            ) from error

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @classmethod
    def _validate_initial_state(
        cls,
        state: Mapping[str, Any],
    ) -> None:
        forbidden = (
            cls._FORBIDDEN_STATE_KEYS
            & set(state)
        )

        if forbidden:
            raise NeedlePersonaWorkflowError(
                "Initial state contains benchmark reference or "
                f"raw-source fields that must remain outside the "
                f"task graph: {sorted(forbidden)}."
            )

        for field_name in (
            "run_id",
            "sample_id",
            "question",
        ):
            value = str(
                state.get(field_name, "")
            ).strip()

            if not value:
                raise NeedlePersonaWorkflowError(
                    f"Initial state field "
                    f"{field_name!r} is required."
                )

        mode = state.get("experiment_mode")

        if mode not in {
            "private_only",
            "ungoverned_shared",
            "governed_shared",
        }:
            raise NeedlePersonaWorkflowError(
                "Initial state contains an invalid "
                f"experiment_mode: {mode!r}."
            )

        private_memory_ids = state.get(
            "private_memory_ids"
        )

        if not isinstance(
            private_memory_ids,
            Mapping,
        ):
            raise NeedlePersonaWorkflowError(
                "Initial state private_memory_ids must be "
                "a mapping of agent ID to memory IDs."
            )

        unknown_agent_ids = (
            set(private_memory_ids)
            - set(AGENT_PERSONA_MAP)
        )

        if unknown_agent_ids:
            raise NeedlePersonaWorkflowError(
                "Initial state private_memory_ids contains "
                f"unknown persona agents: "
                f"{sorted(unknown_agent_ids)}."
            )

        for agent_id in AGENT_PERSONA_MAP:
            memory_ids = private_memory_ids.get(
                agent_id,
                [],
            )

            if not isinstance(memory_ids, list):
                raise NeedlePersonaWorkflowError(
                    "private_memory_ids values must be lists; "
                    f"{agent_id!r} has "
                    f"{type(memory_ids).__name__}."
                )

            cleaned = [
                str(memory_id).strip()
                for memory_id in memory_ids
                if str(memory_id).strip()
            ]

            if len(cleaned) != len(set(cleaned)):
                raise NeedlePersonaWorkflowError(
                    f"Duplicate memory IDs found for "
                    f"{agent_id!r}."
                )

    @staticmethod
    def _validate_final_state(
        state: NeedlePersonaWorkflowState,
    ) -> None:
        if state.get("workflow_status") != "answered":
            raise NeedlePersonaWorkflowError(
                "Workflow ended without reaching answered "
                f"status: "
                f"{state.get('workflow_status')!r}."
            )

        if state.get("agent_answer") is None:
            raise NeedlePersonaWorkflowError(
                "Workflow ended without agent_answer."
            )

        responder_id = str(
            state.get("responder_agent_id", "")
        ).strip()

        if responder_id not in AGENT_PERSONA_MAP:
            raise NeedlePersonaWorkflowError(
                "Workflow ended with an invalid "
                f"responder_agent_id: "
                f"{responder_id!r}."
            )

        selected = set(
            state.get("selected_agent_ids", [])
        )

        if responder_id not in selected:
            raise NeedlePersonaWorkflowError(
                "Final responder is not present in "
                "selected_agent_ids."
            )

        mode = state.get("experiment_mode")
        shared_ids = state.get(
            "shared_memory_ids",
            [],
        )

        if (
            mode == "private_only"
            and shared_ids
        ):
            raise NeedlePersonaWorkflowError(
                "private_only workflow unexpectedly produced "
                "shared_memory_ids."
            )

        retrieved_ids = set(
            state.get(
                "responder_retrieved_memory_ids",
                [],
            )
        )
        used_ids = set(
            state["agent_answer"].used_memory_ids
        )

        unavailable_used_ids = (
            used_ids - retrieved_ids
        )

        if unavailable_used_ids:
            raise NeedlePersonaWorkflowError(
                "AgentAnswer references memories that were not "
                "retrieved for the responder: "
                f"{sorted(unavailable_used_ids)}."
            )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def draw_mermaid(self) -> str:
        """
        Return a Mermaid representation of the compiled graph.
        """
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception as error:
            raise NeedlePersonaWorkflowError(
                f"Failed to render Mermaid graph: {error}",
                cause=error,
            ) from error


def build_needle_persona_workflow(
    *,
    nodes: NeedlePersonaNodes,
    workflow_config: NeedlePersonaWorkflowConfig | None = None,
) -> NeedlePersonaWorkflow:
    """
    Build a workflow from an already configured NeedlePersonaNodes object.
    """
    return NeedlePersonaWorkflow(
        nodes=nodes,
        config=workflow_config,
    )


def create_needle_persona_workflow(
    *,
    dependencies: NeedlePersonaNodeDependencies,
    node_config: NeedlePersonaNodeConfig | None = None,
    workflow_config: NeedlePersonaWorkflowConfig | None = None,
) -> NeedlePersonaWorkflow:
    """
    Convenience factory that creates both the node collection and workflow.
    """
    nodes = NeedlePersonaNodes(
        dependencies,
        config=node_config,
    )

    return build_needle_persona_workflow(
        nodes=nodes,
        workflow_config=workflow_config,
    )


__all__ = [
    "NeedlePersonaWorkflowError",
    "NeedlePersonaWorkflowConfig",
    "NeedlePersonaWorkflow",
    "build_needle_persona_workflow",
    "create_needle_persona_workflow",
]