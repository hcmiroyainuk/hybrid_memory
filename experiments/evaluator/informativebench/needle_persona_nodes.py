from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, Protocol, runtime_checkable

from src.llm import (
    AgentAnswer,
    CoordinatorPromptTemplate,
    CriticPromptTemplate,
    LLMClient,
    MemoryReviewOutput,
    PromotionDecisionOutput,
    PromptTemplates,
    TaskRoutingOutput,
    WorkerPromptTemplate,
)
from src.memory.entities import (
    Agent,
    AgentRole,
)

from .needle_persona_ingestion import (
    build_default_worker_prompts,
)
from .needle_persona_models import (
    AGENT_PERSONA_MAP,
    PERSONA_AGENT_MAP,
    PERSONA_NAMES,
)
from .needle_persona_state import (
    ExperimentMode,
    NeedlePersonaWorkflowState,
)


RoutingStrategy = Literal["deterministic", "llm"]


@runtime_checkable
class NodeMemoryServiceProtocol(Protocol):
    """
    MemoryService operations required by task-workflow nodes.
    """

    def get_memory(
        self,
        agent: Agent,
        memory_id: str,
    ) -> Any:
        ...

    def retrieve_memories(
        self,
        agent: Agent,
        query: str,
        top_k: int = 5,
        include_private: bool = True,
        include_shared: bool = True,
    ) -> list[Any]:
        ...

    def list_accessible_memories(
        self,
        agent: Agent,
        include_private: bool = True,
        include_shared: bool = True,
        active_only: bool = True,
    ) -> list[Any]:
        ...


@runtime_checkable
class NodePromotionServiceProtocol(Protocol):
    """
    PromotionService operations required by governed and baseline sharing.

    ``memory_store`` is used only by the deliberately ungoverned baseline.
    The governed path uses the public promotion-service methods.
    """

    memory_store: Any

    def submit_promotion_request(
        self,
        agent: Agent,
        memory_id: str,
        reason: str,
    ) -> Any:
        ...

    def review_promotion_request(
        self,
        agent: Agent,
        request_id: str,
        comment: str,
    ) -> Any:
        ...

    def approve_promotion_request(
        self,
        agent: Agent,
        request_id: str,
        readable_by: list[str],
        comment: str | None = None,
    ) -> Any:
        ...

    def reject_promotion_request(
        self,
        agent: Agent,
        request_id: str,
        comment: str | None = None,
    ) -> Any:
        ...

    def get_request(
        self,
        request_id: str,
    ) -> Any:
        ...


class NeedlePersonaNodeError(Exception):
    """
    Raised when one workflow node cannot complete safely.
    """

    def __init__(
        self,
        node_name: str,
        message: str,
        *,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(f"{node_name}: {message}")
        self.node_name = node_name
        self.cause = cause


@dataclass(frozen=True)
class NeedlePersonaNodeConfig:
    """
    Behavioural configuration shared by the task nodes.
    """

    routing_strategy: RoutingStrategy = "deterministic"

    contributor_top_k: int = 5
    responder_top_k: int = 8
    existing_context_limit: int = 8

    # When False, a missing/unconfigured semantic retriever is treated as an
    # error. Enable only for smoke tests that should fall back to simple
    # lexical ranking over the run's known memory IDs.
    allow_lexical_retrieval_fallback: bool = False

    # Remove hallucinated memory/source/agent references from AgentAnswer.
    sanitise_answer_references: bool = True

    # The ungoverned baseline intentionally bypasses Critic and Coordinator
    # review, but still restricts the resulting shared memory to the source
    # owner and the responder rather than exposing it globally.
    ungoverned_task_scoped_acl: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "contributor_top_k",
            "responder_top_k",
            "existing_context_limit",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(
                    f"{field_name} must be greater than zero."
                )


@dataclass(frozen=True)
class NeedlePersonaNodeDependencies:
    """
    Concrete services and agents used by the LangGraph nodes.
    """

    llm_client: LLMClient
    memory_service: NodeMemoryServiceProtocol
    promotion_service: NodePromotionServiceProtocol

    persona_agents: Mapping[str, Agent]
    critic_agent: Agent
    coordinator_agent: Agent

    worker_prompts: Mapping[
        str,
        WorkerPromptTemplate,
    ] | None = None
    critic_prompt: CriticPromptTemplate | None = None
    coordinator_prompt: CoordinatorPromptTemplate | None = None


def build_default_governance_agents() -> tuple[Agent, Agent]:
    """
    Build the Critic and Coordinator agents used by the benchmark workflow.
    """
    return (
        Agent.critic(agent_id="critic_agent"),
        Agent.coordinator(agent_id="coordinator_agent"),
    )


class NeedlePersonaNodes:
    """
    Node collection for the Needle in the Persona LangGraph workflow.

    Intended graph sequence:

        route_task
            -> collect_candidate_memories
            -> branch by experiment_mode

        private_only:
            retrieve_for_responder
            -> generate_answer

        ungoverned_shared:
            direct_share
            -> retrieve_for_responder
            -> generate_answer

        governed_shared:
            submit_promotion_requests
            -> critic_review
            -> coordinator_decide
            -> apply_governance
            -> retrieve_for_responder
            -> generate_answer

    No node accepts raw persona conversations or reference answers.
    """

    def __init__(
        self,
        dependencies: NeedlePersonaNodeDependencies,
        *,
        config: NeedlePersonaNodeConfig | None = None,
    ) -> None:
        self.deps = dependencies
        self.config = config or NeedlePersonaNodeConfig()

        self.persona_agents = dict(
            dependencies.persona_agents
        )
        self.worker_prompts = dict(
            dependencies.worker_prompts
            or build_default_worker_prompts()
        )
        self.critic_prompt = (
            dependencies.critic_prompt
            or PromptTemplates.critic(
                critic_id=dependencies.critic_agent.agent_id
            )
        )
        self.coordinator_prompt = (
            dependencies.coordinator_prompt
            or PromptTemplates.coordinator(
                coordinator_id=(
                    dependencies.coordinator_agent.agent_id
                )
            )
        )

        self._validate_dependencies()

    # ------------------------------------------------------------------
    # 1. Routing
    # ------------------------------------------------------------------

    def route_task(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> dict[str, Any]:
        node_name = "route_task"
        started_at = perf_counter()

        try:
            question = self._required_state_text(
                state,
                "question",
            )

            if self.config.routing_strategy == "llm":
                routing = self._route_with_llm(
                    question=question,
                    state=state,
                )
            else:
                routing = self._route_deterministically(
                    question
                )

            selected = self._clean_agent_ids(
                routing.selected_agent_ids,
                allowed=set(self.persona_agents),
            )

            responder = routing.responder_agent_id.strip()

            if responder not in selected:
                raise ValueError(
                    "Routing responder is not included in "
                    "selected_agent_ids."
                )

            contributors = [
                agent_id
                for agent_id in selected
                if agent_id != responder
            ]

            return {
                "routing_output": routing,
                "selected_agent_ids": selected,
                "responder_agent_id": responder,
                "contributor_agent_ids": contributors,
                "workflow_status": "routed",
                "node_trace": [node_name],
                "node_metrics": {
                    f"{node_name}_ms": self._elapsed_ms(
                        started_at
                    )
                },
            }
        except Exception as error:
            raise self._node_error(
                node_name,
                error,
            ) from error

    def _route_deterministically(
        self,
        question: str,
    ) -> TaskRoutingOutput:
        lowered = question.lower()
        mentions: list[tuple[int, str, str]] = []

        for persona in PERSONA_NAMES:
            match = re.search(
                rf"\b{re.escape(persona)}\b",
                lowered,
            )

            if match:
                mentions.append(
                    (
                        match.start(),
                        persona,
                        PERSONA_AGENT_MAP[persona],
                    )
                )

        mentions.sort(key=lambda item: item[0])

        if mentions:
            selected = [
                agent_id
                for _, _, agent_id in mentions
            ]
            mentioned_personas = [
                persona
                for _, persona, _ in mentions
            ]
            reason = (
                "Selected the persona agents explicitly named in "
                "the task, preserving mention order."
            )
        else:
            selected = [
                PERSONA_AGENT_MAP[persona]
                for persona in PERSONA_NAMES
            ]
            mentioned_personas = list(PERSONA_NAMES)
            reason = (
                "No persona name was explicit, so all persona "
                "agents were selected as a safe routing fallback."
            )

        responder = selected[0]

        return TaskRoutingOutput(
            selected_agent_ids=selected,
            responder_agent_id=responder,
            required_information=[
                (
                    f"Relevant task information owned by "
                    f"{persona.title()}."
                )
                for persona in mentioned_personas
            ],
            reason=reason,
            confidence=1.0 if mentions else 0.5,
        )

    def _route_with_llm(
        self,
        *,
        question: str,
        state: NeedlePersonaWorkflowState,
    ) -> TaskRoutingOutput:
        available_agents = {
            agent_id: {
                "persona": AGENT_PERSONA_MAP[agent_id],
                "role": "persona_worker",
            }
            for agent_id in self.persona_agents
        }

        prompt = self.coordinator_prompt.build_task_routing_prompt(
            task=question,
            available_agents=available_agents,
            task_context=None,
            context_sections={
                "Run ID": state.get("run_id"),
                "Sample ID": state.get("sample_id"),
                "Routing restriction": (
                    "Select only persona worker IDs listed in "
                    "Available agents."
                ),
            },
        )

        routing = self.deps.llm_client.invoke_structured(
            prompt,
            TaskRoutingOutput,
        )

        unknown = (
            set(routing.selected_agent_ids)
            - set(self.persona_agents)
        )

        if unknown:
            raise ValueError(
                "LLM routing selected unknown persona agents: "
                f"{sorted(unknown)}."
            )

        return routing

    # ------------------------------------------------------------------
    # 2. Contributor retrieval and candidate collection
    # ------------------------------------------------------------------

    def collect_candidate_memories(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> dict[str, Any]:
        node_name = "collect_candidate_memories"
        started_at = perf_counter()

        try:
            question = self._required_state_text(
                state,
                "question",
            )
            contributors = list(
                state.get("contributor_agent_ids", [])
            )
            private_memory_ids = state.get(
                "private_memory_ids",
                {},
            )

            retrieved_by_agent: dict[str, list[str]] = {}
            candidate_ids: list[str] = []
            owner_by_memory_id: dict[str, str] = {}
            warnings: list[str] = []

            for agent_id in contributors:
                agent = self._persona_agent(agent_id)
                allowed_ids = set(
                    private_memory_ids.get(agent_id, [])
                )

                memories, retrieval_warning = (
                    self._retrieve_and_filter(
                        agent=agent,
                        query=question,
                        top_k=self.config.contributor_top_k,
                        include_private=True,
                        include_shared=False,
                        allowed_private_ids=allowed_ids,
                        allowed_shared_ids=set(),
                    )
                )

                if retrieval_warning:
                    warnings.append(retrieval_warning)

                memory_ids = [
                    self._memory_id(memory)
                    for memory in memories
                ]

                retrieved_by_agent[agent_id] = memory_ids

                for memory_id in memory_ids:
                    if memory_id not in candidate_ids:
                        candidate_ids.append(memory_id)
                    owner_by_memory_id[memory_id] = agent_id

            return {
                "contributor_retrieved_memory_ids": (
                    retrieved_by_agent
                ),
                "candidate_memory_ids": candidate_ids,
                "candidate_owner_by_memory_id": (
                    owner_by_memory_id
                ),
                "workflow_status": "candidates_collected",
                "node_trace": [node_name],
                "warnings": warnings,
                "node_metrics": {
                    f"{node_name}_ms": self._elapsed_ms(
                        started_at
                    )
                },
            }
        except Exception as error:
            raise self._node_error(
                node_name,
                error,
            ) from error

    # ------------------------------------------------------------------
    # 3A. Governed path: submit promotion requests
    # ------------------------------------------------------------------

    def submit_promotion_requests(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> dict[str, Any]:
        node_name = "submit_promotion_requests"
        started_at = perf_counter()

        try:
            self._require_mode(
                state,
                "governed_shared",
            )

            request_ids: dict[str, str] = {}

            for memory_id in state.get(
                "candidate_memory_ids",
                [],
            ):
                owner_id = self._candidate_owner(
                    state,
                    memory_id,
                )
                owner_agent = self._persona_agent(owner_id)

                request = (
                    self.deps.promotion_service
                    .submit_promotion_request(
                        agent=owner_agent,
                        memory_id=memory_id,
                        reason=(
                            "The contributor retrieved this private "
                            "memory as relevant to the active Needle "
                            "persona task and requests controlled "
                            "sharing with the responder."
                        ),
                    )
                )

                request_ids[memory_id] = (
                    self._request_id(request)
                )

            return {
                "promotion_request_ids": request_ids,
                "workflow_status": "promotion_submitted",
                "node_trace": [node_name],
                "node_metrics": {
                    f"{node_name}_ms": self._elapsed_ms(
                        started_at
                    )
                },
            }
        except Exception as error:
            raise self._node_error(
                node_name,
                error,
            ) from error

    # ------------------------------------------------------------------
    # 3B. Governed path: Critic review
    # ------------------------------------------------------------------

    def critic_review(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> dict[str, Any]:
        node_name = "critic_review"
        started_at = perf_counter()

        try:
            self._require_mode(
                state,
                "governed_shared",
            )

            responder_id = self._required_state_text(
                state,
                "responder_agent_id",
            )
            existing_context = (
                self._build_responder_existing_context(
                    state
                )
            )

            reviews: dict[str, MemoryReviewOutput] = {}

            for memory_id in state.get(
                "candidate_memory_ids",
                [],
            ):
                owner_id = self._candidate_owner(
                    state,
                    memory_id,
                )
                owner_agent = self._persona_agent(owner_id)
                memory = self.deps.memory_service.get_memory(
                    owner_agent,
                    memory_id,
                )

                prompt = (
                    self.critic_prompt
                    .build_memory_review_prompt(
                        candidate_memory_id=memory_id,
                        candidate_memory=(
                            self._memory_to_prompt_dict(memory)
                        ),
                        requested_action=(
                            "promote_to_task_scoped_shared"
                        ),
                        requester_agent_id=owner_id,
                        target_agent_ids=[responder_id],
                        task_context=state["question"],
                        policy_context=(
                            self._governance_policy_context(
                                state
                            )
                        ),
                        existing_memory_context=(
                            existing_context
                        ),
                        context_sections={
                            "Run ID": state.get("run_id"),
                            "Sample ID": state.get(
                                "sample_id"
                            ),
                        },
                    )
                )

                review = (
                    self.deps.llm_client.invoke_structured(
                        prompt,
                        MemoryReviewOutput,
                    )
                )

                if review.memory_id != memory_id:
                    raise ValueError(
                        "Critic returned a review for the wrong "
                        f"memory: {review.memory_id!r}."
                    )

                request_id = self._promotion_request_id(
                    state,
                    memory_id,
                )

                (
                    self.deps.promotion_service
                    .review_promotion_request(
                        agent=self.deps.critic_agent,
                        request_id=request_id,
                        comment=review.reason,
                    )
                )

                reviews[memory_id] = review

            return {
                "critic_reviews": reviews,
                "workflow_status": "reviewed",
                "node_trace": [node_name],
                "node_metrics": {
                    f"{node_name}_ms": self._elapsed_ms(
                        started_at
                    )
                },
            }
        except Exception as error:
            raise self._node_error(
                node_name,
                error,
            ) from error

    # ------------------------------------------------------------------
    # 3C. Governed path: Coordinator decisions
    # ------------------------------------------------------------------

    def coordinator_decide(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> dict[str, Any]:
        node_name = "coordinator_decide"
        started_at = perf_counter()

        try:
            self._require_mode(
                state,
                "governed_shared",
            )

            decisions: dict[
                str,
                PromotionDecisionOutput,
            ] = {}
            warnings: list[str] = []

            for memory_id in state.get(
                "candidate_memory_ids",
                [],
            ):
                review = state.get(
                    "critic_reviews",
                    {},
                ).get(memory_id)

                if review is None:
                    raise ValueError(
                        f"Missing Critic review for {memory_id!r}."
                    )

                owner_id = self._candidate_owner(
                    state,
                    memory_id,
                )
                memory = self.deps.memory_service.get_memory(
                    self._persona_agent(owner_id),
                    memory_id,
                )

                request_id = self._promotion_request_id(
                    state,
                    memory_id,
                )
                request = (
                    self.deps.promotion_service.get_request(
                        request_id
                    )
                )

                prompt = (
                    self.coordinator_prompt
                    .build_promotion_decision_prompt(
                        candidate_memory_id=memory_id,
                        candidate_memory=(
                            self._memory_to_prompt_dict(memory)
                        ),
                        promotion_request=request,
                        critic_review=review,
                        policy_context=(
                            self._governance_policy_context(
                                state
                            )
                        ),
                        existing_memory_context=(
                            self._build_responder_existing_context(
                                state
                            )
                        ),
                        task_context=state["question"],
                        context_sections={
                            "Run ID": state.get("run_id"),
                            "Sample ID": state.get(
                                "sample_id"
                            ),
                            "Source owner agent ID": owner_id,
                        },
                    )
                )

                decision = (
                    self.deps.llm_client.invoke_structured(
                        prompt,
                        PromotionDecisionOutput,
                    )
                )

                if decision.memory_id != memory_id:
                    raise ValueError(
                        "Coordinator returned a decision for "
                        f"the wrong memory: "
                        f"{decision.memory_id!r}."
                    )

                decision, decision_warnings = (
                    self._normalise_promotion_decision(
                        state=state,
                        memory_id=memory_id,
                        owner_id=owner_id,
                        decision=decision,
                    )
                )

                warnings.extend(decision_warnings)
                decisions[memory_id] = decision

            return {
                "promotion_decisions": decisions,
                "workflow_status": "decided",
                "node_trace": [node_name],
                "warnings": warnings,
                "node_metrics": {
                    f"{node_name}_ms": self._elapsed_ms(
                        started_at
                    )
                },
            }
        except Exception as error:
            raise self._node_error(
                node_name,
                error,
            ) from error

    # ------------------------------------------------------------------
    # 3D. Governed path: apply decisions
    # ------------------------------------------------------------------

    def apply_governance(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> dict[str, Any]:
        node_name = "apply_governance"
        started_at = perf_counter()

        try:
            self._require_mode(
                state,
                "governed_shared",
            )

            shared_ids: list[str] = []
            rejected_ids: list[str] = []
            warnings: list[str] = []

            for memory_id in state.get(
                "candidate_memory_ids",
                [],
            ):
                decision = state.get(
                    "promotion_decisions",
                    {},
                ).get(memory_id)

                if decision is None:
                    raise ValueError(
                        "Missing Coordinator decision for "
                        f"{memory_id!r}."
                    )

                request_id = self._promotion_request_id(
                    state,
                    memory_id,
                )

                if decision.decision == "approve":
                    readable_by = list(
                        decision.allowed_agent_ids
                    )

                    promoted_memory = (
                        self.deps.promotion_service
                        .approve_promotion_request(
                            agent=(
                                self.deps.coordinator_agent
                            ),
                            request_id=request_id,
                            readable_by=readable_by,
                            comment=decision.reason,
                        )
                    )

                    shared_ids.append(
                        self._memory_id(promoted_memory)
                    )
                    continue

                # The current PromotionService supports approve/reject.
                # Merge/supersede require a separate conflict-resolution
                # service, so keep the memory private and record a warning.
                if decision.decision in {
                    "merge",
                    "supersede",
                }:
                    warnings.append(
                        f"{decision.decision} for {memory_id} "
                        "is not implemented by PromotionService; "
                        "the request was rejected and the memory "
                        "remained private."
                    )

                (
                    self.deps.promotion_service
                    .reject_promotion_request(
                        agent=self.deps.coordinator_agent,
                        request_id=request_id,
                        comment=decision.reason,
                    )
                )
                rejected_ids.append(memory_id)

            return {
                "shared_memory_ids": shared_ids,
                "rejected_memory_ids": rejected_ids,
                "workflow_status": "shared",
                "node_trace": [node_name],
                "warnings": warnings,
                "node_metrics": {
                    f"{node_name}_ms": self._elapsed_ms(
                        started_at
                    )
                },
            }
        except Exception as error:
            raise self._node_error(
                node_name,
                error,
            ) from error

    # ------------------------------------------------------------------
    # 3E. Ungoverned baseline
    # ------------------------------------------------------------------

    def direct_share(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> dict[str, Any]:
        """
        Deliberately bypass review and approval for the ungoverned baseline.

        This directly changes the memory scope through MemoryStore. It must
        never be used by the governed path. Run each experiment mode with a
        fresh ingestion/run because this operation mutates the private memory.
        """
        node_name = "direct_share"
        started_at = perf_counter()

        try:
            self._require_mode(
                state,
                "ungoverned_shared",
            )

            memory_store = getattr(
                self.deps.promotion_service,
                "memory_store",
                None,
            )

            if memory_store is None or not callable(
                getattr(memory_store, "update_scope", None)
            ):
                raise RuntimeError(
                    "The ungoverned baseline requires "
                    "promotion_service.memory_store.update_scope()."
                )

            responder_id = self._required_state_text(
                state,
                "responder_agent_id",
            )
            shared_ids: list[str] = []

            for memory_id in state.get(
                "candidate_memory_ids",
                [],
            ):
                owner_id = self._candidate_owner(
                    state,
                    memory_id,
                )

                if self.config.ungoverned_task_scoped_acl:
                    readable_by = self._unique_strings(
                        [owner_id, responder_id]
                    )
                else:
                    readable_by = ["*"]

                shared_memory = memory_store.update_scope(
                    memory_id=memory_id,
                    scope="shared",
                    owner_agent_id=(
                        self.deps.coordinator_agent.agent_id
                    ),
                    readable_by=readable_by,
                    writable_by=[
                        self.deps.coordinator_agent.agent_id
                    ],
                )

                shared_ids.append(
                    self._memory_id(shared_memory)
                )

            return {
                "shared_memory_ids": shared_ids,
                "workflow_status": "shared",
                "node_trace": [node_name],
                "warnings": [
                    (
                        "Ungoverned baseline bypassed Critic review "
                        "and Coordinator approval."
                    )
                ],
                "node_metrics": {
                    f"{node_name}_ms": self._elapsed_ms(
                        started_at
                    )
                },
            }
        except Exception as error:
            raise self._node_error(
                node_name,
                error,
            ) from error

    # ------------------------------------------------------------------
    # 4. Responder retrieval
    # ------------------------------------------------------------------

    def retrieve_for_responder(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> dict[str, Any]:
        node_name = "retrieve_for_responder"
        started_at = perf_counter()

        try:
            question = self._required_state_text(
                state,
                "question",
            )
            responder_id = self._required_state_text(
                state,
                "responder_agent_id",
            )
            responder = self._persona_agent(responder_id)
            mode = self._experiment_mode(state)

            allowed_private_ids = set(
                state.get(
                    "private_memory_ids",
                    {},
                ).get(responder_id, [])
            )
            allowed_shared_ids = (
                set(state.get("shared_memory_ids", []))
                if mode != "private_only"
                else set()
            )

            memories, retrieval_warning = (
                self._retrieve_and_filter(
                    agent=responder,
                    query=question,
                    top_k=self.config.responder_top_k,
                    include_private=True,
                    include_shared=(
                        mode != "private_only"
                    ),
                    allowed_private_ids=(
                        allowed_private_ids
                    ),
                    allowed_shared_ids=(
                        allowed_shared_ids
                    ),
                )
            )

            memory_ids = [
                self._memory_id(memory)
                for memory in memories
            ]

            context = self._format_memory_context(
                memories
            )

            warnings = (
                [retrieval_warning]
                if retrieval_warning
                else []
            )

            return {
                "responder_retrieved_memory_ids": (
                    memory_ids
                ),
                "responder_memory_context": context,
                "workflow_status": "retrieved",
                "node_trace": [node_name],
                "warnings": warnings,
                "node_metrics": {
                    f"{node_name}_ms": self._elapsed_ms(
                        started_at
                    )
                },
            }
        except Exception as error:
            raise self._node_error(
                node_name,
                error,
            ) from error

    # ------------------------------------------------------------------
    # 5. Final answer
    # ------------------------------------------------------------------

    def generate_answer(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> dict[str, Any]:
        node_name = "generate_answer"
        started_at = perf_counter()

        try:
            responder_id = self._required_state_text(
                state,
                "responder_agent_id",
            )
            prompt_template = self.worker_prompts[
                responder_id
            ]

            prompt = prompt_template.build_answer_prompt(
                task=state["question"],
                task_input=None,
                accessible_memory_context=state.get(
                    "responder_memory_context",
                    "",
                ),
                context_sections={
                    "Run ID": state.get("run_id"),
                    "Sample ID": state.get("sample_id"),
                    "Experiment mode": state.get(
                        "experiment_mode"
                    ),
                    "Responder agent ID": responder_id,
                    "Selected agent IDs": state.get(
                        "selected_agent_ids",
                        [],
                    ),
                    "Accessible memory IDs": state.get(
                        "responder_retrieved_memory_ids",
                        [],
                    ),
                },
            )

            answer = (
                self.deps.llm_client.invoke_structured(
                    prompt,
                    AgentAnswer,
                )
            )

            answer, warnings = (
                self._validate_answer_references(
                    state=state,
                    answer=answer,
                )
            )

            return {
                "agent_answer": answer,
                "workflow_status": "answered",
                "node_trace": [node_name],
                "warnings": warnings,
                "node_metrics": {
                    f"{node_name}_ms": self._elapsed_ms(
                        started_at
                    )
                },
            }
        except Exception as error:
            raise self._node_error(
                node_name,
                error,
            ) from error

    # ------------------------------------------------------------------
    # Branch helpers for StateGraph.add_conditional_edges
    # ------------------------------------------------------------------

    @staticmethod
    def select_experiment_branch(
        state: NeedlePersonaWorkflowState,
    ) -> ExperimentMode:
        mode = state.get("experiment_mode")

        if mode not in {
            "private_only",
            "ungoverned_shared",
            "governed_shared",
        }:
            raise ValueError(
                f"Invalid experiment_mode: {mode!r}."
            )

        return mode

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _retrieve_and_filter(
        self,
        *,
        agent: Agent,
        query: str,
        top_k: int,
        include_private: bool,
        include_shared: bool,
        allowed_private_ids: set[str],
        allowed_shared_ids: set[str],
    ) -> tuple[list[Any], str | None]:
        allowed_ids = (
            allowed_private_ids | allowed_shared_ids
        )

        if not allowed_ids:
            return [], None

        requested_top_k = max(
            top_k,
            min(len(allowed_ids) * 2, 100),
        )

        try:
            retrieved = (
                self.deps.memory_service
                .retrieve_memories(
                    agent=agent,
                    query=query,
                    top_k=requested_top_k,
                    include_private=include_private,
                    include_shared=include_shared,
                )
            )
        except Exception as error:
            if not self.config.allow_lexical_retrieval_fallback:
                raise

            fallback = self._lexical_retrieval(
                agent=agent,
                query=query,
                allowed_memory_ids=allowed_ids,
                top_k=top_k,
            )

            return (
                fallback,
                (
                    "Semantic retrieval failed and lexical "
                    f"fallback was used for {agent.agent_id}: "
                    f"{error}"
                ),
            )

        filtered: list[Any] = []
        seen: set[str] = set()

        for memory in retrieved:
            memory_id = self._memory_id(memory)

            if memory_id not in allowed_ids:
                continue

            if memory_id in seen:
                continue

            scope = self._memory_scope(memory)

            if (
                scope == "private"
                and memory_id not in allowed_private_ids
            ):
                continue

            if (
                scope == "shared"
                and memory_id not in allowed_shared_ids
            ):
                continue

            filtered.append(memory)
            seen.add(memory_id)

            if len(filtered) >= top_k:
                break

        return filtered, None

    def _lexical_retrieval(
        self,
        *,
        agent: Agent,
        query: str,
        allowed_memory_ids: set[str],
        top_k: int,
    ) -> list[Any]:
        query_terms = self._tokenise(query)
        scored: list[tuple[float, str, Any]] = []

        for memory_id in sorted(allowed_memory_ids):
            try:
                memory = (
                    self.deps.memory_service.get_memory(
                        agent,
                        memory_id,
                    )
                )
            except Exception:
                continue

            memory_terms = self._tokenise(
                self._memory_search_text(memory)
            )

            overlap = len(
                query_terms & memory_terms
            )
            denominator = max(
                1,
                len(query_terms),
            )
            score = overlap / denominator

            scored.append(
                (
                    score,
                    memory_id,
                    memory,
                )
            )

        scored.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )

        return [
            memory
            for _, _, memory in scored[:top_k]
        ]

    # ------------------------------------------------------------------
    # Governance helpers
    # ------------------------------------------------------------------

    def _governance_policy_context(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> dict[str, Any]:
        responder_id = self._required_state_text(
            state,
            "responder_agent_id",
        )

        return {
            "policy_name": (
                "Needle task-scoped controlled sharing"
            ),
            "rules": [
                (
                    "A contributor memory remains private unless "
                    "the Critic reviews it and the Coordinator "
                    "approves it."
                ),
                (
                    "Only information relevant to the active "
                    "question may be shared."
                ),
                (
                    "The responder must be an explicitly "
                    "authorised reader."
                ),
                (
                    "Unrelated, policy-violating, duplicate, "
                    "outdated, or conflicting memories should "
                    "not be blindly promoted."
                ),
            ],
            "required_reader_agent_id": responder_id,
            "global_sharing_allowed": False,
        }

    def _normalise_promotion_decision(
        self,
        *,
        state: NeedlePersonaWorkflowState,
        memory_id: str,
        owner_id: str,
        decision: PromotionDecisionOutput,
    ) -> tuple[PromotionDecisionOutput, list[str]]:
        warnings: list[str] = []
        responder_id = self._required_state_text(
            state,
            "responder_agent_id",
        )
        known_agent_ids = {
            *self.persona_agents,
            self.deps.critic_agent.agent_id,
            self.deps.coordinator_agent.agent_id,
        }

        if decision.decision in {
            "reject",
            "keep_private",
            "merge",
            "supersede",
        }:
            if decision.target_scope != "private":
                warnings.append(
                    f"Coordinator decision for {memory_id} "
                    "was normalised to private scope."
                )

            return (
                decision.model_copy(
                    update={
                        "target_scope": "private",
                        "allowed_agent_ids": [],
                    }
                ),
                warnings,
            )

        allowed_readers = self._clean_agent_ids(
            decision.allowed_agent_ids,
            allowed=known_agent_ids,
        )

        # Preserve source-owner access and ensure that the selected responder
        # can read the approved task-scoped shared memory.
        required_readers = [
            owner_id,
            responder_id,
        ]

        for agent_id in required_readers:
            if agent_id not in allowed_readers:
                allowed_readers.append(agent_id)
                warnings.append(
                    f"{agent_id} was added to the read ACL "
                    f"for approved memory {memory_id}."
                )

        return (
            decision.model_copy(
                update={
                    "target_scope": "shared",
                    "allowed_agent_ids": allowed_readers,
                }
            ),
            warnings,
        )

    def _build_responder_existing_context(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> list[dict[str, Any]]:
        responder_id = self._required_state_text(
            state,
            "responder_agent_id",
        )
        responder = self._persona_agent(responder_id)
        memory_ids = list(
            state.get(
                "private_memory_ids",
                {},
            ).get(responder_id, [])
        )

        context: list[dict[str, Any]] = []

        for memory_id in memory_ids[
            : self.config.existing_context_limit
        ]:
            try:
                memory = (
                    self.deps.memory_service.get_memory(
                        responder,
                        memory_id,
                    )
                )
            except Exception:
                continue

            context.append(
                self._memory_to_prompt_dict(memory)
            )

        return context

    # ------------------------------------------------------------------
    # Answer validation
    # ------------------------------------------------------------------

    def _validate_answer_references(
        self,
        *,
        state: NeedlePersonaWorkflowState,
        answer: AgentAnswer,
    ) -> tuple[AgentAnswer, list[str]]:
        if not self.config.sanitise_answer_references:
            return answer, []

        warnings: list[str] = []
        allowed_memory_ids = set(
            state.get(
                "responder_retrieved_memory_ids",
                [],
            )
        )
        allowed_agent_ids = set(
            state.get(
                "selected_agent_ids",
                [],
            )
        )

        valid_used_memory_ids = [
            memory_id
            for memory_id in answer.used_memory_ids
            if memory_id in allowed_memory_ids
        ]

        invalid_memory_ids = (
            set(answer.used_memory_ids)
            - allowed_memory_ids
        )

        if invalid_memory_ids:
            warnings.append(
                "Removed unavailable used_memory_ids from "
                f"AgentAnswer: {sorted(invalid_memory_ids)}."
            )

        valid_contributing_agents = [
            agent_id
            for agent_id in answer.contributing_agent_ids
            if agent_id in allowed_agent_ids
        ]

        invalid_agent_ids = (
            set(answer.contributing_agent_ids)
            - allowed_agent_ids
        )

        if invalid_agent_ids:
            warnings.append(
                "Removed unavailable contributing_agent_ids "
                f"from AgentAnswer: {sorted(invalid_agent_ids)}."
            )

        accessible_source_ids: set[str] = set()

        responder = self._persona_agent(
            self._required_state_text(
                state,
                "responder_agent_id",
            )
        )

        for memory_id in allowed_memory_ids:
            try:
                memory = (
                    self.deps.memory_service.get_memory(
                        responder,
                        memory_id,
                    )
                )
            except Exception:
                continue

            accessible_source_ids.update(
                self._memory_source_ids(memory)
            )

        valid_source_ids = [
            source_id
            for source_id in answer.supporting_source_ids
            if source_id in accessible_source_ids
        ]

        invalid_source_ids = (
            set(answer.supporting_source_ids)
            - accessible_source_ids
        )

        if invalid_source_ids:
            warnings.append(
                "Removed unavailable supporting_source_ids "
                f"from AgentAnswer: {sorted(invalid_source_ids)}."
            )

        return (
            answer.model_copy(
                update={
                    "used_memory_ids": (
                        valid_used_memory_ids
                    ),
                    "supporting_source_ids": (
                        valid_source_ids
                    ),
                    "contributing_agent_ids": (
                        valid_contributing_agents
                    ),
                }
            ),
            warnings,
        )

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_memory_context(
        self,
        memories: Sequence[Any],
    ) -> str:
        if not memories:
            return "No accessible memory."

        sections: list[str] = []

        for index, memory in enumerate(
            memories,
            start=1,
        ):
            data = self._memory_to_prompt_dict(memory)

            sections.append(
                "\n".join(
                    [
                        f"[Memory {index}]",
                        f"memory_id: {data['memory_id']}",
                        f"owner_agent_id: "
                        f"{data['owner_agent_id']}",
                        f"scope: {data['scope']}",
                        f"memory_type: "
                        f"{data['memory_type']}",
                        f"content: {data['content']}",
                        f"summary: {data['summary']}",
                        "source_ids: "
                        + ", ".join(data["source_ids"]),
                    ]
                )
            )

        return "\n\n".join(sections)

    def _memory_to_prompt_dict(
        self,
        memory: Any,
    ) -> dict[str, Any]:
        metadata = self._read_field(
            memory,
            "metadata",
            {},
        )

        return {
            "memory_id": self._memory_id(memory),
            "content": str(
                self._read_field(
                    memory,
                    "content",
                    "",
                )
            ).strip(),
            "summary": (
                str(
                    self._read_field(
                        memory,
                        "summary",
                        "",
                    )
                ).strip()
                or None
            ),
            "owner_agent_id": str(
                self._read_field(
                    metadata,
                    "owner_agent_id",
                    "",
                )
            ).strip(),
            "scope": self._memory_scope(memory),
            "memory_type": self._enum_value(
                self._read_field(
                    metadata,
                    "memory_type",
                    "note",
                )
            ),
            "status": self._enum_value(
                self._read_field(
                    metadata,
                    "status",
                    "active",
                )
            ),
            "source_ids": self._memory_source_ids(
                memory
            ),
            "tags": list(
                self._read_field(
                    metadata,
                    "tags",
                    [],
                )
                or []
            ),
            "importance": self._read_field(
                metadata,
                "importance",
                None,
            ),
            "confidence": self._read_field(
                metadata,
                "confidence",
                None,
            ),
        }

    # ------------------------------------------------------------------
    # Validation and low-level helpers
    # ------------------------------------------------------------------

    def _validate_dependencies(self) -> None:
        expected_persona_ids = set(
            AGENT_PERSONA_MAP
        )
        missing_agents = (
            expected_persona_ids
            - set(self.persona_agents)
        )

        if missing_agents:
            raise ValueError(
                "Missing persona agents: "
                f"{sorted(missing_agents)}."
            )

        missing_prompts = (
            expected_persona_ids
            - set(self.worker_prompts)
        )

        if missing_prompts:
            raise ValueError(
                "Missing Worker prompt templates: "
                f"{sorted(missing_prompts)}."
            )

        for agent_id in expected_persona_ids:
            agent = self.persona_agents[agent_id]

            if agent.agent_id != agent_id:
                raise ValueError(
                    f"Persona-agent key {agent_id!r} does not "
                    f"match Agent.agent_id "
                    f"{agent.agent_id!r}."
                )

            if agent.role != AgentRole.WORKER:
                raise ValueError(
                    f"{agent_id!r} must be a Worker."
                )

            if self.worker_prompts[
                agent_id
            ].agent_id != agent_id:
                raise ValueError(
                    f"Worker prompt for {agent_id!r} has "
                    "the wrong agent_id."
                )

        if (
            self.deps.critic_agent.role
            != AgentRole.CRITIC
        ):
            raise ValueError(
                "critic_agent must have AgentRole.CRITIC."
            )

        if (
            self.deps.coordinator_agent.role
            != AgentRole.COORDINATOR
        ):
            raise ValueError(
                "coordinator_agent must have "
                "AgentRole.COORDINATOR."
            )

    def _persona_agent(
        self,
        agent_id: str,
    ) -> Agent:
        try:
            return self.persona_agents[agent_id]
        except KeyError as error:
            raise KeyError(
                f"Unknown persona agent: {agent_id!r}."
            ) from error

    @staticmethod
    def _required_state_text(
        state: NeedlePersonaWorkflowState,
        field_name: str,
    ) -> str:
        value = str(
            state.get(field_name, "")
        ).strip()

        if not value:
            raise ValueError(
                f"State field {field_name!r} is required."
            )

        return value

    @staticmethod
    def _experiment_mode(
        state: NeedlePersonaWorkflowState,
    ) -> ExperimentMode:
        return NeedlePersonaNodes.select_experiment_branch(
            state
        )

    @classmethod
    def _require_mode(
        cls,
        state: NeedlePersonaWorkflowState,
        expected_mode: ExperimentMode,
    ) -> None:
        actual = cls._experiment_mode(state)

        if actual != expected_mode:
            raise ValueError(
                f"Node requires experiment_mode "
                f"{expected_mode!r}, got {actual!r}."
            )

    @staticmethod
    def _candidate_owner(
        state: NeedlePersonaWorkflowState,
        memory_id: str,
    ) -> str:
        owner_id = state.get(
            "candidate_owner_by_memory_id",
            {},
        ).get(memory_id)

        if not owner_id:
            raise ValueError(
                f"No candidate owner recorded for "
                f"{memory_id!r}."
            )

        return owner_id

    @staticmethod
    def _promotion_request_id(
        state: NeedlePersonaWorkflowState,
        memory_id: str,
    ) -> str:
        request_id = state.get(
            "promotion_request_ids",
            {},
        ).get(memory_id)

        if not request_id:
            raise ValueError(
                f"No promotion request recorded for "
                f"{memory_id!r}."
            )

        return request_id

    @staticmethod
    def _memory_id(memory: Any) -> str:
        memory_id = NeedlePersonaNodes._read_field(
            memory,
            "memory_id",
            None,
        )
        text = (
            str(memory_id).strip()
            if memory_id is not None
            else ""
        )

        if not text:
            raise ValueError(
                "Memory object does not contain memory_id."
            )

        return text

    @staticmethod
    def _request_id(request: Any) -> str:
        request_id = NeedlePersonaNodes._read_field(
            request,
            "request_id",
            None,
        )
        text = (
            str(request_id).strip()
            if request_id is not None
            else ""
        )

        if not text:
            raise ValueError(
                "Promotion request does not contain "
                "request_id."
            )

        return text

    @staticmethod
    def _memory_scope(memory: Any) -> str:
        metadata = NeedlePersonaNodes._read_field(
            memory,
            "metadata",
            {},
        )
        return NeedlePersonaNodes._enum_value(
            NeedlePersonaNodes._read_field(
                metadata,
                "scope",
                "",
            )
        ).lower()

    @staticmethod
    def _memory_source_ids(
        memory: Any,
    ) -> list[str]:
        metadata = NeedlePersonaNodes._read_field(
            memory,
            "metadata",
            {},
        )
        values = NeedlePersonaNodes._read_field(
            metadata,
            "source_message_ids",
            [],
        )

        return NeedlePersonaNodes._unique_strings(
            list(values or [])
        )

    @staticmethod
    def _memory_search_text(memory: Any) -> str:
        content = str(
            NeedlePersonaNodes._read_field(
                memory,
                "content",
                "",
            )
        )
        summary = str(
            NeedlePersonaNodes._read_field(
                memory,
                "summary",
                "",
            )
            or ""
        )

        return f"{summary}\n{content}"

    @staticmethod
    def _read_field(
        value: Any,
        field_name: str,
        default: Any = None,
    ) -> Any:
        if value is None:
            return default

        if isinstance(value, Mapping):
            return value.get(field_name, default)

        return getattr(value, field_name, default)

    @staticmethod
    def _enum_value(value: Any) -> str:
        if hasattr(value, "value"):
            value = value.value

        return str(value).strip()

    @staticmethod
    def _tokenise(text: str) -> set[str]:
        return {
            token
            for token in re.findall(
                r"[A-Za-z0-9]+",
                text.lower(),
            )
            if token
        }

    @staticmethod
    def _unique_strings(
        values: Sequence[str],
    ) -> list[str]:
        result: list[str] = []

        for value in values:
            text = str(value).strip()

            if text and text not in result:
                result.append(text)

        return result

    @classmethod
    def _clean_agent_ids(
        cls,
        values: Sequence[str],
        *,
        allowed: set[str],
    ) -> list[str]:
        return [
            agent_id
            for agent_id in cls._unique_strings(values)
            if agent_id in allowed
        ]

    @staticmethod
    def _elapsed_ms(
        started_at: float,
    ) -> float:
        return (
            perf_counter() - started_at
        ) * 1000.0

    @staticmethod
    def _node_error(
        node_name: str,
        error: Exception,
    ) -> NeedlePersonaNodeError:
        if isinstance(
            error,
            NeedlePersonaNodeError,
        ):
            return error

        return NeedlePersonaNodeError(
            node_name=node_name,
            message=str(error),
            cause=error,
        )


def build_failure_state_update(
    *,
    node_name: str,
    error: Exception,
) -> dict[str, Any]:
    """
    Optional adapter for workflows that prefer a failure state over raising.

    Do not wrap nodes with this unless the graph has a conditional edge that
    routes ``workflow_status == "failed"`` directly to END.
    """
    return {
        "workflow_status": "failed",
        "node_trace": [node_name],
        "errors": [f"{node_name}: {error}"],
    }


__all__ = [
    "RoutingStrategy",
    "NodeMemoryServiceProtocol",
    "NodePromotionServiceProtocol",
    "NeedlePersonaNodeError",
    "NeedlePersonaNodeConfig",
    "NeedlePersonaNodeDependencies",
    "NeedlePersonaNodes",
    "build_default_governance_agents",
    "build_failure_state_update",
]