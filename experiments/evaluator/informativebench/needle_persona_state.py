from __future__ import annotations

from operator import add
from typing import Annotated, Literal, TypedDict

from .needle_persona_models import (
    AGENT_PERSONA_MAP,
    NeedlePersonaSample,
    PersonaName,
    PrivateMemoryIndex,
)

from src.llm import (
    AgentAnswer,
    MemoryReviewOutput,
    PromotionDecisionOutput,
    TaskRoutingOutput,
)

from .needle_persona_models import (
    AGENT_PERSONA_MAP,
    NeedlePersonaSample,
    PrivateMemoryIndex,
)


ExperimentMode = Literal[
    "private_only",
    "ungoverned_shared",
    "governed_shared",
]

WorkflowStatus = Literal[
    "initialized",
    "routed",
    "candidates_collected",
    "reviewed",
    "decided",
    "shared",
    "retrieved",
    "answered",
    "failed",
]


def merge_unique_strings(
    left: list[str] | None,
    right: list[str] | None,
) -> list[str]:
    """
    LangGraph reducer for ID lists.

    It preserves first-seen order and removes duplicates. This is useful when
    several nodes or parallel branches contribute memory IDs.
    """
    merged: list[str] = []

    for value in [*(left or []), *(right or [])]:
        item = str(value).strip()

        if item and item not in merged:
            merged.append(item)

    return merged


def merge_string_list_mapping(
    left: dict[str, list[str]] | None,
    right: dict[str, list[str]] | None,
) -> dict[str, list[str]]:
    """
    Merge mappings such as agent_id -> retrieved memory IDs.
    """
    merged: dict[str, list[str]] = {
        key: list(values)
        for key, values in (left or {}).items()
    }

    for key, values in (right or {}).items():
        merged[key] = merge_unique_strings(
            merged.get(key),
            values,
        )

    return merged


def merge_string_mapping(
    left: dict[str, str] | None,
    right: dict[str, str] | None,
) -> dict[str, str]:
    """
    Merge string mappings. Values from the right-hand update take precedence.
    """
    return {
        **(left or {}),
        **(right or {}),
    }


def merge_review_mapping(
    left: dict[str, MemoryReviewOutput] | None,
    right: dict[str, MemoryReviewOutput] | None,
) -> dict[str, MemoryReviewOutput]:
    """
    Merge memory_id -> Critic review mappings.
    """
    return {
        **(left or {}),
        **(right or {}),
    }


def merge_decision_mapping(
    left: dict[str, PromotionDecisionOutput] | None,
    right: dict[str, PromotionDecisionOutput] | None,
) -> dict[str, PromotionDecisionOutput]:
    """
    Merge memory_id -> Coordinator decision mappings.
    """
    return {
        **(left or {}),
        **(right or {}),
    }


def merge_float_mapping(
    left: dict[str, float] | None,
    right: dict[str, float] | None,
) -> dict[str, float]:
    """
    Merge node metric mappings, such as latency in milliseconds.
    """
    return {
        **(left or {}),
        **(right or {}),
    }


class NeedlePersonaWorkflowState(TypedDict, total=False):
    """
    LangGraph state for one Needle in the Persona task run.

    Security boundary:
    - contains no raw persona dialogue;
    - contains no chat_bob_charlie;
    - contains no needle_detail;
    - contains no gold answer or accepted answers.

    The answer reference must remain outside the task graph and be supplied to
    the evaluator only after answer generation.
    """

    # ------------------------------------------------------------------
    # Run identity and experiment configuration
    # ------------------------------------------------------------------

    run_id: str
    sample_id: str
    experiment_mode: ExperimentMode
    workflow_status: WorkflowStatus

    # ------------------------------------------------------------------
    # Task input
    # ------------------------------------------------------------------

    question: str

    # IDs of private memories produced by the ingestion stage.
    private_memory_ids: dict[str, list[str]]

    # ------------------------------------------------------------------
    # Coordinator routing
    # ------------------------------------------------------------------

    routing_output: TaskRoutingOutput
    selected_agent_ids: list[str]
    responder_agent_id: str
    contributor_agent_ids: list[str]

    # ------------------------------------------------------------------
    # Contributor private-memory retrieval
    # ------------------------------------------------------------------

    contributor_retrieved_memory_ids: Annotated[
        dict[str, list[str]],
        merge_string_list_mapping,
    ]

    candidate_memory_ids: Annotated[
        list[str],
        merge_unique_strings,
    ]

    # Maps each candidate memory to the Worker that owns it.
    candidate_owner_by_memory_id: Annotated[
        dict[str, str],
        merge_string_mapping,
    ]

    # ------------------------------------------------------------------
    # Governed-sharing path
    # ------------------------------------------------------------------

    critic_reviews: Annotated[
        dict[str, MemoryReviewOutput],
        merge_review_mapping,
    ]

    promotion_decisions: Annotated[
        dict[str, PromotionDecisionOutput],
        merge_decision_mapping,
    ]

    # IDs made visible to the responder. In private_only this should remain
    # empty. In ungoverned_shared it is filled by direct sharing. In
    # governed_shared it is filled only after approved governance decisions.
    shared_memory_ids: Annotated[
        list[str],
        merge_unique_strings,
    ]

    rejected_memory_ids: Annotated[
        list[str],
        merge_unique_strings,
    ]

    # ------------------------------------------------------------------
    # Responder retrieval and answer generation
    # ------------------------------------------------------------------

    responder_retrieved_memory_ids: Annotated[
        list[str],
        merge_unique_strings,
    ]

    # Permission-filtered, formatted memory context supplied to the responder.
    responder_memory_context: str

    agent_answer: AgentAnswer

    # ------------------------------------------------------------------
    # Observability and failure handling
    # ------------------------------------------------------------------

    # Node names are appended in execution order. Duplicates are intentionally
    # preserved because retries may execute the same node more than once.
    node_trace: Annotated[list[str], add]

    errors: Annotated[list[str], add]
    warnings: Annotated[list[str], add]

    # Example keys: "route_task_ms", "critic_review_ms".
    node_metrics: Annotated[
        dict[str, float],
        merge_float_mapping,
    ]

    # Contributor retrieval
    contributor_retrieved_memory_ids: Annotated[
        dict[str, list[str]],
        merge_string_list_mapping,
    ]

    candidate_memory_ids: Annotated[
        list[str],
        merge_unique_strings,
    ]

    candidate_owner_by_memory_id: Annotated[
        dict[str, str],
        merge_string_mapping,
    ]

    # Governed sharing
    promotion_request_ids: Annotated[
        dict[str, str],
        merge_string_mapping,
    ]

    critic_reviews: Annotated[
        dict[str, MemoryReviewOutput],
        merge_review_mapping,
    ]

    promotion_decisions: Annotated[
        dict[str, PromotionDecisionOutput],
        merge_decision_mapping,
    ]

    shared_memory_ids: Annotated[
        list[str],
        merge_unique_strings,
    ]

    rejected_memory_ids: Annotated[
        list[str],
        merge_unique_strings,
    ]

class NeedlePersonaEvaluationEvidence(TypedDict):
    """
    Evaluator-only gold evidence and runtime memory mappings.

    This object is created only after the workflow has completed, so its
    contents cannot leak into the task graph.
    """

    # Personas explicitly required by the benchmark question.
    target_personas: list[PersonaName]

    # All source records required to answer the question.
    expected_source_ids: list[str]

    # Required sources owned by the responder.
    expected_local_source_ids: list[str]

    # Required sources owned by other agents.
    expected_cross_agent_source_ids: list[str]

    # Runtime mapping from each memory ID to its supporting source IDs.
    memory_id_to_source_ids: dict[str, list[str]]

    # Runtime mapping from each memory ID to its original owner agent.
    memory_id_to_owner_agent_id: dict[str, str]

class NeedlePersonaEvaluationInput(TypedDict):
    """
    Complete deterministic evaluator input for one finished workflow run.

    Reference answers and gold evidence are added only after the workflow has
    produced agent_answer.
    """

    # ------------------------------------------------------------------
    # Run identity
    # ------------------------------------------------------------------

    run_id: str
    sample_id: str
    experiment_mode: ExperimentMode
    question: str

    # ------------------------------------------------------------------
    # Answer data
    # ------------------------------------------------------------------

    predicted_answer: str
    accepted_answers: list[str]

    responder_agent_id: str

    # Contributors selected during routing.
    routed_contributor_agent_ids: list[str]

    # Contributors reported by the final answering agent.
    answer_contributing_agent_ids: list[str]

    # ------------------------------------------------------------------
    # Answer attribution
    # ------------------------------------------------------------------

    used_memory_ids: list[str]
    supporting_source_ids: list[str]

    # ------------------------------------------------------------------
    # Retrieval and sharing trace
    # ------------------------------------------------------------------

    responder_retrieved_memory_ids: list[str]

    candidate_memory_ids: list[str]
    promotion_request_ids: dict[str, str]

    shared_memory_ids: list[str]
    rejected_memory_ids: list[str]

    # ------------------------------------------------------------------
    # Governance trace
    # ------------------------------------------------------------------

    critic_reviews: dict[str, MemoryReviewOutput]
    promotion_decisions: dict[
        str,
        PromotionDecisionOutput,
    ]

    # ------------------------------------------------------------------
    # Gold evidence and runtime mappings
    # ------------------------------------------------------------------

    target_personas: list[PersonaName]

    expected_source_ids: list[str]
    expected_local_source_ids: list[str]
    expected_cross_agent_source_ids: list[str]

    memory_id_to_source_ids: dict[str, list[str]]
    memory_id_to_owner_agent_id: dict[str, str]


class NeedlePersonaEvaluationResult(TypedDict):
    """
    Final per-sample evaluation result.

    Every result contains the same keys. Metrics that do not apply to a mode
    are represented by None rather than being omitted or incorrectly set to 0.
    """

    # ------------------------------------------------------------------
    # Run identity
    # ------------------------------------------------------------------

    run_id: str
    sample_id: str
    experiment_mode: ExperimentMode
    question: str

    # ------------------------------------------------------------------
    # Answer output and matching
    # ------------------------------------------------------------------

    predicted_answer: str
    matched_reference_answer: str | None

    predicted_entities: list[str]
    matched_reference_entities: list[str]

    answer_accuracy: float
    entity_precision: float
    entity_recall: float
    entity_f1: float

    # ------------------------------------------------------------------
    # Expected evidence
    # ------------------------------------------------------------------

    expected_source_ids: list[str]
    expected_local_source_ids: list[str]
    expected_cross_agent_source_ids: list[str]

    # ------------------------------------------------------------------
    # Retrieval metrics and diagnostics
    # ------------------------------------------------------------------

    retrieved_source_ids: list[str]
    retrieved_target_source_ids: list[str]
    missing_retrieved_target_source_ids: list[str]

    target_source_retrieval_recall: float
    all_target_sources_retrieved: bool

    cross_agent_retrieved_source_ids: list[str]
    cross_agent_source_retrieval_recall: float
    cross_agent_memory_hit: bool

    # ------------------------------------------------------------------
    # Memory-utilisation metrics and diagnostics
    # ------------------------------------------------------------------

    used_source_ids: list[str]
    used_target_source_ids: list[str]

    target_source_utilisation_recall: float

    cross_agent_used_source_ids: list[str]
    cross_agent_source_utilisation_recall: float
    cross_agent_source_utilised: bool

    # ------------------------------------------------------------------
    # Sharing and governance metrics
    # ------------------------------------------------------------------

    shared_source_ids: list[str]
    correctly_shared_source_ids: list[str]
    over_shared_source_ids: list[str]
    missing_shared_cross_agent_source_ids: list[str]

    # None for private_only.
    shared_cross_agent_source_recall: float | None

    # None for private_only.
    governance_precision: float | None
    governance_recall: float | None
    over_sharing_rate: float | None
    under_sharing_rate: float | None

    # ------------------------------------------------------------------
    # Governance trace counts
    # ------------------------------------------------------------------

    candidate_memory_count: int
    critic_approved_memory_count: int
    coordinator_approved_memory_count: int
    shared_memory_count: int
    rejected_memory_count: int

    # ------------------------------------------------------------------
    # End-to-end result
    # ------------------------------------------------------------------

    # None for private_only because no transfer is attempted.
    transfer_success: bool | None


def build_initial_needle_persona_state(
    *,
    sample: NeedlePersonaSample,
    memory_index: PrivateMemoryIndex,
    experiment_mode: ExperimentMode,
) -> NeedlePersonaWorkflowState:
    """
    Build a safe initial LangGraph state.

    Only the question and already-ingested private memory IDs are copied from
    the benchmark objects. The gold answer and raw persona sources are not
    copied into the graph state.
    """
    if memory_index.sample_id != sample.sample_id:
        raise ValueError(
            "memory_index.sample_id does not match sample.sample_id: "
            f"{memory_index.sample_id!r} != {sample.sample_id!r}."
        )

    run_id = memory_index.run_id.strip()

    if not run_id:
        raise ValueError("memory_index.run_id cannot be empty.")

    private_memory_ids = {
        agent_id: list(
            memory_index.private_memory_ids.get(agent_id, [])
        )
        for agent_id in AGENT_PERSONA_MAP
    }

    return NeedlePersonaWorkflowState(
        run_id=run_id,
        sample_id=sample.sample_id,
        experiment_mode=experiment_mode,
        workflow_status="initialized",
        question=sample.question,
        private_memory_ids=private_memory_ids,

        selected_agent_ids=[],
        contributor_agent_ids=[],

        contributor_retrieved_memory_ids={},
        candidate_memory_ids=[],
        candidate_owner_by_memory_id={},

        promotion_request_ids={},

        critic_reviews={},
        promotion_decisions={},
        shared_memory_ids=[],
        rejected_memory_ids=[],

        responder_retrieved_memory_ids=[],
        responder_memory_context="",

        node_trace=[],
        errors=[],
        warnings=[],
        node_metrics={},
    )


def build_needle_persona_evaluation_input(
    *,
    sample: NeedlePersonaSample,
    final_state: NeedlePersonaWorkflowState,
    evidence: NeedlePersonaEvaluationEvidence,
) -> NeedlePersonaEvaluationInput:
    """
    Build the complete evaluator input after workflow execution.

    Reference answers and gold evidence are added only after the graph has
    produced agent_answer, preventing evaluation data from leaking into the
    task workflow.
    """
    answer = final_state.get("agent_answer")

    if answer is None:
        raise ValueError(
            "final_state does not contain agent_answer."
        )

    run_id = str(
        final_state.get("run_id", "")
    ).strip()

    if not run_id:
        raise ValueError(
            "final_state does not contain run_id."
        )

    final_sample_id = str(
        final_state.get("sample_id", "")
    ).strip()

    if final_sample_id != sample.sample_id:
        raise ValueError(
            "final_state.sample_id does not match "
            "sample.sample_id: "
            f"{final_sample_id!r} != "
            f"{sample.sample_id!r}."
        )

    responder_agent_id = str(
        final_state.get(
            "responder_agent_id",
            "",
        )
    ).strip()

    if responder_agent_id not in AGENT_PERSONA_MAP:
        raise ValueError(
            "final_state contains an invalid "
            f"responder_agent_id: "
            f"{responder_agent_id!r}."
        )

    experiment_mode = final_state.get(
        "experiment_mode"
    )

    if experiment_mode not in {
        "private_only",
        "ungoverned_shared",
        "governed_shared",
    }:
        raise ValueError(
            "final_state contains an invalid "
            f"experiment_mode: "
            f"{experiment_mode!r}."
        )

    accepted_answers = [
        str(reference).strip()
        for reference in sample.accepted_answers()
        if str(reference).strip()
    ]

    if not accepted_answers:
        raise ValueError(
            "sample does not contain any accepted answer."
        )

    target_personas = list(
        evidence["target_personas"]
    )

    expected_source_ids = list(
        evidence["expected_source_ids"]
    )

    expected_local_source_ids = list(
        evidence["expected_local_source_ids"]
    )

    expected_cross_agent_source_ids = list(
        evidence[
            "expected_cross_agent_source_ids"
        ]
    )

    if not target_personas:
        raise ValueError(
            "evidence.target_personas cannot be empty."
        )

    if not expected_source_ids:
        raise ValueError(
            "evidence.expected_source_ids cannot be empty."
        )

    expected_set = set(expected_source_ids)
    local_set = set(expected_local_source_ids)
    cross_agent_set = set(
        expected_cross_agent_source_ids
    )

    if local_set & cross_agent_set:
        raise ValueError(
            "Local and cross-agent expected source IDs "
            "must be disjoint."
        )

    if local_set | cross_agent_set != expected_set:
        raise ValueError(
            "Local and cross-agent expected source IDs "
            "must partition expected_source_ids."
        )

    memory_id_to_source_ids = {
        str(memory_id).strip(): [
            str(source_id).strip()
            for source_id in source_ids
            if str(source_id).strip()
        ]
        for memory_id, source_ids
        in evidence[
            "memory_id_to_source_ids"
        ].items()
        if str(memory_id).strip()
    }

    memory_id_to_owner_agent_id = {
        str(memory_id).strip(): (
            str(owner_agent_id).strip()
        )
        for memory_id, owner_agent_id
        in evidence[
            "memory_id_to_owner_agent_id"
        ].items()
        if str(memory_id).strip()
    }

    used_memory_ids = list(
        answer.used_memory_ids
    )

    responder_retrieved_memory_ids = list(
        final_state.get(
            "responder_retrieved_memory_ids",
            [],
        )
    )

    candidate_memory_ids = list(
        final_state.get(
            "candidate_memory_ids",
            [],
        )
    )

    shared_memory_ids = list(
        final_state.get(
            "shared_memory_ids",
            [],
        )
    )

    rejected_memory_ids = list(
        final_state.get(
            "rejected_memory_ids",
            [],
        )
    )

    runtime_memory_ids = {
        memory_id
        for memory_id in [
            *used_memory_ids,
            *responder_retrieved_memory_ids,
            *candidate_memory_ids,
            *shared_memory_ids,
            *rejected_memory_ids,
        ]
        if memory_id
    }

    unknown_source_mapping_ids = (
        runtime_memory_ids
        - set(memory_id_to_source_ids)
    )

    if unknown_source_mapping_ids:
        raise ValueError(
            "Runtime memory IDs are missing source "
            "mappings: "
            f"{sorted(unknown_source_mapping_ids)}."
        )

    unknown_owner_mapping_ids = (
        runtime_memory_ids
        - set(memory_id_to_owner_agent_id)
    )

    if unknown_owner_mapping_ids:
        raise ValueError(
            "Runtime memory IDs are missing owner "
            "mappings: "
            f"{sorted(unknown_owner_mapping_ids)}."
        )

    return NeedlePersonaEvaluationInput(
        # Run identity
        run_id=run_id,
        sample_id=sample.sample_id,
        experiment_mode=experiment_mode,
        question=sample.question,

        # Answer data
        predicted_answer=answer.answer,
        accepted_answers=accepted_answers,

        responder_agent_id=responder_agent_id,

        routed_contributor_agent_ids=list(
            final_state.get(
                "contributor_agent_ids",
                [],
            )
        ),

        answer_contributing_agent_ids=list(
            answer.contributing_agent_ids
        ),

        # Answer attribution
        used_memory_ids=used_memory_ids,
        supporting_source_ids=list(
            answer.supporting_source_ids
        ),

        # Retrieval and sharing trace
        responder_retrieved_memory_ids=(
            responder_retrieved_memory_ids
        ),

        candidate_memory_ids=(
            candidate_memory_ids
        ),

        promotion_request_ids=dict(
            final_state.get(
                "promotion_request_ids",
                {},
            )
        ),

        shared_memory_ids=shared_memory_ids,
        rejected_memory_ids=rejected_memory_ids,

        # Governance trace
        critic_reviews=dict(
            final_state.get(
                "critic_reviews",
                {},
            )
        ),

        promotion_decisions=dict(
            final_state.get(
                "promotion_decisions",
                {},
            )
        ),

        # Gold evidence
        target_personas=target_personas,

        expected_source_ids=(
            expected_source_ids
        ),

        expected_local_source_ids=(
            expected_local_source_ids
        ),

        expected_cross_agent_source_ids=(
            expected_cross_agent_source_ids
        ),

        # Runtime memory mappings
        memory_id_to_source_ids=(
            memory_id_to_source_ids
        ),

        memory_id_to_owner_agent_id=(
            memory_id_to_owner_agent_id
        ),
    )


__all__ = [
    "ExperimentMode",
    "WorkflowStatus",
    "NeedlePersonaWorkflowState",
    "NeedlePersonaEvaluationInput",
    "NeedlePersonaEvaluationResult",
    "NeedlePersonaEvaluationEvidence",
    "merge_unique_strings",
    "merge_string_list_mapping",
    "merge_string_mapping",
    "merge_review_mapping",
    "merge_decision_mapping",
    "merge_float_mapping",
    "build_initial_needle_persona_state",
    "build_needle_persona_evaluation_input",
]