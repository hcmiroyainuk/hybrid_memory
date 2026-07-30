from __future__ import annotations

from collections.abc import Mapping
from operator import add
from typing import (
    Annotated,
    Any,
    Literal,
    TypeAlias,
    TypedDict,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from src.llm import (
    AgentAnswer,
    MemoryReviewOutput,
    PromotionDecisionOutput,
)
from src.adapters import (
    MemoryAccessDecisionResult,
    MemoryAccessRequestResult,
)

from ...config.needle_persona_config import (
    ExperimentMode,
    validate_experiment_mode,
)
from ...data_preparing.needle_persona_models import (
    AGENT_PERSONA_MAP,
    NeedleEvidenceAssessment,
    NeedlePersonaSample,
    PrivateMemoryIndex,
)


# ---------------------------------------------------------------------------
# Workflow lifecycle
# ---------------------------------------------------------------------------

WorkflowStatus: TypeAlias = Literal[
    "initialized",
    "routed",
    "initial_memory_retrieved",
    "initial_evidence_assessed",
    "candidates_discovered",
    "requests_submitted",
    "requests_reviewed",
    "access_decided",
    "final_memory_retrieved",
    "answered",
    "insufficient_evidence",
    "refused",
    "failed",
]

TERMINAL_WORKFLOW_STATUSES: frozenset[str] = frozenset(
    {
        "answered",
        "insufficient_evidence",
        "refused",
        "failed",
    }
)


def is_terminal_workflow_status(
    status: str,
) -> bool:
    """
    Return whether a workflow status is terminal.
    """
    return str(status).strip() in (
        TERMINAL_WORKFLOW_STATUSES
    )


# ---------------------------------------------------------------------------
# Workflow-facing candidate DTO
# ---------------------------------------------------------------------------


class NeedleMemoryCandidate(BaseModel):
    """
    Reference to a private memory considered for cross-Agent access.

    The candidate intentionally contains no memory content. Candidate
    discovery may identify a private resource, but the responder must not see
    that resource's content until the sharing request has been approved and
    permission-filtered retrieval has run again.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    memory_id: str = Field(min_length=1)
    owner_agent_id: str = Field(min_length=1)

    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
    )

    reason: str = Field(
        min_length=1,
        description=(
            "Why this memory may contain information "
            "needed by the responder."
        ),
    )

    @field_validator(
        "memory_id",
        "owner_agent_id",
        "reason",
        mode="before",
    )
    @classmethod
    def clean_required_text(
        cls,
        value: Any,
    ) -> str:
        cleaned = " ".join(
            str(value or "").split()
        )

        if not cleaned:
            raise ValueError(
                "Candidate text fields cannot be empty."
            )

        return cleaned


# ---------------------------------------------------------------------------
# LangGraph reducers
# ---------------------------------------------------------------------------


def merge_unique_strings(
    current: list[str],
    update: list[str],
) -> list[str]:
    """
    Merge two string lists while preserving first-seen order.
    """
    result: list[str] = []

    for value in [
        *(current or []),
        *(update or []),
    ]:
        cleaned = str(value).strip()

        if cleaned and cleaned not in result:
            result.append(cleaned)

    return result


def _merge_mapping(
    current: Mapping[str, Any],
    update: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Merge mappings with values from ``update`` replacing matching keys.
    """
    result = dict(current or {})
    result.update(dict(update or {}))
    return result


def merge_string_mapping(
    current: dict[str, str],
    update: dict[str, str],
) -> dict[str, str]:
    return _merge_mapping(current, update)


def merge_float_mapping(
    current: dict[str, float],
    update: dict[str, float],
) -> dict[str, float]:
    return _merge_mapping(current, update)


def merge_candidate_mapping(
    current: dict[str, NeedleMemoryCandidate],
    update: dict[str, NeedleMemoryCandidate],
) -> dict[str, NeedleMemoryCandidate]:
    return _merge_mapping(current, update)


def merge_access_request_mapping(
    current: dict[
        str,
        MemoryAccessRequestResult,
    ],
    update: dict[
        str,
        MemoryAccessRequestResult,
    ],
) -> dict[
    str,
    MemoryAccessRequestResult,
]:
    return _merge_mapping(current, update)

def merge_evidence_assessment_mapping(
    current: dict[
        str,
        NeedleEvidenceAssessment,
    ],
    update: dict[
        str,
        NeedleEvidenceAssessment,
    ],
) -> dict[
    str,
    NeedleEvidenceAssessment,
]:
    """
    Merge Critic evidence assessments by memory ID.

    A later assessment replaces an earlier assessment
    for the same candidate memory.
    """
    return _merge_mapping(
        current,
        update,
    )


def merge_review_mapping(
    current: dict[str, MemoryReviewOutput],
    update: dict[str, MemoryReviewOutput],
) -> dict[str, MemoryReviewOutput]:
    return _merge_mapping(current, update)


def merge_coordinator_output_mapping(
    current: dict[
        str,
        PromotionDecisionOutput,
    ],
    update: dict[
        str,
        PromotionDecisionOutput,
    ],
) -> dict[
    str,
    PromotionDecisionOutput,
]:
    return _merge_mapping(current, update)


def merge_access_decision_mapping(
    current: dict[
        str,
        MemoryAccessDecisionResult,
    ],
    update: dict[
        str,
        MemoryAccessDecisionResult,
    ],
) -> dict[
    str,
    MemoryAccessDecisionResult,
]:
    return _merge_mapping(current, update)


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------


class NeedlePersonaWorkflowState(TypedDict):
    """
    Short-lived execution state for one Needle Persona workflow run.

    This state contains task context, memory IDs, Adapter DTOs, LLM outputs,
    and diagnostic information. It deliberately excludes:

    - the gold answer and alternative answers;
    - raw persona source documents;
    - Store, Service, Adapter, LLMClient, and Agent instances;
    - private-memory content that has not been authorised for the responder.

    Persistent long-term memory remains in the custom memory subsystem.
    LangGraph state is used only as workflow execution context.
    """

    # ------------------------------------------------------------------
    # Run identity and task input
    # ------------------------------------------------------------------

    run_id: str
    sample_id: str
    experiment_mode: ExperimentMode
    workflow_status: WorkflowStatus
    question: str

    # Private memory IDs produced by ingestion, grouped by owner Agent.
    # This is an ownership index, not permission to expose every memory to the
    # responder.
    private_memory_ids: dict[
        str,
        list[str],
    ]

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    selected_agent_ids: Annotated[
        list[str],
        merge_unique_strings,
    ]
    responder_agent_id: str

    # ------------------------------------------------------------------
    # Initial permission-filtered retrieval and evidence assessment
    # ------------------------------------------------------------------

    initial_retrieved_memory_ids: Annotated[
        list[str],
        merge_unique_strings,
    ]
    initial_memory_context: str

    # The responder's first attempt. It may be answered, insufficient, or
    # refused. When it is already answered, the workflow can terminate without
    # any cross-Agent request.
    initial_answer: AgentAnswer | None

    missing_information: Annotated[
        list[str],
        merge_unique_strings,
    ]

    # ------------------------------------------------------------------
    # Cross-Agent candidate discovery
    # ------------------------------------------------------------------

    sharing_round_count: int

    # Keyed by memory_id. Values contain identifiers and retrieval rationale
    # only; unauthorised private content must not be stored here.
    candidate_memories: Annotated[
        dict[str, NeedleMemoryCandidate],
        merge_candidate_mapping,
    ]

    # ------------------------------------------------------------------
    # Sharing requests and governance
    # ------------------------------------------------------------------

    # Keyed by memory_id. A later reviewed DTO replaces the earlier pending
    # DTO for the same memory.
    access_requests: Annotated[
        dict[
            str,
            MemoryAccessRequestResult,
        ],
        merge_access_request_mapping,
    ]

    # Structured evidence-contribution assessments produced
    # before the Critic governance recommendation.
    #
    # Keyed by candidate memory_id.
    critic_evidence_assessments: Annotated[
        dict[
            str,
            NeedleEvidenceAssessment,
        ],
        merge_evidence_assessment_mapping,
    ]

    # Structured Critic LLM outputs, keyed by memory_id.
    critic_reviews: Annotated[
        dict[str, MemoryReviewOutput],
        merge_review_mapping,
    ]

    # Independent semantic evidence rechecks performed by the Coordinator
    # for candidates negatively assessed by the Critic.
    #
    # Keyed by candidate memory_id.
    coordinator_evidence_rechecks: Annotated[
        dict[
            str,
            NeedleEvidenceAssessment,
        ],
        merge_evidence_assessment_mapping,
    ]

    # Structured final Coordinator decisions, keyed by memory_id.
    coordinator_outputs: Annotated[
        dict[
            str,
            PromotionDecisionOutput,
        ],
        merge_coordinator_output_mapping,
    ]

    # Applied Gateway results, keyed by memory_id.
    access_decisions: Annotated[
        dict[
            str,
            MemoryAccessDecisionResult,
        ],
        merge_access_decision_mapping,
    ]

    approved_memory_ids: Annotated[
        list[str],
        merge_unique_strings,
    ]

    rejected_memory_ids: Annotated[
        list[str],
        merge_unique_strings,
    ]

    # Memories initially rejected by the Critic but recovered after the
    # Coordinator's independent semantic evidence recheck.
    recovered_memory_ids: Annotated[
        list[str],
        merge_unique_strings,
    ]

    # ------------------------------------------------------------------
    # Final permission-filtered retrieval and answer
    # ------------------------------------------------------------------

    final_retrieved_memory_ids: Annotated[
        list[str],
        merge_unique_strings,
    ]
    final_memory_context: str
    final_answer: AgentAnswer | None

    # ------------------------------------------------------------------
    # Observability and failure handling
    # ------------------------------------------------------------------

    # Duplicates are preserved because retries may execute a node more than
    # once and the trace should show the actual execution order.
    node_trace: Annotated[
        list[str],
        add,
    ]

    errors: Annotated[
        list[str],
        add,
    ]
    warnings: Annotated[
        list[str],
        add,
    ]

    # Example keys:
    # "retrieve_initial_memory_ms", "review_access_requests_ms".
    node_metrics: Annotated[
        dict[str, float],
        merge_float_mapping,
    ]


# ---------------------------------------------------------------------------
# Initial-state construction
# ---------------------------------------------------------------------------


def build_initial_needle_persona_state(
    *,
    sample: NeedlePersonaSample,
    memory_index: PrivateMemoryIndex,
    experiment_mode: ExperimentMode | str,
) -> NeedlePersonaWorkflowState:
    """
    Build a safe initial state for one ``sample × mode`` execution.

    Only the benchmark question and already-ingested private memory IDs enter
    the task graph. Held-out answers and raw persona documents remain outside
    the workflow and are supplied to the evaluator only after execution.
    """
    if (
        memory_index.sample_id
        != sample.sample_id
    ):
        raise ValueError(
            "memory_index.sample_id does not match "
            "sample.sample_id: "
            f"{memory_index.sample_id!r} != "
            f"{sample.sample_id!r}."
        )

    run_id = str(
        memory_index.run_id
    ).strip()
    sample_id = str(
        sample.sample_id
    ).strip()
    question = str(
        sample.question
    ).strip()

    if not run_id:
        raise ValueError(
            "memory_index.run_id cannot be empty."
        )

    if not sample_id:
        raise ValueError(
            "sample.sample_id cannot be empty."
        )

    if not question:
        raise ValueError(
            "sample.question cannot be empty."
        )

    mode = validate_experiment_mode(
        str(experiment_mode)
    )

    expected_agent_ids = set(
        AGENT_PERSONA_MAP
    )
    supplied_agent_ids = set(
        memory_index.private_memory_ids
    )

    unknown_agent_ids = (
        supplied_agent_ids - expected_agent_ids
    )

    if unknown_agent_ids:
        raise ValueError(
            "PrivateMemoryIndex contains unknown "
            "persona Agent IDs: "
            f"{sorted(unknown_agent_ids)}."
        )

    private_memory_ids: dict[
        str,
        list[str],
    ] = {}

    all_memory_ids: set[str] = set()

    for agent_id in AGENT_PERSONA_MAP:
        cleaned_ids: list[str] = []

        for memory_id in (
            memory_index.private_memory_ids.get(
                agent_id,
                [],
            )
        ):
            clean_memory_id = str(
                memory_id
            ).strip()

            if not clean_memory_id:
                continue

            if clean_memory_id in cleaned_ids:
                raise ValueError(
                    "Duplicate memory ID within "
                    f"{agent_id!r}: "
                    f"{clean_memory_id!r}."
                )

            if clean_memory_id in all_memory_ids:
                raise ValueError(
                    "A private memory ID cannot belong "
                    "to multiple persona Agents: "
                    f"{clean_memory_id!r}."
                )

            cleaned_ids.append(
                clean_memory_id
            )
            all_memory_ids.add(
                clean_memory_id
            )

        private_memory_ids[
            agent_id
        ] = cleaned_ids

    return NeedlePersonaWorkflowState(
        run_id=run_id,
        sample_id=sample_id,
        experiment_mode=mode,
        workflow_status="initialized",
        question=question,
        private_memory_ids=private_memory_ids,
        selected_agent_ids=[],
        responder_agent_id="",
        initial_retrieved_memory_ids=[],
        initial_memory_context="",
        initial_answer=None,
        missing_information=[],
        sharing_round_count=0,
        candidate_memories={},
        access_requests={},
        critic_evidence_assessments={},
        critic_reviews={},
        coordinator_evidence_rechecks={},
        coordinator_outputs={},
        access_decisions={},
        approved_memory_ids=[],
        rejected_memory_ids=[],
        recovered_memory_ids=[],
        final_retrieved_memory_ids=[],
        final_memory_context="",
        final_answer=None,
        node_trace=[],
        errors=[],
        warnings=[],
        node_metrics={},
    )


__all__ = [
    "WorkflowStatus",
    "TERMINAL_WORKFLOW_STATUSES",
    "is_terminal_workflow_status",
    "NeedleMemoryCandidate",
    "NeedlePersonaWorkflowState",
    "merge_unique_strings",
    "merge_string_mapping",
    "merge_float_mapping",
    "merge_candidate_mapping",
    "merge_access_request_mapping",
    "merge_evidence_assessment_mapping",
    "merge_review_mapping",
    "merge_coordinator_output_mapping",
    "merge_access_decision_mapping",
    "build_initial_needle_persona_state",
]