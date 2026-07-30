from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Literal,
    Sequence,
    TypeAlias,
    cast,
)


# ---------------------------------------------------------------------------
# Experiment modes
# ---------------------------------------------------------------------------

ExperimentMode: TypeAlias = Literal[
    "private_only",
    "ungoverned_shared",
    "governed_shared",
]

DEFAULT_EXPERIMENT_MODES: tuple[
    ExperimentMode,
    ...,
] = (
    "private_only",
    "ungoverned_shared",
    "governed_shared",
)

VALID_EXPERIMENT_MODES: frozenset[str] = frozenset(
    DEFAULT_EXPERIMENT_MODES
)


# The current first implementation uses deterministic responder routing and
# owner-scoped candidate retrieval. New strategies can be added later without
# changing the workflow-facing configuration structure.
RoutingStrategy: TypeAlias = Literal[
    "deterministic",
]

CandidateDiscoveryStrategy: TypeAlias = Literal[
    "owner_scoped_retrieval",
]

SharedAccessLifetime: TypeAlias = Literal[
    "runtime",
]


def validate_experiment_mode(
    mode: str,
) -> ExperimentMode:
    """
    Validate and normalise one experiment mode.
    """
    clean_mode = str(mode).strip()

    if clean_mode not in VALID_EXPERIMENT_MODES:
        raise ValueError(
            "Unsupported experiment mode: "
            f"{clean_mode!r}. Expected one of "
            f"{sorted(VALID_EXPERIMENT_MODES)}."
        )

    return cast(
        ExperimentMode,
        clean_mode,
    )


def normalise_experiment_modes(
    modes: Sequence[str],
) -> tuple[ExperimentMode, ...]:
    """
    Validate modes, remove duplicates, and preserve input order.
    """
    cleaned: list[ExperimentMode] = []

    for mode in modes:
        validated = validate_experiment_mode(
            mode
        )

        if validated not in cleaned:
            cleaned.append(validated)

    if not cleaned:
        raise ValueError(
            "At least one experiment mode is required."
        )

    return tuple(cleaned)


# ---------------------------------------------------------------------------
# Node behaviour
# ---------------------------------------------------------------------------


@dataclass(
    frozen=True,
    slots=True,
)
class NeedlePersonaNodeConfig:
    """
    Behavioural configuration used by Needle Persona workflow nodes.

    The first implementation intentionally keeps the workflow deterministic
    and inspectable:

    - the responder is selected deterministically;
    - initial retrieval includes only memories currently accessible to the
      responder;
    - candidate discovery runs inside each owner Agent's memory space;
    - unauthorised private-memory content is not exposed to the responder;
    - at most one cross-Agent sharing round is performed.
    """

    routing_strategy: RoutingStrategy = (
        "deterministic"
    )

    candidate_discovery_strategy: (
        CandidateDiscoveryStrategy
    ) = "owner_scoped_retrieval"

    # Responder retrieval before any new sharing request.
    initial_retrieval_top_k: int = 8

    # Candidate retrieval is performed separately in each owner Agent's
    # private memory space.
    candidate_top_k_per_owner: int = 5

    # Global cap after merging and deduplicating candidates from all owners.
    max_candidate_memories: int = 8

    # Responder retrieval after approved memories become accessible.
    final_retrieval_top_k: int = 12

    # Maximum number of memories rendered into one LLM memory context.
    memory_context_limit: int = 12

    # One round is sufficient for the initial reproducible experiment and
    # prevents accidental workflow loops.
    max_sharing_rounds: int = 1

    # Used when the semantic retriever is absent or fails. The fallback may
    # rank only memory IDs already permitted for the current retrieval step.
    allow_lexical_retrieval_fallback: bool = True

    # Remove hallucinated memory/source/agent references from AgentAnswer
    # before the result enters workflow state.
    sanitise_answer_references: bool = True

    # Keep compact, serialisable node updates for debugging and evaluation.
    save_node_trace: bool = True

    def __post_init__(self) -> None:
        positive_fields = (
            "initial_retrieval_top_k",
            "candidate_top_k_per_owner",
            "max_candidate_memories",
            "final_retrieval_top_k",
            "memory_context_limit",
            "max_sharing_rounds",
        )

        for field_name in positive_fields:
            value = getattr(
                self,
                field_name,
            )

            if value <= 0:
                raise ValueError(
                    f"{field_name} must be greater "
                    "than zero."
                )

        if (
            self.routing_strategy
            != "deterministic"
        ):
            raise ValueError(
                "The current Needle Persona workflow "
                "supports only deterministic routing."
            )

        if (
            self.candidate_discovery_strategy
            != "owner_scoped_retrieval"
        ):
            raise ValueError(
                "The current Needle Persona workflow "
                "supports only owner-scoped candidate "
                "retrieval."
            )


# ---------------------------------------------------------------------------
# LangGraph construction
# ---------------------------------------------------------------------------


@dataclass(
    frozen=True,
    slots=True,
)
class NeedlePersonaWorkflowConfig:
    """
    LangGraph-level configuration.

    Node behaviour belongs to ``node_config``. This class controls workflow
    validation and optional graph persistence.
    """

    node_config: NeedlePersonaNodeConfig = field(
        default_factory=NeedlePersonaNodeConfig
    )

    validate_initial_state: bool = True
    validate_final_state: bool = True

    # The benchmark currently executes short-lived isolated runs, so a
    # checkpointer is optional. It can be supplied later for durable execution.
    checkpointer: Any | None = None


# ---------------------------------------------------------------------------
# Dataset, ingestion, and batch execution
# ---------------------------------------------------------------------------


@dataclass(
    frozen=True,
    slots=True,
)
class NeedlePersonaExperimentConfig:
    """
    Top-level configuration for one Needle Persona batch experiment.

    Sharing grants are persistent for the lifetime of one run-specific
    MemoryStore. Therefore every ``sample × mode`` execution must receive a
    fresh isolated runtime.
    """

    dataset_path: Path = Path(
        "data/dataset_2hop.jsonl"
    )

    output_root: Path = Path(
        "outputs/informativebench/"
        "needle_persona"
    )

    runtime_root: Path = Path(
        "data/needle_persona_runtime"
    )

    modes: tuple[
        ExperimentMode,
        ...,
    ] = DEFAULT_EXPERIMENT_MODES

    # None means the complete dataset.
    sample_limit: int | None = None

    # Loader settings fixed for the benchmark protocol.
    source_granularity: Literal[
        "persona_document"
    ] = "persona_document"

    include_collaborative_chat: bool = False
    retain_raw_fields_in_metadata: bool = False
    loader_strict: bool = True

    # Ingestion writes persona-owned memories as private resources.
    ingestion_include_task_context: bool = False
    ingestion_rollback_on_error: bool = True
    require_memory_per_source: bool = True

    # A sharing grant remains valid until the run-specific store is discarded.
    shared_access_lifetime: (
        SharedAccessLifetime
    ) = "runtime"

    # Required because ACL changes are persistent during a runtime.
    fresh_runtime_per_sample_mode: bool = True

    reset_runtime_root_before_batch: bool = True
    overwrite_output: bool = False
    save_detailed_trace: bool = True

    workflow_config: (
        NeedlePersonaWorkflowConfig
    ) = field(
        default_factory=(
            NeedlePersonaWorkflowConfig
        )
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dataset_path",
            Path(self.dataset_path),
        )
        object.__setattr__(
            self,
            "output_root",
            Path(self.output_root),
        )
        object.__setattr__(
            self,
            "runtime_root",
            Path(self.runtime_root),
        )
        object.__setattr__(
            self,
            "modes",
            normalise_experiment_modes(
                self.modes
            ),
        )

        if (
            self.sample_limit is not None
            and self.sample_limit <= 0
        ):
            raise ValueError(
                "sample_limit must be greater than "
                "zero or None."
            )

        if (
            self.source_granularity
            != "persona_document"
        ):
            raise ValueError(
                "The current Needle Persona protocol "
                "requires persona_document source "
                "granularity."
            )

        if (
            self.shared_access_lifetime
            != "runtime"
        ):
            raise ValueError(
                "The current implementation supports "
                "runtime-persistent sharing only."
            )

        if not self.fresh_runtime_per_sample_mode:
            raise ValueError(
                "fresh_runtime_per_sample_mode must "
                "remain True because sharing ACLs are "
                "persistent within a runtime."
            )

    def as_runtime_metadata(
        self,
    ) -> dict[str, Any]:
        """
        Return a compact JSON-compatible configuration summary.

        This is suitable for the experiment manifest and does not include the
        optional checkpointer object.
        """
        node = self.workflow_config.node_config

        return {
            "experiment_modes": list(
                self.modes
            ),
            "dataset_path": str(
                self.dataset_path
            ),
            "output_root": str(
                self.output_root
            ),
            "runtime_root": str(
                self.runtime_root
            ),
            "sample_limit": self.sample_limit,
            "source_granularity": (
                self.source_granularity
            ),
            "include_collaborative_chat": (
                self.include_collaborative_chat
            ),
            "retain_raw_fields_in_metadata": (
                self.retain_raw_fields_in_metadata
            ),
            "loader_strict": self.loader_strict,
            "shared_access_lifetime": (
                self.shared_access_lifetime
            ),
            "fresh_runtime_per_sample_mode": (
                self.fresh_runtime_per_sample_mode
            ),
            "routing_strategy": (
                node.routing_strategy
            ),
            "candidate_discovery_strategy": (
                node.candidate_discovery_strategy
            ),
            "initial_retrieval_top_k": (
                node.initial_retrieval_top_k
            ),
            "candidate_top_k_per_owner": (
                node.candidate_top_k_per_owner
            ),
            "max_candidate_memories": (
                node.max_candidate_memories
            ),
            "final_retrieval_top_k": (
                node.final_retrieval_top_k
            ),
            "memory_context_limit": (
                node.memory_context_limit
            ),
            "max_sharing_rounds": (
                node.max_sharing_rounds
            ),
            "allow_lexical_retrieval_fallback": (
                node.allow_lexical_retrieval_fallback
            ),
            "sanitise_answer_references": (
                node.sanitise_answer_references
            ),
            "save_node_trace": (
                node.save_node_trace
            ),
        }


DEFAULT_NODE_CONFIG = (
    NeedlePersonaNodeConfig()
)

DEFAULT_WORKFLOW_CONFIG = (
    NeedlePersonaWorkflowConfig()
)

DEFAULT_EXPERIMENT_CONFIG = (
    NeedlePersonaExperimentConfig()
)


__all__ = [
    "ExperimentMode",
    "RoutingStrategy",
    "CandidateDiscoveryStrategy",
    "SharedAccessLifetime",
    "DEFAULT_EXPERIMENT_MODES",
    "VALID_EXPERIMENT_MODES",
    "validate_experiment_mode",
    "normalise_experiment_modes",
    "NeedlePersonaNodeConfig",
    "NeedlePersonaWorkflowConfig",
    "NeedlePersonaExperimentConfig",
    "DEFAULT_NODE_CONFIG",
    "DEFAULT_WORKFLOW_CONFIG",
    "DEFAULT_EXPERIMENT_CONFIG",
]