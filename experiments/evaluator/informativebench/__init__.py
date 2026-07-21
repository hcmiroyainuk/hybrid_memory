"""
InformativeBench Needle in the Persona evaluation package.

The package exposes:
- benchmark data models and JSONL loading;
- persona-private-memory ingestion;
- LangGraph workflow state, nodes, and orchestration;
- deterministic answer, evidence, governance, and transfer evaluation;
- sequential batch execution, metric aggregation, and result persistence.
"""

from .needle_persona_answer_metrics import (
    AnswerMetricResult,
    calculate_answer_accuracy,
    calculate_entity_f1,
    calculate_entity_metrics,
    calculate_entity_precision,
    calculate_entity_recall,
    evaluate_answer,
    extract_entity_set,
    find_best_reference_match,
    normalize_answer_text,
    normalize_entity,
    split_answer_entities,
)
from .needle_persona_evaluation_runner import (
    DEFAULT_EXPERIMENT_MODES,
    NeedlePersonaBatchReport,
    NeedlePersonaBatchRunError,
    NeedlePersonaEvaluationRunner,
    NeedlePersonaEvaluationRuntime,
    NeedlePersonaRunFailure,
    ProgressCallback,
    build_needle_persona_summary,
)
from .needle_persona_evaluator import (
    NeedlePersonaEvaluator,
    NeedlePersonaEvaluatorConfig,
    NeedlePersonaEvaluatorError,
    evaluate_needle_persona,
)
from .needle_persona_evidence import (
    DEFAULT_PERSONA_ALIASES,
    NeedlePersonaEvidenceError,
    build_expected_source_ids,
    build_memory_id_to_owner_agent_id,
    build_memory_id_to_source_ids,
    build_needle_persona_evidence,
    build_source_id_to_persona,
    extract_target_personas,
    map_memory_ids_to_owner_agent_ids,
    map_memory_ids_to_source_ids,
    validate_needle_persona_evidence,
)
from .needle_persona_ingestion import (
    MemoryServiceProtocol,
    NeedlePersonaIngestionError,
    NeedlePersonaIngestionService,
    build_default_persona_agents,
    build_default_worker_prompts,
    ingest_samples,
)
from .needle_persona_loader import (
    DialogueTurn,
    NeedlePersonaDatasetError,
    NeedlePersonaLoader,
    NeedlePersonaRecordError,
    ParsedConversation,
    SourceGranularity,
    load_needle_persona_dataset,
)
from .needle_persona_models import (
    AGENT_PERSONA_MAP,
    PERSONA_AGENT_MAP,
    PERSONA_NAMES,
    NeedlePersonaSample,
    PersonaName,
    PersonaSource,
    PrivateMemoryIndex,
    SourceType,
)
from .needle_persona_nodes import (
    NodeMemoryServiceProtocol,
    NodePromotionServiceProtocol,
    NeedlePersonaNodeConfig,
    NeedlePersonaNodeDependencies,
    NeedlePersonaNodeError,
    NeedlePersonaNodes,
    RoutingStrategy,
    build_default_governance_agents,
    build_failure_state_update,
)
from .needle_persona_state import (
    ExperimentMode,
    NeedlePersonaEvaluationEvidence,
    NeedlePersonaEvaluationInput,
    NeedlePersonaEvaluationResult,
    NeedlePersonaWorkflowState,
    WorkflowStatus,
    build_initial_needle_persona_state,
    build_needle_persona_evaluation_input,
    merge_decision_mapping,
    merge_float_mapping,
    merge_review_mapping,
    merge_string_list_mapping,
    merge_string_mapping,
    merge_unique_strings,
)
from .needle_persona_workflow import (
    NeedlePersonaWorkflow,
    NeedlePersonaWorkflowConfig,
    NeedlePersonaWorkflowError,
    build_needle_persona_workflow,
    create_needle_persona_workflow,
)


__all__ = [
    # Models and mappings
    "PersonaName",
    "SourceType",
    "PERSONA_NAMES",
    "PERSONA_AGENT_MAP",
    "AGENT_PERSONA_MAP",
    "PersonaSource",
    "NeedlePersonaSample",
    "PrivateMemoryIndex",

    # Dataset loader
    "SourceGranularity",
    "DialogueTurn",
    "ParsedConversation",
    "NeedlePersonaDatasetError",
    "NeedlePersonaRecordError",
    "NeedlePersonaLoader",
    "load_needle_persona_dataset",

    # Private-memory ingestion
    "MemoryServiceProtocol",
    "NeedlePersonaIngestionError",
    "NeedlePersonaIngestionService",
    "build_default_persona_agents",
    "build_default_worker_prompts",
    "ingest_samples",

    # Workflow state
    "ExperimentMode",
    "WorkflowStatus",
    "NeedlePersonaWorkflowState",
    "NeedlePersonaEvaluationEvidence",
    "NeedlePersonaEvaluationInput",
    "NeedlePersonaEvaluationResult",
    "merge_unique_strings",
    "merge_string_list_mapping",
    "merge_string_mapping",
    "merge_review_mapping",
    "merge_decision_mapping",
    "merge_float_mapping",
    "build_initial_needle_persona_state",
    "build_needle_persona_evaluation_input",

    # Workflow nodes
    "RoutingStrategy",
    "NodeMemoryServiceProtocol",
    "NodePromotionServiceProtocol",
    "NeedlePersonaNodeError",
    "NeedlePersonaNodeConfig",
    "NeedlePersonaNodeDependencies",
    "NeedlePersonaNodes",
    "build_default_governance_agents",
    "build_failure_state_update",

    # Workflow orchestration
    "NeedlePersonaWorkflowError",
    "NeedlePersonaWorkflowConfig",
    "NeedlePersonaWorkflow",
    "build_needle_persona_workflow",
    "create_needle_persona_workflow",

    # Answer metrics
    "AnswerMetricResult",
    "normalize_answer_text",
    "normalize_entity",
    "split_answer_entities",
    "extract_entity_set",
    "calculate_answer_accuracy",
    "calculate_entity_precision",
    "calculate_entity_recall",
    "calculate_entity_f1",
    "calculate_entity_metrics",
    "find_best_reference_match",
    "evaluate_answer",

    # Standard evidence
    "NeedlePersonaEvidenceError",
    "DEFAULT_PERSONA_ALIASES",
    "extract_target_personas",
    "build_source_id_to_persona",
    "build_expected_source_ids",
    "build_memory_id_to_owner_agent_id",
    "build_memory_id_to_source_ids",
    "map_memory_ids_to_source_ids",
    "map_memory_ids_to_owner_agent_ids",
    "validate_needle_persona_evidence",
    "build_needle_persona_evidence",

    # Evaluator
    "NeedlePersonaEvaluatorError",
    "NeedlePersonaEvaluatorConfig",
    "NeedlePersonaEvaluator",
    "evaluate_needle_persona",

    # Batch runner and aggregation
    "DEFAULT_EXPERIMENT_MODES",
    "NeedlePersonaBatchRunError",
    "NeedlePersonaEvaluationRuntime",
    "NeedlePersonaRunFailure",
    "NeedlePersonaBatchReport",
    "ProgressCallback",
    "NeedlePersonaEvaluationRunner",
    "build_needle_persona_summary",
]