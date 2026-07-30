"""
Evaluation components for InformativeBench Needle in the Persona.
"""

from .needle_persona_evaluator import (
    AnswerMetricResult,
    NeedlePersonaEvaluationInput,
    NeedlePersonaEvaluationResult,
    NeedlePersonaEvaluator,
    NeedlePersonaEvaluatorConfig,
    NeedlePersonaEvaluatorError,
    build_memory_id_to_owner_agent_id,
    build_memory_id_to_source_ids,
    build_needle_persona_evaluation_input,
    evaluate_needle_persona,
    extract_target_personas,
)

from .needle_persona_evaluation_runner import (
    ProgressCallback,
    RunRuntimeFactory,
    NeedlePersonaBatchReport,
    NeedlePersonaBatchRunError,
    NeedlePersonaEvaluationRunner,
    NeedlePersonaEvaluationRuntime,
    NeedlePersonaIngestionProtocol,
    NeedlePersonaLoaderProtocol,
    NeedlePersonaRunFailure,
    NeedlePersonaRunTrace,
    build_needle_persona_summary,
)

__all__ = [
    "NeedlePersonaEvaluatorError",
    "NeedlePersonaEvaluationInput",
    "NeedlePersonaEvaluationResult",
    "AnswerMetricResult",
    "NeedlePersonaEvaluatorConfig",
    "NeedlePersonaEvaluator",
    "build_needle_persona_evaluation_input",
    "extract_target_personas",
    "build_memory_id_to_owner_agent_id",
    "build_memory_id_to_source_ids",
    "evaluate_needle_persona",
    "NeedlePersonaLoaderProtocol",
    "NeedlePersonaIngestionProtocol",
    "RunRuntimeFactory",
    "ProgressCallback",
    "NeedlePersonaBatchRunError",
    "NeedlePersonaEvaluationRuntime",
    "NeedlePersonaRunFailure",
    "NeedlePersonaRunTrace",
    "NeedlePersonaBatchReport",
    "NeedlePersonaEvaluationRunner",
    "build_needle_persona_summary",
]