from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .needle_persona_answer_metrics import (
    AnswerMetricResult,
    evaluate_answer,
)
from .needle_persona_evidence import (
    NeedlePersonaEvidenceError,
    map_memory_ids_to_source_ids,
)
from .needle_persona_models import (
    AGENT_PERSONA_MAP,
)
from .needle_persona_state import (
    ExperimentMode,
    NeedlePersonaEvaluationInput,
    NeedlePersonaEvaluationResult,
)


class NeedlePersonaEvaluatorError(ValueError):
    """
    Raised when evaluator input is incomplete or internally inconsistent.
    """


@dataclass(frozen=True, slots=True)
class NeedlePersonaEvaluatorConfig:
    """
    Final evaluator configuration.

    The experiment uses strict mapping validation so missing memory-to-source
    mappings cannot silently distort retrieval or governance metrics.
    """

    strict_memory_mapping: bool = True
    validate_input: bool = True


class NeedlePersonaEvaluator:
    """
    Deterministic evaluator for one completed Needle in the Persona run.

    The evaluator:
    - compares the predicted answer with accepted references;
    - maps runtime memory IDs back to benchmark source IDs;
    - computes retrieval, utilisation, sharing, governance, and transfer
      metrics;
    - performs no LLM calls, retrieval, memory mutation, or workflow execution.
    """

    _VALID_MODES: set[str] = {
        "private_only",
        "ungoverned_shared",
        "governed_shared",
    }

    def __init__(
        self,
        config: NeedlePersonaEvaluatorConfig | None = None,
    ) -> None:
        self.config = config or NeedlePersonaEvaluatorConfig()

    def evaluate(
        self,
        evaluation_input: NeedlePersonaEvaluationInput,
    ) -> NeedlePersonaEvaluationResult:
        """
        Evaluate one finished workflow run.
        """
        if self.config.validate_input:
            self._validate_input(evaluation_input)

        answer_metrics = evaluate_answer(
            evaluation_input["predicted_answer"],
            evaluation_input["accepted_answers"],
        )

        memory_id_to_source_ids = evaluation_input[
            "memory_id_to_source_ids"
        ]

        retrieved_source_ids = self._map_sources(
            evaluation_input[
                "responder_retrieved_memory_ids"
            ],
            memory_id_to_source_ids,
        )
        used_source_ids = self._map_sources(
            evaluation_input["used_memory_ids"],
            memory_id_to_source_ids,
        )
        shared_source_ids = self._map_sources(
            evaluation_input["shared_memory_ids"],
            memory_id_to_source_ids,
        )

        expected_source_ids = self._clean_unique(
            evaluation_input["expected_source_ids"]
        )
        expected_local_source_ids = self._clean_unique(
            evaluation_input[
                "expected_local_source_ids"
            ]
        )
        expected_cross_agent_source_ids = self._clean_unique(
            evaluation_input[
                "expected_cross_agent_source_ids"
            ]
        )

        expected_set = set(expected_source_ids)
        cross_agent_expected_set = set(
            expected_cross_agent_source_ids
        )
        retrieved_set = set(retrieved_source_ids)
        used_set = set(used_source_ids)
        shared_set = set(shared_source_ids)

        retrieved_target_source_ids = self._ordered_intersection(
            expected_source_ids,
            retrieved_set,
        )
        missing_retrieved_target_source_ids = self._ordered_difference(
            expected_source_ids,
            retrieved_set,
        )

        target_source_retrieval_recall = self._recall(
            retrieved_set,
            expected_set,
        )
        all_target_sources_retrieved = (
            expected_set <= retrieved_set
        )

        cross_agent_retrieved_source_ids = (
            self._ordered_intersection(
                expected_cross_agent_source_ids,
                retrieved_set,
            )
        )
        cross_agent_source_retrieval_recall = self._recall(
            retrieved_set,
            cross_agent_expected_set,
        )
        cross_agent_memory_hit = bool(
            cross_agent_retrieved_source_ids
        )

        used_target_source_ids = self._ordered_intersection(
            expected_source_ids,
            used_set,
        )
        target_source_utilisation_recall = self._recall(
            used_set,
            expected_set,
        )

        cross_agent_used_source_ids = self._ordered_intersection(
            expected_cross_agent_source_ids,
            used_set,
        )
        cross_agent_source_utilisation_recall = self._recall(
            used_set,
            cross_agent_expected_set,
        )
        cross_agent_source_utilised = bool(
            cross_agent_used_source_ids
        )

        correctly_shared_source_ids = self._ordered_intersection(
            expected_cross_agent_source_ids,
            shared_set,
        )
        over_shared_source_ids = self._ordered_difference(
            shared_source_ids,
            cross_agent_expected_set,
        )
        missing_shared_cross_agent_source_ids = (
            self._ordered_difference(
                expected_cross_agent_source_ids,
                shared_set,
            )
        )

        mode = evaluation_input["experiment_mode"]

        (
            shared_cross_agent_source_recall,
            governance_precision,
            governance_recall,
            over_sharing_rate,
            under_sharing_rate,
        ) = self._calculate_governance_metrics(
            mode=mode,
            shared_source_ids=shared_source_ids,
            correctly_shared_source_ids=(
                correctly_shared_source_ids
            ),
            over_shared_source_ids=over_shared_source_ids,
            expected_cross_agent_source_ids=(
                expected_cross_agent_source_ids
            ),
            missing_shared_cross_agent_source_ids=(
                missing_shared_cross_agent_source_ids
            ),
        )

        transfer_success = self._calculate_transfer_success(
            mode=mode,
            has_cross_agent_target=bool(
                expected_cross_agent_source_ids
            ),
            answer_metrics=answer_metrics,
            shared_cross_agent_source_recall=(
                shared_cross_agent_source_recall
            ),
            cross_agent_source_retrieval_recall=(
                cross_agent_source_retrieval_recall
            ),
            cross_agent_source_utilisation_recall=(
                cross_agent_source_utilisation_recall
            ),
        )

        critic_approved_memory_count = sum(
            1
            for review in evaluation_input[
                "critic_reviews"
            ].values()
            if self._field(review, "recommendation")
            == "approve"
        )

        coordinator_approved_memory_count = sum(
            1
            for decision in evaluation_input[
                "promotion_decisions"
            ].values()
            if (
                self._field(decision, "decision")
                == "approve"
                and self._field(
                    decision,
                    "target_scope",
                )
                == "shared"
            )
        )

        return NeedlePersonaEvaluationResult(
            # Run identity
            run_id=evaluation_input["run_id"],
            sample_id=evaluation_input["sample_id"],
            experiment_mode=mode,
            question=evaluation_input["question"],

            # Answer output and matching
            predicted_answer=answer_metrics.predicted_answer,
            matched_reference_answer=(
                answer_metrics.matched_reference_answer
            ),
            predicted_entities=list(
                answer_metrics.predicted_entities
            ),
            matched_reference_entities=list(
                answer_metrics.matched_reference_entities
            ),
            answer_accuracy=answer_metrics.answer_accuracy,
            entity_precision=answer_metrics.entity_precision,
            entity_recall=answer_metrics.entity_recall,
            entity_f1=answer_metrics.entity_f1,

            # Expected evidence
            expected_source_ids=expected_source_ids,
            expected_local_source_ids=(
                expected_local_source_ids
            ),
            expected_cross_agent_source_ids=(
                expected_cross_agent_source_ids
            ),

            # Retrieval diagnostics and metrics
            retrieved_source_ids=retrieved_source_ids,
            retrieved_target_source_ids=(
                retrieved_target_source_ids
            ),
            missing_retrieved_target_source_ids=(
                missing_retrieved_target_source_ids
            ),
            target_source_retrieval_recall=(
                target_source_retrieval_recall
            ),
            all_target_sources_retrieved=(
                all_target_sources_retrieved
            ),
            cross_agent_retrieved_source_ids=(
                cross_agent_retrieved_source_ids
            ),
            cross_agent_source_retrieval_recall=(
                cross_agent_source_retrieval_recall
            ),
            cross_agent_memory_hit=cross_agent_memory_hit,

            # Memory utilisation diagnostics and metrics
            used_source_ids=used_source_ids,
            used_target_source_ids=used_target_source_ids,
            target_source_utilisation_recall=(
                target_source_utilisation_recall
            ),
            cross_agent_used_source_ids=(
                cross_agent_used_source_ids
            ),
            cross_agent_source_utilisation_recall=(
                cross_agent_source_utilisation_recall
            ),
            cross_agent_source_utilised=(
                cross_agent_source_utilised
            ),

            # Sharing and governance diagnostics and metrics
            shared_source_ids=shared_source_ids,
            correctly_shared_source_ids=(
                correctly_shared_source_ids
            ),
            over_shared_source_ids=(
                over_shared_source_ids
            ),
            missing_shared_cross_agent_source_ids=(
                missing_shared_cross_agent_source_ids
            ),
            shared_cross_agent_source_recall=(
                shared_cross_agent_source_recall
            ),
            governance_precision=governance_precision,
            governance_recall=governance_recall,
            over_sharing_rate=over_sharing_rate,
            under_sharing_rate=under_sharing_rate,

            # Governance trace counts
            candidate_memory_count=len(
                self._clean_unique(
                    evaluation_input[
                        "candidate_memory_ids"
                    ]
                )
            ),
            critic_approved_memory_count=(
                critic_approved_memory_count
            ),
            coordinator_approved_memory_count=(
                coordinator_approved_memory_count
            ),
            shared_memory_count=len(
                self._clean_unique(
                    evaluation_input[
                        "shared_memory_ids"
                    ]
                )
            ),
            rejected_memory_count=len(
                self._clean_unique(
                    evaluation_input[
                        "rejected_memory_ids"
                    ]
                )
            ),

            # End-to-end result
            transfer_success=transfer_success,
        )

    def evaluate_many(
        self,
        evaluation_inputs: Sequence[
            NeedlePersonaEvaluationInput
        ],
    ) -> list[NeedlePersonaEvaluationResult]:
        """
        Evaluate a sequence of completed runs in input order.
        """
        return [
            self.evaluate(evaluation_input)
            for evaluation_input in evaluation_inputs
        ]

    def _map_sources(
        self,
        memory_ids: Sequence[str],
        memory_id_to_source_ids: Mapping[
            str,
            Sequence[str],
        ],
    ) -> list[str]:
        try:
            return map_memory_ids_to_source_ids(
                memory_ids,
                memory_id_to_source_ids,
                strict=self.config.strict_memory_mapping,
            )
        except NeedlePersonaEvidenceError as error:
            raise NeedlePersonaEvaluatorError(
                str(error)
            ) from error

    @classmethod
    def _validate_input(
        cls,
        evaluation_input: NeedlePersonaEvaluationInput,
    ) -> None:
        for field_name in (
            "run_id",
            "sample_id",
            "question",
            "predicted_answer",
            "responder_agent_id",
        ):
            value = str(
                evaluation_input.get(
                    field_name,
                    "",
                )
            ).strip()

            if not value:
                raise NeedlePersonaEvaluatorError(
                    f"Evaluation input field "
                    f"{field_name!r} cannot be empty."
                )

        mode = evaluation_input.get(
            "experiment_mode"
        )

        if mode not in cls._VALID_MODES:
            raise NeedlePersonaEvaluatorError(
                "Evaluation input contains an invalid "
                f"experiment_mode: {mode!r}."
            )

        responder_agent_id = evaluation_input[
            "responder_agent_id"
        ]

        if responder_agent_id not in AGENT_PERSONA_MAP:
            raise NeedlePersonaEvaluatorError(
                "Evaluation input contains an unknown "
                f"responder_agent_id: "
                f"{responder_agent_id!r}."
            )

        if not evaluation_input["accepted_answers"]:
            raise NeedlePersonaEvaluatorError(
                "accepted_answers cannot be empty."
            )

        expected_source_ids = cls._clean_unique(
            evaluation_input[
                "expected_source_ids"
            ]
        )
        expected_local_source_ids = cls._clean_unique(
            evaluation_input[
                "expected_local_source_ids"
            ]
        )
        expected_cross_agent_source_ids = (
            cls._clean_unique(
                evaluation_input[
                    "expected_cross_agent_source_ids"
                ]
            )
        )

        if not expected_source_ids:
            raise NeedlePersonaEvaluatorError(
                "expected_source_ids cannot be empty."
            )

        expected_set = set(expected_source_ids)
        local_set = set(
            expected_local_source_ids
        )
        cross_set = set(
            expected_cross_agent_source_ids
        )

        if local_set & cross_set:
            raise NeedlePersonaEvaluatorError(
                "Expected local and cross-agent source "
                "sets must be disjoint."
            )

        if local_set | cross_set != expected_set:
            raise NeedlePersonaEvaluatorError(
                "Expected local and cross-agent source "
                "sets must partition expected_source_ids."
            )

        memory_id_to_source_ids = evaluation_input[
            "memory_id_to_source_ids"
        ]
        memory_id_to_owner_agent_id = evaluation_input[
            "memory_id_to_owner_agent_id"
        ]

        if not memory_id_to_source_ids:
            raise NeedlePersonaEvaluatorError(
                "memory_id_to_source_ids cannot be empty."
            )

        source_mapping_memory_ids = set(
            memory_id_to_source_ids
        )
        owner_mapping_memory_ids = set(
            memory_id_to_owner_agent_id
        )

        if (
            source_mapping_memory_ids
            - owner_mapping_memory_ids
        ):
            raise NeedlePersonaEvaluatorError(
                "Some source-mapped memories have no "
                "owner mapping: "
                f"{sorted(
                    source_mapping_memory_ids
                    - owner_mapping_memory_ids
                )}."
            )

        runtime_memory_ids = set(
            cls._clean_unique(
                [
                    *evaluation_input[
                        "used_memory_ids"
                    ],
                    *evaluation_input[
                        "responder_retrieved_memory_ids"
                    ],
                    *evaluation_input[
                        "candidate_memory_ids"
                    ],
                    *evaluation_input[
                        "shared_memory_ids"
                    ],
                    *evaluation_input[
                        "rejected_memory_ids"
                    ],
                ]
            )
        )

        missing_source_mappings = (
            runtime_memory_ids
            - source_mapping_memory_ids
        )

        if missing_source_mappings:
            raise NeedlePersonaEvaluatorError(
                "Runtime memories are missing source "
                "mappings: "
                f"{sorted(missing_source_mappings)}."
            )

        missing_owner_mappings = (
            runtime_memory_ids
            - owner_mapping_memory_ids
        )

        if missing_owner_mappings:
            raise NeedlePersonaEvaluatorError(
                "Runtime memories are missing owner "
                "mappings: "
                f"{sorted(missing_owner_mappings)}."
            )

        used_memory_ids = set(
            evaluation_input["used_memory_ids"]
        )
        retrieved_memory_ids = set(
            evaluation_input[
                "responder_retrieved_memory_ids"
            ]
        )

        unavailable_used_memories = (
            used_memory_ids - retrieved_memory_ids
        )

        if unavailable_used_memories:
            raise NeedlePersonaEvaluatorError(
                "used_memory_ids contains memories that "
                "were not retrieved for the responder: "
                f"{sorted(unavailable_used_memories)}."
            )

        candidate_memory_ids = set(
            evaluation_input[
                "candidate_memory_ids"
            ]
        )
        shared_memory_ids = set(
            evaluation_input["shared_memory_ids"]
        )
        rejected_memory_ids = set(
            evaluation_input[
                "rejected_memory_ids"
            ]
        )

        if mode == "private_only":
            if shared_memory_ids:
                raise NeedlePersonaEvaluatorError(
                    "private_only input must not contain "
                    "shared_memory_ids."
                )

            if candidate_memory_ids:
                raise NeedlePersonaEvaluatorError(
                    "private_only input must not contain "
                    "candidate_memory_ids."
                )

        if not shared_memory_ids <= candidate_memory_ids:
            raise NeedlePersonaEvaluatorError(
                "shared_memory_ids must be a subset of "
                "candidate_memory_ids."
            )

        if not rejected_memory_ids <= candidate_memory_ids:
            raise NeedlePersonaEvaluatorError(
                "rejected_memory_ids must be a subset of "
                "candidate_memory_ids."
            )

        overlapping_outcomes = (
            shared_memory_ids
            & rejected_memory_ids
        )

        if overlapping_outcomes:
            raise NeedlePersonaEvaluatorError(
                "A memory cannot be both shared and "
                "rejected: "
                f"{sorted(overlapping_outcomes)}."
            )

        critic_review_ids = set(
            evaluation_input["critic_reviews"]
        )
        promotion_decision_ids = set(
            evaluation_input[
                "promotion_decisions"
            ]
        )

        if not critic_review_ids <= candidate_memory_ids:
            raise NeedlePersonaEvaluatorError(
                "critic_reviews contains non-candidate "
                "memory IDs."
            )

        if not promotion_decision_ids <= candidate_memory_ids:
            raise NeedlePersonaEvaluatorError(
                "promotion_decisions contains "
                "non-candidate memory IDs."
            )

        if mode == "governed_shared":
            promotion_request_memory_ids = set(
                evaluation_input[
                    "promotion_request_ids"
                ]
            )

            if (
                promotion_request_memory_ids
                != candidate_memory_ids
            ):
                raise NeedlePersonaEvaluatorError(
                    "governed_shared requires one "
                    "promotion request per candidate "
                    "memory."
                )

            if critic_review_ids != candidate_memory_ids:
                raise NeedlePersonaEvaluatorError(
                    "governed_shared requires one Critic "
                    "review per candidate memory."
                )

            if (
                promotion_decision_ids
                != candidate_memory_ids
            ):
                raise NeedlePersonaEvaluatorError(
                    "governed_shared requires one "
                    "Coordinator decision per candidate "
                    "memory."
                )

        known_source_ids = {
            source_id
            for source_ids in (
                memory_id_to_source_ids.values()
            )
            for source_id in source_ids
        }

        unknown_supporting_sources = (
            set(
                evaluation_input[
                    "supporting_source_ids"
                ]
            )
            - known_source_ids
        )

        if unknown_supporting_sources:
            raise NeedlePersonaEvaluatorError(
                "supporting_source_ids contains unknown "
                "source IDs: "
                f"{sorted(unknown_supporting_sources)}."
            )

    @staticmethod
    def _calculate_governance_metrics(
        *,
        mode: ExperimentMode,
        shared_source_ids: Sequence[str],
        correctly_shared_source_ids: Sequence[str],
        over_shared_source_ids: Sequence[str],
        expected_cross_agent_source_ids: Sequence[str],
        missing_shared_cross_agent_source_ids: Sequence[str],
    ) -> tuple[
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
    ]:
        if mode == "private_only":
            return (
                None,
                None,
                None,
                None,
                None,
            )

        actual_count = len(
            set(shared_source_ids)
        )
        correct_count = len(
            set(correctly_shared_source_ids)
        )
        expected_count = len(
            set(expected_cross_agent_source_ids)
        )
        over_shared_count = len(
            set(over_shared_source_ids)
        )
        missing_count = len(
            set(
                missing_shared_cross_agent_source_ids
            )
        )

        if actual_count == 0:
            governance_precision = (
                1.0
                if expected_count == 0
                else 0.0
            )
            over_sharing_rate = 0.0
        else:
            governance_precision = (
                correct_count / actual_count
            )
            over_sharing_rate = (
                over_shared_count / actual_count
            )

        if expected_count == 0:
            governance_recall = 1.0
            under_sharing_rate = 0.0
        else:
            governance_recall = (
                correct_count / expected_count
            )
            under_sharing_rate = (
                missing_count / expected_count
            )

        return (
            governance_recall,
            governance_precision,
            governance_recall,
            over_sharing_rate,
            under_sharing_rate,
        )

    @staticmethod
    def _calculate_transfer_success(
        *,
        mode: ExperimentMode,
        has_cross_agent_target: bool,
        answer_metrics: AnswerMetricResult,
        shared_cross_agent_source_recall: float | None,
        cross_agent_source_retrieval_recall: float,
        cross_agent_source_utilisation_recall: float,
    ) -> bool | None:
        if mode == "private_only":
            return None

        return bool(
            has_cross_agent_target
            and answer_metrics.answer_accuracy == 1.0
            and shared_cross_agent_source_recall == 1.0
            and cross_agent_source_retrieval_recall == 1.0
            and cross_agent_source_utilisation_recall == 1.0
        )

    @staticmethod
    def _recall(
        observed: set[str],
        expected: set[str],
    ) -> float:
        if not expected:
            return 1.0

        return len(observed & expected) / len(expected)

    @staticmethod
    def _ordered_intersection(
        ordered_values: Sequence[str],
        included_values: set[str],
    ) -> list[str]:
        return [
            value
            for value in ordered_values
            if value in included_values
        ]

    @staticmethod
    def _ordered_difference(
        ordered_values: Sequence[str],
        excluded_values: set[str],
    ) -> list[str]:
        return [
            value
            for value in ordered_values
            if value not in excluded_values
        ]

    @staticmethod
    def _clean_unique(
        values: Sequence[str],
    ) -> list[str]:
        cleaned: list[str] = []

        for value in values:
            item = str(value).strip()

            if item and item not in cleaned:
                cleaned.append(item)

        return cleaned

    @staticmethod
    def _field(
        value: Any,
        field_name: str,
        default: Any = None,
    ) -> Any:
        if isinstance(value, Mapping):
            return value.get(
                field_name,
                default,
            )

        return getattr(
            value,
            field_name,
            default,
        )


def evaluate_needle_persona(
    evaluation_input: NeedlePersonaEvaluationInput,
    *,
    config: NeedlePersonaEvaluatorConfig | None = None,
) -> NeedlePersonaEvaluationResult:
    """
    Convenience function for evaluating one completed run.
    """
    return NeedlePersonaEvaluator(
        config=config
    ).evaluate(evaluation_input)


__all__ = [
    "NeedlePersonaEvaluatorError",
    "NeedlePersonaEvaluatorConfig",
    "NeedlePersonaEvaluator",
    "evaluate_needle_persona",
]