from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import (
    Mapping,
    Sequence,
)
from dataclasses import dataclass, field
from typing import (
    Any,
    NotRequired,
    TypedDict,
)

from experiments.evaluator.informativebench.config.needle_persona_config import (
    ExperimentMode,
    validate_experiment_mode,
)
from ..data_preparing.needle_persona_models import (
    AGENT_PERSONA_MAP,
    PERSONA_AGENT_MAP,
    PERSONA_NAMES,
    NeedlePersonaSample,
    PersonaName,
    PrivateMemoryIndex,
)
from ..workflow.state.needle_persona_state import (
    NeedlePersonaWorkflowState,
)


class NeedlePersonaEvaluatorError(ValueError):
    """
    Raised when evaluator input is incomplete or internally inconsistent.
    """


class NeedlePersonaEvaluationInput(TypedDict):
    """
    Held-out evaluation data combined with a completed workflow state.

    This object is built only after workflow execution. Gold answers and raw
    persona sources therefore never enter LangGraph state.
    """

    run_id: str
    sample_id: str
    experiment_mode: ExperimentMode
    question: str

    answer_status: str
    predicted_answer: str
    accepted_answers: list[str]

    responder_agent_id: str
    selected_agent_ids: list[str]
    contributing_agent_ids: list[str]

    initial_retrieved_memory_ids: list[str]
    final_retrieved_memory_ids: list[str]
    responder_retrieved_memory_ids: list[str]
    used_memory_ids: list[str]
    supporting_source_ids: list[str]

    candidate_memory_ids: list[str]
    access_request_ids: dict[str, str]
    critic_reviews: dict[str, Any]
    coordinator_outputs: dict[str, Any]
    access_decisions: dict[str, Any]
    approved_memory_ids: list[str]
    rejected_memory_ids: list[str]

    target_personas: list[str]
    expected_source_ids: list[str]
    expected_local_source_ids: list[str]
    expected_cross_agent_source_ids: list[str]

    memory_id_to_source_ids: dict[
        str,
        list[str],
    ]
    memory_id_to_owner_agent_id: dict[
        str,
        str,
    ]

    node_trace: list[str]
    errors: list[str]
    warnings: list[str]
    node_metrics: dict[str, float]


class NeedlePersonaEvaluationResult(TypedDict):
    """
    Deterministic metrics and diagnostics for one completed run.
    """

    schema_version: str

    run_id: str
    sample_id: str
    experiment_mode: ExperimentMode
    question: str

    answer_status: str
    predicted_answer: str
    matched_reference_answer: str | None

    exact_match: float
    token_precision: float
    token_recall: float
    token_f1: float
    entity_precision: float
    entity_recall: float
    entity_f1: float

    target_personas: list[str]
    expected_source_ids: list[str]
    expected_local_source_ids: list[str]
    expected_cross_agent_source_ids: list[str]

    retrieved_source_ids: list[str]
    retrieved_target_source_ids: list[str]
    missing_retrieved_target_source_ids: list[str]
    target_source_retrieval_recall: float
    retrieval_hit: bool

    cross_agent_retrieved_source_ids: list[str]
    cross_agent_source_retrieval_recall: float
    cross_agent_memory_hit: bool

    used_source_ids: list[str]
    used_target_source_ids: list[str]
    target_source_utilisation_recall: float
    cross_agent_used_source_ids: list[str]
    cross_agent_source_utilisation_recall: float
    cross_agent_source_utilised: bool

    shared_source_ids: list[str]
    correctly_shared_source_ids: list[str]
    over_shared_source_ids: list[str]
    missing_shared_cross_agent_source_ids: list[str]

    shared_cross_agent_source_recall: (
        float | None
    )
    governance_precision: float | None
    governance_recall: float | None
    over_sharing_rate: float | None
    under_sharing_rate: float | None

    candidate_memory_count: int
    access_request_count: int
    critic_review_count: int
    critic_approved_memory_count: int
    coordinator_decision_count: int
    coordinator_approved_memory_count: int
    access_granted_count: int
    approved_memory_count: int
    rejected_memory_count: int

    transfer_success: bool | None

    responder_agent_id: str
    selected_agent_ids: list[str]
    contributing_agent_ids: list[str]

    initial_retrieved_memory_ids: list[str]
    final_retrieved_memory_ids: list[str]
    responder_retrieved_memory_ids: list[str]
    used_memory_ids: list[str]
    supporting_source_ids: list[str]
    candidate_memory_ids: list[str]
    approved_memory_ids: list[str]
    rejected_memory_ids: list[str]

    node_trace: list[str]
    errors: list[str]
    warnings: list[str]
    node_metrics: dict[str, float]

    run_duration_ms: NotRequired[float]
    stage_durations_ms: NotRequired[
        dict[str, float]
    ]
    runtime_metadata: NotRequired[
        dict[str, Any]
    ]


@dataclass(
    frozen=True,
    slots=True,
)
class AnswerMetricResult:
    """
    Best answer match against all accepted references.
    """

    predicted_answer: str
    matched_reference_answer: str | None

    exact_match: float
    token_precision: float
    token_recall: float
    token_f1: float

    entity_precision: float
    entity_recall: float
    entity_f1: float


@dataclass(
    frozen=True,
    slots=True,
)
class NeedlePersonaEvaluatorConfig:
    """
    Deterministic evaluation configuration.
    """

    strict_memory_mapping: bool = True
    validate_input: bool = True

    # InformativeBench uses Nancy as an alias for Dave in some generated
    # questions. Additional aliases may be supplied by the caller.
    persona_aliases: Mapping[
        str,
        PersonaName,
    ] = field(
        default_factory=lambda: {
            "nancy": "dave",
        }
    )

    def __post_init__(self) -> None:
        normalised: dict[
            str,
            PersonaName,
        ] = {}

        for alias, persona in (
            self.persona_aliases.items()
        ):
            clean_alias = str(
                alias
            ).strip().casefold()
            clean_persona = str(
                persona
            ).strip().casefold()

            if not clean_alias:
                raise ValueError(
                    "Persona alias cannot be empty."
                )

            if (
                clean_persona
                not in PERSONA_NAMES
            ):
                raise ValueError(
                    "Persona alias target must be "
                    "one of PERSONA_NAMES; received "
                    f"{clean_persona!r}."
                )

            normalised[
                clean_alias
            ] = clean_persona  # type: ignore[assignment]

        object.__setattr__(
            self,
            "persona_aliases",
            normalised,
        )


class NeedlePersonaEvaluator:
    """
    Deterministic evaluator for Needle in the Persona.

    It performs no LLM calls, retrieval, sharing, or memory mutation.
    """

    RESULT_SCHEMA_VERSION = "2.0"

    def __init__(
        self,
        config: (
            NeedlePersonaEvaluatorConfig | None
        ) = None,
    ) -> None:
        self.config = (
            config
            or NeedlePersonaEvaluatorConfig()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        evaluation_input: (
            NeedlePersonaEvaluationInput
        ),
    ) -> NeedlePersonaEvaluationResult:
        """
        Evaluate one completed run.
        """
        if self.config.validate_input:
            self._validate_input(
                evaluation_input
            )

        answer_metrics = (
            self.evaluate_answer(
                predicted_answer=(
                    evaluation_input[
                        "predicted_answer"
                    ]
                ),
                accepted_answers=(
                    evaluation_input[
                        "accepted_answers"
                    ]
                ),
            )
        )

        memory_to_sources = (
            evaluation_input[
                "memory_id_to_source_ids"
            ]
        )

        retrieved_source_ids = (
            self._map_memory_ids_to_sources(
                evaluation_input[
                    "responder_retrieved_memory_ids"
                ],
                memory_to_sources,
            )
        )
        used_source_ids = (
            self._map_memory_ids_to_sources(
                evaluation_input[
                    "used_memory_ids"
                ],
                memory_to_sources,
            )
        )
        shared_source_ids = (
            self._map_memory_ids_to_sources(
                evaluation_input[
                    "approved_memory_ids"
                ],
                memory_to_sources,
            )
        )

        expected_source_ids = (
            self._clean_unique(
                evaluation_input[
                    "expected_source_ids"
                ]
            )
        )
        expected_local_source_ids = (
            self._clean_unique(
                evaluation_input[
                    "expected_local_source_ids"
                ]
            )
        )
        expected_cross_source_ids = (
            self._clean_unique(
                evaluation_input[
                    "expected_cross_agent_source_ids"
                ]
            )
        )

        expected_set = set(
            expected_source_ids
        )
        cross_expected_set = set(
            expected_cross_source_ids
        )
        retrieved_set = set(
            retrieved_source_ids
        )
        used_set = set(
            used_source_ids
        )
        shared_set = set(
            shared_source_ids
        )

        retrieved_target_source_ids = (
            self._ordered_intersection(
                expected_source_ids,
                retrieved_set,
            )
        )
        missing_retrieved_target_source_ids = (
            self._ordered_difference(
                expected_source_ids,
                retrieved_set,
            )
        )
        target_source_retrieval_recall = (
            self._recall(
                observed=retrieved_set,
                expected=expected_set,
            )
        )
        retrieval_hit = bool(
            expected_set
            and expected_set <= retrieved_set
        )

        cross_agent_retrieved_source_ids = (
            self._ordered_intersection(
                expected_cross_source_ids,
                retrieved_set,
            )
        )
        cross_agent_source_retrieval_recall = (
            self._recall(
                observed=retrieved_set,
                expected=cross_expected_set,
            )
        )
        cross_agent_memory_hit = bool(
            cross_agent_retrieved_source_ids
        )

        used_target_source_ids = (
            self._ordered_intersection(
                expected_source_ids,
                used_set,
            )
        )
        target_source_utilisation_recall = (
            self._recall(
                observed=used_set,
                expected=expected_set,
            )
        )
        cross_agent_used_source_ids = (
            self._ordered_intersection(
                expected_cross_source_ids,
                used_set,
            )
        )
        cross_agent_source_utilisation_recall = (
            self._recall(
                observed=used_set,
                expected=cross_expected_set,
            )
        )
        cross_agent_source_utilised = bool(
            cross_agent_used_source_ids
        )

        correctly_shared_source_ids = (
            self._ordered_intersection(
                expected_cross_source_ids,
                shared_set,
            )
        )
        over_shared_source_ids = (
            self._ordered_difference(
                shared_source_ids,
                cross_expected_set,
            )
        )
        missing_shared_cross_source_ids = (
            self._ordered_difference(
                expected_cross_source_ids,
                shared_set,
            )
        )

        (
            shared_cross_source_recall,
            governance_precision,
            governance_recall,
            over_sharing_rate,
            under_sharing_rate,
        ) = self._governance_metrics(
            mode=evaluation_input[
                "experiment_mode"
            ],
            shared_source_ids=(
                shared_source_ids
            ),
            correctly_shared_source_ids=(
                correctly_shared_source_ids
            ),
            over_shared_source_ids=(
                over_shared_source_ids
            ),
            expected_cross_source_ids=(
                expected_cross_source_ids
            ),
            missing_shared_cross_source_ids=(
                missing_shared_cross_source_ids
            ),
        )

        transfer_success = (
            self._transfer_success(
                mode=evaluation_input[
                    "experiment_mode"
                ],
                has_cross_agent_target=bool(
                    expected_cross_source_ids
                ),
                exact_match=(
                    answer_metrics.exact_match
                ),
                shared_cross_source_recall=(
                    shared_cross_source_recall
                ),
                cross_agent_retrieval_recall=(
                    cross_agent_source_retrieval_recall
                ),
                cross_agent_utilisation_recall=(
                    cross_agent_source_utilisation_recall
                ),
            )
        )

        critic_approved_count = sum(
            1
            for review in evaluation_input[
                "critic_reviews"
            ].values()
            if self._field(
                review,
                "recommendation",
            )
            == "approve"
        )

        coordinator_approved_count = sum(
            1
            for decision in evaluation_input[
                "coordinator_outputs"
            ].values()
            if (
                self._field(
                    decision,
                    "decision",
                )
                == "approve"
                and self._field(
                    decision,
                    "target_scope",
                )
                == "shared"
            )
        )

        access_granted_count = sum(
            1
            for decision in evaluation_input[
                "access_decisions"
            ].values()
            if bool(
                self._field(
                    decision,
                    "access_granted",
                    False,
                )
            )
        )

        return NeedlePersonaEvaluationResult(
            schema_version=(
                self.RESULT_SCHEMA_VERSION
            ),
            run_id=evaluation_input[
                "run_id"
            ],
            sample_id=evaluation_input[
                "sample_id"
            ],
            experiment_mode=(
                evaluation_input[
                    "experiment_mode"
                ]
            ),
            question=evaluation_input[
                "question"
            ],
            answer_status=(
                evaluation_input[
                    "answer_status"
                ]
            ),
            predicted_answer=(
                answer_metrics
                .predicted_answer
            ),
            matched_reference_answer=(
                answer_metrics
                .matched_reference_answer
            ),
            exact_match=(
                answer_metrics.exact_match
            ),
            token_precision=(
                answer_metrics.token_precision
            ),
            token_recall=(
                answer_metrics.token_recall
            ),
            token_f1=(
                answer_metrics.token_f1
            ),
            entity_precision=(
                answer_metrics.entity_precision
            ),
            entity_recall=(
                answer_metrics.entity_recall
            ),
            entity_f1=(
                answer_metrics.entity_f1
            ),
            target_personas=list(
                evaluation_input[
                    "target_personas"
                ]
            ),
            expected_source_ids=(
                expected_source_ids
            ),
            expected_local_source_ids=(
                expected_local_source_ids
            ),
            expected_cross_agent_source_ids=(
                expected_cross_source_ids
            ),
            retrieved_source_ids=(
                retrieved_source_ids
            ),
            retrieved_target_source_ids=(
                retrieved_target_source_ids
            ),
            missing_retrieved_target_source_ids=(
                missing_retrieved_target_source_ids
            ),
            target_source_retrieval_recall=(
                target_source_retrieval_recall
            ),
            retrieval_hit=retrieval_hit,
            cross_agent_retrieved_source_ids=(
                cross_agent_retrieved_source_ids
            ),
            cross_agent_source_retrieval_recall=(
                cross_agent_source_retrieval_recall
            ),
            cross_agent_memory_hit=(
                cross_agent_memory_hit
            ),
            used_source_ids=used_source_ids,
            used_target_source_ids=(
                used_target_source_ids
            ),
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
            shared_source_ids=(
                shared_source_ids
            ),
            correctly_shared_source_ids=(
                correctly_shared_source_ids
            ),
            over_shared_source_ids=(
                over_shared_source_ids
            ),
            missing_shared_cross_agent_source_ids=(
                missing_shared_cross_source_ids
            ),
            shared_cross_agent_source_recall=(
                shared_cross_source_recall
            ),
            governance_precision=(
                governance_precision
            ),
            governance_recall=(
                governance_recall
            ),
            over_sharing_rate=(
                over_sharing_rate
            ),
            under_sharing_rate=(
                under_sharing_rate
            ),
            candidate_memory_count=len(
                evaluation_input[
                    "candidate_memory_ids"
                ]
            ),
            access_request_count=len(
                evaluation_input[
                    "access_request_ids"
                ]
            ),
            critic_review_count=len(
                evaluation_input[
                    "critic_reviews"
                ]
            ),
            critic_approved_memory_count=(
                critic_approved_count
            ),
            coordinator_decision_count=len(
                evaluation_input[
                    "coordinator_outputs"
                ]
            ),
            coordinator_approved_memory_count=(
                coordinator_approved_count
            ),
            access_granted_count=(
                access_granted_count
            ),
            approved_memory_count=len(
                evaluation_input[
                    "approved_memory_ids"
                ]
            ),
            rejected_memory_count=len(
                evaluation_input[
                    "rejected_memory_ids"
                ]
            ),
            transfer_success=transfer_success,
            responder_agent_id=(
                evaluation_input[
                    "responder_agent_id"
                ]
            ),
            selected_agent_ids=list(
                evaluation_input[
                    "selected_agent_ids"
                ]
            ),
            contributing_agent_ids=list(
                evaluation_input[
                    "contributing_agent_ids"
                ]
            ),
            initial_retrieved_memory_ids=list(
                evaluation_input[
                    "initial_retrieved_memory_ids"
                ]
            ),
            final_retrieved_memory_ids=list(
                evaluation_input[
                    "final_retrieved_memory_ids"
                ]
            ),
            responder_retrieved_memory_ids=list(
                evaluation_input[
                    "responder_retrieved_memory_ids"
                ]
            ),
            used_memory_ids=list(
                evaluation_input[
                    "used_memory_ids"
                ]
            ),
            supporting_source_ids=list(
                evaluation_input[
                    "supporting_source_ids"
                ]
            ),
            candidate_memory_ids=list(
                evaluation_input[
                    "candidate_memory_ids"
                ]
            ),
            approved_memory_ids=list(
                evaluation_input[
                    "approved_memory_ids"
                ]
            ),
            rejected_memory_ids=list(
                evaluation_input[
                    "rejected_memory_ids"
                ]
            ),
            node_trace=list(
                evaluation_input[
                    "node_trace"
                ]
            ),
            errors=list(
                evaluation_input[
                    "errors"
                ]
            ),
            warnings=list(
                evaluation_input[
                    "warnings"
                ]
            ),
            node_metrics=dict(
                evaluation_input[
                    "node_metrics"
                ]
            ),
        )

    def evaluate_run(
        self,
        *,
        sample: NeedlePersonaSample,
        memory_index: PrivateMemoryIndex,
        final_state: (
            NeedlePersonaWorkflowState
        ),
    ) -> NeedlePersonaEvaluationResult:
        """
        Build held-out evidence and evaluate one completed workflow.
        """
        return self.evaluate(
            build_needle_persona_evaluation_input(
                sample=sample,
                memory_index=memory_index,
                final_state=final_state,
                config=self.config,
            )
        )

    def evaluate_many(
        self,
        evaluation_inputs: Sequence[
            NeedlePersonaEvaluationInput
        ],
    ) -> list[
        NeedlePersonaEvaluationResult
    ]:
        return [
            self.evaluate(item)
            for item in evaluation_inputs
        ]

    # ------------------------------------------------------------------
    # Answer metrics
    # ------------------------------------------------------------------

    @classmethod
    def evaluate_answer(
        cls,
        *,
        predicted_answer: str,
        accepted_answers: Sequence[str],
    ) -> AnswerMetricResult:
        """
        Select the accepted reference with the highest token F1.

        Exact match is used as the first tie-breaker, followed by entity F1 and
        original reference order.
        """
        prediction = str(
            predicted_answer or ""
        ).strip()
        references = cls._clean_unique(
            accepted_answers
        )

        if not references:
            raise NeedlePersonaEvaluatorError(
                "accepted_answers cannot be empty."
            )

        best: tuple[
            float,
            float,
            float,
            int,
            str,
            float,
            float,
            float,
            float,
            float,
            float,
        ] | None = None

        for index, reference in enumerate(
            references
        ):
            exact_match = float(
                cls.normalise_answer(
                    prediction
                )
                == cls.normalise_answer(
                    reference
                )
            )

            (
                token_precision,
                token_recall,
                token_f1,
            ) = cls._token_scores(
                prediction,
                reference,
            )

            (
                entity_precision,
                entity_recall,
                entity_f1,
            ) = cls._entity_scores(
                prediction,
                reference,
            )

            candidate = (
                token_f1,
                exact_match,
                entity_f1,
                -index,
                reference,
                token_precision,
                token_recall,
                entity_precision,
                entity_recall,
                token_f1,
                entity_f1,
            )

            if (
                best is None
                or candidate[:4]
                > best[:4]
            ):
                best = candidate

        assert best is not None

        return AnswerMetricResult(
            predicted_answer=prediction,
            matched_reference_answer=(
                best[4]
            ),
            exact_match=best[1],
            token_precision=best[5],
            token_recall=best[6],
            token_f1=best[9],
            entity_precision=best[7],
            entity_recall=best[8],
            entity_f1=best[10],
        )

    @staticmethod
    def normalise_answer(
        text: str,
    ) -> str:
        """
        SQuAD-style normalisation with Unicode-safe punctuation removal.
        """
        value = str(
            text or ""
        ).casefold()

        value = "".join(
            " "
            if character in string.punctuation
            else character
            for character in value
        )

        value = re.sub(
            r"\b(a|an|the|and)\b",
            " ",
            value,
        )

        return " ".join(
            value.split()
        )

    @classmethod
    def split_answer_entities(
        cls,
        text: str,
    ) -> list[str]:
        """
        Split the benchmark's short multi-item answer into normalised entities.
        """
        raw_parts = re.split(
            r"\s*(?:,|;|\n|\band\b|&)\s*",
            str(text or ""),
            flags=re.IGNORECASE,
        )

        result: list[str] = []

        for part in raw_parts:
            normalised = cls.normalise_answer(
                part
            )

            if (
                normalised
                and normalised not in result
            ):
                result.append(
                    normalised
                )

        return result

    @classmethod
    def _token_scores(
        cls,
        prediction: str,
        reference: str,
    ) -> tuple[
        float,
        float,
        float,
    ]:
        predicted_tokens = (
            cls.normalise_answer(
                prediction
            ).split()
        )
        reference_tokens = (
            cls.normalise_answer(
                reference
            ).split()
        )

        if (
            not predicted_tokens
            and not reference_tokens
        ):
            return 1.0, 1.0, 1.0

        if (
            not predicted_tokens
            or not reference_tokens
        ):
            return 0.0, 0.0, 0.0

        overlap = (
            Counter(predicted_tokens)
            & Counter(reference_tokens)
        )
        common_count = sum(
            overlap.values()
        )

        if common_count == 0:
            return 0.0, 0.0, 0.0

        precision = (
            common_count
            / len(predicted_tokens)
        )
        recall = (
            common_count
            / len(reference_tokens)
        )
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

        return precision, recall, f1

    @classmethod
    def _entity_scores(
        cls,
        prediction: str,
        reference: str,
    ) -> tuple[
        float,
        float,
        float,
    ]:
        predicted = set(
            cls.split_answer_entities(
                prediction
            )
        )
        expected = set(
            cls.split_answer_entities(
                reference
            )
        )

        if not predicted and not expected:
            return 1.0, 1.0, 1.0

        if not predicted or not expected:
            return 0.0, 0.0, 0.0

        common = len(
            predicted & expected
        )
        precision = common / len(
            predicted
        )
        recall = common / len(
            expected
        )
        f1 = (
            0.0
            if precision + recall == 0
            else (
                2 * precision * recall
                / (precision + recall)
            )
        )

        return precision, recall, f1

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    @classmethod
    def _validate_input(
        cls,
        evaluation_input: (
            NeedlePersonaEvaluationInput
        ),
    ) -> None:
        for field_name in (
            "run_id",
            "sample_id",
            "question",
            "answer_status",
            "responder_agent_id",
        ):
            if not str(
                evaluation_input.get(
                    field_name,
                    "",
                )
            ).strip():
                raise NeedlePersonaEvaluatorError(
                    "Evaluation input field "
                    f"{field_name!r} cannot be empty."
                )

        validate_experiment_mode(
            str(
                evaluation_input[
                    "experiment_mode"
                ]
            )
        )

        if (
            evaluation_input[
                "responder_agent_id"
            ]
            not in AGENT_PERSONA_MAP
        ):
            raise NeedlePersonaEvaluatorError(
                "Unknown responder_agent_id: "
                f"{evaluation_input[
                    'responder_agent_id'
                ]!r}."
            )

        if not evaluation_input[
            "accepted_answers"
        ]:
            raise NeedlePersonaEvaluatorError(
                "accepted_answers cannot be empty."
            )

        expected = set(
            evaluation_input[
                "expected_source_ids"
            ]
        )
        local = set(
            evaluation_input[
                "expected_local_source_ids"
            ]
        )
        cross = set(
            evaluation_input[
                "expected_cross_agent_source_ids"
            ]
        )

        if not expected:
            raise NeedlePersonaEvaluatorError(
                "expected_source_ids cannot be empty."
            )

        if local & cross:
            raise NeedlePersonaEvaluatorError(
                "Local and cross-Agent expected "
                "source sets must be disjoint."
            )

        if local | cross != expected:
            raise NeedlePersonaEvaluatorError(
                "Local and cross-Agent expected "
                "sources must partition "
                "expected_source_ids."
            )

        memory_to_sources = (
            evaluation_input[
                "memory_id_to_source_ids"
            ]
        )
        memory_to_owner = (
            evaluation_input[
                "memory_id_to_owner_agent_id"
            ]
        )

        runtime_memory_ids = set(
            cls._clean_unique(
                [
                    *evaluation_input[
                        "initial_retrieved_memory_ids"
                    ],
                    *evaluation_input[
                        "final_retrieved_memory_ids"
                    ],
                    *evaluation_input[
                        "used_memory_ids"
                    ],
                    *evaluation_input[
                        "candidate_memory_ids"
                    ],
                    *evaluation_input[
                        "approved_memory_ids"
                    ],
                    *evaluation_input[
                        "rejected_memory_ids"
                    ],
                ]
            )
        )

        missing_source_mappings = (
            runtime_memory_ids
            - set(memory_to_sources)
        )
        missing_owner_mappings = (
            runtime_memory_ids
            - set(memory_to_owner)
        )

        if missing_source_mappings:
            raise NeedlePersonaEvaluatorError(
                "Runtime memories are missing "
                "source mappings: "
                f"{sorted(
                    missing_source_mappings
                )}."
            )

        if missing_owner_mappings:
            raise NeedlePersonaEvaluatorError(
                "Runtime memories are missing "
                "owner mappings: "
                f"{sorted(
                    missing_owner_mappings
                )}."
            )

        used = set(
            evaluation_input[
                "used_memory_ids"
            ]
        )
        retrieved = set(
            evaluation_input[
                "responder_retrieved_memory_ids"
            ]
        )

        if not used <= retrieved:
            raise NeedlePersonaEvaluatorError(
                "used_memory_ids must be a subset "
                "of responder_retrieved_memory_ids."
            )

        candidates = set(
            evaluation_input[
                "candidate_memory_ids"
            ]
        )
        approved = set(
            evaluation_input[
                "approved_memory_ids"
            ]
        )
        rejected = set(
            evaluation_input[
                "rejected_memory_ids"
            ]
        )

        if approved & rejected:
            raise NeedlePersonaEvaluatorError(
                "A memory cannot be both approved "
                "and rejected."
            )

        mode = evaluation_input[
            "experiment_mode"
        ]

        if mode == "private_only":
            if candidates:
                raise NeedlePersonaEvaluatorError(
                    "private_only must not contain "
                    "candidate memories."
                )

            if evaluation_input[
                "access_request_ids"
            ]:
                raise NeedlePersonaEvaluatorError(
                    "private_only must not contain "
                    "access requests."
                )

            if approved:
                raise NeedlePersonaEvaluatorError(
                    "private_only must not contain "
                    "approved shared memories."
                )

        request_memory_ids = set(
            evaluation_input[
                "access_request_ids"
            ]
        )

        if not request_memory_ids <= candidates:
            raise NeedlePersonaEvaluatorError(
                "Access requests contain "
                "non-candidate memory IDs."
            )

        if not set(
            evaluation_input[
                "critic_reviews"
            ]
        ) <= candidates:
            raise NeedlePersonaEvaluatorError(
                "critic_reviews contains "
                "non-candidate memory IDs."
            )

        if not set(
            evaluation_input[
                "coordinator_outputs"
            ]
        ) <= candidates:
            raise NeedlePersonaEvaluatorError(
                "coordinator_outputs contains "
                "non-candidate memory IDs."
            )

    # ------------------------------------------------------------------
    # Evidence and governance metrics
    # ------------------------------------------------------------------

    def _map_memory_ids_to_sources(
        self,
        memory_ids: Sequence[str],
        memory_to_sources: Mapping[
            str,
            Sequence[str],
        ],
    ) -> list[str]:
        result: list[str] = []
        unknown: list[str] = []

        for memory_id in self._clean_unique(
            memory_ids
        ):
            if memory_id not in memory_to_sources:
                unknown.append(
                    memory_id
                )
                continue

            for source_id in (
                memory_to_sources[
                    memory_id
                ]
            ):
                clean_source_id = str(
                    source_id
                ).strip()

                if (
                    clean_source_id
                    and clean_source_id
                    not in result
                ):
                    result.append(
                        clean_source_id
                    )

        if (
            self.config.strict_memory_mapping
            and unknown
        ):
            raise NeedlePersonaEvaluatorError(
                "No source mapping exists for "
                f"memory IDs: {unknown}."
            )

        return result

    @staticmethod
    def _governance_metrics(
        *,
        mode: ExperimentMode,
        shared_source_ids: Sequence[str],
        correctly_shared_source_ids: (
            Sequence[str]
        ),
        over_shared_source_ids: (
            Sequence[str]
        ),
        expected_cross_source_ids: (
            Sequence[str]
        ),
        missing_shared_cross_source_ids: (
            Sequence[str]
        ),
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
            set(
                correctly_shared_source_ids
            )
        )
        over_count = len(
            set(over_shared_source_ids)
        )
        expected_count = len(
            set(expected_cross_source_ids)
        )
        missing_count = len(
            set(
                missing_shared_cross_source_ids
            )
        )

        if actual_count == 0:
            precision = (
                1.0
                if expected_count == 0
                else 0.0
            )
            over_rate = 0.0
        else:
            precision = (
                correct_count / actual_count
            )
            over_rate = (
                over_count / actual_count
            )

        if expected_count == 0:
            recall = 1.0
            under_rate = 0.0
        else:
            recall = (
                correct_count / expected_count
            )
            under_rate = (
                missing_count / expected_count
            )

        return (
            recall,
            precision,
            recall,
            over_rate,
            under_rate,
        )

    @staticmethod
    def _transfer_success(
        *,
        mode: ExperimentMode,
        has_cross_agent_target: bool,
        exact_match: float,
        shared_cross_source_recall: (
            float | None
        ),
        cross_agent_retrieval_recall: float,
        cross_agent_utilisation_recall: (
            float
        ),
    ) -> bool | None:
        if mode == "private_only":
            return None

        return bool(
            has_cross_agent_target
            and exact_match == 1.0
            and shared_cross_source_recall
            == 1.0
            and cross_agent_retrieval_recall
            == 1.0
            and cross_agent_utilisation_recall
            == 1.0
        )

    @staticmethod
    def _recall(
        *,
        observed: set[str],
        expected: set[str],
    ) -> float:
        if not expected:
            return 1.0

        return len(
            observed & expected
        ) / len(expected)

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
    def _field(
        value: Any,
        field_name: str,
        default: Any = None,
    ) -> Any:
        if isinstance(
            value,
            Mapping,
        ):
            return value.get(
                field_name,
                default,
            )

        return getattr(
            value,
            field_name,
            default,
        )

    @staticmethod
    def _clean_unique(
        values: Sequence[Any],
    ) -> list[str]:
        result: list[str] = []

        for value in values:
            clean_value = str(
                value
            ).strip()

            if (
                clean_value
                and clean_value not in result
            ):
                result.append(
                    clean_value
                )

        return result


def build_needle_persona_evaluation_input(
    *,
    sample: NeedlePersonaSample,
    memory_index: PrivateMemoryIndex,
    final_state: NeedlePersonaWorkflowState,
    config: (
        NeedlePersonaEvaluatorConfig | None
    ) = None,
) -> NeedlePersonaEvaluationInput:
    """
    Combine held-out benchmark evidence with a completed workflow state.

    This function must be called after workflow execution.
    """
    evaluator_config = (
        config
        or NeedlePersonaEvaluatorConfig()
    )

    if (
        memory_index.sample_id
        != sample.sample_id
    ):
        raise NeedlePersonaEvaluatorError(
            "memory_index.sample_id does not "
            "match sample.sample_id."
        )

    if (
        final_state["sample_id"]
        != sample.sample_id
    ):
        raise NeedlePersonaEvaluatorError(
            "final_state.sample_id does not "
            "match sample.sample_id."
        )

    if (
        final_state["run_id"]
        != memory_index.run_id
    ):
        raise NeedlePersonaEvaluatorError(
            "final_state.run_id does not match "
            "memory_index.run_id."
        )

    final_answer = final_state.get(
        "final_answer"
    )

    if final_answer is None:
        raise NeedlePersonaEvaluatorError(
            "final_state.final_answer is missing."
        )

    responder_agent_id = str(
        final_state[
            "responder_agent_id"
        ]
    ).strip()
    responder_persona = (
        AGENT_PERSONA_MAP.get(
            responder_agent_id
        )
    )

    if responder_persona is None:
        raise NeedlePersonaEvaluatorError(
            "Unknown responder_agent_id: "
            f"{responder_agent_id!r}."
        )

    target_personas = (
        extract_target_personas(
            question=sample.question,
            aliases=(
                evaluator_config
                .persona_aliases
            ),
        )
    )

    if not target_personas:
        target_personas = [
            AGENT_PERSONA_MAP[
                agent_id
            ]
            for agent_id in final_state[
                "selected_agent_ids"
            ]
            if agent_id
            in AGENT_PERSONA_MAP
        ]

    if not target_personas:
        raise NeedlePersonaEvaluatorError(
            "No target persona could be "
            "identified for the question."
        )

    if (
        responder_persona
        not in target_personas
    ):
        raise NeedlePersonaEvaluatorError(
            "Responder persona is not one of "
            "the question targets."
        )

    expected_source_ids: list[str] = []
    source_to_persona: dict[
        str,
        str,
    ] = {}

    for persona in target_personas:
        for source in sample.get_sources(
            persona
        ):
            source_id = str(
                source.source_id
            ).strip()

            if not source_id:
                continue

            source_to_persona[
                source_id
            ] = persona

            if (
                source_id
                not in expected_source_ids
            ):
                expected_source_ids.append(
                    source_id
                )

    if not expected_source_ids:
        raise NeedlePersonaEvaluatorError(
            "Question targets have no source "
            "evidence in the sample."
        )

    expected_local_source_ids = [
        source_id
        for source_id in expected_source_ids
        if source_to_persona[
            source_id
        ]
        == responder_persona
    ]
    expected_cross_source_ids = [
        source_id
        for source_id in expected_source_ids
        if source_to_persona[
            source_id
        ]
        != responder_persona
    ]

    memory_to_owner = (
        build_memory_id_to_owner_agent_id(
            memory_index
        )
    )
    memory_to_sources = (
        build_memory_id_to_source_ids(
            memory_index
        )
    )

    mapped_source_ids = {
        source_id
        for source_ids in (
            memory_to_sources.values()
        )
        for source_id in source_ids
    }

    missing_expected_sources = (
        set(expected_source_ids)
        - mapped_source_ids
    )

    if (
        evaluator_config
        .strict_memory_mapping
        and missing_expected_sources
    ):
        raise NeedlePersonaEvaluatorError(
            "Expected sources were not mapped "
            "to ingested memories: "
            f"{sorted(
                missing_expected_sources
            )}."
        )

    final_retrieved_ids = (
        NeedlePersonaEvaluator
        ._clean_unique(
            final_state[
                "final_retrieved_memory_ids"
            ]
        )
    )
    initial_retrieved_ids = (
        NeedlePersonaEvaluator
        ._clean_unique(
            final_state[
                "initial_retrieved_memory_ids"
            ]
        )
    )
    responder_retrieved_ids = (
        final_retrieved_ids
        if final_retrieved_ids
        else initial_retrieved_ids
    )

    access_request_ids: dict[
        str,
        str,
    ] = {}

    for memory_id, request in (
        final_state[
            "access_requests"
        ].items()
    ):
        request_id = (
            NeedlePersonaEvaluator
            ._field(
                request,
                "request_id",
                "",
            )
        )
        clean_request_id = str(
            request_id or ""
        ).strip()

        if clean_request_id:
            access_request_ids[
                memory_id
            ] = clean_request_id

    return NeedlePersonaEvaluationInput(
        run_id=final_state[
            "run_id"
        ],
        sample_id=sample.sample_id,
        experiment_mode=(
            final_state[
                "experiment_mode"
            ]
        ),
        question=sample.question,
        answer_status=(
            final_answer.status
        ),
        predicted_answer=(
            final_answer.answer or ""
        ),
        accepted_answers=(
            sample.accepted_answers()
        ),
        responder_agent_id=(
            responder_agent_id
        ),
        selected_agent_ids=(
            NeedlePersonaEvaluator
            ._clean_unique(
                final_state[
                    "selected_agent_ids"
                ]
            )
        ),
        contributing_agent_ids=(
            NeedlePersonaEvaluator
            ._clean_unique(
                final_answer
                .contributing_agent_ids
            )
        ),
        initial_retrieved_memory_ids=(
            initial_retrieved_ids
        ),
        final_retrieved_memory_ids=(
            final_retrieved_ids
        ),
        responder_retrieved_memory_ids=(
            responder_retrieved_ids
        ),
        used_memory_ids=(
            NeedlePersonaEvaluator
            ._clean_unique(
                final_answer
                .used_memory_ids
            )
        ),
        supporting_source_ids=(
            NeedlePersonaEvaluator
            ._clean_unique(
                final_answer
                .supporting_source_ids
            )
        ),
        candidate_memory_ids=(
            list(
                final_state[
                    "candidate_memories"
                ]
            )
        ),
        access_request_ids=(
            access_request_ids
        ),
        critic_reviews=dict(
            final_state[
                "critic_reviews"
            ]
        ),
        coordinator_outputs=dict(
            final_state[
                "coordinator_outputs"
            ]
        ),
        access_decisions=dict(
            final_state[
                "access_decisions"
            ]
        ),
        approved_memory_ids=(
            NeedlePersonaEvaluator
            ._clean_unique(
                final_state[
                    "approved_memory_ids"
                ]
            )
        ),
        rejected_memory_ids=(
            NeedlePersonaEvaluator
            ._clean_unique(
                final_state[
                    "rejected_memory_ids"
                ]
            )
        ),
        target_personas=list(
            target_personas
        ),
        expected_source_ids=(
            expected_source_ids
        ),
        expected_local_source_ids=(
            expected_local_source_ids
        ),
        expected_cross_agent_source_ids=(
            expected_cross_source_ids
        ),
        memory_id_to_source_ids=(
            memory_to_sources
        ),
        memory_id_to_owner_agent_id=(
            memory_to_owner
        ),
        node_trace=list(
            final_state[
                "node_trace"
            ]
        ),
        errors=list(
            final_state[
                "errors"
            ]
        ),
        warnings=list(
            final_state[
                "warnings"
            ]
        ),
        node_metrics=dict(
            final_state[
                "node_metrics"
            ]
        ),
    )


def extract_target_personas(
    *,
    question: str,
    aliases: Mapping[
        str,
        PersonaName,
    ] | None = None,
) -> list[PersonaName]:
    """
    Extract target personas in question order.
    """
    text = str(
        question or ""
    )
    mentions: list[
        tuple[int, PersonaName]
    ] = []

    names: dict[
        str,
        PersonaName,
    ] = {
        persona: persona
        for persona in PERSONA_NAMES
    }

    for alias, persona in (
        aliases or {}
    ).items():
        names[
            str(alias).casefold()
        ] = persona

    for name, persona in names.items():
        match = re.search(
            rf"\b{re.escape(name)}\b",
            text,
            flags=re.IGNORECASE,
        )

        if match is not None:
            mentions.append(
                (
                    match.start(),
                    persona,
                )
            )

    mentions.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    result: list[PersonaName] = []

    for _, persona in mentions:
        if persona not in result:
            result.append(
                persona
            )

    return result


def build_memory_id_to_owner_agent_id(
    memory_index: PrivateMemoryIndex,
) -> dict[str, str]:
    """
    Build an ownership mapping from the ingestion index.
    """
    result: dict[str, str] = {}

    for agent_id, memory_ids in (
        memory_index
        .private_memory_ids
        .items()
    ):
        for memory_id in memory_ids:
            clean_memory_id = str(
                memory_id
            ).strip()

            if not clean_memory_id:
                continue

            existing_owner = result.get(
                clean_memory_id
            )

            if (
                existing_owner is not None
                and existing_owner
                != agent_id
            ):
                raise NeedlePersonaEvaluatorError(
                    "Memory ID is assigned to "
                    "multiple owners: "
                    f"{clean_memory_id!r}."
                )

            result[
                clean_memory_id
            ] = agent_id

    return result


def build_memory_id_to_source_ids(
    memory_index: PrivateMemoryIndex,
) -> dict[str, list[str]]:
    """
    Reverse ``source_to_memory_ids`` while preserving source order.
    """
    known_memory_ids = set(
        build_memory_id_to_owner_agent_id(
            memory_index
        )
    )
    result: dict[
        str,
        list[str],
    ] = {
        memory_id: []
        for memory_id in known_memory_ids
    }

    for source_id, memory_ids in (
        memory_index
        .source_to_memory_ids
        .items()
    ):
        clean_source_id = str(
            source_id
        ).strip()

        if not clean_source_id:
            continue

        for memory_id in memory_ids:
            clean_memory_id = str(
                memory_id
            ).strip()

            if not clean_memory_id:
                continue

            if (
                clean_memory_id
                not in known_memory_ids
            ):
                raise NeedlePersonaEvaluatorError(
                    "source_to_memory_ids "
                    "contains an unknown memory ID: "
                    f"{clean_memory_id!r}."
                )

            if (
                clean_source_id
                not in result[
                    clean_memory_id
                ]
            ):
                result[
                    clean_memory_id
                ].append(
                    clean_source_id
                )

    return result


def evaluate_needle_persona(
    *,
    sample: NeedlePersonaSample,
    memory_index: PrivateMemoryIndex,
    final_state: NeedlePersonaWorkflowState,
    config: (
        NeedlePersonaEvaluatorConfig | None
    ) = None,
) -> NeedlePersonaEvaluationResult:
    """
    Convenience wrapper for one complete evaluation.
    """
    return NeedlePersonaEvaluator(
        config=config
    ).evaluate_run(
        sample=sample,
        memory_index=memory_index,
        final_state=final_state,
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
]