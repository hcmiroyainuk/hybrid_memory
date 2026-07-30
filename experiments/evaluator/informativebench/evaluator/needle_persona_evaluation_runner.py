from __future__ import annotations

import csv
import json
import math
import re
import shutil
import statistics
import traceback
from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)
from dataclasses import (
    asdict,
    dataclass,
    field,
    is_dataclass,
)
from datetime import (
    datetime,
    timezone,
)
from enum import Enum
from io import StringIO
from pathlib import Path
from time import perf_counter
from typing import (
    Any,
    Protocol,
    runtime_checkable,
)
from uuid import uuid4

from pydantic import BaseModel

from ..config.needle_persona_config import (
    DEFAULT_EXPERIMENT_MODES,
    ExperimentMode,
    NeedlePersonaExperimentConfig,
    normalise_experiment_modes,
)
from ..data_preparing.needle_persona_models import (
    NeedlePersonaSample,
    PrivateMemoryIndex,
)
from .needle_persona_evaluator import (
    NeedlePersonaEvaluationResult,
    NeedlePersonaEvaluator,
    build_needle_persona_evaluation_input,
)
from ..workflow.needle_persona_workflow import (
    NeedlePersonaWorkflow,
)
from ..workflow.state.needle_persona_state import (
    NeedlePersonaWorkflowState,
    build_initial_needle_persona_state,
)


_RESULT_SCHEMA_VERSION = "2.0"

_AGGREGATE_METRIC_FIELDS: tuple[
    str,
    ...,
] = (
    "exact_match",
    "token_precision",
    "token_recall",
    "token_f1",
    "entity_precision",
    "entity_recall",
    "entity_f1",
    "target_source_retrieval_recall",
    "retrieval_hit",
    "cross_agent_source_retrieval_recall",
    "cross_agent_memory_hit",
    "target_source_utilisation_recall",
    "cross_agent_source_utilisation_recall",
    "cross_agent_source_utilised",
    "shared_cross_agent_source_recall",
    "governance_precision",
    "governance_recall",
    "over_sharing_rate",
    "under_sharing_rate",
    "candidate_memory_count",
    "access_request_count",
    "critic_review_count",
    "critic_approved_memory_count",
    "coordinator_decision_count",
    "coordinator_approved_memory_count",
    "access_granted_count",
    "approved_memory_count",
    "rejected_memory_count",
    "transfer_success",
    "run_duration_ms",
)

_RESULT_CSV_FIELDS: tuple[
    str,
    ...,
] = (
    "run_id",
    "sample_id",
    "experiment_mode",
    "answer_status",
    "predicted_answer",
    "matched_reference_answer",
    "exact_match",
    "token_precision",
    "token_recall",
    "token_f1",
    "entity_precision",
    "entity_recall",
    "entity_f1",
    "target_source_retrieval_recall",
    "retrieval_hit",
    "cross_agent_source_retrieval_recall",
    "cross_agent_memory_hit",
    "target_source_utilisation_recall",
    "cross_agent_source_utilisation_recall",
    "cross_agent_source_utilised",
    "shared_cross_agent_source_recall",
    "governance_precision",
    "governance_recall",
    "over_sharing_rate",
    "under_sharing_rate",
    "transfer_success",
    "candidate_memory_count",
    "access_request_count",
    "critic_review_count",
    "coordinator_decision_count",
    "approved_memory_count",
    "rejected_memory_count",
    "responder_agent_id",
    "target_personas",
    "expected_source_ids",
    "retrieved_source_ids",
    "used_source_ids",
    "shared_source_ids",
    "initial_retrieved_memory_ids",
    "final_retrieved_memory_ids",
    "used_memory_ids",
    "approved_memory_ids",
    "rejected_memory_ids",
    "run_duration_ms",
)


@runtime_checkable
class NeedlePersonaLoaderProtocol(
    Protocol
):
    """
    Dataset loader contract required by the runner.
    """

    def iter_samples(
        self,
        limit: int | None = None,
    ) -> Any:
        ...


@runtime_checkable
class NeedlePersonaIngestionProtocol(
    Protocol
):
    """
    Ingestion contract required by one isolated runtime.
    """

    def ingest(
        self,
        *,
        sample: NeedlePersonaSample,
        run_id: str,
    ) -> PrivateMemoryIndex:
        ...


@dataclass(
    frozen=True,
    slots=True,
)
class NeedlePersonaIngestionSnapshot:
    """
    Canonical private-memory state produced once for one benchmark sample.

    Every experiment mode receives an isolated file copy of this snapshot.
    The memory IDs and extracted content therefore remain identical across
    ``private_only``, ``ungoverned_shared``, and ``governed_shared``.
    """

    seed_run_id: str
    sample_id: str
    memory_index: PrivateMemoryIndex

    memories_path: Path
    operation_logs_path: Path

    created_at_utc: str
    ingestion_duration_ms: float
    metadata: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        seed_run_id = str(
            self.seed_run_id
        ).strip()
        sample_id = str(
            self.sample_id
        ).strip()
        memories_path = Path(
            self.memories_path
        )
        operation_logs_path = Path(
            self.operation_logs_path
        )

        if not seed_run_id:
            raise ValueError(
                "seed_run_id cannot be empty."
            )

        if not sample_id:
            raise ValueError(
                "sample_id cannot be empty."
            )

        if not isinstance(
            self.memory_index,
            PrivateMemoryIndex,
        ):
            raise TypeError(
                "memory_index must be a "
                "PrivateMemoryIndex."
            )

        if (
            self.memory_index.sample_id
            != sample_id
        ):
            raise ValueError(
                "Snapshot sample_id does not match "
                "memory_index.sample_id."
            )

        if (
            self.memory_index.run_id
            != seed_run_id
        ):
            raise ValueError(
                "Snapshot seed_run_id does not match "
                "memory_index.run_id."
            )

        if self.ingestion_duration_ms < 0:
            raise ValueError(
                "ingestion_duration_ms cannot be "
                "negative."
            )

        if not memories_path.is_file():
            raise FileNotFoundError(
                "Snapshot memories file does not "
                f"exist: {memories_path}."
            )

        if not operation_logs_path.is_file():
            raise FileNotFoundError(
                "Snapshot operation-log file does "
                f"not exist: {operation_logs_path}."
            )

        object.__setattr__(
            self,
            "seed_run_id",
            seed_run_id,
        )
        object.__setattr__(
            self,
            "sample_id",
            sample_id,
        )
        object.__setattr__(
            self,
            "memories_path",
            memories_path,
        )
        object.__setattr__(
            self,
            "operation_logs_path",
            operation_logs_path,
        )
        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )

    def memory_index_for_run(
        self,
        run_id: str,
    ) -> PrivateMemoryIndex:
        """
        Return an independent index carrying the mode-specific run ID.

        The underlying memory IDs remain unchanged because every mode starts
        from the same canonical JSON snapshot.
        """
        clean_run_id = str(
            run_id
        ).strip()

        if not clean_run_id:
            raise ValueError(
                "run_id cannot be empty."
            )

        return self.memory_index.model_copy(
            deep=True,
            update={
                "run_id": clean_run_id,
            },
        )

    def as_metadata(
        self,
    ) -> dict[str, Any]:
        return {
            "canonical_ingestion_seed_run_id": (
                self.seed_run_id
            ),
            "canonical_ingestion_sample_id": (
                self.sample_id
            ),
            "canonical_memories_path": str(
                self.memories_path.resolve()
            ),
            "canonical_operation_logs_path": str(
                self.operation_logs_path.resolve()
            ),
            "canonical_ingestion_created_at_utc": (
                self.created_at_utc
            ),
            "canonical_ingestion_duration_ms": round(
                self.ingestion_duration_ms,
                3,
            ),
            "canonical_memory_count": sum(
                len(memory_ids)
                for memory_ids in (
                    self.memory_index
                    .private_memory_ids
                    .values()
                )
            ),
            **dict(self.metadata),
        }


@runtime_checkable
class NeedlePersonaRuntimeFactoryProtocol(
    Protocol
):
    """
    Runtime-factory contract for canonical ingestion and isolated mode runs.
    """

    def __call__(
        self,
        run_id: str,
        sample: NeedlePersonaSample,
        mode: ExperimentMode,
    ) -> "NeedlePersonaEvaluationRuntime":
        ...

    def create_ingestion_snapshot(
        self,
        *,
        run_id: str,
        sample: NeedlePersonaSample,
    ) -> NeedlePersonaIngestionSnapshot:
        ...

    def build_from_snapshot(
        self,
        *,
        run_id: str,
        sample: NeedlePersonaSample,
        mode: ExperimentMode,
        snapshot: NeedlePersonaIngestionSnapshot,
    ) -> "NeedlePersonaEvaluationRuntime":
        ...


RunRuntimeFactory = (
    NeedlePersonaRuntimeFactoryProtocol
)

ProgressCallback = Callable[
    [Mapping[str, Any]],
    None,
]


class NeedlePersonaBatchRunError(
    RuntimeError
):
    """
    Raised when a batch is configured to stop after the first failed run.
    """

    def __init__(
        self,
        message: str,
        *,
        report: (
            "NeedlePersonaBatchReport | None"
        ) = None,
    ) -> None:
        super().__init__(message)
        self.report = report


@dataclass(
    frozen=True,
    slots=True,
)
class NeedlePersonaEvaluationRuntime:
    """
    Fresh service graph for one ``sample × mode`` run.

    The runtime factory must create isolated JSON persistence paths because
    approved ACL changes remain valid until this runtime is discarded.
    """

    ingestion_service: (
        NeedlePersonaIngestionProtocol
    )
    workflow: NeedlePersonaWorkflow
    evaluator: NeedlePersonaEvaluator = field(
        default_factory=NeedlePersonaEvaluator
    )
    metadata: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.ingestion_service,
            NeedlePersonaIngestionProtocol,
        ):
            raise TypeError(
                "ingestion_service must implement "
                "NeedlePersonaIngestionProtocol."
            )

        if not isinstance(
            self.evaluator,
            NeedlePersonaEvaluator,
        ):
            raise TypeError(
                "evaluator must be a "
                "NeedlePersonaEvaluator."
            )


@dataclass(
    frozen=True,
    slots=True,
)
class NeedlePersonaRunFailure:
    """
    Serializable failure record for one sample-mode run.
    """

    run_id: str
    sample_id: str
    experiment_mode: ExperimentMode

    stage: str
    error_type: str
    error_message: str
    traceback_text: str

    started_at_utc: str
    completed_at_utc: str
    duration_ms: float

    def as_dict(
        self,
    ) -> dict[str, Any]:
        return asdict(self)


@dataclass(
    frozen=True,
    slots=True,
)
class NeedlePersonaRunTrace:
    """
    Sanitised execution trace saved after a successful run.

    It contains IDs, structured outputs, metrics, and workflow state, but no
    raw persona source documents or gold-answer fields.
    """

    run_id: str
    sample_id: str
    experiment_mode: ExperimentMode

    memory_index: Mapping[
        str,
        Any,
    ]
    final_state: Mapping[
        str,
        Any,
    ]
    runtime_metadata: Mapping[
        str,
        Any,
    ]

    started_at_utc: str
    completed_at_utc: str
    duration_ms: float

    def as_dict(
        self,
    ) -> dict[str, Any]:
        return asdict(self)


@dataclass(
    frozen=True,
    slots=True,
)
class NeedlePersonaBatchReport:
    """
    Files and records produced by one batch.
    """

    experiment_id: str
    output_dir: Path

    results: tuple[
        NeedlePersonaEvaluationResult,
        ...,
    ]
    failures: tuple[
        NeedlePersonaRunFailure,
        ...,
    ]
    traces: tuple[
        NeedlePersonaRunTrace,
        ...,
    ]

    summary: Mapping[
        str,
        Any,
    ]

    @property
    def manifest_path(
        self,
    ) -> Path:
        return (
            self.output_dir
            / "manifest.json"
        )

    @property
    def results_jsonl_path(
        self,
    ) -> Path:
        return (
            self.output_dir
            / "results.jsonl"
        )

    @property
    def results_csv_path(
        self,
    ) -> Path:
        return (
            self.output_dir
            / "results.csv"
        )

    @property
    def failures_jsonl_path(
        self,
    ) -> Path:
        return (
            self.output_dir
            / "failures.jsonl"
        )

    @property
    def traces_jsonl_path(
        self,
    ) -> Path:
        return (
            self.output_dir
            / "traces.jsonl"
        )

    @property
    def summary_path(
        self,
    ) -> Path:
        return (
            self.output_dir
            / "summary.json"
        )


class NeedlePersonaEvaluationRunner:
    """
    Sequential runner with one canonical ingestion per selected sample.

    LLM extraction runs once for each sample. Every experiment mode then receives
    an isolated copy of the same initial memories and operation-log history, so
    ACL and promotion changes cannot leak across modes while ingestion variance
    is removed from the comparison.
    """

    def __init__(
        self,
        *,
        loader: NeedlePersonaLoaderProtocol,
        runtime_factory: RunRuntimeFactory,
        config: (
            NeedlePersonaExperimentConfig
            | None
        ) = None,
        runtime_metadata: (
            Mapping[str, Any] | None
        ) = None,
    ) -> None:
        if not isinstance(
            loader,
            NeedlePersonaLoaderProtocol,
        ):
            raise TypeError(
                "loader must implement "
                "NeedlePersonaLoaderProtocol."
            )

        if not callable(
            runtime_factory
        ):
            raise TypeError(
                "runtime_factory must be callable."
            )

        required_factory_methods = (
            "create_ingestion_snapshot",
            "build_from_snapshot",
        )

        missing_factory_methods = [
            method_name
            for method_name in (
                required_factory_methods
            )
            if not callable(
                getattr(
                    runtime_factory,
                    method_name,
                    None,
                )
            )
        ]

        if missing_factory_methods:
            raise TypeError(
                "runtime_factory must implement "
                "canonical snapshot methods: "
                f"{missing_factory_methods}."
            )

        self.loader = loader
        self.runtime_factory = (
            runtime_factory
        )
        self.config = (
            config
            or NeedlePersonaExperimentConfig()
        )
        self.runtime_metadata = dict(
            runtime_metadata or {}
        )

    # ------------------------------------------------------------------
    # Public execution
    # ------------------------------------------------------------------

    def run_batch(
        self,
        *,
        experiment_id: str | None = None,
        modes: (
            Sequence[ExperimentMode]
            | None
        ) = None,
        limit: int | None = None,
        sample_ids: (
            Sequence[str] | None
        ) = None,
        continue_on_error: bool = True,
        save_after_each_run: bool = True,
        overwrite: bool | None = None,
        progress_callback: (
            ProgressCallback | None
        ) = None,
    ) -> NeedlePersonaBatchReport:
        """
        Execute all selected sample-mode pairs and persist outputs.
        """
        selected_modes = (
            normalise_experiment_modes(
                modes
                if modes is not None
                else self.config.modes
            )
        )

        selected_limit = (
            limit
            if limit is not None
            else self.config.sample_limit
        )

        if (
            selected_limit is not None
            and selected_limit <= 0
        ):
            raise ValueError(
                "limit must be greater than zero "
                "or None."
            )

        allow_overwrite = (
            self.config.overwrite_output
            if overwrite is None
            else bool(overwrite)
        )

        clean_experiment_id = (
            self._safe_identifier(
                experiment_id
            )
            if experiment_id
            else self._new_experiment_id()
        )

        output_dir = (
            self.config.output_root
            / clean_experiment_id
        )
        self._prepare_output_dir(
            output_dir,
            overwrite=allow_overwrite,
        )

        if (
            self.config
            .reset_runtime_root_before_batch
        ):
            self._reset_runtime_root()

        samples = self._load_samples(
            limit=selected_limit,
            sample_ids=sample_ids,
        )

        if not samples:
            raise ValueError(
                "No benchmark samples were selected."
            )

        expected_run_count = (
            len(samples)
            * len(selected_modes)
        )
        started_at = self._utc_now()

        results: list[
            NeedlePersonaEvaluationResult
        ] = []
        failures: list[
            NeedlePersonaRunFailure
        ] = []
        traces: list[
            NeedlePersonaRunTrace
        ] = []

        manifest = self._build_manifest(
            experiment_id=(
                clean_experiment_id
            ),
            output_dir=output_dir,
            modes=selected_modes,
            samples=samples,
            expected_run_count=(
                expected_run_count
            ),
            started_at_utc=started_at,
            status="running",
        )

        self._write_json(
            output_dir / "manifest.json",
            manifest,
        )
        self._persist_outputs(
            output_dir=output_dir,
            experiment_id=(
                clean_experiment_id
            ),
            modes=selected_modes,
            samples=samples,
            expected_run_count=(
                expected_run_count
            ),
            started_at_utc=started_at,
            completed_at_utc=None,
            status="running",
            results=results,
            failures=failures,
            traces=traces,
        )

        completed_run_count = 0

        try:
            for sample_index, sample in (
                enumerate(samples)
            ):
                snapshot_run_id = (
                    self._build_ingestion_run_id(
                        experiment_id=(
                            clean_experiment_id
                        ),
                        sample_index=(
                            sample_index
                        ),
                        sample_id=(
                            sample.sample_id
                        ),
                    )
                )
                snapshot_started_at = (
                    self._utc_now()
                )
                snapshot_started_perf = (
                    perf_counter()
                )

                self._notify(
                    progress_callback,
                    {
                        "event": (
                            "ingestion_started"
                        ),
                        "run_id": snapshot_run_id,
                        "sample_id": (
                            sample.sample_id
                        ),
                        "completed_run_count": (
                            completed_run_count
                        ),
                        "expected_run_count": (
                            expected_run_count
                        ),
                    },
                )

                try:
                    snapshot = (
                        self.runtime_factory
                        .create_ingestion_snapshot(
                            run_id=(
                                snapshot_run_id
                            ),
                            sample=sample,
                        )
                    )

                    if not isinstance(
                        snapshot,
                        NeedlePersonaIngestionSnapshot,
                    ):
                        raise TypeError(
                            "create_ingestion_snapshot "
                            "must return "
                            "NeedlePersonaIngestionSnapshot."
                        )

                except Exception as error:
                    snapshot_completed_at = (
                        self._utc_now()
                    )
                    snapshot_duration_ms = (
                        self._elapsed_ms(
                            snapshot_started_perf
                        )
                    )
                    traceback_text = (
                        traceback.format_exc()
                    )

                    for mode in selected_modes:
                        run_id = (
                            self._build_run_id(
                                experiment_id=(
                                    clean_experiment_id
                                ),
                                sample_index=(
                                    sample_index
                                ),
                                sample_id=(
                                    sample.sample_id
                                ),
                                mode=mode,
                            )
                        )

                        failure = (
                            NeedlePersonaRunFailure(
                                run_id=run_id,
                                sample_id=(
                                    sample.sample_id
                                ),
                                experiment_mode=mode,
                                stage=(
                                    "canonical_ingestion"
                                ),
                                error_type=(
                                    type(error).__name__
                                ),
                                error_message=str(
                                    error
                                ),
                                traceback_text=(
                                    traceback_text
                                ),
                                started_at_utc=(
                                    snapshot_started_at
                                ),
                                completed_at_utc=(
                                    snapshot_completed_at
                                ),
                                duration_ms=round(
                                    snapshot_duration_ms,
                                    3,
                                ),
                            )
                        )
                        failures.append(
                            failure
                        )
                        completed_run_count += 1

                        self._notify(
                            progress_callback,
                            {
                                "event": (
                                    "run_failed"
                                ),
                                "run_id": run_id,
                                "sample_id": (
                                    sample.sample_id
                                ),
                                "experiment_mode": (
                                    mode
                                ),
                                "completed_run_count": (
                                    completed_run_count
                                ),
                                "expected_run_count": (
                                    expected_run_count
                                ),
                                "failure": (
                                    failure.as_dict()
                                ),
                            },
                        )

                    self._notify(
                        progress_callback,
                        {
                            "event": (
                                "ingestion_failed"
                            ),
                            "run_id": (
                                snapshot_run_id
                            ),
                            "sample_id": (
                                sample.sample_id
                            ),
                            "error_type": (
                                type(error).__name__
                            ),
                            "error_message": str(
                                error
                            ),
                        },
                    )

                    if save_after_each_run:
                        self._persist_outputs(
                            output_dir=(
                                output_dir
                            ),
                            experiment_id=(
                                clean_experiment_id
                            ),
                            modes=(
                                selected_modes
                            ),
                            samples=samples,
                            expected_run_count=(
                                expected_run_count
                            ),
                            started_at_utc=(
                                started_at
                            ),
                            completed_at_utc=None,
                            status="running",
                            results=results,
                            failures=failures,
                            traces=traces,
                        )

                    if not continue_on_error:
                        completed_at = (
                            self._utc_now()
                        )
                        report = self._finalise(
                            output_dir=(
                                output_dir
                            ),
                            experiment_id=(
                                clean_experiment_id
                            ),
                            modes=(
                                selected_modes
                            ),
                            samples=samples,
                            expected_run_count=(
                                expected_run_count
                            ),
                            started_at_utc=(
                                started_at
                            ),
                            completed_at_utc=(
                                completed_at
                            ),
                            status=(
                                "stopped_after_failure"
                            ),
                            results=results,
                            failures=failures,
                            traces=traces,
                            manifest=manifest,
                        )

                        raise (
                            NeedlePersonaBatchRunError(
                                "Batch stopped after "
                                "canonical ingestion "
                                "failed. See "
                                f"{report.failures_jsonl_path}.",
                                report=report,
                            )
                        )

                    continue

                self._notify(
                    progress_callback,
                    {
                        "event": (
                            "ingestion_succeeded"
                        ),
                        "run_id": (
                            snapshot_run_id
                        ),
                        "sample_id": (
                            sample.sample_id
                        ),
                        "ingestion_duration_ms": (
                            snapshot
                            .ingestion_duration_ms
                        ),
                        "canonical_memory_count": (
                            sum(
                                len(memory_ids)
                                for memory_ids in (
                                    snapshot
                                    .memory_index
                                    .private_memory_ids
                                    .values()
                                )
                            )
                        ),
                    },
                )

                for mode in selected_modes:
                    run_id = self._build_run_id(
                        experiment_id=(
                            clean_experiment_id
                        ),
                        sample_index=(
                            sample_index
                        ),
                        sample_id=(
                            sample.sample_id
                        ),
                        mode=mode,
                    )

                    self._notify(
                        progress_callback,
                        {
                            "event": (
                                "run_started"
                            ),
                            "run_id": run_id,
                            "sample_id": (
                                sample.sample_id
                            ),
                            "experiment_mode": (
                                mode
                            ),
                            "completed_run_count": (
                                completed_run_count
                            ),
                            "expected_run_count": (
                                expected_run_count
                            ),
                            "canonical_ingestion_run_id": (
                                snapshot.seed_run_id
                            ),
                        },
                    )

                    (
                        result,
                        failure,
                        trace,
                    ) = self._run_one(
                        sample=sample,
                        mode=mode,
                        run_id=run_id,
                        snapshot=snapshot,
                    )

                    if result is not None:
                        results.append(
                            result
                        )

                    if failure is not None:
                        failures.append(
                            failure
                        )

                    if trace is not None:
                        traces.append(
                            trace
                        )

                    completed_run_count += 1

                    if save_after_each_run:
                        self._persist_outputs(
                            output_dir=(
                                output_dir
                            ),
                            experiment_id=(
                                clean_experiment_id
                            ),
                            modes=(
                                selected_modes
                            ),
                            samples=samples,
                            expected_run_count=(
                                expected_run_count
                            ),
                            started_at_utc=(
                                started_at
                            ),
                            completed_at_utc=None,
                            status="running",
                            results=results,
                            failures=failures,
                            traces=traces,
                        )

                    self._notify(
                        progress_callback,
                        {
                            "event": (
                                "run_succeeded"
                                if result is not None
                                else "run_failed"
                            ),
                            "run_id": run_id,
                            "sample_id": (
                                sample.sample_id
                            ),
                            "experiment_mode": (
                                mode
                            ),
                            "completed_run_count": (
                                completed_run_count
                            ),
                            "expected_run_count": (
                                expected_run_count
                            ),
                            "failure": (
                                failure.as_dict()
                                if failure
                                is not None
                                else None
                            ),
                        },
                    )

                    if (
                        failure is not None
                        and not continue_on_error
                    ):
                        completed_at = (
                            self._utc_now()
                        )
                        report = self._finalise(
                            output_dir=(
                                output_dir
                            ),
                            experiment_id=(
                                clean_experiment_id
                            ),
                            modes=(
                                selected_modes
                            ),
                            samples=samples,
                            expected_run_count=(
                                expected_run_count
                            ),
                            started_at_utc=(
                                started_at
                            ),
                            completed_at_utc=(
                                completed_at
                            ),
                            status=(
                                "stopped_after_failure"
                            ),
                            results=results,
                            failures=failures,
                            traces=traces,
                            manifest=manifest,
                        )

                        raise (
                            NeedlePersonaBatchRunError(
                                "Batch stopped after "
                                "a failed run. See "
                                f"{report.failures_jsonl_path}.",
                                report=report,
                            )
                        )

        except KeyboardInterrupt:
            completed_at = self._utc_now()
            self._finalise(
                output_dir=output_dir,
                experiment_id=(
                    clean_experiment_id
                ),
                modes=selected_modes,
                samples=samples,
                expected_run_count=(
                    expected_run_count
                ),
                started_at_utc=started_at,
                completed_at_utc=(
                    completed_at
                ),
                status="interrupted",
                results=results,
                failures=failures,
                traces=traces,
                manifest=manifest,
            )
            raise

        completed_at = self._utc_now()
        final_status = (
            "completed_with_failures"
            if failures
            else "completed"
        )

        return self._finalise(
            output_dir=output_dir,
            experiment_id=(
                clean_experiment_id
            ),
            modes=selected_modes,
            samples=samples,
            expected_run_count=(
                expected_run_count
            ),
            started_at_utc=started_at,
            completed_at_utc=(
                completed_at
            ),
            status=final_status,
            results=results,
            failures=failures,
            traces=traces,
            manifest=manifest,
        )

    # ------------------------------------------------------------------
    # One isolated run
    # ------------------------------------------------------------------

    def _run_one(
        self,
        *,
        sample: NeedlePersonaSample,
        mode: ExperimentMode,
        run_id: str,
        snapshot: (
            NeedlePersonaIngestionSnapshot
        ),
    ) -> tuple[
        NeedlePersonaEvaluationResult
        | None,
        NeedlePersonaRunFailure | None,
        NeedlePersonaRunTrace | None,
    ]:
        started_at = self._utc_now()
        started_perf = perf_counter()
        stage = "runtime_initialisation"

        stage_durations: dict[
            str,
            float,
        ] = {}

        try:
            stage_started = perf_counter()
            runtime = (
                self.runtime_factory
                .build_from_snapshot(
                    run_id=run_id,
                    sample=sample,
                    mode=mode,
                    snapshot=snapshot,
                )
            )
            stage_durations[
                stage
            ] = self._elapsed_ms(
                stage_started
            )

            if not isinstance(
                runtime,
                NeedlePersonaEvaluationRuntime,
            ):
                raise TypeError(
                    "runtime_factory must return "
                    "NeedlePersonaEvaluationRuntime."
                )

            stage = "snapshot_index"
            stage_started = perf_counter()
            memory_index = (
                snapshot.memory_index_for_run(
                    run_id
                )
            )
            stage_durations[
                stage
            ] = self._elapsed_ms(
                stage_started
            )

            stage = "initial_state"
            stage_started = perf_counter()
            initial_state = (
                build_initial_needle_persona_state(
                    sample=sample,
                    memory_index=memory_index,
                    experiment_mode=mode,
                )
            )
            stage_durations[
                stage
            ] = self._elapsed_ms(
                stage_started
            )

            stage = "workflow"
            stage_started = perf_counter()
            final_state = (
                runtime.workflow.invoke(
                    initial_state
                )
            )
            stage_durations[
                stage
            ] = self._elapsed_ms(
                stage_started
            )

            stage = "evaluation_input"
            stage_started = perf_counter()
            evaluation_input = (
                build_needle_persona_evaluation_input(
                    sample=sample,
                    memory_index=memory_index,
                    final_state=final_state,
                    config=runtime.evaluator.config,
                )
            )
            stage_durations[
                stage
            ] = self._elapsed_ms(
                stage_started
            )

            stage = "evaluation"
            stage_started = perf_counter()
            result = runtime.evaluator.evaluate(
                evaluation_input
            )
            stage_durations[
                stage
            ] = self._elapsed_ms(
                stage_started
            )

            completed_at = self._utc_now()
            duration_ms = self._elapsed_ms(
                started_perf
            )

            result[
                "run_duration_ms"
            ] = round(
                duration_ms,
                3,
            )
            result[
                "stage_durations_ms"
            ] = {
                key: round(
                    value,
                    3,
                )
                for key, value in (
                    stage_durations.items()
                )
            }
            result[
                "runtime_metadata"
            ] = {
                **self.runtime_metadata,
                **snapshot.as_metadata(),
                **dict(runtime.metadata),
            }

            trace = (
                NeedlePersonaRunTrace(
                    run_id=run_id,
                    sample_id=(
                        sample.sample_id
                    ),
                    experiment_mode=mode,
                    memory_index=(
                        self._jsonable(
                            memory_index
                        )
                    ),
                    final_state=(
                        self._sanitise_final_state(
                            final_state
                        )
                    ),
                    runtime_metadata={
                        **self.runtime_metadata,
                        **snapshot.as_metadata(),
                        **dict(
                            runtime.metadata
                        ),
                    },
                    started_at_utc=(
                        started_at
                    ),
                    completed_at_utc=(
                        completed_at
                    ),
                    duration_ms=round(
                        duration_ms,
                        3,
                    ),
                )
            )

            return result, None, trace

        except Exception as error:
            completed_at = self._utc_now()
            duration_ms = self._elapsed_ms(
                started_perf
            )

            failure = (
                NeedlePersonaRunFailure(
                    run_id=run_id,
                    sample_id=(
                        sample.sample_id
                    ),
                    experiment_mode=mode,
                    stage=stage,
                    error_type=(
                        type(error).__name__
                    ),
                    error_message=str(error),
                    traceback_text=(
                        traceback.format_exc()
                    ),
                    started_at_utc=(
                        started_at
                    ),
                    completed_at_utc=(
                        completed_at
                    ),
                    duration_ms=round(
                        duration_ms,
                        3,
                    ),
                )
            )

            return None, failure, None

    # ------------------------------------------------------------------
    # Persistence and reporting
    # ------------------------------------------------------------------

    def _finalise(
        self,
        *,
        output_dir: Path,
        experiment_id: str,
        modes: Sequence[
            ExperimentMode
        ],
        samples: Sequence[
            NeedlePersonaSample
        ],
        expected_run_count: int,
        started_at_utc: str,
        completed_at_utc: str,
        status: str,
        results: Sequence[
            NeedlePersonaEvaluationResult
        ],
        failures: Sequence[
            NeedlePersonaRunFailure
        ],
        traces: Sequence[
            NeedlePersonaRunTrace
        ],
        manifest: Mapping[str, Any],
    ) -> NeedlePersonaBatchReport:
        summary = self._persist_outputs(
            output_dir=output_dir,
            experiment_id=experiment_id,
            modes=modes,
            samples=samples,
            expected_run_count=(
                expected_run_count
            ),
            started_at_utc=(
                started_at_utc
            ),
            completed_at_utc=(
                completed_at_utc
            ),
            status=status,
            results=results,
            failures=failures,
            traces=traces,
        )

        final_manifest = dict(
            manifest
        )
        final_manifest.update(
            {
                "status": status,
                "completed_at_utc": (
                    completed_at_utc
                ),
                "successful_run_count": (
                    len(results)
                ),
                "failed_run_count": (
                    len(failures)
                ),
            }
        )

        self._write_json(
            output_dir / "manifest.json",
            final_manifest,
        )

        return NeedlePersonaBatchReport(
            experiment_id=experiment_id,
            output_dir=output_dir,
            results=tuple(results),
            failures=tuple(failures),
            traces=tuple(traces),
            summary=summary,
        )

    def _persist_outputs(
        self,
        *,
        output_dir: Path,
        experiment_id: str,
        modes: Sequence[
            ExperimentMode
        ],
        samples: Sequence[
            NeedlePersonaSample
        ],
        expected_run_count: int,
        started_at_utc: str,
        completed_at_utc: (
            str | None
        ),
        status: str,
        results: Sequence[
            NeedlePersonaEvaluationResult
        ],
        failures: Sequence[
            NeedlePersonaRunFailure
        ],
        traces: Sequence[
            NeedlePersonaRunTrace
        ],
    ) -> dict[str, Any]:
        self._write_jsonl(
            output_dir / "results.jsonl",
            list(results),
        )
        self._write_results_csv(
            output_dir / "results.csv",
            results,
        )
        self._write_jsonl(
            output_dir / "failures.jsonl",
            [
                failure.as_dict()
                for failure in failures
            ],
        )

        if (
            self.config
            .save_detailed_trace
        ):
            self._write_jsonl(
                output_dir / "traces.jsonl",
                [
                    trace.as_dict()
                    for trace in traces
                ],
            )

        summary = (
            build_needle_persona_summary(
                experiment_id=(
                    experiment_id
                ),
                status=status,
                started_at_utc=(
                    started_at_utc
                ),
                completed_at_utc=(
                    completed_at_utc
                ),
                requested_modes=modes,
                sample_count=len(
                    samples
                ),
                expected_run_count=(
                    expected_run_count
                ),
                results=results,
                failures=failures,
            )
        )

        self._write_json(
            output_dir / "summary.json",
            summary,
        )

        return summary

    def _build_manifest(
        self,
        *,
        experiment_id: str,
        output_dir: Path,
        modes: Sequence[
            ExperimentMode
        ],
        samples: Sequence[
            NeedlePersonaSample
        ],
        expected_run_count: int,
        started_at_utc: str,
        status: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                _RESULT_SCHEMA_VERSION
            ),
            "experiment_id": (
                experiment_id
            ),
            "status": status,
            "started_at_utc": (
                started_at_utc
            ),
            "completed_at_utc": None,
            "dataset_path": str(
                self.config
                .dataset_path
                .resolve()
            ),
            "runtime_root": str(
                self.config
                .runtime_root
                .resolve()
            ),
            "requested_modes": list(
                modes
            ),
            "sample_count": len(
                samples
            ),
            "sample_ids": [
                sample.sample_id
                for sample in samples
            ],
            "expected_run_count": (
                expected_run_count
            ),
            "successful_run_count": 0,
            "failed_run_count": 0,
            "execution_policy": {
                "sequential": True,
                "single_llm_ingestion_per_sample": (
                    True
                ),
                "canonical_private_memory_snapshot": (
                    True
                ),
                "snapshot_cloned_per_mode": (
                    True
                ),
                "fresh_runtime_per_sample_mode": (
                    True
                ),
                "persistent_acl_lifetime": (
                    "isolated_runtime"
                ),
                "gold_evidence_added_after_workflow": (
                    True
                ),
                "raw_persona_sources_in_graph_state": (
                    False
                ),
            },
            "experiment_config": (
                self.config
                .as_runtime_metadata()
            ),
            "runtime_metadata": (
                self.runtime_metadata
            ),
            "output_dir": str(
                output_dir.resolve()
            ),
            "output_files": {
                "manifest": (
                    "manifest.json"
                ),
                "results_jsonl": (
                    "results.jsonl"
                ),
                "results_csv": (
                    "results.csv"
                ),
                "failures_jsonl": (
                    "failures.jsonl"
                ),
                "traces_jsonl": (
                    "traces.jsonl"
                    if self.config
                    .save_detailed_trace
                    else None
                ),
                "summary": (
                    "summary.json"
                ),
            },
        }

    # ------------------------------------------------------------------
    # Sample selection and runtime paths
    # ------------------------------------------------------------------

    def _load_samples(
        self,
        *,
        limit: int | None,
        sample_ids: (
            Sequence[str] | None
        ),
    ) -> list[
        NeedlePersonaSample
    ]:
        requested_ids = (
            self._clean_unique(
                sample_ids or []
            )
        )

        if not requested_ids:
            return list(
                self.loader.iter_samples(
                    limit=limit
                )
            )

        requested_set = set(
            requested_ids
        )
        selected: list[
            NeedlePersonaSample
        ] = []

        for sample in (
            self.loader.iter_samples()
        ):
            if (
                sample.sample_id
                in requested_set
            ):
                selected.append(
                    sample
                )

        found_ids = {
            sample.sample_id
            for sample in selected
        }
        missing_ids = (
            requested_set - found_ids
        )

        if missing_ids:
            raise KeyError(
                "Requested sample IDs were not "
                "found: "
                f"{sorted(missing_ids)}."
            )

        if limit is None:
            return selected

        return selected[:limit]

    def _reset_runtime_root(
        self,
    ) -> None:
        root = (
            self.config.runtime_root
            .resolve()
        )

        self._assert_safe_reset_path(
            root
        )

        if root.exists():
            shutil.rmtree(root)

        root.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _assert_safe_reset_path(
        path: Path,
    ) -> None:
        forbidden = {
            Path(path.anchor).resolve(),
            Path.home().resolve(),
            Path.cwd().resolve(),
        }

        if path in forbidden:
            raise ValueError(
                "Refusing to reset unsafe "
                f"runtime_root: {path}."
            )

        if len(path.parts) < 3:
            raise ValueError(
                "runtime_root path is too shallow "
                f"to reset safely: {path}."
            )

    @staticmethod
    def _prepare_output_dir(
        output_dir: Path,
        *,
        overwrite: bool,
    ) -> None:
        if output_dir.exists():
            has_entries = any(
                output_dir.iterdir()
            )

            if (
                has_entries
                and not overwrite
            ):
                raise FileExistsError(
                    "Experiment output directory "
                    "already exists and is not "
                    f"empty: {output_dir}."
                )

            if (
                has_entries
                and overwrite
            ):
                shutil.rmtree(
                    output_dir
                )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @classmethod
    def _write_json(
        cls,
        path: Path,
        payload: Any,
    ) -> None:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=cls._json_default,
            allow_nan=False,
        )

        cls._atomic_write_text(
            path,
            text + "\n",
        )

    @classmethod
    def _write_jsonl(
        cls,
        path: Path,
        records: Sequence[Any],
    ) -> None:
        lines = [
            json.dumps(
                record,
                ensure_ascii=False,
                default=cls._json_default,
                allow_nan=False,
            )
            for record in records
        ]

        cls._atomic_write_text(
            path,
            (
                "\n".join(lines) + "\n"
                if lines
                else ""
            ),
        )

    @classmethod
    def _write_results_csv(
        cls,
        path: Path,
        results: Sequence[
            NeedlePersonaEvaluationResult
        ],
    ) -> None:
        field_names = list(
            _RESULT_CSV_FIELDS
        )
        extra_fields = sorted(
            {
                key
                for result in results
                for key in result
                if key not in field_names
            }
        )
        field_names.extend(
            extra_fields
        )

        buffer = StringIO(
            newline=""
        )
        writer = csv.DictWriter(
            buffer,
            fieldnames=field_names,
            extrasaction="ignore",
        )
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    field_name: (
                        cls._csv_value(
                            result.get(
                                field_name
                            )
                        )
                    )
                    for field_name
                    in field_names
                }
            )

        cls._atomic_write_text(
            path,
            buffer.getvalue(),
        )

    @classmethod
    def _csv_value(
        cls,
        value: Any,
    ) -> Any:
        if value is None:
            return ""

        if isinstance(
            value,
            bool,
        ):
            return (
                "true"
                if value
                else "false"
            )

        if isinstance(
            value,
            (
                list,
                tuple,
                set,
                dict,
            ),
        ):
            return json.dumps(
                value,
                ensure_ascii=False,
                default=cls._json_default,
                allow_nan=False,
                sort_keys=isinstance(
                    value,
                    dict,
                ),
            )

        if isinstance(
            value,
            Enum,
        ):
            return value.value

        return value

    @staticmethod
    def _atomic_write_text(
        path: Path,
        text: str,
    ) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = (
            path.with_name(
                path.name + ".tmp"
            )
        )
        temporary_path.write_text(
            text,
            encoding="utf-8",
        )
        temporary_path.replace(
            path
        )

    @classmethod
    def _json_default(
        cls,
        value: Any,
    ) -> Any:
        return cls._jsonable(
            value
        )

    @classmethod
    def _jsonable(
        cls,
        value: Any,
    ) -> Any:
        if isinstance(
            value,
            BaseModel,
        ):
            return value.model_dump(
                mode="json"
            )

        if is_dataclass(value):
            return {
                key: cls._jsonable(
                    item
                )
                for key, item in (
                    asdict(value).items()
                )
            }

        if isinstance(
            value,
            Enum,
        ):
            return value.value

        if isinstance(
            value,
            Path,
        ):
            return str(value)

        if isinstance(
            value,
            Mapping,
        ):
            return {
                str(key): cls._jsonable(
                    item
                )
                for key, item in (
                    value.items()
                )
            }

        if isinstance(
            value,
            (
                list,
                tuple,
                set,
            ),
        ):
            return [
                cls._jsonable(item)
                for item in value
            ]

        if isinstance(
            value,
            float,
        ) and not math.isfinite(value):
            return None

        attributes = getattr(
            value,
            "__dict__",
            None,
        )

        if isinstance(
            attributes,
            Mapping,
        ):
            return {
                str(key): cls._jsonable(
                    item
                )
                for key, item in (
                    attributes.items()
                )
                if not str(key).startswith(
                    "_"
                )
            }

        return str(value)

    @classmethod
    def _sanitise_final_state(
        cls,
        state: (
            NeedlePersonaWorkflowState
        ),
    ) -> dict[str, Any]:
        forbidden = {
            "gold_answer",
            "alternative_answers",
            "persona_sources",
            "needle_detail",
        }

        return {
            key: cls._jsonable(
                value
            )
            for key, value in state.items()
            if key not in forbidden
        }

    # ------------------------------------------------------------------
    # General utility
    # ------------------------------------------------------------------

    @classmethod
    def _build_run_id(
        cls,
        *,
        experiment_id: str,
        sample_index: int,
        sample_id: str,
        mode: ExperimentMode,
    ) -> str:
        return (
            f"{experiment_id}__"
            f"{sample_index:04d}__"
            f"{cls._safe_identifier(sample_id)}__"
            f"{mode}"
        )

    @classmethod
    def _build_ingestion_run_id(
        cls,
        *,
        experiment_id: str,
        sample_index: int,
        sample_id: str,
    ) -> str:
        return (
            f"{experiment_id}__"
            f"{sample_index:04d}__"
            f"{cls._safe_identifier(sample_id)}__"
            "canonical_ingestion"
        )

    @staticmethod
    def _new_experiment_id(
    ) -> str:
        timestamp = datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        suffix = uuid4().hex[:8]

        return (
            f"needle_persona_"
            f"{timestamp}_{suffix}"
        )

    @staticmethod
    def _safe_identifier(
        value: str | None,
    ) -> str:
        cleaned = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            str(
                value or ""
            ).strip(),
        ).strip("_")

        if not cleaned:
            raise ValueError(
                "Identifier cannot be empty "
                "after sanitisation."
            )

        return cleaned

    @staticmethod
    def _clean_unique(
        values: Sequence[Any],
    ) -> list[str]:
        result: list[str] = []

        for value in values:
            cleaned = str(
                value
            ).strip()

            if (
                cleaned
                and cleaned not in result
            ):
                result.append(
                    cleaned
                )

        return result

    @staticmethod
    def _utc_now(
    ) -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()

    @staticmethod
    def _elapsed_ms(
        started: float,
    ) -> float:
        return (
            perf_counter() - started
        ) * 1000.0

    @staticmethod
    def _notify(
        callback: ProgressCallback | None,
        event: Mapping[str, Any],
    ) -> None:
        if callback is not None:
            callback(
                dict(event)
            )


def build_needle_persona_summary(
    *,
    experiment_id: str,
    status: str,
    started_at_utc: str,
    completed_at_utc: str | None,
    requested_modes: Sequence[
        ExperimentMode
    ],
    sample_count: int,
    expected_run_count: int,
    results: Sequence[
        NeedlePersonaEvaluationResult
    ],
    failures: Sequence[
        NeedlePersonaRunFailure
    ],
) -> dict[str, Any]:
    """
    Aggregate result metrics overall and by experiment mode.
    """
    results_by_mode = {
        mode: [
            result
            for result in results
            if result[
                "experiment_mode"
            ]
            == mode
        ]
        for mode in requested_modes
    }

    return {
        "schema_version": (
            _RESULT_SCHEMA_VERSION
        ),
        "experiment_id": (
            experiment_id
        ),
        "status": status,
        "started_at_utc": (
            started_at_utc
        ),
        "completed_at_utc": (
            completed_at_utc
        ),
        "requested_modes": list(
            requested_modes
        ),
        "sample_count": (
            sample_count
        ),
        "expected_run_count": (
            expected_run_count
        ),
        "successful_run_count": (
            len(results)
        ),
        "failed_run_count": (
            len(failures)
        ),
        "overall": (
            _aggregate_results(
                results
            )
        ),
        "by_mode": {
            mode: _aggregate_results(
                mode_results
            )
            for mode, mode_results
            in results_by_mode.items()
        },
        "answer_status_counts": (
            _count_values(
                results,
                "answer_status",
            )
        ),
        "failure_stage_counts": (
            _count_failure_stages(
                failures
            )
        ),
    }


def _aggregate_results(
    results: Sequence[
        NeedlePersonaEvaluationResult
    ],
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "run_count": len(results),
    }

    for field_name in (
        _AGGREGATE_METRIC_FIELDS
    ):
        values: list[float] = []

        for result in results:
            value = result.get(
                field_name
            )

            if value is None:
                continue

            if isinstance(
                value,
                bool,
            ):
                values.append(
                    float(value)
                )
            elif isinstance(
                value,
                (int, float),
            ):
                number = float(value)

                if math.isfinite(number):
                    values.append(
                        number
                    )

        aggregate[
            f"mean_{field_name}"
        ] = (
            statistics.fmean(values)
            if values
            else None
        )

        aggregate[
            f"count_{field_name}"
        ] = len(values)

    return aggregate


def _count_values(
    results: Sequence[
        NeedlePersonaEvaluationResult
    ],
    field_name: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}

    for result in results:
        value = str(
            result.get(
                field_name,
                "",
            )
        ).strip()

        if value:
            counts[value] = (
                counts.get(
                    value,
                    0,
                )
                + 1
            )

    return counts


def _count_failure_stages(
    failures: Sequence[
        NeedlePersonaRunFailure
    ],
) -> dict[str, int]:
    counts: dict[str, int] = {}

    for failure in failures:
        counts[
            failure.stage
        ] = (
            counts.get(
                failure.stage,
                0,
            )
            + 1
        )

    return counts


__all__ = [
    "NeedlePersonaLoaderProtocol",
    "NeedlePersonaIngestionProtocol",
    "NeedlePersonaIngestionSnapshot",
    "NeedlePersonaRuntimeFactoryProtocol",
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