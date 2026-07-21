from __future__ import annotations

import csv
import json
import math
import re
import statistics
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from io import StringIO
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from .needle_persona_evaluator import NeedlePersonaEvaluator
from .needle_persona_evidence import build_needle_persona_evidence
from .needle_persona_ingestion import NeedlePersonaIngestionService
from .needle_persona_loader import NeedlePersonaLoader
from .needle_persona_models import NeedlePersonaSample
from .needle_persona_state import (
    ExperimentMode,
    NeedlePersonaEvaluationResult,
    build_initial_needle_persona_state,
    build_needle_persona_evaluation_input,
)
from .needle_persona_workflow import NeedlePersonaWorkflow


DEFAULT_EXPERIMENT_MODES: tuple[ExperimentMode, ...] = (
    "private_only",
    "ungoverned_shared",
    "governed_shared",
)

_RESULT_SCHEMA_VERSION = "1.0"

_AGGREGATE_METRIC_FIELDS: tuple[str, ...] = (
    # Answer quality
    "answer_accuracy",
    "entity_precision",
    "entity_recall",
    "entity_f1",
    # Retrieval
    "target_source_retrieval_recall",
    "all_target_sources_retrieved",
    "cross_agent_source_retrieval_recall",
    "cross_agent_memory_hit",
    # Utilisation
    "target_source_utilisation_recall",
    "cross_agent_source_utilisation_recall",
    "cross_agent_source_utilised",
    # Sharing and governance
    "shared_cross_agent_source_recall",
    "governance_precision",
    "governance_recall",
    "over_sharing_rate",
    "under_sharing_rate",
    # Governance trace counts
    "candidate_memory_count",
    "critic_approved_memory_count",
    "coordinator_approved_memory_count",
    "shared_memory_count",
    "rejected_memory_count",
    # End-to-end
    "transfer_success",
)

_RESULT_CSV_FIELDS: tuple[str, ...] = (
    "run_id",
    "sample_id",
    "experiment_mode",
    "question",
    "predicted_answer",
    "matched_reference_answer",
    "predicted_entities",
    "matched_reference_entities",
    "answer_accuracy",
    "entity_precision",
    "entity_recall",
    "entity_f1",
    "expected_source_ids",
    "expected_local_source_ids",
    "expected_cross_agent_source_ids",
    "retrieved_source_ids",
    "retrieved_target_source_ids",
    "missing_retrieved_target_source_ids",
    "target_source_retrieval_recall",
    "all_target_sources_retrieved",
    "cross_agent_retrieved_source_ids",
    "cross_agent_source_retrieval_recall",
    "cross_agent_memory_hit",
    "used_source_ids",
    "used_target_source_ids",
    "target_source_utilisation_recall",
    "cross_agent_used_source_ids",
    "cross_agent_source_utilisation_recall",
    "cross_agent_source_utilised",
    "shared_source_ids",
    "correctly_shared_source_ids",
    "over_shared_source_ids",
    "missing_shared_cross_agent_source_ids",
    "shared_cross_agent_source_recall",
    "governance_precision",
    "governance_recall",
    "over_sharing_rate",
    "under_sharing_rate",
    "candidate_memory_count",
    "critic_approved_memory_count",
    "coordinator_approved_memory_count",
    "shared_memory_count",
    "rejected_memory_count",
    "transfer_success",
)


class NeedlePersonaBatchRunError(RuntimeError):
    """
    Raised when a batch run is configured to stop after the first failed run.
    """

    def __init__(
        self,
        message: str,
        *,
        report: "NeedlePersonaBatchReport | None" = None,
    ) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True, slots=True)
class NeedlePersonaEvaluationRuntime:
    """
    Application services required by the final experiment entry point.

    A runtime factory may construct these services using any project-specific
    persistence, retrieval, and LLM configuration. The benchmark runner remains
    independent of those bootstrap details.
    """

    ingestion_service: NeedlePersonaIngestionService
    workflow: NeedlePersonaWorkflow
    evaluator: NeedlePersonaEvaluator = field(
        default_factory=NeedlePersonaEvaluator
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NeedlePersonaBatchReport:
    """
    Result returned after the batch has finished or stopped.
    """

    experiment_id: str
    output_dir: Path
    results: tuple[NeedlePersonaEvaluationResult, ...]
    failures: tuple[NeedlePersonaRunFailure, ...]
    summary: Mapping[str, Any]

    @property
    def results_jsonl_path(self) -> Path:
        return self.output_dir / "results.jsonl"

    @property
    def results_csv_path(self) -> Path:
        return self.output_dir / "results.csv"

    @property
    def failures_jsonl_path(self) -> Path:
        return self.output_dir / "failures.jsonl"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "summary.json"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


ProgressCallback = Callable[[Mapping[str, Any]], None]
RunRuntimeFactory = Callable[
    [str, NeedlePersonaSample, ExperimentMode],
    NeedlePersonaEvaluationRuntime,
]


class NeedlePersonaEvaluationRunner:
    """
    Sequential batch runner for the three Needle in the Persona conditions.

    Every sample-mode pair receives a fresh ingestion run. This is required
    because ungoverned and governed sharing mutate memory scope. The runner is
    intentionally sequential because the project's JSON-backed stores are not
    designed for concurrent writes.
    """

    def __init__(
        self,
        *,
        loader: NeedlePersonaLoader,
        runtime_factory: RunRuntimeFactory,
        runtime_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if loader.source_granularity != "persona_document":
            raise ValueError(
                "The final experiment requires "
                "source_granularity='persona_document'."
            )

        if loader.include_collaborative_chat:
            raise ValueError(
                "The final experiment must exclude collaborative chat."
            )

        if loader.retain_raw_fields_in_metadata:
            raise ValueError(
                "The final experiment must not retain raw benchmark fields "
                "in sample metadata."
            )

        if not callable(runtime_factory):
            raise TypeError("runtime_factory must be callable.")

        self.loader = loader
        self.runtime_factory = runtime_factory
        self.runtime_metadata = dict(runtime_metadata or {})

    def run_batch(
        self,
        *,
        output_root: str | Path,
        modes: Sequence[ExperimentMode] = DEFAULT_EXPERIMENT_MODES,
        limit: int | None = None,
        sample_ids: Sequence[str] | None = None,
        experiment_id: str | None = None,
        continue_on_error: bool = True,
        save_after_each_run: bool = True,
        overwrite: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> NeedlePersonaBatchReport:
        """
        Run all selected samples under every selected experiment mode.

        Output layout::

            <output_root>/<experiment_id>/
                manifest.json
                results.jsonl
                results.csv
                failures.jsonl
                summary.json
        """
        selected_modes = self._validate_modes(modes)

        if limit is not None and limit < 0:
            raise ValueError("limit cannot be negative.")

        clean_experiment_id = (
            self._safe_identifier(experiment_id)
            if experiment_id
            else self._new_experiment_id()
        )

        output_dir = Path(output_root) / clean_experiment_id
        self._prepare_output_dir(
            output_dir,
            overwrite=overwrite,
        )

        selected_samples = self._load_samples(
            limit=limit,
            sample_ids=sample_ids,
        )

        if not selected_samples:
            raise ValueError(
                "No benchmark samples were selected."
            )

        expected_run_count = (
            len(selected_samples) * len(selected_modes)
        )
        started_at = self._utc_now()
        results: list[NeedlePersonaEvaluationResult] = []
        failures: list[NeedlePersonaRunFailure] = []

        manifest = self._build_manifest(
            experiment_id=clean_experiment_id,
            output_dir=output_dir,
            modes=selected_modes,
            samples=selected_samples,
            expected_run_count=expected_run_count,
            started_at_utc=started_at,
            status="running",
        )
        self._write_json(
            output_dir / "manifest.json",
            manifest,
        )
        self._persist_outputs(
            output_dir=output_dir,
            experiment_id=clean_experiment_id,
            modes=selected_modes,
            samples=selected_samples,
            expected_run_count=expected_run_count,
            started_at_utc=started_at,
            completed_at_utc=None,
            status="running",
            results=results,
            failures=failures,
        )

        completed_run_count = 0

        try:
            for sample_index, sample in enumerate(
                selected_samples
            ):
                for mode in selected_modes:
                    run_id = self._build_run_id(
                        experiment_id=clean_experiment_id,
                        sample_index=sample_index,
                        sample_id=sample.sample_id,
                        mode=mode,
                    )

                    self._notify(
                        progress_callback,
                        {
                            "event": "run_started",
                            "run_id": run_id,
                            "sample_id": sample.sample_id,
                            "experiment_mode": mode,
                            "completed_run_count": completed_run_count,
                            "expected_run_count": expected_run_count,
                        },
                    )

                    result, failure = self._run_one(
                        sample=sample,
                        mode=mode,
                        run_id=run_id,
                    )

                    if result is not None:
                        results.append(result)

                    if failure is not None:
                        failures.append(failure)

                    completed_run_count += 1

                    if save_after_each_run:
                        self._persist_outputs(
                            output_dir=output_dir,
                            experiment_id=clean_experiment_id,
                            modes=selected_modes,
                            samples=selected_samples,
                            expected_run_count=expected_run_count,
                            started_at_utc=started_at,
                            completed_at_utc=None,
                            status="running",
                            results=results,
                            failures=failures,
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
                            "sample_id": sample.sample_id,
                            "experiment_mode": mode,
                            "completed_run_count": completed_run_count,
                            "expected_run_count": expected_run_count,
                            "failure": (
                                failure.as_dict()
                                if failure is not None
                                else None
                            ),
                        },
                    )

                    if (
                        failure is not None
                        and not continue_on_error
                    ):
                        completed_at = self._utc_now()
                        report = self._finalise(
                            output_dir=output_dir,
                            experiment_id=clean_experiment_id,
                            modes=selected_modes,
                            samples=selected_samples,
                            expected_run_count=expected_run_count,
                            started_at_utc=started_at,
                            completed_at_utc=completed_at,
                            status="stopped_after_failure",
                            results=results,
                            failures=failures,
                            manifest=manifest,
                        )
                        raise NeedlePersonaBatchRunError(
                            "Batch stopped after a failed run. "
                            f"See {report.failures_jsonl_path}.",
                            report=report,
                        )

        except KeyboardInterrupt:
            completed_at = self._utc_now()
            self._finalise(
                output_dir=output_dir,
                experiment_id=clean_experiment_id,
                modes=selected_modes,
                samples=selected_samples,
                expected_run_count=expected_run_count,
                started_at_utc=started_at,
                completed_at_utc=completed_at,
                status="interrupted",
                results=results,
                failures=failures,
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
            experiment_id=clean_experiment_id,
            modes=selected_modes,
            samples=selected_samples,
            expected_run_count=expected_run_count,
            started_at_utc=started_at,
            completed_at_utc=completed_at,
            status=final_status,
            results=results,
            failures=failures,
            manifest=manifest,
        )

    def _run_one(
        self,
        *,
        sample: NeedlePersonaSample,
        mode: ExperimentMode,
        run_id: str,
    ) -> tuple[
        NeedlePersonaEvaluationResult | None,
        NeedlePersonaRunFailure | None,
    ]:
        started_at = self._utc_now()
        started_perf = perf_counter()
        stage = "runtime_initialisation"

        try:
            # Build an isolated service graph for this exact sample-mode run.
            # The factory must use run-specific persistence paths so no two
            # runs read or write the same memories.json file.
            runtime = self.runtime_factory(
                run_id,
                sample,
                mode,
            )

            if not isinstance(
                runtime,
                NeedlePersonaEvaluationRuntime,
            ):
                raise TypeError(
                    "runtime_factory must return "
                    "NeedlePersonaEvaluationRuntime."
                )

            stage = "ingestion"
            memory_index = runtime.ingestion_service.ingest(
                sample=sample,
                run_id=run_id,
            )

            stage = "initial_state"
            initial_state = (
                build_initial_needle_persona_state(
                    sample=sample,
                    memory_index=memory_index,
                    experiment_mode=mode,
                )
            )

            stage = "workflow"
            final_state = runtime.workflow.invoke(
                initial_state
            )

            stage = "evidence"
            evidence = build_needle_persona_evidence(
                sample=sample,
                memory_index=memory_index,
                responder_agent_id=final_state[
                    "responder_agent_id"
                ],
            )

            stage = "evaluation_input"
            evaluation_input = (
                build_needle_persona_evaluation_input(
                    sample=sample,
                    final_state=final_state,
                    evidence=evidence,
                )
            )

            stage = "evaluation"
            result = runtime.evaluator.evaluate(
                evaluation_input
            )

            return result, None

        except Exception as error:
            completed_at = self._utc_now()
            duration_ms = (
                perf_counter() - started_perf
            ) * 1000.0

            failure = NeedlePersonaRunFailure(
                run_id=run_id,
                sample_id=sample.sample_id,
                experiment_mode=mode,
                stage=stage,
                error_type=type(error).__name__,
                error_message=str(error),
                traceback_text=traceback.format_exc(),
                started_at_utc=started_at,
                completed_at_utc=completed_at,
                duration_ms=round(duration_ms, 3),
            )

            return None, failure

    def _finalise(
        self,
        *,
        output_dir: Path,
        experiment_id: str,
        modes: Sequence[ExperimentMode],
        samples: Sequence[NeedlePersonaSample],
        expected_run_count: int,
        started_at_utc: str,
        completed_at_utc: str,
        status: str,
        results: Sequence[NeedlePersonaEvaluationResult],
        failures: Sequence[NeedlePersonaRunFailure],
        manifest: Mapping[str, Any],
    ) -> NeedlePersonaBatchReport:
        summary = self._persist_outputs(
            output_dir=output_dir,
            experiment_id=experiment_id,
            modes=modes,
            samples=samples,
            expected_run_count=expected_run_count,
            started_at_utc=started_at_utc,
            completed_at_utc=completed_at_utc,
            status=status,
            results=results,
            failures=failures,
        )

        final_manifest = dict(manifest)
        final_manifest.update(
            {
                "status": status,
                "completed_at_utc": completed_at_utc,
                "successful_run_count": len(results),
                "failed_run_count": len(failures),
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
            summary=summary,
        )

    def _persist_outputs(
        self,
        *,
        output_dir: Path,
        experiment_id: str,
        modes: Sequence[ExperimentMode],
        samples: Sequence[NeedlePersonaSample],
        expected_run_count: int,
        started_at_utc: str,
        completed_at_utc: str | None,
        status: str,
        results: Sequence[NeedlePersonaEvaluationResult],
        failures: Sequence[NeedlePersonaRunFailure],
    ) -> dict[str, Any]:
        self._write_jsonl(
            output_dir / "results.jsonl",
            results,
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

        summary = build_needle_persona_summary(
            experiment_id=experiment_id,
            status=status,
            started_at_utc=started_at_utc,
            completed_at_utc=completed_at_utc,
            requested_modes=modes,
            sample_count=len(samples),
            expected_run_count=expected_run_count,
            results=results,
            failures=failures,
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
        modes: Sequence[ExperimentMode],
        samples: Sequence[NeedlePersonaSample],
        expected_run_count: int,
        started_at_utc: str,
        status: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": _RESULT_SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "status": status,
            "started_at_utc": started_at_utc,
            "completed_at_utc": None,
            "dataset_path": str(
                self.loader.dataset_path.resolve()
            ),
            "source_granularity": (
                self.loader.source_granularity
            ),
            "include_collaborative_chat": (
                self.loader.include_collaborative_chat
            ),
            "retain_raw_fields_in_metadata": (
                self.loader.retain_raw_fields_in_metadata
            ),
            "loader_strict": self.loader.strict,
            "requested_modes": list(modes),
            "sample_count": len(samples),
            "sample_ids": [
                sample.sample_id
                for sample in samples
            ],
            "expected_run_count": expected_run_count,
            "successful_run_count": 0,
            "failed_run_count": 0,
            "execution_policy": {
                "sequential": True,
                "fresh_ingestion_per_sample_mode": True,
                "gold_evidence_added_after_workflow": True,
            },
            "runtime_metadata": self.runtime_metadata,
            "output_dir": str(output_dir.resolve()),
            "output_files": {
                "manifest": "manifest.json",
                "detailed_jsonl": "results.jsonl",
                "detailed_csv": "results.csv",
                "failures": "failures.jsonl",
                "summary": "summary.json",
            },
        }

    def _load_samples(
        self,
        *,
        limit: int | None,
        sample_ids: Sequence[str] | None,
    ) -> list[NeedlePersonaSample]:
        requested_ids = self._clean_unique(
            sample_ids or []
        )

        if not requested_ids:
            return list(
                self.loader.iter_samples(limit=limit)
            )

        requested_set = set(requested_ids)
        matched: list[NeedlePersonaSample] = []

        for sample in self.loader.iter_samples():
            if sample.sample_id in requested_set:
                matched.append(sample)

        found_ids = {
            sample.sample_id
            for sample in matched
        }
        missing_ids = requested_set - found_ids

        if missing_ids:
            raise KeyError(
                "Requested sample IDs were not found: "
                f"{sorted(missing_ids)}."
            )

        if limit is None:
            return matched

        return matched[:limit]

    @staticmethod
    def _validate_modes(
        modes: Sequence[ExperimentMode],
    ) -> tuple[ExperimentMode, ...]:
        valid_modes = set(DEFAULT_EXPERIMENT_MODES)
        cleaned: list[ExperimentMode] = []

        for mode in modes:
            if mode not in valid_modes:
                raise ValueError(
                    f"Invalid experiment mode: {mode!r}."
                )

            if mode not in cleaned:
                cleaned.append(mode)

        if not cleaned:
            raise ValueError(
                "At least one experiment mode is required."
            )

        return tuple(cleaned)

    @staticmethod
    def _prepare_output_dir(
        output_dir: Path,
        *,
        overwrite: bool,
    ) -> None:
        if output_dir.exists():
            has_entries = any(output_dir.iterdir())

            if has_entries and not overwrite:
                raise FileExistsError(
                    "Experiment output directory already exists "
                    f"and is not empty: {output_dir}."
                )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

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
    def _new_experiment_id(cls) -> str:
        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")
        suffix = uuid4().hex[:8]

        return (
            f"needle_persona_{timestamp}_{suffix}"
        )

    @staticmethod
    def _safe_identifier(
        value: str | None,
    ) -> str:
        cleaned = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            str(value or "").strip(),
        ).strip("_")

        if not cleaned:
            raise ValueError(
                "Identifier cannot be empty after sanitisation."
            )

        return cleaned

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
    def _utc_now() -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()

    @staticmethod
    def _notify(
        callback: ProgressCallback | None,
        event: Mapping[str, Any],
    ) -> None:
        if callback is not None:
            callback(dict(event))

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
        text = (
            "\n".join(lines) + "\n"
            if lines
            else ""
        )
        cls._atomic_write_text(path, text)

    @classmethod
    def _write_results_csv(
        cls,
        path: Path,
        results: Sequence[
            NeedlePersonaEvaluationResult
        ],
    ) -> None:
        field_names = list(_RESULT_CSV_FIELDS)

        extra_fields = sorted(
            {
                key
                for result in results
                for key in result
                if key not in field_names
            }
        )
        field_names.extend(extra_fields)

        buffer = StringIO(newline="")
        writer = csv.DictWriter(
            buffer,
            fieldnames=field_names,
            extrasaction="ignore",
        )
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    field_name: cls._csv_value(
                        result.get(field_name)
                    )
                    for field_name in field_names
                }
            )

        cls._atomic_write_text(
            path,
            buffer.getvalue(),
        )

    @classmethod
    def _csv_value(cls, value: Any) -> Any:
        if value is None:
            return ""

        if isinstance(value, bool):
            return "true" if value else "false"

        if isinstance(
            value,
            (list, tuple, set, dict),
        ):
            return json.dumps(
                value,
                ensure_ascii=False,
                default=cls._json_default,
                allow_nan=False,
                sort_keys=isinstance(value, dict),
            )

        if isinstance(value, Enum):
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
        temporary_path = path.with_name(
            f".{path.name}.{uuid4().hex}.tmp"
        )
        temporary_path.write_text(
            text,
            encoding="utf-8",
        )
        temporary_path.replace(path)

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")

        if isinstance(value, Enum):
            return value.value

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, datetime):
            return value.isoformat()

        if is_dataclass(value):
            return asdict(value)

        if isinstance(value, set):
            return sorted(value)

        raise TypeError(
            f"Object of type {type(value).__name__} "
            "is not JSON serialisable."
        )


def build_needle_persona_summary(
    *,
    experiment_id: str,
    status: str,
    started_at_utc: str,
    completed_at_utc: str | None,
    requested_modes: Sequence[ExperimentMode],
    sample_count: int,
    expected_run_count: int,
    results: Sequence[NeedlePersonaEvaluationResult],
    failures: Sequence[
        NeedlePersonaRunFailure | Mapping[str, Any]
    ],
) -> dict[str, Any]:
    """
    Build aggregate metrics for the whole batch and for each experiment mode.

    Metrics whose value is ``None`` are excluded from that metric's denominator.
    Every statistic therefore reports ``valid_count`` and ``missing_count``.
    """
    failure_records = [
        (
            failure.as_dict()
            if isinstance(
                failure,
                NeedlePersonaRunFailure,
            )
            else dict(failure)
        )
        for failure in failures
    ]

    results_by_mode: dict[
        ExperimentMode,
        list[NeedlePersonaEvaluationResult],
    ] = {
        mode: []
        for mode in requested_modes
    }
    failures_by_mode: dict[
        ExperimentMode,
        list[dict[str, Any]],
    ] = {
        mode: []
        for mode in requested_modes
    }

    for result in results:
        mode = result["experiment_mode"]
        results_by_mode.setdefault(
            mode,
            [],
        ).append(result)

    for failure in failure_records:
        mode = failure.get(
            "experiment_mode"
        )

        if mode in results_by_mode:
            failures_by_mode.setdefault(
                mode,
                [],
            ).append(failure)

    by_mode: dict[str, Any] = {}

    for mode in requested_modes:
        mode_results = results_by_mode.get(
            mode,
            [],
        )
        mode_failures = failures_by_mode.get(
            mode,
            [],
        )

        by_mode[mode] = {
            "expected_run_count": sample_count,
            "successful_run_count": len(
                mode_results
            ),
            "failed_run_count": len(
                mode_failures
            ),
            "metrics": _aggregate_metrics(
                mode_results
            ),
        }

    overall_metrics = _aggregate_metrics(
        results
    )

    failure_stage_counts: dict[str, int] = {}

    for failure in failure_records:
        stage = str(
            failure.get("stage", "unknown")
        )
        failure_stage_counts[stage] = (
            failure_stage_counts.get(stage, 0)
            + 1
        )

    return {
        "schema_version": _RESULT_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "status": status,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "requested_modes": list(
            requested_modes
        ),
        "sample_count": sample_count,
        "expected_run_count": expected_run_count,
        "successful_run_count": len(results),
        "failed_run_count": len(
            failure_records
        ),
        "execution_completion_rate": (
            (
                len(results)
                + len(failure_records)
            )
            / expected_run_count
            if expected_run_count
            else 0.0
        ),
        "success_rate": (
            len(results) / expected_run_count
            if expected_run_count
            else 0.0
        ),
        "failure_rate": (
            len(failure_records)
            / expected_run_count
            if expected_run_count
            else 0.0
        ),
        "overall_metrics": overall_metrics,
        "by_mode": by_mode,
        "failure_stage_counts": (
            failure_stage_counts
        ),
        "aggregation_policy": {
            "numeric_and_boolean_metrics": (
                "macro mean over successful runs"
            ),
            "none_values": (
                "excluded from the metric denominator"
            ),
            "boolean_values": (
                "converted to 1.0/0.0; mean is a rate"
            ),
            "primary_comparison_unit": (
                "by_mode metrics"
            ),
        },
    }


def _aggregate_metrics(
    results: Sequence[
        NeedlePersonaEvaluationResult
    ],
) -> dict[str, Any]:
    aggregated: dict[str, Any] = {}

    for field_name in _AGGREGATE_METRIC_FIELDS:
        values: list[float] = []

        for result in results:
            raw_value = result.get(field_name)

            if raw_value is None:
                continue

            if isinstance(raw_value, bool):
                values.append(
                    1.0 if raw_value else 0.0
                )
                continue

            if isinstance(raw_value, (int, float)):
                numeric_value = float(raw_value)

                if not math.isfinite(
                    numeric_value
                ):
                    raise ValueError(
                        f"Metric {field_name!r} contains "
                        f"a non-finite value: {raw_value!r}."
                    )

                values.append(numeric_value)

        total_count = len(results)
        valid_count = len(values)
        missing_count = (
            total_count - valid_count
        )

        if not values:
            aggregated[field_name] = {
                "mean": None,
                "stddev": None,
                "min": None,
                "max": None,
                "sum": None,
                "valid_count": 0,
                "missing_count": missing_count,
                "total_count": total_count,
            }
            continue

        aggregated[field_name] = {
            "mean": statistics.fmean(values),
            "stddev": (
                statistics.pstdev(values)
                if len(values) > 1
                else 0.0
            ),
            "min": min(values),
            "max": max(values),
            "sum": sum(values),
            "valid_count": valid_count,
            "missing_count": missing_count,
            "total_count": total_count,
        }

    return aggregated


__all__ = [
    "DEFAULT_EXPERIMENT_MODES",
    "NeedlePersonaBatchRunError",
    "NeedlePersonaEvaluationRuntime",
    "NeedlePersonaRunFailure",
    "NeedlePersonaBatchReport",
    "ProgressCallback",
    "NeedlePersonaEvaluationRunner",
    "build_needle_persona_summary",
]