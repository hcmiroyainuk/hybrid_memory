from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from experiments.evaluator.informativebench.config.needle_persona_config import (
    DEFAULT_EXPERIMENT_MODES,
    ExperimentMode,
    NeedlePersonaExperimentConfig,
    normalise_experiment_modes,
)
from experiments.evaluator.informativebench.evaluator.needle_persona_runtime import (
    NeedlePersonaLLMConfig,
    build_needle_persona_runner,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run InformativeBench Needle in the Persona "
            "with isolated sample × mode runtimes."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/informativebench_data/dataset_2hop.jsonl"),
        help=(
            "Path to the Needle Persona JSONL dataset. "
            "Relative paths are resolved from the project root."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "outputs/informativebench/needle_persona"
        ),
        help=(
            "Root directory for experiment outputs."
        ),
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path(
            "data/needle_persona_runtime"
        ),
        help=(
            "Root directory for isolated MemoryStore "
            "and PromotionRequestStore files."
        ),
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help=(
            "Optional stable experiment ID. When omitted, "
            "a timestamped ID is generated."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help=(
            "Maximum number of dataset samples. "
            "Use 1 for the first smoke test."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Run the complete dataset. This overrides --limit."
        ),
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        default=None,
        help=(
            "Run only the specified sample ID. "
            "Repeat this option to select multiple IDs."
        ),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(DEFAULT_EXPERIMENT_MODES),
        choices=list(DEFAULT_EXPERIMENT_MODES),
        help=(
            "Experiment modes to execute."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM sampling temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2000,
        help=(
            "Maximum structured-output completion tokens. "
            "2000 is safer than 800 for ingestion output."
        ),
    )
    parser.add_argument(
        "--env-path",
        type=Path,
        default=Path(".env"),
        help="Path to the environment file containing the API key.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing non-empty output directory "
            "with the same experiment ID."
        ),
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help=(
            "Stop the batch after the first failed sample-mode run."
        ),
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Do not write traces.jsonl.",
    )

    return parser.parse_args()


def resolve_project_path(
    value: Path,
) -> Path:
    if value.is_absolute():
        return value

    return (
        PROJECT_ROOT / value
    ).resolve()


def progress(event: dict[str, Any]) -> None:
    event_name = event.get(
        "event",
        "event",
    )
    run_id = event.get(
        "run_id",
        "",
    )
    completed = event.get(
        "completed_run_count",
        0,
    )
    expected = event.get(
        "expected_run_count",
        "?",
    )

    print(
        f"[{event_name}] "
        f"{completed}/{expected} "
        f"{run_id}",
        flush=True,
    )

    failure = event.get("failure")

    if failure:
        print(
            "  "
            f"{failure.get('stage')}: "
            f"{failure.get('error_type')}: "
            f"{failure.get('error_message')}",
            flush=True,
        )


def main() -> int:
    args = parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError(
            "--limit must be greater than zero."
        )

    selected_limit = (
        None
        if args.all
        else args.limit
    )

    modes: tuple[
        ExperimentMode,
        ...,
    ] = normalise_experiment_modes(
        args.modes
    )

    dataset_path = resolve_project_path(
        args.dataset
    )
    output_root = resolve_project_path(
        args.output_root
    )
    runtime_root = resolve_project_path(
        args.runtime_root
    )
    env_path = resolve_project_path(
        args.env_path
    )

    if not dataset_path.is_file():
        raise FileNotFoundError(
            "Dataset file does not exist: "
            f"{dataset_path}"
        )

    if not env_path.is_file():
        print(
            "Warning: .env file was not found at "
            f"{env_path}. The API key must already "
            "exist in the environment.",
            file=sys.stderr,
        )

    experiment_config = (
        NeedlePersonaExperimentConfig(
            dataset_path=dataset_path,
            output_root=output_root,
            runtime_root=runtime_root,
            modes=modes,
            sample_limit=selected_limit,
            overwrite_output=(
                args.overwrite
            ),
            save_detailed_trace=(
                not args.no_trace
            ),
            # The runtime factory creates a fresh directory for
            # every sample × mode. Resetting the root before this
            # batch additionally removes stale smoke-test state.
            reset_runtime_root_before_batch=True,
        )
    )

    llm_config = NeedlePersonaLLMConfig(
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        env_path=env_path,
    )

    runner = build_needle_persona_runner(
        experiment_config=(
            experiment_config
        ),
        llm_config=llm_config,
        extra_metadata={
            "entrypoint": (
                "run_needle_persona.py"
            ),
        },
    )

    print("Needle Persona experiment")
    print(f"  dataset: {dataset_path}")
    print(f"  modes: {', '.join(modes)}")
    print(
        "  sample limit: "
        + (
            "all"
            if selected_limit is None
            else str(selected_limit)
        )
    )
    print(f"  model: {args.model}")
    print(f"  max tokens: {args.max_tokens}")
    print(f"  output root: {output_root}")
    print(f"  runtime root: {runtime_root}")
    print()

    report = runner.run_batch(
        experiment_id=(
            args.experiment_id
        ),
        modes=modes,
        limit=selected_limit,
        sample_ids=args.sample_id,
        continue_on_error=(
            not args.stop_on_error
        ),
        save_after_each_run=True,
        overwrite=args.overwrite,
        progress_callback=progress,
    )

    print()
    print("Experiment completed.")
    print(
        f"  successful runs: "
        f"{len(report.results)}"
    )
    print(
        f"  failed runs: "
        f"{len(report.failures)}"
    )
    print(
        f"  output directory: "
        f"{report.output_dir}"
    )
    print(
        f"  summary: "
        f"{report.summary_path}"
    )
    print(
        f"  results: "
        f"{report.results_jsonl_path}"
    )
    print(
        f"  failures: "
        f"{report.failures_jsonl_path}"
    )

    return (
        0
        if not report.failures
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())