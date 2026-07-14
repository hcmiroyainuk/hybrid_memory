from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from experiments.evaluator.squad_evaluator import SquadEvaluator
from src.llm import LLMClient
from src.memory.external_knowledge import ExternalKnowledgeRetriever
from src.memory.external_knowledge.squad_loader import SquadLoader
from src.workflow.rag_workflow import (
    RAGQAWorkflow,
    create_initial_rag_qa_state_from_sample,
)


def find_project_root(start_path: Path) -> Path:
    """
    Find the project root by searching upward for a .env file.
    """

    current = start_path.resolve()

    if current.is_file():
        current = current.parent

    for parent in [current, *current.parents]:
        if (parent / ".env").exists():
            return parent

    raise FileNotFoundError(
        f"Could not find .env from {start_path}"
    )


PROJECT_ROOT = find_project_root(Path(__file__))
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE, override=True)

print(f"Loaded .env from: {ENV_FILE}")


SQUAD_FILE = PROJECT_ROOT / "data" / "train-v2.0.json"

CHROMA_DIR = (
    PROJECT_ROOT
    / "data"
    / "test_external_knowledge"
    / "squad_chroma"
)

COLLECTION_NAME = "test_squad_external_knowledge"

RESULT_DIR = PROJECT_ROOT / "experiments" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def _safe_model_dump(value: Any) -> Any:
    """
    Convert Pydantic models and nested objects into JSON-safe values.
    """

    if value is None:
        return None

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")

    if isinstance(value, list):
        return [
            _safe_model_dump(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            _safe_model_dump(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            key: _safe_model_dump(item)
            for key, item in value.items()
        }

    return value


def _get_value_or_call(value: Any) -> Any:
    """
    Return a value directly, or call it first when it is callable.
    """

    if callable(value):
        return value()

    return value


def _evaluate_one_output(
    prediction: str | None,
    gold_answers: list[str],
) -> dict[str, Any]:
    """
    Evaluate one prediction and return a JSON-safe result.
    """

    result = SquadEvaluator.evaluate_prediction(
        prediction=prediction or "",
        gold_answers=gold_answers,
    )

    return {
        "prediction": result.prediction,
        "gold_answers": result.gold_answers,
        "exact_match": result.exact_match,
        "f1": result.f1,
    }


def _normalize_hash_list(value: Any) -> list[str]:
    """
    Convert retrieved context hashes into a clean list of strings.

    Supported input formats:
    - None
    - str
    - list
    - tuple
    - set
    - callable returning one of the formats above
    """

    value = _get_value_or_call(value)

    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, (list, tuple, set)):
        return [
            str(item)
            for item in value
            if item is not None
        ]

    return [str(value)]


def calculate_retrieval_hit(
    gold_context_hash: str | None,
    retrieved_context_hashes: Any,
) -> bool | None:
    """
    Determine whether retrieval found the gold SQuAD context.

    Returns:
        True:
            The gold context hash appears in retrieved context hashes.

        False:
            The gold context hash is available, but it was not retrieved.

        None:
            The gold context hash is missing, so retrieval cannot be evaluated.
    """

    if not gold_context_hash:
        return None

    normalized_retrieved_hashes = _normalize_hash_list(
        retrieved_context_hashes
    )

    return (
        str(gold_context_hash)
        in normalized_retrieved_hashes
    )


def average_worker_b_metric_by_hit(
    records: list[dict[str, Any]],
    *,
    metric_name: str,
    expected_hit: bool,
) -> float | None:
    """
    Calculate a Worker B metric conditioned on retrieval hit or miss.
    """

    values = [
        float(
            record["worker_b"]["evaluation"][metric_name]
        )
        for record in records
        if record["retrieval"]["retrieval_hit"]
        is expected_hit
    ]

    if not values:
        return None

    return sum(values) / len(values)


def run_evaluation(
    *,
    limit: int = 100,
    retrieval_top_k: int = 3,
    model_name: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """
    Run the RAG QA workflow on SQuAD samples and evaluate:

    - Worker A EM/F1
    - Worker B EM/F1
    - Coordinator EM/F1
    - Workflow success rate
    - Retrieval hit rate
    - Worker B performance when retrieval hits or misses

    Important:
    The gold_context_hash and retrieved_context_hashes must be generated
    from the same original SQuAD context representation.
    """

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. "
            "Please check your root .env file."
        )

    if not SQUAD_FILE.exists():
        raise FileNotFoundError(
            f"SQuAD file not found: {SQUAD_FILE}"
        )

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Chroma directory not found: {CHROMA_DIR}"
        )

    loader = SquadLoader(
        dataset_dir=PROJECT_ROOT / "data",
    )

    samples = loader.load_samples_from_file(
        file_path=SQUAD_FILE,
        limit=limit,
        answerable_only=True,
    )

    external_retriever = ExternalKnowledgeRetriever(
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
        default_top_k=retrieval_top_k,
    )

    llm_client = LLMClient(
        model_name=model_name,
        temperature=0.0,
        max_tokens=500,
        env_path=PROJECT_ROOT / ".env",
    )

    workflow = RAGQAWorkflow(
        llm_client=llm_client,
        external_retriever=external_retriever,
    )

    records: list[dict[str, Any]] = []

    for index, sample in enumerate(samples, start=1):
        print(
            f"\n[{index}/{len(samples)}] "
            f"Running sample: {sample.sample_id}"
        )
        print(f"Question: {sample.question}")

        initial_state = create_initial_rag_qa_state_from_sample(
            sample=sample,
            retrieval_top_k=retrieval_top_k,
        )

        final_state = workflow.run(initial_state)

        worker_a_output = final_state.get(
            "worker_a_output"
        )
        worker_b_output = final_state.get(
            "worker_b_output"
        )
        coordinator_output = final_state.get(
            "coordinator_output"
        )

        worker_a_prediction = (
            worker_a_output.answer
            if worker_a_output is not None
            else ""
        )

        worker_b_prediction = (
            worker_b_output.answer
            if worker_b_output is not None
            else ""
        )

        coordinator_prediction = (
            coordinator_output.final_answer
            if coordinator_output is not None
            else final_state.get("prediction") or ""
        )

        gold_answers = final_state.get(
            "gold_answers",
            [],
        )

        worker_a_eval = _evaluate_one_output(
            prediction=worker_a_prediction,
            gold_answers=gold_answers,
        )

        worker_b_eval = _evaluate_one_output(
            prediction=worker_b_prediction,
            gold_answers=gold_answers,
        )

        coordinator_eval = _evaluate_one_output(
            prediction=coordinator_prediction,
            gold_answers=gold_answers,
        )

        retrieved_chunk_ids = _get_value_or_call(
            final_state.get(
                "retrieved_chunk_ids",
                [],
            )
        )

        retrieved_context_hashes = _normalize_hash_list(
            final_state.get(
                "retrieved_context_hashes",
                [],
            )
        )

        gold_context_hash = final_state.get(
            "gold_context_hash"
        )

        retrieval_hit = calculate_retrieval_hit(
            gold_context_hash=gold_context_hash,
            retrieved_context_hashes=(
                retrieved_context_hashes
            ),
        )

        record = {
            "sample_id": final_state.get("sample_id"),
            "title": final_state.get("title"),
            "question": final_state.get("question"),
            "gold_answers": gold_answers,
            "gold_context_hash": gold_context_hash,
            "retrieval": {
                "top_k": final_state.get(
                    "retrieval_top_k"
                ),
                "retrieved_chunk_ids": (
                    retrieved_chunk_ids
                ),
                "retrieved_context_hashes": (
                    retrieved_context_hashes
                ),
                "retrieval_hit": retrieval_hit,
            },
            "worker_a": {
                "output": _safe_model_dump(
                    worker_a_output
                ),
                "evaluation": worker_a_eval,
            },
            "worker_b": {
                "output": _safe_model_dump(
                    worker_b_output
                ),
                "evaluation": worker_b_eval,
            },
            "critic": {
                "output": _safe_model_dump(
                    final_state.get("critic_output")
                ),
            },
            "coordinator": {
                "output": _safe_model_dump(
                    coordinator_output
                ),
                "evaluation": coordinator_eval,
            },
            "workflow": {
                "success": final_state.get("success"),
                "current_step": final_state.get(
                    "current_step"
                ),
                "error_message": final_state.get(
                    "error_message"
                ),
            },
        }

        records.append(record)

        print(
            "Result:",
            {
                "retrieval_hit": retrieval_hit,
                "worker_a_em": (
                    worker_a_eval["exact_match"]
                ),
                "worker_a_f1": round(
                    worker_a_eval["f1"],
                    4,
                ),
                "worker_b_em": (
                    worker_b_eval["exact_match"]
                ),
                "worker_b_f1": round(
                    worker_b_eval["f1"],
                    4,
                ),
                "coordinator_em": (
                    coordinator_eval["exact_match"]
                ),
                "coordinator_f1": round(
                    coordinator_eval["f1"],
                    4,
                ),
            },
        )

    summary = summarize_records(records)

    output = {
        "config": {
            "limit": limit,
            "retrieval_top_k": retrieval_top_k,
            "model_name": model_name,
            "squad_file": str(SQUAD_FILE),
            "chroma_dir": str(CHROMA_DIR),
            "collection_name": COLLECTION_NAME,
            "created_at": datetime.now().isoformat(
                timespec="seconds"
            ),
        },
        "summary": summary,
        "records": records,
    }

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_file = (
        RESULT_DIR
        / f"rag_qa_eval_{timestamp}.json"
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\n========== EVALUATION SUMMARY =========="
    )
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\nSaved result to: {output_file}")

    return output


def summarize_records(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Summarize answer quality, workflow success, and retrieval quality.
    """

    if not records:
        return {
            "count": 0,
            "worker_a": {
                "exact_match": 0.0,
                "f1": 0.0,
            },
            "worker_b": {
                "exact_match": 0.0,
                "f1": 0.0,
            },
            "coordinator": {
                "exact_match": 0.0,
                "f1": 0.0,
            },
            "workflow_success_rate": 0.0,
            "retrieval": {
                "evaluated_count": 0,
                "hit_count": 0,
                "miss_count": 0,
                "hit_rate": None,
            },
            "worker_b_retrieval_analysis": {
                "em_when_hit": None,
                "f1_when_hit": None,
                "em_when_miss": None,
                "f1_when_miss": None,
            },
        }

    count = len(records)

    def avg(path: list[str]) -> float:
        total = 0.0

        for record in records:
            value: Any = record

            for key in path:
                value = value[key]

            total += float(value)

        return total / count

    success_count = sum(
        1
        for record in records
        if record["workflow"]["success"] is True
    )

    retrieval_hit_values = [
        record["retrieval"]["retrieval_hit"]
        for record in records
        if record["retrieval"]["retrieval_hit"]
        is not None
    ]

    retrieval_evaluated_count = len(
        retrieval_hit_values
    )

    retrieval_hit_count = sum(
        1
        for value in retrieval_hit_values
        if value is True
    )

    retrieval_miss_count = sum(
        1
        for value in retrieval_hit_values
        if value is False
    )

    retrieval_hit_rate = None

    if retrieval_evaluated_count > 0:
        retrieval_hit_rate = (
            retrieval_hit_count
            / retrieval_evaluated_count
        )

    return {
        "count": count,
        "worker_a": {
            "exact_match": avg(
                [
                    "worker_a",
                    "evaluation",
                    "exact_match",
                ]
            ),
            "f1": avg(
                [
                    "worker_a",
                    "evaluation",
                    "f1",
                ]
            ),
        },
        "worker_b": {
            "exact_match": avg(
                [
                    "worker_b",
                    "evaluation",
                    "exact_match",
                ]
            ),
            "f1": avg(
                [
                    "worker_b",
                    "evaluation",
                    "f1",
                ]
            ),
        },
        "coordinator": {
            "exact_match": avg(
                [
                    "coordinator",
                    "evaluation",
                    "exact_match",
                ]
            ),
            "f1": avg(
                [
                    "coordinator",
                    "evaluation",
                    "f1",
                ]
            ),
        },
        "workflow_success_rate": (
            success_count / count
        ),
        "retrieval": {
            "evaluated_count": (
                retrieval_evaluated_count
            ),
            "hit_count": retrieval_hit_count,
            "miss_count": retrieval_miss_count,
            "hit_rate": retrieval_hit_rate,
        },
        "worker_b_retrieval_analysis": {
            "em_when_hit": (
                average_worker_b_metric_by_hit(
                    records,
                    metric_name="exact_match",
                    expected_hit=True,
                )
            ),
            "f1_when_hit": (
                average_worker_b_metric_by_hit(
                    records,
                    metric_name="f1",
                    expected_hit=True,
                )
            ),
            "em_when_miss": (
                average_worker_b_metric_by_hit(
                    records,
                    metric_name="exact_match",
                    expected_hit=False,
                )
            ),
            "f1_when_miss": (
                average_worker_b_metric_by_hit(
                    records,
                    metric_name="f1",
                    expected_hit=False,
                )
            ),
        },
    }


if __name__ == "__main__":
    run_evaluation(
        limit=10,
        retrieval_top_k=3,
        model_name="gpt-4o-mini",
    )