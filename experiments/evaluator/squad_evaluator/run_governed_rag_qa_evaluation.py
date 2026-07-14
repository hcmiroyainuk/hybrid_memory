from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.memory.context import GovernedContextBuilder
from src.memory.entities import Agent, MemoryItem, MemoryMetadata
from experiments.evaluator.squad_evaluator import SquadEvaluator
from src.memory.external_knowledge import ExternalKnowledgeRetriever


try:
    from src.memory.external_knowledge import SquadLoader
except ImportError:
    from src.memory.external_knowledge.squad_loader import SquadLoader

from src.llm import LLMClient
from src.memory.manager.memory_store import MemoryStore
from src.workflow.rag_workflow import (
    GovernedRAGQAWorkflow,
    create_initial_governed_rag_qa_state_from_sample,
)
from src.memory.services.memory_service import MemoryService
from src.memory.services.permission_service import PermissionService


# ----------------------------------------------------------------------
# Project root / env
# ----------------------------------------------------------------------

def find_project_root(start_path: Path) -> Path:
    """
    Find project root by searching upward for .env.
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


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

SQUAD_FILE = PROJECT_ROOT / "data" / "train-v2.0.json"

CHROMA_DIR = (
    PROJECT_ROOT
    / "data"
    / "test_external_knowledge"
    / "squad_chroma"
)

COLLECTION_NAME = "test_squad_external_knowledge"

RESULT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)

GOVERNED_MEMORY_DIR = (
    PROJECT_ROOT
    / "data"
    / "governed_eval_memory"
)

GOVERNED_MEMORY_FILE = GOVERNED_MEMORY_DIR / "memories.json"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _safe_model_dump(value: Any) -> Any:
    """
    Convert Pydantic models or normal objects into JSON-safe values.
    """

    if value is None:
        return None

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")

    if isinstance(value, list):
        return [_safe_model_dump(item) for item in value]

    if isinstance(value, dict):
        return {
            key: _safe_model_dump(item)
            for key, item in value.items()
        }

    return value


def _get_value_or_call(value: Any) -> Any:
    """
    If a value is accidentally stored as a method, call it.
    This avoids saving '<bound method ...>' into result JSON.
    """

    if callable(value):
        return value()

    return value


def _evaluate_one_output(
    prediction: str | None,
    gold_answers: list[str],
) -> dict:
    """
    Evaluate one prediction with SQuAD EM/F1.
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


# ----------------------------------------------------------------------
# Governed memory setup
# ----------------------------------------------------------------------

def create_eval_agents():
    """
    Create four agents for governed workflow evaluation.
    """

    worker_a = Agent.worker(
        agent_id="worker_a",
    )

    worker_b = Agent.worker(
        agent_id="worker_b",
    )

    critic = Agent.critic(
        agent_id="critic",
    )

    coordinator = Agent.coordinator(
        agent_id="coordinator",
    )

    return worker_a, worker_b, critic, coordinator


def create_empty_memory_service(
    *,
    reset_memory_file: bool = True,
) -> MemoryService:
    """
    Create memory service for governed evaluation.

    For fair SQuAD EM/F1 evaluation, this starts with empty governed memory
    by default, so no gold answer is leaked into memory.
    """

    GOVERNED_MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    if reset_memory_file and GOVERNED_MEMORY_FILE.exists():
        GOVERNED_MEMORY_FILE.unlink()

    memory_store = MemoryStore(GOVERNED_MEMORY_FILE)

    permission_service = PermissionService()

    memory_service = MemoryService(
        memory_store=memory_store,
        permission_service=permission_service,
        operation_log_store=None,
        memory_retriever=None,
    )

    return memory_service


def create_private_memory(
    *,
    memory_id: str,
    content: str,
    owner_agent_id: str,
    created_by_agent_id: str,
) -> MemoryItem:
    """
    Optional helper for seeding private governed memory.
    Do not seed gold answers if you want fair EM/F1.
    """

    metadata = MemoryMetadata(
        memory_id=memory_id,
        scope="private",
        owner_agent_id=owner_agent_id,
        created_by_agent_id=created_by_agent_id,
        status="active",
        memory_type="fact",
        tags=["eval", "private", owner_agent_id],
        readable_by=[owner_agent_id],
        writable_by=[owner_agent_id],
    )

    return MemoryItem(
        memory_id=memory_id,
        content=content,
        metadata=metadata,
    )


def create_shared_memory(
    *,
    memory_id: str,
    content: str,
    created_by_agent_id: str,
) -> MemoryItem:
    """
    Optional helper for seeding shared governed memory.
    Do not seed gold answers if you want fair EM/F1.
    """

    metadata = MemoryMetadata(
        memory_id=memory_id,
        scope="shared",
        owner_agent_id="shared",
        created_by_agent_id=created_by_agent_id,
        status="active",
        memory_type="fact",
        tags=["eval", "shared"],
        readable_by=["*"],
        writable_by=["coordinator"],
    )

    return MemoryItem(
        memory_id=memory_id,
        content=content,
        metadata=metadata,
    )


def optionally_seed_demo_memories(
    memory_service: MemoryService,
    *,
    seed_demo_memory: bool,
) -> None:
    """
    Optional memory seeding for debugging read governance.

    Default evaluation should keep this disabled to avoid contaminating EM/F1.
    """

    if not seed_demo_memory:
        return

    shared_memory = create_shared_memory(
        memory_id="eval_shared_general_beyonce",
        content="Shared memory: Beyonce is an American singer and performer.",
        created_by_agent_id="coordinator",
    )

    worker_a_memory = create_private_memory(
        memory_id="eval_worker_a_private_note",
        content="Worker A private memory: prefer short answer spans for SQuAD evaluation.",
        owner_agent_id="worker_a",
        created_by_agent_id="worker_a",
    )

    worker_b_memory = create_private_memory(
        memory_id="eval_worker_b_private_note",
        content="Worker B private memory: use retrieved evidence when it directly supports the answer.",
        owner_agent_id="worker_b",
        created_by_agent_id="worker_b",
    )

    memory_service.memory_store.create(shared_memory)
    memory_service.memory_store.create(worker_a_memory)
    memory_service.memory_store.create(worker_b_memory)


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def run_governed_evaluation(
    *,
    limit: int = 20,
    retrieval_top_k: int = 3,
    memory_top_k: int = 3,
    model_name: str = "gpt-4o-mini",
    seed_demo_memory: bool = False,
) -> dict:
    """
    Run GovernedRAGQAWorkflow on SQuAD samples and evaluate EM/F1.

    This evaluates the current governed workflow:
    - permission-filtered memory retrieval
    - external SQuAD Chroma retrieval
    - Worker A
    - Worker B
    - Critic
    - Coordinator
    """

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            f"OPENAI_API_KEY is missing. Tried loading .env from: {ENV_FILE}"
        )

    if not SQUAD_FILE.exists():
        raise FileNotFoundError(
            f"SQuAD file not found: {SQUAD_FILE}"
        )

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Chroma directory not found: {CHROMA_DIR}"
        )

    worker_a, worker_b, critic, coordinator = create_eval_agents()

    memory_service = create_empty_memory_service(
        reset_memory_file=True,
    )

    optionally_seed_demo_memories(
        memory_service=memory_service,
        seed_demo_memory=seed_demo_memory,
    )

    governed_context_builder = GovernedContextBuilder(
        memory_service=memory_service,
        memory_retriever=None,
        default_top_k=memory_top_k,
        max_chars_per_memory=800,
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
        env_path=ENV_FILE,
    )

    workflow = GovernedRAGQAWorkflow(
        llm_client=llm_client,
        external_retriever=external_retriever,
        governed_context_builder=governed_context_builder,
    )

    records: list[dict] = []

    for index, sample in enumerate(samples, start=1):
        print(f"\n[{index}/{len(samples)}] Running governed sample: {sample.sample_id}")
        print(f"Question: {sample.question}")

        initial_state = create_initial_governed_rag_qa_state_from_sample(
            sample=sample,
            worker_a_agent=worker_a,
            worker_b_agent=worker_b,
            critic_agent=critic,
            coordinator_agent=coordinator,
            retrieval_top_k=retrieval_top_k,
            memory_top_k=memory_top_k,
        )

        final_state = workflow.run(initial_state)

        worker_a_output = final_state.get("worker_a_output")
        worker_b_output = final_state.get("worker_b_output")
        coordinator_output = final_state.get("coordinator_output")

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

        gold_answers = final_state.get("gold_answers", [])

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
            final_state.get("retrieved_chunk_ids", [])
        )

        retrieved_context_hashes = _get_value_or_call(
            final_state.get("retrieved_context_hashes", [])
        )

        record = {
            "sample_id": final_state.get("sample_id"),
            "title": final_state.get("title"),
            "question": final_state.get("question"),
            "gold_answers": gold_answers,
            "gold_context_hash": final_state.get("gold_context_hash"),
            "retrieval": {
                "top_k": final_state.get("retrieval_top_k"),
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "retrieved_context_hashes": retrieved_context_hashes,
                "retrieval_hit": final_state.get("retrieval_hit"),
            },
            "memory_access": {
                "memory_top_k": final_state.get("memory_top_k"),
                "worker_a_memory_ids": final_state.get("worker_a_memory_ids"),
                "worker_b_memory_ids": final_state.get("worker_b_memory_ids"),
                "critic_memory_ids": final_state.get("critic_memory_ids"),
                "coordinator_memory_ids": final_state.get("coordinator_memory_ids"),
            },
            "worker_a": {
                "output": _safe_model_dump(worker_a_output),
                "evaluation": worker_a_eval,
            },
            "worker_b": {
                "output": _safe_model_dump(worker_b_output),
                "evaluation": worker_b_eval,
            },
            "critic": {
                "output": _safe_model_dump(final_state.get("critic_output")),
            },
            "coordinator": {
                "output": _safe_model_dump(coordinator_output),
                "evaluation": coordinator_eval,
            },
            "workflow": {
                "success": final_state.get("success"),
                "current_step": final_state.get("current_step"),
                "error_message": final_state.get("error_message"),
            },
        }

        records.append(record)

        print(
            "Result:",
            {
                "worker_a_em": worker_a_eval["exact_match"],
                "worker_a_f1": round(worker_a_eval["f1"], 4),
                "worker_b_em": worker_b_eval["exact_match"],
                "worker_b_f1": round(worker_b_eval["f1"], 4),
                "coordinator_em": coordinator_eval["exact_match"],
                "coordinator_f1": round(coordinator_eval["f1"], 4),
                "success": final_state.get("success"),
            },
        )

    summary = summarize_records(records)

    output = {
        "config": {
            "workflow": "GovernedRAGQAWorkflow",
            "limit": limit,
            "retrieval_top_k": retrieval_top_k,
            "memory_top_k": memory_top_k,
            "model_name": model_name,
            "seed_demo_memory": seed_demo_memory,
            "squad_file": str(SQUAD_FILE),
            "chroma_dir": str(CHROMA_DIR),
            "collection_name": COLLECTION_NAME,
            "memory_file": str(GOVERNED_MEMORY_FILE),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        "summary": summary,
        "records": records,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULT_DIR / f"governed_rag_qa_eval_{timestamp}.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print("\n========== GOVERNED EVALUATION SUMMARY ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved result to: {output_file}")

    return output


def summarize_records(records: list[dict]) -> dict:
    """
    Summarize average EM/F1 for Worker A, Worker B, and Coordinator.
    """

    if not records:
        return {
            "count": 0,
            "worker_a": {"exact_match": 0.0, "f1": 0.0},
            "worker_b": {"exact_match": 0.0, "f1": 0.0},
            "coordinator": {"exact_match": 0.0, "f1": 0.0},
            "workflow_success_rate": 0.0,
            "retrieval_hit_rate": None,
        }

    count = len(records)

    def avg(path: list[str]) -> float:
        total = 0.0

        for record in records:
            value = record
            for key in path:
                value = value[key]
            total += float(value)

        return total / count

    success_count = sum(
        1 for record in records
        if record["workflow"]["success"] is True
    )

    retrieval_hit_values = [
        record["retrieval"]["retrieval_hit"]
        for record in records
        if record["retrieval"]["retrieval_hit"] is not None
    ]

    retrieval_hit_rate = None
    if retrieval_hit_values:
        retrieval_hit_rate = (
            sum(1 for value in retrieval_hit_values if value is True)
            / len(retrieval_hit_values)
        )

    return {
        "count": count,
        "worker_a": {
            "exact_match": avg(["worker_a", "evaluation", "exact_match"]),
            "f1": avg(["worker_a", "evaluation", "f1"]),
        },
        "worker_b": {
            "exact_match": avg(["worker_b", "evaluation", "exact_match"]),
            "f1": avg(["worker_b", "evaluation", "f1"]),
        },
        "coordinator": {
            "exact_match": avg(["coordinator", "evaluation", "exact_match"]),
            "f1": avg(["coordinator", "evaluation", "f1"]),
        },
        "workflow_success_rate": success_count / count,
        "retrieval_hit_rate": retrieval_hit_rate,
    }


if __name__ == "__main__":
    run_governed_evaluation(
        limit=100,
        retrieval_top_k=3,
        memory_top_k=3,
        model_name="gpt-4o-mini",
        seed_demo_memory=False,
    )