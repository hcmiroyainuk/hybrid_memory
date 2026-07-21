from __future__ import annotations

import os
from pathlib import Path

from src.llm import LLMClient
from src.memory.entities import PromotionRequest
from src.memory.manager import MemoryStore, OperationLogStore

try:
    from src.memory.services.memory_service import MemoryService
    from src.memory.services.operation_log_service import (
        OperationLogService,
    )
    from src.memory.services.permission_service import (
        PermissionService,
    )
    from src.memory.services.promotion_service import (
        PromotionService,
    )
except ModuleNotFoundError:
    # Compatibility with projects that name the package ``service``.
    from src.memory.services.memory_service import MemoryService
    from src.memory.services.operation_log_service import (
        OperationLogService,
    )
    from src.memory.services.permission_service import (
        PermissionService,
    )
    from src.memory.services.promotion_service import (
        PromotionService,
    )

from experiments.evaluator.informativebench import (
    DEFAULT_EXPERIMENT_MODES,
    NeedlePersonaEvaluationRunner,
    NeedlePersonaEvaluator,
    NeedlePersonaIngestionService,
    NeedlePersonaLoader,
    NeedlePersonaNodeConfig,
    NeedlePersonaNodeDependencies,
    build_default_governance_agents,
    build_default_persona_agents,
    create_needle_persona_workflow,
)


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

DATASET_PATH = Path("data/informativebench_data/dataset_2hop.jsonl")
OUTPUT_ROOT = Path(
    "experiments/evaluator/experiments/results/informativebench/needle_persona"
)

# Use None to run the complete dataset.
LIMIT: int | None = 100

MODES = DEFAULT_EXPERIMENT_MODES

MODEL_NAME = os.getenv(
    "OPENAI_MODEL",
    "gpt-4o-mini",
)

RUNTIME_DATA_DIR = Path(
    "data/needle_persona_runtime"
)

# The evaluation run starts from an empty persistence layer.
RESET_RUNTIME_DATA = True


class InMemoryPromotionRequestStore:
    """
    Minimal promotion-request store used by the experiment process.

    Promotion requests only need to live for the duration of one batch run;
    memories and operation logs remain JSON-backed.
    """

    def __init__(self) -> None:
        self._requests: dict[
            str,
            PromotionRequest,
        ] = {}

    def create(
        self,
        request: PromotionRequest,
    ) -> PromotionRequest:
        if request.request_id in self._requests:
            raise ValueError(
                "Promotion request already exists: "
                f"{request.request_id!r}."
            )

        stored = request.model_copy(deep=True)
        self._requests[stored.request_id] = stored
        return stored.model_copy(deep=True)

    def get_by_id(
        self,
        request_id: str,
    ) -> PromotionRequest:
        request = self._requests.get(request_id)

        if request is None:
            raise KeyError(
                f"Promotion request not found: {request_id!r}."
            )

        return request.model_copy(deep=True)

    def replace(
        self,
        request: PromotionRequest,
    ) -> PromotionRequest:
        if request.request_id not in self._requests:
            raise KeyError(
                "Promotion request not found: "
                f"{request.request_id!r}."
            )

        stored = request.model_copy(deep=True)
        self._requests[stored.request_id] = stored
        return stored.model_copy(deep=True)

    def list_all(
        self,
    ) -> list[PromotionRequest]:
        return [
            request.model_copy(deep=True)
            for request in self._requests.values()
        ]

    def list_pending(
        self,
    ) -> list[PromotionRequest]:
        return [
            request.model_copy(deep=True)
            for request in self._requests.values()
            if request.is_pending()
        ]


def reset_runtime_data() -> None:
    """
    Remove persistence files left by an earlier experiment process.
    """
    if not RESET_RUNTIME_DATA:
        return

    for filename in (
        "memories.json",
        "operation_logs.json",
    ):
        path = RUNTIME_DATA_DIR / filename

        if path.exists():
            path.unlink()


def build_runner() -> NeedlePersonaEvaluationRunner:
    """
    Construct all services required by the experiment.
    """
    reset_runtime_data()
    RUNTIME_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    llm_client = LLMClient(
        model_name=MODEL_NAME,
        temperature=0.0,
        max_tokens=800,
        env_path=".env",
    )

    memory_store = MemoryStore(
        RUNTIME_DATA_DIR / "memories.json"
    )
    operation_log_store = OperationLogStore(
        RUNTIME_DATA_DIR / "operation_logs.json"
    )

    permission_service = PermissionService()

    memory_service = MemoryService(
        memory_store=memory_store,
        operation_log_store=operation_log_store,
        memory_retriever=None,
        permission_service=permission_service,
    )

    operation_log_service = OperationLogService(
        operation_log_store
    )

    promotion_service = PromotionService(
        memory_store=memory_store,
        request_store=InMemoryPromotionRequestStore(),
        operation_log_service=operation_log_service,
        permission_service=permission_service,
    )

    persona_agents = build_default_persona_agents()
    critic_agent, coordinator_agent = (
        build_default_governance_agents()
    )

    ingestion_service = NeedlePersonaIngestionService(
        llm_client=llm_client,
        memory_service=memory_service,
        agents=persona_agents,
        include_task_context=False,
        rollback_on_error=True,
        require_memory_per_source=True,
    )

    dependencies = NeedlePersonaNodeDependencies(
        llm_client=llm_client,
        memory_service=memory_service,
        promotion_service=promotion_service,
        persona_agents=persona_agents,
        critic_agent=critic_agent,
        coordinator_agent=coordinator_agent,
    )

    workflow = create_needle_persona_workflow(
        dependencies=dependencies,
        node_config=NeedlePersonaNodeConfig(
            routing_strategy="deterministic",
            contributor_top_k=5,
            responder_top_k=8,
            existing_context_limit=8,
            # MemoryService has no semantic retriever in this compact entry
            # point, so workflow retrieval uses its built-in lexical ranking.
            allow_lexical_retrieval_fallback=True,
            sanitise_answer_references=True,
            ungoverned_task_scoped_acl=True,
        ),
    )

    loader = NeedlePersonaLoader(
        DATASET_PATH,
        source_granularity="persona_document",
        include_collaborative_chat=False,
        retain_raw_fields_in_metadata=False,
        strict=True,
    )

    return NeedlePersonaEvaluationRunner(
        loader=loader,
        ingestion_service=ingestion_service,
        workflow=workflow,
        evaluator=NeedlePersonaEvaluator(),
        runtime_metadata={
            "llm_model": MODEL_NAME,
            "routing_strategy": "deterministic",
            "retrieval_strategy": "lexical_fallback",
        },
    )


def print_summary(
    summary: dict,
) -> None:
    """
    Print the primary comparison metrics.
    """
    print()
    print(
        f"Experiment: {summary['experiment_id']}"
    )
    print(
        "Runs: "
        f"{summary['successful_run_count']} successful, "
        f"{summary['failed_run_count']} failed"
    )

    for mode, mode_summary in summary[
        "by_mode"
    ].items():
        metrics = mode_summary["metrics"]

        def mean(name: str) -> str:
            value = metrics[name]["mean"]
            return (
                f"{value:.4f}"
                if value is not None
                else "N/A"
            )

        print()
        print(f"[{mode}]")
        print(
            "answer_accuracy = "
            f"{mean('answer_accuracy')}"
        )
        print(
            "entity_f1 = "
            f"{mean('entity_f1')}"
        )
        print(
            "retrieval_recall = "
            f"{mean('target_source_retrieval_recall')}"
        )
        print(
            "cross_agent_retrieval = "
            f"{mean('cross_agent_source_retrieval_recall')}"
        )
        print(
            "governance_precision = "
            f"{mean('governance_precision')}"
        )
        print(
            "governance_recall = "
            f"{mean('governance_recall')}"
        )
        print(
            "transfer_success = "
            f"{mean('transfer_success')}"
        )


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "Dataset file was not found: "
            f"{DATASET_PATH.resolve()}"
        )

    runner = build_runner()

    report = runner.run_batch(
        output_root=OUTPUT_ROOT,
        modes=MODES,
        limit=LIMIT,
        continue_on_error=False,
        save_after_each_run=True,
    )

    print_summary(dict(report.summary))

    print()
    print(
        f"Results saved to: "
        f"{report.output_dir.resolve()}"
    )

    if report.failures:
        print()
        print("Failed runs:")

        for failure in report.failures:
            print(
                f"[{failure.experiment_mode}] "
                f"stage={failure.stage} | "
                f"{failure.error_type}: "
                f"{failure.error_message}"
            )

        print()
        print(
            "Full failure records: "
            f"{report.failures_jsonl_path.resolve()}"
        )


if __name__ == "__main__":
    main()
