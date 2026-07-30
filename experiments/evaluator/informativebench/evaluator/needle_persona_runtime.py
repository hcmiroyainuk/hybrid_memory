from __future__ import annotations

import hashlib
import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from src.llm import LLMClient
from src.adapters import (
    PromotionMemorySharingAdapter,
)
from src.memory.entities import (
    Agent,
    AgentRole,
)
from src.memory.manager import (
    MemoryStore,
    OperationLogStore,
    PromotionRequestStore,
)
from src.memory.retrieval import (
    MemoryRetriever,
)
from src.memory.services.memory_service import (
    MemoryService,
)
from src.memory.services.permission_service import (
    PermissionService,
)
from src.memory.services.promotion_service import (
    PromotionService,
)

from ..config.needle_persona_config import (
    ExperimentMode,
    NeedlePersonaExperimentConfig,
)
from ..data_preparing.needle_persona_ingestion import (
    NeedlePersonaIngestionService,
    build_default_persona_agents,
)
from ..data_preparing.needle_persona_loader import (
    NeedlePersonaLoader,
)
from ..data_preparing.needle_persona_models import (
    NeedlePersonaSample,
)
from .needle_persona_evaluation_runner import (
    NeedlePersonaEvaluationRunner,
    NeedlePersonaEvaluationRuntime,
    NeedlePersonaIngestionSnapshot,
)
from .needle_persona_evaluator import (
    NeedlePersonaEvaluator,
    NeedlePersonaEvaluatorConfig,
)
from ..prompts.needle_persona_prompts import (
    COORDINATOR_AGENT_ID,
    CRITIC_AGENT_ID,
    build_needle_persona_prompts,
)
from ..workflow.dependencies.needle_persona_dependencies import (
    NeedlePersonaNodeDependencies,
)
from ..workflow.needle_persona_workflow import (
    build_needle_persona_workflow,
)


class NeedlePersonaRuntimeError(
    RuntimeError
):
    """
    Raised when one isolated experiment runtime cannot be constructed.
    """


@dataclass(
    frozen=True,
    slots=True,
)
class NeedlePersonaLLMConfig:
    """
    LLM settings shared by all isolated sample-mode runtimes.

    The LLM client itself is stateless with respect to memory governance, so it
    may be reused. MemoryStore, PromotionRequestStore, services, adapters, and
    workflows are rebuilt for every run.
    """

    model_name: str = field(
        default_factory=lambda: os.getenv(
            "OPENAI_MODEL",
            "gpt-4o-mini",
        )
    )
    temperature: float = 0.0
    max_tokens: int = 800
    env_path: str | Path = ".env"

    def __post_init__(self) -> None:
        model_name = str(
            self.model_name
        ).strip()

        if not model_name:
            raise ValueError(
                "model_name cannot be empty."
            )

        if self.temperature < 0:
            raise ValueError(
                "temperature cannot be negative."
            )

        if self.max_tokens <= 0:
            raise ValueError(
                "max_tokens must be greater "
                "than zero."
            )

        object.__setattr__(
            self,
            "model_name",
            model_name,
        )
        object.__setattr__(
            self,
            "env_path",
            Path(self.env_path),
        )


@dataclass(
    frozen=True,
    slots=True,
)
class NeedlePersonaRuntimePaths:
    """
    Persistence paths belonging to one exact sample-mode run.
    """

    run_dir: Path
    memories_path: Path
    operation_logs_path: Path
    promotion_requests_path: Path

    def as_metadata(
        self,
    ) -> dict[str, str]:
        return {
            "run_dir": str(
                self.run_dir.resolve()
            ),
            "memories_path": str(
                self.memories_path.resolve()
            ),
            "operation_logs_path": str(
                self.operation_logs_path.resolve()
            ),
            "promotion_requests_path": str(
                self.promotion_requests_path.resolve()
            ),
        }


class NeedlePersonaRuntimeFactory:
    """
    Build one canonical ingestion snapshot per sample and isolated mode runtimes.

    LLM memory extraction is executed only by ``create_ingestion_snapshot``.
    ``build_from_snapshot`` copies the resulting memories and operation logs into
    a fresh persistence directory before constructing each mode's services,
    adapter, workflow, and evaluator. Promotion requests and ACL mutations remain
    isolated to the mode runtime.
    """

    def __init__(
        self,
        *,
        experiment_config: (
            NeedlePersonaExperimentConfig
            | None
        ) = None,
        llm_config: (
            NeedlePersonaLLMConfig | None
        ) = None,
        evaluator_config: (
            NeedlePersonaEvaluatorConfig
            | None
        ) = None,
        llm_client: LLMClient | None = None,
        extra_metadata: (
            Mapping[str, Any] | None
        ) = None,
    ) -> None:
        self.experiment_config = (
            experiment_config
            or NeedlePersonaExperimentConfig()
        )
        self.llm_config = (
            llm_config
            or NeedlePersonaLLMConfig()
        )
        self.evaluator_config = (
            evaluator_config
            or NeedlePersonaEvaluatorConfig()
        )
        self.extra_metadata = dict(
            extra_metadata or {}
        )

        self.llm_client = (
            llm_client
            or self._build_llm_client()
        )

    def __call__(
        self,
        run_id: str,
        sample: NeedlePersonaSample,
        mode: ExperimentMode,
    ) -> NeedlePersonaEvaluationRuntime:
        """
        Construct one isolated runtime.
        """
        clean_run_id = self._required_text(
            run_id,
            "run_id",
        )

        if (
            sample.sample_id.strip()
            == ""
        ):
            raise NeedlePersonaRuntimeError(
                "sample.sample_id cannot be empty."
            )

        paths = self.build_runtime_paths(
            run_id=clean_run_id,
            sample_id=sample.sample_id,
            mode=mode,
        )
        self._prepare_run_directory(
            paths.run_dir
        )

        try:
            return self._build_runtime(
                run_id=clean_run_id,
                sample=sample,
                mode=mode,
                paths=paths,
            )
        except Exception as error:
            raise NeedlePersonaRuntimeError(
                "Failed to construct isolated "
                f"runtime for run_id={clean_run_id!r}, "
                f"sample_id={sample.sample_id!r}, "
                f"mode={mode!r}: {error}"
            ) from error

    def create_ingestion_snapshot(
        self,
        *,
        run_id: str,
        sample: NeedlePersonaSample,
    ) -> NeedlePersonaIngestionSnapshot:
        """
        Run LLM ingestion exactly once and preserve its private-memory files.

        The returned snapshot is immutable from the runner's perspective.
        Mode-specific runtimes copy its persistence files before constructing
        their MemoryStore and services.
        """
        clean_run_id = self._required_text(
            run_id,
            "run_id",
        )
        self._validate_sample(
            sample
        )

        paths = self.build_snapshot_paths(
            run_id=clean_run_id,
            sample_id=sample.sample_id,
        )
        self._prepare_run_directory(
            paths.run_dir
        )

        started = perf_counter()

        try:
            seed_runtime = self._build_runtime(
                run_id=clean_run_id,
                sample=sample,
                mode="private_only",
                paths=paths,
                metadata_extra={
                    "runtime_purpose": (
                        "canonical_ingestion_snapshot"
                    ),
                    "canonical_ingestion": True,
                    "mode_workflow_executed": False,
                },
            )

            memory_index = (
                seed_runtime
                .ingestion_service
                .ingest(
                    sample=sample,
                    run_id=clean_run_id,
                )
            )

            self._require_snapshot_file(
                paths.memories_path,
                "memories",
            )
            self._require_snapshot_file(
                paths.operation_logs_path,
                "operation logs",
            )

            duration_ms = (
                perf_counter() - started
            ) * 1000.0

            return NeedlePersonaIngestionSnapshot(
                seed_run_id=clean_run_id,
                sample_id=sample.sample_id,
                memory_index=memory_index,
                memories_path=(
                    paths.memories_path
                ),
                operation_logs_path=(
                    paths.operation_logs_path
                ),
                created_at_utc=(
                    self._utc_now()
                ),
                ingestion_duration_ms=(
                    duration_ms
                ),
                metadata={
                    "snapshot_directory": str(
                        paths.run_dir.resolve()
                    ),
                    "llm_model": (
                        self.llm_config
                        .model_name
                    ),
                    "temperature": (
                        self.llm_config
                        .temperature
                    ),
                    "max_tokens": (
                        self.llm_config
                        .max_tokens
                    ),
                },
            )

        except Exception as error:
            raise NeedlePersonaRuntimeError(
                "Failed to create canonical "
                f"ingestion snapshot for "
                f"run_id={clean_run_id!r}, "
                f"sample_id={sample.sample_id!r}: "
                f"{error}"
            ) from error

    def build_from_snapshot(
        self,
        *,
        run_id: str,
        sample: NeedlePersonaSample,
        mode: ExperimentMode,
        snapshot: (
            NeedlePersonaIngestionSnapshot
        ),
    ) -> NeedlePersonaEvaluationRuntime:
        """
        Build one isolated mode runtime from a canonical ingestion snapshot.
        """
        clean_run_id = self._required_text(
            run_id,
            "run_id",
        )
        self._validate_sample(
            sample
        )

        if not isinstance(
            snapshot,
            NeedlePersonaIngestionSnapshot,
        ):
            raise TypeError(
                "snapshot must be a "
                "NeedlePersonaIngestionSnapshot."
            )

        if (
            snapshot.sample_id
            != sample.sample_id
        ):
            raise NeedlePersonaRuntimeError(
                "Snapshot sample_id does not "
                "match sample.sample_id: "
                f"{snapshot.sample_id!r} != "
                f"{sample.sample_id!r}."
            )

        paths = self.build_runtime_paths(
            run_id=clean_run_id,
            sample_id=sample.sample_id,
            mode=mode,
        )
        self._prepare_run_directory(
            paths.run_dir
        )

        try:
            self._copy_snapshot_file(
                source=(
                    snapshot.memories_path
                ),
                destination=(
                    paths.memories_path
                ),
                label="memories",
            )
            self._copy_snapshot_file(
                source=(
                    snapshot.operation_logs_path
                ),
                destination=(
                    paths.operation_logs_path
                ),
                label="operation logs",
            )

            return self._build_runtime(
                run_id=clean_run_id,
                sample=sample,
                mode=mode,
                paths=paths,
                metadata_extra={
                    "runtime_purpose": (
                        "mode_evaluation"
                    ),
                    "canonical_ingestion": False,
                    "ingestion_reused": True,
                    **snapshot.as_metadata(),
                },
            )

        except Exception as error:
            raise NeedlePersonaRuntimeError(
                "Failed to construct isolated "
                "runtime from canonical snapshot "
                f"for run_id={clean_run_id!r}, "
                f"sample_id={sample.sample_id!r}, "
                f"mode={mode!r}: {error}"
            ) from error

    # ------------------------------------------------------------------
    # Runtime construction
    # ------------------------------------------------------------------

    def _build_runtime(
        self,
        *,
        run_id: str,
        sample: NeedlePersonaSample,
        mode: ExperimentMode,
        paths: NeedlePersonaRuntimePaths,
        metadata_extra: (
            Mapping[str, Any] | None
        ) = None,
    ) -> NeedlePersonaEvaluationRuntime:
        memory_store = MemoryStore(
            paths.memories_path
        )
        operation_log_store = (
            OperationLogStore(
                paths.operation_logs_path
            )
        )
        promotion_request_store = (
            PromotionRequestStore(
                paths.promotion_requests_path
            )
        )

        permission_service = (
            PermissionService()
        )

        memory_retriever = MemoryRetriever(
            memory_store=memory_store,
            embedding_model=(
                "text-embedding-3-small"
            ),
            persist_directory=None,
        )

        memory_service = MemoryService(
            memory_store=memory_store,
            operation_log_store=(
                operation_log_store
            ),
            memory_retriever=(
                memory_retriever
            ),
            permission_service=(
                permission_service
            ),
        )

        promotion_service = (
            PromotionService(
                memory_store=memory_store,
                request_store=(
                    promotion_request_store
                ),
            )
        )

        persona_agents = (
            build_default_persona_agents()
        )
        (
            critic_agent,
            coordinator_agent,
        ) = build_governance_agents()

        all_agents = {
            **persona_agents,
            critic_agent.agent_id: (
                critic_agent
            ),
            coordinator_agent.agent_id: (
                coordinator_agent
            ),
        }

        sharing_gateway = (
            PromotionMemorySharingAdapter(
                promotion_service=(
                    promotion_service
                ),
                agents=all_agents,
            )
        )

        prompt_bundle = (
            build_needle_persona_prompts()
        )

        ingestion_service = (
            NeedlePersonaIngestionService(
                llm_client=self.llm_client,
                memory_service=(
                    memory_service
                ),
                agents=persona_agents,
                worker_prompts=(
                    prompt_bundle
                    .worker_prompts
                ),
                include_task_context=(
                    self.experiment_config
                    .ingestion_include_task_context
                ),
                rollback_on_error=(
                    self.experiment_config
                    .ingestion_rollback_on_error
                ),
                require_memory_per_source=(
                    self.experiment_config
                    .require_memory_per_source
                ),
            )
        )

        dependencies = (
            NeedlePersonaNodeDependencies(
                llm_client=self.llm_client,
                memory_service=(
                    memory_service
                ),
                sharing_gateway=(
                    sharing_gateway
                ),
                persona_agents=(
                    persona_agents
                ),
                critic_agent=(
                    critic_agent
                ),
                coordinator_agent=(
                    coordinator_agent
                ),
                prompt_bundle=(
                    prompt_bundle
                ),
            )
        )

        workflow = (
            build_needle_persona_workflow(
                dependencies=dependencies,
                workflow_config=(
                    self.experiment_config
                    .workflow_config
                ),
            )
        )

        evaluator = NeedlePersonaEvaluator(
            config=self.evaluator_config
        )

        return (
            NeedlePersonaEvaluationRuntime(
                ingestion_service=(
                    ingestion_service
                ),
                workflow=workflow,
                evaluator=evaluator,
                metadata={
                    "run_id": run_id,
                    "sample_id": (
                        sample.sample_id
                    ),
                    "experiment_mode": mode,
                    "llm_model": (
                        self.llm_config
                        .model_name
                    ),
                    "temperature": (
                        self.llm_config
                        .temperature
                    ),
                    "max_tokens": (
                        self.llm_config
                        .max_tokens
                    ),
                    "retrieval_strategy": (
                        "semantic_memory_retriever_"
                        "with_lexical_fallback"
                    ),
                    "sharing_gateway": (
                        "PromotionMemorySharingAdapter"
                    ),
                    "isolated_runtime": True,
                    **paths.as_metadata(),
                    **dict(
                        metadata_extra or {}
                    ),
                    **self.extra_metadata,
                },
            )
        )

    # ------------------------------------------------------------------
    # Paths and cleanup
    # ------------------------------------------------------------------

    def build_runtime_paths(
        self,
        *,
        run_id: str,
        sample_id: str,
        mode: ExperimentMode,
    ) -> NeedlePersonaRuntimePaths:
        """
        Return deterministic, collision-resistant paths for one run.
        """
        clean_run_id = self._required_text(
            run_id,
            "run_id",
        )
        clean_sample_id = (
            self._safe_identifier(
                sample_id
            )
        )
        clean_mode = self._safe_identifier(
            str(mode)
        )

        run_hash = hashlib.sha256(
            clean_run_id.encode(
                "utf-8"
            )
        ).hexdigest()[:12]

        run_dir = (
            self.experiment_config
            .runtime_root
            / (
                f"{clean_sample_id}__"
                f"{clean_mode}__"
                f"{run_hash}"
            )
        )

        return NeedlePersonaRuntimePaths(
            run_dir=run_dir,
            memories_path=(
                run_dir
                / "memories.json"
            ),
            operation_logs_path=(
                run_dir
                / "operation_logs.json"
            ),
            promotion_requests_path=(
                run_dir
                / "promotion_requests.json"
            ),
        )

    def build_snapshot_paths(
        self,
        *,
        run_id: str,
        sample_id: str,
    ) -> NeedlePersonaRuntimePaths:
        """
        Return persistence paths for one canonical sample ingestion.
        """
        clean_run_id = self._required_text(
            run_id,
            "run_id",
        )
        clean_sample_id = (
            self._safe_identifier(
                sample_id
            )
        )
        run_hash = hashlib.sha256(
            clean_run_id.encode(
                "utf-8"
            )
        ).hexdigest()[:12]

        run_dir = (
            self.experiment_config
            .runtime_root
            / (
                f"{clean_sample_id}__"
                "canonical_ingestion__"
                f"{run_hash}"
            )
        )

        return NeedlePersonaRuntimePaths(
            run_dir=run_dir,
            memories_path=(
                run_dir
                / "memories.json"
            ),
            operation_logs_path=(
                run_dir
                / "operation_logs.json"
            ),
            promotion_requests_path=(
                run_dir
                / "promotion_requests.json"
            ),
        )

    @staticmethod
    def _prepare_run_directory(
        run_dir: Path,
    ) -> None:
        """
        Start the exact run from empty persistence.
        """
        if run_dir.exists():
            shutil.rmtree(
                run_dir
            )

        run_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

    @staticmethod
    def _copy_snapshot_file(
        *,
        source: Path,
        destination: Path,
        label: str,
    ) -> None:
        source_path = Path(source)
        destination_path = Path(
            destination
        )

        if not source_path.is_file():
            raise FileNotFoundError(
                f"Canonical {label} file does "
                f"not exist: {source_path}."
            )

        destination_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        shutil.copy2(
            source_path,
            destination_path,
        )

    @staticmethod
    def _require_snapshot_file(
        path: Path,
        label: str,
    ) -> None:
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"Canonical {label} file was "
                f"not created: {path}."
            )

    @staticmethod
    def _validate_sample(
        sample: NeedlePersonaSample,
    ) -> None:
        if not isinstance(
            sample,
            NeedlePersonaSample,
        ):
            raise TypeError(
                "sample must be a "
                "NeedlePersonaSample."
            )

        if not sample.sample_id.strip():
            raise NeedlePersonaRuntimeError(
                "sample.sample_id cannot be empty."
            )

    @staticmethod
    def _utc_now(
    ) -> str:
        from datetime import (
            datetime,
            timezone,
        )

        return datetime.now(
            timezone.utc
        ).isoformat()

    # ------------------------------------------------------------------
    # LLM construction
    # ------------------------------------------------------------------

    def _build_llm_client(
        self,
    ) -> LLMClient:
        return LLMClient(
            model_name=(
                self.llm_config
                .model_name
            ),
            temperature=(
                self.llm_config
                .temperature
            ),
            max_tokens=(
                self.llm_config
                .max_tokens
            ),
            env_path=str(
                self.llm_config
                .env_path
            ),
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _required_text(
        value: Any,
        field_name: str,
    ) -> str:
        cleaned = str(
            value or ""
        ).strip()

        if not cleaned:
            raise ValueError(
                f"{field_name} cannot be empty."
            )

        return cleaned

    @staticmethod
    def _safe_identifier(
        value: Any,
    ) -> str:
        cleaned = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            str(value or "").strip(),
        ).strip("_")

        return cleaned or "value"


def build_governance_agents(
) -> tuple[
    Agent,
    Agent,
]:
    """
    Build fresh Critic and Coordinator entities for one runtime.
    """
    critic = Agent(
        agent_id=CRITIC_AGENT_ID,
        name="Needle Persona Critic",
        role=AgentRole.CRITIC,
    )
    coordinator = Agent(
        agent_id=COORDINATOR_AGENT_ID,
        name=(
            "Needle Persona Coordinator"
        ),
        role=AgentRole.COORDINATOR,
    )

    return critic, coordinator


def build_needle_persona_loader(
    config: (
        NeedlePersonaExperimentConfig
        | None
    ) = None,
) -> NeedlePersonaLoader:
    """
    Build the leakage-safe benchmark loader.
    """
    experiment_config = (
        config
        or NeedlePersonaExperimentConfig()
    )

    return NeedlePersonaLoader(
        experiment_config.dataset_path,
        source_granularity=(
            experiment_config
            .source_granularity
        ),
        include_collaborative_chat=(
            experiment_config
            .include_collaborative_chat
        ),
        retain_raw_fields_in_metadata=(
            experiment_config
            .retain_raw_fields_in_metadata
        ),
        strict=(
            experiment_config
            .loader_strict
        ),
    )


def build_needle_persona_runner(
    *,
    experiment_config: (
        NeedlePersonaExperimentConfig
        | None
    ) = None,
    llm_config: (
        NeedlePersonaLLMConfig | None
    ) = None,
    evaluator_config: (
        NeedlePersonaEvaluatorConfig
        | None
    ) = None,
    llm_client: LLMClient | None = None,
    extra_metadata: (
        Mapping[str, Any] | None
    ) = None,
) -> NeedlePersonaEvaluationRunner:
    """
    Build the loader, isolated runtime factory, and batch runner.
    """
    config = (
        experiment_config
        or NeedlePersonaExperimentConfig()
    )

    loader = build_needle_persona_loader(
        config
    )
    runtime_factory = (
        NeedlePersonaRuntimeFactory(
            experiment_config=config,
            llm_config=llm_config,
            evaluator_config=(
                evaluator_config
            ),
            llm_client=llm_client,
            extra_metadata=(
                extra_metadata
            ),
        )
    )

    return NeedlePersonaEvaluationRunner(
        loader=loader,
        runtime_factory=(
            runtime_factory
        ),
        config=config,
        runtime_metadata={
            "runtime_factory": (
                "NeedlePersonaRuntimeFactory"
            ),
            "single_llm_ingestion_per_sample": (
                True
            ),
            "canonical_snapshot_cloned_per_mode": (
                True
            ),
            "fresh_runtime_per_sample_mode": (
                True
            ),
            "persistent_acl_lifetime": (
                "isolated_runtime"
            ),
            **dict(
                extra_metadata or {}
            ),
        },
    )


__all__ = [
    "NeedlePersonaRuntimeError",
    "NeedlePersonaLLMConfig",
    "NeedlePersonaRuntimePaths",
    "NeedlePersonaIngestionSnapshot",
    "NeedlePersonaRuntimeFactory",
    "build_governance_agents",
    "build_needle_persona_loader",
    "build_needle_persona_runner",
]