from __future__ import annotations

"""
Isolated runtime construction for GateMem episodes.

This module assembles one complete object graph for one:

    GateMem episode × evaluation mode × run ID

It does not ingest turns, classify deletion/update events, retrieve answer
context, or generate benchmark responses. Those responsibilities belong to
the GateMem system adapter and answer service.
"""

import hashlib
import re
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias

from src.adapters.agent_registry import AgentRegistry
from src.adapters.memory_access_policy_service_adapter import (
    MemoryAccessPolicyServiceAdapter,
)
from src.adapters.promotion_memory_sharing_adapter import (
    PromotionMemorySharingAdapter,
)
from src.memory.entities import Agent
from src.memory.governance.config import GovernanceConfig
from src.memory.governance.decision_protocols import (
    MemoryGovernanceCoordinatorProtocol,
    MemoryGovernanceCriticProtocol,
)
from src.memory.governance.runtime import (
    MemoryGovernanceRuntime,
    WorkflowCompileOptions,
    create_memory_governance_runtime,
)
from src.memory.manager import (
    MemoryStore,
    OperationLogStore,
    PromotionRequestStore,
)
from src.memory.services.memory_access_policy_service import (
    MemoryAccessPolicyService,
)
from src.memory.services.memory_service import MemoryService
from src.memory.services.permission_service import PermissionService
from src.memory.services.promotion_service import PromotionService

from .mapper import MappedEpisode


# ---------------------------------------------------------------------------
# Evaluation mode
# ---------------------------------------------------------------------------


class EvaluationMode(str, Enum):
    """
    GateMem comparison conditions.

    The runtime factory builds the same object graph for every mode. The mode
    is consumed later by the ingestion/query orchestration layer:

    - private_only:
        keep newly ingested memories private;
    - ungoverned_shared:
        apply the baseline sharing rule without Coordinator/Critic review;
    - governed_shared:
        invoke the complete memory-governance runtime.
    """

    PRIVATE_ONLY = "private_only"
    UNGOVERNED_SHARED = "ungoverned_shared"
    GOVERNED_SHARED = "governed_shared"


def normalise_evaluation_mode(
    value: EvaluationMode | str,
) -> EvaluationMode:
    if isinstance(value, EvaluationMode):
        return value

    cleaned = str(value or "").strip().lower()
    try:
        return EvaluationMode(cleaned)
    except ValueError as error:
        supported = ", ".join(mode.value for mode in EvaluationMode)
        raise ValueError(
            f"Unsupported GateMem evaluation mode {value!r}. "
            f"Expected one of: {supported}."
        ) from error


# ---------------------------------------------------------------------------
# Factory dependency types
# ---------------------------------------------------------------------------


CoordinatorFactory: TypeAlias = Callable[
    [Agent, MappedEpisode],
    MemoryGovernanceCoordinatorProtocol,
]

CriticFactory: TypeAlias = Callable[
    [Agent, MappedEpisode],
    MemoryGovernanceCriticProtocol,
]

MemoryRetrieverFactory: TypeAlias = Callable[
    [MemoryStore, "GateMemRuntimePaths"],
    Any,
]


# ---------------------------------------------------------------------------
# Runtime data models
# ---------------------------------------------------------------------------


class GateMemRuntimeFactoryError(RuntimeError):
    """Raised when an isolated GateMem runtime cannot be constructed."""


@dataclass(frozen=True, slots=True)
class GateMemRuntimePaths:
    """Persistence paths belonging to one isolated runtime."""

    runtime_dir: Path
    memories_path: Path
    operation_logs_path: Path
    promotion_requests_path: Path

    def as_metadata(self) -> dict[str, str]:
        return {
            "runtime_dir": str(self.runtime_dir.resolve()),
            "memories_path": str(self.memories_path.resolve()),
            "operation_logs_path": str(
                self.operation_logs_path.resolve()
            ),
            "promotion_requests_path": str(
                self.promotion_requests_path.resolve()
            ),
        }


@dataclass(frozen=True, slots=True)
class GateMemEpisodeRuntime:
    """
    Complete runtime bundle for one GateMem episode.

    GateMem principals are represented by ordinary system Worker entities.
    Their benchmark roles, such as professor, student, or doctor, remain in
    ``principal_roles`` and ``policy_context``. They are not placed in
    ``Agent.role``, because the memory subsystem uses Agent.role for its own
    WORKER / CRITIC / COORDINATOR permission model.
    """

    run_id: str
    episode: MappedEpisode
    mode: EvaluationMode
    paths: GateMemRuntimePaths

    workers: Mapping[str, Agent]
    principal_roles: Mapping[str, str]
    coordinator_agent: Agent
    critic_agent: Agent
    agent_registry: AgentRegistry

    memory_store: MemoryStore
    operation_log_store: OperationLogStore
    promotion_request_store: PromotionRequestStore

    permission_service: PermissionService
    memory_retriever: Any | None
    memory_service: MemoryService
    access_policy_service: MemoryAccessPolicyService
    promotion_service: PromotionService

    access_policy_gateway: MemoryAccessPolicyServiceAdapter
    sharing_gateway: PromotionMemorySharingAdapter

    coordinator: MemoryGovernanceCoordinatorProtocol
    critic: MemoryGovernanceCriticProtocol
    governance_runtime: MemoryGovernanceRuntime

    policy_context: Mapping[str, Any]

    @property
    def namespace_id(self) -> str:
        """Namespace used to isolate this episode's state."""
        return self.episode.namespace_id

    def require_worker(self, principal_id: str) -> Agent:
        clean_id = str(principal_id or "").strip()
        if not clean_id:
            raise ValueError("principal_id cannot be empty.")

        try:
            return self.workers[clean_id]
        except KeyError as error:
            raise KeyError(
                f"GateMem principal {clean_id!r} is not registered "
                f"in episode {self.episode.episode_id!r}."
            ) from error

    def metadata(self) -> dict[str, Any]:
        """Return JSON-compatible runtime metadata for traces/manifests."""
        return {
            "run_id": self.run_id,
            "episode_id": self.episode.episode_id,
            "namespace_id": self.namespace_id,
            "domain": self.episode.domain,
            "evaluation_mode": self.mode.value,
            "isolated_runtime": True,
            "registered_worker_ids": list(self.workers),
            "registered_agent_ids": list(
                self.agent_registry.agent_ids
            ),
            "principal_roles": dict(self.principal_roles),
            "memory_retriever": (
                None
                if self.memory_retriever is None
                else type(self.memory_retriever).__name__
            ),
            **self.paths.as_metadata(),
        }


# ---------------------------------------------------------------------------
# Runtime factory
# ---------------------------------------------------------------------------


class GateMemRuntimeFactory:
    """
    Build a fresh memory-governance object graph for one GateMem episode.

    Mutable persistence is never reused across episode/mode/run combinations.
    The injected Coordinator and Critic factories may reuse stateless LLM
    clients, but they must return fresh decision components for each runtime.

    Parameters
    ----------
    coordinator_factory:
        Creates the workflow-facing Coordinator implementation. The first
        argument is the Coordinator Agent entity used for permissions and the
        second is the mapped GateMem episode.

    critic_factory:
        Creates the workflow-facing Critic implementation.

    memory_retriever_factory:
        Optional semantic retriever constructor. When omitted, MemoryService
        is still able to list permission-filtered memories; answer_service.py
        may apply lexical ranking over that safe set.

    overwrite_existing_runtime:
        When True, rebuilding the same episode/mode/run starts from empty
        persistence. This is suitable for GateMem reset semantics.
    """

    def __init__(
        self,
        *,
        coordinator_factory: CoordinatorFactory,
        critic_factory: CriticFactory,
        runtime_root: str | Path = Path("data/gatemem_runtime"),
        governance_config: GovernanceConfig | None = None,
        review_gate: Any | None = None,
        policy_assignment_compile: WorkflowCompileOptions | None = None,
        access_request_compile: WorkflowCompileOptions | None = None,
        memory_retriever_factory: MemoryRetrieverFactory | None = None,
        coordinator_agent_id: str = "gatemem_coordinator",
        critic_agent_id: str = "gatemem_critic",
        overwrite_existing_runtime: bool = True,
    ) -> None:
        if not callable(coordinator_factory):
            raise TypeError("coordinator_factory must be callable.")
        if not callable(critic_factory):
            raise TypeError("critic_factory must be callable.")
        if (
            memory_retriever_factory is not None
            and not callable(memory_retriever_factory)
        ):
            raise TypeError(
                "memory_retriever_factory must be callable or None."
            )

        self.coordinator_factory = coordinator_factory
        self.critic_factory = critic_factory
        self.runtime_root = Path(runtime_root)
        self.governance_config = (
            governance_config or GovernanceConfig()
        )
        self.review_gate = review_gate
        self.policy_assignment_compile = policy_assignment_compile
        self.access_request_compile = access_request_compile
        self.memory_retriever_factory = memory_retriever_factory
        self.coordinator_agent_id = self._required_text(
            coordinator_agent_id,
            "coordinator_agent_id",
        )
        self.critic_agent_id = self._required_text(
            critic_agent_id,
            "critic_agent_id",
        )
        self.overwrite_existing_runtime = bool(
            overwrite_existing_runtime
        )

        if self.coordinator_agent_id == self.critic_agent_id:
            raise ValueError(
                "coordinator_agent_id and critic_agent_id must differ."
            )

    def __call__(
        self,
        episode: MappedEpisode,
        mode: EvaluationMode | str,
        *,
        run_id: str = "default",
    ) -> GateMemEpisodeRuntime:
        return self.build(
            episode=episode,
            mode=mode,
            run_id=run_id,
        )

    def build(
        self,
        *,
        episode: MappedEpisode,
        mode: EvaluationMode | str,
        run_id: str = "default",
    ) -> GateMemEpisodeRuntime:
        if not isinstance(episode, MappedEpisode):
            raise TypeError(
                "episode must be a MappedEpisode instance."
            )

        resolved_mode = normalise_evaluation_mode(mode)
        clean_run_id = self._required_text(run_id, "run_id")

        self._validate_reserved_agent_ids(episode)

        paths = self.build_runtime_paths(
            episode_id=episode.episode_id,
            mode=resolved_mode,
            run_id=clean_run_id,
        )
        self._prepare_runtime_directory(paths.runtime_dir)

        try:
            return self._build_object_graph(
                episode=episode,
                mode=resolved_mode,
                run_id=clean_run_id,
                paths=paths,
            )
        except Exception as error:
            raise GateMemRuntimeFactoryError(
                "Failed to construct GateMem runtime for "
                f"episode_id={episode.episode_id!r}, "
                f"mode={resolved_mode.value!r}, "
                f"run_id={clean_run_id!r}: {error}"
            ) from error

    def _build_object_graph(
        self,
        *,
        episode: MappedEpisode,
        mode: EvaluationMode,
        run_id: str,
        paths: GateMemRuntimePaths,
    ) -> GateMemEpisodeRuntime:
        # --------------------------------------------------------------
        # Persistence
        # --------------------------------------------------------------

        memory_store = MemoryStore(paths.memories_path)
        operation_log_store = OperationLogStore(
            paths.operation_logs_path
        )
        promotion_request_store = PromotionRequestStore(
            paths.promotion_requests_path
        )

        # --------------------------------------------------------------
        # Permission identities
        # --------------------------------------------------------------

        workers = self._build_workers(episode)
        principal_roles = MappingProxyType(
            {
                principal.agent_id: principal.role
                for principal in episode.principals
            }
        )

        coordinator_agent = Agent.coordinator(
            agent_id=self.coordinator_agent_id
        )
        critic_agent = Agent.critic(
            agent_id=self.critic_agent_id
        )

        all_agents: dict[str, Agent] = {
            **workers,
            coordinator_agent.agent_id: coordinator_agent,
            critic_agent.agent_id: critic_agent,
        }
        agent_registry = AgentRegistry(all_agents)

        # --------------------------------------------------------------
        # Core services
        # --------------------------------------------------------------

        permission_service = PermissionService()

        memory_retriever = None
        if self.memory_retriever_factory is not None:
            memory_retriever = self.memory_retriever_factory(
                memory_store,
                paths,
            )

        memory_service = MemoryService(
            memory_store=memory_store,
            operation_log_store=operation_log_store,
            memory_retriever=memory_retriever,
            permission_service=permission_service,
        )

        access_policy_service = MemoryAccessPolicyService(
            memory_store=memory_store,
            permission_service=permission_service,
            operation_log_store=operation_log_store,
            known_agent_ids=agent_registry.agent_ids,
        )

        promotion_service = PromotionService(
            memory_store=memory_store,
            request_store=promotion_request_store,
            access_policy_service=access_policy_service,
        )

        # --------------------------------------------------------------
        # Workflow-facing adapters
        # --------------------------------------------------------------

        access_policy_gateway = (
            MemoryAccessPolicyServiceAdapter(
                access_policy_service=access_policy_service,
                agents=agent_registry,
            )
        )

        sharing_gateway = PromotionMemorySharingAdapter(
            promotion_service=promotion_service,
            agents=agent_registry.as_mapping(),
        )

        # --------------------------------------------------------------
        # Coordinator / Critic decision components
        # --------------------------------------------------------------

        coordinator = self.coordinator_factory(
            coordinator_agent,
            episode,
        )
        critic = self.critic_factory(
            critic_agent,
            episode,
        )

        # The governance runtime performs its own protocol-method checks.
        governance_runtime = create_memory_governance_runtime(
            coordinator=coordinator,
            critic=critic,
            access_policy_gateway=access_policy_gateway,
            sharing_gateway=sharing_gateway,
            config=self.governance_config,
            review_gate=self.review_gate,
            policy_assignment_compile=(
                self.policy_assignment_compile
            ),
            access_request_compile=(
                self.access_request_compile
            ),
        )

        policy_context = MappingProxyType(
            episode.to_policy_context()
        )

        return GateMemEpisodeRuntime(
            run_id=run_id,
            episode=episode,
            mode=mode,
            paths=paths,
            workers=MappingProxyType(dict(workers)),
            principal_roles=principal_roles,
            coordinator_agent=coordinator_agent,
            critic_agent=critic_agent,
            agent_registry=agent_registry,
            memory_store=memory_store,
            operation_log_store=operation_log_store,
            promotion_request_store=promotion_request_store,
            permission_service=permission_service,
            memory_retriever=memory_retriever,
            memory_service=memory_service,
            access_policy_service=access_policy_service,
            promotion_service=promotion_service,
            access_policy_gateway=access_policy_gateway,
            sharing_gateway=sharing_gateway,
            coordinator=coordinator,
            critic=critic,
            governance_runtime=governance_runtime,
            policy_context=policy_context,
        )

    def _build_workers(
        self,
        episode: MappedEpisode,
    ) -> dict[str, Agent]:
        workers: dict[str, Agent] = {}

        for principal in episode.principals:
            worker = Agent.worker(
                agent_id=principal.agent_id,
                name=principal.display_name,
            )

            # Preserve the benchmark role as descriptive context without
            # corrupting the memory subsystem's AgentRole.WORKER invariant.
            description = (
                f"GateMem principal in domain {episode.domain!r}; "
                f"benchmark role={principal.role!r}."
            )
            worker = worker.model_copy(
                update={"description": description}
            )

            if worker.agent_id in workers:
                raise ValueError(
                    "Duplicate principal worker ID: "
                    f"{worker.agent_id!r}."
                )

            workers[worker.agent_id] = worker

        if not workers:
            raise ValueError(
                f"Episode {episode.episode_id!r} has no principals."
            )

        return workers

    def build_runtime_paths(
        self,
        *,
        episode_id: str,
        mode: EvaluationMode | str,
        run_id: str = "default",
    ) -> GateMemRuntimePaths:
        clean_episode_id = self._required_text(
            episode_id,
            "episode_id",
        )
        clean_run_id = self._required_text(run_id, "run_id")
        resolved_mode = normalise_evaluation_mode(mode)

        digest = hashlib.sha256(
            (
                f"{clean_episode_id}|"
                f"{resolved_mode.value}|"
                f"{clean_run_id}"
            ).encode("utf-8")
        ).hexdigest()[:12]

        directory_name = (
            f"{self._safe_identifier(clean_episode_id)}__"
            f"{resolved_mode.value}__"
            f"{self._safe_identifier(clean_run_id)}__"
            f"{digest}"
        )
        runtime_dir = self.runtime_root / directory_name

        return GateMemRuntimePaths(
            runtime_dir=runtime_dir,
            memories_path=runtime_dir / "memories.json",
            operation_logs_path=(
                runtime_dir / "operation_logs.json"
            ),
            promotion_requests_path=(
                runtime_dir / "promotion_requests.json"
            ),
        )

    def _prepare_runtime_directory(
        self,
        runtime_dir: Path,
    ) -> None:
        if runtime_dir.exists():
            if not self.overwrite_existing_runtime:
                raise FileExistsError(
                    "GateMem runtime directory already exists: "
                    f"{runtime_dir}."
                )
            shutil.rmtree(runtime_dir)

        runtime_dir.mkdir(parents=True, exist_ok=False)

    def _validate_reserved_agent_ids(
        self,
        episode: MappedEpisode,
    ) -> None:
        principal_ids = set(episode.agent_ids)

        collisions = principal_ids.intersection(
            {
                self.coordinator_agent_id,
                self.critic_agent_id,
            }
        )
        if collisions:
            raise ValueError(
                "GateMem principal IDs collide with reserved governance "
                f"Agent IDs: {sorted(collisions)}."
            )

    @staticmethod
    def _required_text(
        value: Any,
        field_name: str,
    ) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError(f"{field_name} cannot be empty.")
        return cleaned

    @staticmethod
    def _safe_identifier(value: Any) -> str:
        cleaned = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            str(value or "").strip(),
        ).strip("_")
        return cleaned or "value"


# ---------------------------------------------------------------------------
# Convenience facade
# ---------------------------------------------------------------------------


def build_episode_runtime(
    *,
    episode: MappedEpisode,
    mode: EvaluationMode | str,
    coordinator_factory: CoordinatorFactory,
    critic_factory: CriticFactory,
    run_id: str = "default",
    runtime_root: str | Path = Path("data/gatemem_runtime"),
    governance_config: GovernanceConfig | None = None,
    review_gate: Any | None = None,
    policy_assignment_compile: WorkflowCompileOptions | None = None,
    access_request_compile: WorkflowCompileOptions | None = None,
    memory_retriever_factory: MemoryRetrieverFactory | None = None,
    coordinator_agent_id: str = "gatemem_coordinator",
    critic_agent_id: str = "gatemem_critic",
    overwrite_existing_runtime: bool = True,
) -> GateMemEpisodeRuntime:
    """Build one isolated runtime without retaining a factory instance."""
    factory = GateMemRuntimeFactory(
        coordinator_factory=coordinator_factory,
        critic_factory=critic_factory,
        runtime_root=runtime_root,
        governance_config=governance_config,
        review_gate=review_gate,
        policy_assignment_compile=policy_assignment_compile,
        access_request_compile=access_request_compile,
        memory_retriever_factory=memory_retriever_factory,
        coordinator_agent_id=coordinator_agent_id,
        critic_agent_id=critic_agent_id,
        overwrite_existing_runtime=overwrite_existing_runtime,
    )
    return factory.build(
        episode=episode,
        mode=mode,
        run_id=run_id,
    )


__all__ = [
    "EvaluationMode",
    "normalise_evaluation_mode",
    "CoordinatorFactory",
    "CriticFactory",
    "MemoryRetrieverFactory",
    "GateMemRuntimeFactoryError",
    "GateMemRuntimePaths",
    "GateMemEpisodeRuntime",
    "GateMemRuntimeFactory",
    "build_episode_runtime",
]