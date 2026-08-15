from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.gatemem import (
    runtime_factory as runtime_factory_module,
)
from experiments.gatemem.mapper import (
    MappedEpisode,
    MappedPrincipal,
    MappedRelationship,
    MappedTurn,
)
from experiments.gatemem.runtime_factory import (
    EvaluationMode,
    GateMemEpisodeRuntime,
    GateMemRuntimeFactory,
    GateMemRuntimeFactoryError,
    build_episode_runtime,
    normalise_evaluation_mode,
)
from src.memory.entities import Agent


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class FakeCoordinator:
    """
    Minimal implementation of the Coordinator protocol.

    These methods are not executed by the runtime-factory unit tests, but
    keeping the complete interface makes the test double usable when the real
    governance-runtime builder is enabled in an integration test.
    """

    agent: Agent
    episode: MappedEpisode

    def assign_initial_policy(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return dict(kwargs)

    def finalise_initial_policy(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return dict(kwargs)

    def evaluate_access_request(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return dict(kwargs)

    def finalise_access_request(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return dict(kwargs)


@dataclass
class FakeCritic:
    """
    Minimal implementation of the Critic protocol.
    """

    agent: Agent
    episode: MappedEpisode

    def review_initial_policy(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return dict(kwargs)

    def review_access_request(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return dict(kwargs)


@dataclass
class FactoryHarness:
    factory: GateMemRuntimeFactory
    governance_runtime: Any
    governance_calls: list[dict[str, Any]]
    coordinator_calls: list[tuple[Agent, MappedEpisode]]
    critic_calls: list[tuple[Agent, MappedEpisode]]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mapped_episode() -> MappedEpisode:
    return MappedEpisode(
        episode_id="education episode 001",
        domain="education",
        principals=(
            MappedPrincipal(
                agent_id="prof_elena",
                role="professor",
                display_name="Professor Elena",
                metadata={
                    "department": "Computer Science",
                },
            ),
            MappedPrincipal(
                agent_id="student_maya",
                role="student",
                display_name="Maya",
                metadata={
                    "programme": "MSc AI",
                },
            ),
        ),
        relationships=(
            MappedRelationship(
                relationship_type="academic_supervision",
                principal_agent_id="prof_elena",
                target_agent_id="student_maya",
                access_scope="supervision_record",
                resource_refs={
                    "programme_id": "msc_ai",
                },
            ),
        ),
        turns=(
            MappedTurn(
                episode_id="education episode 001",
                domain="education",
                turn_id="turn_001",
                sequence_index=0,
                speaker_agent_id="student_maya",
                speaker_role="student",
                text="My dissertation deadline is 20 August.",
                timestamp="2026-08-01T09:00:00Z",
            ),
        ),
        metadata={
            "source": "unit_test",
        },
    )


@pytest.fixture
def factory_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> FactoryHarness:
    governance_runtime = SimpleNamespace(
        name="fake_governance_runtime"
    )
    governance_calls: list[dict[str, Any]] = []
    coordinator_calls: list[
        tuple[Agent, MappedEpisode]
    ] = []
    critic_calls: list[
        tuple[Agent, MappedEpisode]
    ] = []

    def fake_create_memory_governance_runtime(
        **kwargs: Any,
    ) -> Any:
        governance_calls.append(dict(kwargs))
        return governance_runtime

    monkeypatch.setattr(
        runtime_factory_module,
        "create_memory_governance_runtime",
        fake_create_memory_governance_runtime,
    )

    def coordinator_factory(
        agent: Agent,
        episode: MappedEpisode,
    ) -> FakeCoordinator:
        coordinator_calls.append((agent, episode))
        return FakeCoordinator(
            agent=agent,
            episode=episode,
        )

    def critic_factory(
        agent: Agent,
        episode: MappedEpisode,
    ) -> FakeCritic:
        critic_calls.append((agent, episode))
        return FakeCritic(
            agent=agent,
            episode=episode,
        )

    factory = GateMemRuntimeFactory(
        coordinator_factory=coordinator_factory,
        critic_factory=critic_factory,
        runtime_root=tmp_path / "runtime",
    )

    return FactoryHarness(
        factory=factory,
        governance_runtime=governance_runtime,
        governance_calls=governance_calls,
        coordinator_calls=coordinator_calls,
        critic_calls=critic_calls,
    )


# ---------------------------------------------------------------------------
# Evaluation-mode validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw_mode", "expected"),
    [
        (
            "private_only",
            EvaluationMode.PRIVATE_ONLY,
        ),
        (
            " PRIVATE_ONLY ",
            EvaluationMode.PRIVATE_ONLY,
        ),
        (
            "ungoverned_shared",
            EvaluationMode.UNGOVERNED_SHARED,
        ),
        (
            EvaluationMode.GOVERNED_SHARED,
            EvaluationMode.GOVERNED_SHARED,
        ),
    ],
)
def test_normalise_evaluation_mode(
    raw_mode: EvaluationMode | str,
    expected: EvaluationMode,
) -> None:
    assert normalise_evaluation_mode(raw_mode) is expected


def test_normalise_evaluation_mode_rejects_unknown_value() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported GateMem evaluation mode",
    ):
        normalise_evaluation_mode("unknown_mode")


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field_name", "kwargs"),
    [
        (
            "coordinator_factory",
            {
                "coordinator_factory": None,
                "critic_factory": lambda *_: object(),
            },
        ),
        (
            "critic_factory",
            {
                "coordinator_factory": lambda *_: object(),
                "critic_factory": None,
            },
        ),
        (
            "memory_retriever_factory",
            {
                "coordinator_factory": lambda *_: object(),
                "critic_factory": lambda *_: object(),
                "memory_retriever_factory": object(),
            },
        ),
    ],
)
def test_factory_rejects_non_callable_dependencies(
    field_name: str,
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(
        TypeError,
        match=field_name,
    ):
        GateMemRuntimeFactory(**kwargs)


def test_factory_rejects_equal_governance_agent_ids() -> None:
    with pytest.raises(
        ValueError,
        match="must differ",
    ):
        GateMemRuntimeFactory(
            coordinator_factory=lambda *_: object(),
            critic_factory=lambda *_: object(),
            coordinator_agent_id="governance_agent",
            critic_agent_id="governance_agent",
        )


# ---------------------------------------------------------------------------
# Path construction
# ---------------------------------------------------------------------------


def test_runtime_paths_are_deterministic_and_safe(
    factory_harness: FactoryHarness,
) -> None:
    factory = factory_harness.factory

    first = factory.build_runtime_paths(
        episode_id="episode / 001",
        mode="governed_shared",
        run_id="pilot / A",
    )
    second = factory.build_runtime_paths(
        episode_id="episode / 001",
        mode=EvaluationMode.GOVERNED_SHARED,
        run_id="pilot / A",
    )

    assert first == second
    assert first.runtime_dir.name.startswith(
        "episode_001__governed_shared__pilot_A__"
    )
    assert first.memories_path == (
        first.runtime_dir / "memories.json"
    )
    assert first.operation_logs_path == (
        first.runtime_dir / "operation_logs.json"
    )
    assert first.promotion_requests_path == (
        first.runtime_dir / "promotion_requests.json"
    )


def test_runtime_paths_change_with_mode_or_run_id(
    factory_harness: FactoryHarness,
) -> None:
    factory = factory_harness.factory

    governed = factory.build_runtime_paths(
        episode_id="episode_001",
        mode="governed_shared",
        run_id="run_001",
    )
    private = factory.build_runtime_paths(
        episode_id="episode_001",
        mode="private_only",
        run_id="run_001",
    )
    second_run = factory.build_runtime_paths(
        episode_id="episode_001",
        mode="governed_shared",
        run_id="run_002",
    )

    assert governed.runtime_dir != private.runtime_dir
    assert governed.runtime_dir != second_run.runtime_dir
    assert private.runtime_dir != second_run.runtime_dir


# ---------------------------------------------------------------------------
# Successful object-graph construction
# ---------------------------------------------------------------------------


def test_build_constructs_complete_isolated_runtime(
    mapped_episode: MappedEpisode,
    factory_harness: FactoryHarness,
) -> None:
    runtime = factory_harness.factory.build(
        episode=mapped_episode,
        mode="governed_shared",
        run_id="run_001",
    )

    assert isinstance(runtime, GateMemEpisodeRuntime)
    assert runtime.run_id == "run_001"
    assert runtime.episode is mapped_episode
    assert runtime.mode is EvaluationMode.GOVERNED_SHARED
    assert runtime.namespace_id == mapped_episode.episode_id
    assert runtime.paths.runtime_dir.is_dir()

    assert tuple(runtime.workers) == (
        "prof_elena",
        "student_maya",
    )
    assert runtime.principal_roles == {
        "prof_elena": "professor",
        "student_maya": "student",
    }

    professor = runtime.require_worker(
        "prof_elena"
    )
    student = runtime.require_worker(
        "student_maya"
    )

    assert professor.agent_id == "prof_elena"
    assert professor.name == "Professor Elena"
    assert getattr(
        professor.role,
        "value",
        professor.role,
    ) == "worker"
    assert "benchmark role='professor'" in (
        professor.description or ""
    )

    assert student.agent_id == "student_maya"
    assert getattr(
        student.role,
        "value",
        student.role,
    ) == "worker"

    assert getattr(
        runtime.coordinator_agent.role,
        "value",
        runtime.coordinator_agent.role,
    ) == "coordinator"
    assert getattr(
        runtime.critic_agent.role,
        "value",
        runtime.critic_agent.role,
    ) == "critic"

    expected_agent_ids = (
        "prof_elena",
        "student_maya",
        "gatemem_coordinator",
        "gatemem_critic",
    )
    assert runtime.agent_registry.agent_ids == (
        expected_agent_ids
    )

    # The services must share the same run-specific stores and permission
    # service. This is the central composition-root invariant.
    assert (
        runtime.memory_service.memory_store
        is runtime.memory_store
    )
    assert (
        runtime.memory_service.operation_log_store
        is runtime.operation_log_store
    )
    assert (
        runtime.memory_service.permission_service
        is runtime.permission_service
    )
    assert (
        runtime.access_policy_service.memory_store
        is runtime.memory_store
    )
    assert (
        runtime.access_policy_service.operation_log_store
        is runtime.operation_log_store
    )
    assert (
        runtime.access_policy_service.permission_service
        is runtime.permission_service
    )
    assert (
        runtime.promotion_service.memory_store
        is runtime.memory_store
    )
    assert (
        runtime.promotion_service.request_store
        is runtime.promotion_request_store
    )

    assert runtime.memory_retriever is None
    assert (
        runtime.governance_runtime
        is factory_harness.governance_runtime
    )
    assert runtime.policy_context == (
        mapped_episode.to_policy_context()
    )


def test_build_wires_factories_and_governance_builder(
    mapped_episode: MappedEpisode,
    factory_harness: FactoryHarness,
) -> None:
    runtime = factory_harness.factory(
        mapped_episode,
        EvaluationMode.GOVERNED_SHARED,
        run_id="run_002",
    )

    assert len(
        factory_harness.coordinator_calls
    ) == 1
    coordinator_agent, coordinator_episode = (
        factory_harness.coordinator_calls[0]
    )
    assert (
        coordinator_agent
        is runtime.coordinator_agent
    )
    assert coordinator_episode is mapped_episode

    assert len(factory_harness.critic_calls) == 1
    critic_agent, critic_episode = (
        factory_harness.critic_calls[0]
    )
    assert critic_agent is runtime.critic_agent
    assert critic_episode is mapped_episode

    assert len(
        factory_harness.governance_calls
    ) == 1
    call = factory_harness.governance_calls[0]

    assert call["coordinator"] is runtime.coordinator
    assert call["critic"] is runtime.critic
    assert (
        call["access_policy_gateway"]
        is runtime.access_policy_gateway
    )
    assert (
        call["sharing_gateway"]
        is runtime.sharing_gateway
    )
    assert (
        call["config"]
        is factory_harness.factory.governance_config
    )
    assert call["review_gate"] is None
    assert call["policy_assignment_compile"] is None
    assert call["access_request_compile"] is None


def test_memory_retriever_factory_is_injected(
    mapped_episode: MappedEpisode,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = SimpleNamespace(
        name="fake_retriever"
    )
    retriever_calls: list[tuple[Any, Any]] = []

    monkeypatch.setattr(
        runtime_factory_module,
        "create_memory_governance_runtime",
        lambda **_: SimpleNamespace(
            name="governance"
        ),
    )

    def build_retriever(
        memory_store: Any,
        paths: Any,
    ) -> Any:
        retriever_calls.append(
            (memory_store, paths)
        )
        return retriever

    factory = GateMemRuntimeFactory(
        coordinator_factory=(
            lambda agent, episode: FakeCoordinator(
                agent,
                episode,
            )
        ),
        critic_factory=(
            lambda agent, episode: FakeCritic(
                agent,
                episode,
            )
        ),
        memory_retriever_factory=build_retriever,
        runtime_root=tmp_path,
    )

    runtime = factory.build(
        episode=mapped_episode,
        mode="private_only",
        run_id="retriever_run",
    )

    assert retriever_calls == [
        (
            runtime.memory_store,
            runtime.paths,
        )
    ]
    assert runtime.memory_retriever is retriever
    assert (
        runtime.memory_service.memory_retriever
        is retriever
    )
    assert runtime.metadata()[
        "memory_retriever"
    ] == "SimpleNamespace"


# ---------------------------------------------------------------------------
# Runtime facade and immutability
# ---------------------------------------------------------------------------


def test_runtime_metadata_is_json_compatible(
    mapped_episode: MappedEpisode,
    factory_harness: FactoryHarness,
) -> None:
    runtime = factory_harness.factory.build(
        episode=mapped_episode,
        mode="ungoverned_shared",
        run_id="metadata_run",
    )

    metadata = runtime.metadata()

    assert metadata["run_id"] == "metadata_run"
    assert metadata["episode_id"] == (
        mapped_episode.episode_id
    )
    assert metadata["namespace_id"] == (
        mapped_episode.namespace_id
    )
    assert metadata["domain"] == "education"
    assert metadata["evaluation_mode"] == (
        "ungoverned_shared"
    )
    assert metadata["isolated_runtime"] is True
    assert metadata["registered_worker_ids"] == [
        "prof_elena",
        "student_maya",
    ]
    assert metadata["principal_roles"] == {
        "prof_elena": "professor",
        "student_maya": "student",
    }


def test_runtime_require_worker_validates_identifier(
    mapped_episode: MappedEpisode,
    factory_harness: FactoryHarness,
) -> None:
    runtime = factory_harness.factory.build(
        episode=mapped_episode,
        mode="private_only",
        run_id="worker_lookup",
    )

    with pytest.raises(
        ValueError,
        match="principal_id cannot be empty",
    ):
        runtime.require_worker(" ")

    with pytest.raises(
        KeyError,
        match="not registered",
    ):
        runtime.require_worker("unknown_principal")


def test_runtime_exposes_read_only_mappings(
    mapped_episode: MappedEpisode,
    factory_harness: FactoryHarness,
) -> None:
    runtime = factory_harness.factory.build(
        episode=mapped_episode,
        mode="private_only",
        run_id="immutable_mappings",
    )

    with pytest.raises(TypeError):
        runtime.workers["new_agent"] = Agent.worker(
            "new_agent"
        )

    with pytest.raises(TypeError):
        runtime.principal_roles[
            "prof_elena"
        ] = "administrator"

    with pytest.raises(TypeError):
        runtime.policy_context[
            "domain"
        ] = "office"


# ---------------------------------------------------------------------------
# Isolation and reset behaviour
# ---------------------------------------------------------------------------


def test_same_runtime_key_is_reset_when_overwrite_is_enabled(
    mapped_episode: MappedEpisode,
    factory_harness: FactoryHarness,
) -> None:
    first = factory_harness.factory.build(
        episode=mapped_episode,
        mode="governed_shared",
        run_id="reset_run",
    )

    marker = first.paths.runtime_dir / "old_state.marker"
    marker.write_text(
        "stale state",
        encoding="utf-8",
    )
    assert marker.exists()

    second = factory_harness.factory.build(
        episode=mapped_episode,
        mode="governed_shared",
        run_id="reset_run",
    )

    assert second.paths == first.paths
    assert second.paths.runtime_dir.is_dir()
    assert not marker.exists()
    assert first.memory_store is not second.memory_store
    assert (
        first.promotion_request_store
        is not second.promotion_request_store
    )


def test_existing_runtime_is_rejected_when_overwrite_is_disabled(
    mapped_episode: MappedEpisode,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_factory_module,
        "create_memory_governance_runtime",
        lambda **_: object(),
    )

    factory = GateMemRuntimeFactory(
        coordinator_factory=(
            lambda agent, episode: FakeCoordinator(
                agent,
                episode,
            )
        ),
        critic_factory=(
            lambda agent, episode: FakeCritic(
                agent,
                episode,
            )
        ),
        runtime_root=tmp_path,
        overwrite_existing_runtime=False,
    )

    factory.build(
        episode=mapped_episode,
        mode="private_only",
        run_id="same_run",
    )

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        factory.build(
            episode=mapped_episode,
            mode="private_only",
            run_id="same_run",
        )


def test_modes_receive_separate_persistence(
    mapped_episode: MappedEpisode,
    factory_harness: FactoryHarness,
) -> None:
    private_runtime = factory_harness.factory.build(
        episode=mapped_episode,
        mode="private_only",
        run_id="comparison",
    )
    ungoverned_runtime = factory_harness.factory.build(
        episode=mapped_episode,
        mode="ungoverned_shared",
        run_id="comparison",
    )
    governed_runtime = factory_harness.factory.build(
        episode=mapped_episode,
        mode="governed_shared",
        run_id="comparison",
    )

    runtime_dirs = {
        private_runtime.paths.runtime_dir,
        ungoverned_runtime.paths.runtime_dir,
        governed_runtime.paths.runtime_dir,
    }

    assert len(runtime_dirs) == 3
    assert (
        private_runtime.memory_store
        is not ungoverned_runtime.memory_store
    )
    assert (
        ungoverned_runtime.memory_store
        is not governed_runtime.memory_store
    )


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_build_rejects_non_mapped_episode(
    factory_harness: FactoryHarness,
) -> None:
    with pytest.raises(
        TypeError,
        match="MappedEpisode",
    ):
        factory_harness.factory.build(
            episode=object(),  # type: ignore[arg-type]
            mode="private_only",
        )


def test_build_rejects_reserved_principal_id(
    mapped_episode: MappedEpisode,
    factory_harness: FactoryHarness,
) -> None:
    conflicting_episode = mapped_episode.model_copy(
        update={
            "principals": (
                MappedPrincipal(
                    agent_id="gatemem_coordinator",
                    role="student",
                    display_name="Collision",
                ),
            ),
            "relationships": (),
            "turns": (),
        }
    )

    with pytest.raises(
        ValueError,
        match="collide",
    ):
        factory_harness.factory.build(
            episode=conflicting_episode,
            mode="governed_shared",
            run_id="collision",
        )


def test_build_wraps_empty_principal_failure(
    mapped_episode: MappedEpisode,
    factory_harness: FactoryHarness,
) -> None:
    empty_episode = mapped_episode.model_copy(
        update={
            "principals": (),
            "relationships": (),
            "turns": (),
        }
    )

    with pytest.raises(
        GateMemRuntimeFactoryError,
        match="has no principals",
    ):
        factory_harness.factory.build(
            episode=empty_episode,
            mode="governed_shared",
            run_id="empty_principals",
        )


def test_build_wraps_decision_factory_failure(
    mapped_episode: MappedEpisode,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_factory_module,
        "create_memory_governance_runtime",
        lambda **_: object(),
    )

    def failing_coordinator_factory(
        agent: Agent,
        episode: MappedEpisode,
    ) -> Any:
        raise RuntimeError("coordinator construction failed")

    factory = GateMemRuntimeFactory(
        coordinator_factory=failing_coordinator_factory,
        critic_factory=(
            lambda agent, episode: FakeCritic(
                agent,
                episode,
            )
        ),
        runtime_root=tmp_path,
    )

    with pytest.raises(
        GateMemRuntimeFactoryError,
        match="coordinator construction failed",
    ) as exc_info:
        factory.build(
            episode=mapped_episode,
            mode="governed_shared",
            run_id="factory_failure",
        )

    assert isinstance(
        exc_info.value.__cause__,
        RuntimeError,
    )


# ---------------------------------------------------------------------------
# Convenience facade
# ---------------------------------------------------------------------------


def test_build_episode_runtime_forwards_arguments(
    mapped_episode: MappedEpisode,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_runtime = object()
    constructor_calls: list[dict[str, Any]] = []
    build_calls: list[dict[str, Any]] = []

    class FactorySpy:
        def __init__(
            self,
            **kwargs: Any,
        ) -> None:
            constructor_calls.append(
                dict(kwargs)
            )

        def build(
            self,
            **kwargs: Any,
        ) -> Any:
            build_calls.append(dict(kwargs))
            return expected_runtime

    monkeypatch.setattr(
        runtime_factory_module,
        "GateMemRuntimeFactory",
        FactorySpy,
    )

    coordinator_factory = lambda *_: object()
    critic_factory = lambda *_: object()
    retriever_factory = lambda *_: object()
    config = SimpleNamespace(name="config")
    review_gate = SimpleNamespace(name="gate")
    policy_compile = SimpleNamespace(name="policy")
    access_compile = SimpleNamespace(name="access")

    result = build_episode_runtime(
        episode=mapped_episode,
        mode="governed_shared",
        coordinator_factory=coordinator_factory,
        critic_factory=critic_factory,
        run_id="facade_run",
        runtime_root=tmp_path,
        governance_config=config,  # type: ignore[arg-type]
        review_gate=review_gate,
        policy_assignment_compile=(
            policy_compile  # type: ignore[arg-type]
        ),
        access_request_compile=(
            access_compile  # type: ignore[arg-type]
        ),
        memory_retriever_factory=(
            retriever_factory
        ),
        coordinator_agent_id="coord",
        critic_agent_id="critic",
        overwrite_existing_runtime=False,
    )

    assert result is expected_runtime
    assert constructor_calls == [
        {
            "coordinator_factory": (
                coordinator_factory
            ),
            "critic_factory": critic_factory,
            "runtime_root": tmp_path,
            "governance_config": config,
            "review_gate": review_gate,
            "policy_assignment_compile": (
                policy_compile
            ),
            "access_request_compile": (
                access_compile
            ),
            "memory_retriever_factory": (
                retriever_factory
            ),
            "coordinator_agent_id": "coord",
            "critic_agent_id": "critic",
            "overwrite_existing_runtime": False,
        }
    ]
    assert build_calls == [
        {
            "episode": mapped_episode,
            "mode": "governed_shared",
            "run_id": "facade_run",
        }
    ]