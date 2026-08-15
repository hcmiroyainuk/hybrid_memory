from __future__ import annotations

"""
Integration tests for GateMemSystemAgent -> MemoryForgettingService wiring.

These tests deliberately do not test semantic intent extraction or candidate
classification. Those behaviours belong in test_memory_forgetting_service.py.

This file verifies only the adapter boundary:

- GateMemSystemAgent constructs or accepts a forgetting service;
- natural-language deletion turns are routed to that service;
- the deletion command is not stored as an ordinary memory;
- all returned forgotten memory IDs are recorded;
- unresolved/error cases respect strict_lifecycle_events;
- structured ID-based deletion continues to bypass semantic forgetting.
"""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.gatemem import agent as agent_module
from experiments.gatemem.agent import (
    GateMemAgentConfig,
    GateMemIngestionError,
    GateMemPrivateAgent,
)
from experiments.gatemem.mapper import (
    MappedEpisode,
    MappedPrincipal,
    MappedQuery,
    MappedTurn,
)
from experiments.gatemem.runtime_factory import EvaluationMode
from src.memory.entities import Agent
from src.memory.services.memory_forgetting_service import (
    MemoryForgettingService,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeLLMClient:
    """Satisfies the structured LLM protocol without permitting real calls."""

    def invoke_structured(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        raise AssertionError(
            "These adapter integration tests must not call an LLM."
        )


class FakeMapper:
    def __init__(
        self,
        episode: MappedEpisode,
        query: MappedQuery,
    ) -> None:
        self.episode = episode
        self.query = query
        self.episode_calls: list[dict[str, Any]] = []

    def map_episode(
        self,
        payload: dict[str, Any],
    ) -> MappedEpisode:
        self.episode_calls.append(dict(payload))
        return self.episode

    def map_checkpoint(
        self,
        payload: dict[str, Any],
        *,
        episode: MappedEpisode,
    ) -> MappedQuery:
        return self.query


class FakeRuntimeFactory:
    def __init__(
        self,
        runtime: Any,
    ) -> None:
        self.runtime = runtime
        self.calls: list[dict[str, Any]] = []

    def build(
        self,
        *,
        episode: MappedEpisode,
        mode: EvaluationMode,
        run_id: str,
    ) -> Any:
        self.calls.append(
            {
                "episode": episode,
                "mode": mode,
                "run_id": run_id,
            }
        )
        return self.runtime


class FakeMemoryStore:
    def __init__(self) -> None:
        self.memories: dict[str, Any] = {}

    def get_by_id(
        self,
        memory_id: str,
    ) -> Any:
        try:
            return self.memories[memory_id]
        except KeyError as error:
            raise KeyError(memory_id) from error


class FakeMemoryService:
    def __init__(
        self,
        store: FakeMemoryStore,
    ) -> None:
        self.store = store
        self.create_calls: list[dict[str, Any]] = []
        self.deprecate_calls: list[dict[str, Any]] = []
        self._counter = 0

    def create_private_memory(
        self,
        **kwargs: Any,
    ) -> Any:
        self.create_calls.append(dict(kwargs))
        self._counter += 1

        memory_id = f"mem_{self._counter:03d}"
        memory = SimpleNamespace(
            memory_id=memory_id,
            content=kwargs["content"],
            summary=kwargs["summary"],
            metadata=SimpleNamespace(
                owner_agent_id=(
                    kwargs["agent"].agent_id
                ),
                scope="private",
                status="active",
                source_task_id=(
                    kwargs["source_task_id"]
                ),
                source_message_ids=list(
                    kwargs["source_message_ids"]
                ),
            ),
        )
        self.store.memories[memory_id] = memory
        return memory

    def deprecate_memory(
        self,
        **kwargs: Any,
    ) -> Any:
        self.deprecate_calls.append(dict(kwargs))
        memory = self.store.get_by_id(
            kwargs["memory_id"]
        )
        memory.metadata.status = "deprecated"
        return memory


class FakeAccessPolicyService:
    def apply_access_policy(
        self,
        **kwargs: Any,
    ) -> Any:
        return SimpleNamespace(
            metadata=SimpleNamespace(
                scope=kwargs["target_scope"]
            )
        )


class FakeGovernanceRuntime:
    def assign_initial_policy(
        self,
        state: Any,
    ) -> dict[str, Any]:
        return {
            "status": "applied",
            "error_message": None,
        }


class FakeEpisodeRuntime:
    def __init__(
        self,
        episode: MappedEpisode,
    ) -> None:
        self.run_id = "unit_run"
        self.episode = episode
        self.mode = EvaluationMode.PRIVATE_ONLY

        self.workers = {
            "student_maya": Agent.worker(
                agent_id="student_maya",
                name="Maya",
            ),
            "prof_elena": Agent.worker(
                agent_id="prof_elena",
                name="Professor Elena",
            ),
        }
        self.principal_roles = {
            "student_maya": "student",
            "prof_elena": "professor",
        }

        self.coordinator_agent = Agent.coordinator(
            agent_id="gatemem_coordinator"
        )
        self.critic_agent = Agent.critic(
            agent_id="gatemem_critic"
        )
        self.agent_registry = SimpleNamespace(
            agent_ids=(
                "student_maya",
                "prof_elena",
                "gatemem_coordinator",
                "gatemem_critic",
            )
        )

        self.memory_store = FakeMemoryStore()
        self.memory_service = FakeMemoryService(
            self.memory_store
        )
        self.access_policy_service = (
            FakeAccessPolicyService()
        )
        self.governance_runtime = (
            FakeGovernanceRuntime()
        )
        self.policy_context = (
            episode.to_policy_context()
        )

    def require_worker(
        self,
        principal_id: str,
    ) -> Agent:
        return self.workers[principal_id]

    def metadata(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode.episode_id,
            "evaluation_mode": self.mode.value,
        }


class FakeAnswerService:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            no_memory_message="NO MEMORY",
            refusal_message="REFUSED",
        )


class FakeForgettingService:
    def __init__(
        self,
        *,
        forgotten_memory_ids: (
            list[str] | None
        ) = None,
        warnings: list[str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.forgotten_memory_ids = list(
            forgotten_memory_ids or []
        )
        self.warnings = list(
            warnings or []
        )
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def forget(
        self,
        **kwargs: Any,
    ) -> Any:
        self.calls.append(dict(kwargs))

        if self.error is not None:
            raise self.error

        return SimpleNamespace(
            target_fact=(
                "The earlier Project Atlas stipend "
                "was 2,200 USD."
            ),
            forgotten_memory_ids=list(
                self.forgotten_memory_ids
            ),
            warnings=list(self.warnings),
        )


@dataclass
class Harness:
    agent: GateMemPrivateAgent
    runtime: FakeEpisodeRuntime
    forgetting_service: FakeForgettingService
    mapper: FakeMapper
    runtime_factory: FakeRuntimeFactory


# ---------------------------------------------------------------------------
# Fixtures and builders
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def stub_gatemem_base_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Keep the tests focused on the custom adapter rather than GateMem internals.
    """

    def fake_init(
        self: Any,
        **kwargs: Any,
    ) -> None:
        self._base_init_kwargs = dict(kwargs)

    def fake_reset(
        self: Any,
        episode: dict[str, Any],
    ) -> None:
        self._base_reset_episode = episode

    monkeypatch.setattr(
        agent_module.BaseMemoryAgent,
        "__init__",
        fake_init,
    )
    monkeypatch.setattr(
        agent_module.BaseMemoryAgent,
        "reset",
        fake_reset,
    )


def make_episode(
    *,
    deletion_text: str = (
        "Deletion request: remove the earlier "
        "Project Atlas stipend amount 2,200 USD. "
        "Keep 2,350 USD as the current amount."
    ),
) -> MappedEpisode:
    return MappedEpisode(
        episode_id="education_episode_001",
        domain="education",
        principals=(
            MappedPrincipal(
                agent_id="student_maya",
                role="student",
                display_name="Maya",
            ),
            MappedPrincipal(
                agent_id="prof_elena",
                role="professor",
                display_name="Professor Elena",
            ),
        ),
        relationships=(),
        turns=(
            MappedTurn(
                episode_id="education_episode_001",
                domain="education",
                turn_id="t001",
                sequence_index=0,
                speaker_agent_id="student_maya",
                speaker_role="student",
                text=(
                    "The provisional Project Atlas "
                    "stipend is 2,200 USD."
                ),
            ),
            MappedTurn(
                episode_id="education_episode_001",
                domain="education",
                turn_id="t002",
                sequence_index=1,
                speaker_agent_id="student_maya",
                speaker_role="student",
                text=deletion_text,
            ),
        ),
    )


def make_query() -> MappedQuery:
    return MappedQuery(
        checkpoint_id="cp_001",
        episode_id="education_episode_001",
        as_of_turn_id="t002",
        requester_agent_id="student_maya",
        requester_role="student",
        query_text=(
            "What is the current Project Atlas "
            "stipend?"
        ),
    )


def make_raw_episode() -> dict[str, Any]:
    return {
        "episode_id": "education_episode_001",
        "domain": "education",
        "entities": {
            "principals": [],
            "relationships": [],
        },
        "turns": [],
    }


def make_turn(
    turn_id: str,
    *,
    memory_ops: (
        dict[str, Any]
        | list[dict[str, Any]]
        | None
    ) = None,
) -> Any:
    speaker_id = (
        "student_maya"
        if turn_id in {"t001", "t002"}
        else "prof_elena"
    )

    return SimpleNamespace(
        turn_id=turn_id,
        speaker_principal_id=speaker_id,
        speaker_role=(
            "student"
            if speaker_id == "student_maya"
            else "professor"
        ),
        record_refs=[],
        memory_ops=memory_ops,
    )


def build_harness(
    *,
    strict: bool = True,
    forgetting_service: (
        FakeForgettingService | None
    ) = None,
    episode: MappedEpisode | None = None,
) -> Harness:
    episode = episode or make_episode()
    query = make_query()
    runtime = FakeEpisodeRuntime(episode)
    mapper = FakeMapper(episode, query)
    runtime_factory = FakeRuntimeFactory(runtime)

    service = (
        forgetting_service
        or FakeForgettingService(
            forgotten_memory_ids=[
                "mem_001"
            ]
        )
    )

    agent = GateMemPrivateAgent(
        config=GateMemAgentConfig(
            mode=EvaluationMode.PRIVATE_ONLY,
            run_id="unit_run",
            runtime_root=Path(
                "unused/unit-runtime"
            ),
            strict_turn_order=True,
            strict_lifecycle_events=strict,
            expose_memory_audit=False,
            expose_debug=False,
        ),
        mapper=mapper,
        runtime_factory=runtime_factory,
        answer_service=FakeAnswerService(),
        forgetting_service=service,
        llm_client=FakeLLMClient(),
    )

    return Harness(
        agent=agent,
        runtime=runtime,
        forgetting_service=service,
        mapper=mapper,
        runtime_factory=runtime_factory,
    )


def reset_agent(
    harness: Harness,
) -> None:
    harness.agent.reset(
        make_raw_episode()
    )


# ---------------------------------------------------------------------------
# Constructor wiring
# ---------------------------------------------------------------------------


def test_constructor_keeps_injected_forgetting_service() -> None:
    service = FakeForgettingService()
    harness = build_harness(
        forgetting_service=service
    )

    assert (
        harness.agent.forgetting_service
        is service
    )


def test_constructor_builds_default_forgetting_service_when_not_injected() -> None:
    episode = make_episode()
    runtime = FakeEpisodeRuntime(episode)

    agent = GateMemPrivateAgent(
        config=GateMemAgentConfig(
            mode=EvaluationMode.PRIVATE_ONLY,
            run_id="unit_run",
            runtime_root=Path(
                "unused/unit-runtime"
            ),
            expose_memory_audit=False,
            expose_debug=False,
        ),
        mapper=FakeMapper(
            episode,
            make_query(),
        ),
        runtime_factory=FakeRuntimeFactory(
            runtime
        ),
        answer_service=FakeAnswerService(),
        llm_client=FakeLLMClient(),
    )

    assert isinstance(
        agent.forgetting_service,
        MemoryForgettingService,
    )


# ---------------------------------------------------------------------------
# Natural-language forgetting route
# ---------------------------------------------------------------------------


def test_natural_language_delete_calls_forgetting_service_and_records_all_ids(
) -> None:
    service = FakeForgettingService(
        forgotten_memory_ids=[
            "mem_001",
            "mem_derived",
        ],
        warnings=[
            "One unrelated candidate was preserved."
        ],
    )
    harness = build_harness(
        forgetting_service=service
    )
    reset_agent(harness)

    harness.agent.ingest(
        make_turn("t001")
    )

    # Represent a second active copy that the semantic service may resolve.
    harness.runtime.memory_store.memories[
        "mem_derived"
    ] = SimpleNamespace(
        memory_id="mem_derived",
        content=(
            "Earlier reports listed "
            "2,200 USD."
        ),
        summary="",
        metadata=SimpleNamespace(
            owner_agent_id="prof_elena",
            status="active",
            scope="private",
            source_message_ids=["t001"],
        ),
    )

    harness.agent.ingest(
        make_turn("t002")
    )

    assert len(service.calls) == 1
    call = service.calls[0]

    assert call["request_text"] == (
        harness.runtime.episode
        .require_turn("t002")
        .text
    )
    assert (
        call["requesting_agent_id"]
        == "student_maya"
    )
    assert call["source_turn_id"] == "t002"
    assert call["workers"] is (
        harness.runtime.workers
    )
    assert call["memory_service"] is (
        harness.runtime.memory_service
    )
    assert call["memory_store"] is (
        harness.runtime.memory_store
    )

    # Only t001 is an ordinary memory. The deletion command is not stored.
    assert len(
        harness.runtime
        .memory_service
        .create_calls
    ) == 1

    record = (
        harness.agent
        .ingestion_records[1]
    )
    assert record.event_type == "delete"
    assert set(record.memory_ids) == {
        "mem_001",
        "mem_derived",
    }
    assert (
        "One unrelated candidate was preserved."
        in record.warnings
    )

    assert set(
        harness.agent
        .turn_memory_ids["t002"]
    ) == {
        "mem_001",
        "mem_derived",
    }

    # The semantic service owns the deprecation operation; the adapter must not
    # repeat the same mutation after the service returns.
    assert (
        harness.runtime
        .memory_service
        .deprecate_calls
        == []
    )


def test_natural_language_delete_with_no_resolved_memory_becomes_noop_when_non_strict(
) -> None:
    service = FakeForgettingService(
        forgotten_memory_ids=[]
    )
    harness = build_harness(
        strict=False,
        forgetting_service=service,
    )
    reset_agent(harness)

    harness.agent.ingest(
        make_turn("t001")
    )
    harness.agent.ingest(
        make_turn("t002")
    )

    assert len(service.calls) == 1
    assert len(
        harness.runtime
        .memory_service
        .create_calls
    ) == 1

    record = (
        harness.agent
        .ingestion_records[1]
    )
    assert record.event_type == "noop"
    assert record.memory_ids == ()
    assert (
        harness.agent
        .turn_memory_ids["t002"]
        == ()
    )
    assert any(
        "no active memory"
        in warning.lower()
        for warning in record.warnings
    )


def test_natural_language_delete_with_no_resolved_memory_raises_when_strict(
) -> None:
    service = FakeForgettingService(
        forgotten_memory_ids=[]
    )
    harness = build_harness(
        strict=True,
        forgetting_service=service,
    )
    reset_agent(harness)

    harness.agent.ingest(
        make_turn("t001")
    )

    with pytest.raises(
        GateMemIngestionError,
        match="no active memory",
    ):
        harness.agent.ingest(
            make_turn("t002")
        )

    assert len(
        harness.runtime
        .memory_service
        .create_calls
    ) == 1


def test_forgetting_service_error_becomes_noop_when_non_strict() -> None:
    service = FakeForgettingService(
        error=RuntimeError(
            "semantic forgetting failed"
        )
    )
    harness = build_harness(
        strict=False,
        forgetting_service=service,
    )
    reset_agent(harness)

    harness.agent.ingest(
        make_turn("t001")
    )
    harness.agent.ingest(
        make_turn("t002")
    )

    assert len(
        harness.runtime
        .memory_service
        .create_calls
    ) == 1

    record = (
        harness.agent
        .ingestion_records[1]
    )
    assert record.event_type == "noop"
    assert any(
        "semantic forgetting failed"
        in warning
        for warning in record.warnings
    )


# ---------------------------------------------------------------------------
# Structured deletion remains deterministic
# ---------------------------------------------------------------------------


def test_structured_delete_bypasses_semantic_forgetting_service() -> None:
    service = FakeForgettingService(
        error=AssertionError(
            "Structured deletion must not call "
            "semantic forgetting."
        )
    )
    harness = build_harness(
        forgetting_service=service
    )
    reset_agent(harness)

    harness.agent.ingest(
        make_turn("t001")
    )
    harness.agent.ingest(
        make_turn(
            "t002",
            memory_ops={
                "operation": "delete",
                "target_turn_id": "t001",
            },
        )
    )

    assert service.calls == []

    assert len(
        harness.runtime
        .memory_service
        .deprecate_calls
    ) == 1
    delete_call = (
        harness.runtime
        .memory_service
        .deprecate_calls[0]
    )
    assert (
        delete_call["memory_id"]
        == "mem_001"
    )
    assert (
        delete_call["agent"].agent_id
        == "student_maya"
    )

    record = (
        harness.agent
        .ingestion_records[1]
    )
    assert record.event_type == "delete"
    assert record.memory_ids == (
        "mem_001",
    )