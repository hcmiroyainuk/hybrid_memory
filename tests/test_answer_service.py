from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from experiments.gatemem.answer_service import (
    GateMemAnswerConfig,
    GateMemAnswerGenerationError,
    GateMemAnswerResult,
    GateMemAnswerRetrievalError,
    GateMemAnswerService,
    GateMemAnswerValidationError,
)
from experiments.gatemem.mapper import (
    MappedEpisode,
    MappedPrincipal,
    MappedQuery,
    MappedTurn,
)
from experiments.gatemem.runtime_factory import (
    EvaluationMode,
    GateMemEpisodeRuntime,
    GateMemRuntimePaths,
)
from src.llm import AgentAnswer
from src.memory.entities import Agent


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakePromptTemplate:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def build_answer_prompt(
        self,
        *,
        task: str,
        task_input: str | None = None,
        accessible_memory_context: str | None = None,
        context_sections: dict[str, Any] | None = None,
    ) -> str:
        call = {
            "task": task,
            "task_input": task_input,
            "accessible_memory_context": (
                accessible_memory_context
            ),
            "context_sections": dict(
                context_sections or {}
            ),
        }
        self.calls.append(call)

        return (
            f"TASK:\n{task}\n\n"
            f"MEMORY:\n{accessible_memory_context or ''}\n\n"
            f"CONTEXT:\n{context_sections or {}}"
        )


class FakeLLMClient:
    def __init__(
        self,
        response: Any,
        *,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def invoke_structured(
        self,
        prompt: str,
        schema_model: type[Any],
        *,
        apply_parser_normalization: bool = True,
    ) -> Any:
        self.calls.append(
            {
                "prompt": prompt,
                "schema_model": schema_model,
                "apply_parser_normalization": (
                    apply_parser_normalization
                ),
            }
        )

        if self.error is not None:
            raise self.error

        return self.response


class FakeMemoryService:
    def __init__(
        self,
        *,
        accessible: list[Any] | None = None,
        semantic_results: list[Any] | None = None,
        memories_by_id: dict[str, Any] | None = None,
        memory_retriever: Any | None = None,
        list_error: Exception | None = None,
        retrieve_error: Exception | None = None,
        get_errors: dict[str, Exception] | None = None,
    ) -> None:
        self.accessible = list(accessible or [])
        self.semantic_results = list(
            semantic_results or []
        )
        self.memories_by_id = dict(
            memories_by_id
            or {
                memory.memory_id: memory
                for memory in self.accessible
            }
        )
        self.memory_retriever = memory_retriever
        self.list_error = list_error
        self.retrieve_error = retrieve_error
        self.get_errors = dict(get_errors or {})

        self.events: list[str] = []
        self.list_calls: list[dict[str, Any]] = []
        self.retrieve_calls: list[
            dict[str, Any]
        ] = []
        self.get_calls: list[
            tuple[Agent, str]
        ] = []

    def list_accessible_memories(
        self,
        agent: Agent,
        include_private: bool = True,
        include_shared: bool = True,
        active_only: bool = True,
    ) -> list[Any]:
        self.events.append("list")
        self.list_calls.append(
            {
                "agent": agent,
                "include_private": include_private,
                "include_shared": include_shared,
                "active_only": active_only,
            }
        )

        if self.list_error is not None:
            raise self.list_error

        return list(self.accessible)

    def retrieve_memories(
        self,
        agent: Agent,
        query: str,
        top_k: int = 5,
        include_private: bool = True,
        include_shared: bool = True,
    ) -> list[Any]:
        self.events.append("retrieve")
        self.retrieve_calls.append(
            {
                "agent": agent,
                "query": query,
                "top_k": top_k,
                "include_private": include_private,
                "include_shared": include_shared,
            }
        )

        if self.retrieve_error is not None:
            raise self.retrieve_error

        return list(self.semantic_results)

    def get_memory(
        self,
        agent: Agent,
        memory_id: str,
    ) -> Any:
        self.events.append("get")
        self.get_calls.append((agent, memory_id))

        if memory_id in self.get_errors:
            raise self.get_errors[memory_id]

        try:
            return self.memories_by_id[memory_id]
        except KeyError as error:
            raise KeyError(
                f"Unknown memory: {memory_id}"
            ) from error


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def make_memory(
    memory_id: str,
    *,
    content: str,
    owner_agent_id: str = "student_maya",
    episode_id: str | None = (
        "education_episode_001"
    ),
    turn_id: str | None = "t001",
    sequence_index: int | None = 0,
    scope: str = "private",
    source_message_ids: list[str] | None = None,
    source_task_id: str | None = None,
    summary: str = "",
    importance: float = 0.5,
    confidence: float = 1.0,
    readable_by: list[str] | None = None,
) -> Any:
    metadata = SimpleNamespace(
        owner_agent_id=owner_agent_id,
        episode_id=episode_id,
        turn_id=turn_id,
        sequence_index=sequence_index,
        scope=scope,
        source_message_ids=(
            list(source_message_ids)
            if source_message_ids is not None
            else ([turn_id] if turn_id else [])
        ),
        source_task_id=source_task_id,
        importance=importance,
        confidence=confidence,
        readable_by=list(readable_by or []),
    )

    return SimpleNamespace(
        memory_id=memory_id,
        content=content,
        summary=summary,
        metadata=metadata,
    )


def answered(
    answer: str = "20 August",
    *,
    used_memory_ids: list[str] | None = None,
) -> AgentAnswer:
    return AgentAnswer(
        status="answered",
        answer=answer,
        reasoning=(
            "The authorised memory directly states "
            "the requested value."
        ),
        confidence=0.9,
        missing_information=[],
        used_memory_ids=list(
            used_memory_ids or []
        ),
        supporting_source_ids=[],
        contributing_agent_ids=[],
    )


def insufficient() -> AgentAnswer:
    return AgentAnswer(
        status="insufficient_evidence",
        answer=None,
        reasoning=(
            "The authorised context does not contain "
            "the requested information."
        ),
        confidence=0.8,
        missing_information=[
            "The requested information",
        ],
        used_memory_ids=[],
        supporting_source_ids=[],
        contributing_agent_ids=[],
    )


def refused() -> AgentAnswer:
    return AgentAnswer(
        status="refused",
        answer=None,
        reasoning=(
            "The request must be declined under "
            "the supplied authorised context."
        ),
        confidence=0.95,
        missing_information=[],
        used_memory_ids=[],
        supporting_source_ids=[],
        contributing_agent_ids=[],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def episode() -> MappedEpisode:
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
                    "My dissertation deadline is "
                    "20 August."
                ),
            ),
            MappedTurn(
                episode_id="education_episode_001",
                domain="education",
                turn_id="t002",
                sequence_index=1,
                speaker_agent_id="prof_elena",
                speaker_role="professor",
                text=(
                    "The supervision meeting is "
                    "on Monday."
                ),
            ),
            MappedTurn(
                episode_id="education_episode_001",
                domain="education",
                turn_id="t003",
                sequence_index=2,
                speaker_agent_id="prof_elena",
                speaker_role="professor",
                text=(
                    "A future confidential fact."
                ),
            ),
        ),
    )


@pytest.fixture
def query() -> MappedQuery:
    return MappedQuery(
        checkpoint_id="cp_001",
        episode_id="education_episode_001",
        as_of_turn_id="t002",
        requester_agent_id="student_maya",
        requester_role="student",
        query_text=(
            "What is my dissertation deadline?"
        ),
    )


@pytest.fixture
def current_memory() -> Any:
    return make_memory(
        "education_episode_001:t001",
        content=(
            "Maya's dissertation deadline is "
            "20 August."
        ),
        summary="Dissertation deadline",
        source_message_ids=["t001"],
    )


@pytest.fixture
def future_memory() -> Any:
    return make_memory(
        "education_episode_001:t003",
        content="FUTURE_SECRET",
        owner_agent_id="prof_elena",
        turn_id="t003",
        sequence_index=2,
        scope="shared",
        source_message_ids=["t003"],
    )


@pytest.fixture
def other_episode_memory() -> Any:
    return make_memory(
        "education_episode_999:t001",
        content="OTHER_EPISODE_SECRET",
        episode_id="education_episode_999",
        turn_id="t001",
        sequence_index=0,
    )


def build_runtime(
    *,
    tmp_path: Path,
    episode: MappedEpisode,
    memory_service: FakeMemoryService,
) -> GateMemEpisodeRuntime:
    maya = Agent.worker(
        agent_id="student_maya",
        name="Maya",
    )
    elena = Agent.worker(
        agent_id="prof_elena",
        name="Professor Elena",
    )

    runtime_dir = tmp_path / "runtime"
    paths = GateMemRuntimePaths(
        runtime_dir=runtime_dir,
        memories_path=(
            runtime_dir / "memories.json"
        ),
        operation_logs_path=(
            runtime_dir / "operation_logs.json"
        ),
        promotion_requests_path=(
            runtime_dir
            / "promotion_requests.json"
        ),
    )

    placeholder = object()

    return GateMemEpisodeRuntime(
        run_id="test_run",
        episode=episode,
        mode=EvaluationMode.GOVERNED_SHARED,
        paths=paths,
        workers={
            maya.agent_id: maya,
            elena.agent_id: elena,
        },
        principal_roles={
            "student_maya": "student",
            "prof_elena": "professor",
        },
        coordinator_agent=placeholder,  # type: ignore[arg-type]
        critic_agent=placeholder,  # type: ignore[arg-type]
        agent_registry=placeholder,  # type: ignore[arg-type]
        memory_store=placeholder,  # type: ignore[arg-type]
        operation_log_store=placeholder,  # type: ignore[arg-type]
        promotion_request_store=placeholder,  # type: ignore[arg-type]
        permission_service=placeholder,  # type: ignore[arg-type]
        memory_retriever=(
            memory_service.memory_retriever
        ),
        memory_service=memory_service,  # type: ignore[arg-type]
        access_policy_service=placeholder,  # type: ignore[arg-type]
        promotion_service=placeholder,  # type: ignore[arg-type]
        access_policy_gateway=placeholder,  # type: ignore[arg-type]
        sharing_gateway=placeholder,  # type: ignore[arg-type]
        coordinator=placeholder,  # type: ignore[arg-type]
        critic=placeholder,  # type: ignore[arg-type]
        governance_runtime=placeholder,  # type: ignore[arg-type]
        policy_context=(
            episode.to_policy_context()
        ),
    )


def build_service(
    *,
    llm_response: Any,
    config: GateMemAnswerConfig | None = None,
    llm_error: Exception | None = None,
) -> tuple[
    GateMemAnswerService,
    FakeLLMClient,
    FakePromptTemplate,
    list[tuple[Agent, str, str]],
]:
    llm_client = FakeLLMClient(
        llm_response,
        error=llm_error,
    )
    prompt_template = FakePromptTemplate()
    factory_calls: list[
        tuple[Agent, str, str]
    ] = []

    def prompt_factory(
        agent: Agent,
        role: str,
        domain: str,
    ) -> Any:
        factory_calls.append(
            (agent, role, domain)
        )
        return prompt_template

    service = GateMemAnswerService(
        llm_client=llm_client,  # type: ignore[arg-type]
        config=config,
        worker_prompt_factory=prompt_factory,
    )

    return (
        service,
        llm_client,
        prompt_template,
        factory_calls,
    )


# ---------------------------------------------------------------------------
# Configuration and result-model tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_k": 0}, "top_k"),
        (
            {"max_context_chars": 0},
            "max_context_chars",
        ),
        (
            {"max_content_chars_per_memory": 0},
            "max_content_chars_per_memory",
        ),
        (
            {"max_summary_chars_per_memory": 0},
            "max_summary_chars_per_memory",
        ),
        (
            {"no_memory_confidence": 1.5},
            "no_memory_confidence",
        ),
        (
            {
                "include_private": False,
                "include_shared": False,
            },
            "At least one",
        ),
        (
            {"no_memory_message": " "},
            "no_memory_message",
        ),
        (
            {"refusal_message": ""},
            "refusal_message",
        ),
    ],
)
def test_answer_config_rejects_invalid_values(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=message,
    ):
        GateMemAnswerConfig(**kwargs)


def test_answer_service_requires_llm_client() -> None:
    with pytest.raises(
        ValueError,
        match="llm_client",
    ):
        GateMemAnswerService(
            llm_client=None,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("action", "answer_value", "agent_answer"),
    [
        (
            "answer",
            None,
            insufficient(),
        ),
        (
            "refuse",
            None,
            insufficient(),
        ),
        (
            "no_memory",
            None,
            refused(),
        ),
    ],
)
def test_answer_result_enforces_action_contract(
    action: str,
    answer_value: str | None,
    agent_answer: AgentAnswer,
) -> None:
    with pytest.raises(ValidationError):
        GateMemAnswerResult(
            checkpoint_id="cp",
            episode_id="episode",
            requester_agent_id="agent",
            action=action,
            answer=answer_value,
            agent_answer=agent_answer,
        )


def test_answer_result_payload_contains_only_runtime_output() -> None:
    result = GateMemAnswerResult(
        checkpoint_id="cp_001",
        episode_id="education_episode_001",
        requester_agent_id="student_maya",
        action="answer",
        answer="20 August",
        agent_answer=answered(
            "20 August",
            used_memory_ids=[
                "education_episode_001:t001"
            ],
        ),
        retrieved_record_ids=[
            "education_episode_001:t001"
        ],
        used_record_ids=[
            "education_episode_001:t001"
        ],
    )

    payload = result.to_gatemem_payload()

    assert payload["action"] == "answer"
    assert payload["answer"] == "20 August"
    assert payload["used_record_ids"] == [
        "education_episode_001:t001"
    ]
    assert set(payload) == {
        "action",
        "answer",
        "answer_structured",
        "used_record_ids",
    }
    assert "expected_action" not in payload
    assert "judge_spec" not in payload
    assert "leak_targets" not in payload


def test_non_answer_payload_uses_supplied_messages() -> None:
    no_memory_result = GateMemAnswerResult(
        checkpoint_id="cp",
        episode_id="episode",
        requester_agent_id="agent",
        action="no_memory",
        answer=None,
        agent_answer=insufficient(),
    )
    refusal_result = GateMemAnswerResult(
        checkpoint_id="cp",
        episode_id="episode",
        requester_agent_id="agent",
        action="refuse",
        answer=None,
        agent_answer=refused(),
    )

    assert no_memory_result.to_gatemem_payload(
        no_memory_message="NO AUTHORISED EVIDENCE",
    )["answer"] == "NO AUTHORISED EVIDENCE"

    assert refusal_result.to_gatemem_payload(
        refusal_message="DECLINED",
    )["answer"] == "DECLINED"


# ---------------------------------------------------------------------------
# Runtime/query validation
# ---------------------------------------------------------------------------


def test_answer_rejects_non_runtime(
    query: MappedQuery,
) -> None:
    service, _, _, _ = build_service(
        llm_response=answered()
    )

    with pytest.raises(
        TypeError,
        match="GateMemEpisodeRuntime",
    ):
        service.answer(
            runtime=object(),  # type: ignore[arg-type]
            query=query,
        )


def test_answer_rejects_non_query(
    tmp_path: Path,
    episode: MappedEpisode,
) -> None:
    memory_service = FakeMemoryService()
    runtime = build_runtime(
        tmp_path=tmp_path,
        episode=episode,
        memory_service=memory_service,
    )
    service, _, _, _ = build_service(
        llm_response=answered()
    )

    with pytest.raises(
        TypeError,
        match="MappedQuery",
    ):
        service.answer(
            runtime=runtime,
            query=object(),  # type: ignore[arg-type]
        )


def test_answer_rejects_mismatched_episode(
    tmp_path: Path,
    episode: MappedEpisode,
    query: MappedQuery,
) -> None:
    runtime = build_runtime(
        tmp_path=tmp_path,
        episode=episode,
        memory_service=FakeMemoryService(),
    )
    service, _, _, _ = build_service(
        llm_response=answered()
    )
    wrong_query = query.model_copy(
        update={
            "episode_id": "another_episode",
        }
    )

    with pytest.raises(
        GateMemAnswerValidationError,
        match="does not match",
    ):
        service.answer(
            runtime=runtime,
            query=wrong_query,
        )


def test_answer_rejects_mismatched_requester_role(
    tmp_path: Path,
    episode: MappedEpisode,
    query: MappedQuery,
) -> None:
    runtime = build_runtime(
        tmp_path=tmp_path,
        episode=episode,
        memory_service=FakeMemoryService(),
    )
    service, _, _, _ = build_service(
        llm_response=answered()
    )
    wrong_query = query.model_copy(
        update={
            "requester_role": "professor",
        }
    )

    with pytest.raises(
        GateMemAnswerValidationError,
        match="role",
    ):
        service.answer(
            runtime=runtime,
            query=wrong_query,
        )


# ---------------------------------------------------------------------------
# Permission-first and temporal retrieval
# ---------------------------------------------------------------------------


def test_retrieval_lists_accessible_memories_before_semantic_search(
    tmp_path: Path,
    episode: MappedEpisode,
    query: MappedQuery,
    current_memory: Any,
) -> None:
    memory_service = FakeMemoryService(
        accessible=[current_memory],
        semantic_results=[current_memory],
        memory_retriever=object(),
    )
    runtime = build_runtime(
        tmp_path=tmp_path,
        episode=episode,
        memory_service=memory_service,
    )
    service, _, _, _ = build_service(
        llm_response=answered()
    )
    requester = runtime.require_worker(
        "student_maya"
    )

    memories, warnings = (
        service.retrieve_authorised_memories(
            runtime=runtime,
            query=query,
            requester=requester,
        )
    )

    assert memory_service.events[:2] == [
        "list",
        "retrieve",
    ]
    assert memories == [current_memory]
    assert warnings == []


def test_semantic_results_are_intersected_with_authorised_set(
    tmp_path: Path,
    episode: MappedEpisode,
    query: MappedQuery,
    current_memory: Any,
) -> None:
    second_allowed = make_memory(
        "education_episode_001:t002",
        content=(
            "Professor Elena confirms the deadline "
            "arrangements."
        ),
        owner_agent_id="prof_elena",
        turn_id="t002",
        sequence_index=1,
        scope="shared",
    )
    unauthorised = make_memory(
        "education_episode_001:secret",
        content="UNAUTHORISED_SECRET",
        owner_agent_id="prof_elena",
        turn_id="t001",
        sequence_index=0,
    )

    memory_service = FakeMemoryService(
        accessible=[
            current_memory,
            second_allowed,
        ],
        semantic_results=[
            unauthorised,
            second_allowed,
            current_memory,
        ],
        memory_retriever=object(),
    )
    runtime = build_runtime(
        tmp_path=tmp_path,
        episode=episode,
        memory_service=memory_service,
    )
    service, _, _, _ = build_service(
        llm_response=answered()
    )

    memories, _ = (
        service.retrieve_authorised_memories(
            runtime=runtime,
            query=query,
            requester=runtime.require_worker(
                "student_maya"
            ),
        )
    )

    assert memories == [
        second_allowed,
        current_memory,
    ]
    assert unauthorised not in memories


def test_temporal_and_episode_boundaries_fail_closed(
    tmp_path: Path,
    episode: MappedEpisode,
    query: MappedQuery,
    current_memory: Any,
    future_memory: Any,
    other_episode_memory: Any,
) -> None:
    unknown_boundary = make_memory(
        "unscoped_memory",
        content="UNKNOWN_BOUNDARY_SECRET",
        episode_id=None,
        turn_id=None,
        sequence_index=None,
        source_message_ids=[],
    )

    memory_service = FakeMemoryService(
        accessible=[
            current_memory,
            future_memory,
            other_episode_memory,
            unknown_boundary,
        ]
    )
    runtime = build_runtime(
        tmp_path=tmp_path,
        episode=episode,
        memory_service=memory_service,
    )
    service, _, _, _ = build_service(
        llm_response=answered(),
        config=GateMemAnswerConfig(
            use_semantic_retrieval=False,
        ),
    )

    memories, warnings = (
        service.retrieve_authorised_memories(
            runtime=runtime,
            query=query,
            requester=runtime.require_worker(
                "student_maya"
            ),
        )
    )

    assert memories == [current_memory]
    warning_text = "\n".join(warnings)
    assert "future memory" in warning_text
    assert "another episode" in warning_text
    assert "no resolvable" in warning_text


def test_boundary_can_be_resolved_from_source_fields(
    tmp_path: Path,
    episode: MappedEpisode,
    query: MappedQuery,
) -> None:
    memory = make_memory(
        "memory_with_source_fields",
        content="The deadline is 20 August.",
        episode_id=None,
        turn_id=None,
        sequence_index=None,
        source_task_id="education_episode_001",
        source_message_ids=["t001"],
    )

    memory_service = FakeMemoryService(
        accessible=[memory]
    )
    runtime = build_runtime(
        tmp_path=tmp_path,
        episode=episode,
        memory_service=memory_service,
    )
    service, _, _, _ = build_service(
        llm_response=answered(),
        config=GateMemAnswerConfig(
            use_semantic_retrieval=False,
        ),
    )

    memories, warnings = (
        service.retrieve_authorised_memories(
            runtime=runtime,
            query=query,
            requester=runtime.require_worker(
                "student_maya"
            ),
        )
    )

    assert memories == [memory]
    assert warnings == []


def test_lexical_fallback_ranks_relevant_authorised_memory_first(
    tmp_path: Path,
    episode: MappedEpisode,
    query: MappedQuery,
) -> None:
    unrelated = make_memory(
        "education_episode_001:t001",
        content="The library closes at six.",
        turn_id="t001",
        sequence_index=0,
    )
    relevant = make_memory(
        "education_episode_001:t002",
        content=(
            "The dissertation deadline is "
            "20 August."
        ),
        turn_id="t002",
        sequence_index=1,
    )

    memory_service = FakeMemoryService(
        accessible=[unrelated, relevant]
    )
    runtime = build_runtime(
        tmp_path=tmp_path,
        episode=episode,
        memory_service=memory_service,
    )
    service, _, _, _ = build_service(
        llm_response=answered(),
        config=GateMemAnswerConfig(
            use_semantic_retrieval=False,
            top_k=2,
        ),
    )

    memories, _ = (
        service.retrieve_authorised_memories(
            runtime=runtime,
            query=query,
            requester=runtime.require_worker(
                "student_maya"
            ),
        )
    )

    assert memories[0] is relevant
    assert memories[1] is unrelated


def test_semantic_failure_uses_lexical_fallback_when_enabled(
    tmp_path: Path,
    episode: MappedEpisode,
    query: MappedQuery,
    current_memory: Any,
) -> None:
    memory_service = FakeMemoryService(
        accessible=[current_memory],
        memory_retriever=object(),
        retrieve_error=RuntimeError(
            "vector index unavailable"
        ),
    )
    runtime = build_runtime(
        tmp_path=tmp_path,
        episode=episode,
        memory_service=memory_service,
    )
    service, _, _, _ = build_service(
        llm_response=answered(),
        config=GateMemAnswerConfig(
            allow_lexical_fallback=True,
        ),
    )

    memories, warnings = (
        service.retrieve_authorised_memories(
            runtime=runtime,
            query=query,
            requester=runtime.require_worker(
                "student_maya"
            ),
        )
    )

    assert memories == [current_memory]
    assert any(
        "lexical fallback" in warning
        for warning in warnings
    )


def test_semantic_failure_raises_when_fallback_disabled(
    tmp_path: Path,
    episode: MappedEpisode,
    query: MappedQuery,
    current_memory: Any,
) -> None:
    memory_service = FakeMemoryService(
        accessible=[current_memory],
        memory_retriever=object(),
        retrieve_error=RuntimeError(
            "vector index unavailable"
        ),
    )
    runtime = build_runtime(
        tmp_path=tmp_path,
        episode=episode,
        memory_service=memory_service,
    )
    service, _, _, _ = build_service(
        llm_response=answered(),
        config=GateMemAnswerConfig(
            allow_lexical_fallback=False,
        ),
    )

    with pytest.raises(
        GateMemAnswerRetrievalError,
        match="fallback is disabled",
    ):
        service.retrieve_authorised_memories(
            runtime=runtime,
            query=query,
            requester=runtime.require_worker(
                "student_maya"
            ),
        )


def test_missing_semantic_retriever_raises_when_fallback_disabled(
    tmp_path: Path,
    episode: MappedEpisode,
    query: MappedQuery,
    current_memory: Any,
) -> None:
    runtime = build_runtime(
        tmp_path=tmp_path,
        episode=episode,
        memory_service=FakeMemoryService(
            accessible=[current_memory],
            memory_retriever=None,
        ),
    )
    service, _, _, _ = build_service(
        llm_response=answered(),
        config=GateMemAnswerConfig(
            use_semantic_retrieval=True,
            allow_lexical_fallback=False,
        ),
    )

    with pytest.raises(
        GateMemAnswerRetrievalError,
        match="no MemoryRetriever",
    ):
        service.retrieve_authorised_memories(
            runtime=runtime,
            query=query,
            requester=runtime.require_worker(
                "student_maya"
            ),
        )


def test_accessible_memory_listing_failure_is_wrapped(
    tmp_path: Path,
    episode: MappedEpisode,
    query: MappedQuery,
) -> None:
    runtime = build_runtime(
        tmp_path=tmp_path,
        episode=episode,
        memory_service=FakeMemoryService(
            list_error=RuntimeError(
                "store unavailable"
            )
        ),
    )
    service, _, _, _ = build_service(
        llm_response=answered()
    )

    with pytest.raises(
        GateMemAnswerRetrievalError,
        match="store unavailable",
    ):
        service.retrieve_authorised_memories(
            runtime=runtime,
            query=query,
            requester=runtime.require_worker(
                "student_maya"
            ),
        )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_memory_context_omits_acl_membership(
    current_memory: Any,
) -> None:
    current_memory.metadata.readable_by = [
        "prof_elena",
        "finance_officer",
    ]
    service, _, _, _ = build_service(
        llm_response=answered()
    )

    context = service.format_memory_context(
        [current_memory]
    )

    assert current_memory.memory_id in context
    assert "20 August" in context
    assert "readable_by" not in context
    assert "finance_officer" not in context


def test_memory_context_respects_character_limit(
    current_memory: Any,
) -> None:
    current_memory.content = "A" * 1_000
    second = make_memory(
        "education_episode_001:t002",
        content="SECOND_MEMORY",
        turn_id="t002",
        sequence_index=1,
    )
    service, _, _, _ = build_service(
        llm_response=answered(),
        config=GateMemAnswerConfig(
            max_context_chars=160,
            max_content_chars_per_memory=1_000,
        ),
    )

    context = service.format_memory_context(
        [current_memory, second]
    )

    assert len(context) <= 160
    assert "SECOND_MEMORY" not in context


# ---------------------------------------------------------------------------
# End-to-end answer behaviour
# ---------------------------------------------------------------------------


def test_no_accessible_memory_returns_no_memory_without_llm_call(
    tmp_path: Path,
    episode: MappedEpisode,
    query: MappedQuery,
) -> None:
    runtime = build_runtime(
        tmp_path=tmp_path,
        episode=episode,
        memory_service=FakeMemoryService(
            accessible=[]
        ),
    )
    service, llm, prompt, factory_calls = (
        build_service(
            llm_response=answered()
        )
    )

    result = service.answer(
        runtime=runtime,
        query=query,
    )

    assert result.action == "no_memory"
    assert (
        result.agent_answer.status
        == "insufficient_evidence"
    )
    assert result.retrieved_record_ids == []
    assert result.used_record_ids == []
    assert llm.calls == []
    assert prompt.calls == []
    assert factory_calls == []


def test_answer_prompt_contains_only_authorised_bounded_memory(
    tmp_path: Path,
    episode: MappedEpisode,
    query: MappedQuery,
    current_memory: Any,
    future_memory: Any,
    other_episode_memory: Any,
) -> None:
    memory_service = FakeMemoryService(
        accessible=[
            current_memory,
            future_memory,
            other_episode_memory,
        ]
    )
    runtime = build_runtime(
        tmp_path=tmp_path,
        episode=episode,
        memory_service=memory_service,
    )
    service, llm, prompt, factory_calls = (
        build_service(
            llm_response=answered(
                used_memory_ids=[
                    current_memory.memory_id
                ]
            ),
            config=GateMemAnswerConfig(
                use_semantic_retrieval=False,
            ),
        )
    )

    result = service.answer(
        runtime=runtime,
        query=query,
    )

    assert result.action == "answer"
    assert result.answer == "20 August"
    assert len(llm.calls) == 1
    rendered_prompt = llm.calls[0]["prompt"]
    assert "20 August" in rendered_prompt
    assert "FUTURE_SECRET" not in rendered_prompt
    assert (
        "OTHER_EPISODE_SECRET"
        not in rendered_prompt
    )

    assert len(prompt.calls) == 1
    assert prompt.calls[0][
        "context_sections"
    ]["Authorised memory IDs"] == [
        current_memory.memory_id
    ]

    assert len(factory_calls) == 1
    _, role, domain = factory_calls[0]
    assert role == "student"
    assert domain == "education"


def test_answer_references_are_reverified_and_sanitised(
    tmp_path: Path,
    episode: MappedEpisode,
    query: MappedQuery,
    current_memory: Any,
) -> None:
    memory_service = FakeMemoryService(
        accessible=[current_memory],
        memories_by_id={
            current_memory.memory_id: (
                current_memory
            )
        },
    )
    runtime = build_runtime(
        tmp_path=tmp_path,
        episode=episode,
        memory_service=memory_service,
    )
    service, _, _, _ = build_service(
        llm_response=answered(
            used_memory_ids=[
                current_memory.memory_id,
                "hallucinated_memory",
            ]
        ),
        config=GateMemAnswerConfig(
            use_semantic_retrieval=False,
        ),
    )

    result = service.answer(
        runtime=runtime,
        query=query,
    )

    assert result.used_record_ids == [
        current_memory.memory_id
    ]
    assert (
        result.agent_answer.used_memory_ids
        == [current_memory.memory_id]
    )
    assert (
        result.agent_answer
        .supporting_source_ids
        == ["t001"]
    )
    assert (
        result.agent_answer
        .contributing_agent_ids
        == ["student_maya"]
    )
    assert memory_service.get_calls == [
        (
            runtime.require_worker(
                "student_maya"
            ),
            current_memory.memory_id,
        )
    ]
    assert any(
        "hallucinated_memory" in warning
        for warning in result.warnings
    )


def test_unverifiable_memory_reference_is_removed(
    tmp_path: Path,
    episode: MappedEpisode,
    query: MappedQuery,
    current_memory: Any,
) -> None:
    memory_service = FakeMemoryService(
        accessible=[current_memory],
        get_errors={
            current_memory.memory_id: (
                PermissionError("access revoked")
            )
        },
    )
    runtime = build_runtime(
        tmp_path=tmp_path,
        episode=episode,
        memory_service=memory_service,
    )
    service, _, _, _ = build_service(
        llm_response=answered(
            used_memory_ids=[
                current_memory.memory_id
            ]
        ),
        config=GateMemAnswerConfig(
            use_semantic_retrieval=False,
        ),
    )

    result = service.answer(
        runtime=runtime,
        query=query,
    )

    assert result.action == "answer"
    assert result.used_record_ids == []
    assert (
        result.agent_answer.used_memory_ids
        == []
    )
    assert any(
        "access revoked" in warning
        for warning in result.warnings
    )


def test_answer_is_downgraded_when_reference_is_required_but_missing(
    tmp_path: Path,
    episode: MappedEpisode,
    query: MappedQuery,
    current_memory: Any,
) -> None:
    runtime = build_runtime(
        tmp_path=tmp_path,
        episode=episode,
        memory_service=FakeMemoryService(
            accessible=[current_memory]
        ),
    )
    service, _, _, _ = build_service(
        llm_response=answered(
            used_memory_ids=[]
        ),
        config=GateMemAnswerConfig(
            use_semantic_retrieval=False,
            require_memory_reference_for_answer=True,
        ),
    )

    result = service.answer(
        runtime=runtime,
        query=query,
    )

    assert result.action == "no_memory"
    assert result.answer is None
    assert (
        result.agent_answer.status
        == "insufficient_evidence"
    )
    assert any(
        "Downgraded answer" in warning
        for warning in result.warnings
    )


def test_refused_agent_answer_maps_to_refuse_action(
    tmp_path: Path,
    episode: MappedEpisode,
    query: MappedQuery,
    current_memory: Any,
) -> None:
    runtime = build_runtime(
        tmp_path=tmp_path,
        episode=episode,
        memory_service=FakeMemoryService(
            accessible=[current_memory]
        ),
    )
    service, _, _, _ = build_service(
        llm_response=refused(),
        config=GateMemAnswerConfig(
            use_semantic_retrieval=False,
        ),
    )

    result = service.answer(
        runtime=runtime,
        query=query,
    )

    assert result.action == "refuse"
    assert result.answer is None
    assert result.used_record_ids == []
    assert (
        result.to_gatemem_payload(
            refusal_message="DECLINED"
        )["answer"]
        == "DECLINED"
    )


def test_insufficient_agent_answer_maps_to_no_memory_action(
    tmp_path: Path,
    episode: MappedEpisode,
    query: MappedQuery,
    current_memory: Any,
) -> None:
    runtime = build_runtime(
        tmp_path=tmp_path,
        episode=episode,
        memory_service=FakeMemoryService(
            accessible=[current_memory]
        ),
    )
    service, _, _, _ = build_service(
        llm_response=insufficient(),
        config=GateMemAnswerConfig(
            use_semantic_retrieval=False,
        ),
    )

    result = service.answer(
        runtime=runtime,
        query=query,
    )

    assert result.action == "no_memory"
    assert (
        result.agent_answer.status
        == "insufficient_evidence"
    )
    assert result.used_record_ids == []


def test_mapping_llm_response_is_validated_as_agent_answer(
    tmp_path: Path,
    episode: MappedEpisode,
    query: MappedQuery,
    current_memory: Any,
) -> None:
    runtime = build_runtime(
        tmp_path=tmp_path,
        episode=episode,
        memory_service=FakeMemoryService(
            accessible=[current_memory]
        ),
    )
    service, _, _, _ = build_service(
        llm_response=answered(
            used_memory_ids=[
                current_memory.memory_id
            ]
        ).model_dump(mode="python"),
        config=GateMemAnswerConfig(
            use_semantic_retrieval=False,
        ),
    )

    result = service.answer(
        runtime=runtime,
        query=query,
    )

    assert isinstance(
        result.agent_answer,
        AgentAnswer,
    )
    assert result.action == "answer"


def test_llm_failure_is_wrapped(
    tmp_path: Path,
    episode: MappedEpisode,
    query: MappedQuery,
    current_memory: Any,
) -> None:
    runtime = build_runtime(
        tmp_path=tmp_path,
        episode=episode,
        memory_service=FakeMemoryService(
            accessible=[current_memory]
        ),
    )
    service, _, _, _ = build_service(
        llm_response=None,
        llm_error=RuntimeError(
            "provider unavailable"
        ),
        config=GateMemAnswerConfig(
            use_semantic_retrieval=False,
        ),
    )

    with pytest.raises(
        GateMemAnswerGenerationError,
        match="provider unavailable",
    ):
        service.answer(
            runtime=runtime,
            query=query,
        )


def test_invalid_llm_payload_is_wrapped(
    tmp_path: Path,
    episode: MappedEpisode,
    query: MappedQuery,
    current_memory: Any,
) -> None:
    runtime = build_runtime(
        tmp_path=tmp_path,
        episode=episode,
        memory_service=FakeMemoryService(
            accessible=[current_memory]
        ),
    )
    service, _, _, _ = build_service(
        llm_response={
            "status": "answered",
            "answer": None,
            "reasoning": "No concrete answer.",
        },
        config=GateMemAnswerConfig(
            use_semantic_retrieval=False,
        ),
    )

    with pytest.raises(
        GateMemAnswerValidationError,
        match="AgentAnswer",
    ):
        service.answer(
            runtime=runtime,
            query=query,
        )


def test_worker_prompt_is_cached_per_principal_role_and_domain(
    tmp_path: Path,
    episode: MappedEpisode,
    query: MappedQuery,
    current_memory: Any,
) -> None:
    runtime = build_runtime(
        tmp_path=tmp_path,
        episode=episode,
        memory_service=FakeMemoryService(
            accessible=[current_memory]
        ),
    )
    service, llm, _, factory_calls = (
        build_service(
            llm_response=answered(
                used_memory_ids=[
                    current_memory.memory_id
                ]
            ),
            config=GateMemAnswerConfig(
                use_semantic_retrieval=False,
            ),
        )
    )

    first = service.answer(
        runtime=runtime,
        query=query,
    )
    second = service.answer(
        runtime=runtime,
        query=query,
    )

    assert first.action == "answer"
    assert second.action == "answer"
    assert len(factory_calls) == 1
    assert len(llm.calls) == 2