from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytest

from src.memory.services.memory_forgetting_service import (
    ForgettingCandidateDecision,
    ForgettingDecisionBatch,
    ForgettingExecutionError,
    ForgettingIntent,
    ForgettingIntentError,
    MemoryForgettingConfig,
    MemoryForgettingService,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeLLMClient:
    """
    Queue-based structured LLM double.

    Each invocation consumes one response. A queued exception is raised;
    dictionaries are returned unchanged so the service's own Pydantic
    validation remains under test.
    """

    def __init__(
        self,
        responses: list[Any],
    ) -> None:
        self.responses = list(responses)
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

        if not self.responses:
            raise AssertionError(
                "FakeLLMClient received more calls than expected."
            )

        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return response


class FakeMemoryStore:
    def __init__(
        self,
        memories: list[Any] | None = None,
    ) -> None:
        self.memories: dict[str, Any] = {
            memory.memory_id: memory
            for memory in memories or []
        }
        self.list_active_calls = 0
        self.get_calls: list[str] = []

    def list_active(self) -> list[Any]:
        self.list_active_calls += 1
        return [
            memory
            for memory in self.memories.values()
            if memory.metadata.status == "active"
        ]

    def get_by_id(
        self,
        memory_id: str,
    ) -> Any:
        self.get_calls.append(memory_id)

        try:
            return self.memories[memory_id]
        except KeyError as error:
            raise KeyError(
                f"Unknown memory: {memory_id}"
            ) from error


class FakeMemoryService:
    def __init__(
        self,
        store: FakeMemoryStore,
        *,
        memory_retriever: Any | None = None,
        semantic_results: (
            Mapping[str, list[Any]]
            | list[Any]
            | None
        ) = None,
        deprecate_errors: (
            Mapping[str, Exception]
            | None
        ) = None,
    ) -> None:
        self.store = store
        self.memory_retriever = memory_retriever
        self.semantic_results = (
            semantic_results
            if semantic_results is not None
            else []
        )
        self.deprecate_errors = dict(
            deprecate_errors or {}
        )

        self.retrieve_calls: list[
            dict[str, Any]
        ] = []
        self.deprecate_calls: list[
            dict[str, Any]
        ] = []

    def retrieve_memories(
        self,
        agent: Any,
        query: str,
        top_k: int = 5,
        include_private: bool = True,
        include_shared: bool = True,
    ) -> list[Any]:
        self.retrieve_calls.append(
            {
                "agent": agent,
                "query": query,
                "top_k": top_k,
                "include_private": include_private,
                "include_shared": include_shared,
            }
        )

        if isinstance(
            self.semantic_results,
            Mapping,
        ):
            return list(
                self.semantic_results.get(
                    agent.agent_id,
                    [],
                )
            )

        return list(
            self.semantic_results
        )

    def deprecate_memory(
        self,
        agent: Any,
        memory_id: str,
        reason: str | None = None,
    ) -> Any:
        self.deprecate_calls.append(
            {
                "agent": agent,
                "memory_id": memory_id,
                "reason": reason,
            }
        )

        error = self.deprecate_errors.get(
            memory_id
        )
        if error is not None:
            raise error

        memory = self.store.get_by_id(
            memory_id
        )
        memory.metadata.status = "deprecated"
        return memory


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def workers() -> dict[str, Any]:
    return {
        "financial_aid": SimpleNamespace(
            agent_id="financial_aid"
        ),
        "professor": SimpleNamespace(
            agent_id="professor"
        ),
    }


def make_memory(
    memory_id: str,
    *,
    content: str,
    owner_agent_id: str = "financial_aid",
    summary: str = "",
    status: str = "active",
    source_message_ids: list[str] | None = None,
) -> Any:
    return SimpleNamespace(
        memory_id=memory_id,
        content=content,
        summary=summary,
        metadata=SimpleNamespace(
            owner_agent_id=owner_agent_id,
            status=status,
            source_message_ids=list(
                source_message_ids or []
            ),
        ),
    )


def valid_intent(
    *,
    confidence: float = 0.99,
) -> dict[str, Any]:
    return {
        "target_fact": (
            "The earlier provisional Project Atlas "
            "stipend was 2,200 USD."
        ),
        "preserved_facts": [
            (
                "The current approved Project Atlas "
                "stipend is 2,350 USD."
            )
        ],
        "reasoning": (
            "The old amount must be forgotten while "
            "the current amount remains."
        ),
        "confidence": confidence,
    }


def decision(
    memory_id: str,
    action: str,
    confidence: float = 0.99,
) -> dict[str, Any]:
    return {
        "memory_id": memory_id,
        "decision": action,
        "reasoning": f"Classified as {action}.",
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# End-to-end behaviour
# ---------------------------------------------------------------------------


def test_forget_deprecates_all_disclosing_copies_and_preserves_current_fact(
    workers: dict[str, Any],
) -> None:
    old_direct = make_memory(
        "mem_old_direct",
        content=(
            "The provisional Project Atlas stipend "
            "was 2,200 USD."
        ),
    )
    old_summary = make_memory(
        "mem_old_summary",
        content="Project Atlas funding record.",
        summary=(
            "Earlier reports listed a monthly "
            "stipend of 2,200 USD."
        ),
        owner_agent_id="professor",
    )
    transition = make_memory(
        "mem_transition",
        content=(
            "The Project Atlas stipend changed from "
            "2,200 USD to 2,350 USD."
        ),
    )
    current = make_memory(
        "mem_current",
        content=(
            "The current approved Project Atlas "
            "stipend is 2,350 USD."
        ),
    )
    unrelated = make_memory(
        "mem_unrelated",
        content=(
            "The Project Atlas dry-run is scheduled "
            "for May 12."
        ),
        owner_agent_id="professor",
    )

    store = FakeMemoryStore(
        [
            old_direct,
            old_summary,
            transition,
            current,
            unrelated,
        ]
    )
    memory_service = FakeMemoryService(
        store
    )
    llm = FakeLLMClient(
        [
            valid_intent(),
            {
                "decisions": [
                    decision(
                        "mem_old_direct",
                        "forget",
                    ),
                    decision(
                        "mem_old_summary",
                        "forget",
                    ),
                    decision(
                        "mem_transition",
                        "forget",
                    ),
                    decision(
                        "mem_current",
                        "preserve",
                    ),
                    decision(
                        "mem_unrelated",
                        "unrelated",
                    ),
                ]
            },
        ]
    )

    service = MemoryForgettingService(
        llm_client=llm,
        config=MemoryForgettingConfig(
            lexical_top_k=10,
            max_candidates=10,
        ),
    )

    result = service.forget(
        request_text=(
            "Remove the earlier provisional "
            "2,200 USD Atlas stipend. Keep the "
            "approved 2,350 USD amount current."
        ),
        requesting_agent_id="financial_aid",
        source_turn_id="t148",
        workers=workers,
        memory_service=memory_service,
        memory_store=store,
    )

    assert set(
        result.forgotten_memory_ids
    ) == {
        "mem_old_direct",
        "mem_old_summary",
        "mem_transition",
    }
    assert result.preserved_memory_ids == [
        "mem_current"
    ]
    assert result.unrelated_memory_ids == [
        "mem_unrelated"
    ]
    assert result.uncertain_memory_ids == []
    assert result.changed_memory_state is True

    assert old_direct.metadata.status == (
        "deprecated"
    )
    assert old_summary.metadata.status == (
        "deprecated"
    )
    assert transition.metadata.status == (
        "deprecated"
    )
    assert current.metadata.status == "active"
    assert unrelated.metadata.status == "active"

    assert {
        call["memory_id"]
        for call in memory_service.deprecate_calls
    } == {
        "mem_old_direct",
        "mem_old_summary",
        "mem_transition",
    }

    for call in memory_service.deprecate_calls:
        assert "source_turn_id=t148" in (
            call["reason"]
        )
        assert (
            "requesting_agent_id="
            "financial_aid"
        ) in call["reason"]

    assert [
        call["schema_model"]
        for call in llm.calls
    ] == [
        ForgettingIntent,
        ForgettingDecisionBatch,
    ]
    assert all(
        call[
            "apply_parser_normalization"
        ]
        is False
        for call in llm.calls
    )


def test_no_active_memories_returns_warning_without_candidate_classification(
    workers: dict[str, Any],
) -> None:
    store = FakeMemoryStore()
    memory_service = FakeMemoryService(
        store
    )
    llm = FakeLLMClient(
        [valid_intent()]
    )
    service = MemoryForgettingService(
        llm_client=llm
    )

    result = service.forget(
        request_text="Forget the old Atlas stipend.",
        requesting_agent_id="financial_aid",
        source_turn_id="t148",
        workers=workers,
        memory_service=memory_service,
        memory_store=store,
    )

    assert result.candidate_memory_ids == []
    assert result.forgotten_memory_ids == []
    assert result.changed_memory_state is False
    assert result.warnings == [
        (
            "No active memories were available "
            "for forgetting."
        )
    ]
    assert len(llm.calls) == 1
    assert memory_service.deprecate_calls == []


def test_low_confidence_forget_is_not_executed(
    workers: dict[str, Any],
) -> None:
    old_memory = make_memory(
        "mem_old",
        content=(
            "The provisional Project Atlas stipend "
            "was 2,200 USD."
        ),
    )
    store = FakeMemoryStore(
        [old_memory]
    )
    memory_service = FakeMemoryService(
        store
    )
    llm = FakeLLMClient(
        [
            valid_intent(),
            {
                "decisions": [
                    decision(
                        "mem_old",
                        "forget",
                        confidence=0.69,
                    )
                ]
            },
        ]
    )
    service = MemoryForgettingService(
        llm_client=llm,
        config=MemoryForgettingConfig(
            minimum_forget_confidence=0.70
        ),
    )

    result = service.forget(
        request_text="Forget the old 2,200 USD amount.",
        requesting_agent_id="financial_aid",
        source_turn_id="t148",
        workers=workers,
        memory_service=memory_service,
        memory_store=store,
    )

    assert result.forgotten_memory_ids == []
    assert result.uncertain_memory_ids == [
        "mem_old"
    ]
    assert old_memory.metadata.status == "active"
    assert memory_service.deprecate_calls == []
    assert any(
        "Low-confidence forget decision"
        in warning
        for warning in result.warnings
    )


# ---------------------------------------------------------------------------
# Candidate retrieval
# ---------------------------------------------------------------------------


def test_candidate_retrieval_merges_semantic_and_lexical_results_and_deduplicates(
    workers: dict[str, Any],
) -> None:
    semantic = make_memory(
        "mem_semantic",
        content=(
            "An archived Atlas funding entry "
            "records the old amount."
        ),
    )
    lexical = make_memory(
        "mem_lexical",
        content=(
            "The provisional Project Atlas stipend "
            "was 2,200 USD."
        ),
        owner_agent_id="professor",
    )
    deprecated = make_memory(
        "mem_deprecated",
        content=(
            "The provisional Project Atlas stipend "
            "was 2,200 USD."
        ),
        status="deprecated",
    )

    store = FakeMemoryStore(
        [
            semantic,
            lexical,
            deprecated,
        ]
    )
    memory_service = FakeMemoryService(
        store,
        memory_retriever=object(),
        semantic_results={
            "financial_aid": [
                semantic,
                deprecated,
            ],
            "professor": [
                semantic,
            ],
        },
    )
    service = MemoryForgettingService(
        llm_client=FakeLLMClient(
            []
        )
    )

    candidates, warnings = (
        service.retrieve_candidates(
            intent=ForgettingIntent.model_validate(
                valid_intent()
            ),
            workers=workers,
            memory_service=memory_service,
            active_memories=(
                store.list_active()
            ),
        )
    )

    assert [
        memory.memory_id
        for memory in candidates
    ] == [
        "mem_semantic",
        "mem_lexical",
    ]
    assert warnings == []
    assert len(
        memory_service.retrieve_calls
    ) == len(workers)
    assert all(
        call["query"].startswith(
            "The earlier provisional"
        )
        for call in (
            memory_service.retrieve_calls
        )
    )


# ---------------------------------------------------------------------------
# Model-output validation and fail-closed behaviour
# ---------------------------------------------------------------------------


def test_unknown_and_missing_decision_ids_are_fail_closed_as_uncertain() -> None:
    memory = make_memory(
        "mem_real",
        content=(
            "The provisional Project Atlas stipend "
            "was 2,200 USD."
        ),
    )
    llm = FakeLLMClient(
        [
            {
                "decisions": [
                    decision(
                        "mem_invented",
                        "forget",
                    )
                ]
            }
        ]
    )
    service = MemoryForgettingService(
        llm_client=llm
    )

    decisions, warnings = (
        service.classify_candidates(
            intent=(
                ForgettingIntent
                .model_validate(
                    valid_intent()
                )
            ),
            candidates=[memory],
        )
    )

    assert decisions == [
        ForgettingCandidateDecision(
            memory_id="mem_real",
            decision="uncertain",
            reasoning=(
                "No structured decision "
                "was returned."
            ),
            confidence=0.0,
        )
    ]
    assert any(
        "outside the supplied batch"
        in warning
        for warning in warnings
    )
    assert any(
        "No decision was returned"
        in warning
        for warning in warnings
    )


def test_structured_invocation_retries_after_invalid_intent_output() -> None:
    llm = FakeLLMClient(
        [
            {
                "target_fact": "",
                "preserved_facts": [],
                "reasoning": "",
                "confidence": 0.9,
            },
            valid_intent(),
        ]
    )
    service = MemoryForgettingService(
        llm_client=llm,
        config=MemoryForgettingConfig(
            structured_retry_count=1
        ),
    )

    intent = service.extract_intent(
        "Forget the old 2,200 USD amount."
    )

    assert intent.target_fact.startswith(
        "The earlier provisional"
    )
    assert len(llm.calls) == 2
    assert (
        "previous structured response "
        "was invalid"
    ) in llm.calls[1]["prompt"]


def test_low_confidence_intent_is_rejected() -> None:
    llm = FakeLLMClient(
        [
            valid_intent(
                confidence=0.59
            )
        ]
    )
    service = MemoryForgettingService(
        llm_client=llm,
        config=MemoryForgettingConfig(
            minimum_intent_confidence=0.60
        ),
    )

    with pytest.raises(
        ForgettingIntentError,
        match="below 0.600",
    ):
        service.extract_intent(
            "Forget the old Atlas amount."
        )


# ---------------------------------------------------------------------------
# Execution errors
# ---------------------------------------------------------------------------


def test_non_strict_execution_reports_failure_and_keeps_memory_active(
    workers: dict[str, Any],
) -> None:
    memory = make_memory(
        "mem_old",
        content=(
            "The provisional Project Atlas stipend "
            "was 2,200 USD."
        ),
    )
    store = FakeMemoryStore(
        [memory]
    )
    memory_service = FakeMemoryService(
        store,
        deprecate_errors={
            "mem_old": RuntimeError(
                "simulated persistence failure"
            )
        },
    )
    llm = FakeLLMClient(
        [
            valid_intent(),
            {
                "decisions": [
                    decision(
                        "mem_old",
                        "forget",
                    )
                ]
            },
        ]
    )
    service = MemoryForgettingService(
        llm_client=llm,
        config=MemoryForgettingConfig(
            strict_execution=False
        ),
    )

    result = service.forget(
        request_text="Forget the old 2,200 USD amount.",
        requesting_agent_id="financial_aid",
        source_turn_id="t148",
        workers=workers,
        memory_service=memory_service,
        memory_store=store,
    )

    assert result.forgotten_memory_ids == []
    assert result.uncertain_memory_ids == [
        "mem_old"
    ]
    assert memory.metadata.status == "active"
    assert any(
        "simulated persistence failure"
        in warning
        for warning in result.warnings
    )


def test_strict_execution_raises_when_soft_delete_fails(
    workers: dict[str, Any],
) -> None:
    memory = make_memory(
        "mem_old",
        content=(
            "The provisional Project Atlas stipend "
            "was 2,200 USD."
        ),
    )
    store = FakeMemoryStore(
        [memory]
    )
    memory_service = FakeMemoryService(
        store,
        deprecate_errors={
            "mem_old": RuntimeError(
                "simulated persistence failure"
            )
        },
    )
    llm = FakeLLMClient(
        [
            valid_intent(),
            {
                "decisions": [
                    decision(
                        "mem_old",
                        "forget",
                    )
                ]
            },
        ]
    )
    service = MemoryForgettingService(
        llm_client=llm,
        config=MemoryForgettingConfig(
            strict_execution=True
        ),
    )

    with pytest.raises(
        ForgettingExecutionError,
        match=(
            "simulated persistence failure"
        ),
    ):
        service.forget(
            request_text=(
                "Forget the old "
                "2,200 USD amount."
            ),
            requesting_agent_id=(
                "financial_aid"
            ),
            source_turn_id="t148",
            workers=workers,
            memory_service=memory_service,
            memory_store=store,
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("semantic_top_k_per_owner", 0),
        ("lexical_top_k", 0),
        ("max_candidates", 0),
        ("decision_batch_size", 0),
        ("minimum_intent_confidence", 1.1),
        ("minimum_forget_confidence", -0.1),
        ("structured_retry_count", -1),
    ],
)
def test_invalid_config_is_rejected(
    field_name: str,
    field_value: Any,
) -> None:
    kwargs = {
        field_name: field_value
    }

    with pytest.raises(ValueError):
        MemoryForgettingConfig(
            **kwargs
        )