from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from experiments.evaluator.informativebench import (
    AGENT_PERSONA_MAP,
    PERSONA_AGENT_MAP,
    PERSONA_NAMES,
    NeedlePersonaIngestionError,
    NeedlePersonaIngestionService,
    NeedlePersonaLoader,
    NeedlePersonaRecordError,
    NeedlePersonaSample,
    PersonaSource,
    PrivateMemoryIndex,
    build_default_persona_agents,
)
from src.llm import (
    ExtractedMemory,
    MemoryExtractionOutput,
)
from src.memory.entities import (
    MemoryType,
    SourceType as MemorySourceType,
)


# ---------------------------------------------------------------------------
# Test fixtures and fakes
# ---------------------------------------------------------------------------


@pytest.fixture
def raw_record() -> dict[str, Any]:
    """
    Minimal record matching the generated dataset_2hop.jsonl structure.
    """
    return {
        "id": "sample_001",
        "modified_alice_bob_conversation": "\n".join(
            [
                "alice: I love Alexander McQueen's dramatic fashion.",
                "bob: I enjoy jazz and collect vinyl records.",
                "alice: I also prefer minimalist black clothing.",
                "bob: My favourite instrument is the saxophone.",
            ]
        ),
        "modified_charlie_dave_conversation": "\n".join(
            [
                "**Charlie:** I enjoy hiking in the mountains.",
                "**Dave:** Coco Chanel inspires my personal style.",
                "**Charlie:** I usually travel during the summer.",
                "**Dave:** I prefer classic tailored jackets.",
            ]
        ),
        "chat_bob_charlie": "\n".join(
            [
                "bob: Alice likes Alexander McQueen.",
                "charlie: Dave likes Coco Chanel.",
            ]
        ),
        "task_prompt": (
            "What fashion icons inspire Alice and Dave's unique styles?"
        ),
        "answer": "Alexander McQueen and Coco Chanel",
        "needle_detail": (
            "Alice is inspired by Alexander McQueen and Dave is inspired "
            "by Coco Chanel."
        ),
    }


@pytest.fixture
def dataset_path(
    tmp_path: Path,
    raw_record: dict[str, Any],
) -> Path:
    path = tmp_path / "dataset_2hop.jsonl"

    path.write_text(
        json.dumps(raw_record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return path


@pytest.fixture
def loaded_sample(dataset_path: Path) -> NeedlePersonaSample:
    loader = NeedlePersonaLoader(dataset_path)
    return loader.load_by_id("sample_001")


class FakeLLMClient:
    """
    Deterministic replacement for LLMClient.

    Each invoke_structured call returns the next queued
    MemoryExtractionOutput and records the prompt for isolation checks.
    """

    def __init__(
        self,
        outputs: list[MemoryExtractionOutput],
    ) -> None:
        self._outputs = deque(outputs)
        self.prompts: list[str] = []
        self.schemas: list[type[Any]] = []

    def invoke_structured(
        self,
        prompt: str,
        schema_model: type[Any],
        *,
        apply_parser_normalization: bool = True,
    ) -> Any:
        self.prompts.append(prompt)
        self.schemas.append(schema_model)

        if schema_model is not MemoryExtractionOutput:
            raise AssertionError(
                "Ingestion should request MemoryExtractionOutput."
            )

        if not self._outputs:
            raise AssertionError(
                "FakeLLMClient has no queued output for this call."
            )

        return self._outputs.popleft()

    @property
    def remaining_outputs(self) -> int:
        return len(self._outputs)


class FakeMemoryService:
    """
    In-memory replacement for MemoryService.

    It captures every create/deprecate call and can be configured to fail on
    one create call to test ingestion rollback.
    """

    def __init__(
        self,
        *,
        fail_on_create_call: int | None = None,
        fail_on_deprecate_ids: set[str] | None = None,
    ) -> None:
        self.fail_on_create_call = fail_on_create_call
        self.fail_on_deprecate_ids = fail_on_deprecate_ids or set()

        self.create_calls: list[dict[str, Any]] = []
        self.deprecate_calls: list[dict[str, Any]] = []
        self.memories: dict[str, SimpleNamespace] = {}

    def create_private_memory(
        self,
        agent: Any,
        content: str,
        summary: str | None = None,
        memory_type: MemoryType | str = MemoryType.NOTE,
        tags: list[str] | None = None,
        importance: float = 0.5,
        confidence: float = 1.0,
        source_task_id: str | None = None,
        source_message_ids: list[str] | None = None,
        source_type: MemorySourceType | str = (
            MemorySourceType.AGENT_OUTPUT
        ),
        reason: str | None = None,
    ) -> SimpleNamespace:
        call_number = len(self.create_calls) + 1

        if self.fail_on_create_call == call_number:
            raise RuntimeError(
                f"Simulated memory write failure on call {call_number}."
            )

        memory_id = f"mem_{call_number:03d}"

        call = {
            "memory_id": memory_id,
            "agent": agent,
            "content": content,
            "summary": summary,
            "memory_type": memory_type,
            "tags": list(tags or []),
            "importance": importance,
            "confidence": confidence,
            "source_task_id": source_task_id,
            "source_message_ids": list(source_message_ids or []),
            "source_type": source_type,
            "reason": reason,
        }

        self.create_calls.append(call)

        memory = SimpleNamespace(
            memory_id=memory_id,
            content=content,
            summary=summary,
            metadata=SimpleNamespace(
                owner_agent_id=agent.agent_id,
                memory_type=memory_type,
                tags=list(tags or []),
                status="active",
            ),
        )

        self.memories[memory_id] = memory
        return memory

    def deprecate_memory(
        self,
        agent: Any,
        memory_id: str,
        reason: str | None = None,
    ) -> SimpleNamespace:
        if memory_id in self.fail_on_deprecate_ids:
            raise RuntimeError(
                f"Simulated rollback failure for {memory_id}."
            )

        memory = self.memories[memory_id]
        memory.metadata.status = "deprecated"

        self.deprecate_calls.append(
            {
                "agent": agent,
                "memory_id": memory_id,
                "reason": reason,
            }
        )

        return memory


def build_extraction_outputs(
    sample: NeedlePersonaSample,
    *,
    duplicate_alice_content: bool = False,
) -> list[MemoryExtractionOutput]:
    """
    Build one deterministic extraction output for every source in the exact
    order used by NeedlePersonaIngestionService.
    """
    outputs: list[MemoryExtractionOutput] = []

    for persona in PERSONA_NAMES:
        agent_id = PERSONA_AGENT_MAP[persona]

        for source_index, source in enumerate(
            sample.get_sources(persona)
        ):
            if persona == "alice" and duplicate_alice_content:
                content = "Alice likes Alexander McQueen."
            else:
                content = (
                    f"{persona.title()} memory from "
                    f"{source.source_id}."
                )

            outputs.append(
                MemoryExtractionOutput(
                    agent_id=agent_id,
                    memories=[
                        ExtractedMemory(
                            content=content,
                            subject=persona.title(),
                            memory_type="semantic",
                            source_ids=[source.source_id],
                            importance=0.8,
                            shareable=True,
                            confidence=0.9,
                        )
                    ],
                    extraction_summary=(
                        f"Extracted one memory for {agent_id} "
                        f"source {source_index}."
                    ),
                )
            )

    return outputs


def build_service(
    sample: NeedlePersonaSample,
    *,
    outputs: list[MemoryExtractionOutput] | None = None,
    memory_service: FakeMemoryService | None = None,
    include_task_context: bool = False,
    require_memory_per_source: bool = True,
    max_memories_per_source: int | None = None,
) -> tuple[
    NeedlePersonaIngestionService,
    FakeLLMClient,
    FakeMemoryService,
]:
    fake_llm = FakeLLMClient(
        outputs or build_extraction_outputs(sample)
    )
    fake_memory = memory_service or FakeMemoryService()

    service = NeedlePersonaIngestionService(
        llm_client=fake_llm,  # type: ignore[arg-type]
        memory_service=fake_memory,
        agents=build_default_persona_agents(),
        include_task_context=include_task_context,
        rollback_on_error=True,
        require_memory_per_source=require_memory_per_source,
        max_memories_per_source=max_memories_per_source,
    )

    return service, fake_llm, fake_memory


# ---------------------------------------------------------------------------
# Package and model tests
# ---------------------------------------------------------------------------


def test_package_exports_are_available() -> None:
    """
    The package-level __init__.py should expose the main public API.
    """
    import experiments.evaluator.informativebench as package

    expected_names = {
        "NeedlePersonaLoader",
        "NeedlePersonaSample",
        "PersonaSource",
        "PrivateMemoryIndex",
        "NeedlePersonaIngestionService",
        "build_default_persona_agents",
    }

    missing = [
        name
        for name in expected_names
        if not hasattr(package, name)
    ]

    assert not missing, f"Missing package exports: {missing}"


def test_persona_sample_helpers_and_answer_cleaning() -> None:
    alice_source = PersonaSource(
        source_id="alice_001",
        persona="alice",
        content="Alice likes science fiction.",
        source_type="persona",
    )

    sample = NeedlePersonaSample(
        sample_id="sample_001",
        question="What does Alice like?",
        gold_answer="Science fiction",
        alternative_answers=[
            "Sci-fi",
            "Science fiction",
            " ",
            "Sci-fi",
        ],
        hop=2,
        persona_sources={
            "alice": [alice_source],
        },
    )

    assert set(sample.persona_sources) == set(PERSONA_NAMES)
    assert sample.get_sources("alice") == [alice_source]
    assert sample.get_agent_sources("alice_agent") == [
        alice_source
    ]
    assert sample.accepted_answers() == [
        "Science fiction",
        "Sci-fi",
    ]

    with pytest.raises(KeyError):
        sample.get_agent_sources("unknown_agent")


def test_sample_rejects_source_in_wrong_persona_partition() -> None:
    bob_source = PersonaSource(
        source_id="bob_001",
        persona="bob",
        content="Bob likes jazz.",
    )

    with pytest.raises(
        ValidationError,
        match="belongs to 'bob'.*placed under 'alice'",
    ):
        NeedlePersonaSample(
            sample_id="sample_001",
            question="What does Bob like?",
            gold_answer="Jazz",
            persona_sources={
                "alice": [bob_source],
            },
        )


def test_sample_rejects_duplicate_source_ids() -> None:
    alice_source = PersonaSource(
        source_id="duplicate_source",
        persona="alice",
        content="Alice likes art.",
    )
    bob_source = PersonaSource(
        source_id="duplicate_source",
        persona="bob",
        content="Bob likes music.",
    )

    with pytest.raises(
        ValidationError,
        match="Duplicate source_id",
    ):
        NeedlePersonaSample(
            sample_id="sample_001",
            question="What do they like?",
            gold_answer="Art and music",
            persona_sources={
                "alice": [alice_source],
                "bob": [bob_source],
            },
        )


def test_private_memory_index_normalises_all_agent_keys() -> None:
    index = PrivateMemoryIndex(
        run_id="run_001",
        sample_id="sample_001",
        private_memory_ids={
            "alice_agent": [
                "mem_001",
                "mem_001",
                " ",
            ]
        },
        source_to_memory_ids={
            "source_001": [
                "mem_001",
                "mem_001",
            ]
        },
    )

    assert set(index.private_memory_ids) == set(
        AGENT_PERSONA_MAP
    )
    assert index.private_memory_ids["alice_agent"] == [
        "mem_001"
    ]
    assert index.private_memory_ids["bob_agent"] == []
    assert index.source_to_memory_ids["source_001"] == [
        "mem_001"
    ]


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


def test_loader_converts_jsonl_to_four_owned_persona_sources(
    dataset_path: Path,
) -> None:
    loader = NeedlePersonaLoader(dataset_path)

    samples = loader.load_all()

    assert len(samples) == 1

    sample = samples[0]

    assert sample.sample_id == "sample_001"
    assert sample.hop == 2
    assert sample.gold_answer == (
        "Alexander McQueen and Coco Chanel"
    )
    assert set(sample.persona_sources) == set(PERSONA_NAMES)

    for persona in PERSONA_NAMES:
        sources = sample.get_sources(persona)

        assert len(sources) == 1
        assert sources[0].persona == persona
        assert sources[0].source_type == "persona"

    alice_content = sample.get_sources("alice")[0].content
    bob_content = sample.get_sources("bob")[0].content
    charlie_content = sample.get_sources("charlie")[0].content
    dave_content = sample.get_sources("dave")[0].content

    assert "Alexander McQueen" in alice_content
    assert "jazz" not in alice_content.lower()

    assert "jazz" in bob_content.lower()
    assert "Alexander McQueen" not in bob_content

    assert "hiking" in charlie_content.lower()
    assert "Coco Chanel" not in charlie_content

    assert "Coco Chanel" in dave_content
    assert "hiking" not in dave_content.lower()


def test_loader_excludes_collaborative_chat_and_raw_fields_by_default(
    loaded_sample: NeedlePersonaSample,
) -> None:
    bob_sources = loaded_sample.get_sources("bob")
    charlie_sources = loaded_sample.get_sources("charlie")

    assert len(bob_sources) == 1
    assert len(charlie_sources) == 1

    all_owned_content = "\n".join(
        source.content
        for persona in PERSONA_NAMES
        for source in loaded_sample.get_sources(persona)
    )

    assert (
        "Alice likes Alexander McQueen."
        not in all_owned_content
    )
    assert (
        "Dave likes Coco Chanel."
        not in all_owned_content
    )

    assert loaded_sample.metadata[
        "collaborative_chat_included"
    ] is False
    assert loaded_sample.metadata[
        "raw_fields_retained"
    ] is False
    assert "raw_fields" not in loaded_sample.metadata
    assert "needle_detail" not in loaded_sample.metadata


def test_loader_can_include_collaborative_chat_explicitly(
    dataset_path: Path,
) -> None:
    loader = NeedlePersonaLoader(
        dataset_path,
        include_collaborative_chat=True,
    )

    sample = loader.load_by_id("sample_001")

    assert len(sample.get_sources("bob")) == 2
    assert len(sample.get_sources("charlie")) == 2
    assert sample.metadata[
        "collaborative_chat_included"
    ] is True


def test_loader_dialogue_turn_granularity(
    dataset_path: Path,
) -> None:
    loader = NeedlePersonaLoader(
        dataset_path,
        source_granularity="dialogue_turn",
    )

    sample = loader.load_by_id("sample_001")

    assert len(sample.get_sources("alice")) == 2
    assert len(sample.get_sources("bob")) == 2
    assert len(sample.get_sources("charlie")) == 2
    assert len(sample.get_sources("dave")) == 2

    first_alice = sample.get_sources("alice")[0]

    assert first_alice.source_type == "dialogue_turn"
    assert first_alice.turn_index == 0
    assert first_alice.speaker == "alice"


def test_loader_rejects_duplicate_sample_ids(
    tmp_path: Path,
    raw_record: dict[str, Any],
) -> None:
    path = tmp_path / "duplicates.jsonl"
    line = json.dumps(raw_record, ensure_ascii=False)

    path.write_text(
        f"{line}\n{line}\n",
        encoding="utf-8",
    )

    loader = NeedlePersonaLoader(path)

    with pytest.raises(
        NeedlePersonaRecordError,
        match="Duplicate sample ID",
    ):
        loader.load_all()


def test_loader_rejects_missing_required_field(
    tmp_path: Path,
    raw_record: dict[str, Any],
) -> None:
    invalid_record = dict(raw_record)
    invalid_record.pop("answer")

    path = tmp_path / "invalid.jsonl"
    path.write_text(
        json.dumps(invalid_record) + "\n",
        encoding="utf-8",
    )

    loader = NeedlePersonaLoader(path)

    with pytest.raises(
        NeedlePersonaRecordError,
        match="Missing required fields: answer",
    ):
        loader.load_all()


# ---------------------------------------------------------------------------
# Ingestion tests
# ---------------------------------------------------------------------------


def test_ingestion_creates_private_memories_for_all_four_agents(
    loaded_sample: NeedlePersonaSample,
) -> None:
    service, fake_llm, fake_memory = build_service(
        loaded_sample
    )

    index = service.ingest(
        sample=loaded_sample,
        run_id="run_001",
    )

    assert fake_llm.remaining_outputs == 0
    assert len(fake_memory.create_calls) == 4
    assert len(fake_memory.deprecate_calls) == 0

    assert set(index.private_memory_ids) == set(
        AGENT_PERSONA_MAP
    )
    assert all(
        len(memory_ids) == 1
        for memory_ids in index.private_memory_ids.values()
    )
    assert len(index.source_to_memory_ids) == 4

    for call in fake_memory.create_calls:
        owner_agent_id = call["agent"].agent_id
        persona = AGENT_PERSONA_MAP[owner_agent_id]

        assert call["source_type"] == MemorySourceType.MESSAGE
        assert call["importance"] == pytest.approx(0.8)
        assert call["confidence"] == pytest.approx(0.9)
        assert call["source_task_id"] == (
            "needle:sample_001:run:run_001"
        )

        tags = set(call["tags"])

        assert "benchmark:informativebench" in tags
        assert "subset:needle_in_the_persona" in tags
        assert "stage:private_ingestion" in tags
        assert f"persona:{persona}" in tags
        assert f"owner:{owner_agent_id}" in tags
        assert "promotion_candidate" in tags

        assert len(call["source_message_ids"]) == 1

        source_id = call["source_message_ids"][0]

        assert source_id in index.source_to_memory_ids
        assert call["memory_id"] in (
            index.source_to_memory_ids[source_id]
        )
        assert call["memory_id"] in (
            index.private_memory_ids[owner_agent_id]
        )


def test_ingestion_does_not_expose_question_by_default(
    loaded_sample: NeedlePersonaSample,
) -> None:
    service, fake_llm, _ = build_service(
        loaded_sample,
        include_task_context=False,
    )

    service.ingest(
        sample=loaded_sample,
        run_id="run_001",
    )

    assert fake_llm.prompts

    for prompt in fake_llm.prompts:
        assert loaded_sample.question not in prompt
        assert loaded_sample.gold_answer not in prompt
        assert "Task-conditioned extraction:\nFalse" in prompt


def test_ingestion_can_include_question_only_when_explicitly_enabled(
    loaded_sample: NeedlePersonaSample,
) -> None:
    service, fake_llm, _ = build_service(
        loaded_sample,
        include_task_context=True,
    )

    service.ingest(
        sample=loaded_sample,
        run_id="run_001",
    )

    assert all(
        loaded_sample.question in prompt
        for prompt in fake_llm.prompts
    )
    assert all(
        "Task-conditioned extraction:\nTrue" in prompt
        for prompt in fake_llm.prompts
    )


def test_ingestion_maps_semantic_preferences_to_preference_type(
    loaded_sample: NeedlePersonaSample,
) -> None:
    outputs = build_extraction_outputs(loaded_sample)

    outputs[0] = MemoryExtractionOutput(
        agent_id="alice_agent",
        memories=[
            ExtractedMemory(
                content=(
                    "Alice loves Alexander McQueen's fashion."
                ),
                subject="Alice",
                memory_type="semantic",
                source_ids=[
                    loaded_sample.get_sources("alice")[0].source_id
                ],
                importance=0.9,
                shareable=True,
                confidence=0.95,
            )
        ],
    )

    service, _, fake_memory = build_service(
        loaded_sample,
        outputs=outputs,
    )

    service.ingest(
        sample=loaded_sample,
        run_id="run_001",
    )

    alice_call = fake_memory.create_calls[0]

    assert alice_call["agent"].agent_id == "alice_agent"
    assert alice_call["memory_type"] == MemoryType.PREFERENCE


@pytest.mark.parametrize(
    ("extracted_type", "expected_memory_type"),
    [
        ("episodic", MemoryType.EVENT),
        ("procedural", MemoryType.RULE),
        ("semantic", MemoryType.FACT),
    ],
)
def test_ingestion_maps_generic_memory_types(
    loaded_sample: NeedlePersonaSample,
    extracted_type: str,
    expected_memory_type: MemoryType,
) -> None:
    outputs = build_extraction_outputs(loaded_sample)

    outputs[0] = MemoryExtractionOutput(
        agent_id="alice_agent",
        memories=[
            ExtractedMemory(
                content="Alice attended a design event.",
                subject="Alice",
                memory_type=extracted_type,  # type: ignore[arg-type]
                source_ids=[
                    loaded_sample.get_sources("alice")[0].source_id
                ],
                importance=0.7,
                shareable=True,
                confidence=0.9,
            )
        ],
    )

    service, _, fake_memory = build_service(
        loaded_sample,
        outputs=outputs,
    )

    service.ingest(
        sample=loaded_sample,
        run_id="run_001",
    )

    assert (
        fake_memory.create_calls[0]["memory_type"]
        == expected_memory_type
    )


def test_ingestion_attaches_supplied_source_id_when_llm_omits_it(
    loaded_sample: NeedlePersonaSample,
) -> None:
    outputs = build_extraction_outputs(loaded_sample)
    alice_source_id = (
        loaded_sample.get_sources("alice")[0].source_id
    )

    outputs[0] = MemoryExtractionOutput(
        agent_id="alice_agent",
        memories=[
            ExtractedMemory(
                content="Alice likes fashion design.",
                subject="Alice",
                memory_type="semantic",
                source_ids=[],
                importance=0.7,
                shareable=True,
                confidence=0.9,
            )
        ],
    )

    service, _, fake_memory = build_service(
        loaded_sample,
        outputs=outputs,
    )

    service.ingest(
        sample=loaded_sample,
        run_id="run_001",
    )

    assert fake_memory.create_calls[0][
        "source_message_ids"
    ] == [alice_source_id]


def test_ingestion_rejects_unavailable_source_id(
    loaded_sample: NeedlePersonaSample,
) -> None:
    outputs = build_extraction_outputs(loaded_sample)

    outputs[0] = MemoryExtractionOutput(
        agent_id="alice_agent",
        memories=[
            ExtractedMemory(
                content="Alice likes fashion design.",
                subject="Alice",
                memory_type="semantic",
                source_ids=["hidden_source"],
                importance=0.7,
                shareable=True,
                confidence=0.9,
            )
        ],
    )

    service, _, fake_memory = build_service(
        loaded_sample,
        outputs=outputs,
    )

    with pytest.raises(
        NeedlePersonaIngestionError,
        match="unavailable source IDs",
    ):
        service.ingest(
            sample=loaded_sample,
            run_id="run_001",
        )

    assert fake_memory.create_calls == []


def test_ingestion_rejects_wrong_extraction_agent_id(
    loaded_sample: NeedlePersonaSample,
) -> None:
    outputs = build_extraction_outputs(loaded_sample)

    outputs[0] = MemoryExtractionOutput(
        agent_id="bob_agent",
        memories=[
            ExtractedMemory(
                content="Alice likes fashion design.",
                subject="Alice",
                memory_type="semantic",
                source_ids=[
                    loaded_sample.get_sources("alice")[0].source_id
                ],
            )
        ],
    )

    service, _, fake_memory = build_service(
        loaded_sample,
        outputs=outputs,
    )

    with pytest.raises(
        NeedlePersonaIngestionError,
        match="wrong agent_id",
    ):
        service.ingest(
            sample=loaded_sample,
            run_id="run_001",
        )

    assert fake_memory.create_calls == []


def test_ingestion_deduplicates_same_content_for_one_agent(
    loaded_sample: NeedlePersonaSample,
) -> None:
    second_alice_source = PersonaSource(
        source_id="sample_001__alice__second",
        persona="alice",
        content=(
            "alice: Alexander McQueen is one of my favourite "
            "designers."
        ),
        source_type="persona",
    )

    expanded_sample = loaded_sample.model_copy(deep=True)
    expanded_sample.persona_sources["alice"].append(
        second_alice_source
    )
    expanded_sample = NeedlePersonaSample.model_validate(
        expanded_sample.model_dump()
    )

    outputs = build_extraction_outputs(
        expanded_sample,
        duplicate_alice_content=True,
    )

    service, fake_llm, fake_memory = build_service(
        expanded_sample,
        outputs=outputs,
    )

    index = service.ingest(
        sample=expanded_sample,
        run_id="run_001",
    )

    assert fake_llm.remaining_outputs == 0

    alice_calls = [
        call
        for call in fake_memory.create_calls
        if call["agent"].agent_id == "alice_agent"
    ]

    assert len(alice_calls) == 1
    assert len(
        index.private_memory_ids["alice_agent"]
    ) == 1

    first_source = (
        expanded_sample.get_sources("alice")[0].source_id
    )
    second_source = (
        expanded_sample.get_sources("alice")[1].source_id
    )

    assert len(index.source_to_memory_ids[first_source]) == 1
    assert index.source_to_memory_ids[second_source] == []


def test_ingestion_limits_memories_per_source(
    loaded_sample: NeedlePersonaSample,
) -> None:
    outputs = build_extraction_outputs(loaded_sample)
    alice_source_id = (
        loaded_sample.get_sources("alice")[0].source_id
    )

    outputs[0] = MemoryExtractionOutput(
        agent_id="alice_agent",
        memories=[
            ExtractedMemory(
                content=f"Alice memory {index}.",
                source_ids=[alice_source_id],
                memory_type="semantic",
            )
            for index in range(3)
        ],
    )

    service, _, fake_memory = build_service(
        loaded_sample,
        outputs=outputs,
        max_memories_per_source=2,
    )

    index = service.ingest(
        sample=loaded_sample,
        run_id="run_001",
    )

    alice_calls = [
        call
        for call in fake_memory.create_calls
        if call["agent"].agent_id == "alice_agent"
    ]

    assert len(alice_calls) == 2
    assert len(
        index.source_to_memory_ids[alice_source_id]
    ) == 2


def test_ingestion_rolls_back_previous_memories_on_write_failure(
    loaded_sample: NeedlePersonaSample,
) -> None:
    fake_memory = FakeMemoryService(
        fail_on_create_call=3
    )

    service, _, fake_memory = build_service(
        loaded_sample,
        memory_service=fake_memory,
    )

    with pytest.raises(
        NeedlePersonaIngestionError,
        match="Simulated memory write failure",
    ):
        service.ingest(
            sample=loaded_sample,
            run_id="run_rollback",
        )

    assert len(fake_memory.create_calls) == 2
    assert [
        call["memory_id"]
        for call in fake_memory.deprecate_calls
    ] == [
        "mem_002",
        "mem_001",
    ]

    assert all(
        memory.metadata.status == "deprecated"
        for memory in fake_memory.memories.values()
    )


def test_ingestion_reports_rollback_failure(
    loaded_sample: NeedlePersonaSample,
) -> None:
    fake_memory = FakeMemoryService(
        fail_on_create_call=3,
        fail_on_deprecate_ids={"mem_001"},
    )

    service, _, _ = build_service(
        loaded_sample,
        memory_service=fake_memory,
    )

    with pytest.raises(
        NeedlePersonaIngestionError,
        match="Rollback also reported.*mem_001",
    ) as error_info:
        service.ingest(
            sample=loaded_sample,
            run_id="run_rollback_failure",
        )

    assert error_info.value.rollback_errors
    assert "mem_001" in error_info.value.rollback_errors[0]


def test_ingestion_requires_at_least_one_memory_when_configured(
    loaded_sample: NeedlePersonaSample,
) -> None:
    outputs = build_extraction_outputs(loaded_sample)

    outputs[0] = MemoryExtractionOutput(
        agent_id="alice_agent",
        memories=[],
        extraction_summary="Nothing extracted.",
    )

    service, _, fake_memory = build_service(
        loaded_sample,
        outputs=outputs,
        require_memory_per_source=True,
    )

    with pytest.raises(
        NeedlePersonaIngestionError,
        match="no memory candidates",
    ):
        service.ingest(
            sample=loaded_sample,
            run_id="run_001",
        )

    assert fake_memory.create_calls == []


def test_private_memory_index_contains_no_raw_task_content(
    loaded_sample: NeedlePersonaSample,
) -> None:
    service, _, _ = build_service(loaded_sample)

    index = service.ingest(
        sample=loaded_sample,
        run_id="run_001",
    )

    serialised = index.model_dump_json()

    assert loaded_sample.question not in serialised
    assert loaded_sample.gold_answer not in serialised

    for persona in PERSONA_NAMES:
        for source in loaded_sample.get_sources(persona):
            assert source.content not in serialised


# ---------------------------------------------------------------------------
# Optional smoke test using the real generated dataset
# ---------------------------------------------------------------------------


def test_real_generated_dataset_can_load_first_sample() -> None:
    """
    Optional local smoke test.

    Set NEEDLE_DATASET_PATH when the dataset is stored somewhere other than
    data/dataset_2hop.jsonl. This test never calls the LLM.
    """
    dataset = Path(
        os.getenv(
            "NEEDLE_DATASET_PATH",
            "data/informativebench_data/dataset_2hop.jsonl",
        )
    )

    if not dataset.exists():
        pytest.skip(
            "Real dataset not found. Set NEEDLE_DATASET_PATH "
            "to enable this smoke test."
        )

    loader = NeedlePersonaLoader(dataset)

    samples = loader.load_all(limit=1)

    assert len(samples) == 1
    assert samples[0].hop == 2
    assert all(
        samples[0].get_sources(persona)
        for persona in PERSONA_NAMES
    )