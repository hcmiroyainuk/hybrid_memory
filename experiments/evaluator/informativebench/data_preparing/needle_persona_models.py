from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


PersonaName = Literal["alice", "bob", "charlie", "dave"]
SourceType = Literal[
    "persona",
    "profile",
    "dialogue_turn",
    "conversation_summary",
    "other",
]

PERSONA_NAMES: tuple[PersonaName, ...] = (
    "alice",
    "bob",
    "charlie",
    "dave",
)

PERSONA_AGENT_MAP: dict[PersonaName, str] = {
    "alice": "alice_agent",
    "bob": "bob_agent",
    "charlie": "charlie_agent",
    "dave": "dave_agent",
}

AGENT_PERSONA_MAP: dict[str, PersonaName] = {
    agent_id: persona
    for persona, agent_id in PERSONA_AGENT_MAP.items()
}


def _clean_string_list(values: list[str]) -> list[str]:
    """
    Strip whitespace, remove empty values, and preserve order while
    deduplicating.
    """
    cleaned: list[str] = []

    for value in values:
        item = str(value).strip()

        if item and item not in cleaned:
            cleaned.append(item)

    return cleaned


from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)


ContributionType = Literal[
    "direct_answer",
    "partial_hop",
    "context_only",
    "none",
]


class NeedleEvidenceAssessment(BaseModel):
    """
    Critic assessment of one candidate memory's incremental
    contribution to the active benchmark question.

    This model does not make a governance decision.
    """

    memory_id: str = Field(
        description=(
            "Exact ID of the candidate memory."
        )
    )

    contribution_type: ContributionType = Field(
        description=(
            "How the candidate contributes to answering "
            "the active question."
        )
    )

    supported_question_component: str | None = Field(
        default=None,
        description=(
            "The specific part of the question supported "
            "by this candidate."
        ),
    )

    evidence_spans: list[str] = Field(
        default_factory=list,
        description=(
            "Exact short spans from the candidate memory "
            "that provide the contribution."
        ),
    )

    reason: str = Field(
        description=(
            "Concise explanation of the evidence "
            "contribution assessment."
        )
    )

    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    @field_validator(
        "memory_id",
        "reason",
    )
    @classmethod
    def required_text(
        cls,
        value: str,
    ) -> str:
        cleaned = str(value).strip()

        if not cleaned:
            raise ValueError(
                "Required text cannot be empty."
            )

        return cleaned

    @field_validator(
        "supported_question_component",
    )
    @classmethod
    def clean_optional_text(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        cleaned = str(value).strip()
        return cleaned or None

    @field_validator(
        "evidence_spans",
    )
    @classmethod
    def clean_evidence_spans(
        cls,
        values: list[str],
    ) -> list[str]:
        cleaned: list[str] = []

        for value in values:
            text = str(value).strip()

            if text and text not in cleaned:
                cleaned.append(text)

        return cleaned

    @model_validator(mode="after")
    def validate_contribution(
        self,
    ) -> "NeedleEvidenceAssessment":
        useful_types = {
            "direct_answer",
            "partial_hop",
        }

        if self.contribution_type in useful_types:
            if not self.supported_question_component:
                raise ValueError(
                    "A useful contribution requires "
                    "supported_question_component."
                )

            if not self.evidence_spans:
                raise ValueError(
                    "A useful contribution requires at "
                    "least one evidence span."
                )

        if self.contribution_type == "none":
            self.supported_question_component = None
            self.evidence_spans = []

        return self

class PersonaSource(BaseModel):
    """
    One source item owned by a persona agent.

    A source is input to the private-memory ingestion stage. It is not itself
    a persisted memory. The ingestion service may extract one or more memories
    from it.

    Each source must belong to exactly one persona so that the loader cannot
    accidentally expose another persona's information to the wrong agent.
    """

    source_id: str = Field(
        description="Unique source identifier within the benchmark sample."
    )

    persona: PersonaName = Field(
        description="Persona that owns and is allowed to ingest this source."
    )

    content: str = Field(
        description="Text supplied to the persona agent during ingestion."
    )

    source_type: SourceType = Field(
        default="other",
        description="Type of benchmark information represented by this source.",
    )

    conversation_id: str | None = Field(
        default=None,
        description="Optional identifier of the originating conversation.",
    )

    turn_index: int | None = Field(
        default=None,
        ge=0,
        description="Optional zero-based position of a dialogue turn.",
    )

    speaker: str | None = Field(
        default=None,
        description="Optional original speaker name.",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Dataset-specific metadata that should not affect ownership.",
    )

    @field_validator("source_id", "content")
    @classmethod
    def required_text_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Required source text cannot be empty.")

        return value

    @field_validator("conversation_id", "speaker")
    @classmethod
    def clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        value = value.strip()
        return value or None


class NeedlePersonaSample(BaseModel):
    """
    Workflow-independent representation of one Needle in the Persona sample.

    The loader is responsible for converting the official dataset format into
    this model. Downstream ingestion and workflow code should depend on this
    stable representation rather than on the benchmark's raw JSON keys.
    """

    sample_id: str = Field(
        description="Stable identifier for the benchmark sample."
    )

    question: str = Field(
        description="Question presented to the multi-agent task workflow."
    )

    gold_answer: str = Field(
        description="Primary reference answer used for evaluation."
    )

    alternative_answers: list[str] = Field(
        default_factory=list,
        description="Optional additional answers accepted by the evaluator.",
    )

    hop: Literal[1, 2] | None = Field(
        default=None,
        description="Needle sample type when provided by the source dataset.",
    )

    persona_sources: dict[PersonaName, list[PersonaSource]] = Field(
        default_factory=lambda: {
            persona: []
            for persona in PERSONA_NAMES
        },
        description=(
            "Information partitioned by persona ownership. "
            "Every source under a key must belong to that persona."
        ),
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Original dataset metadata retained for traceability.",
    )

    @field_validator("sample_id", "question", "gold_answer")
    @classmethod
    def required_sample_text_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Required sample text cannot be empty.")

        return value

    @field_validator("alternative_answers")
    @classmethod
    def clean_alternative_answers(cls, value: list[str]) -> list[str]:
        return _clean_string_list(value)

    @model_validator(mode="after")
    def validate_persona_partitions(self) -> "NeedlePersonaSample":
        """
        Ensure all four persona partitions exist and that ownership is
        consistent with the partition key.
        """
        normalised_sources: dict[PersonaName, list[PersonaSource]] = {
            persona: list(self.persona_sources.get(persona, []))
            for persona in PERSONA_NAMES
        }

        seen_source_ids: set[str] = set()

        for persona, sources in normalised_sources.items():
            for source in sources:
                if source.persona != persona:
                    raise ValueError(
                        f"Source {source.source_id!r} belongs to "
                        f"{source.persona!r} but was placed under "
                        f"{persona!r}."
                    )

                if source.source_id in seen_source_ids:
                    raise ValueError(
                        f"Duplicate source_id in sample: "
                        f"{source.source_id!r}."
                    )

                seen_source_ids.add(source.source_id)

        self.persona_sources = normalised_sources
        return self

    def get_sources(
        self,
        persona: PersonaName,
    ) -> list[PersonaSource]:
        """
        Return a copy of the sources owned by one persona.
        """
        return list(self.persona_sources.get(persona, []))

    def get_agent_sources(
        self,
        agent_id: str,
    ) -> list[PersonaSource]:
        """
        Return sources for a registered persona agent.
        """
        persona = AGENT_PERSONA_MAP.get(agent_id)

        if persona is None:
            raise KeyError(f"Unknown persona agent: {agent_id!r}.")

        return self.get_sources(persona)

    def accepted_answers(self) -> list[str]:
        """
        Return all accepted answers with the primary answer first.
        """
        return _clean_string_list(
            [self.gold_answer, *self.alternative_answers]
        )


class PrivateMemoryIndex(BaseModel):
    """
    Result of the private-memory ingestion stage for one experiment run.

    Only memory IDs are passed into the task workflow. Raw persona sources and
    conversations remain outside the LangGraph task state to prevent context
    leakage.
    """

    run_id: str = Field(
        description="Identifier of the experiment run."
    )

    sample_id: str = Field(
        description="Identifier of the ingested benchmark sample."
    )

    private_memory_ids: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Private memory IDs grouped by persona agent ID.",
    )

    source_to_memory_ids: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Traceability mapping from source IDs to created memory IDs.",
    )

    @field_validator("run_id", "sample_id")
    @classmethod
    def required_index_text_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Required index field cannot be empty.")

        return value

    @field_validator(
        "private_memory_ids",
        "source_to_memory_ids",
    )
    @classmethod
    def clean_memory_id_mappings(
        cls,
        value: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        cleaned: dict[str, list[str]] = {}

        for key, memory_ids in value.items():
            clean_key = str(key).strip()

            if not clean_key:
                raise ValueError("Memory index keys cannot be empty.")

            cleaned[clean_key] = _clean_string_list(memory_ids)

        return cleaned

    @model_validator(mode="after")
    def validate_persona_agent_ids(self) -> "PrivateMemoryIndex":
        unknown_agent_ids = (
            set(self.private_memory_ids)
            - set(AGENT_PERSONA_MAP)
        )

        if unknown_agent_ids:
            raise ValueError(
                "Unknown persona agent IDs in private_memory_ids: "
                f"{sorted(unknown_agent_ids)}."
            )

        for agent_id in AGENT_PERSONA_MAP:
            self.private_memory_ids.setdefault(agent_id, [])

        return self


__all__ = [
    "PersonaName",
    "SourceType",
    "PERSONA_NAMES",
    "PERSONA_AGENT_MAP",
    "AGENT_PERSONA_MAP",
    "PersonaSource",
    "NeedlePersonaSample",
    "PrivateMemoryIndex",
]