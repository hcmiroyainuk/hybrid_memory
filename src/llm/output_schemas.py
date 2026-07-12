from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AgentAnswer(BaseModel):
    """
    Structured answer produced by a worker agent.

    Used by:
    - Worker A: direct parametric QA agent
    - Worker B: retrieval-grounded QA agent

    For Worker A:
        evidence_chunk_ids is usually empty.

    For Worker B:
        evidence_chunk_ids should contain the retrieved chunks used as evidence.
    """

    answer: str = Field(
        description="Short answer span for evaluation. Do not include explanation."
    )

    reasoning: str = Field(
        description="Brief reasoning behind the answer."
    )

    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )

    evidence_chunk_ids: list[str] = Field(
        default_factory=list,
        description="IDs of retrieved chunks used as evidence. Empty for Worker A.",
    )

    @field_validator("answer")
    @classmethod
    def answer_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("answer cannot be empty.")

        return value

    @field_validator("reasoning")
    @classmethod
    def reasoning_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("reasoning cannot be empty.")

        return value

    @field_validator("evidence_chunk_ids")
    @classmethod
    def clean_evidence_chunk_ids(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []

        for item in value:
            item = str(item).strip()

            if item and item not in cleaned:
                cleaned.append(item)

        return cleaned


class CriticOutput(BaseModel):
    """
    Structured output produced by the critic agent.

    The critic compares Worker A and Worker B, checks whether Worker B's answer
    is supported by retrieved evidence, and recommends one answer.
    """

    recommended_answer: str = Field(
        description="Short answer recommended by the critic."
    )

    preferred_worker: Literal["worker_a", "worker_b", "uncertain"] = Field(
        description="Which worker's answer is preferred."
    )

    comment: str = Field(
        description="Brief critique explaining the recommendation."
    )

    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )

    @field_validator("recommended_answer")
    @classmethod
    def recommended_answer_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("recommended_answer cannot be empty.")

        return value

    @field_validator("comment")
    @classmethod
    def comment_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("comment cannot be empty.")

        return value


class CoordinatorOutput(BaseModel):
    """
    Structured final output produced by the coordinator agent.

    final_answer is the final prediction used for SQuAD EM / F1 evaluation.
    """

    final_answer: str = Field(
        description="Final short answer span for evaluation. Do not include explanation."
    )

    reasoning: str = Field(
        description="Brief explanation for the final decision."
    )

    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )

    @field_validator("final_answer")
    @classmethod
    def final_answer_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("final_answer cannot be empty.")

        return value

    @field_validator("reasoning")
    @classmethod
    def reasoning_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("reasoning cannot be empty.")

        return value


class LLMCallMetadata(BaseModel):
    """
    Optional metadata for recording one LLM call.

    This is not required by the workflow, but it is useful for experiment logging
    and debugging.
    """

    agent_name: str
    model_name: str
    prompt_preview: str | None = None
    raw_output: str | None = None
    success: bool = True
    error_message: str | None = None


class WorkerAOutput(AgentAnswer):
    """
    Alias schema for Worker A.

    Worker A answers from parametric knowledge only.
    """

    pass


class WorkerBOutput(AgentAnswer):
    """
    Alias schema for Worker B.

    Worker B answers using retrieved external evidence.
    """

    pass