from __future__ import annotations

import json
import re
from typing import Any, Optional

from pydantic import ValidationError

from .output_schemas import (
    AgentAnswer,
    CriticOutput,
    CoordinatorOutput,
)


class LLMOutputParseError(Exception):
    """Raised when raw LLM output cannot be parsed into the expected schema."""


class LLMOutputParser:
    """
    Utility parser and normalizer for LLM outputs.

    Main responsibilities:
    - normalize short answers for QA evaluation
    - clamp confidence scores
    - clean evidence chunk ids
    - parse JSON-like raw LLM outputs as fallback
    - convert raw dict/text into Pydantic output schemas

    This class does not call LLMs.
    """

    # ------------------------------------------------------------------
    # Answer normalization
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_short_answer(answer: str) -> str:
        """
        Clean a short answer span returned by the LLM.

        This is not the same as SQuAD EM/F1 normalization.
        It only removes common LLM formatting artifacts before evaluation.

        Examples:
            "The answer is Beyoncé." -> "Beyoncé"
            '"in the late 1990s."' -> "in the late 1990s"
        """

        if answer is None:
            return ""

        text = str(answer).strip()

        if not text:
            return ""

        # Remove wrapping quotes.
        text = text.strip("\"'“”‘’")

        # Remove common answer prefixes.
        prefixes = [
            r"^the answer is\s+",
            r"^answer:\s*",
            r"^final answer:\s*",
            r"^final_answer:\s*",
            r"^prediction:\s*",
        ]

        for pattern in prefixes:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

        # Remove markdown bold markers.
        text = text.replace("**", "").strip()

        # Remove trailing punctuation that often appears in generated answers.
        # Keep punctuation that may be internal, e.g. "U.S."
        text = text.strip()

        if len(text) > 1 and text[-1] in {".", "。"}:
            text = text[:-1].strip()

        return text

    @staticmethod
    def normalize_reasoning(reasoning: str) -> str:
        """
        Clean reasoning text.
        """

        if reasoning is None:
            return ""

        return str(reasoning).strip()

    # ------------------------------------------------------------------
    # Confidence normalization
    # ------------------------------------------------------------------

    @staticmethod
    def clamp_confidence(confidence: Any, default: float = 0.5) -> float:
        """
        Convert confidence to float and clamp it to [0.0, 1.0].
        """

        try:
            value = float(confidence)
        except (TypeError, ValueError):
            value = default

        if value < 0.0:
            return 0.0

        if value > 1.0:
            return 1.0

        return value

    # ------------------------------------------------------------------
    # Evidence ID normalization
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_evidence_chunk_ids(
        evidence_chunk_ids: Optional[list[Any]],
    ) -> list[str]:
        """
        Clean evidence chunk ids.

        Removes:
        - None
        - empty strings
        - duplicates
        """

        if not evidence_chunk_ids:
            return []

        cleaned: list[str] = []

        for item in evidence_chunk_ids:
            text = str(item).strip()

            if text and text not in cleaned:
                cleaned.append(text)

        return cleaned

    # ------------------------------------------------------------------
    # JSON fallback parsing
    # ------------------------------------------------------------------

    @staticmethod
    def extract_json_object(raw_output: str) -> dict[str, Any]:
        """
        Extract a JSON object from raw LLM text.

        Supports:
        - pure JSON string
        - markdown fenced JSON block
        - text containing one JSON object

        Raises:
            LLMOutputParseError if no valid JSON object can be parsed.
        """

        if raw_output is None:
            raise LLMOutputParseError("Raw output is None.")

        text = str(raw_output).strip()

        if not text:
            raise LLMOutputParseError("Raw output is empty.")

        # Remove fenced code block if present.
        fenced_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        if fenced_match:
            text = fenced_match.group(1).strip()

        # Try direct JSON parse.
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Try extracting first {...} object.
        object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)

        if object_match:
            json_text = object_match.group(0)

            try:
                parsed = json.loads(json_text)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError as error:
                raise LLMOutputParseError(
                    f"Found JSON-like object but failed to parse it: {error}"
                ) from error

        raise LLMOutputParseError("Could not extract a valid JSON object.")

    # ------------------------------------------------------------------
    # Schema normalization helpers
    # ------------------------------------------------------------------

    @classmethod
    def normalize_agent_answer_dict(
        cls,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Normalize raw dict fields for AgentAnswer.
        """

        return {
            "answer": cls.normalize_short_answer(data.get("answer", "")),
            "reasoning": cls.normalize_reasoning(data.get("reasoning", "")),
            "confidence": cls.clamp_confidence(data.get("confidence", 0.5)),
            "evidence_chunk_ids": cls.normalize_evidence_chunk_ids(
                data.get("evidence_chunk_ids", [])
            ),
        }

    @classmethod
    def normalize_critic_output_dict(
        cls,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Normalize raw dict fields for CriticOutput.
        """

        preferred_worker = str(
            data.get("preferred_worker", "uncertain")
        ).strip().lower()

        if preferred_worker not in {"worker_a", "worker_b", "uncertain"}:
            preferred_worker = "uncertain"

        return {
            "recommended_answer": cls.normalize_short_answer(
                data.get("recommended_answer", "")
            ),
            "preferred_worker": preferred_worker,
            "comment": cls.normalize_reasoning(data.get("comment", "")),
            "confidence": cls.clamp_confidence(data.get("confidence", 0.5)),
        }

    @classmethod
    def normalize_coordinator_output_dict(
        cls,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Normalize raw dict fields for CoordinatorOutput.
        """

        # Support both final_answer and answer as fallback.
        final_answer = data.get("final_answer", data.get("answer", ""))

        return {
            "final_answer": cls.normalize_short_answer(final_answer),
            "reasoning": cls.normalize_reasoning(data.get("reasoning", "")),
            "confidence": cls.clamp_confidence(data.get("confidence", 0.5)),
        }

    # ------------------------------------------------------------------
    # Parse into Pydantic schemas
    # ------------------------------------------------------------------

    @classmethod
    def parse_agent_answer(
        cls,
        raw_output: str | dict[str, Any],
    ) -> AgentAnswer:
        """
        Parse raw output into AgentAnswer.
        """

        data = (
            raw_output
            if isinstance(raw_output, dict)
            else cls.extract_json_object(raw_output)
        )

        normalized = cls.normalize_agent_answer_dict(data)

        try:
            return AgentAnswer.model_validate(normalized)
        except ValidationError as error:
            raise LLMOutputParseError(
                f"Failed to parse AgentAnswer: {error}"
            ) from error

    @classmethod
    def parse_critic_output(
        cls,
        raw_output: str | dict[str, Any],
    ) -> CriticOutput:
        """
        Parse raw output into CriticOutput.
        """

        data = (
            raw_output
            if isinstance(raw_output, dict)
            else cls.extract_json_object(raw_output)
        )

        normalized = cls.normalize_critic_output_dict(data)

        try:
            return CriticOutput.model_validate(normalized)
        except ValidationError as error:
            raise LLMOutputParseError(
                f"Failed to parse CriticOutput: {error}"
            ) from error

    @classmethod
    def parse_coordinator_output(
        cls,
        raw_output: str | dict[str, Any],
    ) -> CoordinatorOutput:
        """
        Parse raw output into CoordinatorOutput.
        """

        data = (
            raw_output
            if isinstance(raw_output, dict)
            else cls.extract_json_object(raw_output)
        )

        normalized = cls.normalize_coordinator_output_dict(data)

        try:
            return CoordinatorOutput.model_validate(normalized)
        except ValidationError as error:
            raise LLMOutputParseError(
                f"Failed to parse CoordinatorOutput: {error}"
            ) from error

    # ------------------------------------------------------------------
    # Post-process already structured outputs
    # ------------------------------------------------------------------

    @classmethod
    def clean_agent_answer(
        cls,
        answer: AgentAnswer,
    ) -> AgentAnswer:
        """
        Clean an AgentAnswer returned by structured output.
        """

        return AgentAnswer(
            answer=cls.normalize_short_answer(answer.answer),
            reasoning=cls.normalize_reasoning(answer.reasoning),
            confidence=cls.clamp_confidence(answer.confidence),
            evidence_chunk_ids=cls.normalize_evidence_chunk_ids(
                answer.evidence_chunk_ids
            ),
        )

    @classmethod
    def clean_critic_output(
        cls,
        output: CriticOutput,
    ) -> CriticOutput:
        """
        Clean a CriticOutput returned by structured output.
        """

        return CriticOutput(
            recommended_answer=cls.normalize_short_answer(
                output.recommended_answer
            ),
            preferred_worker=output.preferred_worker,
            comment=cls.normalize_reasoning(output.comment),
            confidence=cls.clamp_confidence(output.confidence),
        )

    @classmethod
    def clean_coordinator_output(
        cls,
        output: CoordinatorOutput,
    ) -> CoordinatorOutput:
        """
        Clean a CoordinatorOutput returned by structured output.
        """

        return CoordinatorOutput(
            final_answer=cls.normalize_short_answer(output.final_answer),
            reasoning=cls.normalize_reasoning(output.reasoning),
            confidence=cls.clamp_confidence(output.confidence),
        )