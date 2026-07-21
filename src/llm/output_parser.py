from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar, cast

from pydantic import BaseModel, ValidationError

from .output_schemas import (
    AgentAnswer,
    ExtractedMemory,
    LLMCallMetadata,
    MemoryExtractionOutput,
    MemoryReviewOutput,
    PromotionDecisionOutput,
    TaskRoutingOutput,
)


SchemaT = TypeVar("SchemaT", bound=BaseModel)
Normalizer = Callable[[dict[str, Any]], dict[str, Any]]


class LLMOutputParseError(Exception):
    """
    Raised when raw LLM output cannot be converted into the expected schema.
    """


class LLMOutputParser:
    """
    Generic parser and normalizer for structured LLM outputs.

    Responsibilities:
    - extract a JSON object from raw LLM text;
    - normalize common field types;
    - apply schema-specific cleanup when needed;
    - validate the result with a supplied Pydantic model.

    The parser is independent of any specific workflow, dataset, agent name,
    prompt template, or LLM provider.
    """

    @staticmethod
    def normalize_text(
        value: Any,
        *,
        default: str = "",
    ) -> str:
        if value is None:
            return default

        text = str(value).strip()
        return text if text else default

    @classmethod
    def normalize_optional_text(
        cls,
        value: Any,
    ) -> str | None:
        text = cls.normalize_text(value)
        return text or None

    @classmethod
    def normalize_short_answer(
        cls,
        answer: Any,
    ) -> str:
        """
        Remove common formatting artefacts from a short task answer.

        Dataset-specific EM/F1 normalization belongs in the evaluator.
        """
        text = cls.normalize_text(answer)

        if not text:
            return ""

        text = text.strip("\"'“”‘’")

        prefixes = (
            r"^the answer is\s+",
            r"^answer:\s*",
            r"^final answer:\s*",
            r"^final_answer:\s*",
            r"^prediction:\s*",
        )

        for pattern in prefixes:
            text = re.sub(
                pattern,
                "",
                text,
                flags=re.IGNORECASE,
            ).strip()

        text = text.replace("**", "").strip()

        if len(text) > 1 and text[-1] in {".", "。"}:
            text = text[:-1].strip()

        return text

    @staticmethod
    def clamp_float(
        value: Any,
        *,
        default: float,
        minimum: float = 0.0,
        maximum: float = 1.0,
    ) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = default

        return max(minimum, min(maximum, number))

    @classmethod
    def clamp_confidence(
        cls,
        value: Any,
        *,
        default: float = 0.5,
    ) -> float:
        return cls.clamp_float(
            value,
            default=default,
            minimum=0.0,
            maximum=1.0,
        )

    @staticmethod
    def normalize_bool(
        value: Any,
        *,
        default: bool = False,
    ) -> bool:
        if isinstance(value, bool):
            return value

        if value is None:
            return default

        if isinstance(value, (int, float)):
            return bool(value)

        text = str(value).strip().lower()

        if text in {"true", "yes", "y", "1", "on"}:
            return True

        if text in {"false", "no", "n", "0", "off"}:
            return False

        return default

    @staticmethod
    def normalize_choice(
        value: Any,
        *,
        allowed: set[str],
        default: str,
        aliases: Mapping[str, str] | None = None,
    ) -> str:
        text = str(value).strip().lower() if value is not None else ""

        if aliases and text in aliases:
            text = aliases[text]

        return text if text in allowed else default

    @classmethod
    def normalize_string_list(
        cls,
        values: Any,
    ) -> list[str]:
        """
        Convert a value into an ordered, deduplicated list of strings.
        """
        if values is None:
            return []

        if isinstance(values, str):
            stripped = values.strip()

            if not stripped:
                return []

            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                decoded = None

            if isinstance(decoded, list):
                values = decoded
            else:
                values = [stripped]

        if not isinstance(values, Sequence) or isinstance(
            values,
            (bytes, bytearray),
        ):
            values = [values]

        cleaned: list[str] = []

        for item in values:
            text = cls.normalize_text(item)
            if text and text not in cleaned:
                cleaned.append(text)

        return cleaned

    @classmethod
    def extract_json_object(
        cls,
        raw_output: str,
    ) -> dict[str, Any]:
        """
        Extract the first valid JSON object from raw LLM text.
        """
        if raw_output is None:
            raise LLMOutputParseError("Raw output is None.")

        text = str(raw_output).strip()

        if not text:
            raise LLMOutputParseError("Raw output is empty.")

        fenced_match = re.search(
            r"```(?:json)?\s*(.*?)\s*```",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        if fenced_match:
            fenced_text = fenced_match.group(1).strip()
            try:
                parsed = json.loads(fenced_text)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(parsed, dict):
                    return cast(dict[str, Any], parsed)

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, dict):
            return cast(dict[str, Any], parsed)

        decoder = json.JSONDecoder()

        for index, character in enumerate(text):
            if character != "{":
                continue

            try:
                parsed, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue

            if isinstance(parsed, dict):
                return cast(dict[str, Any], parsed)

        raise LLMOutputParseError(
            "Could not extract a valid JSON object from the LLM output."
        )

    @classmethod
    def to_mapping(
        cls,
        raw_output: str | Mapping[str, Any] | BaseModel,
    ) -> dict[str, Any]:
        if isinstance(raw_output, BaseModel):
            return dict(raw_output.model_dump())

        if isinstance(raw_output, Mapping):
            return dict(raw_output)

        return cls.extract_json_object(raw_output)

    @classmethod
    def normalize_agent_answer_dict(
        cls,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "answer": cls.normalize_short_answer(data.get("answer")),
            "reasoning": cls.normalize_text(data.get("reasoning")),
            "confidence": cls.clamp_confidence(data.get("confidence")),
            "used_memory_ids": cls.normalize_string_list(
                data.get("used_memory_ids")
            ),
            "supporting_source_ids": cls.normalize_string_list(
                data.get("supporting_source_ids")
            ),
            "contributing_agent_ids": cls.normalize_string_list(
                data.get("contributing_agent_ids")
            ),
        }

    @classmethod
    def normalize_extracted_memory_dict(
        cls,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "content": cls.normalize_text(data.get("content")),
            "subject": cls.normalize_optional_text(data.get("subject")),
            "memory_type": cls.normalize_choice(
                data.get("memory_type"),
                allowed={"semantic", "episodic", "procedural"},
                default="semantic",
            ),
            "source_ids": cls.normalize_string_list(data.get("source_ids")),
            "importance": cls.clamp_float(
                data.get("importance"),
                default=0.5,
            ),
            "shareable": cls.normalize_bool(
                data.get("shareable"),
                default=True,
            ),
            "confidence": cls.clamp_confidence(data.get("confidence")),
        }

    @classmethod
    def normalize_memory_extraction_output_dict(
        cls,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        raw_memories = data.get("memories", [])

        if isinstance(raw_memories, Mapping):
            raw_memories = [raw_memories]

        normalized_memories: list[dict[str, Any]] = []

        if isinstance(raw_memories, Sequence) and not isinstance(
            raw_memories,
            (str, bytes, bytearray),
        ):
            for item in raw_memories:
                if isinstance(item, BaseModel):
                    item = item.model_dump()

                if isinstance(item, Mapping):
                    normalized_memories.append(
                        cls.normalize_extracted_memory_dict(dict(item))
                    )

        return {
            "agent_id": cls.normalize_text(data.get("agent_id")),
            "memories": normalized_memories,
            "extraction_summary": cls.normalize_optional_text(
                data.get("extraction_summary")
            ),
        }

    @classmethod
    def normalize_memory_review_output_dict(
        cls,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "memory_id": cls.normalize_text(data.get("memory_id")),
            "classification": cls.normalize_choice(
                data.get("classification"),
                allowed={
                    "new",
                    "duplicate",
                    "conflict",
                    "outdated",
                    "irrelevant",
                    "policy_violation",
                    "uncertain",
                },
                default="uncertain",
                aliases={
                    "policy violation": "policy_violation",
                    "policy-violation": "policy_violation",
                },
            ),
            "recommendation": cls.normalize_choice(
                data.get("recommendation"),
                allowed={
                    "approve",
                    "reject",
                    "merge",
                    "supersede",
                    "keep_private",
                },
                default="reject",
                aliases={
                    "keep private": "keep_private",
                    "keep-private": "keep_private",
                },
            ),
            "relevant": cls.normalize_bool(
                data.get("relevant"),
                default=False,
            ),
            "policy_compliant": cls.normalize_bool(
                data.get("policy_compliant"),
                default=False,
            ),
            "related_memory_ids": cls.normalize_string_list(
                data.get("related_memory_ids")
            ),
            "reason": cls.normalize_text(data.get("reason")),
            "confidence": cls.clamp_confidence(data.get("confidence")),
        }

    @classmethod
    def normalize_task_routing_output_dict(
        cls,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        selected_agent_ids = cls.normalize_string_list(
            data.get("selected_agent_ids")
        )
        responder_agent_id = cls.normalize_text(
            data.get("responder_agent_id")
        )

        if responder_agent_id and responder_agent_id not in selected_agent_ids:
            selected_agent_ids.append(responder_agent_id)

        return {
            "selected_agent_ids": selected_agent_ids,
            "responder_agent_id": responder_agent_id,
            "required_information": cls.normalize_string_list(
                data.get("required_information")
            ),
            "reason": cls.normalize_text(data.get("reason")),
            "confidence": cls.clamp_confidence(data.get("confidence")),
        }

    @classmethod
    def normalize_promotion_decision_output_dict(
        cls,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        decision = cls.normalize_choice(
            data.get("decision"),
            allowed={
                "approve",
                "reject",
                "merge",
                "supersede",
                "keep_private",
            },
            default="reject",
            aliases={
                "keep private": "keep_private",
                "keep-private": "keep_private",
            },
        )

        default_scope = (
            "private"
            if decision in {"reject", "keep_private"}
            else "shared"
        )

        target_scope = cls.normalize_choice(
            data.get("target_scope"),
            allowed={"private", "shared"},
            default=default_scope,
        )

        allowed_agent_ids = cls.normalize_string_list(
            data.get("allowed_agent_ids")
        )

        if decision in {"reject", "keep_private"}:
            target_scope = "private"

        if target_scope == "private":
            allowed_agent_ids = []

        return {
            "memory_id": cls.normalize_text(data.get("memory_id")),
            "decision": decision,
            "target_scope": target_scope,
            "allowed_agent_ids": allowed_agent_ids,
            "related_memory_ids": cls.normalize_string_list(
                data.get("related_memory_ids")
            ),
            "reason": cls.normalize_text(data.get("reason")),
            "confidence": cls.clamp_confidence(data.get("confidence")),
        }

    @classmethod
    def normalize_llm_call_metadata_dict(
        cls,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "agent_id": cls.normalize_text(data.get("agent_id")),
            "role": cls.normalize_optional_text(data.get("role")),
            "model_name": cls.normalize_text(data.get("model_name")),
            "prompt_preview": cls.normalize_optional_text(
                data.get("prompt_preview")
            ),
            "raw_output": cls.normalize_optional_text(
                data.get("raw_output")
            ),
            "success": cls.normalize_bool(
                data.get("success"),
                default=True,
            ),
            "error_message": cls.normalize_optional_text(
                data.get("error_message")
            ),
            "latency_ms": (
                None
                if data.get("latency_ms") is None
                else max(0.0, float(data.get("latency_ms")))
            ),
            "input_tokens": (
                None
                if data.get("input_tokens") is None
                else max(0, int(data.get("input_tokens")))
            ),
            "output_tokens": (
                None
                if data.get("output_tokens") is None
                else max(0, int(data.get("output_tokens")))
            ),
        }

    @classmethod
    def get_normalizer(
        cls,
        schema_model: type[SchemaT],
    ) -> Normalizer | None:
        normalizers: dict[type[BaseModel], Normalizer] = {
            AgentAnswer: cls.normalize_agent_answer_dict,
            ExtractedMemory: cls.normalize_extracted_memory_dict,
            MemoryExtractionOutput: (
                cls.normalize_memory_extraction_output_dict
            ),
            MemoryReviewOutput: cls.normalize_memory_review_output_dict,
            TaskRoutingOutput: cls.normalize_task_routing_output_dict,
            PromotionDecisionOutput: (
                cls.normalize_promotion_decision_output_dict
            ),
            LLMCallMetadata: cls.normalize_llm_call_metadata_dict,
        }

        return normalizers.get(schema_model)

    @classmethod
    def parse_as(
        cls,
        raw_output: str | Mapping[str, Any] | BaseModel,
        schema_model: type[SchemaT],
        *,
        normalizer: Normalizer | None = None,
        apply_default_normalizer: bool = True,
    ) -> SchemaT:
        """
        Parse raw output into any supplied Pydantic schema.
        """
        data = cls.to_mapping(raw_output)

        selected_normalizer = normalizer

        if selected_normalizer is None and apply_default_normalizer:
            selected_normalizer = cls.get_normalizer(schema_model)

        if selected_normalizer is not None:
            data = selected_normalizer(data)

        try:
            return schema_model.model_validate(data)
        except ValidationError as error:
            raw_preview = cls.normalize_text(raw_output)[:500]

            raise LLMOutputParseError(
                f"Failed to parse {schema_model.__name__}: {error}. "
                f"Raw output preview: {raw_preview!r}"
            ) from error

    @classmethod
    def clean_model(
        cls,
        model: SchemaT,
    ) -> SchemaT:
        """
        Re-normalize and revalidate an already structured Pydantic output.
        """
        return cls.parse_as(
            model,
            cast(type[SchemaT], type(model)),
        )

    @classmethod
    def parse_agent_answer(
        cls,
        raw_output: str | Mapping[str, Any] | BaseModel,
    ) -> AgentAnswer:
        return cls.parse_as(raw_output, AgentAnswer)

    @classmethod
    def parse_extracted_memory(
        cls,
        raw_output: str | Mapping[str, Any] | BaseModel,
    ) -> ExtractedMemory:
        return cls.parse_as(raw_output, ExtractedMemory)

    @classmethod
    def parse_memory_extraction_output(
        cls,
        raw_output: str | Mapping[str, Any] | BaseModel,
    ) -> MemoryExtractionOutput:
        return cls.parse_as(raw_output, MemoryExtractionOutput)

    @classmethod
    def parse_memory_review_output(
        cls,
        raw_output: str | Mapping[str, Any] | BaseModel,
    ) -> MemoryReviewOutput:
        return cls.parse_as(raw_output, MemoryReviewOutput)

    @classmethod
    def parse_task_routing_output(
        cls,
        raw_output: str | Mapping[str, Any] | BaseModel,
    ) -> TaskRoutingOutput:
        return cls.parse_as(raw_output, TaskRoutingOutput)

    @classmethod
    def parse_promotion_decision_output(
        cls,
        raw_output: str | Mapping[str, Any] | BaseModel,
    ) -> PromotionDecisionOutput:
        return cls.parse_as(raw_output, PromotionDecisionOutput)

    @classmethod
    def parse_llm_call_metadata(
        cls,
        raw_output: str | Mapping[str, Any] | BaseModel,
    ) -> LLMCallMetadata:
        return cls.parse_as(raw_output, LLMCallMetadata)