from __future__ import annotations

"""
Generic semantic active-forgetting service.

Pipeline:
1. Parse an explicit forgetting request into a target fact and preserved facts.
2. Retrieve a bounded set of active candidate memories.
3. Semantically classify each candidate as forget/preserve/unrelated/uncertain.
4. Soft-delete high-confidence forget candidates through MemoryService.

The service does not import GateMem and never changes ACLs or physically deletes
memory records.
"""

import json
import logging
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, Field, field_validator, model_validator


SchemaT = TypeVar("SchemaT", bound=BaseModel)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class StructuredLLMProtocol(Protocol):
    def invoke_structured(
        self,
        prompt: str,
        schema_model: type[SchemaT],
        *,
        apply_parser_normalization: bool = True,
    ) -> SchemaT:
        ...


@runtime_checkable
class MemoryStoreProtocol(Protocol):
    def list_active(self) -> list[Any]:
        ...

    def get_by_id(self, memory_id: str) -> Any:
        ...


@runtime_checkable
class MemoryServiceProtocol(Protocol):
    memory_retriever: Any | None

    def retrieve_memories(
        self,
        agent: Any,
        query: str,
        top_k: int = 5,
        include_private: bool = True,
        include_shared: bool = True,
    ) -> list[Any]:
        ...

    def deprecate_memory(
        self,
        agent: Any,
        memory_id: str,
        reason: str | None = None,
    ) -> Any:
        ...


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MemoryForgettingError(RuntimeError):
    pass


class ForgettingIntentError(MemoryForgettingError):
    pass


class ForgettingCandidateError(MemoryForgettingError):
    pass


class ForgettingDecisionError(MemoryForgettingError):
    pass


class ForgettingExecutionError(MemoryForgettingError):
    pass


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ForgettingIntent(BaseModel):
    target_fact: str = Field(
        description=(
            "The exact self-contained fact that must no longer be "
            "retrievable or reconstructable."
        )
    )
    preserved_facts: list[str] = Field(
        default_factory=list,
        description="Current facts explicitly requested to remain available.",
    )
    reasoning: str = ""
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("target_fact")
    @classmethod
    def validate_target_fact(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("target_fact cannot be empty.")
        return value

    @field_validator("preserved_facts")
    @classmethod
    def clean_preserved_facts(cls, values: list[str]) -> list[str]:
        return _deduplicate_strings(values)

    @field_validator("reasoning")
    @classmethod
    def clean_reasoning(cls, value: str) -> str:
        return str(value or "").strip()

    @model_validator(mode="after")
    def target_must_not_be_preserved(self) -> "ForgettingIntent":
        target = _normalise_text(self.target_fact)
        preserved = {_normalise_text(item) for item in self.preserved_facts}
        if target in preserved:
            raise ValueError(
                "target_fact must not also appear in preserved_facts."
            )
        return self


DecisionName = Literal["forget", "preserve", "unrelated", "uncertain"]


class ForgettingCandidateDecision(BaseModel):
    memory_id: str
    decision: DecisionName
    reasoning: str = ""
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("memory_id")
    @classmethod
    def validate_memory_id(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("memory_id cannot be empty.")
        return value

    @field_validator("reasoning")
    @classmethod
    def clean_reasoning(cls, value: str) -> str:
        return str(value or "").strip()


class ForgettingDecisionBatch(BaseModel):
    decisions: list[ForgettingCandidateDecision] = Field(default_factory=list)


class ForgettingResult(BaseModel):
    request_text: str
    source_turn_id: str
    requesting_agent_id: str
    target_fact: str
    preserved_facts: list[str]

    candidate_memory_ids: list[str]
    forgotten_memory_ids: list[str]
    preserved_memory_ids: list[str]
    unrelated_memory_ids: list[str]
    uncertain_memory_ids: list[str]
    warnings: list[str] = Field(default_factory=list)

    @property
    def changed_memory_state(self) -> bool:
        return bool(self.forgotten_memory_ids)


@dataclass(frozen=True, slots=True)
class MemoryForgettingConfig:
    semantic_top_k_per_owner: int = 12
    lexical_top_k: int = 60
    max_candidates: int = 60
    decision_batch_size: int = 8

    minimum_intent_confidence: float = 0.60
    minimum_forget_confidence: float = 0.70
    structured_retry_count: int = 1
    strict_execution: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "semantic_top_k_per_owner",
            "lexical_top_k",
            "max_candidates",
            "decision_batch_size",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be greater than zero.")

        for field_name in (
            "minimum_intent_confidence",
            "minimum_forget_confidence",
        ):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1.")

        if self.structured_retry_count < 0:
            raise ValueError("structured_retry_count cannot be negative.")


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class MemoryForgettingService:
    def __init__(
        self,
        *,
        llm_client: StructuredLLMProtocol,
        config: MemoryForgettingConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if not isinstance(llm_client, StructuredLLMProtocol):
            raise TypeError(
                "llm_client must implement invoke_structured()."
            )

        self.llm_client = llm_client
        self.config = config or MemoryForgettingConfig()
        self.logger = logger or logging.getLogger(
            f"{__name__}.{type(self).__name__}"
        )

    def forget(
        self,
        *,
        request_text: str,
        requesting_agent_id: str,
        source_turn_id: str,
        workers: Mapping[str, Any],
        memory_service: MemoryServiceProtocol,
        memory_store: MemoryStoreProtocol,
    ) -> ForgettingResult:
        request_text = _required_text(request_text, "request_text")
        requesting_agent_id = _required_text(
            requesting_agent_id, "requesting_agent_id"
        )
        source_turn_id = _required_text(source_turn_id, "source_turn_id")

        if not isinstance(workers, Mapping):
            raise TypeError("workers must be a mapping of owner IDs to Agents.")

        intent = self.extract_intent(request_text)
        active_memories = self._load_active_memories(memory_store)

        if not active_memories:
            return self._empty_result(
                request_text=request_text,
                requesting_agent_id=requesting_agent_id,
                source_turn_id=source_turn_id,
                intent=intent,
                warning="No active memories were available for forgetting.",
            )

        candidates, warnings = self.retrieve_candidates(
            intent=intent,
            workers=workers,
            memory_service=memory_service,
            active_memories=active_memories,
        )

        if not candidates:
            return self._empty_result(
                request_text=request_text,
                requesting_agent_id=requesting_agent_id,
                source_turn_id=source_turn_id,
                intent=intent,
                warning=(
                    "No active candidate memory was found for the parsed "
                    "target fact."
                ),
                existing_warnings=warnings,
            )

        decisions, decision_warnings = self.classify_candidates(
            intent=intent,
            candidates=candidates,
        )
        warnings.extend(decision_warnings)

        (
            forgotten_ids,
            preserved_ids,
            unrelated_ids,
            uncertain_ids,
            execution_warnings,
        ) = self._execute_decisions(
            decisions=decisions,
            candidates=candidates,
            workers=workers,
            memory_service=memory_service,
            memory_store=memory_store,
            source_turn_id=source_turn_id,
            requesting_agent_id=requesting_agent_id,
        )
        warnings.extend(execution_warnings)

        result = ForgettingResult(
            request_text=request_text,
            source_turn_id=source_turn_id,
            requesting_agent_id=requesting_agent_id,
            target_fact=intent.target_fact,
            preserved_facts=list(intent.preserved_facts),
            candidate_memory_ids=[
                _memory_id(memory) for memory in candidates
            ],
            forgotten_memory_ids=forgotten_ids,
            preserved_memory_ids=preserved_ids,
            unrelated_memory_ids=unrelated_ids,
            uncertain_memory_ids=uncertain_ids,
            warnings=_deduplicate_strings(warnings),
        )

        self.logger.info(
            "Semantic forgetting complete: source_turn=%s candidates=%d "
            "forgotten=%d preserved=%d unrelated=%d uncertain=%d",
            source_turn_id,
            len(result.candidate_memory_ids),
            len(result.forgotten_memory_ids),
            len(result.preserved_memory_ids),
            len(result.unrelated_memory_ids),
            len(result.uncertain_memory_ids),
        )
        return result

    # ------------------------------------------------------------------
    # Intent
    # ------------------------------------------------------------------

    def extract_intent(self, request_text: str) -> ForgettingIntent:
        prompt = (
            "You parse explicit active-forgetting requests.\n\n"
            "Extract the exact fact that must become unrecoverable and "
            "separate it from current facts that must remain available.\n\n"
            "Rules:\n"
            "- target_fact must be self-contained and precise.\n"
            "- target_fact contains only the old/restricted fact to forget.\n"
            "- preserved_facts contains only facts explicitly requested to "
            "remain current or available.\n"
            "- Do not invent facts.\n"
            "- The target may be a name, value, credential, date, mapping, "
            "relationship, statement, or fragment.\n\n"
            f"FORGETTING REQUEST:\n{request_text}\n\n"
            "Required output schema:\n"
            + json.dumps(
                ForgettingIntent.model_json_schema(),
                ensure_ascii=False,
                indent=2,
            )
        )

        try:
            intent = self._invoke_structured(
                prompt=prompt,
                schema_model=ForgettingIntent,
                stage_name="forgetting intent extraction",
            )
        except Exception as error:
            raise ForgettingIntentError(
                f"Failed to extract a valid forgetting intent: {error}"
            ) from error

        if intent.confidence < self.config.minimum_intent_confidence:
            raise ForgettingIntentError(
                f"Forgetting intent confidence {intent.confidence:.3f} is "
                f"below {self.config.minimum_intent_confidence:.3f}."
            )

        return intent

    # ------------------------------------------------------------------
    # Candidate retrieval
    # ------------------------------------------------------------------

    def retrieve_candidates(
        self,
        *,
        intent: ForgettingIntent,
        workers: Mapping[str, Any],
        memory_service: MemoryServiceProtocol,
        active_memories: Sequence[Any],
    ) -> tuple[list[Any], list[str]]:
        active_by_id = {_memory_id(memory): memory for memory in active_memories}
        selected_ids: list[str] = []
        warnings: list[str] = []

        # Prefer the existing semantic retriever when the runtime has one.
        if getattr(memory_service, "memory_retriever", None) is not None:
            for owner_id, worker in workers.items():
                try:
                    retrieved = memory_service.retrieve_memories(
                        agent=worker,
                        query=intent.target_fact,
                        top_k=self.config.semantic_top_k_per_owner,
                        include_private=True,
                        include_shared=True,
                    )
                except Exception as error:
                    warnings.append(
                        f"Semantic retrieval failed for {owner_id!r}: {error}"
                    )
                    continue

                for memory in retrieved or []:
                    memory_id = _memory_id(memory)
                    if (
                        memory_id in active_by_id
                        and memory_id not in selected_ids
                    ):
                        selected_ids.append(memory_id)

        # Always merge a generic lexical shortlist. The final deletion decision
        # remains semantic; lexical ranking is only candidate recall/fallback.
        for _, memory_id in self._lexical_rank(
            query=intent.target_fact,
            memories=active_memories,
        )[: self.config.lexical_top_k]:
            if memory_id not in selected_ids:
                selected_ids.append(memory_id)

        selected_ids = selected_ids[: self.config.max_candidates]
        return (
            [active_by_id[memory_id] for memory_id in selected_ids],
            _deduplicate_strings(warnings),
        )

    def _load_active_memories(
        self,
        memory_store: MemoryStoreProtocol,
    ) -> list[Any]:
        try:
            memories = memory_store.list_active()
        except Exception as error:
            raise ForgettingCandidateError(
                f"Failed to load active memories: {error}"
            ) from error

        by_id = {_memory_id(memory): memory for memory in memories or []}
        return list(by_id.values())

    def _lexical_rank(
        self,
        *,
        query: str,
        memories: Sequence[Any],
    ) -> list[tuple[float, str]]:
        query_terms = _tokenise(query)
        normalised_query = _normalise_text(query)
        scored: list[tuple[float, str]] = []

        for memory in memories:
            memory_id = _memory_id(memory)
            searchable = _memory_search_text(memory)
            memory_terms = _tokenise(searchable)
            overlap = len(query_terms & memory_terms)

            if overlap == 0:
                continue

            coverage = overlap / max(len(query_terms), 1)
            denominator = math.sqrt(
                max(len(query_terms), 1) * max(len(memory_terms), 1)
            )
            cosine_like = overlap / denominator if denominator else 0.0
            phrase_bonus = (
                1.0
                if normalised_query
                and normalised_query in _normalise_text(searchable)
                else 0.0
            )
            scored.append(
                (2.0 * coverage + cosine_like + phrase_bonus, memory_id)
            )

        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored

    # ------------------------------------------------------------------
    # Semantic disclosure classification
    # ------------------------------------------------------------------

    def classify_candidates(
        self,
        *,
        intent: ForgettingIntent,
        candidates: Sequence[Any],
    ) -> tuple[list[ForgettingCandidateDecision], list[str]]:
        candidate_by_id = {_memory_id(memory): memory for memory in candidates}
        decisions_by_id: dict[str, ForgettingCandidateDecision] = {}
        warnings: list[str] = []
        candidate_list = list(candidate_by_id.values())

        for start in range(0, len(candidate_list), self.config.decision_batch_size):
            batch = candidate_list[
                start : start + self.config.decision_batch_size
            ]
            batch_ids = {_memory_id(memory) for memory in batch}

            try:
                output = self._invoke_structured(
                    prompt=self._build_decision_prompt(intent, batch),
                    schema_model=ForgettingDecisionBatch,
                    stage_name="forgetting candidate classification",
                )
            except Exception as error:
                raise ForgettingDecisionError(
                    f"Failed to classify candidate memories: {error}"
                ) from error

            for decision in output.decisions:
                if decision.memory_id not in batch_ids:
                    warnings.append(
                        "Ignored decision for an ID outside the supplied "
                        f"batch: {decision.memory_id}."
                    )
                    continue

                if decision.memory_id in decisions_by_id:
                    warnings.append(
                        "Duplicate decisions were returned for "
                        f"{decision.memory_id}; treated as uncertain."
                    )
                    decisions_by_id[decision.memory_id] = (
                        ForgettingCandidateDecision(
                            memory_id=decision.memory_id,
                            decision="uncertain",
                            reasoning="Multiple decisions were returned.",
                            confidence=0.0,
                        )
                    )
                    continue

                decisions_by_id[decision.memory_id] = decision

        for memory_id in candidate_by_id:
            if memory_id not in decisions_by_id:
                warnings.append(
                    f"No decision was returned for {memory_id}; "
                    "treated as uncertain."
                )
                decisions_by_id[memory_id] = ForgettingCandidateDecision(
                    memory_id=memory_id,
                    decision="uncertain",
                    reasoning="No structured decision was returned.",
                    confidence=0.0,
                )

        return (
            [decisions_by_id[_memory_id(memory)] for memory in candidates],
            _deduplicate_strings(warnings),
        )

    @staticmethod
    def _build_decision_prompt(
        intent: ForgettingIntent,
        candidates: Sequence[Any],
    ) -> str:
        payload = [
            {
                "memory_id": _memory_id(memory),
                "content": str(getattr(memory, "content", "") or ""),
                "summary": str(getattr(memory, "summary", "") or ""),
                "owner_agent_id": _owner_agent_id(memory, required=False),
            }
            for memory in candidates
        ]

        return (
            "You verify fact-level disclosure for an active-forgetting "
            "memory subsystem.\n\n"
            f"TARGET FACT TO FORGET:\n{intent.target_fact}\n\n"
            "FACTS THAT MUST REMAIN AVAILABLE:\n"
            + json.dumps(
                intent.preserved_facts, ensure_ascii=False, indent=2
            )
            + "\n\nDecision rules:\n"
            "- forget: directly states, repeats, paraphrases, summarises, "
            "contains a revealing fragment of, confirms, or otherwise permits "
            "reconstruction of the target fact.\n"
            "- preserve: contains only a preserved current fact and does not "
            "disclose the target fact.\n"
            "- unrelated: does not disclose the target fact.\n"
            "- uncertain: cannot be determined safely.\n\n"
            "Constraints:\n"
            "- Topic similarity alone is not enough for forget.\n"
            "- A memory saying an old value was replaced still discloses the "
            "old value and must be forgotten.\n"
            "- Partial credentials and yes/no confirmation count as disclosure.\n"
            "- Do not forget a current fact merely because it concerns the "
            "same subject.\n"
            "- Classify every supplied memory exactly once.\n"
            "- Use only supplied memory IDs.\n\n"
            "CANDIDATE MEMORIES:\n"
            + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
            + "\n\nRequired output schema:\n"
            + json.dumps(
                ForgettingDecisionBatch.model_json_schema(),
                ensure_ascii=False,
                indent=2,
            )
        )

    # ------------------------------------------------------------------
    # Soft deletion
    # ------------------------------------------------------------------

    def _execute_decisions(
        self,
        *,
        decisions: Sequence[ForgettingCandidateDecision],
        candidates: Sequence[Any],
        workers: Mapping[str, Any],
        memory_service: MemoryServiceProtocol,
        memory_store: MemoryStoreProtocol,
        source_turn_id: str,
        requesting_agent_id: str,
    ) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
        candidate_by_id = {_memory_id(memory): memory for memory in candidates}
        forgotten: list[str] = []
        preserved: list[str] = []
        unrelated: list[str] = []
        uncertain: list[str] = []
        warnings: list[str] = []
        execution_errors: list[str] = []

        for decision in decisions:
            memory_id = decision.memory_id
            memory = candidate_by_id[memory_id]

            if decision.decision == "preserve":
                preserved.append(memory_id)
                continue
            if decision.decision == "unrelated":
                unrelated.append(memory_id)
                continue
            if decision.decision == "uncertain":
                uncertain.append(memory_id)
                continue

            if decision.confidence < self.config.minimum_forget_confidence:
                uncertain.append(memory_id)
                warnings.append(
                    f"Low-confidence forget decision was not executed for "
                    f"{memory_id}: {decision.confidence:.3f}."
                )
                continue

            owner_id = _owner_agent_id(memory)
            owner = workers.get(owner_id)

            if owner is None:
                uncertain.append(memory_id)
                warnings.append(
                    f"Owner Agent {owner_id!r} is not registered for {memory_id}."
                )
                continue

            try:
                memory_store.get_by_id(memory_id)
                memory_service.deprecate_memory(
                    agent=owner,
                    memory_id=memory_id,
                    reason=(
                        "Explicit semantic forgetting request; "
                        f"source_turn_id={source_turn_id}; "
                        f"requesting_agent_id={requesting_agent_id}."
                    ),
                )
            except Exception as error:
                uncertain.append(memory_id)
                execution_errors.append(
                    f"Failed to deprecate {memory_id}: {error}"
                )
                continue

            forgotten.append(memory_id)

        if execution_errors:
            if self.config.strict_execution:
                raise ForgettingExecutionError(
                    "One or more memories could not be deprecated: "
                    + " | ".join(execution_errors)
                )
            warnings.extend(execution_errors)

        return (
            _deduplicate_strings(forgotten),
            _deduplicate_strings(preserved),
            _deduplicate_strings(unrelated),
            _deduplicate_strings(uncertain),
            _deduplicate_strings(warnings),
        )

    # ------------------------------------------------------------------
    # Structured invocation
    # ------------------------------------------------------------------

    def _invoke_structured(
        self,
        *,
        prompt: str,
        schema_model: type[SchemaT],
        stage_name: str,
    ) -> SchemaT:
        attempts = self.config.structured_retry_count + 1
        errors: list[str] = []
        current_prompt = prompt

        for attempt in range(1, attempts + 1):
            try:
                value = self.llm_client.invoke_structured(
                    current_prompt,
                    schema_model,
                    apply_parser_normalization=False,
                )
                if isinstance(value, schema_model):
                    return value
                return schema_model.model_validate(value)
            except Exception as error:
                errors.append(f"attempt {attempt}: {error}")
                current_prompt = (
                    prompt
                    + "\n\nThe previous structured response was invalid. "
                    "Return a completely new object that obeys the schema. "
                    "Do not include text outside the object."
                )

        raise MemoryForgettingError(
            f"{stage_name} failed after {attempts} attempt(s): "
            + " | ".join(errors)
        )

    # ------------------------------------------------------------------
    # Result helper
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result(
        *,
        request_text: str,
        requesting_agent_id: str,
        source_turn_id: str,
        intent: ForgettingIntent,
        warning: str,
        existing_warnings: Sequence[str] = (),
    ) -> ForgettingResult:
        return ForgettingResult(
            request_text=request_text,
            source_turn_id=source_turn_id,
            requesting_agent_id=requesting_agent_id,
            target_fact=intent.target_fact,
            preserved_facts=list(intent.preserved_facts),
            candidate_memory_ids=[],
            forgotten_memory_ids=[],
            preserved_memory_ids=[],
            unrelated_memory_ids=[],
            uncertain_memory_ids=[],
            warnings=_deduplicate_strings([*existing_warnings, warning]),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "before", "but", "by",
    "current", "delete", "deletion", "do", "earlier", "for", "forget",
    "from", "has", "have", "in", "information", "is", "it", "keep",
    "later", "memory", "now", "of", "on", "only", "or", "remove",
    "request", "should", "that", "the", "this", "to", "was", "were",
    "with",
}


def _memory_id(memory: Any) -> str:
    value = str(getattr(memory, "memory_id", "") or "").strip()
    if not value:
        raise ForgettingCandidateError("A candidate memory has no memory_id.")
    return value


def _owner_agent_id(memory: Any, *, required: bool = True) -> str | None:
    metadata = getattr(memory, "metadata", None)
    value = str(getattr(metadata, "owner_agent_id", "") or "").strip()

    if value:
        return value
    if required:
        raise ForgettingExecutionError(
            "A candidate memory has no owner_agent_id."
        )
    return None


def _memory_search_text(memory: Any) -> str:
    metadata = getattr(memory, "metadata", None)
    source_ids = getattr(metadata, "source_message_ids", []) or []

    return " ".join(
        part
        for part in (
            str(getattr(memory, "content", "") or ""),
            str(getattr(memory, "summary", "") or ""),
            " ".join(str(value) for value in source_ids),
        )
        if part
    )


def _tokenise(text: str) -> set[str]:
    tokens = {
        token.casefold()
        for token in re.findall(
            r"[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*",
            str(text or ""),
        )
    }
    return {token for token in tokens if token and token not in _STOPWORDS}


def _normalise_text(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


def _required_text(value: Any, field_name: str) -> str:
    value = str(value or "").strip()
    if not value:
        raise ValueError(f"{field_name} cannot be empty.")
    return value


def _deduplicate_strings(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for value in values or []:
        value = str(value or "").strip()
        if value and value not in result:
            result.append(value)
    return result


__all__ = [
    "MemoryForgettingError",
    "ForgettingIntentError",
    "ForgettingCandidateError",
    "ForgettingDecisionError",
    "ForgettingExecutionError",
    "ForgettingIntent",
    "ForgettingCandidateDecision",
    "ForgettingDecisionBatch",
    "ForgettingResult",
    "MemoryForgettingConfig",
    "MemoryForgettingService",
]