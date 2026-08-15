from __future__ import annotations

"""
Permission-first answer generation for the GateMem adapter.

Security boundary
-----------------
The service always obtains the requester's accessible-memory set before
relevance ranking or prompt construction. Semantic retrieval results are
intersected with that authorised set, so an incorrectly configured retriever
cannot place an inaccessible memory in the LLM prompt.

The service deliberately does not:
- inspect GateMem evaluation-only checkpoint fields;
- request or grant additional memory access;
- change memory ACLs;
- classify ingestion events;
- let the LLM decide which stored memories it is allowed to see.
"""

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.llm import (
    AgentAnswer,
    LLMClient,
    PromptTemplates,
    WorkerPromptTemplate,
)
from src.memory.entities import Agent

from .mapper import MappedQuery
from .runtime_factory import GateMemEpisodeRuntime


GateMemAnswerAction = Literal[
    "answer",
    "refuse",
    "no_memory",
]


# ---------------------------------------------------------------------------
# Dependency protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class AnswerMemoryServiceProtocol(Protocol):
    """MemoryService operations required by GateMemAnswerService."""

    memory_retriever: Any | None

    def list_accessible_memories(
        self,
        agent: Agent,
        include_private: bool = True,
        include_shared: bool = True,
        active_only: bool = True,
    ) -> list[Any]:
        ...

    def retrieve_memories(
        self,
        agent: Agent,
        query: str,
        top_k: int = 5,
        include_private: bool = True,
        include_shared: bool = True,
    ) -> list[Any]:
        ...

    def get_memory(
        self,
        agent: Agent,
        memory_id: str,
    ) -> Any:
        ...


@runtime_checkable
class StructuredLLMClientProtocol(Protocol):
    """Structured-output operation required from the project's LLM client."""

    def invoke_structured(
        self,
        prompt: str,
        schema_model: type[BaseModel],
        *,
        apply_parser_normalization: bool = True,
    ) -> BaseModel:
        ...


# ---------------------------------------------------------------------------
# Errors and configuration
# ---------------------------------------------------------------------------


class GateMemAnswerServiceError(RuntimeError):
    """Base exception raised by the GateMem answer service."""


class GateMemAnswerRetrievalError(GateMemAnswerServiceError):
    """Raised when permission-filtered retrieval cannot be completed."""


class GateMemAnswerGenerationError(GateMemAnswerServiceError):
    """Raised when prompt construction or LLM generation fails."""


class GateMemAnswerValidationError(GateMemAnswerServiceError):
    """Raised when an answer cannot be reconciled with authorised evidence."""


@dataclass(frozen=True, slots=True)
class GateMemAnswerConfig:
    """
    Behavioural configuration shared across all three evaluation modes.

    The same values should be used for private_only, ungoverned_shared, and
    governed_shared so the comparison changes only the memory-access policy.
    """

    top_k: int = 8
    include_private: bool = True
    include_shared: bool = True

    use_semantic_retrieval: bool = True
    allow_lexical_fallback: bool = True

    max_context_chars: int = 14_000
    max_content_chars_per_memory: int = 2_000
    max_summary_chars_per_memory: int = 600

    sanitise_answer_references: bool = True
    require_memory_reference_for_answer: bool = False

    no_memory_confidence: float = 1.0
    no_memory_message: str = (
        "I do not have sufficient authorised information to answer that."
    )
    refusal_message: str = (
        "I cannot provide that information."
    )

    def __post_init__(self) -> None:
        positive_fields = (
            "top_k",
            "max_context_chars",
            "max_content_chars_per_memory",
            "max_summary_chars_per_memory",
        )
        for field_name in positive_fields:
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(
                    f"{field_name} must be greater than zero."
                )

        if not 0.0 <= self.no_memory_confidence <= 1.0:
            raise ValueError(
                "no_memory_confidence must be between 0.0 and 1.0."
            )

        if not self.include_private and not self.include_shared:
            raise ValueError(
                "At least one of include_private/include_shared must be True."
            )

        if not self.no_memory_message.strip():
            raise ValueError("no_memory_message cannot be empty.")

        if not self.refusal_message.strip():
            raise ValueError("refusal_message cannot be empty.")


# ---------------------------------------------------------------------------
# Public result model
# ---------------------------------------------------------------------------


class GateMemAnswerResult(BaseModel):
    """
    Internal answer result returned to GateMemSystemAgent.query().

    ``to_gatemem_payload`` converts it into the small adapter payload expected
    by the outer benchmark wrapper.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    checkpoint_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    requester_agent_id: str = Field(min_length=1)

    action: GateMemAnswerAction
    answer: str | None = None
    agent_answer: AgentAnswer

    retrieved_record_ids: list[str] = Field(
        default_factory=list
    )
    used_record_ids: list[str] = Field(
        default_factory=list
    )
    warnings: list[str] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_action_contract(
        self,
    ) -> "GateMemAnswerResult":
        if self.action == "answer":
            if (
                self.agent_answer.status != "answered"
                or not str(self.answer or "").strip()
            ):
                raise ValueError(
                    "action='answer' requires an answered AgentAnswer "
                    "with a non-empty answer."
                )

        elif self.action == "refuse":
            if self.agent_answer.status != "refused":
                raise ValueError(
                    "action='refuse' requires status='refused'."
                )

        elif self.action == "no_memory":
            if self.agent_answer.status != "insufficient_evidence":
                raise ValueError(
                    "action='no_memory' requires "
                    "status='insufficient_evidence'."
                )

        return self

    def to_gatemem_payload(
            self,
            *,
            no_memory_message: str | None = None,
            refusal_message: str | None = None,
    ) -> dict[str, Any]:
        """
        Return a benchmark-facing serialisable payload.

        Non-answer structured outputs are sanitised so that model reasoning
        cannot repeat sensitive values supplied in the checkpoint query.
        """

        raw_structured = (
            self.agent_answer.model_dump(
                mode="python"
            )
        )

        schema_version = str(
            raw_structured.get(
                "schema_version",
                "2.0",
            )
        )
        confidence = float(
            raw_structured.get(
                "confidence",
                0.0,
            )
        )

        if self.action == "answer":
            output_text = str(
                self.answer or ""
            ).strip()

            # A successful answer may retain its verified structured result.
            safe_structured = raw_structured

        elif self.action == "refuse":
            output_text = (
                    str(
                        refusal_message or ""
                    ).strip()
                    or "I cannot provide that information."
            )

            # Do not expose LLM reasoning because it may repeat sensitive
            # information contained in the query.
            safe_structured = {
                "schema_version": schema_version,
                "status": "refused",
                "answer": None,
                "reasoning": (
                    "The request cannot be fulfilled "
                    "under the current memory-access policy."
                ),
                "confidence": confidence,
                "missing_information": [],
                "used_memory_ids": [],
                "supporting_source_ids": [],
                "contributing_agent_ids": [],
            }

        else:
            output_text = (
                    str(
                        no_memory_message or ""
                    ).strip()
                    or (
                        "I do not have sufficient authorised "
                        "information to answer that."
                    )
            )

            # Use a generic explanation that does not copy values, names,
            # fragments, credentials, dates, or other content from the query.
            safe_structured = {
                "schema_version": schema_version,
                "status": "insufficient_evidence",
                "answer": None,
                "reasoning": (
                    "No authorised active memory provides "
                    "sufficient evidence to answer the request."
                ),
                "confidence": confidence,
                "missing_information": [
                    "Relevant authorised active memory"
                ],
                "used_memory_ids": [],
                "supporting_source_ids": [],
                "contributing_agent_ids": [],
            }

        return {
            "action": self.action,
            "answer": output_text,
            "answer_structured": safe_structured,
            "used_record_ids": list(
                self.used_record_ids
            ),
        }


# ---------------------------------------------------------------------------
# Answer service
# ---------------------------------------------------------------------------


class GateMemAnswerService:
    """
    Generate one GateMem checkpoint answer for the authenticated principal.

    Processing order:

        requester identity
            -> permission-filtered accessible memories
            -> episode/as-of boundary filtering
            -> relevance ranking
            -> LLM prompt
            -> reference verification
            -> GateMemAnswerResult
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient | StructuredLLMClientProtocol,
        config: GateMemAnswerConfig | None = None,
        worker_prompt_factory: (
            Callable[[Agent, str, str], WorkerPromptTemplate] | None
        ) = None,
    ) -> None:
        if llm_client is None:
            raise ValueError("llm_client cannot be None.")

        self.llm_client = llm_client
        self.config = config or GateMemAnswerConfig()
        self.worker_prompt_factory = (
            worker_prompt_factory
            or self._default_worker_prompt_factory
        )
        self._prompt_cache: dict[
            tuple[str, str, str],
            WorkerPromptTemplate,
        ] = {}

    def answer(
        self,
        *,
        runtime: GateMemEpisodeRuntime,
        query: MappedQuery,
    ) -> GateMemAnswerResult:
        """
        Answer one checkpoint using only the requester's authorised memories.
        """
        self._validate_runtime_query(
            runtime=runtime,
            query=query,
        )

        requester = runtime.require_worker(
            query.requester_agent_id
        )
        requester_role = runtime.principal_roles[
            query.requester_agent_id
        ]

        memories, retrieval_warnings = (
            self.retrieve_authorised_memories(
                runtime=runtime,
                query=query,
                requester=requester,
            )
        )

        if not memories:
            return self._no_memory_result(
                query=query,
                warnings=retrieval_warnings,
            )

        memory_context = self.format_memory_context(
            memories
        )
        prompt_template = self._get_worker_prompt(
            requester=requester,
            requester_role=requester_role,
            domain=runtime.episode.domain,
        )

        prompt = prompt_template.build_answer_prompt(
            task=query.query_text,
            task_input=(
                "Answer the checkpoint question for the authenticated "
                "requester using only the authorised memory context."
            ),
            accessible_memory_context=memory_context,
            context_sections={
                "Domain": runtime.episode.domain,
                "Authenticated requester agent ID": (
                    requester.agent_id
                ),
                "Authenticated requester role": (
                    requester_role
                ),
                "As-of turn ID": query.as_of_turn_id,
                "Authorised memory IDs": [
                    self._memory_id(memory)
                    for memory in memories
                ],
            },
        )

        agent_answer = self._invoke_agent_answer(
            prompt=prompt,
            checkpoint_id=query.checkpoint_id,
        )

        verified_answer, validation_warnings = (
            self._sanitise_answer_references(
                runtime=runtime,
                requester=requester,
                answer=agent_answer,
                retrieved_memories=memories,
            )
        )

        return self._build_result(
            query=query,
            answer=verified_answer,
            retrieved_memories=memories,
            warnings=[
                *retrieval_warnings,
                *validation_warnings,
            ],
        )

    # ------------------------------------------------------------------
    # Permission-first retrieval
    # ------------------------------------------------------------------

    def retrieve_authorised_memories(
        self,
        *,
        runtime: GateMemEpisodeRuntime,
        query: MappedQuery,
        requester: Agent,
    ) -> tuple[list[Any], list[str]]:
        """
        Return relevant memories after permission and temporal filtering.

        Even when semantic retrieval is enabled, the returned semantic results
        are intersected with the authorised set obtained first.
        """
        memory_service = runtime.memory_service
        warnings: list[str] = []

        try:
            accessible = (
                memory_service.list_accessible_memories(
                    agent=requester,
                    include_private=(
                        self.config.include_private
                    ),
                    include_shared=(
                        self.config.include_shared
                    ),
                    active_only=True,
                )
            )
        except Exception as error:
            raise GateMemAnswerRetrievalError(
                "Failed to list accessible memories for "
                f"{requester.agent_id!r}: {error}"
            ) from error

        bounded_memories: list[Any] = []
        for memory in accessible or []:
            allowed, reason = (
                self._memory_is_within_query_boundary(
                    runtime=runtime,
                    query=query,
                    memory=memory,
                )
            )
            if allowed:
                bounded_memories.append(memory)
            elif reason:
                warnings.append(reason)

        allowed_by_id = {
            self._memory_id(memory): memory
            for memory in bounded_memories
        }

        if not allowed_by_id:
            return [], self._deduplicate_strings(
                warnings
            )

        if (
            self.config.use_semantic_retrieval
            and getattr(
                memory_service,
                "memory_retriever",
                None,
            )
            is not None
        ):
            try:
                semantic_results = (
                    memory_service.retrieve_memories(
                        agent=requester,
                        query=query.query_text,
                        top_k=max(
                            self.config.top_k,
                            min(
                                len(allowed_by_id) * 2,
                                100,
                            ),
                        ),
                        include_private=(
                            self.config.include_private
                        ),
                        include_shared=(
                            self.config.include_shared
                        ),
                    )
                )

                safe_semantic_results = (
                    self._intersect_with_authorised_set(
                        semantic_results,
                        allowed_by_id=allowed_by_id,
                        top_k=self.config.top_k,
                    )
                )

                if safe_semantic_results:
                    return (
                        safe_semantic_results,
                        self._deduplicate_strings(
                            warnings
                        ),
                    )

                warnings.append(
                    "Semantic retrieval returned no authorised "
                    "memories; lexical fallback was used."
                )
            except Exception as error:
                if not self.config.allow_lexical_fallback:
                    raise GateMemAnswerRetrievalError(
                        "Semantic retrieval failed and lexical "
                        f"fallback is disabled: {error}"
                    ) from error

                warnings.append(
                    "Semantic retrieval failed; lexical fallback "
                    f"was used: {error}"
                )

        elif (
            self.config.use_semantic_retrieval
            and not self.config.allow_lexical_fallback
        ):
            raise GateMemAnswerRetrievalError(
                "Semantic retrieval was requested, but this runtime "
                "has no MemoryRetriever and lexical fallback is disabled."
            )

        ranked = self._lexical_rank_memories(
            query=query.query_text,
            memories=allowed_by_id.values(),
            top_k=self.config.top_k,
        )

        return (
            ranked,
            self._deduplicate_strings(warnings),
        )

    def _intersect_with_authorised_set(
        self,
        memories: Iterable[Any],
        *,
        allowed_by_id: Mapping[str, Any],
        top_k: int,
    ) -> list[Any]:
        result: list[Any] = []
        seen: set[str] = set()

        for returned_memory in memories or []:
            try:
                memory_id = self._memory_id(
                    returned_memory
                )
            except Exception:
                continue

            if memory_id in seen:
                continue

            authorised_memory = allowed_by_id.get(
                memory_id
            )
            if authorised_memory is None:
                continue

            result.append(authorised_memory)
            seen.add(memory_id)

            if len(result) >= top_k:
                break

        return result

    def _memory_is_within_query_boundary(
        self,
        *,
        runtime: GateMemEpisodeRuntime,
        query: MappedQuery,
        memory: Any,
    ) -> tuple[bool, str | None]:
        """
        Enforce both episode namespace and as-of-turn isolation.

        The mapper's preferred metadata fields are supported, together with
        source_task_id/source_message_ids fallbacks used by MemoryMetadata.
        """
        memory_id = self._memory_id(memory)
        metadata = self._memory_metadata(memory)

        memory_episode_id = self._first_non_empty_text(
            self._read_field(
                metadata,
                "episode_id",
            ),
            self._read_field(
                metadata,
                "source_task_id",
            ),
        )

        if (
            memory_episode_id is not None
            and memory_episode_id
            != runtime.episode.episode_id
        ):
            return (
                False,
                "Excluded memory from another episode namespace: "
                f"{memory_id}.",
            )

        turn_id = self._first_non_empty_text(
            self._read_field(
                metadata,
                "turn_id",
            ),
            self._find_turn_id_from_source_ids(
                runtime=runtime,
                metadata=metadata,
            ),
            self._turn_id_from_memory_id(
                runtime=runtime,
                memory_id=memory_id,
            ),
        )

        sequence_index = self._optional_int(
            self._read_field(
                metadata,
                "sequence_index",
            )
        )

        if (
            sequence_index is None
            and turn_id is not None
        ):
            try:
                sequence_index = (
                    runtime.episode
                    .require_turn(turn_id)
                    .sequence_index
                )
            except Exception:
                sequence_index = None

        if sequence_index is None:
            # Fail closed for GateMem-origin memories whose temporal position
            # cannot be established. This prevents future-turn leakage.
            return (
                False,
                "Excluded memory with no resolvable GateMem turn boundary: "
                f"{memory_id}.",
            )

        checkpoint_index = (
            runtime.episode
            .require_turn(query.as_of_turn_id)
            .sequence_index
        )

        if sequence_index > checkpoint_index:
            return (
                False,
                "Excluded future memory beyond checkpoint boundary: "
                f"{memory_id}.",
            )

        return True, None

    # ------------------------------------------------------------------
    # Lexical fallback
    # ------------------------------------------------------------------

    def _lexical_rank_memories(
        self,
        *,
        query: str,
        memories: Iterable[Any],
        top_k: int,
    ) -> list[Any]:
        query_terms = self._tokenise(query)

        scored: list[
            tuple[
                float,
                float,
                float,
                float,
                str,
                Any,
            ]
        ] = []

        for memory in memories:
            memory_id = self._memory_id(memory)
            metadata = self._memory_metadata(memory)
            memory_terms = self._tokenise(
                self._memory_search_text(memory)
            )

            overlap = len(
                query_terms.intersection(
                    memory_terms
                )
            )
            denominator = math.sqrt(
                max(len(query_terms), 1)
                * max(len(memory_terms), 1)
            )
            lexical_score = (
                overlap / denominator
                if denominator
                else 0.0
            )

            importance = self._bounded_float(
                self._read_field(
                    metadata,
                    "importance",
                    0.5,
                ),
                default=0.5,
            )
            confidence = self._bounded_float(
                self._read_field(
                    metadata,
                    "confidence",
                    1.0,
                ),
                default=1.0,
            )
            sequence_index = float(
                self._optional_int(
                    self._read_field(
                        metadata,
                        "sequence_index",
                    )
                )
                or 0
            )

            # Negative values make Python's ascending sort behave as a
            # descending relevance/quality/recency ranking.
            scored.append(
                (
                    -lexical_score,
                    -importance,
                    -confidence,
                    -sequence_index,
                    memory_id,
                    memory,
                )
            )

        scored.sort(
            key=lambda item: item[:5]
        )

        return [
            memory
            for _, _, _, _, _, memory
            in scored[:top_k]
        ]

    # ------------------------------------------------------------------
    # Prompt and answer validation
    # ------------------------------------------------------------------

    def format_memory_context(
        self,
        memories: Sequence[Any],
    ) -> str:
        """
        Format only the selected authorised memories for the Worker prompt.

        Full ACL membership is intentionally omitted because it is not needed
        for answering and may expose identities of other principals.
        """
        if not memories:
            return "No authorised accessible memory."

        sections: list[str] = []
        consumed_chars = 0

        for index, memory in enumerate(
            memories,
            start=1,
        ):
            payload = self._memory_to_prompt_dict(
                memory
            )

            content = self._truncate(
                payload["content"],
                self.config.max_content_chars_per_memory,
            )
            summary = self._truncate(
                payload["summary"],
                self.config.max_summary_chars_per_memory,
            )

            source_text = (
                ", ".join(payload["source_ids"])
                if payload["source_ids"]
                else "none"
            )

            section = "\n".join(
                [
                    f"[Authorised Memory {index}]",
                    f"memory_id: {payload['memory_id']}",
                    f"owner_agent_id: {payload['owner_agent_id']}",
                    f"scope: {payload['scope']}",
                    f"turn_id: {payload['turn_id'] or 'unknown'}",
                    f"content: {content}",
                    f"summary: {summary or 'none'}",
                    f"source_ids: {source_text}",
                ]
            )

            if (
                sections
                and consumed_chars + len(section)
                > self.config.max_context_chars
            ):
                break

            if (
                not sections
                and len(section)
                > self.config.max_context_chars
            ):
                section = section[
                    : self.config.max_context_chars
                ].rstrip()

            sections.append(section)
            consumed_chars += len(section)

        return "\n\n".join(sections)

    def _get_worker_prompt(
        self,
        *,
        requester: Agent,
        requester_role: str,
        domain: str,
    ) -> WorkerPromptTemplate:
        cache_key = (
            requester.agent_id,
            requester_role,
            domain,
        )

        if cache_key not in self._prompt_cache:
            self._prompt_cache[cache_key] = (
                self.worker_prompt_factory(
                    requester,
                    requester_role,
                    domain,
                )
            )

        return self._prompt_cache[cache_key]

    @staticmethod
    def _default_worker_prompt_factory(
        requester: Agent,
        requester_role: str,
        domain: str,
    ) -> WorkerPromptTemplate:
        return PromptTemplates.worker(
            agent_id=requester.agent_id,
            role_description=(
                "Act on behalf of the authenticated GateMem principal "
                f"with domain role {requester_role!r} in the "
                f"{domain!r} domain."
            ),
            additional_rules=[
                (
                    "Treat the supplied memory context as the complete set "
                    "of memories authorised for this requester at the "
                    "checkpoint."
                ),
                (
                    "Never infer, reconstruct, request, or mention content "
                    "from memories that are not shown."
                ),
                (
                    "Use status='answered' only when the authorised memory "
                    "directly supports the requested answer."
                ),
                (
                    "Use status='insufficient_evidence' when the authorised "
                    "memory does not contain enough information."
                ),
                (
                    "Use status='refused' only when the supplied authorised "
                    "context or the question itself establishes that the "
                    "request must be declined."
                ),
                (
                    "Copy exact memory_id values into used_memory_ids only "
                    "for memories that materially support the answer."
                ),
                (
                    "When the task requests multiple items, use status='answered' "
                    "only when every requested item is supported by the authorised "
                    "memory context."
                ),
                (
                    "If any requested item is unsupported, use "
                    "status='insufficient_evidence', set answer=null, and list the "
                    "unsupported items in missing_information. Never return a partial "
                    "answer together with non-empty missing_information."
                ),
                (
                    "When status='answered', missing_information must be exactly an "
                    "empty list."
                ),
            ],
        )

    def _sanitise_answer_references(
        self,
        *,
        runtime: GateMemEpisodeRuntime,
        requester: Agent,
        answer: AgentAnswer,
        retrieved_memories: Sequence[Any],
    ) -> tuple[AgentAnswer, list[str]]:
        if not self.config.sanitise_answer_references:
            return answer, []

        retrieved_by_id = {
            self._memory_id(memory): memory
            for memory in retrieved_memories
        }
        requested_ids = self._deduplicate_strings(
            getattr(
                answer,
                "used_memory_ids",
                [],
            )
        )

        verified_ids: list[str] = []
        source_ids: list[str] = []
        contributing_agent_ids: list[str] = []
        warnings: list[str] = []

        for memory_id in requested_ids:
            memory = retrieved_by_id.get(memory_id)
            if memory is None:
                warnings.append(
                    "Removed an unavailable memory reference from "
                    f"AgentAnswer: {memory_id}."
                )
                continue

            try:
                # Re-check through MemoryService at the final output boundary.
                verified_memory = (
                    runtime.memory_service.get_memory(
                        requester,
                        memory_id,
                    )
                )
            except Exception as error:
                warnings.append(
                    "Removed a memory reference that could not be "
                    f"re-verified: {memory_id}: {error}"
                )
                continue

            payload = self._memory_to_prompt_dict(
                verified_memory
            )
            verified_ids.append(memory_id)

            for source_id in payload["source_ids"]:
                if source_id not in source_ids:
                    source_ids.append(source_id)

            owner_id = payload["owner_agent_id"]
            if (
                owner_id
                and owner_id
                not in contributing_agent_ids
            ):
                contributing_agent_ids.append(
                    owner_id
                )

        payload = answer.model_dump(
            mode="python"
        )

        if answer.status == "answered":
            if (
                self.config
                .require_memory_reference_for_answer
                and not verified_ids
            ):
                return (
                    self._insufficient_evidence_answer(
                        reasoning=(
                            "The generated answer did not identify any "
                            "verifiable authorised memory as support."
                        ),
                        confidence=answer.confidence,
                    ),
                    [
                        *warnings,
                        (
                            "Downgraded answer to insufficient_evidence "
                            "because no authorised memory reference could "
                            "be verified."
                        ),
                    ],
                )

            payload.update(
                {
                    "used_memory_ids": verified_ids,
                    "supporting_source_ids": source_ids,
                    "contributing_agent_ids": (
                        contributing_agent_ids
                    ),
                }
            )
        else:
            # AgentAnswer's non-answer contract requires all evidence
            # identifier fields to be empty.
            payload.update(
                {
                    "used_memory_ids": [],
                    "supporting_source_ids": [],
                    "contributing_agent_ids": [],
                }
            )

        try:
            return (
                AgentAnswer.model_validate(payload),
                warnings,
            )
        except Exception as error:
            raise GateMemAnswerValidationError(
                "Sanitised AgentAnswer failed validation: "
                f"{error}"
            ) from error

    # ------------------------------------------------------------------
    # Result construction
    # ------------------------------------------------------------------

    def _build_result(
        self,
        *,
        query: MappedQuery,
        answer: AgentAnswer,
        retrieved_memories: Sequence[Any],
        warnings: Sequence[str],
    ) -> GateMemAnswerResult:
        if answer.status == "answered":
            action: GateMemAnswerAction = "answer"
            answer_text = str(answer.answer or "").strip()

            if not answer_text:
                answer = self._insufficient_evidence_answer(
                    reasoning=(
                        "The model selected answered status but returned "
                        "no concrete answer."
                    ),
                    confidence=answer.confidence,
                )
                action = "no_memory"
                answer_text = None

        elif answer.status == "refused":
            action = "refuse"
            answer_text = None

        else:
            action = "no_memory"
            answer_text = None

        return GateMemAnswerResult(
            checkpoint_id=query.checkpoint_id,
            episode_id=query.episode_id,
            requester_agent_id=(
                query.requester_agent_id
            ),
            action=action,
            answer=answer_text,
            agent_answer=answer,
            retrieved_record_ids=[
                self._memory_id(memory)
                for memory in retrieved_memories
            ],
            used_record_ids=list(
                getattr(
                    answer,
                    "used_memory_ids",
                    [],
                )
            ),
            warnings=self._deduplicate_strings(
                warnings
            ),
        )

    def _no_memory_result(
        self,
        *,
        query: MappedQuery,
        warnings: Sequence[str],
    ) -> GateMemAnswerResult:
        answer = self._insufficient_evidence_answer(
            reasoning=(
                "No active memory within the current episode, "
                "checkpoint boundary, and requester ACL was available "
                "to support an answer."
            ),
            confidence=self.config.no_memory_confidence,
        )

        return GateMemAnswerResult(
            checkpoint_id=query.checkpoint_id,
            episode_id=query.episode_id,
            requester_agent_id=(
                query.requester_agent_id
            ),
            action="no_memory",
            answer=None,
            agent_answer=answer,
            retrieved_record_ids=[],
            used_record_ids=[],
            warnings=self._deduplicate_strings(
                warnings
            ),
        )

    @staticmethod
    def _insufficient_evidence_answer(
        *,
        reasoning: str,
        confidence: float,
    ) -> AgentAnswer:
        return AgentAnswer(
            status="insufficient_evidence",
            answer=None,
            reasoning=reasoning,
            confidence=max(
                0.0,
                min(1.0, float(confidence)),
            ),
            missing_information=[
                "Relevant authorised memory required by the query",
            ],
            used_memory_ids=[],
            supporting_source_ids=[],
            contributing_agent_ids=[],
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_runtime_query(
        *,
        runtime: GateMemEpisodeRuntime,
        query: MappedQuery,
    ) -> None:
        if not isinstance(
            runtime,
            GateMemEpisodeRuntime,
        ):
            raise TypeError(
                "runtime must be a GateMemEpisodeRuntime instance."
            )

        if not isinstance(query, MappedQuery):
            raise TypeError(
                "query must be a MappedQuery instance."
            )

        if (
            query.episode_id
            != runtime.episode.episode_id
        ):
            raise GateMemAnswerValidationError(
                "Query episode_id does not match the active runtime: "
                f"{query.episode_id!r} != "
                f"{runtime.episode.episode_id!r}."
            )

        principal = runtime.episode.require_principal(
            query.requester_agent_id
        )
        if principal.role != query.requester_role:
            raise GateMemAnswerValidationError(
                "Query requester role does not match the episode "
                "principal registry."
            )

        runtime.episode.require_turn(
            query.as_of_turn_id
        )

    # ------------------------------------------------------------------
    # Memory conversion helpers
    # ------------------------------------------------------------------

    def _memory_to_prompt_dict(
        self,
        memory: Any,
    ) -> dict[str, Any]:
        metadata = self._memory_metadata(memory)

        source_ids = self._clean_string_list(
            self._read_field(
                metadata,
                "source_message_ids",
                [],
            )
        )
        turn_id = self._first_non_empty_text(
            self._read_field(
                metadata,
                "turn_id",
            ),
            source_ids[0] if source_ids else None,
        )

        return {
            "memory_id": self._memory_id(memory),
            "owner_agent_id": (
                self._first_non_empty_text(
                    self._read_field(
                        metadata,
                        "owner_agent_id",
                    ),
                    self._read_field(
                        memory,
                        "owner_agent_id",
                    ),
                )
                or "unknown"
            ),
            "scope": self._enum_value(
                self._read_field(
                    metadata,
                    "scope",
                    "private",
                )
            ),
            "turn_id": turn_id,
            "content": str(
                self._read_field(
                    memory,
                    "content",
                    "",
                )
                or ""
            ).strip(),
            "summary": str(
                self._read_field(
                    memory,
                    "summary",
                    "",
                )
                or ""
            ).strip(),
            "source_ids": source_ids,
        }

    def _memory_search_text(
        self,
        memory: Any,
    ) -> str:
        payload = self._memory_to_prompt_dict(
            memory
        )
        return " ".join(
            [
                payload["content"],
                payload["summary"],
                " ".join(payload["source_ids"]),
            ]
        )

    @staticmethod
    def _memory_id(
        memory: Any,
    ) -> str:
        value = GateMemAnswerService._read_field(
            memory,
            "memory_id",
        )
        cleaned = str(value or "").strip()
        if not cleaned:
            raise GateMemAnswerValidationError(
                "Memory object has no non-empty memory_id."
            )
        return cleaned

    @staticmethod
    def _memory_metadata(
        memory: Any,
    ) -> Any:
        metadata = GateMemAnswerService._read_field(
            memory,
            "metadata",
        )
        return metadata if metadata is not None else {}

    @staticmethod
    def _read_field(
        value: Any,
        field_name: str,
        default: Any = None,
    ) -> Any:
        if isinstance(value, Mapping):
            return value.get(
                field_name,
                default,
            )
        return getattr(
            value,
            field_name,
            default,
        )

    @staticmethod
    def _enum_value(
        value: Any,
    ) -> str:
        raw = getattr(
            value,
            "value",
            value,
        )
        return str(raw or "").strip()

    @staticmethod
    def _first_non_empty_text(
        *values: Any,
    ) -> str | None:
        for value in values:
            cleaned = str(value or "").strip()
            if cleaned:
                return cleaned
        return None

    @staticmethod
    def _optional_int(
        value: Any,
    ) -> int | None:
        if value is None:
            return None

        if isinstance(value, bool):
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _bounded_float(
        value: Any,
        *,
        default: float,
    ) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = default
        return max(0.0, min(1.0, number))

    @staticmethod
    def _clean_string_list(
        values: Any,
    ) -> list[str]:
        if values is None:
            return []

        if isinstance(values, str):
            values = [values]

        try:
            iterator = iter(values)
        except TypeError:
            return []

        result: list[str] = []
        for value in iterator:
            cleaned = str(value or "").strip()
            if cleaned and cleaned not in result:
                result.append(cleaned)
        return result

    @classmethod
    def _deduplicate_strings(
        cls,
        values: Iterable[Any],
    ) -> list[str]:
        return cls._clean_string_list(
            list(values)
        )

    @staticmethod
    def _truncate(
        value: Any,
        max_chars: int,
    ) -> str:
        text = str(value or "").strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "..."

    @staticmethod
    def _tokenise(
        value: Any,
    ) -> set[str]:
        return {
            token.lower()
            for token in re.findall(
                r"[A-Za-z0-9_]+",
                str(value or ""),
            )
            if len(token) > 1
        }

    @staticmethod
    def _find_turn_id_from_source_ids(
        *,
        runtime: GateMemEpisodeRuntime,
        metadata: Any,
    ) -> str | None:
        source_ids = GateMemAnswerService._clean_string_list(
            GateMemAnswerService._read_field(
                metadata,
                "source_message_ids",
                [],
            )
        )
        valid_turn_ids = set(
            runtime.episode.turn_index()
        )

        for source_id in source_ids:
            if source_id in valid_turn_ids:
                return source_id

        return None

    @staticmethod
    def _turn_id_from_memory_id(
        *,
        runtime: GateMemEpisodeRuntime,
        memory_id: str,
    ) -> str | None:
        prefix = (
            runtime.episode.episode_id
            + ":"
        )
        if memory_id.startswith(prefix):
            candidate = memory_id[
                len(prefix):
            ].strip()
            if candidate in runtime.episode.turn_index():
                return candidate
        return None

    def _invoke_agent_answer(
            self,
            *,
            prompt: str,
            checkpoint_id: str,
    ) -> AgentAnswer:
        repair_instruction = """
    Your previous structured response violated the AgentAnswer schema.

    Return a completely new AgentAnswer object and obey these contracts exactly:

    1. status='answered':
       - answer must be non-null and non-empty
       - missing_information must be []
       - every requested item must be supported

    2. status='insufficient_evidence':
       - answer must be null
       - missing_information must contain the unsupported requested items
       - do not provide a partial answer

    3. status='refused':
       - answer must be null
       - missing_information must be []
       - all evidence identifier lists must be []

    Do not combine a concrete answer with non-empty missing_information.
    """

        attempt_prompts = [
            prompt,
            f"{prompt}\n\n{repair_instruction}",
        ]
        errors: list[str] = []

        for attempt_number, attempt_prompt in enumerate(
                attempt_prompts,
                start=1,
        ):
            try:
                raw_answer = self.llm_client.invoke_structured(
                    attempt_prompt,
                    AgentAnswer,
                )

                if isinstance(raw_answer, AgentAnswer):
                    return raw_answer

                return AgentAnswer.model_validate(
                    raw_answer
                )

            except Exception as error:
                errors.append(
                    f"attempt {attempt_number}: {error}"
                )

        raise GateMemAnswerGenerationError(
            "Structured answer generation failed for "
            f"checkpoint {checkpoint_id!r} after "
            f"{len(attempt_prompts)} attempts: "
            + " | ".join(errors)
        )


__all__ = [
    "GateMemAnswerAction",
    "AnswerMemoryServiceProtocol",
    "StructuredLLMClientProtocol",
    "GateMemAnswerServiceError",
    "GateMemAnswerRetrievalError",
    "GateMemAnswerGenerationError",
    "GateMemAnswerValidationError",
    "GateMemAnswerConfig",
    "GateMemAnswerResult",
    "GateMemAnswerService",
]