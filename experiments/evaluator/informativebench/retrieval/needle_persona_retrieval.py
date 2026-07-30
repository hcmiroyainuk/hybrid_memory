from __future__ import annotations

import json
import re
from collections.abc import (
    Iterable,
    Mapping,
)
from dataclasses import dataclass
from typing import Any

from src.memory.entities import Agent

from ..config.needle_persona_config import (
    NeedlePersonaNodeConfig,
)
from ..data_preparing.needle_persona_models import (
    AGENT_PERSONA_MAP,
)
from ..workflow.dependencies.needle_persona_dependencies import (
    NeedlePersonaMemoryServiceProtocol,
)
from ..workflow.state.needle_persona_state import (
    NeedleMemoryCandidate,
)


class NeedlePersonaRetrievalError(Exception):
    """
    Raised when benchmark retrieval cannot be completed safely.
    """


@dataclass(
    frozen=True,
    slots=True,
)
class NeedlePersonaRetrievalResult:
    """
    Permission-filtered retrieval result supplied to a responder.

    ``memories`` may contain MemoryItem instances from the core memory layer.
    Workflow state should normally persist ``memory_ids`` and ``context``
    rather than these service-layer objects.
    """

    memories: tuple[Any, ...]
    memory_ids: tuple[str, ...]
    context: str

    fallback_used: bool = False
    warnings: tuple[str, ...] = ()

    def as_state_update(
        self,
        *,
        memory_ids_field: str,
        context_field: str,
    ) -> dict[str, Any]:
        """
        Convert the result into a compact LangGraph state update.
        """
        clean_ids_field = str(
            memory_ids_field
        ).strip()
        clean_context_field = str(
            context_field
        ).strip()

        if not clean_ids_field:
            raise ValueError(
                "memory_ids_field cannot be empty."
            )

        if not clean_context_field:
            raise ValueError(
                "context_field cannot be empty."
            )

        return {
            clean_ids_field: list(
                self.memory_ids
            ),
            clean_context_field: self.context,
            "warnings": list(self.warnings),
        }


@dataclass(
    frozen=True,
    slots=True,
)
class NeedlePersonaCandidateDiscoveryResult:
    """
    Owner-scoped cross-Agent candidate-discovery result.

    Candidate objects contain identifiers and ranking rationale only. They do
    not expose private-memory content to the responder.
    """

    candidates: tuple[
        NeedleMemoryCandidate,
        ...,
    ]

    fallback_owner_agent_ids: tuple[
        str,
        ...,
    ] = ()

    warnings: tuple[str, ...] = ()

    @property
    def candidate_ids(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            candidate.memory_id
            for candidate in self.candidates
        )

    def as_mapping(
        self,
    ) -> dict[
        str,
        NeedleMemoryCandidate,
    ]:
        """
        Return candidates keyed by memory ID for workflow state.
        """
        return {
            candidate.memory_id: candidate
            for candidate in self.candidates
        }


class NeedlePersonaRetrievalService:
    """
    Retrieval helper for the Needle in the Persona experiment.

    Responsibilities:

    1. retrieve only memories currently accessible to the responder;
    2. discover cross-Agent candidates inside each owner's own memory space;
    3. prevent unauthorised private content from entering responder state;
    4. provide deterministic lexical fallback when semantic retrieval is
       unavailable;
    5. render authorised memories into a stable prompt context.

    This class does not:

    - submit or approve sharing requests;
    - modify Memory scope or ACLs;
    - call the LLM;
    - use benchmark gold answers;
    - persist workflow state.
    """

    _TOKEN_PATTERN = re.compile(
        r"[A-Za-z0-9]+|[\u4e00-\u9fff]"
    )

    _STOP_WORDS = frozenset(
        {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "did",
            "do",
            "does",
            "for",
            "from",
            "had",
            "has",
            "have",
            "he",
            "her",
            "hers",
            "him",
            "his",
            "how",
            "i",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "or",
            "she",
            "that",
            "the",
            "their",
            "them",
            "they",
            "this",
            "to",
            "was",
            "were",
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "with",
        }
    )

    def __init__(
        self,
        *,
        memory_service: (
            NeedlePersonaMemoryServiceProtocol
        ),
        config: (
            NeedlePersonaNodeConfig | None
        ) = None,
    ) -> None:
        if not isinstance(
            memory_service,
            NeedlePersonaMemoryServiceProtocol,
        ):
            raise TypeError(
                "memory_service must implement "
                "NeedlePersonaMemoryServiceProtocol."
            )

        self.memory_service = memory_service
        self.config = (
            config
            or NeedlePersonaNodeConfig()
        )

    # ------------------------------------------------------------------
    # Responder retrieval
    # ------------------------------------------------------------------

    def retrieve_initial_for_responder(
        self,
        *,
        responder: Agent,
        query: str,
        allowed_private_memory_ids: (
            Iterable[str] | None
        ) = None,
        include_shared: bool = True,
    ) -> NeedlePersonaRetrievalResult:
        """
        Retrieve the responder's currently accessible evidence before any new
        sharing request is approved.
        """
        return self.retrieve_accessible(
            agent=responder,
            query=query,
            top_k=(
                self.config
                .initial_retrieval_top_k
            ),
            include_private=True,
            include_shared=include_shared,
            allowed_private_memory_ids=(
                allowed_private_memory_ids
            ),
        )

    def retrieve_final_for_responder(
        self,
        *,
        responder: Agent,
        query: str,
        allowed_private_memory_ids: (
            Iterable[str] | None
        ) = None,
    ) -> NeedlePersonaRetrievalResult:
        """
        Retrieve evidence after approved promotion requests have updated the
        responder's persistent read ACL.
        """
        return self.retrieve_accessible(
            agent=responder,
            query=query,
            top_k=(
                self.config
                .final_retrieval_top_k
            ),
            include_private=True,
            include_shared=True,
            allowed_private_memory_ids=(
                allowed_private_memory_ids
            ),
        )

    def retrieve_accessible(
        self,
        *,
        agent: Agent,
        query: str,
        top_k: int,
        include_private: bool = True,
        include_shared: bool = True,
        allowed_private_memory_ids: (
            Iterable[str] | None
        ) = None,
        allowed_shared_memory_ids: (
            Iterable[str] | None
        ) = None,
    ) -> NeedlePersonaRetrievalResult:
        """
        Retrieve and double-check memories accessible to one Agent.

        ``MemoryService`` already performs permission filtering. This helper
        additionally intersects semantic results with the service's accessible
        memory listing and optional run-scoped ID allowlists.
        """
        clean_query = self._required_text(
            query,
            "query",
        )
        clean_top_k = self._positive_int(
            top_k,
            "top_k",
        )

        accessible_memories = self._list_accessible(
            agent=agent,
            include_private=include_private,
            include_shared=include_shared,
        )

        accessible_by_id = {
            self._memory_id(memory): memory
            for memory in accessible_memories
            if self._is_active(memory)
        }

        private_allowlist = (
            self._clean_id_set(
                allowed_private_memory_ids
            )
        )
        shared_allowlist = (
            self._clean_id_set(
                allowed_shared_memory_ids
            )
        )

        allowed_by_id: dict[str, Any] = {}

        for memory_id, memory in (
            accessible_by_id.items()
        ):
            scope = self._memory_scope(
                memory
            )

            if scope == "private":
                if not include_private:
                    continue

                if (
                    private_allowlist
                    and memory_id
                    not in private_allowlist
                ):
                    continue

            elif scope == "shared":
                if not include_shared:
                    continue

                if (
                    shared_allowlist
                    and memory_id
                    not in shared_allowlist
                ):
                    continue

            else:
                continue

            if not self._can_be_read_by(
                memory,
                agent.agent_id,
            ):
                continue

            allowed_by_id[
                memory_id
            ] = memory

        if not allowed_by_id:
            return NeedlePersonaRetrievalResult(
                memories=(),
                memory_ids=(),
                context=self.build_memory_context(
                    ()
                ),
            )

        fallback_used = False
        warnings: list[str] = []

        try:
            semantic_memories = (
                self.memory_service
                .retrieve_memories(
                    agent=agent,
                    query=clean_query,
                    top_k=max(
                        clean_top_k,
                        min(
                            len(allowed_by_id) * 2,
                            100,
                        ),
                    ),
                    include_private=(
                        include_private
                    ),
                    include_shared=(
                        include_shared
                    ),
                )
            )
        except Exception as error:
            if not (
                self.config
                .allow_lexical_retrieval_fallback
            ):
                raise NeedlePersonaRetrievalError(
                    "Semantic retrieval failed for "
                    f"{agent.agent_id!r}: {error}"
                ) from error

            semantic_memories = (
                self._lexical_rank_memories(
                    query=clean_query,
                    memories=(
                        allowed_by_id.values()
                    ),
                    top_k=clean_top_k,
                )
            )
            fallback_used = True
            warnings.append(
                "Semantic retrieval failed and "
                "lexical fallback was used for "
                f"{agent.agent_id}: {error}"
            )

        filtered = self._filter_ranked_memories(
            memories=semantic_memories,
            allowed_by_id=allowed_by_id,
            agent_id=agent.agent_id,
            top_k=clean_top_k,
        )

        # A semantic retriever can legally return an empty result. The fallback
        # is applied only when explicitly enabled so smoke tests remain useful
        # even when the vector index is absent or incomplete.
        if (
            not filtered
            and allowed_by_id
            and self.config
            .allow_lexical_retrieval_fallback
        ):
            filtered = (
                self._lexical_rank_memories(
                    query=clean_query,
                    memories=(
                        allowed_by_id.values()
                    ),
                    top_k=clean_top_k,
                )
            )

            if not fallback_used:
                fallback_used = True
                warnings.append(
                    "Semantic retrieval returned no "
                    "allowed memories; lexical fallback "
                    f"was used for {agent.agent_id}."
                )

        memory_ids = tuple(
            self._memory_id(memory)
            for memory in filtered
        )

        return NeedlePersonaRetrievalResult(
            memories=tuple(filtered),
            memory_ids=memory_ids,
            context=self.build_memory_context(
                filtered
            ),
            fallback_used=fallback_used,
            warnings=tuple(warnings),
        )

    # ------------------------------------------------------------------
    # Owner-scoped candidate discovery
    # ------------------------------------------------------------------

    def discover_cross_agent_candidates(
        self,
        *,
        query: str,
        responder_agent_id: str,
        owner_agents: Mapping[str, Agent],
        private_memory_ids: Mapping[
            str,
            Iterable[str],
        ],
        missing_information: (
            Iterable[str] | None
        ) = None,
    ) -> (
        NeedlePersonaCandidateDiscoveryResult
    ):
        """
        Discover candidate memories inside each non-responder owner's private
        memory space.

        Each owner retrieves only memories it is authorised to read. The
        returned candidate DTOs contain no memory content.
        """
        clean_query = self._build_candidate_query(
            question=query,
            missing_information=(
                missing_information
            ),
        )
        clean_responder_id = (
            self._required_text(
                responder_agent_id,
                "responder_agent_id",
            )
        )

        expected_agent_ids = set(
            AGENT_PERSONA_MAP
        )
        supplied_owner_ids = set(
            owner_agents
        )

        missing_owner_agents = (
            expected_agent_ids
            - supplied_owner_ids
        )

        if missing_owner_agents:
            raise NeedlePersonaRetrievalError(
                "Missing owner Agents for candidate "
                "discovery: "
                f"{sorted(missing_owner_agents)}."
            )

        if (
            clean_responder_id
            not in expected_agent_ids
        ):
            raise NeedlePersonaRetrievalError(
                "responder_agent_id must be one "
                "of the persona Agents; received "
                f"{clean_responder_id!r}."
            )

        ranked_candidates: list[
            tuple[
                float,
                int,
                str,
                NeedleMemoryCandidate,
            ]
        ] = []
        fallback_owner_ids: list[str] = []
        warnings: list[str] = []

        for owner_order, owner_agent_id in (
            enumerate(AGENT_PERSONA_MAP)
        ):
            if (
                owner_agent_id
                == clean_responder_id
            ):
                continue

            owner = owner_agents[
                owner_agent_id
            ]

            if (
                owner.agent_id
                != owner_agent_id
            ):
                raise NeedlePersonaRetrievalError(
                    "Owner Agent mapping key does "
                    "not match Agent.agent_id: "
                    f"{owner_agent_id!r} != "
                    f"{owner.agent_id!r}."
                )

            allowed_ids = self._clean_id_set(
                private_memory_ids.get(
                    owner_agent_id,
                    (),
                )
            )

            if not allowed_ids:
                continue

            owner_result = (
                self.retrieve_accessible(
                    agent=owner,
                    query=clean_query,
                    top_k=(
                        self.config
                        .candidate_top_k_per_owner
                    ),
                    include_private=True,
                    include_shared=False,
                    allowed_private_memory_ids=(
                        allowed_ids
                    ),
                )
            )

            if owner_result.fallback_used:
                fallback_owner_ids.append(
                    owner_agent_id
                )

            warnings.extend(
                owner_result.warnings
            )

            for local_rank, memory in (
                enumerate(owner_result.memories)
            ):
                memory_id = self._memory_id(
                    memory
                )

                if (
                    self._memory_scope(memory)
                    != "private"
                ):
                    continue

                if (
                    self._memory_owner_id(
                        memory
                    )
                    != owner_agent_id
                ):
                    continue

                score = (
                    self._candidate_score(
                        query=clean_query,
                        memory=memory,
                        rank=local_rank,
                    )
                )

                candidate = (
                    NeedleMemoryCandidate(
                        memory_id=memory_id,
                        owner_agent_id=(
                            owner_agent_id
                        ),
                        relevance_score=score,
                        reason=(
                            "Owner-scoped retrieval "
                            "identified this private "
                            "memory as potentially "
                            "relevant to the responder's "
                            "missing information."
                        ),
                    )
                )

                ranked_candidates.append(
                    (
                        -score,
                        owner_order,
                        memory_id,
                        candidate,
                    )
                )

        ranked_candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
            )
        )

        unique_candidates: list[
            NeedleMemoryCandidate
        ] = []
        seen_memory_ids: set[str] = set()

        for (
            _,
            _,
            memory_id,
            candidate,
        ) in ranked_candidates:
            if memory_id in seen_memory_ids:
                continue

            unique_candidates.append(
                candidate
            )
            seen_memory_ids.add(
                memory_id
            )

            if (
                len(unique_candidates)
                >= self.config
                .max_candidate_memories
            ):
                break

        return (
            NeedlePersonaCandidateDiscoveryResult(
                candidates=tuple(
                    unique_candidates
                ),
                fallback_owner_agent_ids=tuple(
                    dict.fromkeys(
                        fallback_owner_ids
                    )
                ),
                warnings=tuple(
                    dict.fromkeys(warnings)
                ),
            )
        )

    def get_candidate_memory(
        self,
        *,
        owner: Agent,
        memory_id: str,
        expected_owner_agent_id: (
            str | None
        ) = None,
    ) -> Any:
        """
        Load one candidate through its owner Agent for Critic/Coordinator prompt
        construction.

        The returned memory must remain inside the governance node. Its content
        must not be copied into responder state before approval.
        """
        clean_memory_id = self._required_text(
            memory_id,
            "memory_id",
        )

        try:
            memory = self.memory_service.get_memory(
                owner,
                clean_memory_id,
            )
        except Exception as error:
            raise NeedlePersonaRetrievalError(
                "Failed to load candidate memory "
                f"{clean_memory_id!r} through owner "
                f"{owner.agent_id!r}: {error}"
            ) from error

        actual_owner_id = (
            self._memory_owner_id(
                memory
            )
        )
        required_owner_id = (
            self._required_text(
                expected_owner_agent_id,
                "expected_owner_agent_id",
            )
            if expected_owner_agent_id
            is not None
            else owner.agent_id
        )

        if (
            actual_owner_id
            != required_owner_id
        ):
            raise NeedlePersonaRetrievalError(
                "Candidate owner mismatch for "
                f"{clean_memory_id!r}: "
                f"{actual_owner_id!r} != "
                f"{required_owner_id!r}."
            )

        if (
            self._memory_scope(memory)
            != "private"
        ):
            raise NeedlePersonaRetrievalError(
                "Candidate memory must still be "
                "private before access approval: "
                f"{clean_memory_id!r}."
            )

        if not self._is_active(memory):
            raise NeedlePersonaRetrievalError(
                "Candidate memory is not active: "
                f"{clean_memory_id!r}."
            )

        return memory

    # ------------------------------------------------------------------
    # Prompt context
    # ------------------------------------------------------------------

    def build_memory_context(
        self,
        memories: Iterable[Any],
        *,
        limit: int | None = None,
    ) -> str:
        """
        Render permission-filtered memories into a stable LLM context.

        Exact memory IDs, owner IDs, and source IDs are shown so AgentAnswer
        can report grounded references.
        """
        context_limit = (
            self.config.memory_context_limit
            if limit is None
            else self._positive_int(
                limit,
                "limit",
            )
        )

        unique_memories: list[Any] = []
        seen_ids: set[str] = set()

        for memory in memories:
            memory_id = self._memory_id(
                memory
            )

            if memory_id in seen_ids:
                continue

            unique_memories.append(
                memory
            )
            seen_ids.add(memory_id)

            if (
                len(unique_memories)
                >= context_limit
            ):
                break

        if not unique_memories:
            return "No accessible memory."

        sections: list[str] = []

        for index, memory in enumerate(
            unique_memories,
            start=1,
        ):
            prompt_memory = (
                self.memory_to_prompt_dict(
                    memory
                )
            )

            sections.append(
                "\n".join(
                    [
                        f"[Memory {index}]",
                        (
                            "memory_id: "
                            f"{prompt_memory['memory_id']}"
                        ),
                        (
                            "owner_agent_id: "
                            f"{prompt_memory['owner_agent_id']}"
                        ),
                        (
                            "scope: "
                            f"{prompt_memory['scope']}"
                        ),
                        (
                            "readable_by: "
                            + json.dumps(
                                prompt_memory[
                                    "readable_by"
                                ],
                                ensure_ascii=False,
                            )
                        ),
                        (
                            "source_ids: "
                            + json.dumps(
                                prompt_memory[
                                    "source_ids"
                                ],
                                ensure_ascii=False,
                            )
                        ),
                        (
                            "memory_type: "
                            f"{prompt_memory['memory_type']}"
                        ),
                        (
                            "summary: "
                            f"{prompt_memory['summary']}"
                        ),
                        (
                            "content: "
                            f"{prompt_memory['content']}"
                        ),
                    ]
                )
            )

        return "\n\n".join(sections)

    def memory_to_prompt_dict(
        self,
        memory: Any,
    ) -> dict[str, Any]:
        """
        Convert one authorised memory into a JSON-compatible prompt record.
        """
        metadata = self._memory_metadata(
            memory
        )

        content = str(
            self._read_field(
                memory,
                "content",
                "",
            )
            or ""
        ).strip()

        if not content:
            raise NeedlePersonaRetrievalError(
                "Memory content cannot be empty: "
                f"{self._memory_id(memory)!r}."
            )

        summary_value = self._read_field(
            memory,
            "summary",
            None,
        )
        summary = (
            str(summary_value).strip()
            if summary_value is not None
            else ""
        )

        return {
            "memory_id": self._memory_id(
                memory
            ),
            "content": content,
            "summary": (
                summary
                or "No summary."
            ),
            "owner_agent_id": (
                self._memory_owner_id(
                    memory
                )
            ),
            "scope": self._memory_scope(
                memory
            ),
            "readable_by": (
                self._clean_string_list(
                    self._read_field(
                        metadata,
                        "readable_by",
                        [],
                    )
                )
            ),
            "source_ids": (
                self._clean_string_list(
                    self._read_field(
                        metadata,
                        "source_message_ids",
                        [],
                    )
                )
            ),
            "memory_type": (
                self._enum_value(
                    self._read_field(
                        metadata,
                        "memory_type",
                        "note",
                    )
                )
            ),
            "status": self._enum_value(
                self._read_field(
                    metadata,
                    "status",
                    "active",
                )
            ),
            "tags": self._clean_string_list(
                self._read_field(
                    metadata,
                    "tags",
                    [],
                )
            ),
            "importance": self._safe_float(
                self._read_field(
                    metadata,
                    "importance",
                    0.5,
                ),
                default=0.5,
            ),
            "confidence": self._safe_float(
                self._read_field(
                    metadata,
                    "confidence",
                    1.0,
                ),
                default=1.0,
            ),
        }

    # ------------------------------------------------------------------
    # Retrieval internals
    # ------------------------------------------------------------------

    def _list_accessible(
        self,
        *,
        agent: Agent,
        include_private: bool,
        include_shared: bool,
    ) -> list[Any]:
        try:
            memories = (
                self.memory_service
                .list_accessible_memories(
                    agent=agent,
                    include_private=(
                        include_private
                    ),
                    include_shared=(
                        include_shared
                    ),
                    active_only=True,
                )
            )
        except Exception as error:
            raise NeedlePersonaRetrievalError(
                "Failed to list accessible "
                f"memories for {agent.agent_id!r}: "
                f"{error}"
            ) from error

        return list(memories or [])

    def _filter_ranked_memories(
        self,
        *,
        memories: Iterable[Any],
        allowed_by_id: Mapping[str, Any],
        agent_id: str,
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

            allowed_memory = (
                allowed_by_id.get(
                    memory_id
                )
            )

            if allowed_memory is None:
                continue

            if not self._is_active(
                allowed_memory
            ):
                continue

            if not self._can_be_read_by(
                allowed_memory,
                agent_id,
            ):
                continue

            result.append(
                allowed_memory
            )
            seen.add(memory_id)

            if len(result) >= top_k:
                break

        return result

    def _lexical_rank_memories(
        self,
        *,
        query: str,
        memories: Iterable[Any],
        top_k: int,
    ) -> list[Any]:
        query_terms = self._tokenise(
            query
        )
        scored: list[
            tuple[
                float,
                float,
                float,
                str,
                Any,
            ]
        ] = []

        for memory in memories:
            memory_id = self._memory_id(
                memory
            )
            memory_terms = self._tokenise(
                self._memory_search_text(
                    memory
                )
            )

            lexical_score = (
                self._term_overlap_score(
                    query_terms,
                    memory_terms,
                )
            )
            metadata = self._memory_metadata(
                memory
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

            scored.append(
                (
                    -lexical_score,
                    -importance,
                    -confidence,
                    memory_id,
                    memory,
                )
            )

        scored.sort(
            key=lambda item: item[:4]
        )

        return [
            memory
            for _, _, _, _, memory
            in scored[:top_k]
        ]

    def _candidate_score(
        self,
        *,
        query: str,
        memory: Any,
        rank: int,
    ) -> float:
        query_terms = self._tokenise(
            query
        )
        memory_terms = self._tokenise(
            self._memory_search_text(
                memory
            )
        )

        lexical_score = (
            self._term_overlap_score(
                query_terms,
                memory_terms,
            )
        )
        reciprocal_rank = 1.0 / (
            rank + 1
        )

        metadata = self._memory_metadata(
            memory
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

        score = (
            0.65 * lexical_score
            + 0.25 * reciprocal_rank
            + 0.05 * importance
            + 0.05 * confidence
        )

        return round(
            min(max(score, 0.0), 1.0),
            6,
        )

    def _memory_search_text(
        self,
        memory: Any,
    ) -> str:
        metadata = self._memory_metadata(
            memory
        )

        values: list[Any] = [
            self._read_field(
                memory,
                "content",
                "",
            ),
            self._read_field(
                memory,
                "summary",
                "",
            ),
            self._read_field(
                metadata,
                "memory_type",
                "",
            ),
            *(
                self._read_field(
                    metadata,
                    "tags",
                    [],
                )
                or []
            ),
        ]

        return " ".join(
            self._enum_value(value)
            for value in values
            if value is not None
        )

    @classmethod
    def _build_candidate_query(
        cls,
        *,
        question: str,
        missing_information: (
            Iterable[str] | None
        ),
    ) -> str:
        clean_question = cls._required_text(
            question,
            "question",
        )

        missing = [
            " ".join(
                str(item).split()
            )
            for item in (
                missing_information or []
            )
            if " ".join(
                str(item).split()
            )
        ]

        if not missing:
            return clean_question

        return (
            clean_question
            + "\nMissing information: "
            + "; ".join(missing)
        )

    @classmethod
    def _tokenise(
        cls,
        text: Any,
    ) -> set[str]:
        tokens = {
            token.casefold()
            for token in cls._TOKEN_PATTERN.findall(
                str(text or "")
            )
        }

        return {
            token
            for token in tokens
            if token not in cls._STOP_WORDS
        }

    @staticmethod
    def _term_overlap_score(
        query_terms: set[str],
        memory_terms: set[str],
    ) -> float:
        if not query_terms:
            return 0.0

        return len(
            query_terms & memory_terms
        ) / len(query_terms)

    # ------------------------------------------------------------------
    # Memory-schema helpers
    # ------------------------------------------------------------------

    @classmethod
    def _memory_id(
        cls,
        memory: Any,
    ) -> str:
        memory_id = cls._required_text(
            cls._read_field(
                memory,
                "memory_id",
                "",
            ),
            "memory_id",
        )
        return memory_id

    @classmethod
    def _memory_metadata(
        cls,
        memory: Any,
    ) -> Any:
        metadata = cls._read_field(
            memory,
            "metadata",
            None,
        )

        if metadata is None:
            raise NeedlePersonaRetrievalError(
                "Memory is missing metadata: "
                f"{cls._read_field(memory, 'memory_id', '<unknown>')!r}."
            )

        return metadata

    @classmethod
    def _memory_owner_id(
        cls,
        memory: Any,
    ) -> str:
        metadata = cls._memory_metadata(
            memory
        )

        return cls._required_text(
            cls._read_field(
                metadata,
                "owner_agent_id",
                "",
            ),
            "owner_agent_id",
        )

    @classmethod
    def _memory_scope(
        cls,
        memory: Any,
    ) -> str:
        metadata = cls._memory_metadata(
            memory
        )

        return cls._enum_value(
            cls._read_field(
                metadata,
                "scope",
                "",
            )
        ).strip().lower()

    @classmethod
    def _is_active(
        cls,
        memory: Any,
    ) -> bool:
        metadata = cls._memory_metadata(
            memory
        )
        status = cls._enum_value(
            cls._read_field(
                metadata,
                "status",
                "active",
            )
        ).strip().lower()

        return status == "active"

    @classmethod
    def _can_be_read_by(
        cls,
        memory: Any,
        agent_id: str,
    ) -> bool:
        clean_agent_id = cls._required_text(
            agent_id,
            "agent_id",
        )

        method = getattr(
            memory,
            "can_be_read_by",
            None,
        )

        if callable(method):
            try:
                return bool(
                    method(clean_agent_id)
                )
            except Exception:
                pass

        metadata = cls._memory_metadata(
            memory
        )
        readers = cls._clean_string_list(
            cls._read_field(
                metadata,
                "readable_by",
                [],
            )
        )

        return (
            "*" in readers
            or clean_agent_id in readers
        )

    @staticmethod
    def _read_field(
        instance: Any,
        field_name: str,
        default: Any = None,
    ) -> Any:
        if isinstance(
            instance,
            Mapping,
        ):
            return instance.get(
                field_name,
                default,
            )

        return getattr(
            instance,
            field_name,
            default,
        )

    @staticmethod
    def _enum_value(
        value: Any,
    ) -> str:
        raw_value = getattr(
            value,
            "value",
            value,
        )

        return str(
            raw_value
        ).strip()

    @classmethod
    def _clean_id_set(
        cls,
        values: Iterable[str] | None,
    ) -> set[str]:
        return set(
            cls._clean_string_list(
                values
            )
        )

    @staticmethod
    def _clean_string_list(
        values: Iterable[Any] | None,
    ) -> list[str]:
        if values is None:
            return []

        if isinstance(values, str):
            values = [values]

        result: list[str] = []

        for value in values:
            cleaned = str(
                getattr(
                    value,
                    "value",
                    value,
                )
            ).strip()

            if (
                cleaned
                and cleaned not in result
            ):
                result.append(cleaned)

        return result

    @staticmethod
    def _safe_float(
        value: Any,
        *,
        default: float,
    ) -> float:
        try:
            return float(value)
        except (
            TypeError,
            ValueError,
        ):
            return default

    @classmethod
    def _bounded_float(
        cls,
        value: Any,
        *,
        default: float,
    ) -> float:
        number = cls._safe_float(
            value,
            default=default,
        )

        return min(
            max(number, 0.0),
            1.0,
        )

    @staticmethod
    def _positive_int(
        value: Any,
        field_name: str,
    ) -> int:
        try:
            number = int(value)
        except (
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                f"{field_name} must be an integer."
            ) from error

        if number <= 0:
            raise ValueError(
                f"{field_name} must be greater "
                "than zero."
            )

        return number

    @staticmethod
    def _required_text(
        value: Any,
        field_name: str,
    ) -> str:
        cleaned = " ".join(
            str(value or "").split()
        )

        if not cleaned:
            raise ValueError(
                f"{field_name} cannot be empty."
            )

        return cleaned


__all__ = [
    "NeedlePersonaRetrievalError",
    "NeedlePersonaRetrievalResult",
    "NeedlePersonaCandidateDiscoveryResult",
    "NeedlePersonaRetrievalService",
]