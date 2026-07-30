from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any, Protocol, runtime_checkable

from src.llm import (
    LLMClient,
    MemoryExtractionOutput,
    PromptTemplates,
    WorkerPromptTemplate,
)
from src.memory.entities import (
    Agent,
    AgentRole,
    MemoryType,
    SourceType,
)

from .needle_persona_models import (
    AGENT_PERSONA_MAP,
    PERSONA_AGENT_MAP,
    PERSONA_NAMES,
    NeedlePersonaSample,
    PersonaName,
    PersonaSource,
    PrivateMemoryIndex,
)


@runtime_checkable
class MemoryServiceProtocol(Protocol):
    """
    Minimal MemoryService interface required by this ingestion module.

    Using a protocol keeps the benchmark code independent of whether the
    concrete service package is named ``service`` or ``services``.
    """

    def create_private_memory(
        self,
        agent: Agent,
        content: str,
        summary: str | None = None,
        memory_type: MemoryType | str = MemoryType.NOTE,
        tags: list[str] | None = None,
        importance: float = 0.5,
        confidence: float = 1.0,
        source_task_id: str | None = None,
        source_message_ids: list[str] | None = None,
        source_type: SourceType | str = SourceType.AGENT_OUTPUT,
        reason: str | None = None,
    ) -> Any:
        ...

    def deprecate_memory(
        self,
        agent: Agent,
        memory_id: str,
        reason: str | None = None,
    ) -> Any:
        ...


class NeedlePersonaIngestionError(Exception):
    """
    Raised when private-memory ingestion cannot be completed safely.
    """

    def __init__(
        self,
        message: str,
        *,
        run_id: str | None = None,
        sample_id: str | None = None,
        agent_id: str | None = None,
        source_id: str | None = None,
        rollback_errors: list[str] | None = None,
    ) -> None:
        context: list[str] = []

        if run_id:
            context.append(f"run_id={run_id!r}")

        if sample_id:
            context.append(f"sample_id={sample_id!r}")

        if agent_id:
            context.append(f"agent_id={agent_id!r}")

        if source_id:
            context.append(f"source_id={source_id!r}")

        context_text = (
            f" ({', '.join(context)})"
            if context
            else ""
        )

        rollback_text = ""

        if rollback_errors:
            rollback_text = (
                " Rollback also reported: "
                + " | ".join(rollback_errors)
            )

        super().__init__(
            f"{message}{context_text}.{rollback_text}".strip()
        )

        self.run_id = run_id
        self.sample_id = sample_id
        self.agent_id = agent_id
        self.source_id = source_id
        self.rollback_errors = rollback_errors or []


class NeedlePersonaIngestionService:
    """
    Extract and persist persona-owned private memories for one benchmark sample.

    Flow:
        NeedlePersonaSample
            -> select each persona's own sources
            -> build a generic worker memory-extraction prompt
            -> invoke MemoryExtractionOutput
            -> validate provenance and ownership
            -> create private memories through MemoryService
            -> return only memory IDs in PrivateMemoryIndex

    Important isolation rule:
        The output index contains no raw dialogue, persona source text, gold
        answer, or collaborative chat. Only IDs are passed to the task workflow.

    By default, the benchmark question is not shown during ingestion. This
    avoids task-conditioned extraction that would selectively surface the
    needle before the actual workflow begins.
    """

    _PREFERENCE_PATTERNS = (
        r"\bfavou?rite\b",
        r"\bprefers?\b",
        r"\blikes?\b",
        r"\bloves?\b",
        r"\benjoys?\b",
        r"\binterested in\b",
        r"\bnot interested in\b",
        r"\bdislikes?\b",
        r"\bhates?\b",
        r"\bpassionate about\b",
        r"\benthusiastic about\b",
    )

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        memory_service: MemoryServiceProtocol,
        agents: Mapping[str, Agent],
        worker_prompts: Mapping[str, WorkerPromptTemplate] | None = None,
        include_task_context: bool = False,
        rollback_on_error: bool = True,
        require_memory_per_source: bool = False,
        max_memories_per_source: int | None = None,
    ) -> None:
        """
        Args:
            llm_client:
                Generic structured-output client.
            memory_service:
                Existing memory service. It must support private-memory creation
                and deprecation.
            agents:
                Mapping containing alice_agent, bob_agent, charlie_agent, and
                dave_agent.
            worker_prompts:
                Optional preconfigured prompt templates for the four agents.
                When omitted, safe default templates are created.
            include_task_context:
                Whether to show the benchmark question during extraction.
                Keep False for the main evaluation.
            rollback_on_error:
                Deprecate memories already created in the current ingestion
                attempt if a later extraction or write fails.
            require_memory_per_source:
                Raise when a source produces no memory candidates.
            max_memories_per_source:
                Optional cap applied after extraction and deduplication.
        """
        if max_memories_per_source is not None:
            if max_memories_per_source <= 0:
                raise ValueError(
                    "max_memories_per_source must be greater than zero."
                )

        self.llm_client = llm_client
        self.memory_service = memory_service
        self.agents = dict(agents)
        self.include_task_context = include_task_context
        self.rollback_on_error = rollback_on_error
        self.require_memory_per_source = require_memory_per_source
        self.max_memories_per_source = max_memories_per_source

        self._validate_agents()

        self.worker_prompts = (
            dict(worker_prompts)
            if worker_prompts is not None
            else build_default_worker_prompts()
        )

        self._validate_worker_prompts()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def ingest(
        self,
        *,
        sample: NeedlePersonaSample,
        run_id: str,
    ) -> PrivateMemoryIndex:
        """
        Extract and persist all four agents' private memories for one sample.

        This method is intentionally outside LangGraph. Raw source content
        should be discarded after this method returns.
        """
        run_id = self._required_text(run_id, "run_id")

        private_memory_ids: dict[str, list[str]] = {
            agent_id: []
            for agent_id in AGENT_PERSONA_MAP
        }
        source_to_memory_ids: dict[str, list[str]] = {}

        # Used for rollback in reverse creation order.
        created_records: list[tuple[str, str]] = []

        # Prevent the same LLM statement from being written repeatedly for one
        # persona when multiple sources overlap.
        seen_content_by_agent: dict[str, set[str]] = {
            agent_id: set()
            for agent_id in AGENT_PERSONA_MAP
        }

        current_agent_id: str | None = None
        current_source_id: str | None = None

        try:
            for persona in PERSONA_NAMES:
                current_agent_id = PERSONA_AGENT_MAP[persona]
                agent = self.agents[current_agent_id]
                prompt_template = self.worker_prompts[current_agent_id]

                for source in sample.get_sources(persona):
                    current_source_id = source.source_id

                    extracted = self._extract_source_memories(
                        sample=sample,
                        run_id=run_id,
                        agent=agent,
                        source=source,
                        prompt_template=prompt_template,
                    )

                    candidates = self._prepare_candidates(
                        extraction=extracted,
                        expected_agent_id=current_agent_id,
                        allowed_source_ids={source.source_id},
                        seen_content=seen_content_by_agent[
                            current_agent_id
                        ],
                    )

                    if (
                        self.require_memory_per_source
                        and not candidates
                    ):
                        raise NeedlePersonaIngestionError(
                            "The source produced no valid memory candidates",
                            run_id=run_id,
                            sample_id=sample.sample_id,
                            agent_id=current_agent_id,
                            source_id=source.source_id,
                        )

                    source_memory_ids: list[str] = []

                    for candidate in candidates:
                        created_memory = (
                            self.memory_service.create_private_memory(
                                agent=agent,
                                content=candidate.content,
                                summary=candidate.subject,
                                memory_type=self._map_memory_type(
                                    candidate.memory_type,
                                    candidate.content,
                                ),
                                tags=self._build_tags(
                                    run_id=run_id,
                                    sample=sample,
                                    persona=persona,
                                    agent_id=current_agent_id,
                                    source=source,
                                    shareable=candidate.shareable,
                                ),
                                importance=candidate.importance,
                                confidence=candidate.confidence,
                                source_task_id=self._source_task_id(
                                    run_id=run_id,
                                    sample_id=sample.sample_id,
                                ),
                                source_message_ids=candidate.source_ids,
                                source_type=SourceType.MESSAGE,
                                reason=(
                                    "Private persona memory created during "
                                    "InformativeBench Needle ingestion."
                                ),
                            )
                        )

                        memory_id = self._extract_memory_id(
                            created_memory
                        )

                        private_memory_ids[
                            current_agent_id
                        ].append(memory_id)
                        source_memory_ids.append(memory_id)
                        created_records.append(
                            (current_agent_id, memory_id)
                        )

                    source_to_memory_ids[
                        source.source_id
                    ] = source_memory_ids

            return PrivateMemoryIndex(
                run_id=run_id,
                sample_id=sample.sample_id,
                private_memory_ids=private_memory_ids,
                source_to_memory_ids=source_to_memory_ids,
            )

        except NeedlePersonaIngestionError:
            rollback_errors = self._rollback(
                created_records=created_records,
                run_id=run_id,
                sample_id=sample.sample_id,
            )

            if rollback_errors:
                raise NeedlePersonaIngestionError(
                    "Private-memory ingestion failed",
                    run_id=run_id,
                    sample_id=sample.sample_id,
                    agent_id=current_agent_id,
                    source_id=current_source_id,
                    rollback_errors=rollback_errors,
                )

            raise

        except Exception as error:
            rollback_errors = self._rollback(
                created_records=created_records,
                run_id=run_id,
                sample_id=sample.sample_id,
            )

            raise NeedlePersonaIngestionError(
                f"Private-memory ingestion failed: {error}",
                run_id=run_id,
                sample_id=sample.sample_id,
                agent_id=current_agent_id,
                source_id=current_source_id,
                rollback_errors=rollback_errors,
            ) from error

    # ------------------------------------------------------------------
    # LLM extraction
    # ------------------------------------------------------------------

    def _extract_source_memories(
        self,
        *,
        sample: NeedlePersonaSample,
        run_id: str,
        agent: Agent,
        source: PersonaSource,
        prompt_template: WorkerPromptTemplate,
    ) -> MemoryExtractionOutput:
        task_context = (
            sample.question
            if self.include_task_context
            else None
        )

        prompt = prompt_template.build_memory_extraction_prompt(
            source_content=source.content,
            source_ids=[source.source_id],
            task_context=task_context,
            context_sections={
                "Benchmark": "InformativeBench",
                "Subset": "Needle in the Persona",
                "Run ID": run_id,
                "Sample ID": sample.sample_id,
                "Owner agent ID": agent.agent_id,
                "Owner persona": source.persona,
                "Source type": source.source_type,
                "Conversation ID": source.conversation_id,
                "Task-conditioned extraction": (
                    self.include_task_context
                ),
            },
        )

        return self.llm_client.invoke_structured(
            prompt,
            MemoryExtractionOutput,
        )

    def _prepare_candidates(
        self,
        *,
        extraction: MemoryExtractionOutput,
        expected_agent_id: str,
        allowed_source_ids: set[str],
        seen_content: set[str],
    ) -> list[Any]:
        """
        Validate agent identity and source provenance, then remove duplicate
        candidates while preserving order.
        """
        if extraction.agent_id != expected_agent_id:
            raise NeedlePersonaIngestionError(
                "The extraction output returned the wrong agent_id",
                agent_id=expected_agent_id,
            )

        prepared: list[Any] = []

        for candidate in extraction.memories:
            source_ids = list(candidate.source_ids)

            # The model may omit source_ids even though only one source was
            # supplied. In that case, attach the known source deterministically.
            if not source_ids:
                source_ids = sorted(allowed_source_ids)

            unexpected_source_ids = (
                set(source_ids) - allowed_source_ids
            )

            if unexpected_source_ids:
                raise NeedlePersonaIngestionError(
                    "The extraction output referenced unavailable source IDs: "
                    f"{sorted(unexpected_source_ids)}",
                    agent_id=expected_agent_id,
                )

            content_key = self._normalise_content_key(
                candidate.content
            )

            if content_key in seen_content:
                continue

            seen_content.add(content_key)

            prepared.append(
                candidate.model_copy(
                    update={
                        "source_ids": source_ids,
                    }
                )
            )

        if self.max_memories_per_source is not None:
            prepared = prepared[
                : self.max_memories_per_source
            ]

        return prepared

    # ------------------------------------------------------------------
    # Memory mapping and metadata
    # ------------------------------------------------------------------

    @classmethod
    def _map_memory_type(
        cls,
        extracted_type: str,
        content: str,
    ) -> MemoryType:
        """
        Map the generic LLM extraction taxonomy onto the project's memory
        taxonomy.

        semantic:
            preference-like statements -> PREFERENCE
            all other statements -> FACT
        episodic -> EVENT
        procedural -> RULE
        """
        normalised_type = str(extracted_type).strip().lower()

        if normalised_type == "episodic":
            return MemoryType.EVENT

        if normalised_type == "procedural":
            return MemoryType.RULE

        if cls._looks_like_preference(content):
            return MemoryType.PREFERENCE

        return MemoryType.FACT

    @classmethod
    def _looks_like_preference(
        cls,
        content: str,
    ) -> bool:
        text = content.lower()

        return any(
            re.search(pattern, text)
            for pattern in cls._PREFERENCE_PATTERNS
        )

    @classmethod
    def _build_tags(
        cls,
        *,
        run_id: str,
        sample: NeedlePersonaSample,
        persona: PersonaName,
        agent_id: str,
        source: PersonaSource,
        shareable: bool,
    ) -> list[str]:
        return [
            "benchmark:informativebench",
            "subset:needle_in_the_persona",
            "stage:private_ingestion",
            f"run:{cls._tag_value(run_id)}",
            f"sample:{cls._tag_value(sample.sample_id)}",
            f"persona:{persona}",
            f"owner:{cls._tag_value(agent_id)}",
            f"source:{cls._tag_value(source.source_id)}",
            f"source_type:{source.source_type}",
            (
                "promotion_candidate"
                if shareable
                else "keep_private"
            ),
        ]

    @staticmethod
    def _source_task_id(
        *,
        run_id: str,
        sample_id: str,
    ) -> str:
        return f"needle:{sample_id}:run:{run_id}"

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def _rollback(
        self,
        *,
        created_records: list[tuple[str, str]],
        run_id: str,
        sample_id: str,
    ) -> list[str]:
        """
        Soft-delete memories created in a failed ingestion attempt.

        MemoryStore uses lifecycle-aware deprecation rather than hard deletion,
        so rollback remains auditable.
        """
        if not self.rollback_on_error:
            return []

        rollback_errors: list[str] = []

        for agent_id, memory_id in reversed(created_records):
            try:
                self.memory_service.deprecate_memory(
                    agent=self.agents[agent_id],
                    memory_id=memory_id,
                    reason=(
                        "Rollback after failed Needle persona ingestion "
                        f"for run={run_id}, sample={sample_id}."
                    ),
                )
            except Exception as error:
                rollback_errors.append(
                    f"{memory_id}: {error}"
                )

        return rollback_errors

    # ------------------------------------------------------------------
    # Constructor validation
    # ------------------------------------------------------------------

    def _validate_agents(self) -> None:
        expected_agent_ids = set(AGENT_PERSONA_MAP)
        actual_agent_ids = set(self.agents)

        missing = expected_agent_ids - actual_agent_ids

        if missing:
            raise ValueError(
                "Missing persona agents: "
                f"{sorted(missing)}."
            )

        for agent_id in expected_agent_ids:
            agent = self.agents[agent_id]

            if agent.agent_id != agent_id:
                raise ValueError(
                    f"Agent mapping key {agent_id!r} does not match "
                    f"Agent.agent_id {agent.agent_id!r}."
                )

            if agent.role != AgentRole.WORKER:
                raise ValueError(
                    f"Persona agent {agent_id!r} must have "
                    "AgentRole.WORKER."
                )

            if not agent.can_write_private:
                raise ValueError(
                    f"Persona agent {agent_id!r} cannot write "
                    "private memory."
                )

    def _validate_worker_prompts(self) -> None:
        expected_agent_ids = set(AGENT_PERSONA_MAP)
        actual_agent_ids = set(self.worker_prompts)

        missing = expected_agent_ids - actual_agent_ids

        if missing:
            raise ValueError(
                "Missing worker prompt templates: "
                f"{sorted(missing)}."
            )

        for agent_id in expected_agent_ids:
            template = self.worker_prompts[agent_id]

            if template.agent_id != agent_id:
                raise ValueError(
                    f"Prompt template key {agent_id!r} does not "
                    f"match template.agent_id "
                    f"{template.agent_id!r}."
                )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_memory_id(
        created_memory: Any,
    ) -> str:
        memory_id = getattr(
            created_memory,
            "memory_id",
            None,
        )

        if memory_id is None and isinstance(
            created_memory,
            Mapping,
        ):
            memory_id = created_memory.get("memory_id")

        memory_id = (
            str(memory_id).strip()
            if memory_id is not None
            else ""
        )

        if not memory_id:
            raise NeedlePersonaIngestionError(
                "MemoryService.create_private_memory() returned "
                "an object without memory_id"
            )

        return memory_id

    @staticmethod
    def _normalise_content_key(
        content: str,
    ) -> str:
        return re.sub(
            r"\s+",
            " ",
            content.strip().lower(),
        )

    @staticmethod
    def _tag_value(
        value: str,
    ) -> str:
        cleaned = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            str(value).strip(),
        ).strip("_")

        return cleaned or "unknown"

    @staticmethod
    def _required_text(
        value: str,
        field_name: str,
    ) -> str:
        text = str(value).strip()

        if not text:
            raise ValueError(
                f"{field_name} cannot be empty."
            )

        return text


def build_default_persona_agents() -> dict[str, Agent]:
    """
    Create the four Worker agents used by the benchmark workflow.
    """
    return {
        agent_id: Agent.worker(
            agent_id=agent_id,
            name=f"{persona.title()} Agent",
        )
        for persona, agent_id in PERSONA_AGENT_MAP.items()
    }


def build_default_worker_prompts() -> dict[
    str,
    WorkerPromptTemplate,
]:
    """
    Build one shared prompt design instantiated for four persona owners.
    """
    prompts: dict[str, WorkerPromptTemplate] = {}

    for persona, agent_id in PERSONA_AGENT_MAP.items():
        display_name = persona.title()

        prompts[agent_id] = PromptTemplates.worker(
            agent_id=agent_id,
            role_description=(
                f"Represent {display_name}. Extract, retain, and use only "
                f"information owned by {display_name}."
            ),
            additional_rules=[
                (
                    f"Do not attribute another person's statements, "
                    f"preferences, experiences, or plans to "
                    f"{display_name}."
                ),
                (
                    "A memory may be marked shareable, but it must remain "
                    "private until the governance workflow explicitly "
                    "approves promotion."
                ),
            ],
        )

    return prompts


def ingest_samples(
    *,
    service: NeedlePersonaIngestionService,
    samples: Iterable[NeedlePersonaSample],
    run_id_prefix: str,
) -> list[PrivateMemoryIndex]:
    """
    Convenience helper for sequential ingestion of multiple samples.

    Each sample receives its own run ID so memory provenance and experimental
    cleanup remain isolated.
    """
    prefix = str(run_id_prefix).strip()

    if not prefix:
        raise ValueError("run_id_prefix cannot be empty.")

    results: list[PrivateMemoryIndex] = []

    for index, sample in enumerate(samples):
        run_id = (
            f"{prefix}_{index:04d}_"
            f"{NeedlePersonaIngestionService._tag_value(sample.sample_id)}"
        )

        results.append(
            service.ingest(
                sample=sample,
                run_id=run_id,
            )
        )

    return results


__all__ = [
    "MemoryServiceProtocol",
    "NeedlePersonaIngestionError",
    "NeedlePersonaIngestionService",
    "build_default_persona_agents",
    "build_default_worker_prompts",
    "ingest_samples",
]