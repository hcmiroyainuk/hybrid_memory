from __future__ import annotations

"""
GateMem adapter for the hybrid multi-agent memory-governance system.

The adapter implements GateMem's incremental lifecycle:

    reset(episode)
        -> ingest(turn) in chronological order
        -> query(checkpoint)

Design invariants
-----------------
1. Every GateMem principal is represented by one Worker.
2. Every episode receives an isolated runtime and persistence namespace.
3. Every ordinary turn is created as private memory first.
4. The three experiment modes differ only in the access-policy path:
   - private_only: keep the memory private;
   - ungoverned_shared: grant global read access directly;
   - governed_shared: run Coordinator/Critic policy assignment.
5. Query-time retrieval is permission-first. This adapter never requests or
   grants access merely because a checkpoint query was asked.
6. GateMem evaluation-only fields never enter the runtime, prompt, or answer.
7. Deletion commands are lifecycle events rather than retrievable memories.
"""

import json
import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

from src.memory.services.memory_forgetting_service import (
    MemoryForgettingService,
)

from pydantic import BaseModel

from bench.agents.base import (
    BaseMemoryAgent,
    Checkpoint,
    Turn,
)

from src.llm import LLMClient
from src.llm.output_schemas import (
    MemoryAccessDecisionOutput,
    MemoryAccessRequestDecisionOutput,
    MemoryAccessRequestReviewOutput,
    MemoryAccessReviewOutput,
)
from src.memory.entities import Agent
from src.memory.governance.policy_assignment_state import (
    build_initial_policy_assignment_state,
)

from .answer_service import (
    GateMemAnswerConfig,
    GateMemAnswerResult,
    GateMemAnswerService,
)
from .mapper import (
    GateMemMapper,
    MappedEpisode,
    MappedQuery,
    MappedTurn,
)
from .runtime_factory import (
    CoordinatorFactory,
    CriticFactory,
    EvaluationMode,
    GateMemEpisodeRuntime,
    GateMemRuntimeFactory,
    normalise_evaluation_mode,
)


SchemaT = TypeVar(
    "SchemaT",
    bound=BaseModel,
)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class StructuredLLMProtocol(Protocol):
    """Structured-output capability used by answer and governance adapters."""

    def invoke_structured(
        self,
        prompt: str,
        schema_model: type[SchemaT],
        *,
        apply_parser_normalization: bool = True,
    ) -> SchemaT:
        ...


# ---------------------------------------------------------------------------
# Errors and configuration
# ---------------------------------------------------------------------------


class GateMemAgentError(RuntimeError):
    """Base exception raised by the GateMem system adapter."""


class GateMemAgentConfigurationError(GateMemAgentError):
    """Raised when the adapter dependencies are incomplete or inconsistent."""


class GateMemAgentStateError(GateMemAgentError):
    """Raised when reset/ingest/query lifecycle state is invalid."""


class GateMemIngestionError(GateMemAgentError):
    """Raised when a turn cannot be safely ingested."""


class GateMemQueryError(GateMemAgentError):
    """Raised when a checkpoint cannot be safely answered."""


@dataclass(frozen=True, slots=True)
class GateMemAgentConfig:
    """Adapter-level behaviour shared across all evaluation modes."""

    mode: EvaluationMode = EvaluationMode.GOVERNED_SHARED
    run_id: str = "gatemem"
    runtime_root: Path = Path("data/gatemem_runtime")

    strict_turn_order: bool = True
    strict_lifecycle_events: bool = False
    fail_closed_governance: bool = True

    expose_memory_audit: bool = True
    expose_debug: bool = True

    use_llm_governance: bool = True
    global_share_ungoverned: bool = True

    natural_language_deletion: bool = True

    summary_max_chars: int = 240

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mode",
            normalise_evaluation_mode(
                self.mode
            ),
        )
        object.__setattr__(
            self,
            "runtime_root",
            Path(self.runtime_root),
        )

        clean_run_id = str(
            self.run_id or ""
        ).strip()
        if not clean_run_id:
            raise ValueError(
                "run_id cannot be empty."
            )
        object.__setattr__(
            self,
            "run_id",
            clean_run_id,
        )

        if self.summary_max_chars <= 0:
            raise ValueError(
                "summary_max_chars must be greater than zero."
            )


@dataclass(frozen=True, slots=True)
class GateMemIngestionRecord:
    """Serializable trace of one processed GateMem turn."""

    turn_id: str
    sequence_index: int
    event_type: Literal[
        "create",
        "update",
        "delete",
        "noop",
    ]
    memory_ids: tuple[str, ...] = ()
    policy_status: str | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GateMemLifecycleOperation:
    """Normalised structured or natural-language lifecycle operation."""

    operation: Literal[
        "delete",
        "update",
    ]
    target_refs: tuple[str, ...] = ()
    replacement_content: str | None = None
    source: Literal[
        "structured",
        "natural_language",
    ] = "structured"


# ---------------------------------------------------------------------------
# Generic LLM-backed governance decision components
# ---------------------------------------------------------------------------


class GateMemLLMCoordinator:
    """
    Generic Coordinator implementation for GateMem initial-policy assignment.

    It produces decisions only. Policy persistence remains the responsibility
    of MemoryAccessPolicyGateway inside the governance workflow.
    """

    def __init__(
        self,
        *,
        llm_client: StructuredLLMProtocol,
        coordinator_agent_id: str,
        logger: logging.Logger | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.coordinator_agent_id = self._required_text(
            coordinator_agent_id,
            "coordinator_agent_id",
        )
        self.logger = logger or logging.getLogger(
            __name__
        )

    def assign_initial_policy(
        self,
        *,
        memory_id: str,
        owner_agent_id: str,
        available_agent_ids: Sequence[str],
        memory_context: Mapping[str, Any],
        policy_context: Mapping[str, Any],
        task_context: Mapping[str, Any],
    ) -> MemoryAccessDecisionOutput:
        prompt = self._build_prompt(
            task=(
                "Assign the initial read-access policy for one newly "
                "created private memory."
            ),
            rules=[
                "The memory is private and owner-only by default.",
                (
                    "Share only when the supplied relationships, access "
                    "scope, and content justify access."
                ),
                (
                    "Use targeted sharing with the minimum necessary "
                    "non-owner principal IDs whenever possible."
                ),
                (
                    "Use ['*'] only when the content is explicitly public "
                    "to every principal in this episode."
                ),
                (
                    "Do not grant access to Coordinator or Critic identities."
                ),
                (
                    "When evidence is ambiguous, keep the memory private."
                ),
                (
                    "Return memory_id exactly as supplied and use only "
                    "available principal IDs."
                ),
            ],
            context={
                "memory_id": memory_id,
                "owner_agent_id": owner_agent_id,
                "available_agent_ids": list(
                    available_agent_ids
                ),
                "memory_context": dict(
                    memory_context
                ),
                "policy_context": dict(
                    policy_context
                ),
                "task_context": dict(
                    task_context
                ),
            },
            schema_model=MemoryAccessDecisionOutput,
        )

        raw = self.llm_client.invoke_structured(
            prompt,
            MemoryAccessDecisionOutput,
        )
        return self._normalise_policy_decision(
            decision=raw,
            memory_id=memory_id,
            owner_agent_id=owner_agent_id,
            available_agent_ids=(
                available_agent_ids
            ),
            policy_context=policy_context,
        )

    def finalise_initial_policy(
        self,
        *,
        initial_decision: MemoryAccessDecisionOutput,
        critic_review: MemoryAccessReviewOutput,
        owner_agent_id: str,
        available_agent_ids: Sequence[str],
        memory_context: Mapping[str, Any],
        policy_context: Mapping[str, Any],
        task_context: Mapping[str, Any],
    ) -> MemoryAccessDecisionOutput:
        """
        Deterministically reconcile the proposal with Critic advice.

        The Coordinator retains authority, while the deterministic normaliser
        prevents a second LLM call from reintroducing unknown reader IDs.
        """
        if critic_review.recommendation == "approve":
            candidate = initial_decision.model_copy(
                update={
                    "review_required": False,
                    "reason": (
                        f"{initial_decision.reason} "
                        "Critic review approved the policy."
                    ).strip(),
                    "confidence": max(
                        initial_decision.confidence,
                        critic_review.confidence,
                    ),
                }
            )
        elif critic_review.recommendation == "revise":
            candidate = MemoryAccessDecisionOutput(
                memory_id=(
                    initial_decision.memory_id
                ),
                target_scope=(
                    critic_review.suggested_scope
                ),
                allowed_agent_ids=list(
                    critic_review.suggested_agent_ids
                ),
                review_required=False,
                reason=(
                    "Coordinator accepted the Critic's "
                    f"least-privilege revision: {critic_review.reason}"
                ),
                confidence=critic_review.confidence,
            )
        else:
            candidate = MemoryAccessDecisionOutput(
                memory_id=(
                    initial_decision.memory_id
                ),
                target_scope="private",
                allowed_agent_ids=[],
                review_required=False,
                reason=(
                    "Coordinator kept the memory private after "
                    f"Critic rejection: {critic_review.reason}"
                ),
                confidence=critic_review.confidence,
            )

        return self._normalise_policy_decision(
            decision=candidate,
            memory_id=(
                initial_decision.memory_id
            ),
            owner_agent_id=owner_agent_id,
            available_agent_ids=(
                available_agent_ids
            ),
            policy_context=policy_context,
        )

    def evaluate_access_request(
        self,
        *,
        request_id: str,
        memory_id: str,
        owner_agent_id: str,
        requester_agent_id: str,
        request_reason: str,
        task_id: str | None,
        available_agent_ids: Sequence[str],
        memory_context: Mapping[str, Any],
        policy_context: Mapping[str, Any],
        task_context: Mapping[str, Any],
    ) -> MemoryAccessRequestDecisionOutput:
        """
        Runtime queries never call this method automatically.

        When an explicit access-request workflow invokes it, the Coordinator
        uses the same least-privilege policy.
        """
        prompt = self._build_prompt(
            task=(
                "Evaluate one explicit runtime request for read access "
                "to an existing memory."
            ),
            rules=[
                "Approve only when the reason is specific and task-relevant.",
                "Reject role mismatch, vague need, or unnecessary access.",
                "Prefer a Critic review for sensitive or ambiguous content.",
                "Return request_id and memory_id exactly as supplied.",
            ],
            context={
                "request_id": request_id,
                "memory_id": memory_id,
                "owner_agent_id": owner_agent_id,
                "requester_agent_id": (
                    requester_agent_id
                ),
                "request_reason": request_reason,
                "task_id": task_id,
                "available_agent_ids": list(
                    available_agent_ids
                ),
                "memory_context": dict(
                    memory_context
                ),
                "policy_context": dict(
                    policy_context
                ),
                "task_context": dict(
                    task_context
                ),
            },
            schema_model=(
                MemoryAccessRequestDecisionOutput
            ),
        )

        raw = self.llm_client.invoke_structured(
            prompt,
            MemoryAccessRequestDecisionOutput,
        )
        payload = raw.model_dump(
            mode="python"
        )
        payload["request_id"] = request_id
        payload["memory_id"] = memory_id

        if requester_agent_id not in set(
            available_agent_ids
        ):
            payload.update(
                {
                    "approved": False,
                    "review_required": False,
                    "reason": (
                        "Requester is not a registered Agent "
                        "in the current runtime."
                    ),
                }
            )

        return (
            MemoryAccessRequestDecisionOutput
            .model_validate(payload)
        )

    def finalise_access_request(
        self,
        *,
        initial_decision: MemoryAccessRequestDecisionOutput,
        critic_review: MemoryAccessRequestReviewOutput,
        request_id: str,
        memory_id: str,
        owner_agent_id: str,
        requester_agent_id: str,
        request_reason: str,
        task_id: str | None,
        available_agent_ids: Sequence[str],
        memory_context: Mapping[str, Any],
        policy_context: Mapping[str, Any],
        task_context: Mapping[str, Any],
    ) -> MemoryAccessRequestDecisionOutput:
        approved = (
            initial_decision.approved
            and critic_review.recommendation
            == "approve"
            and critic_review.request_justified
            and critic_review.policy_compliant
            and critic_review.least_privilege_satisfied
        )

        return MemoryAccessRequestDecisionOutput(
            request_id=request_id,
            memory_id=memory_id,
            approved=approved,
            review_required=False,
            reason=(
                "Coordinator approved the request after "
                "positive Critic review."
                if approved
                else (
                    "Coordinator rejected the request after "
                    f"Critic review: {critic_review.reason}"
                )
            ),
            confidence=max(
                initial_decision.confidence,
                critic_review.confidence,
            ),
        )

    def _normalise_policy_decision(
        self,
        *,
        decision: MemoryAccessDecisionOutput,
        memory_id: str,
        owner_agent_id: str,
        available_agent_ids: Sequence[str],
        policy_context: Mapping[str, Any],
    ) -> MemoryAccessDecisionOutput:
        payload = decision.model_dump(
            mode="python"
        )
        payload["memory_id"] = memory_id

        permitted_principals = (
            self._principal_ids(
                policy_context
            )
        )
        if not permitted_principals:
            permitted_principals = {
                agent_id
                for agent_id
                in available_agent_ids
                if agent_id
                not in {
                    self.coordinator_agent_id,
                    "gatemem_critic",
                }
            }

        permitted_principals.discard(
            owner_agent_id
        )

        requested_ids = self._clean_ids(
            payload.get(
                "allowed_agent_ids",
                [],
            )
        )

        if "*" in requested_ids:
            requested_ids = ["*"]
        else:
            requested_ids = [
                agent_id
                for agent_id in requested_ids
                if agent_id
                in permitted_principals
            ]

        target_scope = str(
            payload.get(
                "target_scope",
                "private",
            )
        ).strip().lower()

        if (
            target_scope != "shared"
            or not requested_ids
        ):
            payload["target_scope"] = "private"
            payload["allowed_agent_ids"] = []
        else:
            payload["target_scope"] = "shared"
            payload["allowed_agent_ids"] = (
                requested_ids
            )

        payload["review_required"] = bool(
            payload.get(
                "review_required",
                False,
            )
        )

        return MemoryAccessDecisionOutput.model_validate(
            payload
        )

    @staticmethod
    def _principal_ids(
        policy_context: Mapping[str, Any],
    ) -> set[str]:
        result: set[str] = set()
        principals = policy_context.get(
            "principals",
            [],
        )

        if not isinstance(
            principals,
            Sequence,
        ) or isinstance(
            principals,
            (str, bytes, bytearray),
        ):
            return result

        for principal in principals:
            if not isinstance(
                principal,
                Mapping,
            ):
                continue

            value = (
                principal.get("agent_id")
                or principal.get(
                    "principal_id"
                )
            )
            cleaned = str(
                value or ""
            ).strip()
            if cleaned:
                result.add(cleaned)

        return result

    @staticmethod
    def _build_prompt(
        *,
        task: str,
        rules: Sequence[str],
        context: Mapping[str, Any],
        schema_model: type[BaseModel],
    ) -> str:
        return (
            "You are the Coordinator in a multi-agent memory-governance "
            "system.\n\n"
            f"Task:\n{task}\n\n"
            "Rules:\n"
            + "\n".join(
                f"- {rule}"
                for rule in rules
            )
            + "\n\nDecision context:\n"
            + json.dumps(
                context,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
            + "\n\nRequired structured output:\n"
            + json.dumps(
                schema_model.model_json_schema(),
                ensure_ascii=False,
                indent=2,
            )
        )

    @staticmethod
    def _clean_ids(
        values: Iterable[Any],
    ) -> list[str]:
        result: list[str] = []
        for value in values or []:
            cleaned = str(
                value or ""
            ).strip()
            if (
                cleaned
                and cleaned not in result
            ):
                result.append(cleaned)
        return result

    @staticmethod
    def _required_text(
        value: Any,
        field_name: str,
    ) -> str:
        cleaned = str(
            value or ""
        ).strip()
        if not cleaned:
            raise ValueError(
                f"{field_name} cannot be empty."
            )
        return cleaned


class GateMemLLMCritic:
    """Generic LLM-backed advisory Critic."""

    def __init__(
        self,
        *,
        llm_client: StructuredLLMProtocol,
        critic_agent_id: str,
        logger: logging.Logger | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.critic_agent_id = str(
            critic_agent_id or ""
        ).strip()
        if not self.critic_agent_id:
            raise ValueError(
                "critic_agent_id cannot be empty."
            )
        self.logger = logger or logging.getLogger(
            __name__
        )

    def review_initial_policy(
        self,
        *,
        proposed_decision: MemoryAccessDecisionOutput,
        owner_agent_id: str,
        available_agent_ids: Sequence[str],
        memory_context: Mapping[str, Any],
        policy_context: Mapping[str, Any],
        task_context: Mapping[str, Any],
    ) -> MemoryAccessReviewOutput:
        prompt = self._build_prompt(
            task=(
                "Review one proposed initial memory read-access policy."
            ),
            rules=[
                "Apply least privilege.",
                "Check relationship scope and resource-specific constraints.",
                "Reject unknown readers and unnecessary global sharing.",
                (
                    "Use revise to recommend a smaller targeted reader set "
                    "when the proposal is broader than necessary."
                ),
                "When evidence is ambiguous, recommend private access.",
                "Return memory_id exactly as supplied.",
            ],
            context={
                "proposed_decision": (
                    proposed_decision.model_dump(
                        mode="python"
                    )
                ),
                "owner_agent_id": owner_agent_id,
                "available_agent_ids": list(
                    available_agent_ids
                ),
                "memory_context": dict(
                    memory_context
                ),
                "policy_context": dict(
                    policy_context
                ),
                "task_context": dict(
                    task_context
                ),
            },
            schema_model=(
                MemoryAccessReviewOutput
            ),
        )

        raw = self.llm_client.invoke_structured(
            prompt,
            MemoryAccessReviewOutput,
        )
        payload = raw.model_dump(
            mode="python"
        )
        payload["memory_id"] = (
            proposed_decision.memory_id
        )

        allowed_ids = set(
            available_agent_ids
        )
        suggested = [
            agent_id
            for agent_id
            in self._clean_ids(
                payload.get(
                    "suggested_agent_ids",
                    [],
                )
            )
            if (
                agent_id == "*"
                or agent_id in allowed_ids
            )
        ]
        if "*" in suggested:
            suggested = ["*"]

        if (
            payload.get(
                "suggested_scope"
            )
            != "shared"
            or not suggested
        ):
            payload[
                "suggested_scope"
            ] = "private"
            payload[
                "suggested_agent_ids"
            ] = []
        else:
            payload[
                "suggested_agent_ids"
            ] = suggested

        if payload.get(
            "recommendation"
        ) == "reject":
            payload[
                "suggested_scope"
            ] = "private"
            payload[
                "suggested_agent_ids"
            ] = []

        return MemoryAccessReviewOutput.model_validate(
            payload
        )

    def review_access_request(
        self,
        *,
        proposed_decision: MemoryAccessRequestDecisionOutput,
        request_id: str,
        memory_id: str,
        owner_agent_id: str,
        requester_agent_id: str,
        request_reason: str,
        task_id: str | None,
        available_agent_ids: Sequence[str],
        memory_context: Mapping[str, Any],
        policy_context: Mapping[str, Any],
        task_context: Mapping[str, Any],
    ) -> MemoryAccessRequestReviewOutput:
        prompt = self._build_prompt(
            task=(
                "Review one proposed runtime memory-access decision."
            ),
            rules=[
                "Approve only a specific, justified, policy-compliant request.",
                "Check requester role, relationship, scope, and least privilege.",
                "Reject vague, unrelated, or excessive access.",
                "Return request_id and memory_id exactly as supplied.",
            ],
            context={
                "proposed_decision": (
                    proposed_decision.model_dump(
                        mode="python"
                    )
                ),
                "request_id": request_id,
                "memory_id": memory_id,
                "owner_agent_id": owner_agent_id,
                "requester_agent_id": (
                    requester_agent_id
                ),
                "request_reason": request_reason,
                "task_id": task_id,
                "available_agent_ids": list(
                    available_agent_ids
                ),
                "memory_context": dict(
                    memory_context
                ),
                "policy_context": dict(
                    policy_context
                ),
                "task_context": dict(
                    task_context
                ),
            },
            schema_model=(
                MemoryAccessRequestReviewOutput
            ),
        )

        raw = self.llm_client.invoke_structured(
            prompt,
            MemoryAccessRequestReviewOutput,
        )
        payload = raw.model_dump(
            mode="python"
        )
        payload["request_id"] = request_id
        payload["memory_id"] = memory_id

        return (
            MemoryAccessRequestReviewOutput
            .model_validate(payload)
        )

    @staticmethod
    def _build_prompt(
        *,
        task: str,
        rules: Sequence[str],
        context: Mapping[str, Any],
        schema_model: type[BaseModel],
    ) -> str:
        return (
            "You are the Critic in a multi-agent memory-governance "
            "system. You review but never mutate memory or ACL state.\n\n"
            f"Task:\n{task}\n\n"
            "Rules:\n"
            + "\n".join(
                f"- {rule}"
                for rule in rules
            )
            + "\n\nReview context:\n"
            + json.dumps(
                context,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
            + "\n\nRequired structured output:\n"
            + json.dumps(
                schema_model.model_json_schema(),
                ensure_ascii=False,
                indent=2,
            )
        )

    @staticmethod
    def _clean_ids(
        values: Iterable[Any],
    ) -> list[str]:
        result: list[str] = []
        for value in values or []:
            cleaned = str(
                value or ""
            ).strip()
            if (
                cleaned
                and cleaned not in result
            ):
                result.append(cleaned)
        return result


# ---------------------------------------------------------------------------
# GateMem system agent
# ---------------------------------------------------------------------------


class GateMemSystemAgent(BaseMemoryAgent):
    """
    One GateMem-visible agent wrapping the complete internal MAS.

    Subclasses set ``MODE`` for the three comparison conditions.
    """

    MODE: EvaluationMode = (
        EvaluationMode.GOVERNED_SHARED
    )

    def __init__(
        self,
        top_k: int = 8,
        llm_mode: str = "hosted",
        llm_router: Any | None = None,
        query_prompt_path: str | Path | None = None,
        logger: logging.Logger | None = None,
        answer_protocol: Any | None = None,
        *,
        mode: EvaluationMode | str | None = None,
        config: GateMemAgentConfig | None = None,
        mapper: GateMemMapper | None = None,
        runtime_factory: GateMemRuntimeFactory | None = None,
        answer_service: GateMemAnswerService | None = None,
        forgetting_service: (
                MemoryForgettingService | None
        ) = None,
        llm_client: StructuredLLMProtocol | None = None,
        coordinator_factory: CoordinatorFactory | None = None,
        critic_factory: CriticFactory | None = None,
        runtime_root: str | Path = Path(
            "data/gatemem_runtime"
        ),
        run_id: str = "gatemem",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1_200,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(
            top_k=top_k,
            llm_mode=llm_mode,
            llm_router=llm_router,
            query_prompt_path=(
                query_prompt_path
            ),
            logger=logger,
            answer_protocol=(
                answer_protocol
            ),
            **base_kwargs,
        )

        self.logger = logger or logging.getLogger(
            f"{__name__}.{type(self).__name__}"
        )

        resolved_mode = (
            normalise_evaluation_mode(
                mode
            )
            if mode is not None
            else self.MODE
        )

        self.config = config or GateMemAgentConfig(
            mode=resolved_mode,
            run_id=run_id,
            runtime_root=Path(
                runtime_root
            ),
        )

        if (
            config is not None
            and mode is not None
            and config.mode
            != resolved_mode
        ):
            raise GateMemAgentConfigurationError(
                "mode conflicts with config.mode."
            )

        self.mapper = mapper or GateMemMapper()

        if llm_client is None:
            llm_client = self._build_default_llm_client(
                llm_router=llm_router,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        self.llm_client = llm_client

        self.forgetting_service = (
                forgetting_service
                or MemoryForgettingService(
            llm_client=self.llm_client,
            logger=self.logger,
        )
        )

        if coordinator_factory is None:
            coordinator_factory = (
                self._default_coordinator_factory
            )
        if critic_factory is None:
            critic_factory = (
                self._default_critic_factory
            )

        self.runtime_factory = (
            runtime_factory
            or GateMemRuntimeFactory(
                coordinator_factory=(
                    coordinator_factory
                ),
                critic_factory=(
                    critic_factory
                ),
                runtime_root=(
                    self.config.runtime_root
                ),
                overwrite_existing_runtime=True,
            )
        )

        self.answer_service = (
            answer_service
            or GateMemAnswerService(
                llm_client=self.llm_client,
                config=GateMemAnswerConfig(
                    top_k=top_k,
                ),
            )
        )

        self._episode: MappedEpisode | None = None
        self._runtime: (
            GateMemEpisodeRuntime | None
        ) = None

        self._ingested_turn_ids: set[str] = set()
        self._turn_memory_ids: dict[
            str,
            list[str],
        ] = {}
        self._ingestion_records: list[
            GateMemIngestionRecord
        ] = []
        self._adapter_warnings: list[str] = []
        self._highest_ingested_index = -1

    # ------------------------------------------------------------------
    # GateMem lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        episode: dict[str, Any],
    ) -> None:
        super().reset(episode)

        try:
            mapped_episode = (
                self.mapper.map_episode(
                    episode
                )
            )
            runtime = (
                self.runtime_factory.build(
                    episode=mapped_episode,
                    mode=self.config.mode,
                    run_id=self.config.run_id,
                )
            )
        except Exception as error:
            raise GateMemAgentStateError(
                "Failed to reset GateMem adapter "
                f"for episode {episode.get('episode_id')!r}: "
                f"{error}"
            ) from error

        self._episode = mapped_episode
        self._runtime = runtime
        self._ingested_turn_ids.clear()
        self._turn_memory_ids.clear()
        self._ingestion_records.clear()
        self._adapter_warnings.clear()
        self._highest_ingested_index = -1

        self.logger.debug(
            "Reset GateMem runtime: %s",
            runtime.metadata(),
        )

    def ingest(
        self,
        turn: Turn,
    ) -> None:
        episode, runtime = (
            self._require_active_runtime()
        )
        mapped_turn = self._map_turn(
            turn
        )

        if (
            mapped_turn.turn_id
            in self._ingested_turn_ids
        ):
            self.logger.debug(
                "Skipping duplicate GateMem turn %s.",
                mapped_turn.turn_id,
            )
            return

        self._validate_turn_order(
            mapped_turn
        )

        try:
            lifecycle_operation = (
                self._classify_lifecycle_operation(
                    turn=turn,
                    mapped_turn=mapped_turn,
                )
            )

            if (
                lifecycle_operation is not None
            ):
                record = (
                    self._apply_lifecycle_operation(
                        turn=turn,
                        mapped_turn=mapped_turn,
                        operation=(
                            lifecycle_operation
                        ),
                        runtime=runtime,
                    )
                )
            else:
                record = self._ingest_memory_turn(
                    mapped_turn=mapped_turn,
                    turn=turn,
                    runtime=runtime,
                )
        except Exception as error:
            if self.config.strict_lifecycle_events:
                raise GateMemIngestionError(
                    "Failed to ingest turn "
                    f"{mapped_turn.turn_id!r}: {error}"
                ) from error

            warning = (
                "Turn ingestion failed and was skipped: "
                f"{mapped_turn.turn_id}: {error}"
            )
            self._adapter_warnings.append(
                warning
            )
            self.logger.warning(
                warning,
                exc_info=True,
            )
            record = GateMemIngestionRecord(
                turn_id=mapped_turn.turn_id,
                sequence_index=(
                    mapped_turn.sequence_index
                ),
                event_type="noop",
                warnings=(warning,),
            )

        self._ingested_turn_ids.add(
            mapped_turn.turn_id
        )
        self._highest_ingested_index = max(
            self._highest_ingested_index,
            mapped_turn.sequence_index,
        )
        self._ingestion_records.append(
            record
        )

    def query(
        self,
        checkpoint: Checkpoint,
    ) -> dict[str, Any]:
        episode, runtime = (
            self._require_active_runtime()
        )
        mapped_query = self._map_checkpoint(
            checkpoint
        )

        checkpoint_index = (
            episode.require_turn(
                mapped_query.as_of_turn_id
            ).sequence_index
        )
        if (
            self._highest_ingested_index
            < checkpoint_index
        ):
            raise GateMemQueryError(
                "Checkpoint was requested before all "
                "turns through as_of_turn_id were ingested: "
                f"{mapped_query.as_of_turn_id!r}."
            )

        try:
            result = self.answer_service.answer(
                runtime=runtime,
                query=mapped_query,
            )
        except Exception as error:
            raise GateMemQueryError(
                "Failed to answer checkpoint "
                f"{mapped_query.checkpoint_id!r}: {error}"
            ) from error

        payload = result.to_gatemem_payload(
            no_memory_message=(
                self.answer_service
                .config
                .no_memory_message
            ),
            refusal_message=(
                self.answer_service
                .config
                .refusal_message
            ),
        )

        if self.config.expose_memory_audit:
            payload.update(
                self._build_memory_audit(
                    runtime=runtime,
                    query=mapped_query,
                    result=result,
                )
            )

        if self.config.expose_debug:
            payload["debug"] = {
                "episode_id": (
                    episode.episode_id
                ),
                "checkpoint_id": (
                    mapped_query.checkpoint_id
                ),
                "mode": self.config.mode.value,
                "requester_agent_id": (
                    mapped_query
                    .requester_agent_id
                ),
                "as_of_turn_id": (
                    mapped_query.as_of_turn_id
                ),
                "ingested_turn_count": len(
                    self._ingested_turn_ids
                ),
                "warnings": [
                    *self._adapter_warnings,
                    *result.warnings,
                ],
            }

        return payload

    def _summary(
            self,
            text: str,
    ) -> str:
        cleaned = " ".join(
            str(text or "").split()
        )

        if (
                len(cleaned)
                <= self.config.summary_max_chars
        ):
            return cleaned

        return (
                cleaned[
                    : self.config.summary_max_chars
                ].rstrip()
                + "..."
        )

    # ------------------------------------------------------------------
    # Normal turn ingestion
    # ------------------------------------------------------------------

    def _ingest_memory_turn(
        self,
        *,
        mapped_turn: MappedTurn,
        turn: Turn,
        runtime: GateMemEpisodeRuntime,
    ) -> GateMemIngestionRecord:
        owner = runtime.require_worker(
            mapped_turn.speaker_agent_id
        )

        source_ids = [
            mapped_turn.turn_id,
            *self._turn_record_refs(turn),
        ]
        source_ids = self._clean_ids(
            source_ids
        )

        memory = (
            runtime.memory_service
            .create_private_memory(
                agent=owner,
                content=mapped_turn.text,
                summary=self._summary(
                    mapped_turn.text
                ),
                source_task_id=(
                    mapped_turn.episode_id
                ),
                source_message_ids=(
                    source_ids
                ),
                reason=(
                    "GateMem turn ingested as "
                    "owner-private memory."
                ),
            )
        )

        self._turn_memory_ids[
            mapped_turn.turn_id
        ] = [memory.memory_id]

        policy_status: str | None = None
        warnings: list[str] = []

        if (
            self.config.mode
            == EvaluationMode.PRIVATE_ONLY
        ):
            policy_status = "private"

        elif (
            self.config.mode
            == EvaluationMode.UNGOVERNED_SHARED
        ):
            policy_status = (
                self._apply_ungoverned_policy(
                    runtime=runtime,
                    memory=memory,
                )
            )

        else:
            final_state = (
                self._apply_governed_policy(
                    runtime=runtime,
                    mapped_turn=mapped_turn,
                    memory=memory,
                )
            )
            policy_status = str(
                final_state.get(
                    "status",
                    "unknown",
                )
            )

            if policy_status != "applied":
                warning = (
                    "Governed policy assignment did not "
                    "reach applied status for memory "
                    f"{memory.memory_id}: {final_state.get('error_message')}"
                )
                warnings.append(warning)

                if (
                    not self.config
                    .fail_closed_governance
                ):
                    raise GateMemIngestionError(
                        warning
                    )

        return GateMemIngestionRecord(
            turn_id=mapped_turn.turn_id,
            sequence_index=(
                mapped_turn.sequence_index
            ),
            event_type="create",
            memory_ids=(memory.memory_id,),
            policy_status=policy_status,
            warnings=tuple(warnings),
        )

    def _apply_ungoverned_policy(
        self,
        *,
        runtime: GateMemEpisodeRuntime,
        memory: Any,
    ) -> str:
        allowed_agent_ids = (
            ["*"]
            if self.config.global_share_ungoverned
            else list(runtime.workers)
        )

        result = (
            runtime.access_policy_service
            .apply_access_policy(
                governance_actor=(
                    runtime.coordinator_agent
                ),
                memory_id=memory.memory_id,
                target_scope="shared",
                allowed_agent_ids=(
                    allowed_agent_ids
                ),
                reason=(
                    "GateMem ungoverned-shared "
                    "baseline policy."
                ),
            )
        )

        scope = getattr(
            getattr(result, "metadata", None),
            "scope",
            "shared",
        )
        return str(
            getattr(scope, "value", scope)
        )

    def _apply_governed_policy(
        self,
        *,
        runtime: GateMemEpisodeRuntime,
        mapped_turn: MappedTurn,
        memory: Any,
    ) -> Mapping[str, Any]:
        policy_context = dict(
            runtime.policy_context
        )
        policy_context.update(
            {
                "policy_name": (
                    "GateMem principal-aware "
                    "least-privilege policy"
                ),
                "default_scope": "private",
                "global_sharing_requires_explicit_public_content": True,
                "rules": [
                    (
                        "A memory is private and owner-only "
                        "unless a relationship and access scope "
                        "justify sharing."
                    ),
                    (
                        "Use the smallest targeted reader set "
                        "that satisfies the active relationship."
                    ),
                    (
                        "Do not infer access from role name alone "
                        "when relationship scope is narrower."
                    ),
                    (
                        "Coordinator and Critic are governance "
                        "actors, not automatic memory readers."
                    ),
                ],
            }
        )

        memory_context = {
            "memory_id": memory.memory_id,
            "content": memory.content,
            "summary": memory.summary,
            "owner_agent_id": (
                mapped_turn.speaker_agent_id
            ),
            "owner_domain_role": (
                mapped_turn.speaker_role
            ),
            "turn_id": mapped_turn.turn_id,
            "turn_kind": (
                mapped_turn.turn_kind
            ),
            "timestamp": (
                mapped_turn.timestamp
            ),
        }

        task_context = {
            "source": "gatemem_ingestion",
            "episode_id": (
                mapped_turn.episode_id
            ),
            "domain": mapped_turn.domain,
            "turn_id": mapped_turn.turn_id,
            "sequence_index": (
                mapped_turn.sequence_index
            ),
        }

        state = (
            build_initial_policy_assignment_state(
                run_id=(
                    f"{runtime.run_id}:"
                    f"{mapped_turn.turn_id}"
                ),
                memory_id=memory.memory_id,
                owner_agent_id=(
                    mapped_turn
                    .speaker_agent_id
                ),
                governance_actor_id=(
                    runtime
                    .coordinator_agent
                    .agent_id
                ),
                available_agent_ids=(
                    runtime
                    .agent_registry
                    .agent_ids
                ),
                memory_context=(
                    memory_context
                ),
                policy_context=(
                    policy_context
                ),
                task_context=(
                    task_context
                ),
                risk_labels=(
                    self._risk_labels(
                        mapped_turn.text
                    )
                ),
            )
        )

        return (
            runtime.governance_runtime
            .assign_initial_policy(state)
        )

    # ------------------------------------------------------------------
    # Lifecycle events
    # ------------------------------------------------------------------

    def _classify_lifecycle_operation(
        self,
        *,
        turn: Turn,
        mapped_turn: MappedTurn,
    ) -> GateMemLifecycleOperation | None:
        structured = (
            self._structured_operations(
                turn
            )
        )

        for operation in structured:
            operation_name = str(
                operation.get("operation")
                or operation.get("op")
                or operation.get("action")
                or operation.get("type")
                or ""
            ).strip().lower()

            if operation_name in {
                "delete",
                "deprecate",
                "remove",
                "forget",
                "erase",
            }:
                return GateMemLifecycleOperation(
                    operation="delete",
                    target_refs=tuple(
                        self._operation_refs(
                            operation
                        )
                    ),
                    source="structured",
                )

            if operation_name in {
                "update",
                "replace",
                "correct",
                "supersede",
            }:
                replacement = self._first_text(
                    operation.get(
                        "new_content"
                    ),
                    operation.get(
                        "replacement_content"
                    ),
                    operation.get(
                        "content"
                    ),
                    operation.get(
                        "text"
                    ),
                )
                return GateMemLifecycleOperation(
                    operation="update",
                    target_refs=tuple(
                        self._operation_refs(
                            operation
                        )
                    ),
                    replacement_content=(
                        replacement
                    ),
                    source="structured",
                )

        if (
            self.config.natural_language_deletion
            and self._looks_like_deletion(
                mapped_turn.text
            )
        ):
            refs = self._clean_ids(
                [
                    *self._turn_record_refs(
                        turn
                    ),
                    *self._turn_ids_in_text(
                        mapped_turn.text
                    ),
                ]
            )
            return GateMemLifecycleOperation(
                operation="delete",
                target_refs=tuple(refs),
                source="natural_language",
            )

        return None

    def _apply_natural_language_forgetting(
            self,
            *,
            mapped_turn: MappedTurn,
            runtime: GateMemEpisodeRuntime,
    ) -> GateMemIngestionRecord:
        """
        Resolve and execute one natural-language forgetting request.

        The deletion command itself is never stored as memory.
        """

        result = self.forgetting_service.forget(
            request_text=mapped_turn.text,
            requesting_agent_id=(
                mapped_turn.speaker_agent_id
            ),
            source_turn_id=mapped_turn.turn_id,
            workers=runtime.workers,
            memory_service=runtime.memory_service,
            memory_store=runtime.memory_store,
        )

        affected = self._clean_ids(
            result.forgotten_memory_ids
        )
        warnings = self._clean_ids(
            result.warnings
        )

        if not affected:
            warning = (
                "Semantic forgetting resolved no active "
                "memory for deletion event "
                f"{mapped_turn.turn_id!r}."
            )

            if warning not in warnings:
                warnings.append(warning)

            self._adapter_warnings.extend(
                warning
                for warning in warnings
                if warning
                not in self._adapter_warnings
            )

            self._turn_memory_ids[
                mapped_turn.turn_id
            ] = []

            if self.config.strict_lifecycle_events:
                raise GateMemIngestionError(
                    warning
                )

            return GateMemIngestionRecord(
                turn_id=mapped_turn.turn_id,
                sequence_index=(
                    mapped_turn.sequence_index
                ),
                event_type="noop",
                memory_ids=(),
                policy_status=None,
                warnings=tuple(warnings),
            )

        self._turn_memory_ids[
            mapped_turn.turn_id
        ] = list(affected)

        self._adapter_warnings.extend(
            warning
            for warning in warnings
            if warning
            not in self._adapter_warnings
        )

        self.logger.info(
            "Semantic forgetting applied: "
            "turn_id=%s forgotten_memory_ids=%s "
            "target_fact=%r",
            mapped_turn.turn_id,
            affected,
            result.target_fact,
        )

        return GateMemIngestionRecord(
            turn_id=mapped_turn.turn_id,
            sequence_index=(
                mapped_turn.sequence_index
            ),
            event_type="delete",
            memory_ids=tuple(affected),
            policy_status=None,
            warnings=tuple(warnings),
        )

    def _apply_lifecycle_operation(
            self,
            *,
            turn: Turn,
            mapped_turn: MappedTurn,
            operation: GateMemLifecycleOperation,
            runtime: GateMemEpisodeRuntime,
    ) -> GateMemIngestionRecord:

        if (
                operation.operation == "delete"
                and operation.source
                == "natural_language"
        ):
            return (
                self._apply_natural_language_forgetting(
                    mapped_turn=mapped_turn,
                    runtime=runtime,
                )
            )

        target_ids = (
            self._resolve_target_memory_ids(
                turn=turn,
                mapped_turn=mapped_turn,
                operation=operation,
                runtime=runtime,
            )
        )

        if not target_ids:
            warning = (
                "Lifecycle operation had no uniquely "
                "resolvable target and raw command text "
                "was not stored: "
                f"{mapped_turn.turn_id}."
            )
            self._adapter_warnings.append(
                warning
            )

            if (
                self.config
                .strict_lifecycle_events
            ):
                raise GateMemIngestionError(
                    warning
                )

            self._turn_memory_ids[
                mapped_turn.turn_id
            ] = []
            return GateMemIngestionRecord(
                turn_id=mapped_turn.turn_id,
                sequence_index=(
                    mapped_turn.sequence_index
                ),
                event_type="noop",
                warnings=(warning,),
            )

        affected: list[str] = []
        warnings: list[str] = []

        for memory_id in target_ids:
            memory = (
                runtime.memory_store
                .get_by_id(memory_id)
            )
            owner_agent_id = str(
                getattr(
                    memory.metadata,
                    "owner_agent_id",
                    "",
                )
                or ""
            ).strip()
            owner = runtime.require_worker(
                owner_agent_id
            )

            if operation.operation == "delete":
                runtime.memory_service.deprecate_memory(
                    agent=owner,
                    memory_id=memory_id,
                    reason=(
                        "GateMem deletion event "
                        f"{mapped_turn.turn_id}; "
                        "requesting principal="
                        f"{mapped_turn.speaker_agent_id}."
                    ),
                )
            else:
                replacement = (
                    operation.replacement_content
                    or mapped_turn.text
                )
                runtime.memory_service.update_memory_content(
                    agent=owner,
                    memory_id=memory_id,
                    content=replacement,
                    reason=(
                        "GateMem update event "
                        f"{mapped_turn.turn_id}; "
                        "requesting principal="
                        f"{mapped_turn.speaker_agent_id}."
                    ),
                )

            affected.append(memory_id)

        self._turn_memory_ids[
            mapped_turn.turn_id
        ] = list(affected)

        return GateMemIngestionRecord(
            turn_id=mapped_turn.turn_id,
            sequence_index=(
                mapped_turn.sequence_index
            ),
            event_type=operation.operation,
            memory_ids=tuple(affected),
            policy_status=None,
            warnings=tuple(warnings),
        )

    def _resolve_target_memory_ids(
        self,
        *,
        turn: Turn,
        mapped_turn: MappedTurn,
        operation: GateMemLifecycleOperation,
        runtime: GateMemEpisodeRuntime,
    ) -> list[str]:
        resolved: list[str] = []

        for ref in operation.target_refs:
            if ref in self._turn_memory_ids:
                resolved.extend(
                    self._turn_memory_ids[
                        ref
                    ]
                )
                continue

            try:
                runtime.memory_store.get_by_id(
                    ref
                )
            except Exception:
                continue
            else:
                resolved.append(ref)

        return self._clean_ids(
            resolved
        )

    # ------------------------------------------------------------------
    # GateMem mapping
    # ------------------------------------------------------------------

    def _map_turn(
        self,
        turn: Turn,
    ) -> MappedTurn:
        episode, _ = (
            self._require_active_runtime()
        )

        turn_id = self._required_text(
            getattr(
                turn,
                "turn_id",
                None,
            ),
            "turn.turn_id",
        )

        canonical = episode.require_turn(
            turn_id
        )

        incoming_speaker = (
            self._required_text(
                getattr(
                    turn,
                    "speaker_principal_id",
                    None,
                ),
                (
                    "turn."
                    "speaker_principal_id"
                ),
            )
        )
        incoming_role = (
            self._required_text(
                getattr(
                    turn,
                    "speaker_role",
                    None,
                ),
                "turn.speaker_role",
            )
        )

        if (
            incoming_speaker
            != canonical.speaker_agent_id
            or incoming_role
            != canonical.speaker_role
        ):
            raise GateMemIngestionError(
                "Incoming GateMem turn identity does "
                "not match the mapped episode."
            )

        return canonical

    def _map_checkpoint(
        self,
        checkpoint: Checkpoint,
    ) -> MappedQuery:
        episode, _ = (
            self._require_active_runtime()
        )

        safe_payload = {
            "checkpoint_id": (
                self._required_text(
                    getattr(
                        checkpoint,
                        "checkpoint_id",
                        None,
                    ),
                    (
                        "checkpoint."
                        "checkpoint_id"
                    ),
                )
            ),
            "episode_id": (
                self._required_text(
                    getattr(
                        checkpoint,
                        "episode_id",
                        None,
                    ),
                    (
                        "checkpoint."
                        "episode_id"
                    ),
                )
            ),
            "as_of_turn_id": (
                self._required_text(
                    getattr(
                        checkpoint,
                        "as_of_turn_id",
                        None,
                    ),
                    (
                        "checkpoint."
                        "as_of_turn_id"
                    ),
                )
            ),
            "asker": {
                "principal_id": (
                    self._required_text(
                        getattr(
                            checkpoint,
                            (
                                "asker_"
                                "principal_id"
                            ),
                            None,
                        ),
                        (
                            "checkpoint."
                            "asker_principal_id"
                        ),
                    )
                ),
                "role": (
                    self._required_text(
                        getattr(
                            checkpoint,
                            "asker_role",
                            None,
                        ),
                        (
                            "checkpoint."
                            "asker_role"
                        ),
                    )
                ),
            },
            "query_text": (
                self._required_text(
                    getattr(
                        checkpoint,
                        "query_text",
                        None,
                    ),
                    (
                        "checkpoint."
                        "query_text"
                    ),
                )
            ),
        }

        return self.mapper.map_checkpoint(
            safe_payload,
            episode=episode,
        )

    # ------------------------------------------------------------------
    # Output audit
    # ------------------------------------------------------------------

    def _build_memory_audit(
        self,
        *,
        runtime: GateMemEpisodeRuntime,
        query: MappedQuery,
        result: GateMemAnswerResult,
    ) -> dict[str, Any]:
        requester = runtime.require_worker(
            query.requester_agent_id
        )
        retrieved: list[Any] = []
        records: list[dict[str, Any]] = []

        for memory_id in (
            result.retrieved_record_ids
        ):
            try:
                memory = (
                    runtime.memory_service
                    .get_memory(
                        requester,
                        memory_id,
                    )
                )
            except Exception as error:
                self._adapter_warnings.append(
                    "Could not reconstruct memory "
                    f"audit entry {memory_id}: {error}"
                )
                continue

            retrieved.append(memory)
            metadata = getattr(
                memory,
                "metadata",
                None,
            )
            source_ids = self._clean_ids(
                getattr(
                    metadata,
                    "source_message_ids",
                    [],
                )
            )

            records.append(
                {
                    "record_id": (
                        memory.memory_id
                    ),
                    "memory_id": (
                        memory.memory_id
                    ),
                    "principal_id": (
                        getattr(
                            metadata,
                            "owner_agent_id",
                            None,
                        )
                    ),
                    "role": (
                        runtime.principal_roles.get(
                            getattr(
                                metadata,
                                "owner_agent_id",
                                "",
                            )
                        )
                    ),
                    "turn_id": (
                        next(
                            (
                                source_id
                                for source_id
                                in source_ids
                                if source_id
                                in runtime
                                .episode
                                .turn_index()
                            ),
                            None,
                        )
                    ),
                    "record_refs": (
                        source_ids
                    ),
                    "text": str(
                        getattr(
                            memory,
                            "content",
                            "",
                        )
                        or ""
                    ),
                }
            )

        prompt_memory_block = (
            self.answer_service
            .format_memory_context(
                retrieved
            )
            if retrieved
            else (
                "No authorised accessible memory."
            )
        )

        return {
            "retrieved_memory": records,
            "prompt_memory_block": (
                prompt_memory_block
            ),
            "memory_audit": {
                "schema_version": "1.0",
                "episode_id": (
                    runtime.episode.episode_id
                ),
                "checkpoint_id": (
                    query.checkpoint_id
                ),
                "requester_principal_id": (
                    query.requester_agent_id
                ),
                "requester_role": (
                    query.requester_role
                ),
                "as_of_turn_id": (
                    query.as_of_turn_id
                ),
                "retrieved_record_ids": [
                    record["record_id"]
                    for record in records
                ],
                "used_record_ids": list(
                    result.used_record_ids
                ),
                "prompt_context": {
                    "text": (
                        prompt_memory_block
                    ),
                },
            },
        }

    # ------------------------------------------------------------------
    # Validation and helpers
    # ------------------------------------------------------------------

    def _require_active_runtime(
        self,
    ) -> tuple[
        MappedEpisode,
        GateMemEpisodeRuntime,
    ]:
        if (
            self._episode is None
            or self._runtime is None
        ):
            raise GateMemAgentStateError(
                "reset(episode) must be called "
                "before ingest() or query()."
            )

        return self._episode, self._runtime

    def _validate_turn_order(
        self,
        mapped_turn: MappedTurn,
    ) -> None:
        if not self.config.strict_turn_order:
            return

        expected_index = (
            self._highest_ingested_index + 1
        )
        if (
            mapped_turn.sequence_index
            != expected_index
        ):
            raise GateMemIngestionError(
                "GateMem turns must be ingested in "
                "chronological order: expected "
                f"sequence_index={expected_index}, "
                f"received "
                f"{mapped_turn.sequence_index}."
            )

    def _default_coordinator_factory(
        self,
        agent: Agent,
        episode: MappedEpisode,
    ) -> GateMemLLMCoordinator:
        return GateMemLLMCoordinator(
            llm_client=self.llm_client,
            coordinator_agent_id=(
                agent.agent_id
            ),
            logger=self.logger,
        )

    def _default_critic_factory(
        self,
        agent: Agent,
        episode: MappedEpisode,
    ) -> GateMemLLMCritic:
        return GateMemLLMCritic(
            llm_client=self.llm_client,
            critic_agent_id=agent.agent_id,
            logger=self.logger,
        )

    @staticmethod
    def _build_default_llm_client(
        *,
        llm_router: Any | None,
        model_name: str,
        temperature: float,
        max_tokens: int,
    ) -> StructuredLLMProtocol:
        """
        Prefer an explicitly exposed LangChain chat model from GateMem's router.

        Otherwise initialise the project's ordinary LLMClient from environment
        configuration. The custom adapter does not inspect hidden checkpoint
        annotations or use GateMem's judge model.
        """
        chat_model = getattr(
            llm_router,
            "chat_model",
            None,
        )

        if chat_model is not None:
            return LLMClient(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                validate_api_key=False,
                chat_model=chat_model,
            )

        return LLMClient(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @staticmethod
    def _structured_operations(
        turn: Turn,
    ) -> list[Mapping[str, Any]]:
        raw = getattr(
            turn,
            "memory_ops",
            None,
        )
        if raw is None:
            return []

        if isinstance(raw, Mapping):
            return [raw]

        if isinstance(
            raw,
            Sequence,
        ) and not isinstance(
            raw,
            (str, bytes, bytearray),
        ):
            return [
                item
                for item in raw
                if isinstance(
                    item,
                    Mapping,
                )
            ]

        return []

    @classmethod
    def _operation_refs(
        cls,
        operation: Mapping[str, Any],
    ) -> list[str]:
        scalar_keys = (
            "memory_id",
            "record_id",
            "target_memory_id",
            "target_record_id",
            "target_turn_id",
            "source_turn_id",
        )
        sequence_keys = (
            "memory_ids",
            "record_ids",
            "target_memory_ids",
            "target_record_ids",
            "target_turn_ids",
            "record_refs",
        )

        values: list[Any] = []
        for key in scalar_keys:
            values.append(
                operation.get(key)
            )

        for key in sequence_keys:
            raw = operation.get(key)
            if isinstance(
                raw,
                Sequence,
            ) and not isinstance(
                raw,
                (str, bytes, bytearray),
            ):
                values.extend(raw)
            elif raw is not None:
                values.append(raw)

        return cls._clean_ids(values)

    @staticmethod
    def _turn_record_refs(
        turn: Turn,
    ) -> list[str]:
        raw = getattr(
            turn,
            "record_refs",
            None,
        )
        return GateMemSystemAgent._clean_ids(
            raw or []
        )

    @staticmethod
    def _looks_like_deletion(
        text: str,
    ) -> bool:
        lowered = str(
            text or ""
        ).lower()

        patterns = (
            r"\bdeletion request\b",
            r"\bdelete\b",
            r"\bremove\b",
            r"\bforget\b",
            r"\berase\b",
            r"\bpurge\b",
            r"\bdo not retain\b",
        )
        return any(
            re.search(
                pattern,
                lowered,
            )
            for pattern in patterns
        )

    @staticmethod
    def _turn_ids_in_text(
        text: str,
    ) -> list[str]:
        return GateMemSystemAgent._clean_ids(
            re.findall(
                r"\b(?:t|turn[_-]?)[0-9]{1,8}\b",
                str(text or ""),
                flags=re.IGNORECASE,
            )
        )

    @classmethod
    def _risk_labels(
        cls,
        text: str,
    ) -> list[str]:
        lowered = str(
            text or ""
        ).lower()
        labels: list[str] = []

        label_patterns = {
            "credential_or_secret": (
                r"\bpassword\b",
                r"\btoken\b",
                r"\bcredential\b",
                r"\bsecret\b",
                r"\bapi[_ -]?key\b",
            ),
            "confidential_content": (
                r"\bconfidential\b",
                r"\bprivate\b",
                r"\brestricted\b",
            ),
            "medical_content": (
                r"\bdiagnos",
                r"\bpatient\b",
                r"\bmedication\b",
                r"\bmedical\b",
            ),
            "financial_content": (
                r"\bsalary\b",
                r"\bstipend\b",
                r"\bpayment\b",
                r"\bbank\b",
                r"\bsponsor\b",
            ),
            "education_sensitive_content": (
                r"\bgrade\b",
                r"\bdisciplin",
                r"\bacademic misconduct\b",
            ),
            "personal_identifier": (
                r"\bssn\b",
                r"\bpassport\b",
                r"\bdate of birth\b",
                r"\baddress\b",
            ),
        }

        for label, patterns in (
            label_patterns.items()
        ):
            if any(
                re.search(
                    pattern,
                    lowered,
                )
                for pattern in patterns
            ):
                labels.append(label)

        return labels


    @staticmethod
    def _first_text(
        *values: Any,
    ) -> str | None:
        for value in values:
            cleaned = str(
                value or ""
            ).strip()
            if cleaned:
                return cleaned
        return None

    @staticmethod
    def _required_text(
        value: Any,
        field_name: str,
    ) -> str:
        cleaned = str(
            value or ""
        ).strip()
        if not cleaned:
            raise ValueError(
                f"{field_name} cannot be empty."
            )
        return cleaned

    @staticmethod
    def _clean_ids(
        values: Iterable[Any],
    ) -> list[str]:
        result: list[str] = []
        for value in values or []:
            cleaned = str(
                value or ""
            ).strip()
            if (
                cleaned
                and cleaned not in result
            ):
                result.append(cleaned)
        return result

    # ------------------------------------------------------------------
    # Introspection useful for tests and experiment logs
    # ------------------------------------------------------------------

    @property
    def runtime(
        self,
    ) -> GateMemEpisodeRuntime:
        _, runtime = (
            self._require_active_runtime()
        )
        return runtime

    @property
    def mapped_episode(
        self,
    ) -> MappedEpisode:
        episode, _ = (
            self._require_active_runtime()
        )
        return episode

    @property
    def ingestion_records(
        self,
    ) -> tuple[GateMemIngestionRecord, ...]:
        return tuple(
            self._ingestion_records
        )

    @property
    def turn_memory_ids(
        self,
    ) -> Mapping[str, tuple[str, ...]]:
        return {
            turn_id: tuple(memory_ids)
            for turn_id, memory_ids
            in self._turn_memory_ids.items()
        }


class GateMemPrivateAgent(
    GateMemSystemAgent
):
    """All memories remain owner-private."""

    MODE = EvaluationMode.PRIVATE_ONLY


class GateMemUngovernedAgent(
    GateMemSystemAgent
):
    """All ordinary memories become globally shared."""

    MODE = (
        EvaluationMode.UNGOVERNED_SHARED
    )


class GateMemGovernedAgent(
    GateMemSystemAgent
):
    """Coordinator/Critic-governed sharing."""

    MODE = EvaluationMode.GOVERNED_SHARED


__all__ = [
    "StructuredLLMProtocol",
    "GateMemAgentError",
    "GateMemAgentConfigurationError",
    "GateMemAgentStateError",
    "GateMemIngestionError",
    "GateMemQueryError",
    "GateMemAgentConfig",
    "GateMemIngestionRecord",
    "GateMemLifecycleOperation",
    "GateMemLLMCoordinator",
    "GateMemLLMCritic",
    "GateMemSystemAgent",
    "GateMemPrivateAgent",
    "GateMemUngovernedAgent",
    "GateMemGovernedAgent",
]