from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    Any,
    Protocol,
    runtime_checkable,
)

from src.llm import (
    LLMClient,
    WorkerPromptTemplate,
)
from src.adapters import (
    MemorySharingGateway,
)
from src.memory.entities import (
    Agent,
    AgentRole,
)

from ...data_preparing.needle_persona_models import (
    AGENT_PERSONA_MAP,
)
from ...prompts.needle_persona_prompts import (
    COORDINATOR_AGENT_ID,
    CRITIC_AGENT_ID,
    NeedlePersonaPromptBundle,
)


@runtime_checkable
class NeedlePersonaMemoryServiceProtocol(Protocol):
    """
    Memory-service operations required by Needle Persona retrieval and nodes.

    The workflow depends only on this narrow contract rather than on the
    complete concrete MemoryService API.
    """

    def get_memory(
        self,
        agent: Agent,
        memory_id: str,
    ) -> Any:
        """
        Return one memory after applying the service's permission checks.
        """
        ...

    def retrieve_memories(
        self,
        agent: Agent,
        query: str,
        top_k: int = 5,
        include_private: bool = True,
        include_shared: bool = True,
    ) -> list[Any]:
        """
        Retrieve memories accessible to the supplied Agent.
        """
        ...

    def list_accessible_memories(
        self,
        agent: Agent,
        include_private: bool = True,
        include_shared: bool = True,
        active_only: bool = True,
    ) -> list[Any]:
        """
        List memories currently accessible to the supplied Agent.
        """
        ...


@dataclass(
    frozen=True,
    slots=True,
)
class NeedlePersonaNodeDependencies:
    """
    Runtime dependencies injected into Needle Persona workflow nodes.

    The dependency container keeps services and Agent objects outside
    LangGraph state. Nodes receive one validated dependency object when the
    workflow is constructed.

    Important boundaries:

    - workflow nodes use ``sharing_gateway`` rather than PromotionService;
    - workflow nodes use ``memory_service`` through a narrow Protocol;
    - prompt templates are supplied as one dataset-specific bundle;
    - Stores, PromotionRequest entities, and concrete Adapter internals are
      not exposed here.
    """

    llm_client: LLMClient

    memory_service: (
        NeedlePersonaMemoryServiceProtocol
    )

    sharing_gateway: MemorySharingGateway

    persona_agents: Mapping[str, Agent]

    critic_agent: Agent
    coordinator_agent: Agent

    prompt_bundle: NeedlePersonaPromptBundle

    def __post_init__(self) -> None:
        """
        Validate and freeze the dependency registry.
        """
        frozen_persona_agents = MappingProxyType(
            dict(self.persona_agents)
        )

        object.__setattr__(
            self,
            "persona_agents",
            frozen_persona_agents,
        )

        self._validate_memory_service()
        self._validate_sharing_gateway()
        self._validate_persona_agents()
        self._validate_governance_agents()
        self._validate_prompt_bundle()

    # ------------------------------------------------------------------
    # Public lookup helpers
    # ------------------------------------------------------------------

    def persona_agent(
        self,
        agent_id: str,
    ) -> Agent:
        """
        Return one registered persona Worker.
        """
        clean_agent_id = self._required_text(
            agent_id,
            "agent_id",
        )

        try:
            return self.persona_agents[
                clean_agent_id
            ]
        except KeyError as error:
            raise KeyError(
                "Unknown persona Agent ID: "
                f"{clean_agent_id!r}."
            ) from error

    def worker_prompt(
        self,
        agent_id: str,
    ) -> WorkerPromptTemplate:
        """
        Return the Worker prompt associated with one persona Agent.
        """
        clean_agent_id = self._required_text(
            agent_id,
            "agent_id",
        )

        # NeedlePersonaPromptBundle.worker() performs its own key validation.
        return self.prompt_bundle.worker(
            clean_agent_id
        )

    @property
    def all_agents(
        self,
    ) -> Mapping[str, Agent]:
        """
        Return a read-only registry containing Workers and governance Agents.
        """
        combined = dict(self.persona_agents)
        combined[
            self.critic_agent.agent_id
        ] = self.critic_agent
        combined[
            self.coordinator_agent.agent_id
        ] = self.coordinator_agent

        return MappingProxyType(combined)

    @property
    def persona_agent_ids(
        self,
    ) -> tuple[str, ...]:
        """
        Return persona Agent IDs in the benchmark's canonical order.
        """
        return tuple(
            AGENT_PERSONA_MAP
        )

    # ------------------------------------------------------------------
    # Dependency validation
    # ------------------------------------------------------------------

    def _validate_memory_service(
        self,
    ) -> None:
        if not isinstance(
            self.memory_service,
            NeedlePersonaMemoryServiceProtocol,
        ):
            raise TypeError(
                "memory_service must implement "
                "NeedlePersonaMemoryServiceProtocol."
            )

    def _validate_sharing_gateway(
        self,
    ) -> None:
        if not isinstance(
            self.sharing_gateway,
            MemorySharingGateway,
        ):
            raise TypeError(
                "sharing_gateway must implement "
                "MemorySharingGateway."
            )

    def _validate_persona_agents(
        self,
    ) -> None:
        expected_agent_ids = set(
            AGENT_PERSONA_MAP
        )
        actual_agent_ids = set(
            self.persona_agents
        )

        missing_agent_ids = (
            expected_agent_ids
            - actual_agent_ids
        )
        extra_agent_ids = (
            actual_agent_ids
            - expected_agent_ids
        )

        if missing_agent_ids:
            raise ValueError(
                "Missing persona Agents: "
                f"{sorted(missing_agent_ids)}."
            )

        if extra_agent_ids:
            raise ValueError(
                "persona_agents contains unsupported "
                "Agent IDs: "
                f"{sorted(extra_agent_ids)}."
            )

        for agent_id in AGENT_PERSONA_MAP:
            agent = self.persona_agents[
                agent_id
            ]

            if agent.agent_id != agent_id:
                raise ValueError(
                    "Persona Agent mapping key does not "
                    "match Agent.agent_id: "
                    f"{agent_id!r} != "
                    f"{agent.agent_id!r}."
                )

            if agent.role != AgentRole.WORKER:
                raise ValueError(
                    f"{agent_id!r} must have "
                    "AgentRole.WORKER."
                )

    def _validate_governance_agents(
        self,
    ) -> None:
        if (
            self.critic_agent.agent_id
            != CRITIC_AGENT_ID
        ):
            raise ValueError(
                "critic_agent.agent_id must match "
                f"{CRITIC_AGENT_ID!r}."
            )

        if (
            self.critic_agent.role
            != AgentRole.CRITIC
        ):
            raise ValueError(
                "critic_agent must have "
                "AgentRole.CRITIC."
            )

        if (
            self.coordinator_agent.agent_id
            != COORDINATOR_AGENT_ID
        ):
            raise ValueError(
                "coordinator_agent.agent_id must match "
                f"{COORDINATOR_AGENT_ID!r}."
            )

        if (
            self.coordinator_agent.role
            != AgentRole.COORDINATOR
        ):
            raise ValueError(
                "coordinator_agent must have "
                "AgentRole.COORDINATOR."
            )

        all_ids = [
            *self.persona_agents,
            self.critic_agent.agent_id,
            self.coordinator_agent.agent_id,
        ]

        if len(all_ids) != len(set(all_ids)):
            raise ValueError(
                "Worker, Critic, and Coordinator Agent "
                "IDs must be unique."
            )

    def _validate_prompt_bundle(
        self,
    ) -> None:
        expected_agent_ids = set(
            AGENT_PERSONA_MAP
        )
        prompt_agent_ids = set(
            self.prompt_bundle.worker_prompts
        )

        missing_prompt_ids = (
            expected_agent_ids
            - prompt_agent_ids
        )
        extra_prompt_ids = (
            prompt_agent_ids
            - expected_agent_ids
        )

        if missing_prompt_ids:
            raise ValueError(
                "Missing Worker prompt templates: "
                f"{sorted(missing_prompt_ids)}."
            )

        if extra_prompt_ids:
            raise ValueError(
                "Prompt bundle contains unsupported "
                "Worker prompt IDs: "
                f"{sorted(extra_prompt_ids)}."
            )

        for agent_id in AGENT_PERSONA_MAP:
            template = (
                self.prompt_bundle.worker_prompts[
                    agent_id
                ]
            )

            if template.agent_id != agent_id:
                raise ValueError(
                    "Worker prompt key does not match "
                    "template.agent_id: "
                    f"{agent_id!r} != "
                    f"{template.agent_id!r}."
                )

        if (
            self.prompt_bundle.critic_prompt.critic_id
            != self.critic_agent.agent_id
        ):
            raise ValueError(
                "Critic prompt ID does not match "
                "critic_agent.agent_id."
            )

        if (
            self.prompt_bundle
            .coordinator_prompt
            .coordinator_id
            != self.coordinator_agent.agent_id
        ):
            raise ValueError(
                "Coordinator prompt ID does not match "
                "coordinator_agent.agent_id."
            )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

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


__all__ = [
    "NeedlePersonaMemoryServiceProtocol",
    "NeedlePersonaNodeDependencies",
]