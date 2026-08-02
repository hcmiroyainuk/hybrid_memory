from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any

from ..memory.entities import Agent


class AgentRegistryError(Exception):
    """Base exception raised by AgentRegistry."""


class InvalidAgentRegistryError(
    AgentRegistryError,
    ValueError,
):
    """Raised when the supplied Agent registry is invalid."""


class AgentNotRegisteredError(
    AgentRegistryError,
    LookupError,
):
    """Raised when an Agent ID is not registered."""


class AgentRegistry(Mapping[str, Agent]):
    """
    Immutable registry that resolves Agent IDs to Agent objects.

    The registry normalises and validates mapping keys once during
    construction so adapters do not need to duplicate Agent lookup logic.

    Invariants:
    - every registry key is a non-empty Agent ID;
    - every Agent exposes a non-empty ``agent_id``;
    - a registry key must match the corresponding ``Agent.agent_id``;
    - normalised Agent IDs must be unique;
    - the registry cannot be modified after construction.
    """

    def __init__(
        self,
        agents: Mapping[str, Agent],
    ) -> None:
        if agents is None:
            raise InvalidAgentRegistryError(
                "agents cannot be None."
            )

        normalised_agents: dict[str, Agent] = {}

        for mapping_id, agent in agents.items():
            clean_mapping_id = self._required_text(
                mapping_id,
                "agent mapping key",
            )

            if agent is None:
                raise InvalidAgentRegistryError(
                    "Agent registry values cannot be None. "
                    f"Invalid entry: {clean_mapping_id!r}."
                )

            actual_agent_id = self._required_text(
                getattr(agent, "agent_id", None),
                "Agent.agent_id",
            )

            if clean_mapping_id != actual_agent_id:
                raise InvalidAgentRegistryError(
                    "Agent mapping key does not match "
                    "Agent.agent_id: "
                    f"{clean_mapping_id!r} != "
                    f"{actual_agent_id!r}."
                )

            if clean_mapping_id in normalised_agents:
                raise InvalidAgentRegistryError(
                    "Duplicate normalised Agent ID: "
                    f"{clean_mapping_id!r}."
                )

            normalised_agents[
                clean_mapping_id
            ] = agent

        self._agents: Mapping[str, Agent] = (
            MappingProxyType(normalised_agents)
        )

    # ------------------------------------------------------------------
    # Mapping interface
    # ------------------------------------------------------------------

    def __getitem__(
        self,
        agent_id: str,
    ) -> Agent:
        return self.require(agent_id)

    def __iter__(self) -> Iterator[str]:
        return iter(self._agents)

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(
        self,
        agent_id: object,
    ) -> bool:
        try:
            clean_agent_id = self._required_text(
                agent_id,
                "agent_id",
            )
        except (
            InvalidAgentRegistryError,
            TypeError,
        ):
            return False

        return clean_agent_id in self._agents

    # ------------------------------------------------------------------
    # Public lookup API
    # ------------------------------------------------------------------

    def require(
        self,
        agent_id: str,
    ) -> Agent:
        """
        Return the registered Agent.

        Raises:
            AgentNotRegisteredError:
                If the Agent ID is not present.
        """
        clean_agent_id = self._required_text(
            agent_id,
            "agent_id",
        )

        try:
            return self._agents[clean_agent_id]
        except KeyError as error:
            raise AgentNotRegisteredError(
                f"Agent {clean_agent_id!r} is not registered."
            ) from error

    def get_optional(
        self,
        agent_id: str,
    ) -> Agent | None:
        """
        Return the registered Agent, or ``None`` when absent.

        Invalid or blank identifiers still raise an error because they
        represent malformed input rather than a missing registry entry.
        """
        clean_agent_id = self._required_text(
            agent_id,
            "agent_id",
        )

        return self._agents.get(clean_agent_id)

    def contains(
        self,
        agent_id: str,
    ) -> bool:
        """Return whether an Agent ID is registered."""
        clean_agent_id = self._required_text(
            agent_id,
            "agent_id",
        )

        return clean_agent_id in self._agents

    @property
    def agent_ids(self) -> tuple[str, ...]:
        """Return registered Agent IDs in insertion order."""
        return tuple(self._agents)

    @property
    def agents(self) -> tuple[Agent, ...]:
        """Return registered Agent objects in insertion order."""
        return tuple(self._agents.values())

    def as_mapping(self) -> Mapping[str, Agent]:
        """Return a read-only mapping view of the registry."""
        return self._agents

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _required_text(
        value: Any,
        field_name: str,
    ) -> str:
        if value is None:
            raise InvalidAgentRegistryError(
                f"{field_name} cannot be None."
            )

        cleaned = str(value).strip()

        if not cleaned:
            raise InvalidAgentRegistryError(
                f"{field_name} cannot be empty."
            )

        return cleaned


__all__ = [
    "AgentRegistry",
    "AgentRegistryError",
    "AgentNotRegisteredError",
    "InvalidAgentRegistryError",
]