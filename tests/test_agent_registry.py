from __future__ import annotations

from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from src.adapters.agent_registry import (
    AgentNotRegisteredError,
    AgentRegistry,
    InvalidAgentRegistryError,
)


def _agent(agent_id: str) -> Any:
    return SimpleNamespace(agent_id=agent_id)


def test_registry_resolves_agents_and_implements_mapping_interface() -> None:
    alice = _agent("alice_agent")
    bob = _agent("bob_agent")

    registry = AgentRegistry(
        {
            "alice_agent": alice,
            "bob_agent": bob,
        }
    )

    assert registry.require("alice_agent") is alice
    assert registry["bob_agent"] is bob
    assert registry.get_optional("alice_agent") is alice
    assert registry.get_optional("unknown_agent") is None

    assert registry.contains("bob_agent") is True
    assert "alice_agent" in registry
    assert "unknown_agent" not in registry

    assert len(registry) == 2
    assert tuple(registry) == (
        "alice_agent",
        "bob_agent",
    )
    assert registry.agent_ids == (
        "alice_agent",
        "bob_agent",
    )
    assert registry.agents == (
        alice,
        bob,
    )


def test_registry_normalises_lookup_identifiers() -> None:
    alice = _agent("alice_agent")

    registry = AgentRegistry(
        {
            " alice_agent ": alice,
        }
    )

    assert registry.require(" alice_agent ") is alice
    assert registry.agent_ids == ("alice_agent",)


def test_registry_returns_read_only_mapping_view() -> None:
    alice = _agent("alice_agent")
    registry = AgentRegistry(
        {
            "alice_agent": alice,
        }
    )

    mapping = registry.as_mapping()

    assert isinstance(mapping, MappingProxyType)
    assert mapping["alice_agent"] is alice

    with pytest.raises(TypeError):
        mapping["bob_agent"] = _agent("bob_agent")  # type: ignore[index]


def test_registry_rejects_none_mapping() -> None:
    with pytest.raises(
        InvalidAgentRegistryError,
        match="agents cannot be None",
    ):
        AgentRegistry(None)  # type: ignore[arg-type]


def test_registry_rejects_empty_mapping_key() -> None:
    with pytest.raises(
        InvalidAgentRegistryError,
        match="agent mapping key cannot be empty",
    ):
        AgentRegistry(
            {
                " ": _agent("alice_agent"),
            }
        )


def test_registry_rejects_none_agent_value() -> None:
    with pytest.raises(
        InvalidAgentRegistryError,
        match="values cannot be None",
    ):
        AgentRegistry(
            {
                "alice_agent": None,
            }
        )  # type: ignore[dict-item]


def test_registry_rejects_agent_without_agent_id() -> None:
    with pytest.raises(
        InvalidAgentRegistryError,
        match="Agent.agent_id cannot be None",
    ):
        AgentRegistry(
            {
                "alice_agent": SimpleNamespace(),
            }
        )


def test_registry_rejects_empty_agent_id() -> None:
    with pytest.raises(
        InvalidAgentRegistryError,
        match="Agent.agent_id cannot be empty",
    ):
        AgentRegistry(
            {
                "alice_agent": _agent(" "),
            }
        )


def test_registry_rejects_mapping_key_mismatch() -> None:
    with pytest.raises(
        InvalidAgentRegistryError,
        match="does not match",
    ):
        AgentRegistry(
            {
                "alice_agent": _agent("bob_agent"),
            }
        )


def test_registry_rejects_duplicate_normalised_ids() -> None:
    first = _agent("alice_agent")
    second = _agent("alice_agent")

    with pytest.raises(
        InvalidAgentRegistryError,
        match="Duplicate normalised Agent ID",
    ):
        AgentRegistry(
            {
                "alice_agent": first,
                " alice_agent ": second,
            }
        )


def test_require_raises_for_unknown_agent() -> None:
    registry = AgentRegistry(
        {
            "alice_agent": _agent("alice_agent"),
        }
    )

    with pytest.raises(
        AgentNotRegisteredError,
        match="unknown_agent",
    ):
        registry.require("unknown_agent")


@pytest.mark.parametrize(
    "invalid_agent_id",
    [
        "",
        "   ",
        None,
    ],
)
def test_lookup_rejects_malformed_agent_id(
    invalid_agent_id: Any,
) -> None:
    registry = AgentRegistry(
        {
            "alice_agent": _agent("alice_agent"),
        }
    )

    with pytest.raises(InvalidAgentRegistryError):
        registry.require(invalid_agent_id)

    with pytest.raises(InvalidAgentRegistryError):
        registry.get_optional(invalid_agent_id)

    with pytest.raises(InvalidAgentRegistryError):
        registry.contains(invalid_agent_id)


@pytest.mark.parametrize(
    "invalid_agent_id",
    [
        "",
        "   ",
        None,
        object(),
    ],
)
def test_contains_operator_returns_false_for_malformed_identifier(
    invalid_agent_id: Any,
) -> None:
    registry = AgentRegistry(
        {
            "alice_agent": _agent("alice_agent"),
        }
    )

    assert invalid_agent_id not in registry