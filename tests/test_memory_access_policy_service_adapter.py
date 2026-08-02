from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from src.adapters.agent_registry import (
    AgentRegistry,
)
from src.adapters.memory_access_policy_exceptions import (
    MemoryAccessPolicyConflictError,
    MemoryAccessPolicyGatewayError,
    MemoryAccessPolicyNotFoundError,
    MemoryAccessPolicyPermissionError,
    MemoryAccessPolicyPersistenceError,
    MemoryAccessPolicyValidationError,
)
from src.adapters.memory_access_policy_gateway import (
    MemoryAccessPolicyGateway,
)
from src.adapters.memory_access_policy_models import (
    AccessPolicyScope,
    MemoryAccessPolicyResult,
)
from src.adapters.memory_access_policy_service_adapter import (
    MemoryAccessPolicyServiceAdapter,
)
from src.memory.services.memory_access_policy_service import (
    AccessPolicyPersistenceError,
    InvalidAccessPolicyError,
    MemoryAccessPolicyError,
)
from src.memory.services.permission_service import (
    PermissionDeniedError,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class FakeMemoryMetadata:
    owner_agent_id: str
    scope: str
    readable_by: list[str]
    writable_by: list[str]
    last_operation_id: str | None = None


@dataclass
class FakeMemory:
    memory_id: str
    metadata: FakeMemoryMetadata


class FakeMemoryStore:
    def __init__(
        self,
        memories: list[FakeMemory] | None = None,
    ) -> None:
        self._memories: dict[str, FakeMemory] = {
            memory.memory_id: deepcopy(memory)
            for memory in memories or []
        }
        self.load_error: Exception | None = None
        self.get_calls: list[str] = []
        self.replace_calls: list[FakeMemory] = []

    def get_by_id(
        self,
        memory_id: str,
    ) -> FakeMemory:
        self.get_calls.append(memory_id)

        if self.load_error is not None:
            raise self.load_error

        try:
            return deepcopy(
                self._memories[memory_id]
            )
        except KeyError as error:
            raise KeyError(
                f"Memory not found: {memory_id}"
            ) from error

    def replace(
        self,
        memory: FakeMemory,
    ) -> FakeMemory:
        persisted = deepcopy(memory)
        self._memories[
            persisted.memory_id
        ] = persisted
        self.replace_calls.append(
            deepcopy(persisted)
        )
        return deepcopy(persisted)


@dataclass
class ServiceCall:
    method: str
    governance_actor: Any
    memory_id: str
    agent_ids: list[str] | None = None
    target_scope: str | None = None
    reason: str | None = None


class FakeMemoryAccessPolicyService:
    """
    Deterministic service double used to verify the adapter boundary.

    It intentionally exposes the same public methods used by the adapter,
    while keeping ACL mutation simple and observable.
    """

    def __init__(
        self,
        memory_store: FakeMemoryStore,
    ) -> None:
        self.memory_store = memory_store
        self.calls: list[ServiceCall] = []
        self.error_to_raise: Exception | None = None
        self.result_override: Any = None
        self._operation_counter = 0

    def apply_access_policy(
        self,
        *,
        governance_actor: Any,
        memory_id: str,
        target_scope: str,
        allowed_agent_ids: list[str],
        reason: str | None = None,
    ) -> Any:
        self.calls.append(
            ServiceCall(
                method="apply_access_policy",
                governance_actor=governance_actor,
                memory_id=memory_id,
                target_scope=target_scope,
                agent_ids=list(
                    allowed_agent_ids
                ),
                reason=reason,
            )
        )
        self._raise_configured_error()

        if self.result_override is not None:
            return self.result_override

        memory = self.memory_store.get_by_id(
            memory_id
        )
        owner_id = (
            memory.metadata.owner_agent_id
        )

        if target_scope == "private":
            memory.metadata.scope = "private"
            memory.metadata.readable_by = [
                owner_id
            ]
            memory.metadata.writable_by = [
                owner_id
            ]
        elif allowed_agent_ids == ["*"]:
            memory.metadata.scope = "shared"
            memory.metadata.readable_by = ["*"]
            self._ensure_owner_writer(
                memory
            )
        else:
            memory.metadata.scope = "shared"
            memory.metadata.readable_by = (
                self._owner_first(
                    owner_id,
                    allowed_agent_ids,
                )
            )
            self._ensure_owner_writer(
                memory
            )

        return self._persist_when_changed(
            memory
        )

    def grant_read_access(
        self,
        *,
        governance_actor: Any,
        memory_id: str,
        agent_ids: list[str],
        reason: str | None = None,
    ) -> Any:
        self.calls.append(
            ServiceCall(
                method="grant_read_access",
                governance_actor=governance_actor,
                memory_id=memory_id,
                agent_ids=list(agent_ids),
                reason=reason,
            )
        )
        self._raise_configured_error()

        if self.result_override is not None:
            return self.result_override

        memory = self.memory_store.get_by_id(
            memory_id
        )

        if "*" in memory.metadata.readable_by:
            return memory

        owner_id = (
            memory.metadata.owner_agent_id
        )
        non_owner_readers = [
            reader
            for reader in (
                memory.metadata.readable_by
            )
            if reader != owner_id
        ]

        memory.metadata.scope = "shared"
        memory.metadata.readable_by = (
            self._owner_first(
                owner_id,
                [
                    *non_owner_readers,
                    *agent_ids,
                ],
            )
        )
        self._ensure_owner_writer(memory)

        return self._persist_when_changed(
            memory
        )

    def revoke_read_access(
        self,
        *,
        governance_actor: Any,
        memory_id: str,
        agent_ids: list[str],
        reason: str | None = None,
    ) -> Any:
        self.calls.append(
            ServiceCall(
                method="revoke_read_access",
                governance_actor=governance_actor,
                memory_id=memory_id,
                agent_ids=list(agent_ids),
                reason=reason,
            )
        )
        self._raise_configured_error()

        if self.result_override is not None:
            return self.result_override

        memory = self.memory_store.get_by_id(
            memory_id
        )
        owner_id = (
            memory.metadata.owner_agent_id
        )
        revoked = set(agent_ids)
        remaining = [
            reader
            for reader in (
                memory.metadata.readable_by
            )
            if (
                reader != owner_id
                and reader not in revoked
            )
        ]

        if not remaining:
            memory.metadata.scope = "private"
            memory.metadata.readable_by = [
                owner_id
            ]
            memory.metadata.writable_by = [
                owner_id
            ]
        else:
            memory.metadata.scope = "shared"
            memory.metadata.readable_by = (
                self._owner_first(
                    owner_id,
                    remaining,
                )
            )
            self._ensure_owner_writer(memory)

        return self._persist_when_changed(
            memory
        )

    def make_private(
        self,
        *,
        governance_actor: Any,
        memory_id: str,
        reason: str | None = None,
    ) -> Any:
        self.calls.append(
            ServiceCall(
                method="make_private",
                governance_actor=governance_actor,
                memory_id=memory_id,
                reason=reason,
            )
        )
        self._raise_configured_error()

        if self.result_override is not None:
            return self.result_override

        memory = self.memory_store.get_by_id(
            memory_id
        )
        owner_id = (
            memory.metadata.owner_agent_id
        )
        memory.metadata.scope = "private"
        memory.metadata.readable_by = [
            owner_id
        ]
        memory.metadata.writable_by = [
            owner_id
        ]

        return self._persist_when_changed(
            memory
        )

    def make_globally_shared(
        self,
        *,
        governance_actor: Any,
        memory_id: str,
        reason: str | None = None,
    ) -> Any:
        self.calls.append(
            ServiceCall(
                method="make_globally_shared",
                governance_actor=governance_actor,
                memory_id=memory_id,
                reason=reason,
            )
        )
        self._raise_configured_error()

        if self.result_override is not None:
            return self.result_override

        memory = self.memory_store.get_by_id(
            memory_id
        )
        memory.metadata.scope = "shared"
        memory.metadata.readable_by = ["*"]
        self._ensure_owner_writer(memory)

        return self._persist_when_changed(
            memory
        )

    def _raise_configured_error(self) -> None:
        if self.error_to_raise is not None:
            raise self.error_to_raise

    def _persist_when_changed(
        self,
        memory: FakeMemory,
    ) -> FakeMemory:
        current = self.memory_store.get_by_id(
            memory.memory_id
        )

        before = (
            current.metadata.scope,
            tuple(
                current.metadata.readable_by
            ),
            tuple(
                current.metadata.writable_by
            ),
        )
        after = (
            memory.metadata.scope,
            tuple(
                memory.metadata.readable_by
            ),
            tuple(
                memory.metadata.writable_by
            ),
        )

        if before == after:
            return current

        self._operation_counter += 1
        memory.metadata.last_operation_id = (
            f"op_{self._operation_counter:03d}"
        )
        return self.memory_store.replace(
            memory
        )

    @staticmethod
    def _owner_first(
        owner_agent_id: str,
        values: list[str],
    ) -> list[str]:
        result = [owner_agent_id]

        for value in values:
            if (
                value != owner_agent_id
                and value not in result
            ):
                result.append(value)

        return result

    @staticmethod
    def _ensure_owner_writer(
        memory: FakeMemory,
    ) -> None:
        owner_id = (
            memory.metadata.owner_agent_id
        )
        writers = [
            owner_id,
            *[
                writer
                for writer in (
                    memory.metadata.writable_by
                )
                if writer != owner_id
            ],
        ]
        memory.metadata.writable_by = (
            list(dict.fromkeys(writers))
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def coordinator_agent() -> Any:
    return SimpleNamespace(
        agent_id="coordinator_agent",
    )


@pytest.fixture
def agents(
    coordinator_agent: Any,
) -> dict[str, Any]:
    return {
        "alice_agent": SimpleNamespace(
            agent_id="alice_agent",
        ),
        "bob_agent": SimpleNamespace(
            agent_id="bob_agent",
        ),
        "charlie_agent": SimpleNamespace(
            agent_id="charlie_agent",
        ),
        "coordinator_agent": (
            coordinator_agent
        ),
    }


@pytest.fixture
def private_memory() -> FakeMemory:
    return FakeMemory(
        memory_id="mem_001",
        metadata=FakeMemoryMetadata(
            owner_agent_id="alice_agent",
            scope="private",
            readable_by=[
                "alice_agent",
            ],
            writable_by=[
                "alice_agent",
            ],
            last_operation_id=None,
        ),
    )


@pytest.fixture
def adapter_bundle(
    agents: dict[str, Any],
    private_memory: FakeMemory,
) -> tuple[
    MemoryAccessPolicyServiceAdapter,
    FakeMemoryAccessPolicyService,
    FakeMemoryStore,
]:
    store = FakeMemoryStore(
        [private_memory]
    )
    service = FakeMemoryAccessPolicyService(
        store
    )
    adapter = (
        MemoryAccessPolicyServiceAdapter(
            access_policy_service=service,
            agents=agents,
        )
    )
    return adapter, service, store


# ---------------------------------------------------------------------------
# Construction and protocol conformance
# ---------------------------------------------------------------------------


def test_adapter_rejects_missing_service(
    agents: dict[str, Any],
) -> None:
    with pytest.raises(
        ValueError,
        match="cannot be None",
    ):
        MemoryAccessPolicyServiceAdapter(
            access_policy_service=None,
            agents=agents,
        )  # type: ignore[arg-type]


def test_adapter_accepts_existing_agent_registry(
    coordinator_agent: Any,
    private_memory: FakeMemory,
) -> None:
    registry = AgentRegistry(
        {
            "coordinator_agent": (
                coordinator_agent
            ),
        }
    )
    service = FakeMemoryAccessPolicyService(
        FakeMemoryStore(
            [private_memory]
        )
    )

    adapter = (
        MemoryAccessPolicyServiceAdapter(
            access_policy_service=service,
            agents=registry,
        )
    )

    assert isinstance(
        adapter,
        MemoryAccessPolicyGateway,
    )


# ---------------------------------------------------------------------------
# Gateway operation delegation
# ---------------------------------------------------------------------------


def test_apply_policy_normalises_input_and_returns_result(
    adapter_bundle: tuple[
        MemoryAccessPolicyServiceAdapter,
        FakeMemoryAccessPolicyService,
        FakeMemoryStore,
    ],
    coordinator_agent: Any,
) -> None:
    adapter, service, _ = adapter_bundle

    result = adapter.apply_policy(
        governance_actor_id=(
            " coordinator_agent "
        ),
        memory_id=" mem_001 ",
        target_scope=" SHARED ",
        allowed_agent_ids=[
            " bob_agent ",
            "",
            "bob_agent",
        ],
        reason="  Initial policy approved.  ",
    )

    assert isinstance(
        result,
        MemoryAccessPolicyResult,
    )
    assert result.memory_id == "mem_001"
    assert result.owner_agent_id == (
        "alice_agent"
    )
    assert result.scope == (
        AccessPolicyScope.SHARED
    )
    assert result.readable_by == (
        "alice_agent",
        "bob_agent",
    )
    assert result.writable_by == (
        "alice_agent",
    )
    assert result.changed is True
    assert result.operation_id == "op_001"

    call = service.calls[-1]
    assert call.method == (
        "apply_access_policy"
    )
    assert call.governance_actor is (
        coordinator_agent
    )
    assert call.memory_id == "mem_001"
    assert call.target_scope == "shared"
    assert call.agent_ids == [
        "bob_agent",
    ]
    assert call.reason == (
        "Initial policy approved."
    )


def test_reapplying_same_policy_returns_changed_false(
    adapter_bundle: tuple[
        MemoryAccessPolicyServiceAdapter,
        FakeMemoryAccessPolicyService,
        FakeMemoryStore,
    ],
) -> None:
    adapter, _, store = adapter_bundle

    result = adapter.apply_policy(
        governance_actor_id=(
            "coordinator_agent"
        ),
        memory_id="mem_001",
        target_scope="private",
        allowed_agent_ids=[],
        reason="No change.",
    )

    assert result.scope == (
        AccessPolicyScope.PRIVATE
    )
    assert result.readable_by == (
        "alice_agent",
    )
    assert result.writable_by == (
        "alice_agent",
    )
    assert result.changed is False
    assert result.operation_id is None
    assert store.replace_calls == []


def test_grant_read_access_delegates_clean_agent_ids(
    adapter_bundle: tuple[
        MemoryAccessPolicyServiceAdapter,
        FakeMemoryAccessPolicyService,
        FakeMemoryStore,
    ],
) -> None:
    adapter, service, _ = adapter_bundle

    result = adapter.grant_read_access(
        governance_actor_id=(
            "coordinator_agent"
        ),
        memory_id="mem_001",
        agent_ids=[
            " bob_agent ",
            "charlie_agent",
            "bob_agent",
            "",
        ],
        reason="  Approved request. ",
    )

    assert result.scope == (
        AccessPolicyScope.SHARED
    )
    assert result.readable_by == (
        "alice_agent",
        "bob_agent",
        "charlie_agent",
    )
    assert result.changed is True

    call = service.calls[-1]
    assert call.method == "grant_read_access"
    assert call.agent_ids == [
        "bob_agent",
        "charlie_agent",
    ]
    assert call.reason == "Approved request."


def test_revoke_last_reader_returns_private_policy(
    adapter_bundle: tuple[
        MemoryAccessPolicyServiceAdapter,
        FakeMemoryAccessPolicyService,
        FakeMemoryStore,
    ],
) -> None:
    adapter, service, _ = adapter_bundle

    adapter.grant_read_access(
        governance_actor_id=(
            "coordinator_agent"
        ),
        memory_id="mem_001",
        agent_ids=["bob_agent"],
    )

    result = adapter.revoke_read_access(
        governance_actor_id=(
            "coordinator_agent"
        ),
        memory_id="mem_001",
        agent_ids=[" bob_agent "],
        reason="Access expired.",
    )

    assert result.scope == (
        AccessPolicyScope.PRIVATE
    )
    assert result.readable_by == (
        "alice_agent",
    )
    assert result.writable_by == (
        "alice_agent",
    )
    assert result.changed is True

    call = service.calls[-1]
    assert call.method == (
        "revoke_read_access"
    )
    assert call.agent_ids == [
        "bob_agent",
    ]


def test_make_private_delegates_to_service(
    adapter_bundle: tuple[
        MemoryAccessPolicyServiceAdapter,
        FakeMemoryAccessPolicyService,
        FakeMemoryStore,
    ],
) -> None:
    adapter, service, _ = adapter_bundle

    adapter.grant_read_access(
        governance_actor_id=(
            "coordinator_agent"
        ),
        memory_id="mem_001",
        agent_ids=["bob_agent"],
    )

    result = adapter.make_private(
        governance_actor_id=(
            "coordinator_agent"
        ),
        memory_id="mem_001",
        reason="Reset to owner-only.",
    )

    assert result.scope == (
        AccessPolicyScope.PRIVATE
    )
    assert result.readable_by == (
        "alice_agent",
    )

    call = service.calls[-1]
    assert call.method == "make_private"
    assert call.reason == (
        "Reset to owner-only."
    )


def test_make_globally_shared_returns_wildcard_acl(
    adapter_bundle: tuple[
        MemoryAccessPolicyServiceAdapter,
        FakeMemoryAccessPolicyService,
        FakeMemoryStore,
    ],
) -> None:
    adapter, service, _ = adapter_bundle

    result = adapter.make_globally_shared(
        governance_actor_id=(
            "coordinator_agent"
        ),
        memory_id="mem_001",
        reason="Public system memory.",
    )

    assert result.scope == (
        AccessPolicyScope.SHARED
    )
    assert result.readable_by == ("*",)
    assert result.writable_by == (
        "alice_agent",
    )
    assert result.changed is True

    call = service.calls[-1]
    assert call.method == (
        "make_globally_shared"
    )


# ---------------------------------------------------------------------------
# Input and lookup validation
# ---------------------------------------------------------------------------


def test_unknown_governance_actor_is_translated_to_not_found(
    adapter_bundle: tuple[
        MemoryAccessPolicyServiceAdapter,
        FakeMemoryAccessPolicyService,
        FakeMemoryStore,
    ],
) -> None:
    adapter, service, _ = adapter_bundle

    with pytest.raises(
        MemoryAccessPolicyNotFoundError,
        match="unknown_agent",
    ):
        adapter.make_private(
            governance_actor_id=(
                "unknown_agent"
            ),
            memory_id="mem_001",
        )

    assert service.calls == []


@pytest.mark.parametrize(
    ("expected_pattern", "kwargs"),
    [
        (
            "agent_id",
            {
                "governance_actor_id": " ",
                "memory_id": "mem_001",
                "target_scope": "private",
            },
        ),
        (
            "memory_id",
            {
                "governance_actor_id": (
                    "coordinator_agent"
                ),
                "memory_id": " ",
                "target_scope": "private",
            },
        ),
        (
            "target_scope",
            {
                "governance_actor_id": (
                    "coordinator_agent"
                ),
                "memory_id": "mem_001",
                "target_scope": "public",
            },
        ),
    ],
)
def test_apply_policy_rejects_invalid_input(
    adapter_bundle: tuple[
        MemoryAccessPolicyServiceAdapter,
        FakeMemoryAccessPolicyService,
        FakeMemoryStore,
    ],
    expected_pattern: str,
    kwargs: dict[str, Any],
) -> None:
    adapter, service, _ = adapter_bundle

    with pytest.raises(
        MemoryAccessPolicyValidationError,
        match=expected_pattern,
    ):
        adapter.apply_policy(**kwargs)

    assert service.calls == []


@pytest.mark.parametrize(
    "method_name",
    [
        "grant_read_access",
        "revoke_read_access",
    ],
)
def test_incremental_acl_operations_require_agent_ids(
    adapter_bundle: tuple[
        MemoryAccessPolicyServiceAdapter,
        FakeMemoryAccessPolicyService,
        FakeMemoryStore,
    ],
    method_name: str,
) -> None:
    adapter, service, _ = adapter_bundle
    method = getattr(
        adapter,
        method_name,
    )

    with pytest.raises(
        MemoryAccessPolicyValidationError,
        match="at least one Agent ID",
    ):
        method(
            governance_actor_id=(
                "coordinator_agent"
            ),
            memory_id="mem_001",
            agent_ids=[
                "",
                " ",
            ],
        )

    assert service.calls == []


def test_missing_memory_is_translated_to_not_found(
    adapter_bundle: tuple[
        MemoryAccessPolicyServiceAdapter,
        FakeMemoryAccessPolicyService,
        FakeMemoryStore,
    ],
) -> None:
    adapter, service, _ = adapter_bundle

    with pytest.raises(
        MemoryAccessPolicyNotFoundError,
        match="missing",
    ):
        adapter.make_private(
            governance_actor_id=(
                "coordinator_agent"
            ),
            memory_id="missing",
        )

    assert service.calls == []


def test_memory_load_failure_is_translated_to_gateway_error(
    adapter_bundle: tuple[
        MemoryAccessPolicyServiceAdapter,
        FakeMemoryAccessPolicyService,
        FakeMemoryStore,
    ],
) -> None:
    adapter, service, store = (
        adapter_bundle
    )
    store.load_error = RuntimeError(
        "storage unavailable"
    )

    with pytest.raises(
        MemoryAccessPolicyGatewayError,
        match="Failed to load memory",
    ):
        adapter.make_private(
            governance_actor_id=(
                "coordinator_agent"
            ),
            memory_id="mem_001",
        )

    assert service.calls == []


# ---------------------------------------------------------------------------
# Exception translation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("service_error", "expected_gateway_error"),
    [
        (
            AccessPolicyPersistenceError(
                "audit logging failed"
            ),
            MemoryAccessPolicyPersistenceError,
        ),
        (
            PermissionDeniedError(
                "actor is not authorised"
            ),
            MemoryAccessPolicyPermissionError,
        ),
        (
            InvalidAccessPolicyError(
                "A shared memory requires a reader."
            ),
            MemoryAccessPolicyValidationError,
        ),
        (
            InvalidAccessPolicyError(
                "Access policy cannot be changed "
                "for deprecated memory 'mem_001'."
            ),
            MemoryAccessPolicyConflictError,
        ),
        (
            KeyError("memory disappeared"),
            MemoryAccessPolicyNotFoundError,
        ),
        (
            MemoryAccessPolicyError(
                "unexpected policy failure"
            ),
            MemoryAccessPolicyGatewayError,
        ),
    ],
)
def test_service_exceptions_are_translated(
    adapter_bundle: tuple[
        MemoryAccessPolicyServiceAdapter,
        FakeMemoryAccessPolicyService,
        FakeMemoryStore,
    ],
    service_error: Exception,
    expected_gateway_error: type[Exception],
) -> None:
    adapter, service, _ = adapter_bundle
    service.error_to_raise = service_error

    with pytest.raises(
        expected_gateway_error
    ):
        adapter.make_private(
            governance_actor_id=(
                "coordinator_agent"
            ),
            memory_id="mem_001",
        )


# ---------------------------------------------------------------------------
# Invalid service results
# ---------------------------------------------------------------------------


def test_service_result_without_metadata_is_rejected(
    adapter_bundle: tuple[
        MemoryAccessPolicyServiceAdapter,
        FakeMemoryAccessPolicyService,
        FakeMemoryStore,
    ],
) -> None:
    adapter, service, _ = adapter_bundle
    service.result_override = (
        SimpleNamespace(
            memory_id="mem_001",
            metadata=None,
        )
    )

    with pytest.raises(
        MemoryAccessPolicyGatewayError,
        match="no metadata",
    ):
        adapter.make_private(
            governance_actor_id=(
                "coordinator_agent"
            ),
            memory_id="mem_001",
        )


def test_invalid_persisted_acl_is_rejected(
    adapter_bundle: tuple[
        MemoryAccessPolicyServiceAdapter,
        FakeMemoryAccessPolicyService,
        FakeMemoryStore,
    ],
) -> None:
    adapter, service, _ = adapter_bundle
    service.result_override = FakeMemory(
        memory_id="mem_001",
        metadata=FakeMemoryMetadata(
            owner_agent_id="alice_agent",
            scope="shared",
            readable_by=[],
            writable_by=[
                "alice_agent",
            ],
            last_operation_id="op_bad",
        ),
    )

    with pytest.raises(
        MemoryAccessPolicyGatewayError,
        match="invalid persisted policy",
    ):
        adapter.make_globally_shared(
            governance_actor_id=(
                "coordinator_agent"
            ),
            memory_id="mem_001",
        )


def test_changed_false_hides_previous_operation_id(
    agents: dict[str, Any],
) -> None:
    memory = FakeMemory(
        memory_id="mem_001",
        metadata=FakeMemoryMetadata(
            owner_agent_id="alice_agent",
            scope="private",
            readable_by=[
                "alice_agent",
            ],
            writable_by=[
                "alice_agent",
            ],
            last_operation_id="op_old",
        ),
    )
    service = FakeMemoryAccessPolicyService(
        FakeMemoryStore([memory])
    )
    adapter = (
        MemoryAccessPolicyServiceAdapter(
            access_policy_service=service,
            agents=agents,
        )
    )

    result = adapter.make_private(
        governance_actor_id=(
            "coordinator_agent"
        ),
        memory_id="mem_001",
    )

    assert result.changed is False
    assert result.operation_id is None