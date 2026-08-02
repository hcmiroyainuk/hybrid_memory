from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from src.llm.output_schemas import MemoryAccessDecisionOutput
from src.memory.entities import (
    MemoryItem,
    MemoryMetadata,
    MemoryOperationType,
    MemoryScope,
    MemoryStatus,
)
from src.memory.services.memory_access_policy_service import (
    AccessPolicyPersistenceError,
    InvalidAccessPolicyError,
    MemoryAccessPolicyService,
)
from src.memory.services.memory_service import MemoryService
from src.memory.services.permission_service import (
    PermissionDeniedError,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeMemoryStore:
    """
    Minimal in-memory replacement for MemoryStore.

    It implements the operations used by MemoryService and
    MemoryAccessPolicyService.
    """

    def __init__(
        self,
        memories: list[MemoryItem] | None = None,
    ) -> None:
        self._memories: dict[str, MemoryItem] = {}
        self.create_calls = 0
        self.replace_calls = 0

        for memory in memories or []:
            self._memories[memory.memory_id] = (
                memory.model_copy(deep=True)
            )

    def create(
        self,
        memory: MemoryItem,
    ) -> MemoryItem:
        self.create_calls += 1

        if memory.memory_id in self._memories:
            raise ValueError(
                f"Duplicate memory ID: {memory.memory_id}"
            )

        stored = memory.model_copy(deep=True)
        self._memories[stored.memory_id] = stored

        return stored.model_copy(deep=True)

    def get_by_id(
        self,
        memory_id: str,
    ) -> MemoryItem:
        try:
            memory = self._memories[memory_id]
        except KeyError as error:
            raise KeyError(
                f"Memory not found: {memory_id}"
            ) from error

        return memory.model_copy(deep=True)

    def replace(
        self,
        memory: MemoryItem,
    ) -> MemoryItem:
        self.replace_calls += 1

        if memory.memory_id not in self._memories:
            raise KeyError(
                f"Memory not found: {memory.memory_id}"
            )

        stored = memory.model_copy(deep=True)
        self._memories[stored.memory_id] = stored

        return stored.model_copy(deep=True)


class FakeOperationLogStore:
    def __init__(
        self,
        *,
        fail_on_append: bool = False,
    ) -> None:
        self.fail_on_append = fail_on_append
        self.records: list[Any] = []

    def append(
        self,
        record: Any,
    ) -> Any:
        if self.fail_on_append:
            raise RuntimeError(
                "Simulated operation-log failure."
            )

        copied_record = record.model_copy(deep=True)
        self.records.append(copied_record)
        return copied_record


class FakePermissionService:
    """
    Deterministic replacement for PermissionService.

    The access-policy service currently reuses
    assert_can_approve_promotion() as its governance-authority check.
    """

    def __init__(
        self,
        *,
        allow_policy_change: bool = True,
        allow_shared_creation: bool = True,
    ) -> None:
        self.allow_policy_change = allow_policy_change
        self.allow_shared_creation = allow_shared_creation
        self.policy_checks: list[str] = []
        self.shared_creation_checks: list[str] = []

    def assert_can_approve_promotion(
        self,
        agent: Any,
    ) -> None:
        self.policy_checks.append(agent.agent_id)

        if not self.allow_policy_change:
            raise PermissionDeniedError(
                "Governance actor cannot manage access policy."
            )

    def assert_can_create_shared_memory(
        self,
        agent: Any,
    ) -> None:
        self.shared_creation_checks.append(
            agent.agent_id
        )

        if not self.allow_shared_creation:
            raise PermissionDeniedError(
                "Agent cannot create shared memory."
            )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def alice_agent() -> Any:
    return SimpleNamespace(
        agent_id="alice_agent",
    )


@pytest.fixture
def bob_agent() -> Any:
    return SimpleNamespace(
        agent_id="bob_agent",
    )


@pytest.fixture
def coordinator_agent() -> Any:
    return SimpleNamespace(
        agent_id="coordinator_agent",
    )


@pytest.fixture
def private_memory() -> MemoryItem:
    return MemoryItem(
        memory_id="mem_alice_001",
        content="Alice's project deadline is Friday.",
        summary="Alice's project deadline",
        metadata=MemoryMetadata.private(
            owner_agent_id="alice_agent",
            created_by_agent_id="alice_agent",
        ),
    )


@pytest.fixture
def policy_service(
    private_memory: MemoryItem,
) -> tuple[
    MemoryAccessPolicyService,
    FakeMemoryStore,
    FakePermissionService,
]:
    memory_store = FakeMemoryStore(
        [private_memory]
    )
    permission_service = FakePermissionService()

    service = MemoryAccessPolicyService(
        memory_store=memory_store,
        permission_service=permission_service,
        known_agent_ids={
            "alice_agent",
            "bob_agent",
            "charlie_agent",
            "coordinator_agent",
        },
    )

    return (
        service,
        memory_store,
        permission_service,
    )


# ---------------------------------------------------------------------------
# MemoryMetadata.shared()
# ---------------------------------------------------------------------------


def test_shared_metadata_supports_targeted_readers() -> None:
    metadata = MemoryMetadata.shared(
        owner_agent_id="alice_agent",
        created_by_agent_id="alice_agent",
        readable_by=[
            "bob_agent",
            "bob_agent",
        ],
    )

    assert metadata.scope == MemoryScope.SHARED
    assert metadata.readable_by == [
        "alice_agent",
        "bob_agent",
    ]
    assert metadata.writable_by == [
        "alice_agent",
    ]


def test_shared_metadata_supports_global_acl() -> None:
    metadata = MemoryMetadata.shared(
        owner_agent_id="alice_agent",
        created_by_agent_id="alice_agent",
        readable_by=["*"],
    )

    assert metadata.scope == MemoryScope.SHARED
    assert metadata.readable_by == ["*"]
    assert metadata.writable_by == [
        "alice_agent",
    ]


def test_shared_metadata_without_readers_keeps_owner_only() -> None:
    metadata = MemoryMetadata.shared(
        owner_agent_id="alice_agent",
        created_by_agent_id="alice_agent",
    )

    assert metadata.readable_by == [
        "alice_agent",
    ]


# ---------------------------------------------------------------------------
# MemoryService.create_shared_memory()
# ---------------------------------------------------------------------------


def test_memory_service_can_create_targeted_shared_memory(
    coordinator_agent: Any,
) -> None:
    memory_store = FakeMemoryStore()
    permission_service = FakePermissionService()

    service = MemoryService(
        memory_store=memory_store,
        permission_service=permission_service,
    )

    created = service.create_shared_memory(
        agent=coordinator_agent,
        content="A shared system rule.",
        readable_by=[
            "alice_agent",
            "bob_agent",
        ],
        writable_by=[
            "coordinator_agent",
        ],
    )

    assert created.metadata.scope == (
        MemoryScope.SHARED
    )
    assert created.metadata.owner_agent_id == (
        "coordinator_agent"
    )
    assert created.metadata.readable_by == [
        "coordinator_agent",
        "alice_agent",
        "bob_agent",
    ]
    assert created.metadata.writable_by == [
        "coordinator_agent",
    ]
    assert memory_store.create_calls == 1
    assert permission_service.shared_creation_checks == [
        "coordinator_agent"
    ]


def test_memory_service_shared_creation_defaults_to_global_access(
    coordinator_agent: Any,
) -> None:
    memory_store = FakeMemoryStore()

    service = MemoryService(
        memory_store=memory_store,
        permission_service=FakePermissionService(),
    )

    created = service.create_shared_memory(
        agent=coordinator_agent,
        content="A globally shared system rule.",
    )

    assert created.metadata.readable_by == ["*"]


# ---------------------------------------------------------------------------
# MemoryAccessDecisionOutput
# ---------------------------------------------------------------------------


def test_access_decision_accepts_targeted_shared_policy() -> None:
    decision = MemoryAccessDecisionOutput(
        memory_id="mem_alice_001",
        target_scope="shared",
        allowed_agent_ids=[
            " bob_agent ",
            "bob_agent",
            "charlie_agent",
        ],
        review_required=False,
        reason="Approved for selected collaborators.",
        confidence=0.9,
    )

    assert decision.allowed_agent_ids == [
        "bob_agent",
        "charlie_agent",
    ]


def test_access_decision_accepts_private_policy() -> None:
    decision = MemoryAccessDecisionOutput(
        memory_id="mem_alice_001",
        target_scope="private",
        allowed_agent_ids=[],
        review_required=True,
        reason="The information is sensitive.",
        confidence=0.6,
    )

    assert decision.target_scope == "private"
    assert decision.allowed_agent_ids == []


@pytest.mark.parametrize(
    ("target_scope", "allowed_agent_ids"),
    [
        ("private", ["bob_agent"]),
        ("shared", []),
        ("shared", ["*", "bob_agent"]),
    ],
)
def test_access_decision_rejects_invalid_policy_combinations(
    target_scope: str,
    allowed_agent_ids: list[str],
) -> None:
    with pytest.raises(ValidationError):
        MemoryAccessDecisionOutput(
            memory_id="mem_alice_001",
            target_scope=target_scope,
            allowed_agent_ids=allowed_agent_ids,
            review_required=False,
            reason="Invalid test policy.",
            confidence=0.8,
        )


# ---------------------------------------------------------------------------
# MemoryAccessPolicyService.apply_access_policy()
# ---------------------------------------------------------------------------


def test_apply_targeted_shared_policy_preserves_owner(
    policy_service: tuple[
        MemoryAccessPolicyService,
        FakeMemoryStore,
        FakePermissionService,
    ],
    coordinator_agent: Any,
) -> None:
    service, memory_store, permission = (
        policy_service
    )

    updated = service.apply_access_policy(
        governance_actor=coordinator_agent,
        memory_id="mem_alice_001",
        target_scope=MemoryScope.SHARED,
        allowed_agent_ids=[
            "bob_agent",
        ],
        reason="Bob requires authorised access.",
    )

    assert updated.metadata.owner_agent_id == (
        "alice_agent"
    )
    assert updated.metadata.scope == (
        MemoryScope.SHARED
    )
    assert updated.metadata.readable_by == [
        "alice_agent",
        "bob_agent",
    ]
    assert updated.metadata.writable_by == [
        "alice_agent",
    ]
    assert (
        "coordinator_agent"
        not in updated.metadata.readable_by
    )
    assert memory_store.replace_calls == 1
    assert permission.policy_checks == [
        "coordinator_agent"
    ]


def test_apply_private_policy_rejects_non_owner_reader(
    policy_service: tuple[
        MemoryAccessPolicyService,
        FakeMemoryStore,
        FakePermissionService,
    ],
    coordinator_agent: Any,
) -> None:
    service, memory_store, _ = policy_service

    with pytest.raises(
        InvalidAccessPolicyError,
        match="private memory",
    ):
        service.apply_access_policy(
            governance_actor=coordinator_agent,
            memory_id="mem_alice_001",
            target_scope=MemoryScope.PRIVATE,
            allowed_agent_ids=[
                "bob_agent",
            ],
        )

    stored = memory_store.get_by_id(
        "mem_alice_001"
    )
    assert stored.metadata.scope == (
        MemoryScope.PRIVATE
    )
    assert stored.metadata.readable_by == [
        "alice_agent",
    ]


def test_apply_shared_policy_rejects_missing_reader(
    policy_service: tuple[
        MemoryAccessPolicyService,
        FakeMemoryStore,
        FakePermissionService,
    ],
    coordinator_agent: Any,
) -> None:
    service, _, _ = policy_service

    with pytest.raises(
        InvalidAccessPolicyError,
        match="requires at least one",
    ):
        service.apply_access_policy(
            governance_actor=coordinator_agent,
            memory_id="mem_alice_001",
            target_scope=MemoryScope.SHARED,
            allowed_agent_ids=[],
        )


def test_apply_policy_rejects_unknown_agent(
    policy_service: tuple[
        MemoryAccessPolicyService,
        FakeMemoryStore,
        FakePermissionService,
    ],
    coordinator_agent: Any,
) -> None:
    service, _, _ = policy_service

    with pytest.raises(
        InvalidAccessPolicyError,
        match="unknown Agent IDs",
    ):
        service.apply_access_policy(
            governance_actor=coordinator_agent,
            memory_id="mem_alice_001",
            target_scope=MemoryScope.SHARED,
            allowed_agent_ids=[
                "unknown_agent",
            ],
        )


def test_apply_policy_requires_authorised_governance_actor(
    private_memory: MemoryItem,
    coordinator_agent: Any,
) -> None:
    service = MemoryAccessPolicyService(
        memory_store=FakeMemoryStore(
            [private_memory]
        ),
        permission_service=FakePermissionService(
            allow_policy_change=False
        ),
    )

    with pytest.raises(PermissionDeniedError):
        service.apply_access_policy(
            governance_actor=coordinator_agent,
            memory_id="mem_alice_001",
            target_scope=MemoryScope.SHARED,
            allowed_agent_ids=[
                "bob_agent",
            ],
        )


def test_deprecated_memory_policy_cannot_be_changed(
    private_memory: MemoryItem,
    coordinator_agent: Any,
) -> None:
    deprecated = private_memory.model_copy(
        deep=True
    )
    deprecated.metadata.status = (
        MemoryStatus.DEPRECATED
    )

    service = MemoryAccessPolicyService(
        memory_store=FakeMemoryStore(
            [deprecated]
        ),
        permission_service=FakePermissionService(),
    )

    with pytest.raises(
        InvalidAccessPolicyError,
        match="deprecated memory",
    ):
        service.apply_access_policy(
            governance_actor=coordinator_agent,
            memory_id=deprecated.memory_id,
            target_scope=MemoryScope.SHARED,
            allowed_agent_ids=[
                "bob_agent",
            ],
        )


def test_reapplying_same_policy_is_no_op(
    policy_service: tuple[
        MemoryAccessPolicyService,
        FakeMemoryStore,
        FakePermissionService,
    ],
    coordinator_agent: Any,
) -> None:
    service, memory_store, _ = policy_service

    result = service.apply_access_policy(
        governance_actor=coordinator_agent,
        memory_id="mem_alice_001",
        target_scope=MemoryScope.PRIVATE,
        allowed_agent_ids=[],
    )

    assert result.metadata.scope == (
        MemoryScope.PRIVATE
    )
    assert memory_store.replace_calls == 0


# ---------------------------------------------------------------------------
# Convenience operations
# ---------------------------------------------------------------------------


def test_grant_and_revoke_read_access(
    policy_service: tuple[
        MemoryAccessPolicyService,
        FakeMemoryStore,
        FakePermissionService,
    ],
    coordinator_agent: Any,
) -> None:
    service, _, _ = policy_service

    shared = service.grant_read_access(
        governance_actor=coordinator_agent,
        memory_id="mem_alice_001",
        agent_ids=[
            "bob_agent",
            "charlie_agent",
        ],
    )

    assert shared.metadata.readable_by == [
        "alice_agent",
        "bob_agent",
        "charlie_agent",
    ]

    remaining = service.revoke_read_access(
        governance_actor=coordinator_agent,
        memory_id="mem_alice_001",
        agent_ids=[
            "bob_agent",
        ],
    )

    assert remaining.metadata.scope == (
        MemoryScope.SHARED
    )
    assert remaining.metadata.readable_by == [
        "alice_agent",
        "charlie_agent",
    ]


def test_revoking_last_non_owner_reader_makes_memory_private(
    policy_service: tuple[
        MemoryAccessPolicyService,
        FakeMemoryStore,
        FakePermissionService,
    ],
    coordinator_agent: Any,
) -> None:
    service, _, _ = policy_service

    service.grant_read_access(
        governance_actor=coordinator_agent,
        memory_id="mem_alice_001",
        agent_ids=["bob_agent"],
    )

    private = service.revoke_read_access(
        governance_actor=coordinator_agent,
        memory_id="mem_alice_001",
        agent_ids=["bob_agent"],
    )

    assert private.metadata.scope == (
        MemoryScope.PRIVATE
    )
    assert private.metadata.readable_by == [
        "alice_agent",
    ]
    assert private.metadata.writable_by == [
        "alice_agent",
    ]


def test_owner_read_access_cannot_be_revoked(
    policy_service: tuple[
        MemoryAccessPolicyService,
        FakeMemoryStore,
        FakePermissionService,
    ],
    coordinator_agent: Any,
) -> None:
    service, _, _ = policy_service

    with pytest.raises(
        InvalidAccessPolicyError,
        match="owner cannot",
    ):
        service.revoke_read_access(
            governance_actor=coordinator_agent,
            memory_id="mem_alice_001",
            agent_ids=[
                "alice_agent",
            ],
        )


def test_specific_reader_cannot_be_revoked_from_global_acl(
    policy_service: tuple[
        MemoryAccessPolicyService,
        FakeMemoryStore,
        FakePermissionService,
    ],
    coordinator_agent: Any,
) -> None:
    service, _, _ = policy_service

    service.make_globally_shared(
        governance_actor=coordinator_agent,
        memory_id="mem_alice_001",
    )

    with pytest.raises(
        InvalidAccessPolicyError,
        match="global '\\*' ACL",
    ):
        service.revoke_read_access(
            governance_actor=coordinator_agent,
            memory_id="mem_alice_001",
            agent_ids=[
                "bob_agent",
            ],
        )


# ---------------------------------------------------------------------------
# Audit logging and rollback
# ---------------------------------------------------------------------------


def test_policy_change_creates_scope_change_log(
    private_memory: MemoryItem,
    coordinator_agent: Any,
) -> None:
    memory_store = FakeMemoryStore(
        [private_memory]
    )
    operation_log_store = (
        FakeOperationLogStore()
    )

    service = MemoryAccessPolicyService(
        memory_store=memory_store,
        permission_service=FakePermissionService(),
        operation_log_store=operation_log_store,
        known_agent_ids={
            "alice_agent",
            "bob_agent",
            "coordinator_agent",
        },
    )

    updated = service.apply_access_policy(
        governance_actor=coordinator_agent,
        memory_id="mem_alice_001",
        target_scope=MemoryScope.SHARED,
        allowed_agent_ids=[
            "bob_agent",
        ],
        reason="Approved targeted access.",
    )

    assert len(operation_log_store.records) == 1

    record = operation_log_store.records[0]

    assert record.operation_type == (
        MemoryOperationType.SCOPE_CHANGE
    )
    assert record.target_memory_ids == [
        "mem_alice_001"
    ]
    assert record.actor_agent_id == (
        "coordinator_agent"
    )
    assert record.before_state["scope"] == (
        "private"
    )
    assert record.after_state["scope"] == (
        "shared"
    )
    assert (
        updated.metadata.last_operation_id
        == record.record_id
    )
    assert (
        record.record_id
        in updated.metadata.related_operation_ids
    )


def test_logging_failure_rolls_back_memory_change(
    private_memory: MemoryItem,
    coordinator_agent: Any,
) -> None:
    memory_store = FakeMemoryStore(
        [private_memory]
    )

    service = MemoryAccessPolicyService(
        memory_store=memory_store,
        permission_service=FakePermissionService(),
        operation_log_store=FakeOperationLogStore(
            fail_on_append=True
        ),
    )

    with pytest.raises(
        AccessPolicyPersistenceError,
        match="rolled back",
    ):
        service.apply_access_policy(
            governance_actor=coordinator_agent,
            memory_id="mem_alice_001",
            target_scope=MemoryScope.SHARED,
            allowed_agent_ids=[
                "bob_agent",
            ],
        )

    stored = memory_store.get_by_id(
        "mem_alice_001"
    )

    assert stored.metadata.scope == (
        MemoryScope.PRIVATE
    )
    assert stored.metadata.readable_by == [
        "alice_agent",
    ]