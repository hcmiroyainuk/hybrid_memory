from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

import src.memory.governance.runtime as runtime_module
from src.memory.governance.config import (
    GovernanceConfig,
)
from src.memory.governance.review_gate import (
    GovernanceReviewGate,
)
from src.memory.governance.runtime import (
    MemoryGovernanceRuntime,
    WorkflowCompileOptions,
    build_memory_governance_runtime,
    create_memory_governance_runtime,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeCoordinator:
    def assign_initial_policy(
        self,
        **kwargs: Any,
    ) -> Any:
        return kwargs

    def finalise_initial_policy(
        self,
        **kwargs: Any,
    ) -> Any:
        return kwargs

    def evaluate_access_request(
        self,
        **kwargs: Any,
    ) -> Any:
        return kwargs

    def finalise_access_request(
        self,
        **kwargs: Any,
    ) -> Any:
        return kwargs


class FakeCritic:
    def review_initial_policy(
        self,
        **kwargs: Any,
    ) -> Any:
        return kwargs

    def review_access_request(
        self,
        **kwargs: Any,
    ) -> Any:
        return kwargs


class FakePolicyGateway:
    def apply_policy(
        self,
        **kwargs: Any,
    ) -> Any:
        return kwargs

    def grant_read_access(
        self,
        **kwargs: Any,
    ) -> Any:
        return kwargs

    def revoke_read_access(
        self,
        **kwargs: Any,
    ) -> Any:
        return kwargs

    def make_private(
        self,
        **kwargs: Any,
    ) -> Any:
        return kwargs

    def make_globally_shared(
        self,
        **kwargs: Any,
    ) -> Any:
        return kwargs


class FakeSharingGateway:
    def has_access(
        self,
        **kwargs: Any,
    ) -> bool:
        return False

    def request_access(
        self,
        **kwargs: Any,
    ) -> Any:
        return kwargs

    def review_request(
        self,
        **kwargs: Any,
    ) -> Any:
        return kwargs

    def approve_request(
        self,
        **kwargs: Any,
    ) -> Any:
        return kwargs

    def approve_direct_request(
        self,
        **kwargs: Any,
    ) -> Any:
        return kwargs

    def reject_request(
        self,
        **kwargs: Any,
    ) -> Any:
        return kwargs

    def get_request(
        self,
        **kwargs: Any,
    ) -> Any:
        return kwargs


class FakeReviewGate:
    def requires_policy_review(
        self,
        **kwargs: Any,
    ) -> Any:
        return kwargs

    def requires_access_request_review(
        self,
        **kwargs: Any,
    ) -> Any:
        return kwargs


@dataclass
class FakeCompiledWorkflow:
    dependencies: Any
    sync_result: dict[str, Any]
    async_result: dict[str, Any]
    invoke_calls: list[dict[str, Any]] = field(
        default_factory=list
    )
    ainvoke_calls: list[dict[str, Any]] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        self.nodes = SimpleNamespace(
            dependencies=self.dependencies
        )

    def invoke(
        self,
        state: dict[str, Any],
        *,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.invoke_calls.append(
            {
                "state": dict(state),
                "config": config,
            }
        )
        return dict(self.sync_result)

    async def ainvoke(
        self,
        state: dict[str, Any],
        *,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.ainvoke_calls.append(
            {
                "state": dict(state),
                "config": config,
            }
        )
        return dict(self.async_result)


@dataclass
class WorkflowBuilderSpy:
    kind: str
    calls: list[dict[str, Any]] = field(
        default_factory=list
    )
    workflows: list[
        FakeCompiledWorkflow
    ] = field(
        default_factory=list
    )

    def __call__(
        self,
        **kwargs: Any,
    ) -> FakeCompiledWorkflow:
        self.calls.append(
            dict(kwargs)
        )

        workflow = FakeCompiledWorkflow(
            dependencies=kwargs[
                "dependencies"
            ],
            sync_result={
                "workflow": self.kind,
                "mode": "sync",
            },
            async_result={
                "workflow": self.kind,
                "mode": "async",
            },
        )
        self.workflows.append(workflow)
        return workflow


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def runtime_parts() -> dict[str, Any]:
    return {
        "coordinator": FakeCoordinator(),
        "critic": FakeCritic(),
        "access_policy_gateway": (
            FakePolicyGateway()
        ),
        "sharing_gateway": (
            FakeSharingGateway()
        ),
    }


@pytest.fixture
def workflow_spies(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    WorkflowBuilderSpy,
    WorkflowBuilderSpy,
]:
    policy_spy = WorkflowBuilderSpy(
        kind="policy"
    )
    access_spy = WorkflowBuilderSpy(
        kind="access"
    )

    monkeypatch.setattr(
        runtime_module,
        "build_policy_assignment_workflow",
        policy_spy,
    )
    monkeypatch.setattr(
        runtime_module,
        "build_access_request_workflow",
        access_spy,
    )

    return policy_spy, access_spy


def _build_runtime(
    *,
    runtime_parts: dict[str, Any],
    workflow_spies: tuple[
        WorkflowBuilderSpy,
        WorkflowBuilderSpy,
    ],
    **overrides: Any,
) -> MemoryGovernanceRuntime:
    # Access the fixture so the monkeypatches are installed.
    _ = workflow_spies

    arguments = {
        **runtime_parts,
        **overrides,
    }

    return build_memory_governance_runtime(
        **arguments
    )


# ---------------------------------------------------------------------------
# WorkflowCompileOptions
# ---------------------------------------------------------------------------


def test_compile_options_create_normalises_and_deduplicates() -> None:
    checkpointer = object()

    options = WorkflowCompileOptions.create(
        checkpointer=checkpointer,
        interrupt_before=[
            " node_a ",
            "node_a",
            "node_b",
        ],
        interrupt_after=[
            "node_c",
            " node_c ",
        ],
        debug=1,
    )

    assert options.checkpointer is checkpointer
    assert options.interrupt_before == (
        "node_a",
        "node_b",
    )
    assert options.interrupt_after == (
        "node_c",
    )
    assert options.debug is True


def test_compile_options_accepts_single_string() -> None:
    options = WorkflowCompileOptions(
        interrupt_before=(
            "apply_access_policy"
        ),  # type: ignore[arg-type]
    )

    assert options.interrupt_before == (
        "apply_access_policy",
    )


@pytest.mark.parametrize(
    "field_name",
    [
        "interrupt_before",
        "interrupt_after",
    ],
)
def test_compile_options_rejects_empty_node_name(
    field_name: str,
) -> None:
    arguments = {
        field_name: (
            "valid_node",
            " ",
        )
    }

    with pytest.raises(
        ValueError,
        match="empty node name",
    ):
        WorkflowCompileOptions(
            **arguments
        )


# ---------------------------------------------------------------------------
# Runtime construction
# ---------------------------------------------------------------------------


def test_build_runtime_wires_shared_config_gate_and_dependencies(
    runtime_parts: dict[str, Any],
    workflow_spies: tuple[
        WorkflowBuilderSpy,
        WorkflowBuilderSpy,
    ],
) -> None:
    config = GovernanceConfig(
        fail_closed=False
    )
    review_gate = FakeReviewGate()

    runtime = _build_runtime(
        runtime_parts=runtime_parts,
        workflow_spies=workflow_spies,
        config=config,
        review_gate=review_gate,
    )

    assert runtime.config is config
    assert runtime.review_gate is review_gate

    policy_dependencies = (
        runtime
        .policy_assignment_dependencies
    )
    access_dependencies = (
        runtime
        .access_request_dependencies
    )

    assert (
        policy_dependencies.config
        is config
    )
    assert (
        access_dependencies.config
        is config
    )
    assert (
        policy_dependencies.review_gate
        is review_gate
    )
    assert (
        access_dependencies.review_gate
        is review_gate
    )

    assert (
        policy_dependencies.coordinator
        is runtime_parts["coordinator"]
    )
    assert (
        access_dependencies.coordinator
        is runtime_parts["coordinator"]
    )
    assert (
        policy_dependencies.critic
        is runtime_parts["critic"]
    )
    assert (
        access_dependencies.critic
        is runtime_parts["critic"]
    )

    assert (
        policy_dependencies
        .access_policy_gateway
        is runtime_parts[
            "access_policy_gateway"
        ]
    )
    assert (
        access_dependencies
        .sharing_gateway
        is runtime_parts[
            "sharing_gateway"
        ]
    )


def test_build_runtime_creates_default_config_and_review_gate(
    runtime_parts: dict[str, Any],
    workflow_spies: tuple[
        WorkflowBuilderSpy,
        WorkflowBuilderSpy,
    ],
) -> None:
    runtime = _build_runtime(
        runtime_parts=runtime_parts,
        workflow_spies=workflow_spies,
    )

    assert isinstance(
        runtime.config,
        GovernanceConfig,
    )
    assert isinstance(
        runtime.review_gate,
        GovernanceReviewGate,
    )
    assert (
        runtime.review_gate.config
        == runtime.config
    )


def test_build_runtime_forwards_independent_compile_options(
    runtime_parts: dict[str, Any],
    workflow_spies: tuple[
        WorkflowBuilderSpy,
        WorkflowBuilderSpy,
    ],
) -> None:
    policy_spy, access_spy = (
        workflow_spies
    )
    policy_checkpointer = object()
    access_checkpointer = object()

    policy_options = (
        WorkflowCompileOptions.create(
            checkpointer=(
                policy_checkpointer
            ),
            interrupt_before=[
                "apply_access_policy",
            ],
            interrupt_after=[
                "assign_initial_policy",
            ],
            debug=True,
        )
    )
    access_options = (
        WorkflowCompileOptions.create(
            checkpointer=(
                access_checkpointer
            ),
            interrupt_before=[
                "execute_access_decision",
            ],
            interrupt_after=[
                "submit_access_request",
            ],
            debug=False,
        )
    )

    runtime = _build_runtime(
        runtime_parts=runtime_parts,
        workflow_spies=workflow_spies,
        policy_assignment_compile=(
            policy_options
        ),
        access_request_compile=(
            access_options
        ),
    )

    assert (
        runtime
        .policy_assignment_workflow
        is policy_spy.workflows[0]
    )
    assert (
        runtime
        .access_request_workflow
        is access_spy.workflows[0]
    )

    policy_call = policy_spy.calls[0]
    assert (
        policy_call["checkpointer"]
        is policy_checkpointer
    )
    assert policy_call[
        "interrupt_before"
    ] == (
        "apply_access_policy",
    )
    assert policy_call[
        "interrupt_after"
    ] == (
        "assign_initial_policy",
    )
    assert policy_call["debug"] is True

    access_call = access_spy.calls[0]
    assert (
        access_call["checkpointer"]
        is access_checkpointer
    )
    assert access_call[
        "interrupt_before"
    ] == (
        "execute_access_decision",
    )
    assert access_call[
        "interrupt_after"
    ] == (
        "submit_access_request",
    )
    assert access_call["debug"] is False


def test_default_compile_options_are_forwarded_as_none(
    runtime_parts: dict[str, Any],
    workflow_spies: tuple[
        WorkflowBuilderSpy,
        WorkflowBuilderSpy,
    ],
) -> None:
    policy_spy, access_spy = (
        workflow_spies
    )

    _build_runtime(
        runtime_parts=runtime_parts,
        workflow_spies=workflow_spies,
    )

    for call in (
        policy_spy.calls[0],
        access_spy.calls[0],
    ):
        assert call["checkpointer"] is None
        assert call[
            "interrupt_before"
        ] is None
        assert call[
            "interrupt_after"
        ] is None
        assert call["debug"] is False


def test_build_runtime_rejects_review_gate_with_different_config(
    runtime_parts: dict[str, Any],
    workflow_spies: tuple[
        WorkflowBuilderSpy,
        WorkflowBuilderSpy,
    ],
) -> None:
    runtime_config = GovernanceConfig(
        fail_closed=True
    )
    gate_config = GovernanceConfig(
        fail_closed=False
    )
    review_gate = GovernanceReviewGate(
        gate_config
    )

    with pytest.raises(
        ValueError,
        match="same GovernanceConfig",
    ):
        _build_runtime(
            runtime_parts=runtime_parts,
            workflow_spies=workflow_spies,
            config=runtime_config,
            review_gate=review_gate,
        )


@pytest.mark.parametrize(
    (
        "argument_name",
        "invalid_value",
        "expected_message",
    ),
    [
        (
            "policy_assignment_compile",
            object(),
            "policy_assignment_compile",
        ),
        (
            "access_request_compile",
            object(),
            "access_request_compile",
        ),
    ],
)
def test_build_runtime_rejects_invalid_compile_options(
    runtime_parts: dict[str, Any],
    workflow_spies: tuple[
        WorkflowBuilderSpy,
        WorkflowBuilderSpy,
    ],
    argument_name: str,
    invalid_value: object,
    expected_message: str,
) -> None:
    with pytest.raises(
        TypeError,
        match=expected_message,
    ):
        _build_runtime(
            runtime_parts=runtime_parts,
            workflow_spies=workflow_spies,
            **{
                argument_name: invalid_value,
            },
        )


# ---------------------------------------------------------------------------
# Dependency interface validation
# ---------------------------------------------------------------------------


def test_build_runtime_rejects_incomplete_coordinator(
    runtime_parts: dict[str, Any],
    workflow_spies: tuple[
        WorkflowBuilderSpy,
        WorkflowBuilderSpy,
    ],
) -> None:
    class IncompleteCoordinator:
        def assign_initial_policy(
            self,
            **kwargs: Any,
        ) -> Any:
            return kwargs

    with pytest.raises(
        TypeError,
        match="coordinator.*missing",
    ):
        _build_runtime(
            runtime_parts=runtime_parts,
            workflow_spies=workflow_spies,
            coordinator=(
                IncompleteCoordinator()
            ),
        )


def test_build_runtime_rejects_incomplete_critic(
    runtime_parts: dict[str, Any],
    workflow_spies: tuple[
        WorkflowBuilderSpy,
        WorkflowBuilderSpy,
    ],
) -> None:
    class IncompleteCritic:
        def review_initial_policy(
            self,
            **kwargs: Any,
        ) -> Any:
            return kwargs

    with pytest.raises(
        TypeError,
        match="critic.*missing",
    ):
        _build_runtime(
            runtime_parts=runtime_parts,
            workflow_spies=workflow_spies,
            critic=IncompleteCritic(),
        )


def test_build_runtime_rejects_incomplete_policy_gateway(
    runtime_parts: dict[str, Any],
    workflow_spies: tuple[
        WorkflowBuilderSpy,
        WorkflowBuilderSpy,
    ],
) -> None:
    class IncompletePolicyGateway:
        def apply_policy(
            self,
            **kwargs: Any,
        ) -> Any:
            return kwargs

    with pytest.raises(
        TypeError,
        match=(
            "access_policy_gateway.*missing"
        ),
    ):
        _build_runtime(
            runtime_parts=runtime_parts,
            workflow_spies=workflow_spies,
            access_policy_gateway=(
                IncompletePolicyGateway()
            ),
        )


def test_build_runtime_rejects_sharing_gateway_without_direct_approval(
    runtime_parts: dict[str, Any],
    workflow_spies: tuple[
        WorkflowBuilderSpy,
        WorkflowBuilderSpy,
    ],
) -> None:
    gateway = FakeSharingGateway()
    gateway.approve_direct_request = (
        None  # type: ignore[method-assign]
    )

    with pytest.raises(
        TypeError,
        match="approve_direct_request",
    ):
        _build_runtime(
            runtime_parts=runtime_parts,
            workflow_spies=workflow_spies,
            sharing_gateway=gateway,
        )


# ---------------------------------------------------------------------------
# Runtime execution facade
# ---------------------------------------------------------------------------


def test_runtime_sync_methods_delegate_to_correct_workflow(
    runtime_parts: dict[str, Any],
    workflow_spies: tuple[
        WorkflowBuilderSpy,
        WorkflowBuilderSpy,
    ],
) -> None:
    runtime = _build_runtime(
        runtime_parts=runtime_parts,
        workflow_spies=workflow_spies,
    )
    invocation_config = {
        "configurable": {
            "thread_id": "thread_001",
        }
    }

    policy_result = (
        runtime.assign_initial_policy(
            {
                "memory_id": "mem_001",
            },
            config=invocation_config,
        )
    )
    access_result = (
        runtime.request_memory_access(
            {
                "request_id": "req_001",
            },
            config=invocation_config,
        )
    )

    assert policy_result == {
        "workflow": "policy",
        "mode": "sync",
    }
    assert access_result == {
        "workflow": "access",
        "mode": "sync",
    }

    policy_workflow = (
        runtime.policy_assignment_workflow
    )
    access_workflow = (
        runtime.access_request_workflow
    )

    assert policy_workflow.invoke_calls == [
        {
            "state": {
                "memory_id": "mem_001",
            },
            "config": invocation_config,
        }
    ]
    assert access_workflow.invoke_calls == [
        {
            "state": {
                "request_id": "req_001",
            },
            "config": invocation_config,
        }
    ]


def test_runtime_async_methods_delegate_to_correct_workflow(
    runtime_parts: dict[str, Any],
    workflow_spies: tuple[
        WorkflowBuilderSpy,
        WorkflowBuilderSpy,
    ],
) -> None:
    runtime = _build_runtime(
        runtime_parts=runtime_parts,
        workflow_spies=workflow_spies,
    )
    invocation_config = {
        "configurable": {
            "thread_id": "thread_002",
        }
    }

    async def run() -> tuple[
        dict[str, Any],
        dict[str, Any],
    ]:
        policy_result = await (
            runtime.aassign_initial_policy(
                {
                    "memory_id": "mem_002",
                },
                config=invocation_config,
            )
        )
        access_result = await (
            runtime.arequest_memory_access(
                {
                    "request_id": "req_002",
                },
                config=invocation_config,
            )
        )
        return (
            policy_result,
            access_result,
        )

    policy_result, access_result = (
        asyncio.run(run())
    )

    assert policy_result == {
        "workflow": "policy",
        "mode": "async",
    }
    assert access_result == {
        "workflow": "access",
        "mode": "async",
    }


# ---------------------------------------------------------------------------
# Runtime consistency checks
# ---------------------------------------------------------------------------


def test_runtime_rejects_workflow_bound_to_wrong_dependencies(
    runtime_parts: dict[str, Any],
    workflow_spies: tuple[
        WorkflowBuilderSpy,
        WorkflowBuilderSpy,
    ],
) -> None:
    runtime = _build_runtime(
        runtime_parts=runtime_parts,
        workflow_spies=workflow_spies,
    )

    wrong_policy_workflow = (
        FakeCompiledWorkflow(
            dependencies=object(),
            sync_result={},
            async_result={},
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Policy-assignment workflow"
        ),
    ):
        MemoryGovernanceRuntime(
            config=runtime.config,
            review_gate=runtime.review_gate,
            policy_assignment_dependencies=(
                runtime
                .policy_assignment_dependencies
            ),
            access_request_dependencies=(
                runtime
                .access_request_dependencies
            ),
            policy_assignment_workflow=(
                wrong_policy_workflow
            ),
            access_request_workflow=(
                runtime
                .access_request_workflow
            ),
        )


def test_runtime_rejects_dependencies_with_wrong_review_gate(
    runtime_parts: dict[str, Any],
    workflow_spies: tuple[
        WorkflowBuilderSpy,
        WorkflowBuilderSpy,
    ],
) -> None:
    runtime = _build_runtime(
        runtime_parts=runtime_parts,
        workflow_spies=workflow_spies,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Policy-assignment dependencies"
        ),
    ):
        MemoryGovernanceRuntime(
            config=runtime.config,
            review_gate=FakeReviewGate(),
            policy_assignment_dependencies=(
                runtime
                .policy_assignment_dependencies
            ),
            access_request_dependencies=(
                runtime
                .access_request_dependencies
            ),
            policy_assignment_workflow=(
                runtime
                .policy_assignment_workflow
            ),
            access_request_workflow=(
                runtime
                .access_request_workflow
            ),
        )


# ---------------------------------------------------------------------------
# Alias
# ---------------------------------------------------------------------------


def test_create_runtime_alias_forwards_all_arguments(
    monkeypatch: pytest.MonkeyPatch,
    runtime_parts: dict[str, Any],
) -> None:
    sentinel = object()
    calls: list[dict[str, Any]] = []

    def fake_builder(
        **kwargs: Any,
    ) -> Any:
        calls.append(dict(kwargs))
        return sentinel

    monkeypatch.setattr(
        runtime_module,
        "build_memory_governance_runtime",
        fake_builder,
    )

    config = GovernanceConfig()
    review_gate = FakeReviewGate()
    policy_options = (
        WorkflowCompileOptions()
    )
    access_options = (
        WorkflowCompileOptions()
    )

    result = create_memory_governance_runtime(
        **runtime_parts,
        config=config,
        review_gate=review_gate,
        policy_assignment_compile=(
            policy_options
        ),
        access_request_compile=(
            access_options
        ),
    )

    assert result is sentinel
    assert calls == [
        {
            **runtime_parts,
            "config": config,
            "review_gate": review_gate,
            "policy_assignment_compile": (
                policy_options
            ),
            "access_request_compile": (
                access_options
            ),
        }
    ]