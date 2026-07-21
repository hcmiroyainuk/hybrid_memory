from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import pytest

from experiments.evaluator.informativebench import (
    AGENT_PERSONA_MAP,
    NeedlePersonaNodeError,
    NeedlePersonaSample,
    NeedlePersonaWorkflow,
    NeedlePersonaWorkflowConfig,
    NeedlePersonaWorkflowError,
    NeedlePersonaWorkflowState,
    PersonaSource,
    PrivateMemoryIndex,
    build_initial_needle_persona_state,
    build_needle_persona_workflow,
)
from src.llm import (
    AgentAnswer,
    MemoryReviewOutput,
    PromotionDecisionOutput,
    TaskRoutingOutput,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample() -> NeedlePersonaSample:
    """
    A two-person task in which Alice is the responder and Dave contributes
    information through memory sharing.
    """
    return NeedlePersonaSample(
        sample_id="sample_001",
        question=(
            "What fashion icons inspire Alice and Dave's "
            "unique styles?"
        ),
        gold_answer="Alexander McQueen and Coco Chanel",
        alternative_answers=[
            "Coco Chanel and Alexander McQueen",
        ],
        hop=2,
        persona_sources={
            "alice": [
                PersonaSource(
                    source_id="source_alice_001",
                    persona="alice",
                    content=(
                        "Alice is inspired by Alexander "
                        "McQueen."
                    ),
                    source_type="persona",
                )
            ],
            "bob": [
                PersonaSource(
                    source_id="source_bob_001",
                    persona="bob",
                    content="Bob likes jazz.",
                    source_type="persona",
                )
            ],
            "charlie": [
                PersonaSource(
                    source_id="source_charlie_001",
                    persona="charlie",
                    content="Charlie likes hiking.",
                    source_type="persona",
                )
            ],
            "dave": [
                PersonaSource(
                    source_id="source_dave_001",
                    persona="dave",
                    content=(
                        "Dave is inspired by Coco Chanel."
                    ),
                    source_type="persona",
                )
            ],
        },
    )


@pytest.fixture
def memory_index() -> PrivateMemoryIndex:
    return PrivateMemoryIndex(
        run_id="run_001",
        sample_id="sample_001",
        private_memory_ids={
            "alice_agent": ["mem_alice_001"],
            "bob_agent": ["mem_bob_001"],
            "charlie_agent": ["mem_charlie_001"],
            "dave_agent": ["mem_dave_001"],
        },
        source_to_memory_ids={
            "source_alice_001": ["mem_alice_001"],
            "source_bob_001": ["mem_bob_001"],
            "source_charlie_001": [
                "mem_charlie_001"
            ],
            "source_dave_001": ["mem_dave_001"],
        },
    )


def build_initial_state(
    *,
    sample: NeedlePersonaSample,
    memory_index: PrivateMemoryIndex,
    mode: str,
) -> NeedlePersonaWorkflowState:
    return build_initial_needle_persona_state(
        sample=sample,
        memory_index=memory_index,
        experiment_mode=cast(Any, mode),
    )


# ---------------------------------------------------------------------------
# Fake nodes
# ---------------------------------------------------------------------------


class FakeNodes:
    """
    Deterministic node implementation used to test workflow orchestration.

    It contains no LLM calls and no memory-service calls. Every method records
    its name in ``calls`` and returns a state update compatible with the real
    NeedlePersonaNodes methods.
    """

    def __init__(
        self,
        *,
        fail_node: str | None = None,
        hallucinated_used_memory_id: bool = False,
        private_mode_returns_shared_memory: bool = False,
        invalid_responder: bool = False,
        omit_answer: bool = False,
        final_status: str = "answered",
    ) -> None:
        self.calls: list[str] = []
        self.fail_node = fail_node
        self.hallucinated_used_memory_id = (
            hallucinated_used_memory_id
        )
        self.private_mode_returns_shared_memory = (
            private_mode_returns_shared_memory
        )
        self.invalid_responder = invalid_responder
        self.omit_answer = omit_answer
        self.final_status = final_status

    def _record(self, node_name: str) -> None:
        self.calls.append(node_name)

        if self.fail_node == node_name:
            raise NeedlePersonaNodeError(
                node_name,
                "Simulated node failure.",
            )

    def route_task(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> dict[str, Any]:
        self._record("route_task")

        responder = (
            "unknown_agent"
            if self.invalid_responder
            else "alice_agent"
        )

        routing = TaskRoutingOutput(
            selected_agent_ids=[
                "alice_agent",
                "dave_agent",
            ],
            # TaskRoutingOutput itself requires the responder to be selected.
            # For the invalid-responder test, keep the structured object valid
            # but return a conflicting state-level responder below.
            responder_agent_id="alice_agent",
            required_information=[
                "Alice's fashion influence.",
                "Dave's fashion influence.",
            ],
            reason=(
                "Alice and Dave are explicitly named in "
                "the question."
            ),
            confidence=1.0,
        )

        return {
            "routing_output": routing,
            "selected_agent_ids": [
                "alice_agent",
                "dave_agent",
            ],
            "responder_agent_id": responder,
            "contributor_agent_ids": [
                "dave_agent",
            ],
            "workflow_status": "routed",
            "node_trace": ["route_task"],
            "node_metrics": {
                "route_task_ms": 1.0,
            },
        }

    def collect_candidate_memories(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> dict[str, Any]:
        self._record("collect_candidate_memories")

        return {
            "contributor_retrieved_memory_ids": {
                "dave_agent": ["mem_dave_001"],
            },
            "candidate_memory_ids": [
                "mem_dave_001",
            ],
            "candidate_owner_by_memory_id": {
                "mem_dave_001": "dave_agent",
            },
            "workflow_status": (
                "candidates_collected"
            ),
            "node_trace": [
                "collect_candidate_memories"
            ],
            "node_metrics": {
                "collect_candidate_memories_ms": 2.0,
            },
        }

    def direct_share(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> dict[str, Any]:
        self._record("direct_share")

        return {
            "shared_memory_ids": [
                "mem_dave_001",
            ],
            "workflow_status": "shared",
            "node_trace": ["direct_share"],
            "warnings": [
                "Ungoverned sharing was used.",
            ],
            "node_metrics": {
                "direct_share_ms": 3.0,
            },
        }

    def submit_promotion_requests(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> dict[str, Any]:
        self._record("submit_promotion_requests")

        return {
            "promotion_request_ids": {
                "mem_dave_001": "request_001",
            },
            "workflow_status": (
                "promotion_submitted"
            ),
            "node_trace": [
                "submit_promotion_requests"
            ],
            "node_metrics": {
                "submit_promotion_requests_ms": 3.0,
            },
        }

    def critic_review(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> dict[str, Any]:
        self._record("critic_review")

        review = MemoryReviewOutput(
            memory_id="mem_dave_001",
            classification="new",
            recommendation="approve",
            relevant=True,
            policy_compliant=True,
            related_memory_ids=[],
            reason=(
                "Dave's preference is relevant and safe "
                "to share with Alice for this task."
            ),
            confidence=0.95,
        )

        return {
            "critic_reviews": {
                "mem_dave_001": review,
            },
            "workflow_status": "reviewed",
            "node_trace": ["critic_review"],
            "node_metrics": {
                "critic_review_ms": 4.0,
            },
        }

    def coordinator_decide(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> dict[str, Any]:
        self._record("coordinator_decide")

        decision = PromotionDecisionOutput(
            memory_id="mem_dave_001",
            decision="approve",
            target_scope="shared",
            allowed_agent_ids=[
                "dave_agent",
                "alice_agent",
            ],
            related_memory_ids=[],
            reason=(
                "Approve task-scoped sharing with the "
                "source owner and responder."
            ),
            confidence=0.95,
        )

        return {
            "promotion_decisions": {
                "mem_dave_001": decision,
            },
            "workflow_status": "decided",
            "node_trace": [
                "coordinator_decide"
            ],
            "node_metrics": {
                "coordinator_decide_ms": 5.0,
            },
        }

    def apply_governance(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> dict[str, Any]:
        self._record("apply_governance")

        return {
            "shared_memory_ids": [
                "mem_dave_001",
            ],
            "rejected_memory_ids": [],
            "workflow_status": "shared",
            "node_trace": ["apply_governance"],
            "node_metrics": {
                "apply_governance_ms": 6.0,
            },
        }

    def retrieve_for_responder(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> dict[str, Any]:
        self._record("retrieve_for_responder")

        mode = state["experiment_mode"]

        if mode == "private_only":
            retrieved_ids = ["mem_alice_001"]
            context = (
                "[Memory 1]\n"
                "memory_id: mem_alice_001\n"
                "content: Alice is inspired by "
                "Alexander McQueen."
            )
            shared_ids = (
                ["mem_dave_001"]
                if self.private_mode_returns_shared_memory
                else []
            )
        else:
            retrieved_ids = [
                "mem_alice_001",
                "mem_dave_001",
            ]
            context = (
                "[Memory 1]\n"
                "memory_id: mem_alice_001\n"
                "content: Alice is inspired by "
                "Alexander McQueen.\n\n"
                "[Memory 2]\n"
                "memory_id: mem_dave_001\n"
                "content: Dave is inspired by "
                "Coco Chanel."
            )
            shared_ids = []

        update: dict[str, Any] = {
            "responder_retrieved_memory_ids": (
                retrieved_ids
            ),
            "responder_memory_context": context,
            "workflow_status": "retrieved",
            "node_trace": [
                "retrieve_for_responder"
            ],
            "node_metrics": {
                "retrieve_for_responder_ms": 7.0,
            },
        }

        if shared_ids:
            update["shared_memory_ids"] = shared_ids

        return update

    def generate_answer(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> dict[str, Any]:
        self._record("generate_answer")

        if self.omit_answer:
            return {
                "workflow_status": self.final_status,
                "node_trace": ["generate_answer"],
            }

        mode = state["experiment_mode"]

        if mode == "private_only":
            answer_text = "Alexander McQueen"
            used_memory_ids = ["mem_alice_001"]
            contributing_agents = ["alice_agent"]
        else:
            answer_text = (
                "Alexander McQueen and Coco Chanel"
            )
            used_memory_ids = [
                "mem_alice_001",
                "mem_dave_001",
            ]
            contributing_agents = [
                "alice_agent",
                "dave_agent",
            ]

        if self.hallucinated_used_memory_id:
            used_memory_ids.append("mem_hidden_999")

        answer = AgentAnswer(
            answer=answer_text,
            reasoning=(
                "The answer uses the responder's retrieved "
                "memory context."
            ),
            confidence=0.95,
            used_memory_ids=used_memory_ids,
            supporting_source_ids=[],
            contributing_agent_ids=(
                contributing_agents
            ),
        )

        return {
            "agent_answer": answer,
            "workflow_status": self.final_status,
            "node_trace": ["generate_answer"],
            "node_metrics": {
                "generate_answer_ms": 8.0,
            },
        }


def build_workflow(
    fake_nodes: FakeNodes | None = None,
    *,
    validate_initial_state: bool = True,
    validate_final_state: bool = True,
) -> tuple[NeedlePersonaWorkflow, FakeNodes]:
    nodes = fake_nodes or FakeNodes()

    workflow = NeedlePersonaWorkflow(
        nodes=cast(Any, nodes),
        config=NeedlePersonaWorkflowConfig(
            validate_initial_state=(
                validate_initial_state
            ),
            validate_final_state=(
                validate_final_state
            ),
        ),
    )

    return workflow, nodes


# ---------------------------------------------------------------------------
# Branch execution tests
# ---------------------------------------------------------------------------


def test_private_only_executes_only_responder_path(
    sample: NeedlePersonaSample,
    memory_index: PrivateMemoryIndex,
) -> None:
    workflow, nodes = build_workflow()

    initial_state = build_initial_state(
        sample=sample,
        memory_index=memory_index,
        mode="private_only",
    )

    final_state = workflow.invoke(initial_state)

    assert nodes.calls == [
        "route_task",
        "retrieve_for_responder",
        "generate_answer",
    ]

    assert final_state["node_trace"] == [
        "route_task",
        "retrieve_for_responder",
        "generate_answer",
    ]

    assert (
        "collect_candidate_memories"
        not in nodes.calls
    )
    assert "direct_share" not in nodes.calls
    assert "critic_review" not in nodes.calls

    assert final_state["shared_memory_ids"] == []
    assert (
        final_state[
            "responder_retrieved_memory_ids"
        ]
        == ["mem_alice_001"]
    )
    assert (
        final_state["agent_answer"].answer
        == "Alexander McQueen"
    )
    assert (
        final_state["agent_answer"].used_memory_ids
        == ["mem_alice_001"]
    )
    assert final_state["workflow_status"] == "answered"


def test_ungoverned_shared_executes_direct_share_path(
    sample: NeedlePersonaSample,
    memory_index: PrivateMemoryIndex,
) -> None:
    workflow, nodes = build_workflow()

    initial_state = build_initial_state(
        sample=sample,
        memory_index=memory_index,
        mode="ungoverned_shared",
    )

    final_state = workflow.invoke(initial_state)

    assert nodes.calls == [
        "route_task",
        "collect_candidate_memories",
        "direct_share",
        "retrieve_for_responder",
        "generate_answer",
    ]

    assert "submit_promotion_requests" not in nodes.calls
    assert "critic_review" not in nodes.calls
    assert "coordinator_decide" not in nodes.calls
    assert "apply_governance" not in nodes.calls

    assert final_state["candidate_memory_ids"] == [
        "mem_dave_001"
    ]
    assert final_state["shared_memory_ids"] == [
        "mem_dave_001"
    ]
    assert (
        final_state[
            "responder_retrieved_memory_ids"
        ]
        == [
            "mem_alice_001",
            "mem_dave_001",
        ]
    )
    assert (
        final_state["agent_answer"].answer
        == "Alexander McQueen and Coco Chanel"
    )
    assert (
        "Ungoverned sharing was used."
        in final_state["warnings"]
    )


def test_governed_shared_executes_complete_governance_path(
    sample: NeedlePersonaSample,
    memory_index: PrivateMemoryIndex,
) -> None:
    workflow, nodes = build_workflow()

    initial_state = build_initial_state(
        sample=sample,
        memory_index=memory_index,
        mode="governed_shared",
    )

    final_state = workflow.invoke(initial_state)

    assert nodes.calls == [
        "route_task",
        "collect_candidate_memories",
        "submit_promotion_requests",
        "critic_review",
        "coordinator_decide",
        "apply_governance",
        "retrieve_for_responder",
        "generate_answer",
    ]

    assert "direct_share" not in nodes.calls

    assert final_state[
        "promotion_request_ids"
    ] == {
        "mem_dave_001": "request_001"
    }

    review = final_state["critic_reviews"][
        "mem_dave_001"
    ]
    assert review.recommendation == "approve"
    assert review.relevant is True
    assert review.policy_compliant is True

    decision = final_state[
        "promotion_decisions"
    ]["mem_dave_001"]
    assert decision.decision == "approve"
    assert decision.target_scope == "shared"
    assert set(decision.allowed_agent_ids) == {
        "alice_agent",
        "dave_agent",
    }

    assert final_state["shared_memory_ids"] == [
        "mem_dave_001"
    ]
    assert (
        final_state["agent_answer"].used_memory_ids
        == [
            "mem_alice_001",
            "mem_dave_001",
        ]
    )


@pytest.mark.parametrize(
    ("mode", "expected_calls"),
    [
        (
            "private_only",
            [
                "route_task",
                "retrieve_for_responder",
                "generate_answer",
            ],
        ),
        (
            "ungoverned_shared",
            [
                "route_task",
                "collect_candidate_memories",
                "direct_share",
                "retrieve_for_responder",
                "generate_answer",
            ],
        ),
        (
            "governed_shared",
            [
                "route_task",
                "collect_candidate_memories",
                "submit_promotion_requests",
                "critic_review",
                "coordinator_decide",
                "apply_governance",
                "retrieve_for_responder",
                "generate_answer",
            ],
        ),
    ],
)
def test_node_trace_matches_execution_order(
    sample: NeedlePersonaSample,
    memory_index: PrivateMemoryIndex,
    mode: str,
    expected_calls: list[str],
) -> None:
    workflow, nodes = build_workflow()

    initial_state = build_initial_state(
        sample=sample,
        memory_index=memory_index,
        mode=mode,
    )

    final_state = workflow.invoke(initial_state)

    assert nodes.calls == expected_calls
    assert final_state["node_trace"] == expected_calls


def test_node_metrics_are_merged_across_nodes(
    sample: NeedlePersonaSample,
    memory_index: PrivateMemoryIndex,
) -> None:
    workflow, _ = build_workflow()

    final_state = workflow.invoke(
        build_initial_state(
            sample=sample,
            memory_index=memory_index,
            mode="governed_shared",
        )
    )

    assert set(final_state["node_metrics"]) == {
        "route_task_ms",
        "collect_candidate_memories_ms",
        "submit_promotion_requests_ms",
        "critic_review_ms",
        "coordinator_decide_ms",
        "apply_governance_ms",
        "retrieve_for_responder_ms",
        "generate_answer_ms",
    }


# ---------------------------------------------------------------------------
# Streaming and graph introspection
# ---------------------------------------------------------------------------


def test_stream_updates_exposes_nodes_in_execution_order(
    sample: NeedlePersonaSample,
    memory_index: PrivateMemoryIndex,
) -> None:
    workflow, _ = build_workflow()

    initial_state = build_initial_state(
        sample=sample,
        memory_index=memory_index,
        mode="ungoverned_shared",
    )

    updates = list(
        workflow.stream(
            initial_state,
            stream_mode="updates",
        )
    )

    update_node_names = [
        next(iter(update))
        for update in updates
    ]

    assert update_node_names == [
        "route_task",
        "collect_candidate_memories",
        "direct_share",
        "retrieve_for_responder",
        "generate_answer",
    ]


def test_draw_mermaid_contains_all_workflow_nodes() -> None:
    workflow, _ = build_workflow()

    mermaid = workflow.draw_mermaid()

    expected_nodes = {
        "route_task",
        "collect_candidate_memories",
        "direct_share",
        "submit_promotion_requests",
        "critic_review",
        "coordinator_decide",
        "apply_governance",
        "retrieve_for_responder",
        "generate_answer",
    }

    for node_name in expected_nodes:
        assert node_name in mermaid


def test_build_workflow_factory_accepts_configured_nodes() -> None:
    nodes = FakeNodes()

    workflow = build_needle_persona_workflow(
        nodes=cast(Any, nodes),
        workflow_config=NeedlePersonaWorkflowConfig(),
    )

    assert isinstance(
        workflow,
        NeedlePersonaWorkflow,
    )
    assert workflow.nodes is nodes


# ---------------------------------------------------------------------------
# Initial-state validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "gold_answer",
        "alternative_answers",
        "accepted_answers",
        "persona_sources",
        "raw_conversations",
        "raw_dialogues",
        "chat_bob_charlie",
        "needle_detail",
    ],
)
def test_initial_state_rejects_reference_and_raw_source_fields(
    sample: NeedlePersonaSample,
    memory_index: PrivateMemoryIndex,
    forbidden_key: str,
) -> None:
    workflow, nodes = build_workflow()

    state = build_initial_state(
        sample=sample,
        memory_index=memory_index,
        mode="private_only",
    )
    state[forbidden_key] = "leaked value"  # type: ignore[literal-required]

    with pytest.raises(
        NeedlePersonaWorkflowError,
        match="must remain outside the task graph",
    ):
        workflow.invoke(state)

    assert nodes.calls == []


@pytest.mark.parametrize(
    "missing_field",
    [
        "run_id",
        "sample_id",
        "question",
    ],
)
def test_initial_state_requires_identity_and_question_fields(
    sample: NeedlePersonaSample,
    memory_index: PrivateMemoryIndex,
    missing_field: str,
) -> None:
    workflow, nodes = build_workflow()

    state = dict(
        build_initial_state(
            sample=sample,
            memory_index=memory_index,
            mode="private_only",
        )
    )
    state.pop(missing_field)

    with pytest.raises(
        NeedlePersonaWorkflowError,
        match=missing_field,
    ):
        workflow.invoke(
            cast(
                NeedlePersonaWorkflowState,
                state,
            )
        )

    assert nodes.calls == []


def test_initial_state_rejects_invalid_experiment_mode(
    sample: NeedlePersonaSample,
    memory_index: PrivateMemoryIndex,
) -> None:
    workflow, nodes = build_workflow()

    state = build_initial_state(
        sample=sample,
        memory_index=memory_index,
        mode="unsupported_mode",
    )

    with pytest.raises(
        NeedlePersonaWorkflowError,
        match="invalid experiment_mode",
    ):
        workflow.invoke(state)

    assert nodes.calls == []


def test_initial_state_rejects_unknown_persona_agent(
    sample: NeedlePersonaSample,
    memory_index: PrivateMemoryIndex,
) -> None:
    workflow, nodes = build_workflow()

    state = build_initial_state(
        sample=sample,
        memory_index=memory_index,
        mode="private_only",
    )
    state["private_memory_ids"][
        "unknown_agent"
    ] = ["mem_unknown"]  # type: ignore[typeddict-unknown-key]

    with pytest.raises(
        NeedlePersonaWorkflowError,
        match="unknown persona agents",
    ):
        workflow.invoke(state)

    assert nodes.calls == []


def test_initial_state_rejects_non_list_memory_ids(
    sample: NeedlePersonaSample,
    memory_index: PrivateMemoryIndex,
) -> None:
    workflow, nodes = build_workflow()

    state = build_initial_state(
        sample=sample,
        memory_index=memory_index,
        mode="private_only",
    )
    state["private_memory_ids"][
        "alice_agent"
    ] = "mem_alice_001"  # type: ignore[assignment]

    with pytest.raises(
        NeedlePersonaWorkflowError,
        match="values must be lists",
    ):
        workflow.invoke(state)

    assert nodes.calls == []


def test_initial_state_rejects_duplicate_memory_ids(
    sample: NeedlePersonaSample,
    memory_index: PrivateMemoryIndex,
) -> None:
    workflow, nodes = build_workflow()

    state = build_initial_state(
        sample=sample,
        memory_index=memory_index,
        mode="private_only",
    )
    state["private_memory_ids"][
        "alice_agent"
    ] = [
        "mem_alice_001",
        "mem_alice_001",
    ]

    with pytest.raises(
        NeedlePersonaWorkflowError,
        match="Duplicate memory IDs",
    ):
        workflow.invoke(state)

    assert nodes.calls == []


def test_initial_state_helper_excludes_reference_answer(
    sample: NeedlePersonaSample,
    memory_index: PrivateMemoryIndex,
) -> None:
    state = build_initial_state(
        sample=sample,
        memory_index=memory_index,
        mode="governed_shared",
    )

    serialised = repr(state)

    assert "gold_answer" not in state
    assert "alternative_answers" not in state
    assert "persona_sources" not in state

    assert sample.gold_answer not in serialised

    for sources in sample.persona_sources.values():
        for source in sources:
            assert source.content not in serialised


# ---------------------------------------------------------------------------
# Final-state validation
# ---------------------------------------------------------------------------


def test_final_state_rejects_hallucinated_used_memory_id(
    sample: NeedlePersonaSample,
    memory_index: PrivateMemoryIndex,
) -> None:
    nodes = FakeNodes(
        hallucinated_used_memory_id=True
    )
    workflow, _ = build_workflow(nodes)

    with pytest.raises(
        NeedlePersonaWorkflowError,
        match="were not retrieved",
    ):
        workflow.invoke(
            build_initial_state(
                sample=sample,
                memory_index=memory_index,
                mode="governed_shared",
            )
        )


def test_final_state_rejects_shared_memory_in_private_only_mode(
    sample: NeedlePersonaSample,
    memory_index: PrivateMemoryIndex,
) -> None:
    nodes = FakeNodes(
        private_mode_returns_shared_memory=True
    )
    workflow, _ = build_workflow(nodes)

    with pytest.raises(
        NeedlePersonaWorkflowError,
        match="private_only.*shared_memory_ids",
    ):
        workflow.invoke(
            build_initial_state(
                sample=sample,
                memory_index=memory_index,
                mode="private_only",
            )
        )


def test_final_state_rejects_invalid_responder(
    sample: NeedlePersonaSample,
    memory_index: PrivateMemoryIndex,
) -> None:
    nodes = FakeNodes(
        invalid_responder=True
    )
    workflow, _ = build_workflow(nodes)

    with pytest.raises(
        NeedlePersonaWorkflowError,
        match="invalid responder_agent_id",
    ):
        workflow.invoke(
            build_initial_state(
                sample=sample,
                memory_index=memory_index,
                mode="private_only",
            )
        )


def test_final_state_requires_agent_answer(
    sample: NeedlePersonaSample,
    memory_index: PrivateMemoryIndex,
) -> None:
    nodes = FakeNodes(
        omit_answer=True,
        final_status="answered",
    )
    workflow, _ = build_workflow(nodes)

    with pytest.raises(
        NeedlePersonaWorkflowError,
        match="without agent_answer",
    ):
        workflow.invoke(
            build_initial_state(
                sample=sample,
                memory_index=memory_index,
                mode="private_only",
            )
        )


def test_final_state_requires_answered_status(
    sample: NeedlePersonaSample,
    memory_index: PrivateMemoryIndex,
) -> None:
    nodes = FakeNodes(
        final_status="retrieved"
    )
    workflow, _ = build_workflow(nodes)

    with pytest.raises(
        NeedlePersonaWorkflowError,
        match="without reaching answered status",
    ):
        workflow.invoke(
            build_initial_state(
                sample=sample,
                memory_index=memory_index,
                mode="private_only",
            )
        )


def test_final_validation_can_be_disabled_for_debugging(
    sample: NeedlePersonaSample,
    memory_index: PrivateMemoryIndex,
) -> None:
    nodes = FakeNodes(
        omit_answer=True,
        final_status="retrieved",
    )
    workflow, _ = build_workflow(
        nodes,
        validate_final_state=False,
    )

    final_state = workflow.invoke(
        build_initial_state(
            sample=sample,
            memory_index=memory_index,
            mode="private_only",
        )
    )

    assert final_state["workflow_status"] == "retrieved"
    assert "agent_answer" not in final_state


# ---------------------------------------------------------------------------
# Error wrapping
# ---------------------------------------------------------------------------


def test_node_error_is_wrapped_as_workflow_error(
    sample: NeedlePersonaSample,
    memory_index: PrivateMemoryIndex,
) -> None:
    nodes = FakeNodes(
        fail_node="critic_review"
    )
    workflow, _ = build_workflow(nodes)

    with pytest.raises(
        NeedlePersonaWorkflowError,
        match="Simulated node failure",
    ) as error_info:
        workflow.invoke(
            build_initial_state(
                sample=sample,
                memory_index=memory_index,
                mode="governed_shared",
            )
        )

    assert (
        error_info.value.node_name
        == "critic_review"
    )
    assert isinstance(
        error_info.value.cause,
        NeedlePersonaNodeError,
    )

    assert nodes.calls == [
        "route_task",
        "collect_candidate_memories",
        "submit_promotion_requests",
        "critic_review",
    ]


def test_stream_wraps_node_error(
    sample: NeedlePersonaSample,
    memory_index: PrivateMemoryIndex,
) -> None:
    nodes = FakeNodes(
        fail_node="direct_share"
    )
    workflow, _ = build_workflow(nodes)

    stream = workflow.stream(
        build_initial_state(
            sample=sample,
            memory_index=memory_index,
            mode="ungoverned_shared",
        ),
        stream_mode="updates",
    )

    with pytest.raises(
        NeedlePersonaWorkflowError,
        match="Simulated node failure",
    ):
        list(stream)


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------


def test_package_exports_workflow_api() -> None:
    import experiments.evaluator.informativebench as package

    expected_names = {
        "NeedlePersonaWorkflow",
        "NeedlePersonaWorkflowConfig",
        "NeedlePersonaWorkflowError",
        "build_needle_persona_workflow",
        "create_needle_persona_workflow",
    }

    missing = [
        name
        for name in expected_names
        if not hasattr(package, name)
    ]

    assert not missing, (
        f"Missing workflow package exports: {missing}"
    )