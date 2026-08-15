from __future__ import annotations

"""
Decision contracts used by the generic memory-governance subgraphs.

This module defines the capabilities that Coordinator and Critic
implementations must provide. It does not:

- call an LLM;
- build prompts;
- contain LangGraph state or routing logic;
- execute access-policy changes;
- persist requests or memories.

Concrete experiment adapters may use different prompts and LLM clients, but
they expose the same interfaces to the generic governance nodes.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    from src.llm.output_schemas import (
        MemoryAccessDecisionOutput,
        MemoryAccessRequestDecisionOutput,
        MemoryAccessRequestReviewOutput,
        MemoryAccessReviewOutput,
    )


DecisionContext = Mapping[str, Any]
AgentIdSequence = Sequence[str]


# ---------------------------------------------------------------------------
# Initial access-policy assignment
# ---------------------------------------------------------------------------


@runtime_checkable
class InitialPolicyCoordinatorProtocol(Protocol):
    """
    Coordinator capabilities required by the policy-assignment subgraph.
    """

    def assign_initial_policy(
        self,
        *,
        memory_id: str,
        owner_agent_id: str,
        available_agent_ids: AgentIdSequence,
        memory_context: DecisionContext,
        policy_context: DecisionContext,
        task_context: DecisionContext,
    ) -> MemoryAccessDecisionOutput:
        """
        Propose the initial read-access policy for an existing memory.

        The returned object is a decision proposal only. It must not mutate
        the memory or persist ACL changes.
        """
        ...

    def finalise_initial_policy(
        self,
        *,
        initial_decision: MemoryAccessDecisionOutput,
        critic_review: MemoryAccessReviewOutput,
        owner_agent_id: str,
        available_agent_ids: AgentIdSequence,
        memory_context: DecisionContext,
        policy_context: DecisionContext,
        task_context: DecisionContext,
    ) -> MemoryAccessDecisionOutput:
        """
        Produce the final access-policy decision after Critic review.

        The Coordinator retains final authority. The returned decision is
        subsequently executed by MemoryAccessPolicyGateway.
        """
        ...


@runtime_checkable
class InitialPolicyCriticProtocol(Protocol):
    """
    Critic capability required by the policy-assignment subgraph.
    """

    def review_initial_policy(
        self,
        *,
        proposed_decision: MemoryAccessDecisionOutput,
        owner_agent_id: str,
        available_agent_ids: AgentIdSequence,
        memory_context: DecisionContext,
        policy_context: DecisionContext,
        task_context: DecisionContext,
    ) -> MemoryAccessReviewOutput:
        """
        Review a proposed initial access policy.

        The review is advisory. It must not directly modify the proposed
        decision, memory metadata, or persisted ACL.
        """
        ...


# ---------------------------------------------------------------------------
# Runtime access-request decisions
# ---------------------------------------------------------------------------


@runtime_checkable
class AccessRequestCoordinatorProtocol(Protocol):
    """
    Coordinator capabilities required by the access-request subgraph.
    """

    def evaluate_access_request(
        self,
        *,
        request_id: str,
        memory_id: str,
        owner_agent_id: str,
        requester_agent_id: str,
        request_reason: str,
        task_id: str | None,
        available_agent_ids: AgentIdSequence,
        memory_context: DecisionContext,
        policy_context: DecisionContext,
        task_context: DecisionContext,
    ) -> MemoryAccessRequestDecisionOutput:
        """
        Produce an initial decision for a runtime access request.

        The result may indicate that Critic review is required. It does not
        approve, reject, or persist the request by itself.
        """
        ...

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
        available_agent_ids: AgentIdSequence,
        memory_context: DecisionContext,
        policy_context: DecisionContext,
        task_context: DecisionContext,
    ) -> MemoryAccessRequestDecisionOutput:
        """
        Produce the final access-request decision after Critic review.

        The returned decision is subsequently executed through
        MemorySharingGateway.
        """
        ...


@runtime_checkable
class AccessRequestCriticProtocol(Protocol):
    """
    Critic capability required by the access-request subgraph.
    """

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
        available_agent_ids: AgentIdSequence,
        memory_context: DecisionContext,
        policy_context: DecisionContext,
        task_context: DecisionContext,
    ) -> MemoryAccessRequestReviewOutput:
        """
        Review an initial runtime access-request decision.

        The review is advisory and must not call MemorySharingGateway or
        modify the persisted request state directly.
        """
        ...


# ---------------------------------------------------------------------------
# Composite interfaces
# ---------------------------------------------------------------------------


@runtime_checkable
class MemoryGovernanceCoordinatorProtocol(
    InitialPolicyCoordinatorProtocol,
    AccessRequestCoordinatorProtocol,
    Protocol,
):
    """
    Complete Coordinator interface for both governance subgraphs.

    Dependencies that need only one workflow should use the narrower
    InitialPolicyCoordinatorProtocol or AccessRequestCoordinatorProtocol.
    """


@runtime_checkable
class MemoryGovernanceCriticProtocol(
    InitialPolicyCriticProtocol,
    AccessRequestCriticProtocol,
    Protocol,
):
    """
    Complete Critic interface for both governance subgraphs.

    Dependencies that need only one workflow should use the narrower
    InitialPolicyCriticProtocol or AccessRequestCriticProtocol.
    """


__all__ = [
    "DecisionContext",
    "AgentIdSequence",
    "InitialPolicyCoordinatorProtocol",
    "InitialPolicyCriticProtocol",
    "AccessRequestCoordinatorProtocol",
    "AccessRequestCriticProtocol",
    "MemoryGovernanceCoordinatorProtocol",
    "MemoryGovernanceCriticProtocol",
]