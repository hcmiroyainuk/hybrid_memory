from __future__ import annotations

from ..entities import (
    Agent,
    AgentRole,
    MemoryItem,
    MemoryScope,
    MemoryStatus,
)


class PermissionDeniedError(Exception):
    """Raised when an agent does not have permission to perform an operation."""


class PermissionService:
    """
    Centralised permission services for memory governance.

    This services only checks whether an agent is allowed to perform an action.
    It does not modify memories, requests, or logs.

    Permission model:
    - Role-level permission: defined by Agent.role and Agent permission flags.
    - Object-level permission: defined by MemoryMetadata.readable_by / writable_by.
    """

    # ------------------------------------------------------------------
    # Read permissions
    # ------------------------------------------------------------------

    @staticmethod
    def can_read_memory(agent: Agent, memory: MemoryItem) -> bool:
        """
        Check whether an agent can read a memory.

        Rules:
        - Shared memory can be read by agents with shared-read permission.
        - Private memory can only be read if the memory explicitly allows it.
        - Deprecated memories are not considered readable in normal workflow.
        """

        if memory.metadata.status == MemoryStatus.DEPRECATED:
            return False

        if memory.metadata.scope == MemoryScope.SHARED:
            return agent.can_read_shared and memory.can_be_read_by(agent.agent_id)

        if memory.metadata.scope == MemoryScope.PRIVATE:
            return memory.can_be_read_by(agent.agent_id)

        return False

    @staticmethod
    def assert_can_read_memory(agent: Agent, memory: MemoryItem) -> None:
        """
        Raise PermissionDeniedError if the agent cannot read the memory.
        """

        if not PermissionService.can_read_memory(agent, memory):
            raise PermissionDeniedError(
                f"Agent '{agent.agent_id}' is not allowed to read memory "
                f"'{memory.memory_id}'."
            )

    # ------------------------------------------------------------------
    # Write permissions
    # ------------------------------------------------------------------

    @staticmethod
    def can_write_memory(agent: Agent, memory: MemoryItem) -> bool:
        """
        Check whether an agent can update a memory.

        Rules:
        - Shared memory can only be directly written by agents with shared-write permission.
        - Private memory can only be written by agents with private-write permission
          and object-level write access.
        - Deprecated memories should not be updated in normal workflow.
        """

        if memory.metadata.status == MemoryStatus.DEPRECATED:
            return False

        if memory.metadata.scope == MemoryScope.SHARED:
            return (
                agent.can_write_shared
                and memory.can_be_written_by(agent.agent_id)
            )

        if memory.metadata.scope == MemoryScope.PRIVATE:
            return (
                agent.can_write_private
                and memory.can_be_written_by(agent.agent_id)
            )

        return False

    @staticmethod
    def assert_can_write_memory(agent: Agent, memory: MemoryItem) -> None:
        """
        Raise PermissionDeniedError if the agent cannot write the memory.
        """

        if not PermissionService.can_write_memory(agent, memory):
            raise PermissionDeniedError(
                f"Agent '{agent.agent_id}' is not allowed to write memory "
                f"'{memory.memory_id}'."
            )

    # ------------------------------------------------------------------
    # Creation permissions
    # ------------------------------------------------------------------

    @staticmethod
    def can_create_private_memory(agent: Agent) -> bool:
        """
        Check whether an agent can create private memory.

        Baseline rule:
        - Worker, critic, and coordinator can all create their own private memories
          if can_write_private is True.
        """

        return agent.can_write_private

    @staticmethod
    def assert_can_create_private_memory(agent: Agent) -> None:
        """
        Raise PermissionDeniedError if the agent cannot create private memory.
        """

        if not PermissionService.can_create_private_memory(agent):
            raise PermissionDeniedError(
                f"Agent '{agent.agent_id}' is not allowed to create private memory."
            )

    @staticmethod
    def can_create_shared_memory(agent: Agent) -> bool:
        """
        Check whether an agent can directly create shared memory.

        Baseline rule:
        - Only coordinator can directly create shared memory.
        """

        return (
            agent.role == AgentRole.COORDINATOR
            and agent.can_write_shared
        )

    @staticmethod
    def assert_can_create_shared_memory(agent: Agent) -> None:
        """
        Raise PermissionDeniedError if the agent cannot create shared memory.
        """

        if not PermissionService.can_create_shared_memory(agent):
            raise PermissionDeniedError(
                f"Agent '{agent.agent_id}' is not allowed to create shared memory."
            )

    # ------------------------------------------------------------------
    # Promotion permissions
    # ------------------------------------------------------------------

    @staticmethod
    def can_propose_promotion(agent: Agent, memory: MemoryItem) -> bool:
        """
        Check whether an agent can propose a private memory for shared promotion.

        Rules:
        - The memory must be private.
        - The memory must be active.
        - The proposing agent must be able to read the memory.
        - Normally, this means the proposer is the owner of the private memory.
        """

        if memory.metadata.scope != MemoryScope.PRIVATE:
            return False

        if memory.metadata.status != MemoryStatus.ACTIVE:
            return False

        if not memory.can_be_read_by(agent.agent_id):
            return False

        return True

    @staticmethod
    def assert_can_propose_promotion(agent: Agent, memory: MemoryItem) -> None:
        """
        Raise PermissionDeniedError if the agent cannot propose the memory for promotion.
        """

        if not PermissionService.can_propose_promotion(agent, memory):
            raise PermissionDeniedError(
                f"Agent '{agent.agent_id}' is not allowed to propose memory "
                f"'{memory.memory_id}' for promotion."
            )

    @staticmethod
    def can_review_promotion(agent: Agent) -> bool:
        """
        Check whether an agent can review a promotion request.

        Baseline rule:
        - Critic and coordinator can review promotion requests.
        """

        return (
            agent.can_review_memory
            and agent.role in {AgentRole.CRITIC, AgentRole.COORDINATOR}
        )

    @staticmethod
    def assert_can_review_promotion(agent: Agent) -> None:
        """
        Raise PermissionDeniedError if the agent cannot review promotion requests.
        """

        if not PermissionService.can_review_promotion(agent):
            raise PermissionDeniedError(
                f"Agent '{agent.agent_id}' is not allowed to review promotion requests."
            )

    @staticmethod
    def can_approve_promotion(agent: Agent) -> bool:
        """
        Check whether an agent can approve and execute memory promotion.

        Baseline rule:
        - Only coordinator can approve promotion.
        - Critic can review but cannot approve the final shared-memory update.
        """

        return (
            agent.role == AgentRole.COORDINATOR
            and agent.can_review_memory
            and agent.can_write_shared
        )

    @staticmethod
    def assert_can_approve_promotion(agent: Agent) -> None:
        """
        Raise PermissionDeniedError if the agent cannot approve promotion.
        """

        if not PermissionService.can_approve_promotion(agent):
            raise PermissionDeniedError(
                f"Agent '{agent.agent_id}' is not allowed to approve promotion requests."
            )

    @staticmethod
    def can_reject_promotion(agent: Agent) -> bool:
        """
        Check whether an agent can reject a promotion request.

        Baseline rule:
        - Only coordinator can make the final rejection decision.
        - Critic can provide recommendation but not final rejection.
        """

        return (
            agent.role == AgentRole.COORDINATOR
            and agent.can_review_memory
        )

    @staticmethod
    def assert_can_reject_promotion(agent: Agent) -> None:
        """
        Raise PermissionDeniedError if the agent cannot reject promotion.
        """

        if not PermissionService.can_reject_promotion(agent):
            raise PermissionDeniedError(
                f"Agent '{agent.agent_id}' is not allowed to reject promotion requests."
            )

    # ------------------------------------------------------------------
    # Conflict permissions
    # ------------------------------------------------------------------

    @staticmethod
    def can_detect_conflict(agent: Agent) -> bool:
        """
        Check whether an agent can detect or report memory conflicts.

        Baseline rule:
        - Critic and coordinator can detect/report conflicts.
        """

        return (
            agent.can_review_memory
            and agent.role in {AgentRole.CRITIC, AgentRole.COORDINATOR}
        )

    @staticmethod
    def assert_can_detect_conflict(agent: Agent) -> None:
        """
        Raise PermissionDeniedError if the agent cannot detect conflicts.
        """

        if not PermissionService.can_detect_conflict(agent):
            raise PermissionDeniedError(
                f"Agent '{agent.agent_id}' is not allowed to detect memory conflicts."
            )

    @staticmethod
    def can_resolve_conflict(agent: Agent) -> bool:
        """
        Check whether an agent can resolve memory conflicts.

        Baseline rule:
        - Only coordinator can make final conflict resolution decisions.
        """

        return (
            agent.role == AgentRole.COORDINATOR
            and agent.can_review_memory
            and agent.can_write_shared
        )

    @staticmethod
    def assert_can_resolve_conflict(agent: Agent) -> None:
        """
        Raise PermissionDeniedError if the agent cannot resolve conflicts.
        """

        if not PermissionService.can_resolve_conflict(agent):
            raise PermissionDeniedError(
                f"Agent '{agent.agent_id}' is not allowed to resolve memory conflicts."
            )

    # ------------------------------------------------------------------
    # Retrieval permissions
    # ------------------------------------------------------------------

    @staticmethod
    def can_retrieve_shared_memory(agent: Agent) -> bool:
        """
        Check whether an agent can retrieve shared memories.
        """

        return agent.can_read_shared

    @staticmethod
    def can_retrieve_private_memory(agent: Agent, owner_agent_id: str) -> bool:
        """
        Check whether an agent can retrieve private memories owned by owner_agent_id.

        Baseline rule:
        - Private memory can only be retrieved by its owner.
        """

        return agent.agent_id == owner_agent_id

    # ------------------------------------------------------------------
    # Generic operation helper
    # ------------------------------------------------------------------

    @staticmethod
    def can_perform_memory_operation(
        agent: Agent,
        operation: str,
        memory: MemoryItem | None = None,
    ) -> bool:
        """
        Generic operation dispatcher.

        This is useful when services layer wants to check permission by operation name.
        """

        operation = operation.lower()

        if operation == "create_private":
            return PermissionService.can_create_private_memory(agent)

        if operation == "create_shared":
            return PermissionService.can_create_shared_memory(agent)

        if operation == "read":
            if memory is None:
                return False
            return PermissionService.can_read_memory(agent, memory)

        if operation == "write":
            if memory is None:
                return False
            return PermissionService.can_write_memory(agent, memory)

        if operation == "propose_promotion":
            if memory is None:
                return False
            return PermissionService.can_propose_promotion(agent, memory)

        if operation == "review_promotion":
            return PermissionService.can_review_promotion(agent)

        if operation == "approve_promotion":
            return PermissionService.can_approve_promotion(agent)

        if operation == "reject_promotion":
            return PermissionService.can_reject_promotion(agent)

        if operation == "detect_conflict":
            return PermissionService.can_detect_conflict(agent)

        if operation == "resolve_conflict":
            return PermissionService.can_resolve_conflict(agent)

        return False