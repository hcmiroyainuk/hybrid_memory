from __future__ import annotations

from ..entities import (
    Agent,
    AgentRole,
    MemoryItem,
    MemoryScope,
    MemoryStatus,
)


class PermissionDeniedError(Exception):
    """Raised when an agent is not allowed to perform a memory operation."""


class PermissionService:
    """
    Centralised permission checks for governed multi-agent memory.

    Governance rules enforced by this service:
    - A Worker may create, read, and update only its own private memories.
    - A private memory is owner-only, even if its ACL is accidentally broader.
    - Shared memories are readable only when both the Agent role permission and
      the memory-level ``readable_by`` ACL allow access.
    - Only a Coordinator may directly create or update shared memories.
    - Only the Worker that owns an active private memory may submit it for
      promotion.
    - Only a Critic may review a promotion request and provide a recommendation.
    - Only a Coordinator may approve or reject promotion and resolve conflicts.

    This service performs validation only. It does not mutate memories,
    promotion requests, or operation logs.
    """

    # ------------------------------------------------------------------
    # Read permissions
    # ------------------------------------------------------------------

    @staticmethod
    def can_read_memory(agent: Agent, memory: MemoryItem) -> bool:
        """
        Return whether ``agent`` may read ``memory`` in the normal workflow.

        Deprecated memories are excluded from normal reads. Private memories
        are strictly owner-only. Shared memories require both role-level and
        object-level permission.
        """

        if memory.metadata.status == MemoryStatus.DEPRECATED:
            return False

        if memory.metadata.scope == MemoryScope.PRIVATE:
            return (
                agent.role == AgentRole.WORKER
                and memory.metadata.owner_agent_id == agent.agent_id
                and memory.can_be_read_by(agent.agent_id)
            )

        if memory.metadata.scope == MemoryScope.SHARED:
            return (
                agent.can_read_shared
                and memory.can_be_read_by(agent.agent_id)
            )

        return False

    @staticmethod
    def assert_can_read_memory(
        agent: Agent,
        memory: MemoryItem,
    ) -> None:
        """Raise ``PermissionDeniedError`` when the memory is not readable."""

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
        Return whether ``agent`` may update ``memory``.

        Workers may update only their own active private memories. Shared
        memory updates are restricted to a Coordinator that also passes the
        memory-level ``writable_by`` ACL.
        """

        if memory.metadata.status == MemoryStatus.DEPRECATED:
            return False

        if memory.metadata.scope == MemoryScope.PRIVATE:
            return (
                agent.role == AgentRole.WORKER
                and agent.can_write_private
                and memory.metadata.owner_agent_id == agent.agent_id
                and memory.can_be_written_by(agent.agent_id)
            )

        if memory.metadata.scope == MemoryScope.SHARED:
            return (
                agent.role == AgentRole.COORDINATOR
                and agent.can_write_shared
                and memory.can_be_written_by(agent.agent_id)
            )

        return False

    @staticmethod
    def assert_can_write_memory(
        agent: Agent,
        memory: MemoryItem,
    ) -> None:
        """Raise ``PermissionDeniedError`` when the memory is not writable."""

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
        """Return whether the Agent may create a private memory for itself."""

        return (
            agent.role == AgentRole.WORKER
            and agent.can_write_private
        )

    @staticmethod
    def assert_can_create_private_memory(agent: Agent) -> None:
        """Raise when the Agent may not create private memory."""

        if not PermissionService.can_create_private_memory(agent):
            raise PermissionDeniedError(
                f"Agent '{agent.agent_id}' is not allowed to create "
                "private memory. Only Workers with private-write "
                "permission may create private memories."
            )

    @staticmethod
    def can_create_shared_memory(agent: Agent) -> bool:
        """Return whether the Agent may directly create shared memory."""

        return (
            agent.role == AgentRole.COORDINATOR
            and agent.can_write_shared
        )

    @staticmethod
    def assert_can_create_shared_memory(agent: Agent) -> None:
        """Raise when the Agent may not directly create shared memory."""

        if not PermissionService.can_create_shared_memory(agent):
            raise PermissionDeniedError(
                f"Agent '{agent.agent_id}' is not allowed to create "
                "shared memory. Only the Coordinator may do so."
            )

    # ------------------------------------------------------------------
    # Promotion permissions
    # ------------------------------------------------------------------

    @staticmethod
    def can_propose_promotion(
        agent: Agent,
        memory: MemoryItem,
    ) -> bool:
        """
        Return whether ``agent`` may submit ``memory`` for shared promotion.

        The requester must be a Worker submitting its own active private
        memory. Object-level read and write ACLs must also remain valid.
        """

        if agent.role != AgentRole.WORKER:
            return False

        if not agent.can_write_private:
            return False

        if memory.metadata.scope != MemoryScope.PRIVATE:
            return False

        if memory.metadata.status != MemoryStatus.ACTIVE:
            return False

        if memory.metadata.owner_agent_id != agent.agent_id:
            return False

        if not memory.can_be_read_by(agent.agent_id):
            return False

        if not memory.can_be_written_by(agent.agent_id):
            return False

        return True

    @staticmethod
    def assert_can_propose_promotion(
        agent: Agent,
        memory: MemoryItem,
    ) -> None:
        """Raise when the Worker may not submit the promotion request."""

        if not PermissionService.can_propose_promotion(agent, memory):
            raise PermissionDeniedError(
                f"Agent '{agent.agent_id}' is not allowed to propose memory "
                f"'{memory.memory_id}' for promotion. A Worker may submit "
                "only its own active private memory."
            )

    @staticmethod
    def can_review_promotion(agent: Agent) -> bool:
        """Return whether the Agent may provide a promotion recommendation."""

        return (
            agent.role == AgentRole.CRITIC
            and agent.can_review_memory
        )

    @staticmethod
    def assert_can_review_promotion(agent: Agent) -> None:
        """Raise when the Agent may not review promotion requests."""

        if not PermissionService.can_review_promotion(agent):
            raise PermissionDeniedError(
                f"Agent '{agent.agent_id}' is not allowed to review "
                "promotion requests. Only the Critic may provide the "
                "recommendation."
            )

    @staticmethod
    def can_approve_promotion(agent: Agent) -> bool:
        """Return whether the Agent may make the final approval decision."""

        return (
            agent.role == AgentRole.COORDINATOR
            and agent.can_review_memory
            and agent.can_write_shared
        )

    @staticmethod
    def assert_can_approve_promotion(agent: Agent) -> None:
        """Raise when the Agent may not approve promotion requests."""

        if not PermissionService.can_approve_promotion(agent):
            raise PermissionDeniedError(
                f"Agent '{agent.agent_id}' is not allowed to approve "
                "promotion requests. Only the Coordinator may approve."
            )

    @staticmethod
    def can_reject_promotion(agent: Agent) -> bool:
        """Return whether the Agent may make the final rejection decision."""

        return (
            agent.role == AgentRole.COORDINATOR
            and agent.can_review_memory
        )

    @staticmethod
    def assert_can_reject_promotion(agent: Agent) -> None:
        """Raise when the Agent may not reject promotion requests."""

        if not PermissionService.can_reject_promotion(agent):
            raise PermissionDeniedError(
                f"Agent '{agent.agent_id}' is not allowed to reject "
                "promotion requests. Only the Coordinator may reject."
            )

    # ------------------------------------------------------------------
    # Conflict permissions
    # ------------------------------------------------------------------

    @staticmethod
    def can_detect_conflict(agent: Agent) -> bool:
        """Return whether the Agent may detect and report a conflict."""

        return (
            agent.role == AgentRole.CRITIC
            and agent.can_review_memory
        )

    @staticmethod
    def assert_can_detect_conflict(agent: Agent) -> None:
        """Raise when the Agent may not detect or report conflicts."""

        if not PermissionService.can_detect_conflict(agent):
            raise PermissionDeniedError(
                f"Agent '{agent.agent_id}' is not allowed to detect "
                "memory conflicts. Only the Critic may report them."
            )

    @staticmethod
    def can_resolve_conflict(agent: Agent) -> bool:
        """Return whether the Agent may make a final conflict decision."""

        return (
            agent.role == AgentRole.COORDINATOR
            and agent.can_review_memory
            and agent.can_write_shared
        )

    @staticmethod
    def assert_can_resolve_conflict(agent: Agent) -> None:
        """Raise when the Agent may not resolve conflicts."""

        if not PermissionService.can_resolve_conflict(agent):
            raise PermissionDeniedError(
                f"Agent '{agent.agent_id}' is not allowed to resolve "
                "memory conflicts. Only the Coordinator may decide the "
                "resolution."
            )

    # ------------------------------------------------------------------
    # Retrieval permissions
    # ------------------------------------------------------------------

    @staticmethod
    def can_retrieve_shared_memory(agent: Agent) -> bool:
        """Return whether the Agent may participate in shared retrieval."""

        return agent.can_read_shared

    @staticmethod
    def can_retrieve_private_memory(
        agent: Agent,
        owner_agent_id: str,
    ) -> bool:
        """Return whether the Worker may retrieve the owner's private memory."""

        return (
            agent.role == AgentRole.WORKER
            and agent.agent_id == owner_agent_id
        )

    # ------------------------------------------------------------------
    # Generic operation helper
    # ------------------------------------------------------------------

    @staticmethod
    def can_perform_memory_operation(
        agent: Agent,
        operation: str,
        memory: MemoryItem | None = None,
    ) -> bool:
        """Dispatch a named permission check to the corresponding rule."""

        normalized_operation = operation.strip().lower()

        if normalized_operation == "create_private":
            return PermissionService.can_create_private_memory(agent)

        if normalized_operation == "create_shared":
            return PermissionService.can_create_shared_memory(agent)

        if normalized_operation == "read":
            return (
                memory is not None
                and PermissionService.can_read_memory(agent, memory)
            )

        if normalized_operation == "write":
            return (
                memory is not None
                and PermissionService.can_write_memory(agent, memory)
            )

        if normalized_operation == "propose_promotion":
            return (
                memory is not None
                and PermissionService.can_propose_promotion(agent, memory)
            )

        if normalized_operation == "review_promotion":
            return PermissionService.can_review_promotion(agent)

        if normalized_operation == "approve_promotion":
            return PermissionService.can_approve_promotion(agent)

        if normalized_operation == "reject_promotion":
            return PermissionService.can_reject_promotion(agent)

        if normalized_operation == "detect_conflict":
            return PermissionService.can_detect_conflict(agent)

        if normalized_operation == "resolve_conflict":
            return PermissionService.can_resolve_conflict(agent)

        return False