from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    from .memory_access_policy_models import (
        MemoryAccessPolicyResult,
    )
else:
    # Keeps this Protocol importable before the concrete workflow-facing
    # result model is added. Runtime type enforcement is not performed here.
    MemoryAccessPolicyResult = Any


@runtime_checkable
class MemoryAccessPolicyGateway(Protocol):
    """
    Workflow-facing contract for applying and maintaining memory ACLs.

    Workflow and governance modules should depend on this Protocol rather
    than importing MemoryAccessPolicyService or persistence-layer classes.

    The gateway receives Agent IDs rather than Agent objects. A concrete
    adapter is responsible for:

    - resolving ``governance_actor_id`` to an Agent;
    - calling MemoryAccessPolicyService;
    - converting the returned MemoryItem into MemoryAccessPolicyResult;
    - translating service-specific exceptions into stable gateway errors.

    This gateway does not decide which policy should be applied. It only
    executes a policy that has already been determined by the governance
    workflow.
    """

    def apply_policy(
        self,
        *,
        governance_actor_id: str,
        memory_id: str,
        target_scope: str,
        allowed_agent_ids: Sequence[str] = (),
        reason: str | None = None,
    ) -> MemoryAccessPolicyResult:
        """
        Replace the complete read-access policy of an existing memory.

        Args:
            governance_actor_id:
                ID of the authorised Agent applying the policy.

            memory_id:
                ID of the memory whose ACL will be changed.

            target_scope:
                Target visibility scope, normally ``"private"`` or
                ``"shared"``.

            allowed_agent_ids:
                Additional non-owner readers for a shared policy. The owner
                is preserved automatically by MemoryAccessPolicyService.
                Use exactly ``("*",)`` for global read access.

            reason:
                Optional audit reason for the policy change.

        Returns:
            A stable workflow-facing representation of the persisted policy.
        """
        ...

    def grant_read_access(
        self,
        *,
        governance_actor_id: str,
        memory_id: str,
        agent_ids: Sequence[str],
        reason: str | None = None,
    ) -> MemoryAccessPolicyResult:
        """
        Add one or more Agents to a memory's read ACL.

        Existing targeted readers are retained. A private memory becomes
        shared when at least one non-owner reader is granted access.
        """
        ...

    def revoke_read_access(
        self,
        *,
        governance_actor_id: str,
        memory_id: str,
        agent_ids: Sequence[str],
        reason: str | None = None,
    ) -> MemoryAccessPolicyResult:
        """
        Remove one or more Agents from a memory's read ACL.

        The memory owner cannot be removed. When the final non-owner reader
        is removed, the concrete service may return the memory to private
        owner-only access.
        """
        ...

    def make_private(
        self,
        *,
        governance_actor_id: str,
        memory_id: str,
        reason: str | None = None,
    ) -> MemoryAccessPolicyResult:
        """
        Restore a memory to private owner-only access.
        """
        ...

    def make_globally_shared(
        self,
        *,
        governance_actor_id: str,
        memory_id: str,
        reason: str | None = None,
    ) -> MemoryAccessPolicyResult:
        """
        Make a memory globally readable by applying the ``"*"`` ACL.
        """
        ...


__all__ = ["MemoryAccessPolicyGateway"]