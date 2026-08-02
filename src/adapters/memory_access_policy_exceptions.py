class MemoryAccessPolicyGatewayError(Exception):
    """
    Base exception exposed by the memory access-policy gateway.

    Workflow and governance code should catch this exception hierarchy
    instead of depending on MemoryAccessPolicyService implementation errors.
    """


class MemoryAccessPolicyNotFoundError(
    MemoryAccessPolicyGatewayError
):
    """
    Raised when the target memory or governance actor cannot be found.
    """


class MemoryAccessPolicyValidationError(
    MemoryAccessPolicyGatewayError
):
    """
    Raised when a proposed access policy is structurally invalid.

    Examples:
    - an unsupported target scope;
    - a private policy containing non-owner readers;
    - an empty reader set for shared access;
    - combining the '*' wildcard with specific Agent IDs;
    - attempting to remove the memory owner from required ACL fields.
    """


class MemoryAccessPolicyPermissionError(
    MemoryAccessPolicyGatewayError
):
    """
    Raised when the governance actor is not authorised to modify a memory ACL.
    """


class MemoryAccessPolicyConflictError(
    MemoryAccessPolicyGatewayError
):
    """
    Raised when a valid operation conflicts with the current memory state.

    Examples:
    - changing the policy of a deprecated memory;
    - revoking access that cannot be revoked;
    - applying a transition that violates the memory lifecycle;
    - modifying a memory whose state changed during the operation.
    """


class MemoryAccessPolicyPersistenceError(
    MemoryAccessPolicyGatewayError
):
    """
    Raised when an access-policy change cannot be persisted safely.

    This includes failures where the memory update, audit-log write, or
    rollback operation does not complete successfully.
    """


__all__ = [
    "MemoryAccessPolicyGatewayError",
    "MemoryAccessPolicyNotFoundError",
    "MemoryAccessPolicyValidationError",
    "MemoryAccessPolicyPermissionError",
    "MemoryAccessPolicyConflictError",
    "MemoryAccessPolicyPersistenceError",
]