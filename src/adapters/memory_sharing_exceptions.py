class MemorySharingError(Exception):
    """Base exception exposed by the memory-sharing gateway."""


class MemorySharingNotFoundError(MemorySharingError):
    """Raised when an Agent, Memory, or access request cannot be found."""


class MemorySharingConflictError(MemorySharingError):
    """
    Raised when an operation conflicts with current state.

    Examples include duplicate active requests, existing access, invalid
    lifecycle transitions, or attempts to modify completed requests.
    """


class MemorySharingPermissionError(MemorySharingError):
    """Raised when an actor is not authorised to perform an operation."""


__all__ = [
    "MemorySharingError",
    "MemorySharingNotFoundError",
    "MemorySharingConflictError",
    "MemorySharingPermissionError",
]