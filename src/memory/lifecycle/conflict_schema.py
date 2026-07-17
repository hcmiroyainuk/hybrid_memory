from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ConflictType(str, Enum):
    """
    Types of memory consistency issues.

    NONE:
        No conflict detected.

    DUPLICATE:
        Candidate memory expresses the same question-answer fact as
        an existing memory.

    CONFLICTING_ANSWER:
        Candidate memory is about the same question as an existing memory,
        but the answer is different.

    OUTDATED:
        Candidate memory may supersede an older memory.
        This is reserved for later lifecycle handling.
    """

    NONE = "none"
    DUPLICATE = "duplicate"
    CONFLICTING_ANSWER = "conflicting_answer"
    OUTDATED = "outdated"


class ConflictSeverity(str, Enum):
    """
    Severity level of detected memory conflict.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ConflictDecision(str, Enum):
    """
    Lifecycle decision after conflict checking.

    ALLOW_WRITE_AND_PROMOTE:
        Candidate can be written as private memory and promoted to shared memory.

    ALLOW_WRITE_ONLY:
        Candidate can be written as private memory, but should not be promoted.

    BLOCK_WRITE:
        Candidate should not be written.

    REQUIRE_REVIEW:
        Candidate can be stored, but requires human or critic review before sharing.
    """

    ALLOW_WRITE_AND_PROMOTE = "allow_write_and_promote"
    ALLOW_WRITE_ONLY = "allow_write_only"
    BLOCK_WRITE = "block_write"
    REQUIRE_REVIEW = "require_review"


class MemoryConflictRecord(BaseModel):
    """
    One matched memory involved in a conflict check.

    This is used to keep a compact record of which existing memory caused
    duplicate or conflict detection.
    """

    matched_memory_id: str = Field(
        description="ID of the existing memory matched against the candidate."
    )
    matched_memory_content: str = Field(
        default="",
        description="Content of the matched existing memory."
    )
    matched_question: Optional[str] = Field(
        default=None,
        description="Question extracted from the matched memory."
    )
    matched_answer: Optional[str] = Field(
        default=None,
        description="Answer extracted from the matched memory."
    )
    similarity_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional similarity score between candidate and matched memory."
    )

    @field_validator("matched_memory_id")
    @classmethod
    def validate_matched_memory_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("matched_memory_id cannot be empty.")
        return value

    @field_validator("matched_memory_content")
    @classmethod
    def clean_matched_memory_content(cls, value: str) -> str:
        return value.strip()


class ConflictCheckResult(BaseModel):
    """
    Result of checking a candidate memory against existing memories.

    This object is used by:
    - ConflictDetector
    - MemoryWriteController
    - GovernedMemoryLifecycle
    """

    candidate_memory_id: str = Field(
        description="ID of the candidate memory being checked."
    )

    conflict_type: ConflictType = Field(
        default=ConflictType.NONE,
        description="Detected conflict type."
    )

    severity: ConflictSeverity = Field(
        default=ConflictSeverity.NONE,
        description="Severity of the detected conflict."
    )

    decision: ConflictDecision = Field(
        default=ConflictDecision.ALLOW_WRITE_AND_PROMOTE,
        description="Recommended lifecycle decision."
    )

    matched_records: list[MemoryConflictRecord] = Field(
        default_factory=list,
        description="Existing memories that matched or conflicted with the candidate."
    )

    message: str = Field(
        default="No conflict detected.",
        description="Human-readable explanation of the conflict check result."
    )

    should_write: bool = Field(
        default=True,
        description="Whether the candidate should be written as private memory."
    )

    should_promote: bool = Field(
        default=True,
        description="Whether the candidate should be promoted to shared memory."
    )

    requires_review: bool = Field(
        default=False,
        description="Whether the candidate requires review before promotion."
    )

    candidate_question: Optional[str] = Field(
        default=None,
        description="Question extracted from the candidate memory."
    )

    candidate_answer: Optional[str] = Field(
        default=None,
        description="Answer extracted from the candidate memory."
    )

    @field_validator("candidate_memory_id")
    @classmethod
    def validate_candidate_memory_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("candidate_memory_id cannot be empty.")
        return value

    @field_validator("message")
    @classmethod
    def clean_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            return "No conflict message."
        return value

    @classmethod
    def no_conflict(
        cls,
        *,
        candidate_memory_id: str,
        candidate_question: Optional[str] = None,
        candidate_answer: Optional[str] = None,
        message: str = "No conflict detected.",
    ) -> "ConflictCheckResult":
        """
        Candidate memory is safe to write and promote.
        """

        return cls(
            candidate_memory_id=candidate_memory_id,
            conflict_type=ConflictType.NONE,
            severity=ConflictSeverity.NONE,
            decision=ConflictDecision.ALLOW_WRITE_AND_PROMOTE,
            matched_records=[],
            message=message,
            should_write=True,
            should_promote=True,
            requires_review=False,
            candidate_question=candidate_question,
            candidate_answer=candidate_answer,
        )

    @classmethod
    def duplicate(
        cls,
        *,
        candidate_memory_id: str,
        matched_records: list[MemoryConflictRecord],
        candidate_question: Optional[str] = None,
        candidate_answer: Optional[str] = None,
        message: str = "Duplicate memory detected.",
    ) -> "ConflictCheckResult":
        """
        Candidate duplicates existing memory.

        Default behavior:
        - do not write
        - do not promote
        - no review required
        """

        return cls(
            candidate_memory_id=candidate_memory_id,
            conflict_type=ConflictType.DUPLICATE,
            severity=ConflictSeverity.LOW,
            decision=ConflictDecision.BLOCK_WRITE,
            matched_records=matched_records,
            message=message,
            should_write=False,
            should_promote=False,
            requires_review=False,
            candidate_question=candidate_question,
            candidate_answer=candidate_answer,
        )

    @classmethod
    def conflicting_answer(
        cls,
        *,
        candidate_memory_id: str,
        matched_records: list[MemoryConflictRecord],
        candidate_question: Optional[str] = None,
        candidate_answer: Optional[str] = None,
        message: str = "Conflicting answer detected.",
    ) -> "ConflictCheckResult":
        """
        Candidate conflicts with existing memory.

        Default behavior:
        - write private memory for audit/review
        - do not promote automatically
        - review required
        """

        return cls(
            candidate_memory_id=candidate_memory_id,
            conflict_type=ConflictType.CONFLICTING_ANSWER,
            severity=ConflictSeverity.HIGH,
            decision=ConflictDecision.REQUIRE_REVIEW,
            matched_records=matched_records,
            message=message,
            should_write=True,
            should_promote=False,
            requires_review=True,
            candidate_question=candidate_question,
            candidate_answer=candidate_answer,
        )

    @classmethod
    def outdated(
        cls,
        *,
        candidate_memory_id: str,
        matched_records: list[MemoryConflictRecord],
        candidate_question: Optional[str] = None,
        candidate_answer: Optional[str] = None,
        message: str = "Potential outdated memory detected.",
    ) -> "ConflictCheckResult":
        """
        Candidate may supersede older memory.

        This is reserved for later lifecycle logic.
        """

        return cls(
            candidate_memory_id=candidate_memory_id,
            conflict_type=ConflictType.OUTDATED,
            severity=ConflictSeverity.MEDIUM,
            decision=ConflictDecision.REQUIRE_REVIEW,
            matched_records=matched_records,
            message=message,
            should_write=True,
            should_promote=False,
            requires_review=True,
            candidate_question=candidate_question,
            candidate_answer=candidate_answer,
        )


class GovernedMemoryLifecycleResult(BaseModel):
    """
    Final result of processing a QA output through the governed memory lifecycle.

    This will be returned by governed_memory_lifecycle.py later.
    """

    success: bool = Field(
        default=False,
        description="Whether the lifecycle process completed successfully."
    )

    candidate_memory_id: Optional[str] = Field(
        default=None,
        description="ID of the generated candidate memory."
    )

    private_memory_written: bool = Field(
        default=False,
        description="Whether the candidate was written as private memory."
    )

    private_memory_id: Optional[str] = Field(
        default=None,
        description="ID of the formally persisted private memory.",
    )

    conflict_result: Optional[ConflictCheckResult] = Field(
        default=None,
        description="Conflict detection result."
    )

    promotion_request_id: Optional[str] = Field(
        default=None,
        description="ID of the promotion request, if created."
    )

    promotion_status: Optional[str] = Field(
        default=None,
        description="Status of the promotion request."
    )

    shared_memory_id: Optional[str] = Field(
        default=None,
        description="ID of the shared memory after promotion."
    )

    operation_log_count: Optional[int] = Field(
        default=None,
        description="Number of operation log records after lifecycle processing."
    )

    message: str = Field(
        default="",
        description="Human-readable lifecycle result message."
    )

    @field_validator("message")
    @classmethod
    def clean_message(cls, value: str) -> str:
        return value.strip()