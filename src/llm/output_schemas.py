from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


def _clean_string_list(values: list[str]) -> list[str]:
    """
    Strip whitespace, remove empty values, and preserve order while deduplicating.
    """
    cleaned: list[str] = []

    for value in values:
        item = str(value).strip()
        if item and item not in cleaned:
            cleaned.append(item)

    return cleaned


AnswerStatus = Literal[
    "answered",
    "insufficient_evidence",
    "refused",
]


class AgentAnswer(BaseModel):
    """
    Structured outcome produced by a task-oriented agent.

    The schema explicitly distinguishes a successful answer from an
    evidence-based abstention or a refusal. An empty string is never used as an
    implicit status signal.

    Workflow-level code remains responsible for checking that referenced
    memories exist, are accessible to the responder, and map to the declared
    source and contributing-agent identifiers.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"] = Field(
        default="2.0",
        description="Version of the AgentAnswer data contract.",
    )

    status: AnswerStatus = Field(
        description=(
            "Outcome of the answering attempt. Use 'answered' only when the "
            "accessible evidence supports a final answer; use "
            "'insufficient_evidence' when required information is unavailable; "
            "use 'refused' only when the task must not be answered."
        )
    )

    answer: str | None = Field(
        default=None,
        description=(
            "Final answer items only when status='answered'. For multiple "
            "items, use a comma followed by one space and preserve the order of "
            "the corresponding subjects in the question. Do not include "
            "explanations, labels, prefixes, bullets, quotation marks, brackets, "
            "or full sentences. Must be null for non-answer statuses."
        ),
    )

    reasoning: str = Field(
        description=(
            "Brief rationale for the selected status. For 'answered', explain "
            "how the accessible evidence supports the answer. For "
            "'insufficient_evidence', explain why the available evidence is "
            "incomplete. For 'refused', explain the refusal."
        )
    )

    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the selected status and its associated content, "
            "between 0.0 and 1.0. For 'answered', this is confidence in the "
            "answer; otherwise, it is confidence in the abstention or refusal."
        ),
    )

    missing_information: list[str] = Field(
        default_factory=list,
        description=(
            "Specific information still required to answer the task. This must "
            "be non-empty only when status='insufficient_evidence'."
        ),
    )

    used_memory_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Exact IDs of accessible private or shared memories materially used "
            "in the response. These identifiers must be validated by the "
            "workflow against the current run and responder permissions."
        ),
    )

    supporting_source_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Exact source identifiers associated with the used memories. These "
            "are retained for compatibility and must be verified or derived "
            "deterministically by workflow-level code."
        ),
    )

    contributing_agent_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Exact agent identifiers associated with the used memories. These "
            "are retained for compatibility and must be verified or derived "
            "deterministically by workflow-level code."
        ),
    )

    @field_validator("answer", mode="before")
    @classmethod
    def clean_optional_answer(cls, value: Any) -> str | None:
        if value is None:
            return None

        text = " ".join(str(value).split())
        return text or None

    @field_validator("reasoning", mode="before")
    @classmethod
    def reasoning_must_not_be_empty(cls, value: Any) -> str:
        text = " ".join(str(value or "").split())

        if not text:
            raise ValueError("reasoning cannot be empty.")

        return text

    @field_validator(
        "missing_information",
        "used_memory_ids",
        "supporting_source_ids",
        "contributing_agent_ids",
        mode="before",
    )
    @classmethod
    def clean_string_lists(cls, value: Any) -> list[str]:
        if value is None:
            return []

        if not isinstance(value, list):
            raise TypeError("Identifier and information fields must be lists.")

        return _clean_string_list(value)

    @model_validator(mode="after")
    def validate_status_contract(self) -> "AgentAnswer":
        if self.status == "answered":
            if self.answer is None:
                raise ValueError(
                    "answer must be non-empty when status='answered'."
                )

            if self.missing_information:
                raise ValueError(
                    "missing_information must be empty when status='answered'."
                )

            return self

        if self.answer is not None:
            raise ValueError(
                "answer must be null when status is not 'answered'."
            )

        if self.status == "insufficient_evidence":
            if not self.missing_information:
                raise ValueError(
                    "missing_information must be non-empty when "
                    "status='insufficient_evidence'."
                )

            return self

        if self.status == "refused":
            if self.missing_information:
                raise ValueError(
                    "missing_information must be empty when status='refused'."
                )

            if (
                self.used_memory_ids
                or self.supporting_source_ids
                or self.contributing_agent_ids
            ):
                raise ValueError(
                    "Evidence identifier fields must be empty when "
                    "status='refused'."
                )

            return self

        raise ValueError(f"Unsupported answer status: {self.status!r}.")


class ExtractedMemory(BaseModel):
    """
    One memory candidate extracted from an agent's local information.

    The memory is still a candidate at this stage. Ownership, persistence,
    visibility, and promotion are handled by the memory service and governance
    workflow rather than by this schema.
    """

    content: str = Field(
        description="Atomic, self-contained fact or reusable piece of information."
    )

    subject: str | None = Field(
        default=None,
        description="Main entity or topic described by the memory.",
    )

    memory_type: Literal["semantic", "episodic", "procedural"] = Field(
        default="semantic",
        description="High-level category of the extracted memory.",
    )

    source_ids: list[str] = Field(
        default_factory=list,
        description="IDs of source messages, dialogue turns, or documents supporting the memory.",
    )

    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Estimated usefulness of retaining this memory.",
    )

    shareable: bool = Field(
        default=True,
        description="Whether the content may be considered for promotion to shared memory.",
    )

    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence that the extracted memory is supported by the source.",
    )

    @field_validator("content")
    @classmethod
    def content_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("content cannot be empty.")

        return value

    @field_validator("subject")
    @classmethod
    def clean_optional_subject(cls, value: str | None) -> str | None:
        if value is None:
            return None

        value = value.strip()
        return value or None

    @field_validator("source_ids")
    @classmethod
    def clean_source_ids(cls, value: list[str]) -> list[str]:
        return _clean_string_list(value)


class MemoryExtractionOutput(BaseModel):
    """
    Structured output returned when an agent converts local information into
    one or more memory candidates.
    """

    agent_id: str = Field(
        description="ID of the agent that extracted the memories."
    )

    memories: list[ExtractedMemory] = Field(
        default_factory=list,
        description="Memory candidates extracted from the supplied information.",
    )

    extraction_summary: str | None = Field(
        default=None,
        description="Optional short summary of what was extracted.",
    )

    @field_validator("agent_id")
    @classmethod
    def agent_id_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("agent_id cannot be empty.")

        return value

    @field_validator("extraction_summary")
    @classmethod
    def clean_optional_summary(cls, value: str | None) -> str | None:
        if value is None:
            return None

        value = value.strip()
        return value or None


class MemoryReviewOutput(BaseModel):
    """
    Structured review produced by the critic for one promotion candidate.
    """

    memory_id: str = Field(
        description="ID of the candidate memory under review."
    )

    classification: Literal[
        "new",
        "duplicate",
        "conflict",
        "outdated",
        "irrelevant",
        "policy_violation",
        "uncertain",
    ] = Field(
        description="Primary classification assigned to the candidate memory."
    )

    recommendation: Literal[
        "approve",
        "reject",
        "merge",
        "supersede",
        "keep_private",
    ] = Field(
        description="Recommended governance action."
    )

    relevant: bool = Field(
        description="Whether the candidate is relevant to the current task or sharing request."
    )

    policy_compliant: bool = Field(
        description="Whether sharing the candidate would comply with the active policy."
    )

    related_memory_ids: list[str] = Field(
        default_factory=list,
        description="Existing memories associated with a duplicate, conflict, merge, or supersession decision.",
    )

    reason: str = Field(
        description="Brief justification for the classification and recommendation."
    )

    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )

    @field_validator("memory_id", "reason")
    @classmethod
    def required_text_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Required text field cannot be empty.")

        return value

    @field_validator("related_memory_ids")
    @classmethod
    def clean_related_memory_ids(cls, value: list[str]) -> list[str]:
        return _clean_string_list(value)


class MemoryAccessDecisionOutput(BaseModel):
    """
    Structured decision describing the initial access policy
    assigned to an existing memory.

    This schema only represents the decision. It does not modify
    MemoryMetadata or persist any access-policy changes.
    """

    memory_id: str = Field(
        description=(
            "ID of the memory to which the access policy applies."
        )
    )

    target_scope: Literal[
        "private",
        "shared",
    ] = Field(
        description=(
            "Visibility scope to assign to the memory."
        )
    )

    allowed_agent_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Additional non-owner agents allowed to read the memory "
            "when target_scope is 'shared'. Use ['*'] for global "
            "read access. The memory owner is added automatically "
            "by the access-policy service."
        ),
    )

    review_required: bool = Field(
        default=False,
        description=(
            "Whether the proposed access policy requires additional "
            "review before it is applied."
        ),
    )

    reason: str = Field(
        description=(
            "Brief explanation for the proposed access policy."
        )
    )

    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the access-policy decision."
        ),
    )

    @field_validator(
        "memory_id",
        "reason",
    )
    @classmethod
    def required_text_must_not_be_empty(
        cls,
        value: str,
    ) -> str:
        cleaned = str(value).strip()

        if not cleaned:
            raise ValueError(
                "Required access-decision fields cannot be empty."
            )

        return cleaned

    @field_validator(
        "allowed_agent_ids",
    )
    @classmethod
    def clean_allowed_agent_ids(
        cls,
        value: list[str],
    ) -> list[str]:
        return _clean_string_list(value)

    @model_validator(mode="after")
    def validate_access_policy(
        self,
    ) -> "MemoryAccessDecisionOutput":
        if (
            self.target_scope == "private"
            and self.allowed_agent_ids
        ):
            raise ValueError(
                "allowed_agent_ids must be empty when "
                "target_scope='private'."
            )

        if (
            self.target_scope == "shared"
            and not self.allowed_agent_ids
        ):
            raise ValueError(
                "A shared access policy must specify at least "
                "one allowed agent ID or the '*' wildcard."
            )

        if (
            "*" in self.allowed_agent_ids
            and len(self.allowed_agent_ids) > 1
        ):
            raise ValueError(
                "The '*' wildcard cannot be combined with "
                "specific agent IDs."
            )

        return self


class MemoryAccessReviewOutput(BaseModel):
    """
    Structured Critic review of a proposed initial memory-access policy.

    This output is advisory. The Coordinator retains final authority and the
    review itself must not modify the persisted memory ACL.
    """

    memory_id: str = Field(
        description="ID of the memory whose proposed access policy was reviewed."
    )

    recommendation: Literal[
        "approve",
        "revise",
        "reject",
    ] = Field(
        description=(
            "Critic recommendation for the proposed policy. "
            "'approve' accepts it, 'revise' proposes a safer alternative, "
            "and 'reject' recommends owner-only private access."
        )
    )

    policy_compliant: bool = Field(
        description="Whether the proposed policy complies with the active governance policy."
    )

    least_privilege_satisfied: bool = Field(
        description=(
            "Whether the proposed readers are limited to the minimum set "
            "required by the current task or policy."
        )
    )

    risk_labels: list[str] = Field(
        default_factory=list,
        description=(
            "Compact risk labels identified during review, such as "
            "'sensitive_content', 'over_sharing', or 'unknown_reader'."
        ),
    )

    suggested_scope: Literal[
        "private",
        "shared",
    ] = Field(
        description="Scope recommended by the Critic after review."
    )

    suggested_agent_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Additional non-owner readers recommended when "
            "suggested_scope='shared'. Use ['*'] only for global access."
        ),
    )

    reason: str = Field(
        description="Brief explanation of the Critic's recommendation."
    )

    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )

    @field_validator(
        "memory_id",
        "reason",
    )
    @classmethod
    def required_review_text_must_not_be_empty(
        cls,
        value: str,
    ) -> str:
        cleaned = str(value).strip()

        if not cleaned:
            raise ValueError(
                "Required memory-access review fields cannot be empty."
            )

        return cleaned

    @field_validator(
        "risk_labels",
        "suggested_agent_ids",
    )
    @classmethod
    def clean_review_lists(
        cls,
        value: list[str],
    ) -> list[str]:
        return _clean_string_list(value)

    @model_validator(mode="after")
    def validate_suggested_policy(
        self,
    ) -> "MemoryAccessReviewOutput":
        if (
            self.suggested_scope == "private"
            and self.suggested_agent_ids
        ):
            raise ValueError(
                "suggested_agent_ids must be empty when "
                "suggested_scope='private'."
            )

        if (
            self.suggested_scope == "shared"
            and not self.suggested_agent_ids
        ):
            raise ValueError(
                "A shared suggested policy must specify at least "
                "one Agent ID or the '*' wildcard."
            )

        if (
            "*" in self.suggested_agent_ids
            and len(self.suggested_agent_ids) > 1
        ):
            raise ValueError(
                "The '*' wildcard cannot be combined with specific "
                "Agent IDs."
            )

        if self.recommendation == "reject":
            if self.suggested_scope != "private":
                raise ValueError(
                    "A rejected initial policy must recommend "
                    "suggested_scope='private'."
                )

            if self.suggested_agent_ids:
                raise ValueError(
                    "A rejected initial policy cannot recommend "
                    "additional readers."
                )

        return self


class MemoryAccessRequestDecisionOutput(BaseModel):
    """
    Structured Coordinator decision for a runtime memory-access request.

    This object records what should happen. MemorySharingGateway performs the
    actual approve or reject operation.
    """

    request_id: str = Field(
        description="ID of the access request being evaluated."
    )

    memory_id: str = Field(
        description="ID of the memory referenced by the request."
    )

    approved: bool = Field(
        description="Whether the Coordinator proposes approving the request."
    )

    review_required: bool = Field(
        default=False,
        description=(
            "Whether Critic review is required before the decision "
            "may be executed."
        ),
    )

    reason: str = Field(
        description="Brief explanation of the proposed access-request decision."
    )

    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )

    @field_validator(
        "request_id",
        "memory_id",
        "reason",
    )
    @classmethod
    def required_request_decision_text_must_not_be_empty(
        cls,
        value: str,
    ) -> str:
        cleaned = str(value).strip()

        if not cleaned:
            raise ValueError(
                "Required access-request decision fields cannot be empty."
            )

        return cleaned


class MemoryAccessRequestReviewOutput(BaseModel):
    """
    Structured Critic review of a proposed runtime access-request decision.

    The review is advisory. It does not approve, reject, or persist the
    underlying request.
    """

    request_id: str = Field(
        description="ID of the access request being reviewed."
    )

    memory_id: str = Field(
        description="ID of the memory referenced by the request."
    )

    recommendation: Literal[
        "approve",
        "reject",
    ] = Field(
        description="Critic recommendation for the access request."
    )

    request_justified: bool = Field(
        description=(
            "Whether the requester supplied a sufficiently specific and "
            "task-relevant reason for access."
        )
    )

    policy_compliant: bool = Field(
        description="Whether granting the request would comply with the active policy."
    )

    least_privilege_satisfied: bool = Field(
        description=(
            "Whether granting this request is consistent with least-privilege "
            "access rather than unnecessary broad sharing."
        )
    )

    risk_labels: list[str] = Field(
        default_factory=list,
        description=(
            "Compact risk labels identified during review, such as "
            "'insufficient_reason', 'sensitive_content', or 'role_mismatch'."
        ),
    )

    reason: str = Field(
        description="Brief explanation of the Critic's recommendation."
    )

    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )

    @field_validator(
        "request_id",
        "memory_id",
        "reason",
    )
    @classmethod
    def required_request_review_text_must_not_be_empty(
        cls,
        value: str,
    ) -> str:
        cleaned = str(value).strip()

        if not cleaned:
            raise ValueError(
                "Required access-request review fields cannot be empty."
            )

        return cleaned

    @field_validator("risk_labels")
    @classmethod
    def clean_request_review_risk_labels(
        cls,
        value: list[str],
    ) -> list[str]:
        return _clean_string_list(value)

    @model_validator(mode="after")
    def validate_request_recommendation(
        self,
    ) -> "MemoryAccessRequestReviewOutput":
        if self.recommendation == "approve":
            if not self.request_justified:
                raise ValueError(
                    "An approved recommendation requires "
                    "request_justified=True."
                )

            if not self.policy_compliant:
                raise ValueError(
                    "An approved recommendation requires "
                    "policy_compliant=True."
                )

            if not self.least_privilege_satisfied:
                raise ValueError(
                    "An approved recommendation requires "
                    "least_privilege_satisfied=True."
                )

        return self


class TaskRoutingOutput(BaseModel):
    """
    Structured routing decision produced for a multi-agent task.
    """

    selected_agent_ids: list[str] = Field(
        description="Agents whose local information may be relevant to the task."
    )

    responder_agent_id: str = Field(
        description="Agent responsible for producing the final task answer."
    )

    required_information: list[str] = Field(
        default_factory=list,
        description="Information items that must be gathered before the task can be answered.",
    )

    reason: str = Field(
        description="Brief explanation of the routing decision."
    )

    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )

    @field_validator("selected_agent_ids", "required_information")
    @classmethod
    def clean_list_fields(cls, value: list[str]) -> list[str]:
        return _clean_string_list(value)

    @field_validator("responder_agent_id", "reason")
    @classmethod
    def required_routing_text_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Required routing field cannot be empty.")

        return value

    @model_validator(mode="after")
    def responder_must_be_selected(self) -> "TaskRoutingOutput":
        if self.responder_agent_id not in self.selected_agent_ids:
            raise ValueError(
                "responder_agent_id must also appear in selected_agent_ids."
            )

        return self


class PromotionDecisionOutput(BaseModel):
    """
    Final promotion decision produced by the coordinator or deterministic
    governance controller.
    """

    memory_id: str = Field(
        description="ID of the memory candidate being decided."
    )

    decision: Literal[
        "approve",
        "reject",
        "merge",
        "supersede",
        "keep_private",
    ] = Field(
        description="Final governance action."
    )

    target_scope: Literal["private", "shared"] = Field(
        description="Visibility scope after the decision is applied."
    )

    allowed_agent_ids: list[str] = Field(
        default_factory=list,
        description="Agents allowed to read the memory when target_scope is shared.",
    )

    related_memory_ids: list[str] = Field(
        default_factory=list,
        description="Memories involved in a merge or supersession operation.",
    )

    reason: str = Field(
        description="Brief explanation of the final decision."
    )

    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )

    @field_validator("memory_id", "reason")
    @classmethod
    def required_decision_text_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Required decision field cannot be empty.")

        return value

    @field_validator("allowed_agent_ids", "related_memory_ids")
    @classmethod
    def clean_decision_identifier_lists(cls, value: list[str]) -> list[str]:
        return _clean_string_list(value)

    @model_validator(mode="after")
    def validate_scope_against_decision(self) -> "PromotionDecisionOutput":
        if self.decision in {"reject", "keep_private"} and self.target_scope != "private":
            raise ValueError(
                "Rejected or private-only memories must keep target_scope='private'."
            )

        if self.target_scope == "private" and self.allowed_agent_ids:
            raise ValueError(
                "allowed_agent_ids must be empty when target_scope='private'."
            )

        return self


class LLMCallMetadata(BaseModel):
    """
    Optional metadata for logging, debugging, and experiment analysis.
    """

    agent_id: str
    role: str | None = None
    model_name: str

    prompt_preview: str | None = None
    raw_output: str | None = None

    success: bool = True
    error_message: str | None = None

    latency_ms: float | None = Field(default=None, ge=0.0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)

    @field_validator("agent_id", "model_name")
    @classmethod
    def required_metadata_text_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Required metadata field cannot be empty.")

        return value

    @field_validator("role", "prompt_preview", "raw_output", "error_message")
    @classmethod
    def clean_optional_metadata_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        value = value.strip()
        return value or None