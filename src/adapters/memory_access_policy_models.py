from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class AccessPolicyScope(str, Enum):
    """
    Stable memory-scope values exposed to workflow code.
    """

    PRIVATE = "private"
    SHARED = "shared"


class MemoryAccessPolicyResult(BaseModel):
    """
    Workflow-facing result of a persisted memory access-policy operation.

    This model represents the ACL state after MemoryAccessPolicyService has
    completed an operation. It is intentionally independent of MemoryItem and
    MemoryMetadata so workflow code does not depend on persistence entities.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    memory_id: str = Field(
        min_length=1,
        description=(
            "ID of the memory whose access policy was applied."
        ),
    )

    owner_agent_id: str = Field(
        min_length=1,
        description=(
            "ID of the Agent that owns the memory."
        ),
    )

    scope: AccessPolicyScope = Field(
        description=(
            "Persisted visibility scope of the memory."
        ),
    )

    readable_by: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Persisted read ACL. The '*' wildcard means globally readable."
        ),
    )

    writable_by: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Persisted write ACL."
        ),
    )

    changed: bool = Field(
        default=True,
        description=(
            "Whether the operation changed the persisted access policy."
        ),
    )

    operation_id: str | None = Field(
        default=None,
        description=(
            "Audit-log operation ID associated with the policy change, "
            "when one was created."
        ),
    )

    @field_validator(
        "memory_id",
        "owner_agent_id",
    )
    @classmethod
    def required_text_must_not_be_empty(
        cls,
        value: str,
    ) -> str:
        cleaned = str(value).strip()

        if not cleaned:
            raise ValueError(
                "Required access-policy result fields cannot be empty."
            )

        return cleaned

    @field_validator(
        "operation_id",
        mode="before",
    )
    @classmethod
    def clean_optional_operation_id(
        cls,
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        cleaned = str(value).strip()
        return cleaned or None

    @field_validator(
        "readable_by",
        "writable_by",
        mode="before",
    )
    @classmethod
    def clean_acl_values(
        cls,
        value: Any,
    ) -> tuple[str, ...]:
        if value is None:
            return ()

        if isinstance(value, str):
            values = [value]
        else:
            values = value

        cleaned_values: list[str] = []

        for item in values:
            cleaned = str(item).strip()

            if (
                cleaned
                and cleaned not in cleaned_values
            ):
                cleaned_values.append(cleaned)

        return tuple(cleaned_values)

    @model_validator(mode="after")
    def validate_access_policy(
        self,
    ) -> "MemoryAccessPolicyResult":
        owner_id = self.owner_agent_id

        if self.scope == AccessPolicyScope.PRIVATE:
            if self.readable_by != (owner_id,):
                raise ValueError(
                    "A private policy result must grant read access "
                    "only to the memory owner."
                )

            if self.writable_by != (owner_id,):
                raise ValueError(
                    "A private policy result must grant write access "
                    "only to the memory owner."
                )

            return self

        if not self.readable_by:
            raise ValueError(
                "A shared policy result must contain at least one reader."
            )

        if "*" in self.readable_by:
            if self.readable_by != ("*",):
                raise ValueError(
                    "The '*' wildcard cannot be combined with specific "
                    "Agent IDs in readable_by."
                )
        elif owner_id not in self.readable_by:
            raise ValueError(
                "The memory owner must remain in readable_by for a "
                "targeted shared policy."
            )

        if owner_id not in self.writable_by:
            raise ValueError(
                "The memory owner must remain in writable_by."
            )

        return self


__all__ = [
    "AccessPolicyScope",
    "MemoryAccessPolicyResult",
]