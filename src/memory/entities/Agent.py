from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    COORDINATOR = "coordinator"
    CRITIC = "critic"
    WORKER = "worker"


class Agent(BaseModel):
    """
    Represents an agent in the multi-agent memory system.

    The role determines the default permissions for memory operations.
    """

    agent_id: str = Field(..., description="Unique identifier of the agent.")
    name: str = Field(..., description="Human-readable name of the agent.")

    role: AgentRole = Field(..., description="Role of the agent.")

    can_read_shared: bool = Field(
        default=True,
        description="Whether the agent can read shared memories."
    )
    can_write_private: bool = Field(
        default=True,
        description="Whether the agent can write to its own private memory."
    )
    can_write_shared: bool = Field(
        default=False,
        description="Whether the agent can directly write shared memories."
    )
    can_review_memory: bool = Field(
        default=False,
        description="Whether the agent can review memory promotion or conflicts."
    )

    description: Optional[str] = Field(
        default=None,
        description="Optional description of the agent's responsibility."
    )

    @classmethod
    def coordinator(cls, agent_id: str = "agent_coordinator") -> "Agent":
        return cls(
            agent_id=agent_id,
            name="Coordinator Agent",
            role=AgentRole.COORDINATOR,
            can_read_shared=True,
            can_write_private=True,
            can_write_shared=True,
            can_review_memory=True,
            description="Global decision maker for shared memory governance."
        )

    @classmethod
    def critic(cls, agent_id: str = "agent_critic") -> "Agent":
        return cls(
            agent_id=agent_id,
            name="Critic Agent",
            role=AgentRole.CRITIC,
            can_read_shared=True,
            can_write_private=True,
            can_write_shared=False,
            can_review_memory=True,
            description="Evaluates memory quality, duplication, and conflicts."
        )

    @classmethod
    def worker(cls, agent_id: str, name: Optional[str] = None) -> "Agent":
        return cls(
            agent_id=agent_id,
            name=name or agent_id,
            role=AgentRole.WORKER,
            can_read_shared=True,
            can_write_private=True,
            can_write_shared=False,
            can_review_memory=False,
            description="Task-specific worker agent."
        )