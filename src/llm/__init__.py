"""
Public interface for the generic LLM layer.

This package exposes:
- workflow-independent LLM clients;
- reusable prompt-template builders;
- generic structured-output parsing;
- shared Pydantic output schemas.
"""

from .llm_client import (
    LLMClient,
    LLMClientConfigurationError,
    LLMClientInvocationError,
)
from .output_parser import (
    LLMOutputParseError,
    LLMOutputParser,
)
from .output_schemas import (
    AgentAnswer,
    ExtractedMemory,
    LLMCallMetadata,
    MemoryExtractionOutput,
    MemoryReviewOutput,
    PromotionDecisionOutput,
    TaskRoutingOutput,
)
from .prompt_templates import (
    CoordinatorPromptTemplate,
    CriticPromptTemplate,
    PromptTemplates,
    WorkerPromptTemplate,
)


__all__ = [
    # Client
    "LLMClient",
    "LLMClientConfigurationError",
    "LLMClientInvocationError",

    # Parser
    "LLMOutputParser",
    "LLMOutputParseError",

    # Schemas
    "AgentAnswer",
    "ExtractedMemory",
    "MemoryExtractionOutput",
    "MemoryReviewOutput",
    "TaskRoutingOutput",
    "PromotionDecisionOutput",
    "LLMCallMetadata",

    # Prompt templates
    "PromptTemplates",
    "WorkerPromptTemplate",
    "CriticPromptTemplate",
    "CoordinatorPromptTemplate",
]