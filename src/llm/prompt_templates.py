from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel

from .output_schemas import (
    AgentAnswer,
    ExtractedMemory,
    MemoryExtractionOutput,
    MemoryReviewOutput,
    PromotionDecisionOutput,
    TaskRoutingOutput,
)


SchemaT = TypeVar("SchemaT", bound=BaseModel)


def _clean_lines(values: Sequence[str] | None) -> list[str]:
    """
    Normalize a sequence of prompt instructions while preserving order.
    """
    cleaned: list[str] = []

    for value in values or []:
        item = str(value).strip()
        if item and item not in cleaned:
            cleaned.append(item)

    return cleaned


def _format_bullets(values: Sequence[str]) -> str:
    """
    Render prompt instructions as Markdown bullets.
    """
    return "\n".join(f"- {value}" for value in values)


def _format_value(value: Any) -> str:
    """
    Convert common Python and Pydantic values into readable prompt text.
    """
    if value is None:
        return "Not provided."

    if isinstance(value, BaseModel):
        return value.model_dump_json(indent=2)

    if isinstance(value, str):
        text = value.strip()
        return text or "Not provided."

    if isinstance(value, Mapping):
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return json.dumps(list(value), indent=2, ensure_ascii=False, default=str)

    return str(value)


def _format_context_sections(
    sections: Mapping[str, Any] | None,
) -> str:
    """
    Render arbitrary named context sections.

    This keeps prompt templates independent of a specific dataset, retriever,
    workflow state, or external-knowledge schema.
    """
    if not sections:
        return "No additional context."

    rendered: list[str] = []

    for heading, value in sections.items():
        title = str(heading).strip()
        if not title:
            continue

        rendered.append(f"{title}:\n{_format_value(value)}")

    return "\n\n".join(rendered) if rendered else "No additional context."


def _schema_hint(schema_model: type[SchemaT]) -> str:
    """
    Return the Pydantic JSON schema used by structured-output calls.
    """
    return json.dumps(
        schema_model.model_json_schema(),
        indent=2,
        ensure_ascii=False,
    )


class _BasePromptTemplate:
    """
    Shared constructor logic for role-specific prompt templates.

    additional_tasks and additional_rules are attached when the template
    instance is created. This allows each agent instance to add role-specific
    behaviour without duplicating the core prompt.
    """

    def __init__(
        self,
        *,
        additional_tasks: Sequence[str] | None = None,
        additional_rules: Sequence[str] | None = None,
    ) -> None:
        self.additional_tasks = _clean_lines(additional_tasks)
        self.additional_rules = _clean_lines(additional_rules)

    def _compose_tasks(
        self,
        required_tasks: Sequence[str],
    ) -> list[str]:
        return _clean_lines([*required_tasks, *self.additional_tasks])

    def _compose_rules(
        self,
        required_rules: Sequence[str],
    ) -> list[str]:
        return _clean_lines([*required_rules, *self.additional_rules])


class WorkerPromptTemplate(_BasePromptTemplate):
    """
    Generic prompt template shared by all task agents.

    Alice, Bob, Charlie, Dave, or any future worker can use the same template.
    Agent-specific behaviour is supplied through role_description,
    additional_tasks, and additional_rules at construction time.
    """

    _CORE_RULES = (
        "Use only the assigned input, the permission-filtered accessible memory, and the explicitly provided context.",
        "Do not claim access to hidden, private, or unavailable information.",
        "Do not request or invent memory content that is not shown.",
        "Keep the answer direct and keep the reasoning brief.",
        "Record only memory IDs and source IDs that were actually used.",
        "Follow the required structured-output schema exactly.",
    )

    def __init__(
        self,
        *,
        agent_id: str,
        role_description: str | None = None,
        additional_tasks: Sequence[str] | None = None,
        additional_rules: Sequence[str] | None = None,
    ) -> None:
        super().__init__(
            additional_tasks=additional_tasks,
            additional_rules=additional_rules,
        )

        agent_id = agent_id.strip()
        if not agent_id:
            raise ValueError("agent_id cannot be empty.")

        self.agent_id = agent_id
        self.role_description = (
            role_description.strip()
            if role_description and role_description.strip()
            else "Complete assigned tasks as a task-oriented agent."
        )

    def build_answer_prompt(
        self,
        *,
        task: str,
        task_input: str | None = None,
        accessible_memory_context: str | None = None,
        context_sections: Mapping[str, Any] | None = None,
    ) -> str:
        """
        Build a general task-answering prompt returning AgentAnswer.
        """
        required_tasks = (
            "Complete the assigned task using the available evidence.",
            "Produce one final answer and identify the memories, sources, and agents that materially contributed to it.",
        )

        answer_rules = (
            *self._CORE_RULES,

            "The answer field must contain only the final answer items required "
            "by the task.",

            "When the question requires multiple answer items, separate them "
            "using a comma followed by one space.",

            "Return answer items in the same order as the corresponding people "
            "or subjects appear in the question.",

            "Do not use sentences, labels, bullet points, explanations, quotation "
            "marks, brackets, or answer prefixes in the answer field.",

            "Use commas, not 'and', '&', semicolons, or line breaks, to separate "
            "different answer items. Do not remove 'and' or '&' when it is part "
            "of an answer item's proper name.",

            "Do not repeat the same answer item.",

            "Put all explanation and justification in the reasoning field, "
            "never in the answer field.",

            "Use only answer items supported by the accessible memory or other "
            "explicitly supplied context.",

            "For every accessible memory that materially supports the answer, "
            "copy its exact memory_id into used_memory_ids.",

            "For every used memory, copy its displayed source_ids into "
            "supporting_source_ids. Do not invent or modify source IDs.",

            "For every used memory, copy its displayed owner_agent_id into "
            "contributing_agent_ids. Do not include agents whose memories were "
            "not materially used.",

            "Use the exact identifiers shown in the accessible memory context. "
            "Do not use memory numbering such as 'Memory 1' unless that is the "
            "actual displayed memory_id.",
        )

        tasks = self._compose_tasks(required_tasks)
        rules = self._compose_rules(answer_rules)

        return f"""
You are {self.agent_id}, a task agent in a multi-agent system.

Role:
{self.role_description}

Tasks:
{_format_bullets(tasks)}

Important rules:
{_format_bullets(rules)}

Assigned task:
{_format_value(task)}

Task input:
{_format_value(task_input)}

Accessible memory:
{_format_value(accessible_memory_context)}

Additional context:
{_format_context_sections(context_sections)}

Required output schema:
{_schema_hint(AgentAnswer)}
""".strip()

    def build_memory_extraction_prompt(
        self,
        *,
        source_content: str,
        source_ids: Sequence[str] | None = None,
        task_context: str | None = None,
        context_sections: Mapping[str, Any] | None = None,
    ) -> str:
        """
        Build a prompt that converts local information into memory candidates.
        """
        required_tasks = (
            "Extract reusable memories from the supplied local information.",
            "Represent each memory as an atomic, self-contained statement.",
        )

        extraction_rules = (
            *self._CORE_RULES,
            "Extract only information directly supported by the supplied source.",
            "Do not combine unrelated facts into one memory.",
            "Preserve the original meaning and avoid adding unsupported detail.",
            "Use source_ids only from the source identifiers provided below.",
            "Mark a memory as shareable only when its content is suitable for later controlled sharing.",
        )

        tasks = self._compose_tasks(required_tasks)
        rules = self._compose_rules(extraction_rules)

        extraction_context = {
            "Source IDs": list(source_ids or []),
            "Task context": task_context,
            **dict(context_sections or {}),
        }

        return f"""
You are {self.agent_id}, a task agent preparing local information for memory storage.

Role:
{self.role_description}

Tasks:
{_format_bullets(tasks)}

Important rules:
{_format_bullets(rules)}

Local source content:
{_format_value(source_content)}

Extraction context:
{_format_context_sections(extraction_context)}

Required output schema:
{_schema_hint(MemoryExtractionOutput)}
""".strip()


class CriticPromptTemplate(_BasePromptTemplate):
    """
    Prompt template for memory-governance review.

    The critic's essential review duties and safety rules are fixed. Extra
    tasks and rules may be added when creating a critic instance.
    """

    _CORE_TASKS = (
        "Review one candidate memory for the requested governance action.",
        "Classify the candidate and recommend an appropriate action.",
    )

    _CORE_RULES = (
        "Base the review only on the candidate memory, supplied policy, task context, and existing-memory context.",
        "Check relevance, policy compliance, duplication, conflict, and temporal validity.",
        "Do not invent facts, policies, timestamps, or related memory IDs.",
        "Do not rewrite, persist, promote, delete, merge, or supersede any memory.",
        "The critic provides a recommendation only; the coordinator retains final authority.",
        "Use related_memory_ids only for memories explicitly present in the supplied context.",
        "Explain the recommendation briefly and follow the required structured-output schema exactly.",
    )

    def __init__(
        self,
        *,
        critic_id: str = "critic",
        additional_tasks: Sequence[str] | None = None,
        additional_rules: Sequence[str] | None = None,
    ) -> None:
        super().__init__(
            additional_tasks=additional_tasks,
            additional_rules=additional_rules,
        )

        critic_id = critic_id.strip()
        if not critic_id:
            raise ValueError("critic_id cannot be empty.")

        self.critic_id = critic_id

    def build_memory_review_prompt(
        self,
        *,
        candidate_memory_id: str,
        candidate_memory: ExtractedMemory | BaseModel | Mapping[str, Any] | str,
        requested_action: str,
        requester_agent_id: str,
        target_agent_ids: Sequence[str] | None = None,
        task_context: str | None = None,
        policy_context: Any = None,
        existing_memory_context: Any = None,
        context_sections: Mapping[str, Any] | None = None,
    ) -> str:
        tasks = self._compose_tasks(self._CORE_TASKS)
        rules = self._compose_rules(self._CORE_RULES)

        review_context = {
            "Candidate memory ID": candidate_memory_id,
            "Candidate memory": candidate_memory,
            "Requested action": requested_action,
            "Requester agent ID": requester_agent_id,
            "Target agent IDs": list(target_agent_ids or []),
            "Task context": task_context,
            "Policy context": policy_context,
            "Existing memory context": existing_memory_context,
            **dict(context_sections or {}),
        }

        return f"""
You are {self.critic_id}, the memory-governance critic in a multi-agent system.

Tasks:
{_format_bullets(tasks)}

Important rules:
{_format_bullets(rules)}

Review context:
{_format_context_sections(review_context)}

Required output schema:
{_schema_hint(MemoryReviewOutput)}
""".strip()


class CoordinatorPromptTemplate(_BasePromptTemplate):
    """
    Prompt template for coordinator operations.

    The same coordinator instance can build task-routing prompts and final
    memory-promotion decision prompts. Essential coordinator rules are fixed,
    while extra tasks and rules can be attached at construction time.
    """

    _COMMON_RULES = (
        "Use only the agents, memories, policies, reviews, and context explicitly provided.",
        "Do not expose private memory content to an agent that is not authorised to receive it.",
        "Do not invent agent capabilities, permissions, memories, or policy conditions.",
        "Keep the decision explanation brief and follow the required structured-output schema exactly.",
    )

    _ROUTING_TASKS = (
        "Select the agents whose information or capabilities are necessary for the assigned task.",
        "Choose one selected agent to produce the final task response.",
    )

    _ROUTING_RULES = (
        "Select only agents that are relevant to the task.",
        "The responder must be included in selected_agent_ids.",
        "Do not route private information itself; route only the task and authorised context.",
        "List the information that must be gathered without fabricating unavailable facts.",
    )

    _PROMOTION_TASKS = (
        "Make the final governance decision for the candidate memory.",
        "Determine the resulting scope, authorised readers, and any related-memory operation.",
    )

    _PROMOTION_RULES = (
        "Apply the active policy even when the critic recommends approval.",
        "Treat the critic output as advice rather than an instruction.",
        "A rejected or keep-private candidate must remain private.",
        "A shared candidate must have an explicit allowed_agent_ids list.",
        "Use merge or supersede only when the supplied context identifies valid related memories.",
        "Do not alter the candidate's factual content as part of the decision.",
    )

    def __init__(
        self,
        *,
        coordinator_id: str = "coordinator",
        additional_tasks: Sequence[str] | None = None,
        additional_rules: Sequence[str] | None = None,
    ) -> None:
        super().__init__(
            additional_tasks=additional_tasks,
            additional_rules=additional_rules,
        )

        coordinator_id = coordinator_id.strip()
        if not coordinator_id:
            raise ValueError("coordinator_id cannot be empty.")

        self.coordinator_id = coordinator_id

    def build_task_routing_prompt(
        self,
        *,
        task: str,
        available_agents: Mapping[str, Any] | Sequence[str],
        task_context: str | None = None,
        context_sections: Mapping[str, Any] | None = None,
    ) -> str:
        tasks = self._compose_tasks(self._ROUTING_TASKS)
        rules = self._compose_rules(
            [*self._COMMON_RULES, *self._ROUTING_RULES]
        )

        routing_context = {
            "Assigned task": task,
            "Available agents": available_agents,
            "Task context": task_context,
            **dict(context_sections or {}),
        }

        return f"""
You are {self.coordinator_id}, the task coordinator in a multi-agent system.

Tasks:
{_format_bullets(tasks)}

Important rules:
{_format_bullets(rules)}

Routing context:
{_format_context_sections(routing_context)}

Required output schema:
{_schema_hint(TaskRoutingOutput)}
""".strip()

    def build_promotion_decision_prompt(
        self,
        *,
        candidate_memory_id: str,
        candidate_memory: ExtractedMemory | BaseModel | Mapping[str, Any] | str,
        promotion_request: Any,
        critic_review: MemoryReviewOutput | BaseModel | Mapping[str, Any] | str,
        policy_context: Any,
        existing_memory_context: Any = None,
        task_context: str | None = None,
        context_sections: Mapping[str, Any] | None = None,
    ) -> str:
        tasks = self._compose_tasks(self._PROMOTION_TASKS)
        rules = self._compose_rules(
            [*self._COMMON_RULES, *self._PROMOTION_RULES]
        )

        decision_context = {
            "Candidate memory ID": candidate_memory_id,
            "Candidate memory": candidate_memory,
            "Promotion request": promotion_request,
            "Critic review": critic_review,
            "Policy context": policy_context,
            "Existing memory context": existing_memory_context,
            "Task context": task_context,
            **dict(context_sections or {}),
        }

        return f"""
You are {self.coordinator_id}, the final memory-governance authority in a multi-agent system.

Tasks:
{_format_bullets(tasks)}

Important rules:
{_format_bullets(rules)}

Decision context:
{_format_context_sections(decision_context)}

Required output schema:
{_schema_hint(PromotionDecisionOutput)}
""".strip()


class PromptTemplates:
    """
    Small factory facade for constructing reusable prompt-template instances.

    Example:
        alice_prompts = PromptTemplates.worker(
            agent_id="alice_agent",
            role_description="Represents Alice and manages Alice's private memory.",
            additional_rules=["Share only memories relevant to the active task."],
        )
    """

    @staticmethod
    def worker(
        *,
        agent_id: str,
        role_description: str | None = None,
        additional_tasks: Sequence[str] | None = None,
        additional_rules: Sequence[str] | None = None,
    ) -> WorkerPromptTemplate:
        return WorkerPromptTemplate(
            agent_id=agent_id,
            role_description=role_description,
            additional_tasks=additional_tasks,
            additional_rules=additional_rules,
        )

    @staticmethod
    def critic(
        *,
        critic_id: str = "critic",
        additional_tasks: Sequence[str] | None = None,
        additional_rules: Sequence[str] | None = None,
    ) -> CriticPromptTemplate:
        return CriticPromptTemplate(
            critic_id=critic_id,
            additional_tasks=additional_tasks,
            additional_rules=additional_rules,
        )

    @staticmethod
    def coordinator(
        *,
        coordinator_id: str = "coordinator",
        additional_tasks: Sequence[str] | None = None,
        additional_rules: Sequence[str] | None = None,
    ) -> CoordinatorPromptTemplate:
        return CoordinatorPromptTemplate(
            coordinator_id=coordinator_id,
            additional_tasks=additional_tasks,
            additional_rules=additional_rules,
        )