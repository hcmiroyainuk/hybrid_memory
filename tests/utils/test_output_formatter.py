from __future__ import annotations

import textwrap
from typing import Any, Iterable


DEFAULT_WRAP_WIDTH = 100
DEFAULT_MAX_LIST_ITEMS = 20


def wrap_text(
    text: Any,
    *,
    width: int = DEFAULT_WRAP_WIDTH,
    subsequent_indent: str = "",
) -> str:
    """
    Convert text to string and wrap it to a fixed line width.

    This function does not truncate content.
    """

    if text is None:
        return "None"

    value = str(text).replace("\n", " ").strip()

    if not value:
        return ""

    return textwrap.fill(
        value,
        width=width,
        subsequent_indent=subsequent_indent,
        break_long_words=False,
        break_on_hyphens=False,
    )


def format_list(
    items: Iterable[Any] | None,
    *,
    max_items: int = DEFAULT_MAX_LIST_ITEMS,
    width: int = DEFAULT_WRAP_WIDTH,
) -> str:
    """
    Format a list-like object in a compact readable way.

    This function does not truncate individual items.
    It wraps long lines instead.
    """

    if items is None:
        return "[]"

    if callable(items):
        return f"<callable: {getattr(items, '__name__', type(items).__name__)}>"

    values = list(items)

    if not values:
        return "[]"

    shown_values = values[:max_items]

    lines = ["["]

    for item in shown_values:
        wrapped_item = wrap_text(
            repr(item),
            width=width,
            subsequent_indent="  ",
        )
        lines.append(f"  {wrapped_item},")

    if len(values) > max_items:
        lines.append(f"  ... (+{len(values) - max_items} more)")

    lines.append("]")

    return "\n".join(lines)


def format_section(
    title: str,
    content: Any = "",
    *,
    width: int = 80,
) -> str:
    """
    Format a titled section.
    """

    line = "=" * width
    return f"\n{line}\n{title}\n{line}\n{content}"


def format_field(
    name: str,
    value: Any,
    *,
    width: int = DEFAULT_WRAP_WIDTH,
    indent: str = "  ",
) -> str:
    """
    Format one named field with wrapped value.
    """

    wrapped_value = wrap_text(
        value,
        width=width,
        subsequent_indent=indent,
    )

    return f"{name}: {wrapped_value}"


def format_agent_answer(
    output: Any,
    *,
    width: int = DEFAULT_WRAP_WIDTH,
) -> str:
    """
    Format AgentAnswer-like output.
    """

    if output is None:
        return "None"

    answer = getattr(output, "answer", None)
    reasoning = getattr(output, "reasoning", None)
    confidence = getattr(output, "confidence", None)
    evidence_chunk_ids = getattr(output, "evidence_chunk_ids", [])

    return "\n".join(
        [
            format_field("answer", answer, width=width),
            format_field("confidence", confidence, width=width),
            "evidence_chunk_ids:",
            format_list(evidence_chunk_ids, width=width),
            format_field("reasoning", reasoning, width=width),
        ]
    )


def format_critic_output(
    output: Any,
    *,
    width: int = DEFAULT_WRAP_WIDTH,
) -> str:
    """
    Format CriticOutput-like output.
    """

    if output is None:
        return "None"

    recommended_answer = getattr(output, "recommended_answer", None)
    preferred_worker = getattr(output, "preferred_worker", None)
    confidence = getattr(output, "confidence", None)
    comment = getattr(output, "comment", None)

    return "\n".join(
        [
            format_field("recommended_answer", recommended_answer, width=width),
            format_field("preferred_worker", preferred_worker, width=width),
            format_field("confidence", confidence, width=width),
            format_field("comment", comment, width=width),
        ]
    )


def format_coordinator_output(
    output: Any,
    *,
    width: int = DEFAULT_WRAP_WIDTH,
) -> str:
    """
    Format CoordinatorOutput-like output.
    """

    if output is None:
        return "None"

    final_answer = getattr(output, "final_answer", None)
    reasoning = getattr(output, "reasoning", None)
    confidence = getattr(output, "confidence", None)

    return "\n".join(
        [
            format_field("final_answer", final_answer, width=width),
            format_field("confidence", confidence, width=width),
            format_field("reasoning", reasoning, width=width),
        ]
    )


def format_memory_context_summary(
    *,
    label: str,
    memory_ids: list[str] | None,
    memory_text: str | None,
    width: int = DEFAULT_WRAP_WIDTH,
) -> str:
    """
    Format memory context summary for one agent.
    """

    return "\n".join(
        [
            f"{label} memory ids:",
            format_list(memory_ids, width=width),
            f"{label} memory text:",
            wrap_text(
                memory_text,
                width=width,
                subsequent_indent="  ",
            ),
        ]
    )


def format_rag_qa_state_summary(
    state: dict[str, Any],
    *,
    width: int = DEFAULT_WRAP_WIDTH,
) -> str:
    """
    Format standard RAG QA workflow final state.
    """

    lines = [
        format_section(
            "RAG QA WORKFLOW SUMMARY",
            "\n".join(
                [
                    format_field("current_step", state.get("current_step"), width=width),
                    format_field("success", state.get("success"), width=width),
                    format_field("error_message", state.get("error_message"), width=width),
                    format_field("question", state.get("question"), width=width),
                    "retrieved_chunk_ids:",
                    format_list(state.get("retrieved_chunk_ids"), width=width),
                    format_field("prediction", state.get("prediction"), width=width),
                ]
            ),
        ),
        format_section(
            "Worker A",
            format_agent_answer(state.get("worker_a_output"), width=width),
        ),
        format_section(
            "Worker B",
            format_agent_answer(state.get("worker_b_output"), width=width),
        ),
        format_section(
            "Critic",
            format_critic_output(state.get("critic_output"), width=width),
        ),
        format_section(
            "Coordinator",
            format_coordinator_output(state.get("coordinator_output"), width=width),
        ),
    ]

    return "\n".join(lines)


def format_governed_rag_qa_state_summary(
    state: dict[str, Any],
    *,
    width: int = DEFAULT_WRAP_WIDTH,
) -> str:
    """
    Format governed RAG QA workflow final state.
    """

    lines = [
        format_section(
            "GOVERNED RAG QA WORKFLOW SUMMARY",
            "\n".join(
                [
                    format_field("current_step", state.get("current_step"), width=width),
                    format_field("success", state.get("success"), width=width),
                    format_field("error_message", state.get("error_message"), width=width),
                    format_field("question", state.get("question"), width=width),
                    "retrieved_chunk_ids:",
                    format_list(state.get("retrieved_chunk_ids"), width=width),
                    format_field("prediction", state.get("prediction"), width=width),
                ]
            ),
        ),
        format_section(
            "Memory Access Summary",
            "\n\n".join(
                [
                    format_memory_context_summary(
                        label="Worker A",
                        memory_ids=state.get("worker_a_memory_ids"),
                        memory_text=state.get("worker_a_memory_text"),
                        width=width,
                    ),
                    format_memory_context_summary(
                        label="Worker B",
                        memory_ids=state.get("worker_b_memory_ids"),
                        memory_text=state.get("worker_b_memory_text"),
                        width=width,
                    ),
                    format_memory_context_summary(
                        label="Critic",
                        memory_ids=state.get("critic_memory_ids"),
                        memory_text=state.get("critic_memory_text"),
                        width=width,
                    ),
                    format_memory_context_summary(
                        label="Coordinator",
                        memory_ids=state.get("coordinator_memory_ids"),
                        memory_text=state.get("coordinator_memory_text"),
                        width=width,
                    ),
                ]
            ),
        ),
        format_section(
            "Worker A",
            format_agent_answer(state.get("worker_a_output"), width=width),
        ),
        format_section(
            "Worker B",
            format_agent_answer(state.get("worker_b_output"), width=width),
        ),
        format_section(
            "Critic",
            format_critic_output(state.get("critic_output"), width=width),
        ),
        format_section(
            "Coordinator",
            format_coordinator_output(state.get("coordinator_output"), width=width),
        ),
    ]

    return "\n".join(lines)