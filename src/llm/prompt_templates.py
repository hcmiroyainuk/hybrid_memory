from __future__ import annotations

import json
from typing import Iterable

from src.memory.external_knowledge import RetrievedKnowledge
from .output_schemas import AgentAnswer, CriticOutput, CoordinatorOutput


class PromptTemplates:
    """
    Prompt templates for the RAG-based multi-agent QA baseline.

    Agents:
    - Worker A: direct parametric QA agent
    - Worker B: retrieval-grounded QA agent
    - Critic: compares Worker A and Worker B
    - Coordinator: produces final answer

    This module only builds prompts.
    It does not call LLMs or parse outputs.
    """

    # ------------------------------------------------------------------
    # Worker A prompt
    # ------------------------------------------------------------------

    @staticmethod
    def build_worker_a_prompt(question: str) -> str:
        """
        Build prompt for Worker A.

        Worker A answers using parametric model knowledge only.
        """

        return f"""
You are Worker A, a direct question-answering agent.

Your task:
Answer the question using your own knowledge.

Important rules:
- Return a short answer span.
- Do not return a full sentence unless necessary.
- Do not include explanation in the answer field.
- Put explanation only in the reasoning field.
- If uncertain, still give the best possible answer.
- Your output must follow the required schema.

Question:
{question}

Required output schema:
{PromptTemplates._schema_hint_for_agent_answer()}
""".strip()

    # ------------------------------------------------------------------
    # Worker B prompt
    # ------------------------------------------------------------------

    @staticmethod
    def build_worker_b_prompt(
        question: str,
        retrieved_knowledge: list[RetrievedKnowledge],
    ) -> str:
        """
        Build prompt for Worker B.

        Worker B answers using retrieved external evidence.
        """

        evidence_text = PromptTemplates.format_retrieved_knowledge(
            retrieved_knowledge
        )

        return f"""
You are Worker B, a retrieval-grounded question-answering agent.

Your task:
Answer the question using the retrieved evidence.

Important rules:
- Prefer answers directly supported by the retrieved evidence.
- Return a short answer span.
- Do not return a full sentence unless necessary.
- Do not include explanation in the answer field.
- Put explanation only in the reasoning field.
- Include the chunk IDs that support your answer in evidence_chunk_ids.
- If the retrieved evidence is insufficient, give the best possible answer and use lower confidence.
- Your output must follow the required schema.

Question:
{question}

Retrieved evidence:
{evidence_text}

Required output schema:
{PromptTemplates._schema_hint_for_agent_answer()}
""".strip()

    # ------------------------------------------------------------------
    # Critic prompt
    # ------------------------------------------------------------------

    @staticmethod
    def build_critic_prompt(
        question: str,
        worker_a_output: AgentAnswer,
        worker_b_output: AgentAnswer,
        retrieved_knowledge: list[RetrievedKnowledge],
    ) -> str:
        """
        Build prompt for the Critic.

        The Critic compares Worker A and Worker B and recommends one answer.
        """

        evidence_text = PromptTemplates.format_retrieved_knowledge(
            retrieved_knowledge
        )

        worker_a_text = PromptTemplates.format_agent_answer(
            agent_name="Worker A",
            answer=worker_a_output,
        )

        worker_b_text = PromptTemplates.format_agent_answer(
            agent_name="Worker B",
            answer=worker_b_output,
        )

        return f"""
You are the Critic in a multi-agent question-answering system.

Your task:
Compare Worker A and Worker B, evaluate their answers, and recommend the most likely answer.

Important rules:
- Check whether Worker B's answer is supported by the retrieved evidence.
- Do not blindly prefer Worker B.
- Prefer Worker B only if the evidence is relevant and supports the answer.
- If the retrieved evidence is irrelevant or insufficient, prefer the more plausible answer.
- If both answers are weak, still recommend the best possible short answer.
- recommended_answer must be a short answer span.
- Do not include explanation in recommended_answer.
- Put explanation only in comment.
- Your output must follow the required schema.

Question:
{question}

Worker outputs:
{worker_a_text}

{worker_b_text}

Retrieved evidence:
{evidence_text}

Required output schema:
{PromptTemplates._schema_hint_for_critic_output()}
""".strip()

    # ------------------------------------------------------------------
    # Coordinator prompt
    # ------------------------------------------------------------------

    @staticmethod
    def build_coordinator_prompt(
        question: str,
        worker_a_output: AgentAnswer,
        worker_b_output: AgentAnswer,
        critic_output: CriticOutput,
        retrieved_knowledge: list[RetrievedKnowledge],
    ) -> str:
        """
        Build prompt for the Coordinator.

        The Coordinator produces the final answer used for evaluation.
        """

        evidence_text = PromptTemplates.format_retrieved_knowledge(
            retrieved_knowledge
        )

        worker_a_text = PromptTemplates.format_agent_answer(
            agent_name="Worker A",
            answer=worker_a_output,
        )

        worker_b_text = PromptTemplates.format_agent_answer(
            agent_name="Worker B",
            answer=worker_b_output,
        )

        critic_text = PromptTemplates.format_critic_output(critic_output)

        return f"""
You are the Coordinator in a multi-agent question-answering system.

Your task:
Produce the final answer for evaluation.

You are given:
- The original question
- Worker A's direct answer
- Worker B's retrieval-grounded answer
- The retrieved evidence
- The Critic's recommendation

Important rules:
- Use all available information.
- Prefer evidence-supported answers when the retrieved evidence is relevant.
- Do not blindly follow any single worker.
- final_answer must be one short answer span.
- Do not include explanation in final_answer.
- Put explanation only in reasoning.
- Your output must follow the required schema.

Question:
{question}

Worker outputs:
{worker_a_text}

{worker_b_text}

Critic recommendation:
{critic_text}

Retrieved evidence:
{evidence_text}

Required output schema:
{PromptTemplates._schema_hint_for_coordinator_output()}
""".strip()

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def format_retrieved_knowledge(
        retrieved_knowledge: list[RetrievedKnowledge],
        max_chars_per_chunk: int = 1200,
    ) -> str:
        """
        Format retrieved chunks for prompt input.

        Args:
            retrieved_knowledge:
                Retrieved chunks returned by ExternalKnowledgeRetriever.
            max_chars_per_chunk:
                Prevents very long contexts from making prompts too large.
        """

        if not retrieved_knowledge:
            return "No retrieved evidence."

        formatted_chunks: list[str] = []

        for index, item in enumerate(retrieved_knowledge, start=1):
            content = item.content.strip()

            if len(content) > max_chars_per_chunk:
                content = content[:max_chars_per_chunk].rstrip() + "..."

            score_text = (
                f"{item.score:.4f}"
                if item.score is not None
                else "N/A"
            )

            formatted_chunks.append(
                "\n".join(
                    [
                        f"[Evidence {index}]",
                        f"chunk_id: {item.chunk_id}",
                        f"context_id: {item.context_id}",
                        f"title: {item.title}",
                        f"source: {item.source}",
                        f"score: {score_text}",
                        "content:",
                        content,
                    ]
                )
            )

        return "\n\n".join(formatted_chunks)

    @staticmethod
    def format_agent_answer(
        agent_name: str,
        answer: AgentAnswer,
    ) -> str:
        """
        Format an AgentAnswer for Critic / Coordinator prompts.
        """

        return "\n".join(
            [
                f"{agent_name}:",
                f"answer: {answer.answer}",
                f"reasoning: {answer.reasoning}",
                f"confidence: {answer.confidence}",
                f"evidence_chunk_ids: {answer.evidence_chunk_ids}",
            ]
        )

    @staticmethod
    def format_critic_output(
        critic_output: CriticOutput,
    ) -> str:
        """
        Format CriticOutput for Coordinator prompt.
        """

        return "\n".join(
            [
                f"recommended_answer: {critic_output.recommended_answer}",
                f"preferred_worker: {critic_output.preferred_worker}",
                f"comment: {critic_output.comment}",
                f"confidence: {critic_output.confidence}",
            ]
        )

    # ------------------------------------------------------------------
    # Schema hints
    # ------------------------------------------------------------------

    @staticmethod
    def _schema_hint_for_agent_answer() -> str:
        """
        Human-readable JSON schema hint for AgentAnswer.

        This is included in prompts even when structured output is used,
        because it helps the model follow the expected shape.
        """

        return json.dumps(
            {
                "answer": "short answer span",
                "reasoning": "brief reasoning",
                "confidence": "float between 0.0 and 1.0",
                "evidence_chunk_ids": [
                    "chunk id used as evidence; empty list if no evidence"
                ],
            },
            indent=2,
            ensure_ascii=False,
        )

    @staticmethod
    def _schema_hint_for_critic_output() -> str:
        """
        Human-readable JSON schema hint for CriticOutput.
        """

        return json.dumps(
            {
                "recommended_answer": "short answer span",
                "preferred_worker": "worker_a | worker_b | uncertain",
                "comment": "brief critique explaining the recommendation",
                "confidence": "float between 0.0 and 1.0",
            },
            indent=2,
            ensure_ascii=False,
        )

    @staticmethod
    def _schema_hint_for_coordinator_output() -> str:
        """
        Human-readable JSON schema hint for CoordinatorOutput.
        """

        return json.dumps(
            {
                "final_answer": "short answer span",
                "reasoning": "brief explanation for the final decision",
                "confidence": "float between 0.0 and 1.0",
            },
            indent=2,
            ensure_ascii=False,
        )


# ----------------------------------------------------------------------
# Optional functional wrappers
# ----------------------------------------------------------------------

def build_worker_a_prompt(question: str) -> str:
    return PromptTemplates.build_worker_a_prompt(question)


def build_worker_b_prompt(
    question: str,
    retrieved_knowledge: list[RetrievedKnowledge],
) -> str:
    return PromptTemplates.build_worker_b_prompt(
        question=question,
        retrieved_knowledge=retrieved_knowledge,
    )


def build_critic_prompt(
    question: str,
    worker_a_output: AgentAnswer,
    worker_b_output: AgentAnswer,
    retrieved_knowledge: list[RetrievedKnowledge],
) -> str:
    return PromptTemplates.build_critic_prompt(
        question=question,
        worker_a_output=worker_a_output,
        worker_b_output=worker_b_output,
        retrieved_knowledge=retrieved_knowledge,
    )


def build_coordinator_prompt(
    question: str,
    worker_a_output: AgentAnswer,
    worker_b_output: AgentAnswer,
    critic_output: CriticOutput,
    retrieved_knowledge: list[RetrievedKnowledge],
) -> str:
    return PromptTemplates.build_coordinator_prompt(
        question=question,
        worker_a_output=worker_a_output,
        worker_b_output=worker_b_output,
        critic_output=critic_output,
        retrieved_knowledge=retrieved_knowledge,
    )