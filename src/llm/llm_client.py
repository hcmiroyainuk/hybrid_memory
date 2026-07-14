from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.memory.external_knowledge import RetrievedKnowledge
from .output_parser import LLMOutputParser
from .output_schemas import (
    AgentAnswer,
    CriticOutput,
    CoordinatorOutput,
)
from .prompt_templates import PromptTemplates


class LLMClientConfigurationError(Exception):
    """Raised when LLM client configuration is invalid."""


class LLMClient:
    """
    LLM client for the RAG-based multi-agent QA baseline.

    Responsibilities:
    - load OPENAI_API_KEY from .env
    - initialize ChatOpenAI
    - call role-specific prompts
    - return structured Pydantic outputs

    Agents:
    - Worker A: direct QA
    - Worker B: retrieval-grounded QA
    - Critic: compare Worker A and Worker B
    - Coordinator: final answer generation

    accessible_memory_context:
    - optional permission-filtered memory context
    - should be prepared by GovernedContextBuilder
    - this client does not perform permission checking
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 500,
        env_path: Optional[str | Path] = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._load_environment(env_path)
        self._validate_openai_api_key()

        self.chat_model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.agent_answer_model = self.chat_model.with_structured_output(
            AgentAnswer
        )
        self.critic_output_model = self.chat_model.with_structured_output(
            CriticOutput
        )
        self.coordinator_output_model = self.chat_model.with_structured_output(
            CoordinatorOutput
        )

    # ------------------------------------------------------------------
    # Worker A
    # ------------------------------------------------------------------

    def generate_worker_a_answer(
        self,
        question: str,
        accessible_memory_context: str | None = None,
    ) -> AgentAnswer:
        """
        Generate Worker A's direct answer.

        Worker A receives:
        - question
        - optional permission-filtered accessible memory

        Worker A does not receive external retrieved evidence.
        """

        self._validate_question(question)

        prompt = PromptTemplates.build_worker_a_prompt(
            question=question,
            accessible_memory_context=accessible_memory_context,
        )

        output = self.agent_answer_model.invoke(prompt)

        if not isinstance(output, AgentAnswer):
            output = AgentAnswer.model_validate(output)

        return LLMOutputParser.clean_agent_answer(output)

    # ------------------------------------------------------------------
    # Worker B
    # ------------------------------------------------------------------

    def generate_worker_b_answer(
        self,
        question: str,
        retrieved_knowledge: list[RetrievedKnowledge],
        accessible_memory_context: str | None = None,
    ) -> AgentAnswer:
        """
        Generate Worker B's retrieval-grounded answer.

        Worker B receives:
        - question
        - retrieved external knowledge
        - optional permission-filtered accessible memory
        """

        self._validate_question(question)

        prompt = PromptTemplates.build_worker_b_prompt(
            question=question,
            retrieved_knowledge=retrieved_knowledge,
            accessible_memory_context=accessible_memory_context,
        )

        output = self.agent_answer_model.invoke(prompt)

        if not isinstance(output, AgentAnswer):
            output = AgentAnswer.model_validate(output)

        return LLMOutputParser.clean_agent_answer(output)

    # ------------------------------------------------------------------
    # Critic
    # ------------------------------------------------------------------

    def generate_critic_output(
        self,
        question: str,
        worker_a_output: AgentAnswer,
        worker_b_output: AgentAnswer,
        retrieved_knowledge: list[RetrievedKnowledge],
        accessible_memory_context: str | None = None,
    ) -> CriticOutput:
        """
        Generate Critic's comparison and recommendation.

        Critic receives:
        - question
        - Worker A output
        - Worker B output
        - retrieved external knowledge
        - optional permission-filtered accessible memory
        """

        self._validate_question(question)

        prompt = PromptTemplates.build_critic_prompt(
            question=question,
            worker_a_output=worker_a_output,
            worker_b_output=worker_b_output,
            retrieved_knowledge=retrieved_knowledge,
            accessible_memory_context=accessible_memory_context,
        )

        output = self.critic_output_model.invoke(prompt)

        if not isinstance(output, CriticOutput):
            output = CriticOutput.model_validate(output)

        return LLMOutputParser.clean_critic_output(output)

    # ------------------------------------------------------------------
    # Coordinator
    # ------------------------------------------------------------------

    def generate_coordinator_output(
        self,
        question: str,
        worker_a_output: AgentAnswer,
        worker_b_output: AgentAnswer,
        critic_output: CriticOutput,
        retrieved_knowledge: list[RetrievedKnowledge],
        accessible_memory_context: str | None = None,
    ) -> CoordinatorOutput:
        """
        Generate Coordinator's final answer.

        Coordinator receives:
        - question
        - Worker A output
        - Worker B output
        - Critic output
        - retrieved external knowledge
        - optional permission-filtered accessible memory

        final_answer is the answer used later for SQuAD EM / F1 evaluation.
        """

        self._validate_question(question)

        prompt = PromptTemplates.build_coordinator_prompt(
            question=question,
            worker_a_output=worker_a_output,
            worker_b_output=worker_b_output,
            critic_output=critic_output,
            retrieved_knowledge=retrieved_knowledge,
            accessible_memory_context=accessible_memory_context,
        )

        output = self.coordinator_output_model.invoke(prompt)

        if not isinstance(output, CoordinatorOutput):
            output = CoordinatorOutput.model_validate(output)

        return LLMOutputParser.clean_coordinator_output(output)

    # ------------------------------------------------------------------
    # Full four-agent chain helper
    # ------------------------------------------------------------------

    def run_four_agent_qa(
        self,
        question: str,
        retrieved_knowledge: list[RetrievedKnowledge],
        worker_a_memory_context: str | None = None,
        worker_b_memory_context: str | None = None,
        critic_memory_context: str | None = None,
        coordinator_memory_context: str | None = None,
    ) -> dict:
        """
        Run Worker A, Worker B, Critic, and Coordinator in sequence.

        This helper supports both:
        - standard RAG QA baseline
        - governed memory RAG QA baseline

        For standard baseline, leave all memory context arguments as None.
        For governed baseline, pass permission-filtered memory contexts.
        """

        worker_a_output = self.generate_worker_a_answer(
            question=question,
            accessible_memory_context=worker_a_memory_context,
        )

        worker_b_output = self.generate_worker_b_answer(
            question=question,
            retrieved_knowledge=retrieved_knowledge,
            accessible_memory_context=worker_b_memory_context,
        )

        critic_output = self.generate_critic_output(
            question=question,
            worker_a_output=worker_a_output,
            worker_b_output=worker_b_output,
            retrieved_knowledge=retrieved_knowledge,
            accessible_memory_context=critic_memory_context,
        )

        coordinator_output = self.generate_coordinator_output(
            question=question,
            worker_a_output=worker_a_output,
            worker_b_output=worker_b_output,
            critic_output=critic_output,
            retrieved_knowledge=retrieved_knowledge,
            accessible_memory_context=coordinator_memory_context,
        )

        return {
            "worker_a": worker_a_output,
            "worker_b": worker_b_output,
            "critic": critic_output,
            "coordinator": coordinator_output,
        }

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def build_worker_a_prompt_preview(
        self,
        question: str,
        accessible_memory_context: str | None = None,
    ) -> str:
        """
        Return Worker A prompt without calling the LLM.
        """

        return PromptTemplates.build_worker_a_prompt(
            question=question,
            accessible_memory_context=accessible_memory_context,
        )

    def build_worker_b_prompt_preview(
        self,
        question: str,
        retrieved_knowledge: list[RetrievedKnowledge],
        accessible_memory_context: str | None = None,
    ) -> str:
        """
        Return Worker B prompt without calling the LLM.
        """

        return PromptTemplates.build_worker_b_prompt(
            question=question,
            retrieved_knowledge=retrieved_knowledge,
            accessible_memory_context=accessible_memory_context,
        )

    def build_critic_prompt_preview(
        self,
        question: str,
        worker_a_output: AgentAnswer,
        worker_b_output: AgentAnswer,
        retrieved_knowledge: list[RetrievedKnowledge],
        accessible_memory_context: str | None = None,
    ) -> str:
        """
        Return Critic prompt without calling the LLM.
        """

        return PromptTemplates.build_critic_prompt(
            question=question,
            worker_a_output=worker_a_output,
            worker_b_output=worker_b_output,
            retrieved_knowledge=retrieved_knowledge,
            accessible_memory_context=accessible_memory_context,
        )

    def build_coordinator_prompt_preview(
        self,
        question: str,
        worker_a_output: AgentAnswer,
        worker_b_output: AgentAnswer,
        critic_output: CriticOutput,
        retrieved_knowledge: list[RetrievedKnowledge],
        accessible_memory_context: str | None = None,
    ) -> str:
        """
        Return Coordinator prompt without calling the LLM.
        """

        return PromptTemplates.build_coordinator_prompt(
            question=question,
            worker_a_output=worker_a_output,
            worker_b_output=worker_b_output,
            critic_output=critic_output,
            retrieved_knowledge=retrieved_knowledge,
            accessible_memory_context=accessible_memory_context,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_environment(
        env_path: Optional[str | Path] = None,
    ) -> None:
        """
        Load environment variables from .env.

        If env_path is None, python-dotenv searches from the current working
        directory upward.
        """

        if env_path is not None:
            load_dotenv(dotenv_path=Path(env_path))
        else:
            load_dotenv()

    @staticmethod
    def _validate_openai_api_key() -> None:
        """
        Ensure OPENAI_API_KEY is available.
        """

        if not os.getenv("OPENAI_API_KEY"):
            raise LLMClientConfigurationError(
                "OPENAI_API_KEY is missing. "
                "Put it in the project root .env file or configure it in PyCharm."
            )

    @staticmethod
    def _validate_question(question: str) -> None:
        """
        Validate input question.
        """

        if not question or not question.strip():
            raise ValueError("question cannot be empty.")