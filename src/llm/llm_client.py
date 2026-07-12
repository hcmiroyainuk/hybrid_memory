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
    - Worker A: direct parametric QA
    - Worker B: retrieval-grounded QA
    - Critic: compare Worker A and Worker B
    - Coordinator: final answer generation
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
    ) -> AgentAnswer:
        """
        Generate Worker A's direct answer.

        Worker A only receives the question and does not use retrieved evidence.
        """

        self._validate_question(question)

        prompt = PromptTemplates.build_worker_a_prompt(
            question=question,
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
    ) -> AgentAnswer:
        """
        Generate Worker B's retrieval-grounded answer.

        Worker B receives the question and retrieved external knowledge chunks.
        """

        self._validate_question(question)

        prompt = PromptTemplates.build_worker_b_prompt(
            question=question,
            retrieved_knowledge=retrieved_knowledge,
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
    ) -> CriticOutput:
        """
        Generate Critic's comparison and recommendation.
        """

        self._validate_question(question)

        prompt = PromptTemplates.build_critic_prompt(
            question=question,
            worker_a_output=worker_a_output,
            worker_b_output=worker_b_output,
            retrieved_knowledge=retrieved_knowledge,
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
    ) -> CoordinatorOutput:
        """
        Generate Coordinator's final answer.

        final_answer is the answer used later for SQuAD EM / F1 evaluation.
        """

        self._validate_question(question)

        prompt = PromptTemplates.build_coordinator_prompt(
            question=question,
            worker_a_output=worker_a_output,
            worker_b_output=worker_b_output,
            critic_output=critic_output,
            retrieved_knowledge=retrieved_knowledge,
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
    ) -> dict:
        """
        Run Worker A, Worker B, Critic, and Coordinator in sequence.

        This is useful for quick testing before building the LangGraph workflow.
        """

        worker_a_output = self.generate_worker_a_answer(
            question=question,
        )

        worker_b_output = self.generate_worker_b_answer(
            question=question,
            retrieved_knowledge=retrieved_knowledge,
        )

        critic_output = self.generate_critic_output(
            question=question,
            worker_a_output=worker_a_output,
            worker_b_output=worker_b_output,
            retrieved_knowledge=retrieved_knowledge,
        )

        coordinator_output = self.generate_coordinator_output(
            question=question,
            worker_a_output=worker_a_output,
            worker_b_output=worker_b_output,
            critic_output=critic_output,
            retrieved_knowledge=retrieved_knowledge,
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
    ) -> str:
        """
        Return Worker A prompt without calling the LLM.
        """

        return PromptTemplates.build_worker_a_prompt(question)

    def build_worker_b_prompt_preview(
        self,
        question: str,
        retrieved_knowledge: list[RetrievedKnowledge],
    ) -> str:
        """
        Return Worker B prompt without calling the LLM.
        """

        return PromptTemplates.build_worker_b_prompt(
            question=question,
            retrieved_knowledge=retrieved_knowledge,
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