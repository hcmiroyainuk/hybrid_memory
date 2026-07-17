from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from ..entities import MemoryItem, MemoryMetadata


class CandidateMemoryBuilder:
    """
    Build candidate memories from governed RAG QA workflow outputs.

    Responsibility:
    - Convert a workflow final_state into a MemoryItem.
    - The generated memory is private by default.
    - This class does not write memory to store.
    - This class does not check conflicts.
    - This class does not promote memory.

    Default design:
    - Worker B owns the candidate memory because Worker B is the retrieval-grounded agent.
    - The candidate can later be promoted to shared memory after conflict checking and review.
    """

    DEFAULT_MEMORY_TYPE = "fact"

    DEFAULT_TAGS = [
        "qa",
        "squad",
        "generated",
        "candidate",
        "governed_rag_qa",
    ]

    @classmethod
    def build_from_final_state(
        cls,
        final_state: dict[str, Any],
        *,
        owner_agent_id: str = "worker_b",
        created_by_agent_id: str = "worker_b",
        memory_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> MemoryItem:
        """
        Build a private candidate memory from governed RAG QA final state.

        Args:
            final_state:
                Final state returned by GovernedRAGQAWorkflow.
            owner_agent_id:
                Agent who owns the private candidate memory.
            created_by_agent_id:
                Agent who created the candidate memory.
            memory_id:
                Optional custom memory id. If not provided, a UUID-based id is generated.
            tags:
                Optional additional tags.

        Returns:
            MemoryItem:
                A private memory candidate.
        """

        cls._validate_final_state(final_state)

        actual_memory_id = memory_id or cls.generate_memory_id()

        content = cls.build_memory_content(final_state)

        merged_tags = cls.build_tags(tags)

        metadata = MemoryMetadata(
            memory_id=actual_memory_id,
            scope="private",
            owner_agent_id=owner_agent_id,
            created_by_agent_id=created_by_agent_id,
            status="active",
            memory_type=cls.DEFAULT_MEMORY_TYPE,
            tags=merged_tags,
            readable_by=[owner_agent_id],
            writable_by=[owner_agent_id],
        )

        return MemoryItem(
            memory_id=actual_memory_id,
            content=content,
            metadata=metadata,
        )

    @classmethod
    def build_memory_content(
        cls,
        final_state: dict[str, Any],
    ) -> str:
        """
        Build the textual memory content from workflow final state.

        The format is intentionally structured because conflict_detector.py
        will later extract Question and Final answer from this content.
        """

        question = cls._safe_get_text(final_state, "question")
        prediction = cls._extract_prediction(final_state)

        worker_a_answer = cls._extract_agent_answer(
            final_state.get("worker_a_output")
        )
        worker_b_answer = cls._extract_agent_answer(
            final_state.get("worker_b_output")
        )
        critic_recommendation = cls._extract_critic_recommendation(
            final_state.get("critic_output")
        )

        retrieved_chunk_ids = cls._safe_get_list(
            final_state.get("retrieved_chunk_ids")
        )

        retrieved_context_hashes = cls._safe_get_list(
            final_state.get("retrieved_context_hashes")
        )

        retrieval_hit = final_state.get("retrieval_hit")

        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

        lines = [
            "Memory source: governed_rag_qa_workflow",
            f"Created at: {timestamp}",
            "",
            f"Question: {question}",
            f"Final answer: {prediction}",
            "",
            f"Worker A answer: {worker_a_answer}",
            f"Worker B answer: {worker_b_answer}",
            f"Critic recommendation: {critic_recommendation}",
            "",
            f"Retrieved chunk ids: {retrieved_chunk_ids}",
            f"Retrieved context hashes: {retrieved_context_hashes}",
            f"Retrieval hit: {retrieval_hit}",
        ]

        return "\n".join(lines).strip()

    @classmethod
    def build_tags(
        cls,
        extra_tags: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Build de-duplicated tags for candidate memory.
        """

        tags: list[str] = []

        for tag in cls.DEFAULT_TAGS + (extra_tags or []):
            tag = str(tag).strip()
            if tag and tag not in tags:
                tags.append(tag)

        return tags

    @staticmethod
    def generate_memory_id() -> str:
        """
        Generate candidate memory id.
        """

        return f"candidate_memory_{uuid4().hex}"

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    @classmethod
    def _extract_prediction(
        cls,
        final_state: dict[str, Any],
    ) -> str:
        """
        Extract final prediction from final_state.

        Preference:
        1. coordinator_output.final_answer
        2. final_state["prediction"]
        3. empty string
        """

        coordinator_output = final_state.get("coordinator_output")

        if coordinator_output is not None and hasattr(
            coordinator_output,
            "final_answer",
        ):
            value = getattr(coordinator_output, "final_answer")
            if value:
                return str(value).strip()

        return cls._safe_get_text(final_state, "prediction")

    @staticmethod
    def _extract_agent_answer(
        output: Any,
    ) -> str:
        """
        Extract answer from AgentAnswer-like object.
        """

        if output is None:
            return ""

        if hasattr(output, "answer"):
            value = getattr(output, "answer")
            return "" if value is None else str(value).strip()

        if isinstance(output, dict):
            value = output.get("answer", "")
            return "" if value is None else str(value).strip()

        return str(output).strip()

    @staticmethod
    def _extract_critic_recommendation(
        output: Any,
    ) -> str:
        """
        Extract recommended answer from CriticOutput-like object.
        """

        if output is None:
            return ""

        if hasattr(output, "recommended_answer"):
            value = getattr(output, "recommended_answer")
            return "" if value is None else str(value).strip()

        if isinstance(output, dict):
            value = output.get("recommended_answer", "")
            return "" if value is None else str(value).strip()

        return str(output).strip()

    @staticmethod
    def _safe_get_text(
        data: dict[str, Any],
        key: str,
    ) -> str:
        value = data.get(key, "")
        return "" if value is None else str(value).strip()

    @staticmethod
    def _safe_get_list(
        value: Any,
    ) -> list[str]:
        """
        Normalize values that may be:
        - list
        - tuple
        - callable returning list
        - None
        """

        if callable(value):
            value = value()

        if value is None:
            return []

        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]

        if isinstance(value, tuple):
            return [str(item).strip() for item in value if str(item).strip()]

        return [str(value).strip()] if str(value).strip() else []

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_final_state(
        final_state: dict[str, Any],
    ) -> None:
        if not isinstance(final_state, dict):
            raise TypeError("final_state must be a dictionary-like state.")

        question = final_state.get("question")

        if not question or not str(question).strip():
            raise ValueError("final_state must contain a non-empty question.")

        has_prediction = bool(final_state.get("prediction"))

        coordinator_output = final_state.get("coordinator_output")
        has_coordinator_answer = (
            coordinator_output is not None
            and hasattr(coordinator_output, "final_answer")
            and bool(getattr(coordinator_output, "final_answer"))
        )

        if not has_prediction and not has_coordinator_answer:
            raise ValueError(
                "final_state must contain prediction or coordinator_output.final_answer."
            )