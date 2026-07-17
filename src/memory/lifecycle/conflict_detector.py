from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Optional

from ..entities import MemoryItem
from src.memory.utils.text_normalizer import TextNormalizer
from .conflict_schema import (
    ConflictCheckResult,
    MemoryConflictRecord,
)


class ConflictDetector:
    """
    Rule-based memory conflict detector.

    Current supported conflict types:
    - duplicate:
        Same or highly similar question, same normalized answer.
    - conflicting_answer:
        Same or highly similar question, different normalized answer.
    - none:
        No related memory found.

    This implementation is intentionally deterministic and lightweight.
    It does not call LLMs.
    """

    def __init__(
        self,
        *,
        question_similarity_threshold: float = 0.88,
    ) -> None:
        """
        Args:
            question_similarity_threshold:
                Minimum similarity score for two questions to be considered
                about the same fact.
        """

        if not 0.0 <= question_similarity_threshold <= 1.0:
            raise ValueError(
                "question_similarity_threshold must be between 0.0 and 1.0."
            )

        self.question_similarity_threshold = question_similarity_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_conflict(
        self,
        *,
        candidate_memory: MemoryItem,
        existing_memories: list[MemoryItem],
    ) -> ConflictCheckResult:
        """
        Check one candidate memory against existing memories.

        Args:
            candidate_memory:
                Newly generated candidate memory.
            existing_memories:
                Existing memories to compare against.

        Returns:
            ConflictCheckResult.
        """

        candidate_memory_id = self._get_memory_id(candidate_memory)
        candidate_content = self._get_memory_content(candidate_memory)

        candidate_question = self.extract_question(candidate_content)
        candidate_answer = self.extract_answer(candidate_content)

        if not candidate_question or not candidate_answer:
            return ConflictCheckResult.no_conflict(
                candidate_memory_id=candidate_memory_id,
                candidate_question=candidate_question,
                candidate_answer=candidate_answer,
                message=(
                    "No conflict detected because candidate memory does not "
                    "contain structured Question / Final answer fields."
                ),
            )

        duplicate_records: list[MemoryConflictRecord] = []
        conflicting_records: list[MemoryConflictRecord] = []

        for existing_memory in existing_memories:
            existing_memory_id = self._get_memory_id(existing_memory)

            # Do not compare memory with itself.
            if existing_memory_id == candidate_memory_id:
                continue

            existing_content = self._get_memory_content(existing_memory)
            existing_question = self.extract_question(existing_content)
            existing_answer = self.extract_answer(existing_content)

            if not existing_question or not existing_answer:
                continue

            question_similarity = self.question_similarity(
                candidate_question,
                existing_question,
            )

            if question_similarity < self.question_similarity_threshold:
                continue

            record = MemoryConflictRecord(
                matched_memory_id=existing_memory_id,
                matched_memory_content=existing_content,
                matched_question=existing_question,
                matched_answer=existing_answer,
                similarity_score=question_similarity,
            )

            if self.answers_equivalent(candidate_answer, existing_answer):
                duplicate_records.append(record)
            else:
                conflicting_records.append(record)

        # Conflict is more important than duplicate if both are detected.
        if conflicting_records:
            return ConflictCheckResult.conflicting_answer(
                candidate_memory_id=candidate_memory_id,
                matched_records=conflicting_records,
                candidate_question=candidate_question,
                candidate_answer=candidate_answer,
                message=(
                    "Conflicting answer detected: candidate memory is about "
                    "the same question as existing memory, but the answer differs."
                ),
            )

        if duplicate_records:
            return ConflictCheckResult.duplicate(
                candidate_memory_id=candidate_memory_id,
                matched_records=duplicate_records,
                candidate_question=candidate_question,
                candidate_answer=candidate_answer,
                message=(
                    "Duplicate memory detected: candidate memory has the same "
                    "question and equivalent answer as existing memory."
                ),
            )

        return ConflictCheckResult.no_conflict(
            candidate_memory_id=candidate_memory_id,
            candidate_question=candidate_question,
            candidate_answer=candidate_answer,
            message="No duplicate or conflicting answer detected.",
        )

    def check_against_memory_service(
        self,
        *,
        candidate_memory: MemoryItem,
        memory_service: Any,
        agent: Optional[Any] = None,
    ) -> ConflictCheckResult:
        """
        Check candidate memory against memories from MemoryService.

        If agent is provided, this checks against memories accessible to the agent.
        Otherwise, it tries to list all active memories.

        This helper avoids coupling the detector too tightly to one MemoryService API.
        """

        existing_memories = self._load_existing_memories(
            memory_service=memory_service,
            agent=agent,
        )

        return self.check_conflict(
            candidate_memory=candidate_memory,
            existing_memories=existing_memories,
        )

    # ------------------------------------------------------------------
    # Question / answer extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_question(content: str) -> Optional[str]:
        """
        Extract question from structured memory content.

        Supported line formats:
        - Question: ...
        - Q: ...
        """

        return ConflictDetector._extract_line_value(
            content=content,
            prefixes=["Question", "Q"],
        )

    @staticmethod
    def extract_answer(content: str) -> Optional[str]:
        """
        Extract answer from structured memory content.

        Supported line formats:
        - Final answer: ...
        - Answer: ...
        - A: ...
        """

        return ConflictDetector._extract_line_value(
            content=content,
            prefixes=["Final answer", "Answer", "A"],
        )

    @staticmethod
    def _extract_line_value(
        *,
        content: str,
        prefixes: list[str],
    ) -> Optional[str]:
        """
        Extract value from a line like:
            Prefix: value
        """

        if not content:
            return None

        for line in str(content).splitlines():
            line = line.strip()

            if not line:
                continue

            for prefix in prefixes:
                pattern = rf"^{re.escape(prefix)}\s*:\s*(.+)$"
                match = re.match(pattern, line, flags=re.IGNORECASE)

                if match:
                    value = match.group(1).strip()
                    return value or None

        return None

    # ------------------------------------------------------------------
    # Similarity / normalization
    # ------------------------------------------------------------------

    def question_similarity(
        self,
        question_a: str,
        question_b: str,
    ) -> float:
        """
        Compute normalized question similarity.

        Uses SequenceMatcher after normalization.
        """

        normalized_a = TextNormalizer.normalize_answer(question_a)
        normalized_b = TextNormalizer.normalize_answer(question_b)

        if not normalized_a and not normalized_b:
            return 1.0

        if not normalized_a or not normalized_b:
            return 0.0

        if normalized_a == normalized_b:
            return 1.0

        return SequenceMatcher(
            None,
            normalized_a,
            normalized_b,
        ).ratio()

    @staticmethod
    def answers_equivalent(
        answer_a: str,
        answer_b: str,
    ) -> bool:
        """
        Determine whether two answers are equivalent after normalization.
        """

        normalized_a = TextNormalizer.normalize_answer(answer_a)
        normalized_b = TextNormalizer.normalize_answer(answer_b)

        return normalized_a == normalized_b

    # ------------------------------------------------------------------
    # Memory loading helpers
    # ------------------------------------------------------------------

    def _load_existing_memories(
        self,
        *,
        memory_service: Any,
        agent: Optional[Any] = None,
    ) -> list[MemoryItem]:
        """
        Load existing memories from MemoryService.

        Supports several method names used in this project.
        """

        if agent is not None:
            for method_name in [
                "list_accessible_memories",
                "list_accessible_memory",
            ]:
                if hasattr(memory_service, method_name):
                    method = getattr(memory_service, method_name)

                    try:
                        return method(agent=agent)
                    except TypeError:
                        try:
                            return method(requesting_agent=agent)
                        except TypeError:
                            return method(agent)

        for method_name in [
            "list_active_memories",
            "list_active",
            "list_all_memories",
            "list_all",
        ]:
            if hasattr(memory_service, method_name):
                method = getattr(memory_service, method_name)
                return method()

        if hasattr(memory_service, "memory_store"):
            memory_store = getattr(memory_service, "memory_store")

            for method_name in [
                "list_active",
                "list_all",
            ]:
                if hasattr(memory_store, method_name):
                    method = getattr(memory_store, method_name)
                    return method()

        raise AttributeError(
            "Could not load existing memories from memory_service. "
            "Expected list_accessible_memories, list_active, or list_all."
        )

    # ------------------------------------------------------------------
    # Memory field helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_memory_id(memory_item: MemoryItem) -> str:
        for attr_name in ["memory_id", "id"]:
            if hasattr(memory_item, attr_name):
                value = getattr(memory_item, attr_name)
                if value is not None:
                    return str(value)

        raise AttributeError("MemoryItem must have memory_id or id.")

    @staticmethod
    def _get_memory_content(memory_item: MemoryItem) -> str:
        if hasattr(memory_item, "content"):
            value = getattr(memory_item, "content")
            return "" if value is None else str(value)

        return ""