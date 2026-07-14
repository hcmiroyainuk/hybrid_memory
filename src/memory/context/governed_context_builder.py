from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..entities import Agent, MemoryItem


@dataclass
class GovernedMemoryContext:
    """
    Permission-filtered memory context for one agent.

    This object is used before LLM invocation.
    Only memories that the current agent is allowed to access should appear here.
    """

    agent_id: str
    agent_role: str
    query: str
    memory_items: list[MemoryItem]
    memory_ids: list[str]
    formatted_context: str


class GovernedContextBuilder:
    """
    Build permission-filtered memory context for LLM agents.

    Responsibilities:
    - retrieve memories accessible to a given agent
    - format accessible memories for prompt input
    - keep permission filtering outside the LLM prompt

    This class does NOT:
    - call LLMs
    - modify memory
    - approve promotion requests
    - write operation logs

    Permission logic should be handled by MemoryService / MemoryRetriever /
    PermissionService. This class only coordinates context construction.
    """

    def __init__(
        self,
        *,
        memory_service: Any,
        memory_retriever: Optional[Any] = None,
        default_top_k: int = 3,
        max_chars_per_memory: int = 800,
    ) -> None:
        """
        Args:
            memory_service:
                MemoryService instance. Used as fallback for accessible memory listing.
            memory_retriever:
                Optional MemoryRetriever instance. Prefer using this when available
                because it performs semantic retrieval over accessible memories.
            default_top_k:
                Number of memories to include by default.
            max_chars_per_memory:
                Maximum content length for each formatted memory item.
        """

        if default_top_k <= 0:
            raise ValueError("default_top_k must be greater than 0.")

        if max_chars_per_memory <= 0:
            raise ValueError("max_chars_per_memory must be greater than 0.")

        self.memory_service = memory_service
        self.memory_retriever = memory_retriever
        self.default_top_k = default_top_k
        self.max_chars_per_memory = max_chars_per_memory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_memory_context(
        self,
        *,
        agent: Agent,
        query: str,
        top_k: Optional[int] = None,
    ) -> GovernedMemoryContext:
        """
        Build accessible memory context for one agent.

        Args:
            agent:
                Current agent identity.
            query:
                Query / question used for retrieval.
            top_k:
                Optional retrieval limit.

        Returns:
            GovernedMemoryContext containing only accessible memories.
        """

        self._validate_query(query)

        actual_top_k = top_k or self.default_top_k

        if actual_top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        memory_items = self.retrieve_accessible_memories(
            agent=agent,
            query=query,
            top_k=actual_top_k,
        )

        memory_ids = [
            self._get_memory_id(memory_item)
            for memory_item in memory_items
        ]

        formatted_context = self.format_memory_items(
            memory_items=memory_items,
            max_chars_per_memory=self.max_chars_per_memory,
        )

        return GovernedMemoryContext(
            agent_id=self._get_agent_id(agent),
            agent_role=self._get_agent_role(agent),
            query=query,
            memory_items=memory_items,
            memory_ids=memory_ids,
            formatted_context=formatted_context,
        )

    def build_context_text(
        self,
        *,
        agent: Agent,
        query: str,
        top_k: Optional[int] = None,
    ) -> str:
        """
        Convenience method returning only the formatted memory context string.
        """

        context = self.build_memory_context(
            agent=agent,
            query=query,
            top_k=top_k,
        )

        return context.formatted_context

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_accessible_memories(
        self,
        *,
        agent: Agent,
        query: str,
        top_k: int,
    ) -> list[MemoryItem]:
        """
        Retrieve memories that are accessible to the given agent.

        Preferred path:
            MemoryRetriever.retrieve(query=query, agent=agent, top_k=top_k)

        Fallback path:
            MemoryService.list_accessible_memories(agent)
            then simple keyword ranking.

        The important rule:
            forbidden memories must never be returned.
        """

        if self.memory_retriever is not None:
            return self._retrieve_with_memory_retriever(
                agent=agent,
                query=query,
                top_k=top_k,
            )

        return self._retrieve_with_memory_service_fallback(
            agent=agent,
            query=query,
            top_k=top_k,
        )

    def _retrieve_with_memory_retriever(
        self,
        *,
        agent: Agent,
        query: str,
        top_k: int,
    ) -> list[MemoryItem]:
        """
        Retrieve using MemoryRetriever.

        This method assumes your MemoryRetriever already performs permission-first
        retrieval internally.
        """

        try:
            result = self.memory_retriever.retrieve(
                query=query,
                agent=agent,
                top_k=top_k,
            )
        except TypeError:
            # Fallback for retriever signatures using requesting_agent.
            result = self.memory_retriever.retrieve(
                query=query,
                requesting_agent=agent,
                top_k=top_k,
            )

        return self._normalize_retrieval_result(result)

    def _retrieve_with_memory_service_fallback(
        self,
        *,
        agent: Agent,
        query: str,
        top_k: int,
    ) -> list[MemoryItem]:
        """
        Fallback when MemoryRetriever is not provided.

        It first asks MemoryService for accessible memories, then does a simple
        lexical ranking. This is less powerful than vector retrieval but useful
        for testing permission correctness.
        """

        try:
            accessible_memories = self.memory_service.list_accessible_memories(
                agent=agent,
            )
        except TypeError:
            accessible_memories = self.memory_service.list_accessible_memories(
                requesting_agent=agent,
            )

        ranked = self._simple_rank_memories(
            query=query,
            memory_items=accessible_memories,
        )

        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_memory_items(
        self,
        *,
        memory_items: list[MemoryItem],
        max_chars_per_memory: Optional[int] = None,
    ) -> str:
        """
        Format memory items for prompt input.
        """

        if not memory_items:
            return "No accessible memory."

        actual_max_chars = max_chars_per_memory or self.max_chars_per_memory

        formatted_items: list[str] = []

        for index, memory_item in enumerate(memory_items, start=1):
            memory_id = self._get_memory_id(memory_item)
            content = self._get_memory_content(memory_item)
            scope = self._get_memory_scope(memory_item)
            owner = self._get_memory_owner(memory_item)
            status = self._get_memory_status(memory_item)
            memory_type = self._get_memory_type(memory_item)
            tags = self._get_memory_tags(memory_item)

            if len(content) > actual_max_chars:
                content = content[:actual_max_chars].rstrip() + "..."

            formatted_items.append(
                "\n".join(
                    [
                        f"[Memory {index}]",
                        f"memory_id: {memory_id}",
                        f"scope: {scope}",
                        f"owner: {owner}",
                        f"status: {status}",
                        f"type: {memory_type}",
                        f"tags: {tags}",
                        "content:",
                        content,
                    ]
                )
            )

        return "\n\n".join(formatted_items)

    # ------------------------------------------------------------------
    # Result normalization
    # ------------------------------------------------------------------

    def _normalize_retrieval_result(
        self,
        result: Any,
    ) -> list[MemoryItem]:
        """
        Normalize possible MemoryRetriever outputs into list[MemoryItem].

        Supports common return shapes:
        - list[MemoryItem]
        - object with memory_items
        - object with memories
        - object with retrieved_memories
        - list of objects containing memory_item
        """

        if result is None:
            return []

        if isinstance(result, list):
            return self._normalize_memory_list(result)

        for attr_name in [
            "memory_items",
            "memories",
            "retrieved_memories",
            "results",
        ]:
            if hasattr(result, attr_name):
                value = getattr(result, attr_name)
                return self._normalize_memory_list(value)

        raise TypeError(
            "Unsupported memory retrieval result format. "
            "Expected list[MemoryItem] or an object with memory_items/memories."
        )

    def _normalize_memory_list(
        self,
        items: list[Any],
    ) -> list[MemoryItem]:
        """
        Normalize a list that may contain MemoryItem directly or wrapper objects.
        """

        normalized: list[MemoryItem] = []

        for item in items:
            if isinstance(item, MemoryItem):
                normalized.append(item)
                continue

            if hasattr(item, "memory_item"):
                memory_item = getattr(item, "memory_item")
                if isinstance(memory_item, MemoryItem):
                    normalized.append(memory_item)
                    continue

            if hasattr(item, "memory"):
                memory_item = getattr(item, "memory")
                if isinstance(memory_item, MemoryItem):
                    normalized.append(memory_item)
                    continue

            raise TypeError(
                f"Unsupported retrieved memory item type: {type(item)}"
            )

        return normalized

    # ------------------------------------------------------------------
    # Simple fallback ranking
    # ------------------------------------------------------------------

    def _simple_rank_memories(
        self,
        *,
        query: str,
        memory_items: list[MemoryItem],
    ) -> list[MemoryItem]:
        """
        Simple lexical ranking fallback.

        This is only used when MemoryRetriever is not provided.
        Vector retrieval should be preferred in normal workflow.
        """

        query_terms = self._tokenize(query)

        scored_items: list[tuple[int, MemoryItem]] = []

        for memory_item in memory_items:
            content = self._get_memory_content(memory_item)
            tags = " ".join(self._get_memory_tags(memory_item))

            searchable_text = f"{content} {tags}"
            memory_terms = self._tokenize(searchable_text)

            score = len(query_terms.intersection(memory_terms))
            scored_items.append((score, memory_item))

        scored_items.sort(key=lambda pair: pair[0], reverse=True)

        return [item for _, item in scored_items]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """
        Lightweight tokenizer for fallback lexical ranking.
        """

        return {
            token.strip().lower()
            for token in text.replace("\n", " ").split(" ")
            if token.strip()
        }

    # ------------------------------------------------------------------
    # Field extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_agent_id(agent: Agent) -> str:
        for attr_name in ["agent_id", "id"]:
            if hasattr(agent, attr_name):
                value = getattr(agent, attr_name)
                if value is not None:
                    return str(value)

        raise AttributeError("Agent must have agent_id or id.")

    @staticmethod
    def _get_agent_role(agent: Agent) -> str:
        for attr_name in ["role", "agent_role"]:
            if hasattr(agent, attr_name):
                value = getattr(agent, attr_name)
                if value is not None:
                    return str(value)

        return "unknown"

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

    @staticmethod
    def _get_memory_scope(memory_item: MemoryItem) -> str:
        if hasattr(memory_item, "metadata"):
            metadata = getattr(memory_item, "metadata")
            if hasattr(metadata, "scope"):
                return str(getattr(metadata, "scope"))

        if hasattr(memory_item, "scope"):
            return str(getattr(memory_item, "scope"))

        return "unknown"

    @staticmethod
    def _get_memory_owner(memory_item: MemoryItem) -> str:
        if hasattr(memory_item, "metadata"):
            metadata = getattr(memory_item, "metadata")
            for attr_name in ["owner_agent_id", "owner", "owner_id"]:
                if hasattr(metadata, attr_name):
                    value = getattr(metadata, attr_name)
                    if value is not None:
                        return str(value)

        for attr_name in ["owner_agent_id", "owner", "owner_id"]:
            if hasattr(memory_item, attr_name):
                value = getattr(memory_item, attr_name)
                if value is not None:
                    return str(value)

        return "unknown"

    @staticmethod
    def _get_memory_status(memory_item: MemoryItem) -> str:
        if hasattr(memory_item, "metadata"):
            metadata = getattr(memory_item, "metadata")
            if hasattr(metadata, "status"):
                return str(getattr(metadata, "status"))

        if hasattr(memory_item, "status"):
            return str(getattr(memory_item, "status"))

        return "unknown"

    @staticmethod
    def _get_memory_type(memory_item: MemoryItem) -> str:
        if hasattr(memory_item, "metadata"):
            metadata = getattr(memory_item, "metadata")
            for attr_name in ["memory_type", "type"]:
                if hasattr(metadata, attr_name):
                    value = getattr(metadata, attr_name)
                    if value is not None:
                        return str(value)

        for attr_name in ["memory_type", "type"]:
            if hasattr(memory_item, attr_name):
                value = getattr(memory_item, attr_name)
                if value is not None:
                    return str(value)

        return "unknown"

    @staticmethod
    def _get_memory_tags(memory_item: MemoryItem) -> list[str]:
        if hasattr(memory_item, "metadata"):
            metadata = getattr(memory_item, "metadata")
            if hasattr(metadata, "tags"):
                tags = getattr(metadata, "tags")
                if tags is not None:
                    return [str(tag) for tag in tags]

        if hasattr(memory_item, "tags"):
            tags = getattr(memory_item, "tags")
            if tags is not None:
                return [str(tag) for tag in tags]

        return []

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_query(query: str) -> None:
        if not query or not query.strip():
            raise ValueError("query cannot be empty.")