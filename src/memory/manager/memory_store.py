from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..entities import (
    MemoryItem,
    MemoryScope,
    MemoryStatus,
    MemoryType,
)


class MemoryNotFoundError(Exception):
    """Raised when a memory item cannot be found."""


class MemoryAlreadyExistsError(Exception):
    """Raised when trying to create a memory with an existing memory_id."""


class MemoryStore:
    """
    JSON-backed CRUD store for MemoryItem.

    This store persists memory current states into a JSON file.

    It does not handle:
    - permission checks
    - promotion workflow
    - conflict detection
    - operation logging
    - RAG / vector retrieval

    Those responsibilities belong to services or retrieval layers.
    """

    def __init__(self, file_path: str | Path = "data/memories.json") -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.file_path.exists():
            self._write_raw([])

    # -------------------------
    # Internal file operations
    # -------------------------

    def _read_raw(self) -> list[dict]:
        """
        Read raw memory records from the JSON file.
        """
        try:
            with self.file_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError:
            raise ValueError(
                f"Memory file '{self.file_path}' contains invalid JSON."
            )

        if not isinstance(data, list):
            raise ValueError(
                f"Memory file '{self.file_path}' must contain a JSON list."
            )

        return data

    def _write_raw(self, data: list[dict]) -> None:
        """
        Write raw memory records to the JSON file.
        """
        with self.file_path.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

    def _load_memories(self) -> list[MemoryItem]:
        """
        Load all memory records as MemoryItem objects.
        """
        raw_memories = self._read_raw()
        return [
            MemoryItem.model_validate(memory)
            for memory in raw_memories
        ]

    def _save_memories(self, memories: list[MemoryItem]) -> None:
        """
        Save MemoryItem objects into the JSON file.
        """
        raw_memories = [
            memory.model_dump(mode="json")
            for memory in memories
        ]
        self._write_raw(raw_memories)

    # -------------------------
    # Create
    # -------------------------

    def create(self, memory: MemoryItem) -> MemoryItem:
        """
        Save a new memory item.

        Raises:
            MemoryAlreadyExistsError:
                If memory_id already exists.
        """
        memories = self._load_memories()

        if any(existing.memory_id == memory.memory_id for existing in memories):
            raise MemoryAlreadyExistsError(
                f"Memory with id '{memory.memory_id}' already exists."
            )

        memories.append(deepcopy(memory))
        self._save_memories(memories)

        return deepcopy(memory)

    # -------------------------
    # Read
    # -------------------------

    def get_by_id(self, memory_id: str) -> MemoryItem:
        """
        Get a memory by its ID.

        Raises:
            MemoryNotFoundError:
                If memory does not exist.
        """
        memories = self._load_memories()

        for memory in memories:
            if memory.memory_id == memory_id:
                return deepcopy(memory)

        raise MemoryNotFoundError(f"Memory with id '{memory_id}' was not found.")

    def exists(self, memory_id: str) -> bool:
        """
        Check whether a memory exists.
        """
        memories = self._load_memories()
        return any(memory.memory_id == memory_id for memory in memories)

    def list_all(self, include_deprecated: bool = True) -> list[MemoryItem]:
        """
        Return all memories.

        Args:
            include_deprecated:
                If False, deprecated memories are excluded.
        """
        memories = self._load_memories()

        if not include_deprecated:
            memories = [
                memory
                for memory in memories
                if memory.metadata.status != MemoryStatus.DEPRECATED
            ]

        return deepcopy(memories)

    def list_active(self) -> list[MemoryItem]:
        """
        Return all active memories.
        """
        return self.list_by_status(MemoryStatus.ACTIVE)

    def list_by_status(self, status: MemoryStatus | str) -> list[MemoryItem]:
        """
        Return memories with the given lifecycle status.
        """
        status = MemoryStatus(status)
        memories = self._load_memories()

        result = [
            memory
            for memory in memories
            if memory.metadata.status == status
        ]

        return deepcopy(result)

    def list_by_scope(
        self,
        scope: MemoryScope | str,
        active_only: bool = False,
    ) -> list[MemoryItem]:
        """
        Return memories by scope: private or shared.
        """
        scope = MemoryScope(scope)
        memories = self._load_memories()

        result = [
            memory
            for memory in memories
            if memory.metadata.scope == scope
        ]

        if active_only:
            result = [
                memory
                for memory in result
                if memory.metadata.status == MemoryStatus.ACTIVE
            ]

        return deepcopy(result)

    def list_shared(self, active_only: bool = True) -> list[MemoryItem]:
        """
        Return shared memories.
        """
        return self.list_by_scope(MemoryScope.SHARED, active_only=active_only)

    def list_private(self, active_only: bool = True) -> list[MemoryItem]:
        """
        Return all private memories.
        """
        return self.list_by_scope(MemoryScope.PRIVATE, active_only=active_only)

    def list_private_by_agent(
        self,
        agent_id: str,
        active_only: bool = True,
    ) -> list[MemoryItem]:
        """
        Return private memories owned by a specific agent.
        """
        memories = self._load_memories()

        result = [
            memory
            for memory in memories
            if memory.metadata.scope == MemoryScope.PRIVATE
            and memory.metadata.owner_agent_id == agent_id
        ]

        if active_only:
            result = [
                memory
                for memory in result
                if memory.metadata.status == MemoryStatus.ACTIVE
            ]

        return deepcopy(result)

    def list_by_owner(
        self,
        owner_agent_id: str,
        active_only: bool = False,
    ) -> list[MemoryItem]:
        """
        Return memories currently owned by an agent.
        """
        memories = self._load_memories()

        result = [
            memory
            for memory in memories
            if memory.metadata.owner_agent_id == owner_agent_id
        ]

        if active_only:
            result = [
                memory
                for memory in result
                if memory.metadata.status == MemoryStatus.ACTIVE
            ]

        return deepcopy(result)

    def list_by_creator(
        self,
        created_by_agent_id: str,
        active_only: bool = False,
    ) -> list[MemoryItem]:
        """
        Return memories originally created by an agent.
        """
        memories = self._load_memories()

        result = [
            memory
            for memory in memories
            if memory.metadata.created_by_agent_id == created_by_agent_id
        ]

        if active_only:
            result = [
                memory
                for memory in result
                if memory.metadata.status == MemoryStatus.ACTIVE
            ]

        return deepcopy(result)

    def list_by_type(
        self,
        memory_type: MemoryType | str,
        active_only: bool = False,
    ) -> list[MemoryItem]:
        """
        Return memories by lightweight memory type.
        """
        memory_type = MemoryType(memory_type)
        memories = self._load_memories()

        result = [
            memory
            for memory in memories
            if memory.metadata.memory_type == memory_type
        ]

        if active_only:
            result = [
                memory
                for memory in result
                if memory.metadata.status == MemoryStatus.ACTIVE
            ]

        return deepcopy(result)

    def list_by_tags(
        self,
        tags: list[str],
        match_all: bool = False,
        active_only: bool = False,
    ) -> list[MemoryItem]:
        """
        Return memories by tags.

        Args:
            tags:
                Tags to search for.
            match_all:
                If True, memory must contain all given tags.
                If False, memory only needs to contain at least one tag.
            active_only:
                If True, only active memories are returned.
        """
        if not tags:
            return self.list_active() if active_only else self.list_all()

        memories = self._load_memories()
        query_tags = set(tags)

        result = []

        for memory in memories:
            memory_tags = set(memory.metadata.tags)

            if match_all:
                matched = query_tags.issubset(memory_tags)
            else:
                matched = bool(query_tags.intersection(memory_tags))

            if matched:
                result.append(memory)

        if active_only:
            result = [
                memory
                for memory in result
                if memory.metadata.status == MemoryStatus.ACTIVE
            ]

        return deepcopy(result)

    def filter(
        self,
        *,
        scope: Optional[MemoryScope | str] = None,
        owner_agent_id: Optional[str] = None,
        created_by_agent_id: Optional[str] = None,
        status: Optional[MemoryStatus | str] = None,
        memory_type: Optional[MemoryType | str] = None,
        tags: Optional[list[str]] = None,
        match_all_tags: bool = False,
        source_task_id: Optional[str] = None,
        active_only: bool = False,
    ) -> list[MemoryItem]:
        """
        General metadata-based filtering.

        This is structured filtering, not semantic retrieval.
        RAG / similarity search should be implemented in a retriever layer.
        """
        result = self._load_memories()

        if scope is not None:
            scope = MemoryScope(scope)
            result = [
                memory
                for memory in result
                if memory.metadata.scope == scope
            ]

        if owner_agent_id is not None:
            result = [
                memory
                for memory in result
                if memory.metadata.owner_agent_id == owner_agent_id
            ]

        if created_by_agent_id is not None:
            result = [
                memory
                for memory in result
                if memory.metadata.created_by_agent_id == created_by_agent_id
            ]

        if status is not None:
            status = MemoryStatus(status)
            result = [
                memory
                for memory in result
                if memory.metadata.status == status
            ]

        if memory_type is not None:
            memory_type = MemoryType(memory_type)
            result = [
                memory
                for memory in result
                if memory.metadata.memory_type == memory_type
            ]

        if source_task_id is not None:
            result = [
                memory
                for memory in result
                if memory.metadata.source_task_id == source_task_id
            ]

        if tags:
            query_tags = set(tags)

            if match_all_tags:
                result = [
                    memory
                    for memory in result
                    if query_tags.issubset(set(memory.metadata.tags))
                ]
            else:
                result = [
                    memory
                    for memory in result
                    if query_tags.intersection(set(memory.metadata.tags))
                ]

        if active_only:
            result = [
                memory
                for memory in result
                if memory.metadata.status == MemoryStatus.ACTIVE
            ]

        return deepcopy(result)

    # -------------------------
    # Update
    # -------------------------

    def replace(self, memory: MemoryItem) -> MemoryItem:
        """
        Replace an existing memory item.

        This is a low-level method.
        Business-level update logic should be handled in MemoryService.
        """
        memories = self._load_memories()

        for index, existing in enumerate(memories):
            if existing.memory_id == memory.memory_id:
                memory.metadata.updated_at = datetime.now(timezone.utc)
                memories[index] = deepcopy(memory)
                self._save_memories(memories)
                return deepcopy(memory)

        raise MemoryNotFoundError(
            f"Memory with id '{memory.memory_id}' was not found."
        )

    def update_content(
        self,
        memory_id: str,
        *,
        content: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> MemoryItem:
        """
        Update the semantic content of a memory.

        This modifies content and/or summary.
        """
        memory = self.get_by_id(memory_id)

        if content is not None:
            if not content.strip():
                raise ValueError("Memory content cannot be empty.")
            memory.content = content

        if summary is not None:
            memory.summary = summary

        memory.metadata.updated_at = datetime.now(timezone.utc)

        return self.replace(memory)

    def update_metadata(
        self,
        memory_id: str,
        *,
        tags: Optional[list[str]] = None,
        importance: Optional[float] = None,
        confidence: Optional[float] = None,
        memory_type: Optional[MemoryType | str] = None,
        source_task_id: Optional[str] = None,
        source_message_ids: Optional[list[str]] = None,
    ) -> MemoryItem:
        """
        Update non-access metadata fields.

        Access and scope changes should be handled by dedicated services methods.
        """
        memory = self.get_by_id(memory_id)

        if tags is not None:
            memory.metadata.tags = list(dict.fromkeys(tags))

        if importance is not None:
            if not 0.0 <= importance <= 1.0:
                raise ValueError("importance must be between 0.0 and 1.0.")
            memory.metadata.importance = importance

        if confidence is not None:
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be between 0.0 and 1.0.")
            memory.metadata.confidence = confidence

        if memory_type is not None:
            memory.metadata.memory_type = MemoryType(memory_type)

        if source_task_id is not None:
            memory.metadata.source_task_id = source_task_id

        if source_message_ids is not None:
            memory.metadata.source_message_ids = list(dict.fromkeys(source_message_ids))

        memory.metadata.updated_at = datetime.now(timezone.utc)

        return self.replace(memory)

    def update_status(
        self,
        memory_id: str,
        status: MemoryStatus | str,
        operation_id: Optional[str] = None,
    ) -> MemoryItem:
        """
        Update memory lifecycle status.
        """
        memory = self.get_by_id(memory_id)

        memory.metadata.status = MemoryStatus(status)
        memory.metadata.updated_at = datetime.now(timezone.utc)

        if operation_id is not None:
            memory.metadata.last_operation_id = operation_id
            if operation_id not in memory.metadata.related_operation_ids:
                memory.metadata.related_operation_ids.append(operation_id)

        return self.replace(memory)

    def update_access(
        self,
        memory_id: str,
        *,
        owner_agent_id: Optional[str] = None,
        readable_by: Optional[list[str]] = None,
        writable_by: Optional[list[str]] = None,
    ) -> MemoryItem:
        """
        Update access-related metadata.

        This is low-level. Permission checks should be done in the services layer.
        """
        memory = self.get_by_id(memory_id)

        if owner_agent_id is not None:
            memory.metadata.owner_agent_id = owner_agent_id

        if readable_by is not None:
            memory.metadata.readable_by = list(dict.fromkeys(readable_by))

        if writable_by is not None:
            memory.metadata.writable_by = list(dict.fromkeys(writable_by))

        memory.metadata.updated_at = datetime.now(timezone.utc)

        return self.replace(memory)

    def update_scope(
        self,
        memory_id: str,
        scope: MemoryScope | str,
        *,
        owner_agent_id: Optional[str] = None,
        readable_by: Optional[list[str]] = None,
        writable_by: Optional[list[str]] = None,
    ) -> MemoryItem:
        """
        Update memory scope.

        Use this carefully. In normal business flow, private -> shared promotion
        should be handled by PromotionService.
        """
        memory = self.get_by_id(memory_id)

        memory.metadata.scope = MemoryScope(scope)

        if owner_agent_id is not None:
            memory.metadata.owner_agent_id = owner_agent_id

        if readable_by is not None:
            memory.metadata.readable_by = list(dict.fromkeys(readable_by))

        if writable_by is not None:
            memory.metadata.writable_by = list(dict.fromkeys(writable_by))

        memory.metadata.updated_at = datetime.now(timezone.utc)

        return self.replace(memory)

    # -------------------------
    # Delete
    # -------------------------

    def deprecate(
        self,
        memory_id: str,
        operation_id: Optional[str] = None,
    ) -> MemoryItem:
        """
        Soft delete a memory by marking it as deprecated.

        Deprecated memories remain stored but are excluded from active retrieval.
        """
        return self.update_status(
            memory_id=memory_id,
            status=MemoryStatus.DEPRECATED,
            operation_id=operation_id,
        )

    def hard_delete(self, memory_id: str) -> None:
        """
        Permanently delete a memory.

        This should only be used for tests or development reset.
        Formal business logic should use deprecate().
        """
        memories = self._load_memories()

        new_memories = [
            memory
            for memory in memories
            if memory.memory_id != memory_id
        ]

        if len(new_memories) == len(memories):
            raise MemoryNotFoundError(f"Memory with id '{memory_id}' was not found.")

        self._save_memories(new_memories)

    def clear(self) -> None:
        """
        Clear all memories.

        Intended for testing or demo reset.
        """
        self._write_raw([])

    # -------------------------
    # Utility
    # -------------------------

    def count(self) -> int:
        """
        Return the total number of stored memories.
        """
        return len(self._load_memories())

    def export_as_dicts(self) -> list[dict]:
        """
        Export all memories as plain dictionaries.

        Useful for debugging, tests, or displaying JSON data.
        """
        return [
            memory.model_dump(mode="json")
            for memory in self._load_memories()
        ]