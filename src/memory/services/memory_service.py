from __future__ import annotations

from typing import Optional, Any

from ..entities import (
    Agent,
    MemoryItem,
    MemoryMetadata,
    MemoryType,
    SourceType,
    MemoryStatus,
    MemoryScope,
    MemoryOperationRecord,
    MemoryOperationType,
)
from ..manager import MemoryStore, OperationLogStore
from ..retrieval import MemoryRetriever
from .permission_service import PermissionService


class MemoryService:
    """
    Business services for memory operations.

    This services is responsible for:
    - creating private/shared memories
    - reading memories with permission checks
    - updating memory content and metadata
    - soft-deleting memories by marking them as deprecated
    - retrieving accessible memories through RAG
    - recording memory operation logs

    It should be used as the main entry point for memory operations.
    External modules should avoid directly modifying MemoryStore in formal workflow.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        operation_log_store: Optional[OperationLogStore] = None,
        memory_retriever: Optional[MemoryRetriever] = None,
        permission_service: Optional[PermissionService] = None,
    ) -> None:
        self.memory_store = memory_store
        self.operation_log_store = operation_log_store
        self.memory_retriever = memory_retriever
        self.permission_service = permission_service or PermissionService()

    # ------------------------------------------------------------------
    # Create memory
    # ------------------------------------------------------------------

    def create_private_memory(
        self,
        agent: Agent,
        content: str,
        summary: Optional[str] = None,
        memory_type: MemoryType | str = MemoryType.NOTE,
        tags: Optional[list[str]] = None,
        importance: float = 0.5,
        confidence: float = 1.0,
        source_task_id: Optional[str] = None,
        source_message_ids: Optional[list[str]] = None,
        source_type: SourceType | str = SourceType.AGENT_OUTPUT,
        reason: Optional[str] = None,
    ) -> MemoryItem:
        """
        Create a private memory owned by the acting agent.
        """

        self.permission_service.assert_can_create_private_memory(agent)

        metadata = MemoryMetadata.private(
            owner_agent_id=agent.agent_id,
            created_by_agent_id=agent.agent_id,
            memory_type=MemoryType(memory_type),
            tags=tags or [],
            importance=importance,
            confidence=confidence,
            source_task_id=source_task_id,
            source_message_ids=source_message_ids or [],
            source_type=SourceType(source_type),
        )

        memory = MemoryItem(
            content=content,
            summary=summary,
            metadata=metadata,
        )

        created_memory = self.memory_store.create(memory)

        record = self._record_memory_created(
            memory=created_memory,
            actor_agent=agent,
            reason=reason or "Private memory created.",
        )

        if record is not None:
            created_memory = self._attach_operation_to_memory(
                memory=created_memory,
                operation_id=record.record_id,
            )

        return created_memory

    def create_shared_memory(
            self,
            agent: Agent,
            content: str,
            summary: Optional[str] = None,
            memory_type: MemoryType | str = MemoryType.NOTE,
            tags: Optional[list[str]] = None,
            importance: float = 0.5,
            confidence: float = 1.0,
            source_task_id: Optional[str] = None,
            source_message_ids: Optional[list[str]] = None,
            source_type: SourceType | str = SourceType.AGENT_OUTPUT,
            readable_by: Optional[list[str]] = None,
            writable_by: Optional[list[str]] = None,
            reason: Optional[str] = None,
    ) -> MemoryItem:
        """
        Directly create a shared memory owned by the acting agent.

        This method is intended for system-level shared memories directly
        created by an authorised agent. Persona-owned memories should
        normally be created privately and governed through
        MemoryAccessPolicyService.
        """
        self.permission_service.assert_can_create_shared_memory(
            agent
        )

        metadata = MemoryMetadata.shared(
            owner_agent_id=agent.agent_id,
            created_by_agent_id=agent.agent_id,
            readable_by=(
                ["*"]
                if readable_by is None
                else readable_by
            ),
            writable_by=(
                    writable_by
                    or [agent.agent_id]
            ),
            memory_type=MemoryType(memory_type),
            tags=tags or [],
            importance=importance,
            confidence=confidence,
            source_task_id=source_task_id,
            source_message_ids=(
                    source_message_ids or []
            ),
            source_type=SourceType(source_type),
        )

        memory = MemoryItem(
            content=content,
            summary=summary,
            metadata=metadata,
        )

        created_memory = self.memory_store.create(
            memory
        )

        record = self._record_memory_created(
            memory=created_memory,
            actor_agent=agent,
            reason=reason or "Shared memory created.",
        )

        if record is not None:
            created_memory = (
                self._attach_operation_to_memory(
                    memory=created_memory,
                    operation_id=record.record_id,
                )
            )

        return created_memory

    # ------------------------------------------------------------------
    # Read memory
    # ------------------------------------------------------------------

    def get_memory(
        self,
        agent: Agent,
        memory_id: str,
    ) -> MemoryItem:
        """
        Get a memory by ID with permission check.
        """

        memory = self.memory_store.get_by_id(memory_id)
        self.permission_service.assert_can_read_memory(agent, memory)

        return memory

    def list_accessible_memories(
        self,
        agent: Agent,
        include_private: bool = True,
        include_shared: bool = True,
        active_only: bool = True,
    ) -> list[MemoryItem]:
        """
        List memories accessible to the given agent.

        This is structured access filtering, not semantic RAG retrieval.
        """

        memories: list[MemoryItem] = []

        if include_shared:
            memories.extend(
                self.memory_store.list_shared(active_only=active_only)
            )

        if include_private:
            memories.extend(
                self.memory_store.list_private_by_agent(
                    agent_id=agent.agent_id,
                    active_only=active_only,
                )
            )

        result = []

        for memory in memories:
            if self.permission_service.can_read_memory(agent, memory):
                result.append(memory)

        unique = {
            memory.memory_id: memory
            for memory in result
        }

        return list(unique.values())

    def retrieve_memories(
        self,
        agent: Agent,
        query: str,
        top_k: int = 5,
        include_private: bool = True,
        include_shared: bool = True,
    ) -> list[MemoryItem]:
        """
        Retrieve relevant accessible memories through the configured MemoryRetriever.

        This uses RAG retrieval, not only metadata filtering.
        """

        if self.memory_retriever is None:
            raise RuntimeError(
                "MemoryRetriever is not configured for MemoryService."
            )

        return self.memory_retriever.retrieve(
            agent=agent,
            query=query,
            top_k=top_k,
            include_private=include_private,
            include_shared=include_shared,
        )

    # ------------------------------------------------------------------
    # Update memory
    # ------------------------------------------------------------------

    def update_memory_content(
        self,
        agent: Agent,
        memory_id: str,
        content: Optional[str] = None,
        summary: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> MemoryItem:
        """
        Update memory content and/or summary.

        This is a semantic update, so it is logged with before/after state.
        """

        memory = self.memory_store.get_by_id(memory_id)
        self.permission_service.assert_can_write_memory(agent, memory)

        before_state = self._snapshot_memory_content(memory)

        updated_memory = self.memory_store.update_content(
            memory_id=memory_id,
            content=content,
            summary=summary,
        )

        after_state = self._snapshot_memory_content(updated_memory)

        record = self._record_memory_updated(
            memory=updated_memory,
            actor_agent=agent,
            before_state=before_state,
            after_state=after_state,
            reason=reason or "Memory content updated.",
        )

        if record is not None:
            updated_memory = self._attach_operation_to_memory(
                memory=updated_memory,
                operation_id=record.record_id,
            )

        return updated_memory

    def update_memory_metadata(
        self,
        agent: Agent,
        memory_id: str,
        tags: Optional[list[str]] = None,
        importance: Optional[float] = None,
        confidence: Optional[float] = None,
        memory_type: Optional[MemoryType | str] = None,
        source_task_id: Optional[str] = None,
        source_message_ids: Optional[list[str]] = None,
        reason: Optional[str] = None,
    ) -> MemoryItem:
        """
        Update non-access metadata fields.

        Access and scope changes should not normally be done here.
        """

        memory = self.memory_store.get_by_id(memory_id)
        self.permission_service.assert_can_write_memory(agent, memory)

        before_state = self._snapshot_memory_metadata(memory)

        updated_memory = self.memory_store.update_metadata(
            memory_id=memory_id,
            tags=tags,
            importance=importance,
            confidence=confidence,
            memory_type=memory_type,
            source_task_id=source_task_id,
            source_message_ids=source_message_ids,
        )

        after_state = self._snapshot_memory_metadata(updated_memory)

        record = self._record_memory_updated(
            memory=updated_memory,
            actor_agent=agent,
            before_state=before_state,
            after_state=after_state,
            reason=reason or "Memory metadata updated.",
        )

        if record is not None:
            updated_memory = self._attach_operation_to_memory(
                memory=updated_memory,
                operation_id=record.record_id,
            )

        return updated_memory

    # ------------------------------------------------------------------
    # Delete / deprecate memory
    # ------------------------------------------------------------------

    def deprecate_memory(
        self,
        agent: Agent,
        memory_id: str,
        reason: Optional[str] = None,
    ) -> MemoryItem:
        """
        Soft-delete a memory by marking it as deprecated.

        Deprecated memories remain stored, but are excluded from normal retrieval.
        """

        memory = self.memory_store.get_by_id(memory_id)
        self.permission_service.assert_can_write_memory(agent, memory)

        before_state = self._snapshot_memory_status(memory)

        record = self._record_memory_deprecated(
            memory=memory,
            actor_agent=agent,
            before_state=before_state,
            reason=reason or "Memory deprecated.",
        )

        operation_id = record.record_id if record is not None else None

        deprecated_memory = self.memory_store.deprecate(
            memory_id=memory_id,
            operation_id=operation_id,
        )

        after_state = self._snapshot_memory_status(deprecated_memory)

        if record is not None:
            record.after_state = after_state
            self._replace_operation_record(record)

        return deprecated_memory

    # ------------------------------------------------------------------
    # Internal logging helpers
    # ------------------------------------------------------------------

    def _record_memory_created(
        self,
        memory: MemoryItem,
        actor_agent: Agent,
        reason: str,
    ) -> Optional[MemoryOperationRecord]:
        """
        Create and append a memory creation log.
        """

        if self.operation_log_store is None:
            return None

        record = MemoryOperationRecord.create_record(
            memory_id=memory.memory_id,
            actor_agent_id=actor_agent.agent_id,
            after_state=self._snapshot_memory_basic(memory),
            reason=reason,
        )

        self.operation_log_store.append(record)
        return record

    def _record_memory_updated(
        self,
        memory: MemoryItem,
        actor_agent: Agent,
        before_state: dict[str, Any],
        after_state: dict[str, Any],
        reason: str,
    ) -> Optional[MemoryOperationRecord]:
        """
        Create and append a memory update log.
        """

        if self.operation_log_store is None:
            return None

        record = MemoryOperationRecord.update_record(
            memory_id=memory.memory_id,
            actor_agent_id=actor_agent.agent_id,
            before_state=before_state,
            after_state=after_state,
            reason=reason,
        )

        self.operation_log_store.append(record)
        return record

    def _record_memory_deprecated(
        self,
        memory: MemoryItem,
        actor_agent: Agent,
        before_state: dict[str, Any],
        reason: str,
    ) -> Optional[MemoryOperationRecord]:
        """
        Create and append a memory deprecation log.
        """

        if self.operation_log_store is None:
            return None

        record = MemoryOperationRecord(
            operation_type=MemoryOperationType.DEPRECATE,
            target_memory_ids=[memory.memory_id],
            actor_agent_id=actor_agent.agent_id,
            reason=reason,
            before_state=before_state,
            after_state=None,
        )

        self.operation_log_store.append(record)
        return record

    def _replace_operation_record(
        self,
        record: MemoryOperationRecord,
    ) -> None:
        """
        Replace an operation record if the store supports replacement.

        Current OperationLogStore is append-oriented and may not expose replace().
        For now, this method does nothing unless such method exists later.
        """

        if self.operation_log_store is None:
            return

        replace_method = getattr(self.operation_log_store, "replace", None)

        if callable(replace_method):
            replace_method(record)

    def _attach_operation_to_memory(
        self,
        memory: MemoryItem,
        operation_id: str,
    ) -> MemoryItem:
        """
        Attach operation ID to memory metadata and persist the updated memory.
        """

        memory.metadata.last_operation_id = operation_id

        if operation_id not in memory.metadata.related_operation_ids:
            memory.metadata.related_operation_ids.append(operation_id)

        return self.memory_store.replace(memory)

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _snapshot_memory_basic(memory: MemoryItem) -> dict[str, Any]:
        """
        Create a compact snapshot of memory state.
        """

        return {
            "memory_id": memory.memory_id,
            "content": memory.content,
            "summary": memory.summary,
            "scope": memory.metadata.scope.value,
            "owner_agent_id": memory.metadata.owner_agent_id,
            "created_by_agent_id": memory.metadata.created_by_agent_id,
            "status": memory.metadata.status.value,
            "memory_type": memory.metadata.memory_type.value,
            "tags": list(memory.metadata.tags),
            "importance": memory.metadata.importance,
            "confidence": memory.metadata.confidence,
        }

    @staticmethod
    def _snapshot_memory_content(memory: MemoryItem) -> dict[str, Any]:
        """
        Snapshot content-related fields.
        """

        return {
            "memory_id": memory.memory_id,
            "content": memory.content,
            "summary": memory.summary,
        }

    @staticmethod
    def _snapshot_memory_metadata(memory: MemoryItem) -> dict[str, Any]:
        """
        Snapshot non-access metadata fields.
        """

        return {
            "memory_id": memory.memory_id,
            "memory_type": memory.metadata.memory_type.value,
            "tags": list(memory.metadata.tags),
            "importance": memory.metadata.importance,
            "confidence": memory.metadata.confidence,
            "source_task_id": memory.metadata.source_task_id,
            "source_message_ids": list(memory.metadata.source_message_ids),
            "source_type": memory.metadata.source_type.value,
        }

    @staticmethod
    def _snapshot_memory_status(memory: MemoryItem) -> dict[str, Any]:
        """
        Snapshot lifecycle status fields.
        """

        return {
            "memory_id": memory.memory_id,
            "status": memory.metadata.status.value,
            "last_operation_id": memory.metadata.last_operation_id,
            "related_operation_ids": list(memory.metadata.related_operation_ids),
        }