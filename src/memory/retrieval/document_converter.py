from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from ..entities import MemoryItem


class MemoryDocumentConverter:
    """
    Converts MemoryItem objects into LangChain Document objects.

    In this project:
    - Document.page_content stores the memory content.
    - Document.metadata stores memory governance metadata.
    """

    @staticmethod
    def memory_to_document(memory: MemoryItem) -> Document:
        metadata = memory.metadata

        tags_text = ",".join(metadata.tags)

        document_metadata: dict[str, Any] = {
            "memory_id": memory.memory_id,
            "summary": memory.summary or "",
            "scope": metadata.scope.value if hasattr(metadata.scope, "value") else str(metadata.scope),
            "owner_agent_id": metadata.owner_agent_id,
            "created_by_agent_id": metadata.created_by_agent_id,
            "status": metadata.status.value if hasattr(metadata.status, "value") else str(metadata.status),
            "memory_type": metadata.memory_type.value if hasattr(metadata.memory_type, "value") else str(metadata.memory_type),
            "tags": tags_text,
            "tags_text": tags_text,
            "importance": float(metadata.importance),
            "confidence": float(metadata.confidence),
            "source_task_id": metadata.source_task_id or "",
            "source_type": metadata.source_type.value if hasattr(metadata.source_type, "value") else str(metadata.source_type),
            "created_at": metadata.created_at.isoformat() if metadata.created_at else "",
            "updated_at": metadata.updated_at.isoformat() if metadata.updated_at else "",
        }

        # Add summary and tags into page_content to make retrieval easier.
        # This is useful because short memories may not contain all retrieval keywords.
        page_content = MemoryDocumentConverter._build_page_content(memory)

        return Document(
            page_content=page_content,
            metadata=document_metadata,
        )

    @staticmethod
    def memories_to_documents(memories: list[MemoryItem]) -> list[Document]:
        return [
            MemoryDocumentConverter.memory_to_document(memory)
            for memory in memories
        ]

    @staticmethod
    def document_to_memory_id(document: Document) -> str:
        memory_id = document.metadata.get("memory_id")

        if not memory_id:
            raise ValueError("Document metadata does not contain memory_id.")

        return str(memory_id)

    @staticmethod
    def _build_page_content(memory: MemoryItem) -> str:
        metadata = memory.metadata

        parts = [
            f"Content: {memory.content}",
        ]

        if memory.summary:
            parts.append(f"Summary: {memory.summary}")

        if metadata.tags:
            parts.append(f"Tags: {', '.join(metadata.tags)}")

        parts.append(f"Memory type: {metadata.memory_type.value if hasattr(metadata.memory_type, 'value') else str(metadata.memory_type)}")
        parts.append(f"Scope: {metadata.scope.value if hasattr(metadata.scope, 'value') else str(metadata.scope)}")

        return "\n".join(parts)