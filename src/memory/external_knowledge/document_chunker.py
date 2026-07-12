from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .knowledge_schema import (
    KnowledgeContext,
    KnowledgeChunk,
)


class DocumentChunker:
    """
    Chunker for external knowledge contexts.

    This module converts deduplicated KnowledgeContext objects into smaller
    KnowledgeChunk objects, and then into LangChain Document objects for Chroma.

    It belongs to the external knowledge layer, not the governed memory layer.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        separators: Optional[list[str]] = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
            add_start_index=True,
        )

    # ------------------------------------------------------------------
    # Context -> Chunk
    # ------------------------------------------------------------------

    def split_context(
        self,
        context: KnowledgeContext,
    ) -> list[KnowledgeChunk]:
        """
        Split one KnowledgeContext into KnowledgeChunk objects.
        """

        if not context.content.strip():
            return []

        documents = self.text_splitter.create_documents(
            texts=[context.content],
            metadatas=[
                {
                    "context_id": context.context_id,
                    "title": context.title,
                    "source": context.source,
                    "context_hash": context.context_hash,
                    "sample_ids": context.sample_ids_text(),
                }
            ],
        )

        chunks: list[KnowledgeChunk] = []

        for index, document in enumerate(documents):
            metadata = document.metadata

            start_index = metadata.get("start_index")
            end_index = None

            if isinstance(start_index, int):
                end_index = start_index + len(document.page_content)

            chunk = KnowledgeChunk(
                chunk_id=self._make_chunk_id(
                    context_id=context.context_id,
                    chunk_index=index,
                ),
                context_id=context.context_id,
                title=context.title,
                content=document.page_content,
                source=context.source,
                chunk_index=index,
                context_hash=context.context_hash,
                sample_ids=list(context.sample_ids),
                start_index=start_index if isinstance(start_index, int) else None,
                end_index=end_index,
            )

            chunks.append(chunk)

        return chunks

    def split_contexts(
        self,
        contexts: list[KnowledgeContext],
    ) -> list[KnowledgeChunk]:
        """
        Split multiple KnowledgeContext objects into KnowledgeChunk objects.
        """

        all_chunks: list[KnowledgeChunk] = []

        for context in contexts:
            all_chunks.extend(self.split_context(context))

        return all_chunks

    # ------------------------------------------------------------------
    # Chunk -> LangChain Document
    # ------------------------------------------------------------------

    @staticmethod
    def chunk_to_document(
        chunk: KnowledgeChunk,
    ) -> Document:
        """
        Convert one KnowledgeChunk into a LangChain Document.

        Document.page_content stores the chunk text.
        Document.metadata stores Chroma-compatible primitive metadata.
        """

        return Document(
            page_content=chunk.content,
            metadata=chunk.to_metadata(),
        )

    @classmethod
    def chunks_to_documents(
        cls,
        chunks: list[KnowledgeChunk],
    ) -> list[Document]:
        """
        Convert KnowledgeChunk objects into LangChain Documents.
        """

        return [
            cls.chunk_to_document(chunk)
            for chunk in chunks
        ]

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def contexts_to_documents(
        self,
        contexts: list[KnowledgeContext],
    ) -> list[Document]:
        """
        Split contexts and directly convert them into LangChain Documents.
        """

        chunks = self.split_contexts(contexts)
        return self.chunks_to_documents(chunks)

    def split_text(
        self,
        text: str,
    ) -> list[str]:
        """
        Split plain text into chunks.

        Useful for quick debugging.
        """

        if not text.strip():
            return []

        return self.text_splitter.split_text(text)

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    @staticmethod
    def summarize_chunks(
        chunks: list[KnowledgeChunk],
    ) -> dict[str, int]:
        """
        Return simple statistics for a list of chunks.
        """

        if not chunks:
            return {
                "chunk_count": 0,
                "min_length": 0,
                "max_length": 0,
                "avg_length": 0,
            }

        lengths = [
            len(chunk.content)
            for chunk in chunks
        ]

        return {
            "chunk_count": len(chunks),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "avg_length": int(sum(lengths) / len(lengths)),
        }

    @staticmethod
    def _make_chunk_id(
        context_id: str,
        chunk_index: int,
    ) -> str:
        return f"{context_id}_chunk_{chunk_index:03d}"