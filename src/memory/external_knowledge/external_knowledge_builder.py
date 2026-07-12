from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from .document_chunker import DocumentChunker
from .knowledge_schema import KnowledgeChunk, KnowledgeContext


class ExternalKnowledgeBuilder:
    """
    Builder for persistent external knowledge vector database.

    This builder converts external knowledge chunks into LangChain Documents,
    embeds them with OpenAIEmbeddings, and stores them in a persistent Chroma
    collection.

    This layer is used for external knowledge retrieval, not governed memory.
    It does not handle:
    - private/shared memory permissions
    - promotion workflow
    - operation logs
    - agent-specific memory access
    """

    def __init__(
        self,
        persist_directory: str | Path = "data/external_knowledge/squad_chroma",
        collection_name: str = "squad_external_knowledge",
        embedding_model: str = "text-embedding-3-small",
        chunker: Optional[DocumentChunker] = None,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunker = chunker or DocumentChunker()

        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
        )

    # ------------------------------------------------------------------
    # Build methods
    # ------------------------------------------------------------------

    def build_from_contexts(
        self,
        contexts: list[KnowledgeContext],
        reset: bool = True,
        batch_size: int = 500,
    ) -> Chroma:
        """
        Build Chroma vector database from KnowledgeContext objects.

        Flow:
            KnowledgeContext -> KnowledgeChunk -> Document -> Chroma
        """

        chunks = self.chunker.split_contexts(contexts)

        return self.build_from_chunks(
            chunks=chunks,
            reset=reset,
            batch_size=batch_size,
        )

    def build_from_chunks(
        self,
        chunks: list[KnowledgeChunk],
        reset: bool = True,
        batch_size: int = 500,
    ) -> Chroma:
        """
        Build Chroma vector database from KnowledgeChunk objects.
        """

        documents = self.chunker.chunks_to_documents(chunks)

        return self.build_from_documents(
            documents=documents,
            reset=reset,
            batch_size=batch_size,
        )

    def build_from_documents(
        self,
        documents: list[Document],
        reset: bool = True,
        batch_size: int = 500,
    ) -> Chroma:
        """
        Build Chroma vector database from LangChain Documents.

        Args:
            documents:
                Documents to embed and store.
            reset:
                If True, remove the existing Chroma directory before building.
            batch_size:
                Number of documents to add per batch.

        Returns:
            Chroma vector store.
        """

        if not documents:
            raise ValueError("Cannot build vector database from empty documents.")

        if reset:
            self.clear()

        self.persist_directory.mkdir(parents=True, exist_ok=True)

        vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory),
        )

        for batch_start in range(0, len(documents), batch_size):
            batch = documents[batch_start: batch_start + batch_size]

            ids = [
                self._document_id(document=document, fallback_index=batch_start + index)
                for index, document in enumerate(batch)
            ]

            vector_store.add_documents(
                documents=batch,
                ids=ids,
            )

        return vector_store

    # ------------------------------------------------------------------
    # Load existing Chroma
    # ------------------------------------------------------------------

    def load_vector_store(self) -> Chroma:
        """
        Load an existing persistent Chroma vector store.
        """

        if not self.persist_directory.exists():
            raise FileNotFoundError(
                f"Chroma persist directory does not exist: {self.persist_directory}"
            )

        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory),
        )

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """
        Delete the persistent Chroma directory.

        Intended for rebuilding the external knowledge base during development.
        """

        if self.persist_directory.exists():
            shutil.rmtree(self.persist_directory)

    def exists(self) -> bool:
        """
        Check whether the Chroma persist directory exists and is non-empty.
        """

        return (
            self.persist_directory.exists()
            and any(self.persist_directory.iterdir())
        )

    def count_documents(self) -> int:
        """
        Count documents in the Chroma collection.

        Returns 0 if the vector database does not exist yet.
        """

        if not self.exists():
            return 0

        vector_store = self.load_vector_store()
        collection = vector_store._collection

        return collection.count()

    def get_collection_info(self) -> dict:
        """
        Return basic information about the current collection.
        """

        return {
            "persist_directory": str(self.persist_directory),
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model,
            "exists": self.exists(),
            "document_count": self.count_documents(),
        }

    @staticmethod
    def _document_id(
        document: Document,
        fallback_index: int,
    ) -> str:
        """
        Choose a stable document ID for Chroma.

        Prefer chunk_id from metadata.
        Fall back to a generated document ID if missing.
        """

        chunk_id = document.metadata.get("chunk_id")

        if chunk_id:
            return str(chunk_id)

        return f"external_doc_{fallback_index:06d}"