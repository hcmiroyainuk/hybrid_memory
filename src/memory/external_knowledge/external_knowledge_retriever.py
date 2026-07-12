from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from .knowledge_schema import (
    RetrievedKnowledge,
    RetrievalResult,
)


class ExternalKnowledgeRetriever:
    """
    Retriever for persistent external knowledge vector database.

    This retriever loads a Chroma collection and retrieves relevant chunks
    for a query.

    It is used by Worker B in the RAG QA workflow.

    This retriever does not handle:
    - private/shared memory access
    - memory promotion
    - operation logging
    - agent-specific permissions

    Those belong to the governed memory layer.
    """

    def __init__(
        self,
        persist_directory: str | Path = "data/external_knowledge/squad_chroma",
        collection_name: str = "squad_external_knowledge",
        embedding_model: str = "text-embedding-3-small",
        default_top_k: int = 3,
    ) -> None:
        if default_top_k <= 0:
            raise ValueError("default_top_k must be greater than 0.")

        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.default_top_k = default_top_k

        if not self.persist_directory.exists():
            raise FileNotFoundError(
                f"Chroma persist directory does not exist: {self.persist_directory}. "
                "Build the external knowledge base before using the retriever."
            )

        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
        )

        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory),
        )

    # ------------------------------------------------------------------
    # Main retrieval methods
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Retrieve top-k relevant knowledge chunks for a query.

        Returns:
            RetrievalResult containing RetrievedKnowledge objects.
        """

        self._validate_query(query)

        top_k = top_k or self.default_top_k
        self._validate_top_k(top_k)

        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
        )

        retrieved_chunks = [
            self._document_to_retrieved_knowledge(
                document=document,
                score=score,
            )
            for document, score in results
        ]

        return RetrievalResult(
            query=query,
            top_k=top_k,
            retrieved_chunks=retrieved_chunks,
        )

    def retrieve_chunks(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[RetrievedKnowledge]:
        """
        Convenience method returning only retrieved chunks.
        """

        return self.retrieve(
            query=query,
            top_k=top_k,
        ).retrieved_chunks

    def retrieve_documents(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[Document]:
        """
        Retrieve raw LangChain Documents.

        Useful for debugging Chroma metadata or inspecting retrieval behavior.
        """

        self._validate_query(query)

        top_k = top_k or self.default_top_k
        self._validate_top_k(top_k)

        return self.vector_store.similarity_search(
            query=query,
            k=top_k,
        )

    def retrieve_documents_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[tuple[Document, float]]:
        """
        Retrieve raw LangChain Documents with Chroma scores.

        Lower score usually means closer / more similar in Chroma distance.
        """

        self._validate_query(query)

        top_k = top_k or self.default_top_k
        self._validate_top_k(top_k)

        return self.vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
        )

    # ------------------------------------------------------------------
    # Metadata-filtered retrieval
    # ------------------------------------------------------------------

    def retrieve_by_title(
        self,
        query: str,
        title: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Retrieve chunks for a query, filtered by article title.

        This is useful for debugging or controlled experiments.
        """

        self._validate_query(query)

        if not title.strip():
            raise ValueError("title cannot be empty.")

        top_k = top_k or self.default_top_k
        self._validate_top_k(top_k)

        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter={"title": title},
        )

        retrieved_chunks = [
            self._document_to_retrieved_knowledge(document, score)
            for document, score in results
        ]

        return RetrievalResult(
            query=query,
            top_k=top_k,
            retrieved_chunks=retrieved_chunks,
        )

    def retrieve_by_context_id(
        self,
        query: str,
        context_id: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Retrieve chunks for a query, filtered by context_id.

        Useful for checking whether chunking works for a specific context.
        """

        self._validate_query(query)

        if not context_id.strip():
            raise ValueError("context_id cannot be empty.")

        top_k = top_k or self.default_top_k
        self._validate_top_k(top_k)

        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter={"context_id": context_id},
        )

        retrieved_chunks = [
            self._document_to_retrieved_knowledge(document, score)
            for document, score in results
        ]

        return RetrievalResult(
            query=query,
            top_k=top_k,
            retrieved_chunks=retrieved_chunks,
        )

    def retrieve_by_context_hash(
        self,
        query: str,
        context_hash: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Retrieve chunks for a query, filtered by context_hash.

        Useful for retrieval hit analysis against the gold SQuAD context.
        """

        self._validate_query(query)

        if not context_hash.strip():
            raise ValueError("context_hash cannot be empty.")

        top_k = top_k or self.default_top_k
        self._validate_top_k(top_k)

        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter={"context_hash": context_hash},
        )

        retrieved_chunks = [
            self._document_to_retrieved_knowledge(document, score)
            for document, score in results
        ]

        return RetrievalResult(
            query=query,
            top_k=top_k,
            retrieved_chunks=retrieved_chunks,
        )

    # ------------------------------------------------------------------
    # Collection information
    # ------------------------------------------------------------------

    def count_documents(self) -> int:
        """
        Count documents in the Chroma collection.
        """

        return self.vector_store._collection.count()

    def get_collection_info(self) -> dict:
        """
        Return basic collection information.
        """

        return {
            "persist_directory": str(self.persist_directory),
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model,
            "document_count": self.count_documents(),
            "default_top_k": self.default_top_k,
        }

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def retrieve_evidence_text(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> str:
        """
        Retrieve chunks and format them as a prompt-ready evidence block.
        """

        result = self.retrieve(
            query=query,
            top_k=top_k,
        )

        return result.format_evidence_for_prompt()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _document_to_retrieved_knowledge(
        document: Document,
        score: Optional[float] = None,
    ) -> RetrievedKnowledge:
        """
        Convert a LangChain Document into RetrievedKnowledge.
        """

        return RetrievedKnowledge.from_document(
            content=document.page_content,
            metadata=document.metadata,
            score=score,
        )

    @staticmethod
    def _validate_query(query: str) -> None:
        if not query or not query.strip():
            raise ValueError("query cannot be empty.")

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")