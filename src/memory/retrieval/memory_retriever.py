from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from ..entities import Agent, MemoryItem, MemoryStatus
from ..manager import MemoryStore
from .document_converter import MemoryDocumentConverter


class MemoryRetriever:
    """
    LangChain-based memory retriever.

    This retriever uses access-first retrieval strategy:

    1. Select accessible memories for the requesting agent.
    2. Build a temporary Chroma vector store from only those memories.
    3. Run similarity search on the accessible memory set.

    This prevents private memories from being indexed into another agent's
    retrieval context.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        embedding_model: str = "text-embedding-3-small",
        persist_directory: Optional[str | Path] = None,
        collection_name: str = "memory_retrieval",
    ) -> None:
        self.memory_store = memory_store
        self.embedding_model = embedding_model
        self.persist_directory = Path(persist_directory) if persist_directory else None
        self.collection_name = collection_name

        self.embeddings = OpenAIEmbeddings(model=embedding_model)

    def get_accessible_memories(
        self,
        agent: Agent,
        include_private: bool = True,
        include_shared: bool = True,
        active_only: bool = True,
    ) -> list[MemoryItem]:
        """
        Return memories that the agent is allowed to retrieve.

        Baseline access rule:
        - active shared memories are accessible to all agents
        - active private memories are accessible only to their owner
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

        # Extra safety filter.
        # Even if MemoryStore methods change later, this prevents obvious leakage.
        safe_memories = []

        for memory in memories:
            if active_only and memory.metadata.status != MemoryStatus.ACTIVE:
                continue

            if memory.can_be_read_by(agent.agent_id):
                safe_memories.append(memory)

        # Remove duplicates by memory_id.
        unique: dict[str, MemoryItem] = {
            memory.memory_id: memory
            for memory in safe_memories
        }

        return list(unique.values())

    def retrieve(
        self,
        agent: Agent,
        query: str,
        top_k: int = 5,
        include_private: bool = True,
        include_shared: bool = True,
    ) -> list[MemoryItem]:
        """
        Retrieve top-k relevant accessible memories for an agent.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        accessible_memories = self.get_accessible_memories(
            agent=agent,
            include_private=include_private,
            include_shared=include_shared,
            active_only=True,
        )

        if not accessible_memories:
            return []

        documents = MemoryDocumentConverter.memories_to_documents(accessible_memories)

        vector_store = self._build_vector_store(documents)

        results = vector_store.similarity_search(
            query=query,
            k=min(top_k, len(documents)),
        )

        return self._documents_to_memories(results)

    def retrieve_with_scores(
        self,
        agent: Agent,
        query: str,
        top_k: int = 5,
        include_private: bool = True,
        include_shared: bool = True,
    ) -> list[tuple[MemoryItem, float]]:
        """
        Retrieve top-k relevant accessible memories with similarity scores.

        Lower distance usually means more similar for Chroma.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        accessible_memories = self.get_accessible_memories(
            agent=agent,
            include_private=include_private,
            include_shared=include_shared,
            active_only=True,
        )

        if not accessible_memories:
            return []

        documents = MemoryDocumentConverter.memories_to_documents(accessible_memories)

        vector_store = self._build_vector_store(documents)

        results = vector_store.similarity_search_with_score(
            query=query,
            k=min(top_k, len(documents)),
        )

        output: list[tuple[MemoryItem, float]] = []

        for document, score in results:
            memory_id = MemoryDocumentConverter.document_to_memory_id(document)
            memory = self.memory_store.get_by_id(memory_id)
            output.append((memory, score))

        return output

    def retrieve_documents(
        self,
        agent: Agent,
        query: str,
        top_k: int = 5,
        include_private: bool = True,
        include_shared: bool = True,
    ):
        """
        Retrieve raw LangChain Documents.

        Useful for debugging the retrieval result and metadata.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        accessible_memories = self.get_accessible_memories(
            agent=agent,
            include_private=include_private,
            include_shared=include_shared,
            active_only=True,
        )

        if not accessible_memories:
            return []

        documents = MemoryDocumentConverter.memories_to_documents(accessible_memories)
        vector_store = self._build_vector_store(documents)

        return vector_store.similarity_search(
            query=query,
            k=min(top_k, len(documents)),
        )

    def _build_vector_store(self, documents):
        """
        Build a Chroma vector store from documents.

        For the current baseline, this is rebuilt per retrieval call using
        only accessible memories.
        """
        if self.persist_directory:
            self.persist_directory.mkdir(parents=True, exist_ok=True)

            return Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=str(self.persist_directory),
            )

        return Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
        )

    def _documents_to_memories(self, documents) -> list[MemoryItem]:
        memories: list[MemoryItem] = []

        for document in documents:
            memory_id = MemoryDocumentConverter.document_to_memory_id(document)
            memory = self.memory_store.get_by_id(memory_id)
            memories.append(memory)

        return memories