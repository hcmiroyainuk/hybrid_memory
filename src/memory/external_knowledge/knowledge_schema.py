from __future__ import annotations

import hashlib
from typing import Any, Optional

from pydantic import BaseModel, Field

class SquadSample(BaseModel):
    """
    One QA sample from SQuAD.

    This is the normalized structure used inside this project.
    It avoids depending directly on the raw Hugging Face / JSON dataset format.
    """

    sample_id: str
    title: str
    context: str
    question: str
    answers: list[str] = Field(default_factory=list)

    @property
    def context_hash(self) -> str:
        """
        Stable hash for deduplicating identical contexts.
        """

        return hashlib.md5(
            self.context.strip().encode("utf-8")
        ).hexdigest()

    def has_answer(self) -> bool:
        """
        Return True if this sample has at least one ground-truth answer.

        Useful if you later use SQuAD v2.0 and want to filter unanswerable samples.
        """

        return len(self.answers) > 0

    def primary_answer(self) -> Optional[str]:
        """
        Return the first ground-truth answer if available.
        """

        if not self.answers:
            return None

        return self.answers[0]


class KnowledgeContext(BaseModel):
    """
    A deduplicated external knowledge context.

    One KnowledgeContext may be linked to multiple SQuAD questions,
    because SQuAD often has several questions for the same passage.
    """

    context_id: str
    title: str
    content: str
    source: str = "squad"
    context_hash: str
    sample_ids: list[str] = Field(default_factory=list)

    @classmethod
    def from_sample(
        cls,
        sample: SquadSample,
        context_id: str,
    ) -> "KnowledgeContext":
        """
        Create a KnowledgeContext from one SQuAD sample.
        """

        return cls(
            context_id=context_id,
            title=sample.title,
            content=sample.context,
            source="squad",
            context_hash=sample.context_hash,
            sample_ids=[sample.sample_id],
        )

    def add_sample_id(self, sample_id: str) -> None:
        """
        Add a SQuAD sample ID linked to this context.
        """

        if sample_id not in self.sample_ids:
            self.sample_ids.append(sample_id)

    def sample_ids_text(self) -> str:
        """
        Return sample IDs as a comma-separated string.

        Chroma metadata is safer with primitive values,
        so list fields should be converted before storing.
        """

        return ",".join(self.sample_ids)


class KnowledgeChunk(BaseModel):
    """
    A chunk produced from a KnowledgeContext.

    This is the unit stored in Chroma and retrieved by Worker B.
    """

    chunk_id: str
    context_id: str
    title: str
    content: str
    source: str = "squad"
    chunk_index: int
    context_hash: str
    sample_ids: list[str] = Field(default_factory=list)

    start_index: Optional[int] = None
    end_index: Optional[int] = None

    def sample_ids_text(self) -> str:
        """
        Return sample IDs as a comma-separated string for vector-store metadata.
        """

        return ",".join(self.sample_ids)

    def to_metadata(self) -> dict[str, Any]:
        """
        Convert this chunk into Chroma-compatible metadata.

        Avoid storing list values directly in metadata.
        """

        return {
            "chunk_id": self.chunk_id,
            "context_id": self.context_id,
            "title": self.title,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "context_hash": self.context_hash,
            "sample_ids": self.sample_ids_text(),
            "start_index": self.start_index if self.start_index is not None else -1,
            "end_index": self.end_index if self.end_index is not None else -1,
        }


class RetrievedKnowledge(BaseModel):
    """
    A retrieved external knowledge chunk.

    This is the object passed from ExternalKnowledgeRetriever to Worker B.
    The workflow and LLM layer should use this instead of depending directly
    on LangChain Document objects.
    """

    chunk_id: str
    context_id: str
    title: str
    content: str
    source: str = "squad"
    score: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_document(
        cls,
        content: str,
        metadata: dict[str, Any],
        score: Optional[float] = None,
    ) -> "RetrievedKnowledge":
        """
        Create RetrievedKnowledge from a LangChain Document-like object.

        The caller passes document.page_content as content and document.metadata
        as metadata.
        """

        return cls(
            chunk_id=str(metadata.get("chunk_id", "")),
            context_id=str(metadata.get("context_id", "")),
            title=str(metadata.get("title", "")),
            content=content,
            source=str(metadata.get("source", "squad")),
            score=score,
            metadata=dict(metadata),
        )

    def format_for_prompt(self, index: int) -> str:
        """
        Format retrieved knowledge for Worker B / Critic / Coordinator prompts.
        """

        score_text = (
            f"{self.score:.4f}"
            if self.score is not None
            else "N/A"
        )

        return (
            f"[Evidence {index}]\n"
            f"chunk_id: {self.chunk_id}\n"
            f"context_id: {self.context_id}\n"
            f"title: {self.title}\n"
            f"score: {score_text}\n"
            f"content:\n{self.content}"
        )


class RetrievalResult(BaseModel):
    """
    Retrieval result for one question.

    This wrapper is useful for logging, debugging, and later error analysis.
    """

    query: str
    top_k: int
    retrieved_chunks: list[RetrievedKnowledge] = Field(default_factory=list)

    def retrieved_chunk_ids(self) -> list[str]:
        """
        Return IDs of retrieved chunks.
        """

        return [
            chunk.chunk_id
            for chunk in self.retrieved_chunks
        ]

    def retrieved_context_hashes(self) -> list[str]:
        """
        Return context hashes from retrieved chunks if available.
        """

        hashes: list[str] = []

        for chunk in self.retrieved_chunks:
            context_hash = chunk.metadata.get("context_hash")
            if context_hash is not None:
                hashes.append(str(context_hash))

        return hashes

    def contains_context_hash(self, context_hash: str) -> bool:
        """
        Check whether retrieval hit a specific source context.

        Useful for evaluating whether the gold SQuAD context was retrieved.
        """

        return context_hash in self.retrieved_context_hashes()

    def format_evidence_for_prompt(self) -> str:
        """
        Format all retrieved chunks as one evidence block for LLM prompts.
        """

        if not self.retrieved_chunks:
            return "No retrieved evidence."

        return "\n\n".join(
            chunk.format_for_prompt(index=i + 1)
            for i, chunk in enumerate(self.retrieved_chunks)
        )