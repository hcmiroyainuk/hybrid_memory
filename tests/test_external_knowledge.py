from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from src.memory.external_knowledge import (
    SquadLoader,
    DocumentChunker,
    ExternalKnowledgeBuilder,
    ExternalKnowledgeRetriever,
    SquadSample,
    KnowledgeContext,
    KnowledgeChunk,
    RetrievedKnowledge,
    RetrievalResult,
)


TEST_PERSIST_DIR = Path("data/test_external_knowledge/squad_chroma")
COLLECTION_NAME = "test_squad_external_knowledge"
EMBEDDING_MODEL = "text-embedding-3-small"


@pytest.mark.integration
def test_external_knowledge_layer_end_to_end() -> None:
    """
    End-to-end integration test for the external knowledge layer.

    This test validates:
    - SQuAD loading through KaggleHub
    - sample normalization
    - unique context extraction
    - context chunking
    - LangChain Document conversion
    - persistent Chroma vector DB construction
    - top-k retrieval
    - prompt-ready evidence formatting
    """

    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set. Skipping embedding-based test.")

    # ------------------------------------------------------------------
    # 1. Load SQuAD samples
    # ------------------------------------------------------------------

    loader = SquadLoader(dataset_dir="data")

    samples = loader.load_samples_from_file(
        file_path="data/train-v2.0.json",
        limit=20,
        answerable_only=True,
    )

    assert len(samples) > 0
    assert all(isinstance(sample, SquadSample) for sample in samples)

    first_sample = samples[0]

    assert first_sample.sample_id
    assert first_sample.title
    assert first_sample.context
    assert first_sample.question
    assert len(first_sample.answers) > 0
    assert first_sample.has_answer() is True
    assert first_sample.primary_answer() is not None
    assert first_sample.context_hash

    print("\nFirst sample")
    print("-" * 80)
    print("ID:", first_sample.sample_id)
    print("Title:", first_sample.title)
    print("Question:", first_sample.question)
    print("Answers:", first_sample.answers)
    print("Context hash:", first_sample.context_hash)

    # ------------------------------------------------------------------
    # 2. Extract unique contexts
    # ------------------------------------------------------------------

    contexts = loader.extract_unique_contexts(samples)

    assert len(contexts) > 0
    assert len(contexts) <= len(samples)
    assert all(isinstance(context, KnowledgeContext) for context in contexts)

    first_context = contexts[0]

    assert first_context.context_id
    assert first_context.title
    assert first_context.content
    assert first_context.source == "squad"
    assert first_context.context_hash
    assert len(first_context.sample_ids) > 0
    assert isinstance(first_context.sample_ids_text(), str)

    print("\nUnique contexts")
    print("-" * 80)
    print("Sample count:", len(samples))
    print("Unique context count:", len(contexts))
    print("First context ID:", first_context.context_id)
    print("First context title:", first_context.title)
    print("First context sample IDs:", first_context.sample_ids_text())

    # ------------------------------------------------------------------
    # 3. Chunk contexts
    # ------------------------------------------------------------------

    chunker = DocumentChunker(
        chunk_size=800,
        chunk_overlap=120,
    )

    chunks = chunker.split_contexts(contexts)

    assert len(chunks) > 0
    assert all(isinstance(chunk, KnowledgeChunk) for chunk in chunks)

    first_chunk = chunks[0]

    assert first_chunk.chunk_id
    assert first_chunk.context_id
    assert first_chunk.title
    assert first_chunk.content
    assert first_chunk.source == "squad"
    assert first_chunk.chunk_index >= 0
    assert first_chunk.context_hash
    assert isinstance(first_chunk.sample_ids_text(), str)

    chunk_metadata = first_chunk.to_metadata()

    assert chunk_metadata["chunk_id"] == first_chunk.chunk_id
    assert chunk_metadata["context_id"] == first_chunk.context_id
    assert chunk_metadata["title"] == first_chunk.title
    assert chunk_metadata["source"] == "squad"
    assert "sample_ids" in chunk_metadata

    stats = DocumentChunker.summarize_chunks(chunks)

    assert stats["chunk_count"] == len(chunks)
    assert stats["max_length"] > 0

    print("\nChunks")
    print("-" * 80)
    print("Chunk count:", len(chunks))
    print("Chunk stats:", stats)
    print("First chunk ID:", first_chunk.chunk_id)
    print("First chunk preview:", first_chunk.content[:200])

    # ------------------------------------------------------------------
    # 4. Convert chunks to LangChain Documents
    # ------------------------------------------------------------------

    documents = chunker.chunks_to_documents(chunks)

    assert len(documents) == len(chunks)
    assert documents[0].page_content == first_chunk.content
    assert documents[0].metadata["chunk_id"] == first_chunk.chunk_id
    assert documents[0].metadata["context_id"] == first_chunk.context_id

    # ------------------------------------------------------------------
    # 5. Build persistent Chroma vector DB
    # ------------------------------------------------------------------

    builder = ExternalKnowledgeBuilder(
        persist_directory=TEST_PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
        chunker=chunker,
    )

    vector_store = builder.build_from_chunks(
        chunks=chunks,
        reset=True,
        batch_size=20,
    )

    assert vector_store is not None
    assert builder.exists() is True

    document_count = builder.count_documents()

    assert document_count == len(chunks)

    collection_info = builder.get_collection_info()

    assert collection_info["exists"] is True
    assert collection_info["document_count"] == len(chunks)
    assert collection_info["collection_name"] == COLLECTION_NAME
    assert collection_info["embedding_model"] == EMBEDDING_MODEL

    print("\nChroma collection")
    print("-" * 80)
    print(collection_info)

    # ------------------------------------------------------------------
    # 6. Load Chroma and retrieve top-k chunks
    # ------------------------------------------------------------------

    retriever = ExternalKnowledgeRetriever(
        persist_directory=TEST_PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
        default_top_k=3,
    )

    assert retriever.count_documents() == len(chunks)

    retrieval_result = retriever.retrieve(
        query=first_sample.question,
        top_k=3,
    )

    assert isinstance(retrieval_result, RetrievalResult)
    assert retrieval_result.query == first_sample.question
    assert retrieval_result.top_k == 3
    assert len(retrieval_result.retrieved_chunks) > 0
    assert len(retrieval_result.retrieved_chunks) <= 3

    for retrieved in retrieval_result.retrieved_chunks:
        assert isinstance(retrieved, RetrievedKnowledge)
        assert retrieved.chunk_id
        assert retrieved.context_id
        assert retrieved.title
        assert retrieved.content
        assert retrieved.source == "squad"
        assert retrieved.metadata
        assert "context_hash" in retrieved.metadata

    retrieved_chunk_ids = retrieval_result.retrieved_chunk_ids()

    assert len(retrieved_chunk_ids) == len(retrieval_result.retrieved_chunks)

    evidence_text = retrieval_result.format_evidence_for_prompt()

    assert "Evidence" in evidence_text
    assert "chunk_id:" in evidence_text
    assert "content:" in evidence_text

    print("\nRetrieval result")
    print("-" * 80)
    print("Query:", first_sample.question)
    print("Gold answers:", first_sample.answers)
    print("Retrieved chunk IDs:", retrieved_chunk_ids)

    for item in retrieval_result.retrieved_chunks:
        print()
        print("Chunk ID:", item.chunk_id)
        print("Title:", item.title)
        print("Score:", item.score)
        print("Context hash:", item.metadata.get("context_hash"))
        print("Preview:", item.content[:250])

    # ------------------------------------------------------------------
    # 7. Retrieval hit analysis against gold context
    # ------------------------------------------------------------------

    gold_context_hash = first_sample.context_hash

    retrieved_context_hashes = retrieval_result.retrieved_context_hashes()

    assert isinstance(retrieved_context_hashes, list)

    retrieval_hit = retrieval_result.contains_context_hash(gold_context_hash)

    print("\nRetrieval hit analysis")
    print("-" * 80)
    print("Gold context hash:", gold_context_hash)
    print("Retrieved context hashes:", retrieved_context_hashes)
    print("Retrieval hit:", retrieval_hit)

    # Do not assert retrieval_hit is always True.
    # Dense retrieval may miss the gold context for small top_k or small index.
    # This should be logged for later analysis, not treated as a code failure.

    # ------------------------------------------------------------------
    # 8. Raw document retrieval methods
    # ------------------------------------------------------------------

    raw_docs = retriever.retrieve_documents(
        query=first_sample.question,
        top_k=2,
    )

    assert len(raw_docs) > 0
    assert len(raw_docs) <= 2
    assert raw_docs[0].page_content
    assert raw_docs[0].metadata

    raw_docs_with_scores = retriever.retrieve_documents_with_scores(
        query=first_sample.question,
        top_k=2,
    )

    assert len(raw_docs_with_scores) > 0
    assert len(raw_docs_with_scores) <= 2

    raw_doc, score = raw_docs_with_scores[0]

    assert raw_doc.page_content
    assert isinstance(score, float)

    # ------------------------------------------------------------------
    # 9. Metadata-filtered retrieval
    # ------------------------------------------------------------------

    filtered_by_title = retriever.retrieve_by_title(
        query=first_sample.question,
        title=first_context.title,
        top_k=2,
    )

    assert isinstance(filtered_by_title, RetrievalResult)

    for item in filtered_by_title.retrieved_chunks:
        assert item.title == first_context.title

    filtered_by_context_hash = retriever.retrieve_by_context_hash(
        query=first_sample.question,
        context_hash=first_context.context_hash,
        top_k=2,
    )

    assert isinstance(filtered_by_context_hash, RetrievalResult)

    for item in filtered_by_context_hash.retrieved_chunks:
        assert item.metadata.get("context_hash") == first_context.context_hash

    # ------------------------------------------------------------------
    # 10. Prompt-ready evidence text helper
    # ------------------------------------------------------------------

    prompt_evidence = retriever.retrieve_evidence_text(
        query=first_sample.question,
        top_k=2,
    )

    assert isinstance(prompt_evidence, str)
    assert "Evidence" in prompt_evidence
    assert "content:" in prompt_evidence

    print("\nPrompt evidence preview")
    print("-" * 80)
    print(prompt_evidence[:800])


def test_document_chunker_validation() -> None:
    """
    Test invalid chunker configuration.
    """

    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=100, chunk_overlap=-1)

    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=100, chunk_overlap=100)


def test_retriever_requires_existing_vector_db() -> None:
    """
    ExternalKnowledgeRetriever should fail clearly if Chroma DB has not been built.
    """

    missing_dir = Path("data/test_external_knowledge/missing_chroma")

    if missing_dir.exists():
        import shutil

        shutil.rmtree(missing_dir)

    with pytest.raises(FileNotFoundError):
        ExternalKnowledgeRetriever(
            persist_directory=missing_dir,
            collection_name="missing_collection",
            embedding_model=EMBEDDING_MODEL,
        )