from pathlib import Path

from dotenv import load_dotenv

from src.memory.external_knowledge import ExternalKnowledgeRetriever
from src.llm import LLMClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def test_llm_four_agent_qa() -> None:
    retriever = ExternalKnowledgeRetriever(
        persist_directory="data/test_external_knowledge/squad_chroma",
        collection_name="test_squad_external_knowledge",
        default_top_k=3,
    )

    question = "When did Beyonce start becoming popular?"

    retrieval_result = retriever.retrieve(
        query=question,
        top_k=3,
    )

    client = LLMClient(
        model_name="gpt-4o-mini",
        temperature=0.0,
    )

    result = client.run_four_agent_qa(
        question=question,
        retrieved_knowledge=retrieval_result.retrieved_chunks,
    )

    print("Worker A:", result["worker_a"])
    print("Worker B:", result["worker_b"])
    print("Critic:", result["critic"])
    print("Coordinator:", result["coordinator"])

    assert result["worker_a"].answer
    assert result["worker_b"].answer
    assert result["critic"].recommended_answer
    assert result["coordinator"].final_answer