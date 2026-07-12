from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from src.memory.external_knowledge import ExternalKnowledgeRetriever
from src.llm import LLMClient
from .rag_qa_state import RAGQAState


class RAGQAWorkflow:
    """
    LangGraph workflow for the LLM-RAG multi-agent QA baseline.

    Workflow:
        START
          -> worker_a_answer
          -> retrieve_external_knowledge
          -> worker_b_answer
          -> critic_evaluate
          -> coordinator_finalize
          -> END

    This workflow does NOT include governed memory yet.
    It is the standard LLM-RAG multi-agent baseline.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        external_retriever: ExternalKnowledgeRetriever,
    ) -> None:
        self.llm_client = llm_client
        self.external_retriever = external_retriever

    # ------------------------------------------------------------------
    # Node 1: Worker A direct answer
    # ------------------------------------------------------------------

    def worker_a_answer(
        self,
        state: RAGQAState,
    ) -> dict[str, Any]:
        """
        Worker A answers the question directly without retrieved evidence.
        """

        question = state["question"]

        try:
            worker_a_output = self.llm_client.generate_worker_a_answer(
                question=question,
            )

            return {
                "worker_a_output": worker_a_output,
                "current_step": "worker_a_answered",
                "error_message": None,
            }

        except Exception as error:
            return {
                "current_step": "worker_a_failed",
                "success": False,
                "error_message": f"Worker A failed: {error}",
            }

    # ------------------------------------------------------------------
    # Node 2: External knowledge retrieval
    # ------------------------------------------------------------------

    def retrieve_external_knowledge(
        self,
        state: RAGQAState,
    ) -> dict[str, Any]:
        """
        Retrieve external knowledge chunks for Worker B.
        """

        question = state["question"]
        top_k = state["retrieval_top_k"]
        gold_context_hash = state.get("gold_context_hash")

        try:
            retrieval_result = self.external_retriever.retrieve(
                query=question,
                top_k=top_k,
            )

            retrieved_chunks = retrieval_result.retrieved_chunks
            retrieved_chunk_ids = retrieval_result.retrieved_chunk_ids
            retrieved_context_hashes = retrieval_result.retrieved_context_hashes

            retrieval_hit = None
            if gold_context_hash:
                retrieval_hit = gold_context_hash in retrieved_context_hashes

            return {
                "retrieved_chunks": retrieved_chunks,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "retrieved_context_hashes": retrieved_context_hashes,
                "retrieval_hit": retrieval_hit,
                "current_step": "external_knowledge_retrieved",
                "error_message": None,
            }

        except Exception as error:
            return {
                "current_step": "external_retrieval_failed",
                "success": False,
                "error_message": f"External knowledge retrieval failed: {error}",
            }

    # ------------------------------------------------------------------
    # Node 3: Worker B retrieval-grounded answer
    # ------------------------------------------------------------------

    def worker_b_answer(
        self,
        state: RAGQAState,
    ) -> dict[str, Any]:
        """
        Worker B answers the question using retrieved evidence.
        """

        question = state["question"]
        retrieved_chunks = state["retrieved_chunks"]

        try:
            worker_b_output = self.llm_client.generate_worker_b_answer(
                question=question,
                retrieved_knowledge=retrieved_chunks,
            )

            return {
                "worker_b_output": worker_b_output,
                "current_step": "worker_b_answered",
                "error_message": None,
            }

        except Exception as error:
            return {
                "current_step": "worker_b_failed",
                "success": False,
                "error_message": f"Worker B failed: {error}",
            }

    # ------------------------------------------------------------------
    # Node 4: Critic evaluation
    # ------------------------------------------------------------------

    def critic_evaluate(
        self,
        state: RAGQAState,
    ) -> dict[str, Any]:
        """
        Critic compares Worker A and Worker B.
        """

        question = state["question"]
        worker_a_output = state["worker_a_output"]
        worker_b_output = state["worker_b_output"]
        retrieved_chunks = state["retrieved_chunks"]

        if worker_a_output is None:
            return {
                "current_step": "critic_failed",
                "success": False,
                "error_message": "Critic failed: worker_a_output is missing.",
            }

        if worker_b_output is None:
            return {
                "current_step": "critic_failed",
                "success": False,
                "error_message": "Critic failed: worker_b_output is missing.",
            }

        try:
            critic_output = self.llm_client.generate_critic_output(
                question=question,
                worker_a_output=worker_a_output,
                worker_b_output=worker_b_output,
                retrieved_knowledge=retrieved_chunks,
            )

            return {
                "critic_output": critic_output,
                "current_step": "critic_evaluated",
                "error_message": None,
            }

        except Exception as error:
            return {
                "current_step": "critic_failed",
                "success": False,
                "error_message": f"Critic failed: {error}",
            }

    # ------------------------------------------------------------------
    # Node 5: Coordinator final answer
    # ------------------------------------------------------------------

    def coordinator_finalize(
        self,
        state: RAGQAState,
    ) -> dict[str, Any]:
        """
        Coordinator produces the final answer.
        """

        question = state["question"]
        worker_a_output = state["worker_a_output"]
        worker_b_output = state["worker_b_output"]
        critic_output = state["critic_output"]
        retrieved_chunks = state["retrieved_chunks"]

        if worker_a_output is None:
            return {
                "current_step": "coordinator_failed",
                "success": False,
                "error_message": "Coordinator failed: worker_a_output is missing.",
            }

        if worker_b_output is None:
            return {
                "current_step": "coordinator_failed",
                "success": False,
                "error_message": "Coordinator failed: worker_b_output is missing.",
            }

        if critic_output is None:
            return {
                "current_step": "coordinator_failed",
                "success": False,
                "error_message": "Coordinator failed: critic_output is missing.",
            }

        try:
            coordinator_output = self.llm_client.generate_coordinator_output(
                question=question,
                worker_a_output=worker_a_output,
                worker_b_output=worker_b_output,
                critic_output=critic_output,
                retrieved_knowledge=retrieved_chunks,
            )

            return {
                "coordinator_output": coordinator_output,
                "prediction": coordinator_output.final_answer,
                "current_step": "coordinator_finalized",
                "success": True,
                "error_message": None,
            }

        except Exception as error:
            return {
                "current_step": "coordinator_failed",
                "success": False,
                "error_message": f"Coordinator failed: {error}",
            }

    # ------------------------------------------------------------------
    # Build graph
    # ------------------------------------------------------------------

    def build(self):
        """
        Build and compile the LangGraph workflow.
        """

        graph = StateGraph(RAGQAState)

        graph.add_node("worker_a_answer", self.worker_a_answer)
        graph.add_node(
            "retrieve_external_knowledge",
            self.retrieve_external_knowledge,
        )
        graph.add_node("worker_b_answer", self.worker_b_answer)
        graph.add_node("critic_evaluate", self.critic_evaluate)
        graph.add_node("coordinator_finalize", self.coordinator_finalize)

        graph.add_edge(START, "worker_a_answer")
        graph.add_edge("worker_a_answer", "retrieve_external_knowledge")
        graph.add_edge("retrieve_external_knowledge", "worker_b_answer")
        graph.add_edge("worker_b_answer", "critic_evaluate")
        graph.add_edge("critic_evaluate", "coordinator_finalize")
        graph.add_edge("coordinator_finalize", END)

        return graph.compile()

    def run(
        self,
        initial_state: RAGQAState,
    ) -> RAGQAState:
        """
        Build and run the workflow for one QA sample.
        """

        app = self.build()
        final_state = app.invoke(initial_state)

        return final_state


def build_rag_qa_workflow(
    *,
    llm_client: LLMClient,
    external_retriever: ExternalKnowledgeRetriever,
):
    """
    Convenience function for building the compiled workflow directly.
    """

    workflow = RAGQAWorkflow(
        llm_client=llm_client,
        external_retriever=external_retriever,
    )

    return workflow.build()
