from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from src.memory.context import GovernedContextBuilder
from src.memory.external_knowledge import ExternalKnowledgeRetriever
from src.llm import LLMClient
from .governed_rag_qa_state import GovernedRAGQAState


class GovernedRAGQAWorkflow:
    """
    LangGraph workflow for governed LLM-RAG multi-agent QA.

    This workflow extends the standard RAG QA baseline with read governance.

    Workflow:
        START
          -> worker_a_retrieve_memory
          -> worker_a_answer
          -> worker_b_retrieve_memory
          -> retrieve_external_knowledge
          -> worker_b_answer
          -> critic_retrieve_memory
          -> critic_evaluate
          -> coordinator_retrieve_memory
          -> coordinator_finalize
          -> END

    This version only handles read governance:
    - Each agent receives only permission-filtered accessible memory.
    - Memory writing, promotion, and conflict resolution are not included yet.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        external_retriever: ExternalKnowledgeRetriever,
        governed_context_builder: GovernedContextBuilder,
    ) -> None:
        self.llm_client = llm_client
        self.external_retriever = external_retriever
        self.governed_context_builder = governed_context_builder

    # ------------------------------------------------------------------
    # Node 1: Worker A memory retrieval
    # ------------------------------------------------------------------

    def worker_a_retrieve_memory(
        self,
        state: GovernedRAGQAState,
    ) -> dict[str, Any]:
        """
        Retrieve permission-filtered memory for Worker A.
        """

        try:
            context = self.governed_context_builder.build_memory_context(
                agent=state["worker_a_agent"],
                query=state["question"],
                top_k=state["memory_top_k"],
            )

            return {
                "worker_a_memory_context": context,
                "worker_a_memory_text": context.formatted_context,
                "worker_a_memory_ids": context.memory_ids,
                "current_step": "worker_a_memory_retrieved",
                "error_message": None,
            }

        except Exception as error:
            return {
                "current_step": "worker_a_memory_retrieval_failed",
                "success": False,
                "error_message": f"Worker A memory retrieval failed: {error}",
            }

    # ------------------------------------------------------------------
    # Node 2: Worker A direct answer
    # ------------------------------------------------------------------

    def worker_a_answer(
        self,
        state: GovernedRAGQAState,
    ) -> dict[str, Any]:
        """
        Worker A answers the question using direct LLM reasoning and
        permission-filtered accessible memory.
        """

        try:
            worker_a_output = self.llm_client.generate_worker_a_answer(
                question=state["question"],
                accessible_memory_context=state["worker_a_memory_text"],
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
    # Node 3: Worker B memory retrieval
    # ------------------------------------------------------------------

    def worker_b_retrieve_memory(
        self,
        state: GovernedRAGQAState,
    ) -> dict[str, Any]:
        """
        Retrieve permission-filtered memory for Worker B.
        """

        try:
            context = self.governed_context_builder.build_memory_context(
                agent=state["worker_b_agent"],
                query=state["question"],
                top_k=state["memory_top_k"],
            )

            return {
                "worker_b_memory_context": context,
                "worker_b_memory_text": context.formatted_context,
                "worker_b_memory_ids": context.memory_ids,
                "current_step": "worker_b_memory_retrieved",
                "error_message": None,
            }

        except Exception as error:
            return {
                "current_step": "worker_b_memory_retrieval_failed",
                "success": False,
                "error_message": f"Worker B memory retrieval failed: {error}",
            }

    # ------------------------------------------------------------------
    # Node 4: External knowledge retrieval
    # ------------------------------------------------------------------

    def retrieve_external_knowledge(
        self,
        state: GovernedRAGQAState,
    ) -> dict[str, Any]:
        """
        Retrieve external knowledge chunks for Worker B.

        External knowledge is separate from governed memory:
        - external knowledge is read-only dataset evidence
        - governed memory is agent-produced or system-managed memory
        """

        question = state["question"]
        top_k = state["retrieval_top_k"]
        gold_context_hash = state.get("gold_context_hash")

        # try:
        #     retrieval_result = self.external_retriever.retrieve(
        #         query=question,
        #         top_k=top_k,
        #     )
        #
        #     retrieved_chunks = retrieval_result.retrieved_chunks
        #     retrieved_chunk_ids = retrieval_result.retrieved_chunk_ids
        #     retrieved_context_hashes = retrieval_result.retrieved_context_hashes
        #
        #     retrieval_hit = None
        #     if gold_context_hash:
        #         retrieval_hit = gold_context_hash in retrieved_context_hashes
        #
        #     return {
        #         "retrieved_chunks": retrieved_chunks,
        #         "retrieved_chunk_ids": retrieved_chunk_ids,
        #         "retrieved_context_hashes": retrieved_context_hashes,
        #         "retrieval_hit": retrieval_hit,
        #         "current_step": "external_knowledge_retrieved",
        #         "error_message": None,
        #     }
        try:
            retrieval_result = self.external_retriever.retrieve(
                query=question,
                top_k=top_k,
            )

            retrieved_chunks = retrieval_result.retrieved_chunks

            retrieved_chunk_ids = (
                retrieval_result.retrieved_chunk_ids()
            )

            retrieved_context_hashes = (
                retrieval_result.retrieved_context_hashes()
            )

            retrieval_hit = None

            if gold_context_hash:
                retrieval_hit = (
                        gold_context_hash
                        in retrieved_context_hashes
                )

            return {
                "retrieved_chunks": retrieved_chunks,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "retrieved_context_hashes": (
                    retrieved_context_hashes
                ),
                "retrieval_hit": retrieval_hit,
                "current_step": (
                    "external_knowledge_retrieved"
                ),
                "error_message": None,
            }

        except Exception as error:
            return {
                "current_step": "external_retrieval_failed",
                "success": False,
                "error_message": f"External knowledge retrieval failed: {error}",
            }

    # ------------------------------------------------------------------
    # Node 5: Worker B retrieval-grounded answer
    # ------------------------------------------------------------------

    def worker_b_answer(
        self,
        state: GovernedRAGQAState,
    ) -> dict[str, Any]:
        """
        Worker B answers the question using:
        - permission-filtered accessible memory
        - retrieved external evidence
        """

        try:
            worker_b_output = self.llm_client.generate_worker_b_answer(
                question=state["question"],
                retrieved_knowledge=state["retrieved_chunks"],
                accessible_memory_context=state["worker_b_memory_text"],
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
    # Node 6: Critic memory retrieval
    # ------------------------------------------------------------------

    def critic_retrieve_memory(
        self,
        state: GovernedRAGQAState,
    ) -> dict[str, Any]:
        """
        Retrieve permission-filtered memory for the Critic.
        """

        try:
            context = self.governed_context_builder.build_memory_context(
                agent=state["critic_agent"],
                query=state["question"],
                top_k=state["memory_top_k"],
            )

            return {
                "critic_memory_context": context,
                "critic_memory_text": context.formatted_context,
                "critic_memory_ids": context.memory_ids,
                "current_step": "critic_memory_retrieved",
                "error_message": None,
            }

        except Exception as error:
            return {
                "current_step": "critic_memory_retrieval_failed",
                "success": False,
                "error_message": f"Critic memory retrieval failed: {error}",
            }

    # ------------------------------------------------------------------
    # Node 7: Critic evaluation
    # ------------------------------------------------------------------

    def critic_evaluate(
        self,
        state: GovernedRAGQAState,
    ) -> dict[str, Any]:
        """
        Critic compares Worker A and Worker B using:
        - worker outputs
        - retrieved external evidence
        - permission-filtered accessible memory
        """

        worker_a_output = state["worker_a_output"]
        worker_b_output = state["worker_b_output"]

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
                question=state["question"],
                worker_a_output=worker_a_output,
                worker_b_output=worker_b_output,
                retrieved_knowledge=state["retrieved_chunks"],
                accessible_memory_context=state["critic_memory_text"],
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
    # Node 8: Coordinator memory retrieval
    # ------------------------------------------------------------------

    def coordinator_retrieve_memory(
        self,
        state: GovernedRAGQAState,
    ) -> dict[str, Any]:
        """
        Retrieve permission-filtered memory for the Coordinator.
        """

        try:
            context = self.governed_context_builder.build_memory_context(
                agent=state["coordinator_agent"],
                query=state["question"],
                top_k=state["memory_top_k"],
            )

            return {
                "coordinator_memory_context": context,
                "coordinator_memory_text": context.formatted_context,
                "coordinator_memory_ids": context.memory_ids,
                "current_step": "coordinator_memory_retrieved",
                "error_message": None,
            }

        except Exception as error:
            return {
                "current_step": "coordinator_memory_retrieval_failed",
                "success": False,
                "error_message": f"Coordinator memory retrieval failed: {error}",
            }

    # ------------------------------------------------------------------
    # Node 9: Coordinator final answer
    # ------------------------------------------------------------------

    def coordinator_finalize(
        self,
        state: GovernedRAGQAState,
    ) -> dict[str, Any]:
        """
        Coordinator produces the final answer using:
        - Worker A output
        - Worker B output
        - Critic output
        - retrieved external evidence
        - permission-filtered accessible memory
        """

        worker_a_output = state["worker_a_output"]
        worker_b_output = state["worker_b_output"]
        critic_output = state["critic_output"]

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
                question=state["question"],
                worker_a_output=worker_a_output,
                worker_b_output=worker_b_output,
                critic_output=critic_output,
                retrieved_knowledge=state["retrieved_chunks"],
                accessible_memory_context=state["coordinator_memory_text"],
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
        Build and compile the governed LangGraph workflow.
        """

        graph = StateGraph(GovernedRAGQAState)

        graph.add_node(
            "worker_a_retrieve_memory",
            self.worker_a_retrieve_memory,
        )
        graph.add_node("worker_a_answer", self.worker_a_answer)

        graph.add_node(
            "worker_b_retrieve_memory",
            self.worker_b_retrieve_memory,
        )
        graph.add_node(
            "retrieve_external_knowledge",
            self.retrieve_external_knowledge,
        )
        graph.add_node("worker_b_answer", self.worker_b_answer)

        graph.add_node(
            "critic_retrieve_memory",
            self.critic_retrieve_memory,
        )
        graph.add_node("critic_evaluate", self.critic_evaluate)

        graph.add_node(
            "coordinator_retrieve_memory",
            self.coordinator_retrieve_memory,
        )
        graph.add_node(
            "coordinator_finalize",
            self.coordinator_finalize,
        )

        graph.add_edge(START, "worker_a_retrieve_memory")
        graph.add_edge("worker_a_retrieve_memory", "worker_a_answer")
        graph.add_edge("worker_a_answer", "worker_b_retrieve_memory")
        graph.add_edge("worker_b_retrieve_memory", "retrieve_external_knowledge")
        graph.add_edge("retrieve_external_knowledge", "worker_b_answer")
        graph.add_edge("worker_b_answer", "critic_retrieve_memory")
        graph.add_edge("critic_retrieve_memory", "critic_evaluate")
        graph.add_edge("critic_evaluate", "coordinator_retrieve_memory")
        graph.add_edge("coordinator_retrieve_memory", "coordinator_finalize")
        graph.add_edge("coordinator_finalize", END)

        return graph.compile()

    def run(
        self,
        initial_state: GovernedRAGQAState,
    ) -> GovernedRAGQAState:
        """
        Build and run the workflow for one governed QA sample.
        """

        app = self.build()
        final_state = app.invoke(initial_state)

        return final_state


def build_governed_rag_qa_workflow(
    *,
    llm_client: LLMClient,
    external_retriever: ExternalKnowledgeRetriever,
    governed_context_builder: GovernedContextBuilder,
):
    """
    Convenience function for building the compiled governed workflow directly.
    """

    workflow = GovernedRAGQAWorkflow(
        llm_client=llm_client,
        external_retriever=external_retriever,
        governed_context_builder=governed_context_builder,
    )

    return workflow.build()