from __future__ import annotations

from typing import Any, Optional

from ..entities import Agent, MemoryItem
from .candidate_memory_builder import CandidateMemoryBuilder
from .conflict_detector import ConflictDetector
from .conflict_schema import (
    ConflictCheckResult,
    GovernedMemoryLifecycleResult,
)
from .memory_write_controller import (
    MemoryWriteController,
    MemoryWriteResult,
)

from ..services.operation_log_service import OperationLogService


class GovernedMemoryLifecycle:
    """
    Orchestrate the governed memory lifecycle after a QA workflow run.

    Lifecycle:
        workflow final state
            -> build candidate memory
            -> compare with active memories
            -> detect duplicate / conflict
            -> controlled private-memory write
            -> optional promotion submission
            -> Critic review
            -> Coordinator approval
            -> shared memory

    This class coordinates existing services. It does not directly implement:
    - candidate construction rules
    - conflict detection rules
    - permission rules
    - low-level storage
    - promotion business rules
    """

    def __init__(
            self,
            *,
            memory_service: Any,
            conflict_detector: ConflictDetector,
            memory_write_controller: MemoryWriteController,
            operation_log_service: OperationLogService,
            promotion_service: Optional[Any] = None,
    ) -> None:
        self.memory_service = memory_service
        self.conflict_detector = conflict_detector
        self.memory_write_controller = memory_write_controller
        self.operation_log_service = operation_log_service
        self.promotion_service = promotion_service

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_qa_result(
        self,
        *,
        final_state: dict[str, Any],
        worker_agent: Agent,
        critic_agent: Agent,
        coordinator_agent: Agent,
        auto_promote: bool = True,
    ) -> GovernedMemoryLifecycleResult:
        """
        Process one governed QA result through the memory lifecycle.

        Args:
            final_state:
                Final state returned by GovernedRAGQAWorkflow.
            worker_agent:
                Worker that owns and proposes the generated memory.
                Normally this is Worker B because its answer is evidence-grounded.
            critic_agent:
                Agent responsible for reviewing promotion requests.
            coordinator_agent:
                Agent responsible for final promotion approval.
            auto_promote:
                Whether safe memories should automatically enter the
                review-and-approval promotion process.

        Returns:
            GovernedMemoryLifecycleResult.
        """

        candidate_memory: Optional[MemoryItem] = None
        conflict_result: Optional[ConflictCheckResult] = None
        write_result: Optional[MemoryWriteResult] = None

        try:
            # ----------------------------------------------------------
            # Step 1: build temporary candidate memory
            # ----------------------------------------------------------

            candidate_memory = CandidateMemoryBuilder.build_from_final_state(
                final_state,
                owner_agent_id=worker_agent.agent_id,
                created_by_agent_id=worker_agent.agent_id,
                tags=["memory_lifecycle"],
            )

            # ----------------------------------------------------------
            # Step 2: load current active memories
            # ----------------------------------------------------------

            existing_memories = self._list_active_memories()

            # ----------------------------------------------------------
            # Step 3: detect duplicate / conflicting answer
            # ----------------------------------------------------------

            conflict_result = self.conflict_detector.check_conflict(
                candidate_memory=candidate_memory,
                existing_memories=existing_memories,
            )

            # ----------------------------------------------------------
            # Step 4: controlled private-memory write
            # ----------------------------------------------------------

            write_result = (
                self.memory_write_controller.write_candidate_memory(
                    candidate_memory=candidate_memory,
                    requesting_agent=worker_agent,
                    conflict_result=conflict_result,
                )
            )

            if not write_result.success:
                return self._build_failure_result(
                    candidate_memory=candidate_memory,
                    conflict_result=conflict_result,
                    write_result=write_result,
                    message=write_result.message,
                )

            # Duplicate candidate: successfully blocked.
            if not write_result.written:
                return GovernedMemoryLifecycleResult(
                    success=True,
                    candidate_memory_id=candidate_memory.memory_id,
                    private_memory_written=False,
                    private_memory_id=None,
                    conflict_result=conflict_result,
                    promotion_request_id=None,
                    promotion_status="blocked_duplicate",
                    shared_memory_id=None,
                    operation_log_count=self._count_operation_logs(),
                    message=(
                        "Duplicate candidate was detected. "
                        "The memory was not written or promoted."
                    ),
                )

            private_memory_id = write_result.stored_memory_id

            if not private_memory_id:
                return self._build_failure_result(
                    candidate_memory=candidate_memory,
                    conflict_result=conflict_result,
                    write_result=write_result,
                    message=(
                        "Private memory write reported success, but no "
                        "stored_memory_id was returned."
                    ),
                )

            # ----------------------------------------------------------
            # Step 5: stop automatic promotion for conflict/review cases
            # ----------------------------------------------------------

            if not write_result.should_promote:
                promotion_status = (
                    "requires_review"
                    if write_result.requires_review
                    else "blocked"
                )

                return GovernedMemoryLifecycleResult(
                    success=True,
                    candidate_memory_id=candidate_memory.memory_id,
                    private_memory_written=True,
                    private_memory_id=private_memory_id,
                    conflict_result=conflict_result,
                    promotion_request_id=None,
                    promotion_status=promotion_status,
                    shared_memory_id=None,
                    operation_log_count=self._count_operation_logs(),
                    message=write_result.message,
                )

            # ----------------------------------------------------------
            # Step 6: safe but promotion disabled
            # ----------------------------------------------------------

            if not auto_promote:
                return GovernedMemoryLifecycleResult(
                    success=True,
                    candidate_memory_id=candidate_memory.memory_id,
                    private_memory_written=True,
                    private_memory_id=private_memory_id,
                    conflict_result=conflict_result,
                    promotion_request_id=None,
                    promotion_status="eligible_not_submitted",
                    shared_memory_id=None,
                    operation_log_count=self._count_operation_logs(),
                    message=(
                        "Private memory was written successfully and is "
                        "eligible for promotion, but auto_promote is disabled."
                    ),
                )

            # ----------------------------------------------------------
            # Step 7: promotion service not configured
            # ----------------------------------------------------------

            if self.promotion_service is None:
                return GovernedMemoryLifecycleResult(
                    success=True,
                    candidate_memory_id=candidate_memory.memory_id,
                    private_memory_written=True,
                    private_memory_id=private_memory_id,
                    conflict_result=conflict_result,
                    promotion_request_id=None,
                    promotion_status="promotion_service_not_configured",
                    shared_memory_id=None,
                    operation_log_count=self._count_operation_logs(),
                    message=(
                        "Private memory was written and passed conflict "
                        "checking, but PromotionService is not configured."
                    ),
                )

            # ----------------------------------------------------------
            # Step 8: submit -> Critic review -> Coordinator approval
            # ----------------------------------------------------------

            (
                promotion_request_id,
                promotion_status,
                shared_memory_id,
            ) = self._run_promotion_process(
                private_memory_id=private_memory_id,
                worker_agent=worker_agent,
                critic_agent=critic_agent,
                coordinator_agent=coordinator_agent,
            )

            return GovernedMemoryLifecycleResult(
                success=True,
                candidate_memory_id=candidate_memory.memory_id,
                private_memory_written=True,
                private_memory_id=private_memory_id,
                conflict_result=conflict_result,
                promotion_request_id=promotion_request_id,
                promotion_status=promotion_status,
                shared_memory_id=shared_memory_id,
                operation_log_count=self._count_operation_logs(),
                message=(
                    "Candidate memory completed the governed lifecycle: "
                    "private write, promotion review, and Coordinator approval."
                ),
            )

        except Exception as error:
            return GovernedMemoryLifecycleResult(
                success=False,
                candidate_memory_id=(
                    candidate_memory.memory_id
                    if candidate_memory is not None
                    else None
                ),
                private_memory_written=(
                    write_result.written
                    if write_result is not None
                    else False
                ),
                private_memory_id=(
                    write_result.stored_memory_id
                    if write_result is not None
                    else None
                ),
                conflict_result=conflict_result,
                promotion_request_id=None,
                promotion_status="failed",
                shared_memory_id=None,
                operation_log_count=self._count_operation_logs(),
                message=f"Governed memory lifecycle failed: {error}",
            )

    # ------------------------------------------------------------------
    # Promotion orchestration
    # ------------------------------------------------------------------

    # def _run_promotion_process(
    #     self,
    #     *,
    #     private_memory_id: str,
    #     worker_agent: Agent,
    #     critic_agent: Agent,
    #     coordinator_agent: Agent,
    # ) -> tuple[str, str, Optional[str]]:
    #     """
    #     Execute private-to-shared promotion.
    #
    #     The invocation helper checks the actual method signature and supplies
    #     only parameters declared by the current PromotionService.
    #     """
    #
    #     promotion_request = self._invoke_promotion_method(
    #         method_name="submit_promotion_request",
    #         values={
    #             "agent": worker_agent,
    #             "requesting_agent": worker_agent,
    #             "proposer_agent": worker_agent,
    #             "worker_agent": worker_agent,
    #             "memory_id": private_memory_id,
    #             "target_memory_id": private_memory_id,
    #             "reason": (
    #                 "Evidence-grounded QA memory passed conflict checking "
    #                 "and was proposed for shared memory."
    #             ),
    #         },
    #     )
    #
    #     promotion_request_id = self._extract_request_id(
    #         promotion_request
    #     )
    #
    #     self._invoke_promotion_method(
    #         method_name="review_promotion_request",
    #         values={
    #             "agent": critic_agent,
    #             "reviewer_agent": critic_agent,
    #             "critic_agent": critic_agent,
    #             "request_id": promotion_request_id,
    #             "promotion_request_id": promotion_request_id,
    #             "recommendation": "approve",
    #             "approved": True,
    #             "review_comment": (
    #                 "The candidate passed deterministic conflict checking "
    #                 "and is recommended for promotion."
    #             ),
    #             "comment": (
    #                 "The candidate passed deterministic conflict checking "
    #                 "and is recommended for promotion."
    #             ),
    #             "reason": (
    #                 "Critic recommends promoting the verified QA memory."
    #             ),
    #         },
    #     )
    #
    #     approval_result = self._invoke_promotion_method(
    #         method_name="approve_promotion_request",
    #         values={
    #             "agent": coordinator_agent,
    #             "approving_agent": coordinator_agent,
    #             "coordinator_agent": coordinator_agent,
    #             "request_id": promotion_request_id,
    #             "promotion_request_id": promotion_request_id,
    #             "reason": (
    #                 "Coordinator approved the reviewed memory for "
    #                 "shared access."
    #             ),
    #         },
    #     )
    #
    #     shared_memory_id = self._resolve_shared_memory_id(
    #         private_memory_id=private_memory_id,
    #         approval_result=approval_result,
    #     )
    #
    #     promotion_status = (
    #         self._extract_status(approval_result)
    #         or "approved"
    #     )
    #
    #     return (
    #         promotion_request_id,
    #         promotion_status,
    #         shared_memory_id,
    #     )

    def _run_promotion_process(
            self,
            *,
            private_memory_id: str,
            worker_agent: Agent,
            critic_agent: Agent,
            coordinator_agent: Agent,
    ) -> tuple[str, str, str]:
        """
        Execute the complete private-to-shared promotion process.

        Flow:
            Worker submits promotion request
            -> Critic reviews request
            -> Coordinator approves request
            -> private memory becomes shared
        """

        # --------------------------------------------------------------
        # Step 1: Worker submits promotion request
        # --------------------------------------------------------------

        promotion_request = (
            self.promotion_service.submit_promotion_request(
                agent=worker_agent,
                memory_id=private_memory_id,
                reason=(
                    "The evidence-grounded QA memory passed conflict "
                    "checking and is proposed for shared access."
                ),
            )
        )

        # --------------------------------------------------------------
        # Step 2: Critic reviews the request
        # --------------------------------------------------------------

        reviewed_request = (
            self.promotion_service.review_promotion_request(
                agent=critic_agent,
                request_id=promotion_request.request_id,
                comment=(
                    "The memory passed conflict checking and is "
                    "recommended for promotion."
                ),
            )
        )

        # The review step only adds the critic's recommendation.
        # The request remains pending until the coordinator decides.

        # --------------------------------------------------------------
        # Step 3: Coordinator approves the request
        # --------------------------------------------------------------

        promoted_memory = (
            self.promotion_service.approve_promotion_request(
                agent=coordinator_agent,
                request_id=reviewed_request.request_id,
                comment=(
                    "Coordinator approved the reviewed memory for "
                    "shared access."
                ),
            )
        )

        # --------------------------------------------------------------
        # Step 4: Read final request state
        # --------------------------------------------------------------

        approved_request = self.promotion_service.get_request(
            reviewed_request.request_id
        )

        return (
            approved_request.request_id,
            approved_request.status.value,
            promoted_memory.memory_id,
        )

    # def _invoke_promotion_method(
    #     self,
    #     *,
    #     method_name: str,
    #     values: dict[str, Any],
    # ) -> Any:
    #     """
    #     Invoke one PromotionService method using its declared signature.
    #
    #     Unlike repeated try/except TypeError approaches, this method:
    #     - inspects the service method once
    #     - passes only supported argument names
    #     - reports missing required parameters before invocation
    #     - does not hide TypeError raised inside the service implementation
    #     """
    #
    #     method = getattr(
    #         self.promotion_service,
    #         method_name,
    #         None,
    #     )
    #
    #     if not callable(method):
    #         raise AttributeError(
    #             f"PromotionService does not provide {method_name}()."
    #         )
    #
    #     signature = inspect.signature(method)
    #
    #     positional_args: list[Any] = []
    #     keyword_args: dict[str, Any] = {}
    #     missing_required: list[str] = []
    #
    #     for parameter_name, parameter in signature.parameters.items():
    #         if parameter_name == "self":
    #             continue
    #
    #         if parameter.kind in {
    #             inspect.Parameter.VAR_POSITIONAL,
    #             inspect.Parameter.VAR_KEYWORD,
    #         }:
    #             continue
    #
    #         if parameter_name in values:
    #             value = values[parameter_name]
    #
    #             if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
    #                 positional_args.append(value)
    #             else:
    #                 keyword_args[parameter_name] = value
    #
    #             continue
    #
    #         if parameter.default is inspect.Parameter.empty:
    #             missing_required.append(parameter_name)
    #
    #     if missing_required:
    #         raise TypeError(
    #             f"Cannot call PromotionService.{method_name}(). "
    #             f"Missing required parameters: {missing_required}. "
    #             "Update the lifecycle alias mapping to match the actual "
    #             "PromotionService interface."
    #         )
    #
    #     return method(
    #         *positional_args,
    #         **keyword_args,
    #     )


    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    def _list_active_memories(self) -> list[MemoryItem]:
        """
        Load all current active memories for global consistency checking.

        Conflict checking is a governance-level operation, so it is not
        restricted to the proposing worker's visible memory context.
        """

        memory_store = getattr(
            self.memory_service,
            "memory_store",
            None,
        )

        if memory_store is None:
            raise AttributeError(
                "MemoryService must expose memory_store."
            )

        return memory_store.list_active()

    # def _resolve_shared_memory_id(
    #     self,
    #     *,
    #     private_memory_id: str,
    #     approval_result: Any,
    # ) -> Optional[str]:
    #     """
    #     Resolve the shared memory after Coordinator approval.
    #
    #     Normal promotion updates the existing private memory's scope, so its
    #     memory ID usually remains unchanged.
    #     """
    #
    #     if isinstance(approval_result, MemoryItem):
    #         return approval_result.memory_id
    #
    #     for field_name in [
    #         "shared_memory_id",
    #         "promoted_memory_id",
    #         "memory_id",
    #         "target_memory_id",
    #     ]:
    #         value = self._read_field(
    #             approval_result,
    #             field_name,
    #         )
    #
    #         if value:
    #             return str(value)
    #
    #     memory_store = self.memory_service.memory_store
    #
    #     try:
    #         memory = memory_store.get_by_id(
    #             private_memory_id
    #         )
    #     except Exception:
    #         return None
    #
    #     scope = memory.metadata.scope
    #
    #     if hasattr(scope, "value"):
    #         scope = scope.value
    #
    #     if str(scope).strip().lower() == "shared":
    #         return memory.memory_id
    #
    #     return None

    # ------------------------------------------------------------------
    # Result extraction helpers
    # ------------------------------------------------------------------

    # @classmethod
    # def _extract_request_id(
    #     cls,
    #     promotion_request: Any,
    # ) -> str:
    #     for field_name in [
    #         "request_id",
    #         "promotion_request_id",
    #         "id",
    #     ]:
    #         value = cls._read_field(
    #             promotion_request,
    #             field_name,
    #         )
    #
    #         if value:
    #             return str(value)
    #
    #     raise ValueError(
    #         "PromotionService.submit_promotion_request() did not return "
    #         "an object containing request_id, promotion_request_id, or id."
    #     )
    #
    # @classmethod
    # def _extract_status(
    #     cls,
    #     value: Any,
    # ) -> Optional[str]:
    #     status = cls._read_field(value, "status")
    #
    #     if status is None:
    #         return None
    #
    #     if hasattr(status, "value"):
    #         status = status.value
    #
    #     return str(status)
    #
    # @staticmethod
    # def _read_field(
    #     value: Any,
    #     field_name: str,
    # ) -> Any:
    #     if value is None:
    #         return None
    #
    #     if isinstance(value, dict):
    #         return value.get(field_name)
    #
    #     return getattr(value, field_name, None)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _count_operation_logs(self) -> Optional[int]:
        """
        Return operation log count when logging is configured.

        MemoryService already records private memory creation when an
        OperationLogStore is provided.
        """

        # operation_log_store = getattr(
        #     self.memory_service,
        #     "operation_log_store",
        #     None,
        # )
        #
        # if operation_log_store is None:
        #     return None
        #
        # count_method = getattr(
        #     operation_log_store,
        #     "count",
        #     None,
        # )
        #
        # if callable(count_method):
        #     return int(count_method())
        #
        # list_method = getattr(
        #     operation_log_store,
        #     "list_all",
        #     None,
        # )
        #
        # if callable(list_method):
        #     return len(list_method())
        #
        # return None
        return self.operation_log_service.operation_log_store.count()

    # ------------------------------------------------------------------
    # Failure helper
    # ------------------------------------------------------------------

    def _build_failure_result(
        self,
        *,
        candidate_memory: MemoryItem,
        conflict_result: ConflictCheckResult,
        write_result: MemoryWriteResult,
        message: str,
    ) -> GovernedMemoryLifecycleResult:
        return GovernedMemoryLifecycleResult(
            success=False,
            candidate_memory_id=candidate_memory.memory_id,
            private_memory_written=write_result.written,
            private_memory_id=write_result.stored_memory_id,
            conflict_result=conflict_result,
            promotion_request_id=None,
            promotion_status="failed",
            shared_memory_id=None,
            operation_log_count=self._count_operation_logs(),
            message=message,
        )