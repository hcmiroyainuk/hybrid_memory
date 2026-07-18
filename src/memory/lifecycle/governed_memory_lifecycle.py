from __future__ import annotations

from typing import Any, Optional

from ..services.operation_log_service import OperationLogService
from .candidate_memory_builder import CandidateMemoryBuilder
from .conflict_detector import ConflictDetector
from ..entities import (
    Agent,
    MemoryItem,
    ConflictType as OperationConflictType,
)
from .conflict_schema import (
    ConflictCheckResult,
    ConflictType as LifecycleConflictType,
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

    CONFLICT_TYPE_MAP: dict[
        LifecycleConflictType,
        OperationConflictType,
    ] = {
        LifecycleConflictType.DUPLICATE:
            OperationConflictType.DUPLICATION,

        LifecycleConflictType.CONFLICTING_ANSWER:
            OperationConflictType.CONTRADICTION,

        LifecycleConflictType.OUTDATED:
            OperationConflictType.OUTDATED,
    }

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
        log_count_before = self._count_operation_logs()
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
                    log_count_before=log_count_before,
                    message=write_result.message,
                )

            # Duplicate candidate: log conflict and block persistence.
            if not write_result.written:
                if (
                        conflict_result.conflict_type
                        == LifecycleConflictType.DUPLICATE
                ):
                    self._record_detected_conflicts(
                        subject_memory=candidate_memory,
                        conflict_result=conflict_result,
                        actor_agent=critic_agent,
                    )

                return GovernedMemoryLifecycleResult(
                    success=True,
                    candidate_memory_id=candidate_memory.memory_id,
                    private_memory_written=False,
                    private_memory_id=None,
                    conflict_result=conflict_result,
                    promotion_request_id=None,
                    promotion_status="blocked_duplicate",
                    shared_memory_id=None,
                    operation_log_count=self._count_new_operation_logs(
                        log_count_before
                    ),
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
                    log_count_before=log_count_before,
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

                if (
                        conflict_result.conflict_type
                        == LifecycleConflictType.CONFLICTING_ANSWER
                ):
                    persisted_private_memory = (
                        self.memory_service.memory_store.get_by_id(
                            private_memory_id
                        )
                    )

                    self._record_detected_conflicts(
                        subject_memory=persisted_private_memory,
                        conflict_result=conflict_result,
                        actor_agent=critic_agent,
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
                    operation_log_count=self._count_new_operation_logs(
                        log_count_before
                    ),
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
                    operation_log_count=self._count_new_operation_logs(
                        log_count_before
                    ),
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
                    operation_log_count=self._count_new_operation_logs(
                        log_count_before
                    ),
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
                operation_log_count=self._count_new_operation_logs(
                    log_count_before
                ),
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
                operation_log_count=self._count_new_operation_logs(
                    log_count_before
                ),
                message=f"Governed memory lifecycle failed: {error}",
            )

    # ------------------------------------------------------------------
    # Promotion orchestration
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    @classmethod
    def _map_conflict_type(
        cls,
        conflict_type: LifecycleConflictType,
    ) -> OperationConflictType | None:
        """
        Convert lifecycle conflict type into operation-log conflict type.

        NONE returns None because no conflict log should be created.
        """

        if conflict_type == LifecycleConflictType.NONE:
            return None

        mapped_type = cls.CONFLICT_TYPE_MAP.get(
            conflict_type
        )

        if mapped_type is None:
            raise ValueError(
                "Unsupported lifecycle conflict type for operation logging: "
                f"{conflict_type!r}."
            )

        return mapped_type

    def _record_detected_conflicts(
            self,
            *,
            subject_memory: MemoryItem,
            conflict_result: ConflictCheckResult,
            actor_agent: Agent,
    ) -> int:
        """
        Record pairwise conflict logs between the candidate/new memory and
        every matched existing memory.

        Args:
            subject_memory:
                For duplicate, this is the temporary candidate memory.
                For conflicting answer, this is the persisted private memory.
            conflict_result:
                Conflict detection result containing matched memory records.
            actor_agent:
                Agent responsible for the conflict assessment, normally Critic.

        Returns:
            Number of conflict log records created.
        """

        if conflict_result.conflict_type not in {
            LifecycleConflictType.DUPLICATE,
            LifecycleConflictType.CONFLICTING_ANSWER,
        }:
            return 0

        operation_conflict_type = self._map_conflict_type(
            conflict_result.conflict_type
        )

        if operation_conflict_type is None:
            return 0

        if not conflict_result.matched_records:
            raise ValueError(
                "Conflict result contains no matched memory records."
            )

        created_log_count = 0

        for matched_record in conflict_result.matched_records:
            existing_memory = (
                self.memory_service.memory_store.get_by_id(
                    matched_record.matched_memory_id
                )
            )

            if (
                    conflict_result.conflict_type
                    == LifecycleConflictType.DUPLICATE
            ):
                description = (
                    "Duplicate memory detected. "
                    f"Candidate memory {subject_memory.memory_id!r} "
                    f"duplicates existing memory "
                    f"{existing_memory.memory_id!r}. "
                    "The candidate write and promotion were blocked."
                )

            else:
                description = (
                    "Conflicting answer detected. "
                    f"Private memory {subject_memory.memory_id!r} "
                    f"conflicts with existing memory "
                    f"{existing_memory.memory_id!r}. "
                    "The new memory remains private and automatic "
                    "promotion was blocked."
                )

            self.operation_log_service.record_conflict_detected(
                memory_a=subject_memory,
                memory_b=existing_memory,
                actor_agent=actor_agent,
                conflict_type=operation_conflict_type,
                description=description,
            )

            created_log_count += 1

        return created_log_count

    def _count_operation_logs(self) -> Optional[int]:
        """
        Return operation log count when logging is configured.

        MemoryService already records private memory creation when an
        OperationLogStore is provided.
        """
        return self.operation_log_service.operation_log_store.count()

    def _count_new_operation_logs(
            self,
            log_count_before: int,
    ) -> int:
        """
        Return the number of operation logs created during the current
        lifecycle execution.
        """

        log_count_after = self._count_operation_logs()
        new_log_count = log_count_after - log_count_before

        if new_log_count < 0:
            raise RuntimeError(
                "Operation log count decreased during lifecycle execution. "
                f"before={log_count_before}, after={log_count_after}."
            )

        return new_log_count

    # ------------------------------------------------------------------
    # Failure helper
    # ------------------------------------------------------------------

    def _build_failure_result(
            self,
            *,
            candidate_memory: MemoryItem,
            conflict_result: ConflictCheckResult,
            write_result: MemoryWriteResult,
            log_count_before: int,
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
            # operation_log_count=self._count_operation_logs(),
            operation_log_count=self._count_new_operation_logs(
                log_count_before
            ),
            message=message,
        )