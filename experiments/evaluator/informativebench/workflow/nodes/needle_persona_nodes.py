from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any, TypeVar

from src.llm import (
    MemoryReviewOutput,
)

from ...data_preparing.needle_persona_models import (
    NeedleEvidenceAssessment,
)

from ...prompts.needle_persona_prompts import (
    build_candidate_evidence_prompt,
    build_candidate_evidence_recheck_prompt,
)

from src.llm import (
    AgentAnswer,
    MemoryReviewOutput,
    PromotionDecisionOutput,
)
from src.adapters import (
    MemorySharingError,
)

from ...config.needle_persona_config import (
    NeedlePersonaNodeConfig,
)
from ...data_preparing.needle_persona_models import (
    AGENT_PERSONA_MAP,
)
from ...retrieval.needle_persona_retrieval import (
    NeedlePersonaRetrievalService,
)
from ..dependencies.needle_persona_dependencies import (
    NeedlePersonaNodeDependencies,
)
from ..state.needle_persona_state import (
    NeedleMemoryCandidate,
    NeedlePersonaWorkflowState,
)


NodeUpdate = dict[str, Any]
T = TypeVar("T")


class NeedlePersonaNodeError(Exception):
    """
    Raised when one Needle Persona workflow node cannot complete safely.
    """

    def __init__(
        self,
        node_name: str,
        message: str,
        *,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(
            f"{node_name}: {message}"
        )
        self.node_name = node_name
        self.cause = cause


class NeedlePersonaNodes:
    """
    Concrete LangGraph nodes for the Needle in the Persona experiment.

    The nodes coordinate four independent concerns:

    - deterministic task routing;
    - permission-filtered memory retrieval;
    - structured LLM reasoning;
    - cross-Agent sharing through MemorySharingGateway.

    They never modify MemoryStore, PromotionRequestStore, Memory scope, or ACL
    fields directly.
    """

    def __init__(
        self,
        *,
        dependencies: NeedlePersonaNodeDependencies,
        config: NeedlePersonaNodeConfig | None = None,
        retrieval_service: (
            NeedlePersonaRetrievalService | None
        ) = None,
    ) -> None:
        self.deps = dependencies
        self.config = (
            config
            or NeedlePersonaNodeConfig()
        )
        self.retrieval = (
            retrieval_service
            or NeedlePersonaRetrievalService(
                memory_service=(
                    dependencies.memory_service
                ),
                config=self.config,
            )
        )

    # ------------------------------------------------------------------
    # Routing and initial retrieval
    # ------------------------------------------------------------------

    def route_task(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> NodeUpdate:
        """
        Select the persona Agents explicitly named in the question.

        The first mentioned persona becomes the responder. This deterministic
        rule keeps all three experiment modes comparable and avoids an extra
        routing LLM call.
        """
        return self._execute(
            "route_task",
            lambda: self._route_task_impl(
                state
            ),
        )

    def _route_task_impl(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> NodeUpdate:
        question = self._required_state_text(
            state,
            "question",
        )

        mentions: list[
            tuple[int, str]
        ] = []

        for agent_id, persona in (
            AGENT_PERSONA_MAP.items()
        ):
            match = re.search(
                rf"\b{re.escape(persona)}\b",
                question,
                flags=re.IGNORECASE,
            )

            if match is not None:
                mentions.append(
                    (
                        match.start(),
                        agent_id,
                    )
                )

        mentions.sort(
            key=lambda item: (
                item[0],
                item[1],
            )
        )

        selected_agent_ids = [
            agent_id
            for _, agent_id in mentions
        ]

        if not selected_agent_ids:
            selected_agent_ids = [
                agent_id
                for agent_id in (
                    self.deps.persona_agent_ids
                )
                if state[
                    "private_memory_ids"
                ].get(agent_id)
            ]

        if not selected_agent_ids:
            raise NeedlePersonaNodeError(
                "route_task",
                "No persona Agent could be selected.",
            )

        responder_agent_id = (
            selected_agent_ids[0]
        )

        return {
            "selected_agent_ids": (
                selected_agent_ids
            ),
            "responder_agent_id": (
                responder_agent_id
            ),
            "workflow_status": "routed",
        }

    def retrieve_initial_memory(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> NodeUpdate:
        """
        Retrieve only evidence currently accessible to the responder.
        """
        return self._execute(
            "retrieve_initial_memory",
            lambda: (
                self._retrieve_initial_memory_impl(
                    state
                )
            ),
        )

    def _retrieve_initial_memory_impl(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> NodeUpdate:
        responder = self._responder(
            state
        )

        result = (
            self.retrieval
            .retrieve_initial_for_responder(
                responder=responder,
                query=state["question"],
                allowed_private_memory_ids=(
                    state[
                        "private_memory_ids"
                    ].get(
                        responder.agent_id,
                        [],
                    )
                ),
                include_shared=True,
            )
        )

        return {
            "initial_retrieved_memory_ids": (
                list(result.memory_ids)
            ),
            "initial_memory_context": (
                result.context
            ),
            "workflow_status": (
                "initial_memory_retrieved"
            ),
            "warnings": list(
                result.warnings
            ),
        }

    def assess_initial_evidence(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> NodeUpdate:
        """
        Ask the responder to answer from its currently accessible evidence.

        An answered or refused result terminates immediately. An
        insufficient-evidence result may trigger cross-Agent access requests,
        depending on the experiment mode.
        """
        return self._execute(
            "assess_initial_evidence",
            lambda: (
                self._assess_initial_evidence_impl(
                    state
                )
            ),
        )

    def _assess_initial_evidence_impl(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> NodeUpdate:
        answer, warnings = (
            self._generate_answer(
                state=state,
                memory_context=state[
                    "initial_memory_context"
                ],
                allowed_memory_ids=state[
                    "initial_retrieved_memory_ids"
                ],
                phase="initial",
            )
        )

        return {
            "initial_answer": answer,
            "missing_information": list(
                answer.missing_information
            ),
            "workflow_status": (
                "initial_evidence_assessed"
            ),
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Candidate discovery and request submission
    # ------------------------------------------------------------------

    def discover_memory_candidates(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> NodeUpdate:
        """
        Discover private candidate memory IDs inside non-responder owners'
        memory spaces.
        """
        return self._execute(
            "discover_memory_candidates",
            lambda: (
                self._discover_memory_candidates_impl(
                    state
                )
            ),
        )

    def _discover_memory_candidates_impl(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> NodeUpdate:
        responder_id = (
            self._required_state_text(
                state,
                "responder_agent_id",
            )
        )

        selected_contributors = [
            agent_id
            for agent_id in state[
                "selected_agent_ids"
            ]
            if agent_id != responder_id
        ]

        if not selected_contributors:
            selected_contributors = [
                agent_id
                for agent_id in (
                    self.deps.persona_agent_ids
                )
                if agent_id != responder_id
            ]

        candidate_private_ids = {
            agent_id: (
                state["private_memory_ids"].get(
                    agent_id,
                    [],
                )
                if agent_id
                in selected_contributors
                else []
            )
            for agent_id in (
                self.deps.persona_agent_ids
            )
        }

        result = (
            self.retrieval
            .discover_cross_agent_candidates(
                query=state["question"],
                responder_agent_id=(
                    responder_id
                ),
                owner_agents=(
                    self.deps.persona_agents
                ),
                private_memory_ids=(
                    candidate_private_ids
                ),
                # Candidate discovery is based on the benchmark question
                # itself. The initial responder's free-form
                # missing-information decomposition is intentionally not
                # propagated because it may contain an incorrect task
                # interpretation and bias all later retrieval.
                missing_information=[],
            )
        )

        return {
            "candidate_memories": (
                result.as_mapping()
            ),
            "sharing_round_count": (
                state[
                    "sharing_round_count"
                ]
                + 1
            ),
            "workflow_status": (
                "candidates_discovered"
            ),
            "warnings": list(
                result.warnings
            ),
        }

    def submit_access_requests(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> NodeUpdate:
        """
        Submit one access request for every discovered candidate.
        """
        return self._execute(
            "submit_access_requests",
            lambda: (
                self._submit_access_requests_impl(
                    state
                )
            ),
        )

    def _submit_access_requests_impl(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> NodeUpdate:
        responder_id = (
            self._required_state_text(
                state,
                "responder_agent_id",
            )
        )

        requests: dict[str, Any] = {}
        already_accessible: list[str] = []
        warnings: list[str] = []

        for memory_id, candidate in (
            self._ordered_candidates(state)
        ):
            try:
                if (
                    self.deps.sharing_gateway
                    .has_access(
                        agent_id=responder_id,
                        memory_id=memory_id,
                    )
                ):
                    already_accessible.append(
                        memory_id
                    )
                    warnings.append(
                        f"{memory_id} was already "
                        "accessible to the responder; "
                        "no new request was submitted."
                    )
                    continue

                request = (
                    self.deps.sharing_gateway
                    .request_access(
                        requester_agent_id=(
                            responder_id
                        ),
                        memory_id=memory_id,
                        task_id=state["run_id"],
                        reason=(
                            self._build_request_reason(
                                state=state,
                                candidate=candidate,
                            )
                        ),
                    )
                )
            except MemorySharingError as error:
                warnings.append(
                    "Access request could not be "
                    f"submitted for {memory_id}: "
                    f"{error}"
                )
                continue

            requests[memory_id] = request

        if (
            state["candidate_memories"]
            and not requests
            and not already_accessible
        ):
            raise NeedlePersonaNodeError(
                "submit_access_requests",
                (
                    "All cross-Agent access requests "
                    "failed. No candidate memory could "
                    "be submitted for sharing."
                ),
            )

        return {
            "access_requests": requests,
            "approved_memory_ids": (
                already_accessible
            ),
            "workflow_status": (
                "requests_submitted"
            ),
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Governed sharing
    # ------------------------------------------------------------------

    def review_access_requests(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> NodeUpdate:
        """
        Ask the Critic to review every pending request and persist the review
        through MemorySharingGateway.
        """
        return self._execute(
            "review_access_requests",
            lambda: (
                self._review_access_requests_impl(
                    state
                )
            ),
        )

    def _review_access_requests_impl(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> NodeUpdate:
        responder_id = (
            self._required_state_text(
                state,
                "responder_agent_id",
            )
        )
        policy_context = (
            self._governance_policy_context(
                responder_agent_id=(
                    responder_id
                )
            )
        )

        evidence_assessments: dict[
            str,
            NeedleEvidenceAssessment,
        ] = {}

        critic_reviews: dict[
            str,
            MemoryReviewOutput,
        ] = {}

        reviewed_requests: dict[
            str,
            Any,
        ] = {}

        warnings: list[str] = []

        for memory_id, request in (
            self._ordered_requests(state)
        ):
            candidate = state[
                "candidate_memories"
            ].get(memory_id)

            if candidate is None:
                warnings.append(
                    "No candidate metadata exists for "
                    f"request {request.request_id}."
                )
                continue

            owner = self.deps.persona_agent(
                candidate.owner_agent_id
            )

            try:
                memory = (
                    self.retrieval
                    .get_candidate_memory(
                        owner=owner,
                        memory_id=memory_id,
                        expected_owner_agent_id=(
                            candidate
                            .owner_agent_id
                        ),
                    )
                )
                memory_payload = (
                    self.retrieval
                    .memory_to_prompt_dict(
                        memory
                    )
                )

                evidence_prompt = (
                    build_candidate_evidence_prompt(
                        question=state[
                            "question"
                        ],
                        candidate_memory=(
                            memory_payload
                        ),
                        responder_context=state[
                            "initial_memory_context"
                        ],
                    )
                )

                evidence_assessment = (
                    self.deps.llm_client
                    .invoke_structured(
                        evidence_prompt,
                        NeedleEvidenceAssessment,
                    )
                )

                # The LLM must not be allowed to replace the
                # candidate identifier supplied by the workflow.
                if (
                    evidence_assessment.memory_id
                    != memory_id
                ):
                    assessment_payload = (
                        evidence_assessment.model_dump(
                            mode="python"
                        )
                    )
                    assessment_payload[
                        "memory_id"
                    ] = memory_id

                    evidence_assessment = (
                        NeedleEvidenceAssessment
                        .model_validate(
                            assessment_payload
                        )
                    )

                # Preserve the semantic evidence assessment
                # independently of the later governance review.
                evidence_assessments[
                    memory_id
                ] = evidence_assessment

                useful_contribution = (
                    evidence_assessment
                    .contribution_type
                    in {
                        "direct_answer",
                        "partial_hop",
                    }
                )

                (
                    policy_compliant,
                    policy_reason,
                ) = (
                    self
                    ._is_task_scoped_sharing_allowed(
                        state=state,
                        candidate=candidate,
                        request=request,
                        responder_agent_id=(
                            responder_id
                        ),
                    )
                )

                should_approve = (
                    useful_contribution
                    and policy_compliant
                )

                evidence_spans = "; ".join(
                    evidence_assessment
                    .evidence_spans
                )

                evidence_description = (
                    (
                        " Evidence spans: "
                        f"{evidence_spans}."
                    )
                    if evidence_spans
                    else ""
                )

                review = MemoryReviewOutput(
                    memory_id=memory_id,
                    classification=(
                        "new"
                        if useful_contribution
                        else "irrelevant"
                    ),
                    recommendation=(
                        "approve"
                        if should_approve
                        else "reject"
                    ),
                    relevant=(
                        useful_contribution
                    ),
                    policy_compliant=(
                        policy_compliant
                    ),
                    related_memory_ids=[],
                    reason=(
                        "Contribution type: "
                        f"{evidence_assessment.contribution_type}. "
                        f"{evidence_assessment.reason}"
                        f"{evidence_description} "
                        "Policy assessment: "
                        f"{policy_reason}"
                    ),
                    confidence=(
                        evidence_assessment
                        .confidence
                    ),
                )

                reviewed_request = (
                    self.deps.sharing_gateway
                    .review_request(
                        critic_agent_id=(
                            self.deps
                            .critic_agent
                            .agent_id
                        ),
                        request_id=(
                            request.request_id
                        ),
                        recommendation=(
                            review.recommendation
                        ),
                        comment=review.reason,
                    )
                )
            except Exception as error:
                warnings.append(
                    "Critic review failed for "
                    f"{memory_id}: {error}"
                )
                continue

            critic_reviews[
                memory_id
            ] = review
            reviewed_requests[
                memory_id
            ] = reviewed_request

        return {
            "critic_evidence_assessments": (
                evidence_assessments
            ),
            "critic_reviews": critic_reviews,
            "access_requests": (
                reviewed_requests
            ),
            "workflow_status": (
                "requests_reviewed"
            ),
            "warnings": warnings,
        }

    def decide_access_requests(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> NodeUpdate:
        """
        Ask the Coordinator for final governed decisions and apply them through
        MemorySharingGateway.
        """
        return self._execute(
            "decide_access_requests",
            lambda: (
                self._decide_access_requests_impl(
                    state
                )
            ),
        )

    def _decide_access_requests_impl(
            self,
            state: NeedlePersonaWorkflowState,
    ) -> NodeUpdate:
        """
        Make final governed-sharing decisions.

        The Critic performs the initial semantic evidence assessment.

        When the Critic finds useful evidence, the Coordinator applies the
        deterministic access policy directly.

        When the Critic does not find useful evidence, the Coordinator performs
        an independent semantic evidence recheck before making the final decision.

        Explicit access-policy failures are hard rejections and cannot be
        overridden by an LLM evidence assessment.
        """
        responder_id = (
            self._required_state_text(
                state,
                "responder_agent_id",
            )
        )

        outputs: dict[
            str,
            PromotionDecisionOutput,
        ] = {}

        access_decisions: dict[
            str,
            Any,
        ] = {}

        # Store every independent Coordinator evidence recheck,
        # including both recovered and confirmed-negative results.
        evidence_rechecks: dict[
            str,
            NeedleEvidenceAssessment,
        ] = {}

        approved: list[str] = []
        rejected: list[str] = []

        # A recovered memory is also an approved memory.
        recovered: list[str] = []

        warnings: list[str] = []

        for memory_id, request in (
                self._ordered_requests(state)
        ):
            review = state[
                "critic_reviews"
            ].get(memory_id)

            candidate = state[
                "candidate_memories"
            ].get(memory_id)

            if review is None:
                warnings.append(
                    "Governed decision skipped for "
                    f"{memory_id} because its Critic "
                    "review is missing."
                )
                continue

            if candidate is None:
                warnings.append(
                    "Governed decision skipped for "
                    f"{memory_id} because its candidate "
                    "metadata is missing."
                )
                continue

            try:
                # ----------------------------------------------------------
                # 1. Deterministic final access-policy check
                # ----------------------------------------------------------

                (
                    final_policy_compliant,
                    final_policy_reason,
                ) = (
                    self
                    ._is_task_scoped_sharing_allowed(
                        state=state,
                        candidate=candidate,
                        request=request,
                        responder_agent_id=(
                            responder_id
                        ),
                    )
                )

                # Semantic relevance must be separated from the Critic's
                # combined approve/reject recommendation.
                #
                # review.recommendation may be "reject" because either:
                # 1. semantic evidence was considered insufficient; or
                # 2. the deterministic policy failed.
                #
                # Here we only extract the semantic component.
                critic_supports_sharing = bool(
                    review.relevant
                )

                coordinator_recheck: (
                        NeedleEvidenceAssessment | None
                ) = None

                coordinator_supports_sharing = False

                # ----------------------------------------------------------
                # 2. Negative-only independent Coordinator recheck
                # ----------------------------------------------------------

                # Do not ask an LLM to reconsider a hard policy rejection.
                #
                # Only recheck candidates for which:
                # - deterministic policy allows sharing; and
                # - the Critic found no answer-bearing evidence.
                if (
                        final_policy_compliant
                        and not critic_supports_sharing
                ):
                    owner = (
                        self.deps.persona_agent(
                            candidate.owner_agent_id
                        )
                    )

                    memory = (
                        self.retrieval
                        .get_candidate_memory(
                            owner=owner,
                            memory_id=memory_id,
                            expected_owner_agent_id=(
                                candidate
                                .owner_agent_id
                            ),
                        )
                    )

                    memory_payload = (
                        self.retrieval
                        .memory_to_prompt_dict(
                            memory
                        )
                    )

                    # The prompt contains only the original question,
                    # candidate memory, and authorised responder context.
                    #
                    # It deliberately does not contain the Critic's verdict
                    # or reason, preventing anchoring on the first judgment.
                    recheck_prompt = (
                        build_candidate_evidence_recheck_prompt(
                            question=state[
                                "question"
                            ],
                            candidate_memory=(
                                memory_payload
                            ),
                            responder_context=state[
                                "initial_memory_context"
                            ],
                        )
                    )

                    try:
                        coordinator_recheck = (
                            self.deps.llm_client
                            .invoke_structured(
                                recheck_prompt,
                                NeedleEvidenceAssessment,
                            )
                        )
                    except Exception as recheck_error:
                        # A failed recheck must not silently grant access.
                        # The request will continue to the rejection branch.
                        warnings.append(
                            "Coordinator evidence recheck "
                            f"failed for {memory_id}: "
                            f"{recheck_error}"
                        )

                        coordinator_recheck = None

                    if coordinator_recheck is not None:
                        # The LLM must not be able to replace the memory ID
                        # supplied by the workflow.
                        if (
                                coordinator_recheck.memory_id
                                != memory_id
                        ):
                            recheck_payload = (
                                coordinator_recheck
                                .model_dump(
                                    mode="python"
                                )
                            )

                            recheck_payload[
                                "memory_id"
                            ] = memory_id

                            coordinator_recheck = (
                                NeedleEvidenceAssessment
                                .model_validate(
                                    recheck_payload
                                )
                            )

                        evidence_rechecks[
                            memory_id
                        ] = coordinator_recheck

                        coordinator_supports_sharing = (
                                coordinator_recheck
                                .contribution_type
                                in {
                                    "direct_answer",
                                    "partial_hop",
                                }
                        )

                # ----------------------------------------------------------
                # 3. Combine semantic evidence and deterministic policy
                # ----------------------------------------------------------

                semantic_support = (
                        critic_supports_sharing
                        or coordinator_supports_sharing
                )

                should_approve = (
                        semantic_support
                        and final_policy_compliant
                )

                # ----------------------------------------------------------
                # 4. Final approval
                # ----------------------------------------------------------

                if should_approve:
                    if coordinator_supports_sharing:
                        # Critic initially rejected this candidate, but the
                        # independent Coordinator recheck recovered it.
                        evidence_source = (
                            "independent Coordinator "
                            "evidence recheck"
                        )

                        selected_assessment = (
                            coordinator_recheck
                        )

                        recovered.append(
                            memory_id
                        )

                    else:
                        evidence_source = (
                            "initial Critic evidence "
                            "assessment"
                        )

                        selected_assessment = (
                            state.get(
                                "critic_evidence_assessments",
                                {},
                            ).get(memory_id)
                        )

                    if selected_assessment is not None:
                        evidence_spans = list(
                            selected_assessment
                            .evidence_spans
                        )

                        decision_confidence = (
                            selected_assessment
                            .confidence
                        )
                    else:
                        evidence_spans = []
                        decision_confidence = (
                            review.confidence
                        )

                    evidence_description = (
                        (
                                " Evidence spans: "
                                + "; ".join(
                            evidence_spans
                        )
                                + "."
                        )
                        if evidence_spans
                        else ""
                    )

                    output = PromotionDecisionOutput(
                        memory_id=memory_id,
                        decision="approve",
                        target_scope="shared",
                        allowed_agent_ids=[
                            candidate.owner_agent_id,
                            responder_id,
                        ],
                        related_memory_ids=[],
                        reason=(
                            "The candidate was found to "
                            "contain an answer-bearing fact "
                            "or necessary intermediate hop "
                            f"through the {evidence_source}."
                            f"{evidence_description} "
                            "The Coordinator's deterministic "
                            "policy check confirmed that "
                            "task-scoped sharing is "
                            "permitted. "
                            f"{final_policy_reason}"
                        ),
                        confidence=(
                            decision_confidence
                        ),
                    )

                    access_result = (
                        self.deps
                        .sharing_gateway
                        .approve_request(
                            coordinator_agent_id=(
                                self.deps
                                .coordinator_agent
                                .agent_id
                            ),
                            request_id=(
                                request.request_id
                            ),
                            comment=output.reason,
                        )
                    )

                    approved.append(
                        memory_id
                    )

                # ----------------------------------------------------------
                # 5. Final rejection
                # ----------------------------------------------------------

                else:
                    rejection_reasons: list[str] = []

                    if not final_policy_compliant:
                        # Hard deterministic rejection.
                        rejection_reasons.append(
                            (
                                "The Coordinator's "
                                "deterministic access-policy "
                                "check failed: "
                                f"{final_policy_reason}"
                            )
                        )

                        decision_confidence = 1.0

                    elif coordinator_recheck is not None:
                        # The Critic rejected the memory and the independent
                        # Coordinator recheck confirmed the negative result.
                        rejection_reasons.append(
                            (
                                "The independent Coordinator "
                                "evidence recheck found no "
                                "answer-bearing fact or "
                                "necessary intermediate hop. "
                                "Contribution type: "
                                f"{coordinator_recheck.contribution_type}. "
                                "Coordinator assessment: "
                                f"{coordinator_recheck.reason}"
                            )
                        )

                        decision_confidence = (
                            coordinator_recheck
                            .confidence
                        )

                    else:
                        # The Critic was negative and the independent recheck
                        # could not be completed.
                        rejection_reasons.append(
                            (
                                "The Critic found no "
                                "answer-bearing fact or "
                                "necessary intermediate hop, "
                                "and no successful independent "
                                "Coordinator evidence recheck "
                                "was available."
                            )
                        )

                        decision_confidence = (
                            review.confidence
                        )

                    # Preserve the Critic result for auditability.
                    # This text was not supplied to the Coordinator recheck.
                    rejection_reasons.append(
                        (
                            "Initial Critic assessment: "
                            f"{review.reason}"
                        )
                    )

                    output = PromotionDecisionOutput(
                        memory_id=memory_id,
                        decision="reject",
                        target_scope="private",
                        allowed_agent_ids=[],
                        related_memory_ids=[],
                        reason=" ".join(
                            rejection_reasons
                        ),
                        confidence=(
                            decision_confidence
                        ),
                    )

                    access_result = (
                        self.deps
                        .sharing_gateway
                        .reject_request(
                            coordinator_agent_id=(
                                self.deps
                                .coordinator_agent
                                .agent_id
                            ),
                            request_id=(
                                request.request_id
                            ),
                            comment=output.reason,
                            require_critic_review=True,
                        )
                    )

                    rejected.append(
                        memory_id
                    )

            except Exception as error:
                warnings.append(
                    "Coordinator decision failed for "
                    f"{memory_id}: {error}"
                )
                continue

            outputs[
                memory_id
            ] = output

            access_decisions[
                memory_id
            ] = access_result

        return {
            "coordinator_evidence_rechecks": (
                evidence_rechecks
            ),
            "coordinator_outputs": outputs,
            "access_decisions": (
                access_decisions
            ),
            "approved_memory_ids": (
                approved
            ),
            "rejected_memory_ids": (
                rejected
            ),
            "recovered_memory_ids": (
                recovered
            ),
            "workflow_status": (
                "access_decided"
            ),
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Ungoverned baseline
    # ------------------------------------------------------------------

    def approve_direct_requests(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> NodeUpdate:
        """
        Directly approve pending requests for the ungoverned baseline.
        """
        return self._execute(
            "approve_direct_requests",
            lambda: (
                self._approve_direct_requests_impl(
                    state
                )
            ),
        )

    def _approve_direct_requests_impl(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> NodeUpdate:
        decisions: dict[str, Any] = {}
        approved: list[str] = []
        rejected: list[str] = []
        warnings: list[str] = []

        for memory_id, request in (
            self._ordered_requests(state)
        ):
            try:
                result = (
                    self.deps.sharing_gateway
                    .approve_direct_request(
                        coordinator_agent_id=(
                            self.deps
                            .coordinator_agent
                            .agent_id
                        ),
                        request_id=(
                            request.request_id
                        ),
                        comment=(
                            "Direct approval for the "
                            "ungoverned_shared baseline."
                        ),
                    )
                )
            except MemorySharingError as error:
                warnings.append(
                    "Direct approval failed for "
                    f"{memory_id}: {error}"
                )
                rejected.append(
                    memory_id
                )
                continue

            decisions[memory_id] = result

            if result.access_granted:
                approved.append(
                    memory_id
                )
            else:
                rejected.append(
                    memory_id
                )

        return {
            "access_decisions": decisions,
            "approved_memory_ids": (
                approved
            ),
            "rejected_memory_ids": (
                rejected
            ),
            "workflow_status": (
                "access_decided"
            ),
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Final retrieval and answer
    # ------------------------------------------------------------------

    def retrieve_final_memory(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> NodeUpdate:
        """
        Re-run permission-filtered retrieval after approved ACL changes.
        """
        return self._execute(
            "retrieve_final_memory",
            lambda: (
                self._retrieve_final_memory_impl(
                    state
                )
            ),
        )

    def _retrieve_final_memory_impl(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> NodeUpdate:
        responder = self._responder(
            state
        )

        result = (
            self.retrieval
            .retrieve_final_for_responder(
                responder=responder,
                query=state["question"],
                allowed_private_memory_ids=(
                    state[
                        "private_memory_ids"
                    ].get(
                        responder.agent_id,
                        [],
                    )
                ),
            )
        )

        return {
            "final_retrieved_memory_ids": (
                list(result.memory_ids)
            ),
            "final_memory_context": (
                result.context
            ),
            "workflow_status": (
                "final_memory_retrieved"
            ),
            "warnings": list(
                result.warnings
            ),
        }

    def generate_final_answer(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> NodeUpdate:
        """
        Generate the terminal answer from the final authorised context.
        """
        return self._execute(
            "generate_final_answer",
            lambda: (
                self._generate_final_answer_impl(
                    state
                )
            ),
        )

    def _generate_final_answer_impl(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> NodeUpdate:
        answer, warnings = (
            self._generate_answer(
                state=state,
                memory_context=state[
                    "final_memory_context"
                ],
                allowed_memory_ids=state[
                    "final_retrieved_memory_ids"
                ],
                phase="final",
            )
        )

        return {
            "final_answer": answer,
            "missing_information": list(
                answer.missing_information
            ),
            "workflow_status": (
                self._terminal_status(answer)
            ),
            "warnings": warnings,
        }

    def finalise_initial_answer(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> NodeUpdate:
        """
        Use the initial answer as the terminal result when no sharing path is
        required or no new access was granted.
        """
        return self._execute(
            "finalise_initial_answer",
            lambda: (
                self._finalise_initial_answer_impl(
                    state
                )
            ),
        )

    def _finalise_initial_answer_impl(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> NodeUpdate:
        answer = state.get(
            "initial_answer"
        )

        if answer is None:
            raise NeedlePersonaNodeError(
                "finalise_initial_answer",
                "initial_answer is missing.",
            )

        return {
            "final_answer": answer,
            "workflow_status": (
                self._terminal_status(answer)
            ),
        }

    # ------------------------------------------------------------------
    # Conditional-edge helpers
    # ------------------------------------------------------------------

    def route_after_initial_assessment(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> str:
        answer = state.get(
            "initial_answer"
        )

        if answer is None:
            raise NeedlePersonaNodeError(
                "route_after_initial_assessment",
                "initial_answer is missing.",
            )

        if answer.status in {
            "answered",
            "refused",
        }:
            return "finish"

        if (
            state["experiment_mode"]
            == "private_only"
        ):
            return "finish"

        if (
            state["sharing_round_count"]
            >= self.config.max_sharing_rounds
        ):
            return "finish"

        return "discover"

    @staticmethod
    def route_after_candidate_discovery(
        state: NeedlePersonaWorkflowState,
    ) -> str:
        return (
            "submit"
            if state["candidate_memories"]
            else "finish"
        )

    @staticmethod
    def route_after_request_submission(
        state: NeedlePersonaWorkflowState,
    ) -> str:
        if state["approved_memory_ids"]:
            return "retrieve"

        if not state["access_requests"]:
            return "finish"

        mode = state["experiment_mode"]

        if mode == "ungoverned_shared":
            return "ungoverned"

        if mode == "governed_shared":
            return "governed"

        return "finish"

    @staticmethod
    def route_after_access_decision(
        state: NeedlePersonaWorkflowState,
    ) -> str:
        return (
            "retrieve"
            if state["approved_memory_ids"]
            else "finish"
        )

    # ------------------------------------------------------------------
    # Answer generation and grounding
    # ------------------------------------------------------------------

    def _generate_answer(
        self,
        *,
        state: NeedlePersonaWorkflowState,
        memory_context: str,
        allowed_memory_ids: list[str],
        phase: str,
    ) -> tuple[
        AgentAnswer,
        list[str],
    ]:
        responder = self._responder(
            state
        )
        prompt_template = (
            self.deps.worker_prompt(
                responder.agent_id
            )
        )

        prompt = (
            prompt_template
            .build_answer_prompt(
                task=state["question"],
                task_input=(
                    "Return only the answer items "
                    "requested by the benchmark "
                    "question."
                ),
                accessible_memory_context=(
                    memory_context
                ),
                context_sections={
                    "Experiment mode": (
                        state[
                            "experiment_mode"
                        ]
                    ),
                    "Answering phase": phase,
                    "Responder agent ID": (
                        responder.agent_id
                    ),
                    "Authorised memory IDs": (
                        allowed_memory_ids
                    ),
                    "Answering instruction": (
                        (
                            "Re-evaluate the benchmark question from "
                            "scratch using the complete authorised "
                            "memory context supplied in this final "
                            "phase. Do not preserve or assume the "
                            "initial answer's interpretation or "
                            "missing-information decomposition. Return "
                            "status='answered' whenever the current "
                            "authorised evidence directly supports the "
                            "specific answer requested by the question."
                        )
                        if phase == "final"
                        else (
                            "Evaluate the benchmark question using "
                            "only the currently authorised memory "
                            "context. Identify only information that "
                            "the question explicitly requires."
                        )
                    ),
                },
            )
        )

        answer = (
            self.deps.llm_client
            .invoke_structured(
                prompt,
                AgentAnswer,
            )
        )

        if not (
            self.config
            .sanitise_answer_references
        ):
            return answer, []

        return self._sanitise_answer(
            answer=answer,
            responder=responder,
            allowed_memory_ids=(
                allowed_memory_ids
            ),
        )

    def _sanitise_answer(
        self,
        *,
        answer: AgentAnswer,
        responder: Any,
        allowed_memory_ids: list[str],
    ) -> tuple[
        AgentAnswer,
        list[str],
    ]:
        allowed_ids = set(
            self._clean_strings(
                allowed_memory_ids
            )
        )
        used_ids = [
            memory_id
            for memory_id in (
                self._clean_strings(
                    answer.used_memory_ids
                )
            )
            if memory_id in allowed_ids
        ]

        source_ids: list[str] = []
        contributing_agent_ids: list[
            str
        ] = []
        verified_ids: list[str] = []
        warnings: list[str] = []

        for memory_id in used_ids:
            try:
                memory = (
                    self.deps.memory_service
                    .get_memory(
                        responder,
                        memory_id,
                    )
                )
                payload = (
                    self.retrieval
                    .memory_to_prompt_dict(
                        memory
                    )
                )
            except Exception as error:
                warnings.append(
                    "Could not verify answer "
                    f"reference {memory_id}: "
                    f"{error}"
                )
                continue

            verified_ids.append(
                memory_id
            )

            for source_id in payload[
                "source_ids"
            ]:
                if (
                    source_id
                    not in source_ids
                ):
                    source_ids.append(
                        source_id
                    )

            owner_id = payload[
                "owner_agent_id"
            ]
            if (
                owner_id
                not in contributing_agent_ids
            ):
                contributing_agent_ids.append(
                    owner_id
                )

        removed_ids = [
            memory_id
            for memory_id in (
                answer.used_memory_ids
            )
            if memory_id not in verified_ids
        ]

        if removed_ids:
            warnings.append(
                "Removed unverified memory "
                "references from AgentAnswer: "
                + ", ".join(
                    self._clean_strings(
                        removed_ids
                    )
                )
            )

        payload = answer.model_dump(
            mode="python"
        )
        payload.update(
            {
                "used_memory_ids": (
                    verified_ids
                ),
                "supporting_source_ids": (
                    source_ids
                ),
                "contributing_agent_ids": (
                    contributing_agent_ids
                ),
            }
        )

        return (
            AgentAnswer.model_validate(
                payload
            ),
            warnings,
        )

    # ------------------------------------------------------------------
    # Governance normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _is_task_scoped_sharing_allowed(
        *,
        state: NeedlePersonaWorkflowState,
        candidate: NeedleMemoryCandidate,
        request: Any,
        responder_agent_id: str,
    ) -> tuple[bool, str]:
        """
        Apply deterministic Needle Persona sharing policy.

        This method checks access-control conditions only.
        It does not assess semantic relevance.
        """

        if (
            candidate.owner_agent_id
            == responder_agent_id
        ):
            return (
                False,
                (
                    "The selected responder already "
                    "owns the candidate memory."
                ),
            )

        if (
            request.requester_agent_id
            != responder_agent_id
        ):
            return (
                False,
                (
                    "The access requester is not the "
                    "selected responder."
                ),
            )

        if (
            request.owner_agent_id
            != candidate.owner_agent_id
        ):
            return (
                False,
                (
                    "The request owner does not match "
                    "the candidate memory owner."
                ),
            )

        if (
            request.memory_id
            != candidate.memory_id
        ):
            return (
                False,
                (
                    "The request memory ID does not "
                    "match the candidate memory ID."
                ),
            )

        selected_agent_ids = set(
            state["selected_agent_ids"]
        )

        if (
            candidate.owner_agent_id
            not in selected_agent_ids
        ):
            return (
                False,
                (
                    "The candidate owner was not "
                    "selected for the active task."
                ),
            )

        if (
            responder_agent_id
            not in selected_agent_ids
        ):
            return (
                False,
                (
                    "The responder was not selected "
                    "for the active task."
                ),
            )

        return (
            True,
            (
                "Task-scoped sharing is allowed "
                "between the candidate owner and the "
                "selected responder."
            ),
        )



    @staticmethod
    def _normalise_review(
        *,
        review: MemoryReviewOutput,
        memory_id: str,
        allowed_related_memory_ids: (
            set[str]
        ),
    ) -> MemoryReviewOutput:
        if review.memory_id != memory_id:
            payload = review.model_dump(
                mode="python"
            )
            payload["memory_id"] = (
                memory_id
            )
        else:
            payload = review.model_dump(
                mode="python"
            )

        payload[
            "related_memory_ids"
        ] = [
            related_id
            for related_id in review.related_memory_ids
            if related_id
            in allowed_related_memory_ids
        ]

        return MemoryReviewOutput.model_validate(
            payload
        )

    @staticmethod
    def _normalise_decision(
        *,
        decision: PromotionDecisionOutput,
        memory_id: str,
        owner_agent_id: str,
        responder_agent_id: str,
    ) -> PromotionDecisionOutput:
        payload = decision.model_dump(
            mode="python"
        )
        payload["memory_id"] = memory_id

        if (
            decision.decision == "approve"
            and decision.target_scope
            == "shared"
        ):
            payload["allowed_agent_ids"] = [
                owner_agent_id,
                responder_agent_id,
            ]
            payload[
                "related_memory_ids"
            ] = []
        elif decision.decision in {
            "reject",
            "keep_private",
        }:
            payload["target_scope"] = (
                "private"
            )
            payload[
                "allowed_agent_ids"
            ] = []
            payload[
                "related_memory_ids"
            ] = []

        return (
            PromotionDecisionOutput
            .model_validate(payload)
        )

    @staticmethod
    def _governance_policy_context(
        *,
        responder_agent_id: str,
    ) -> dict[str, Any]:
        return {
            "policy_name": (
                "Needle Persona persistent "
                "controlled sharing"
            ),
            "sharing_scope": (
                "isolated_experiment_runtime"
            ),
            "permission_persistence": (
                "until_runtime_end"
            ),
            "automatic_revocation": False,
            "global_public_access": False,
            "required_reader_agent_id": (
                responder_agent_id
            ),
            "rules": [
                (
                    "Persona memories are private "
                    "by default."
                ),
                (
                    "Only a memory needed for the "
                    "active question may be promoted."
                ),
                (
                    "The owner retains ownership "
                    "and read access."
                ),
                (
                    "Approval changes the memory to "
                    "permission-controlled shared "
                    "scope and permanently adds the "
                    "requester to readable_by for "
                    "this isolated runtime."
                ),
                (
                    "No permission is inherited by "
                    "other Agents."
                ),
            ],
        }

    # ------------------------------------------------------------------
    # Shared utility
    # ------------------------------------------------------------------

    def _responder(
        self,
        state: NeedlePersonaWorkflowState,
    ) -> Any:
        responder_id = (
            self._required_state_text(
                state,
                "responder_agent_id",
            )
        )
        return self.deps.persona_agent(
            responder_id
        )

    @staticmethod
    def _terminal_status(
        answer: AgentAnswer,
    ) -> str:
        return answer.status

    @staticmethod
    def _ordered_candidates(
        state: NeedlePersonaWorkflowState,
    ) -> list[
        tuple[str, NeedleMemoryCandidate]
    ]:
        return sorted(
            state["candidate_memories"].items(),
            key=lambda item: (
                -item[1].relevance_score,
                item[1].owner_agent_id,
                item[0],
            ),
        )

    @staticmethod
    def _ordered_requests(
        state: NeedlePersonaWorkflowState,
    ) -> list[tuple[str, Any]]:
        candidate_order = {
            memory_id: index
            for index, (
                memory_id,
                _,
            ) in enumerate(
                NeedlePersonaNodes
                ._ordered_candidates(
                    state
                )
            )
        }

        return sorted(
            state["access_requests"].items(),
            key=lambda item: (
                candidate_order.get(
                    item[0],
                    10**9,
                ),
                item[0],
            ),
        )

    @staticmethod
    def _build_request_reason(
        *,
        state: NeedlePersonaWorkflowState,
        candidate: NeedleMemoryCandidate,
    ) -> str:
        """
        Build a neutral, candidate-specific access reason.

        ``state`` remains in the signature because callers already supply it,
        but the initial answer's free-form missing-information list is not
        reused here. That list is a model interpretation rather than a
        trusted workflow fact and may incorrectly constrain later governance.
        """
        _ = state

        return (
            candidate.reason
            + " Evaluate whether this candidate contributes "
            "a relevant fact or intermediate hop needed for "
            "the active benchmark question."
        )

    def _execute(
        self,
        node_name: str,
        operation: Callable[[], NodeUpdate],
    ) -> NodeUpdate:
        started = perf_counter()

        try:
            update = dict(
                operation()
            )
        except NeedlePersonaNodeError:
            raise
        except Exception as error:
            raise NeedlePersonaNodeError(
                node_name,
                str(error),
                cause=error,
            ) from error

        duration_ms = (
            perf_counter() - started
        ) * 1000.0

        if self.config.save_node_trace:
            update["node_trace"] = [
                node_name
            ]

        update["node_metrics"] = {
            f"{node_name}_ms": (
                duration_ms
            )
        }

        update.setdefault(
            "warnings",
            [],
        )
        update.setdefault(
            "errors",
            [],
        )

        return update

    @staticmethod
    def _required_state_text(
        state: Mapping[str, Any],
        field_name: str,
    ) -> str:
        value = state.get(
            field_name
        )
        cleaned = str(
            value or ""
        ).strip()

        if not cleaned:
            raise ValueError(
                f"State field {field_name!r} "
                "cannot be empty."
            )

        return cleaned

    @staticmethod
    def _clean_strings(
        values: Any,
    ) -> list[str]:
        if values is None:
            return []

        if isinstance(values, str):
            values = [values]

        result: list[str] = []

        for value in values:
            cleaned = str(
                value
            ).strip()

            if (
                cleaned
                and cleaned not in result
            ):
                result.append(cleaned)

        return result


__all__ = [
    "NeedlePersonaNodeError",
    "NeedlePersonaNodes",
]