from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.llm import (
    CoordinatorPromptTemplate,
    CriticPromptTemplate,
    PromptTemplates,
    WorkerPromptTemplate,
)

from ..data_preparing.needle_persona_models import (
    AGENT_PERSONA_MAP,
    PERSONA_AGENT_MAP,
    PERSONA_NAMES,
    PersonaName,
)

import json

from ..data_preparing.needle_persona_models import (
    NeedleEvidenceAssessment,
)

ExperimentMode = Literal[
    "private_only",
    "ungoverned_shared",
    "governed_shared",
]

CRITIC_AGENT_ID = "critic_agent"
COORDINATOR_AGENT_ID = "coordinator_agent"


EVIDENCE_SEMANTIC_ENTAILMENT_RULES = """
SEMANTIC EVIDENCE AND ENTAILMENT RULES

Evidence does not need to repeat the exact wording of the question.

Use direct_answer when the candidate explicitly states, paraphrases,
or clearly entails the answer to one atomic question component.

Apply the following conservative semantic mappings:

1. Positive preference expressions

Expressions such as:

- enjoys X
- likes X
- loves X
- prefers X
- finds X exciting
- finds X enjoyable

may support a question component asking what the person enjoys,
likes, or prefers.

2. Negative preference or avoidance expressions

Expressions such as:

- avoids X
- dislikes X
- does not like X
- prefers not to use or do X
- finds X daunting
- finds X intimidating
- finds X uncomfortable
- finds X difficult
- is reluctant to use or do X

may support a question component asking what the person avoids,
prefers to avoid, or is uncomfortable with.

This mapping is allowed only when the negative attitude directly
modifies the same entity, activity, object, or concept.

3. Morphological and lexical variants

Treat singular, plural, hyphenated, and common lexical variants as
equivalent when they clearly refer to the same concept.

Examples:

- stickshift / stickshifts / stick-shift
- comic / comics
- pet / pets
- video game / video games

4. Type and subtype expressions

When the question asks for a type, kind, category, style, or form,
a specifically named variant may directly answer that component.

Do not reject evidence merely because the memory uses the name of
the variant rather than repeating the words "type of" or "kind of".

5. Comparative and multi-person questions

For questions containing terms such as:

- while
- whereas
- but
- and
- both
- compared with

evaluate each person's component independently.

When the responder context supplies one person's side and the
candidate supplies the other person's side about the same concept,
the candidate may be direct_answer.

6. Conservative boundary

Do not use speculative or multi-step world knowledge.

A semantic mapping is valid only when:

- the relevant concept is explicitly present;
- the attitude or relation is explicitly stated;
- the candidate clearly supports one identifiable question component.

A neutral mention of a concept is not sufficient evidence of a
preference or avoidance.
""".strip()

def build_candidate_evidence_recheck_prompt(
    *,
    question: str,
    candidate_memory: dict[str, Any],
    responder_context: str,
) -> str:
    """
    Build an independent Coordinator prompt for rechecking a
    Critic-negative semantic evidence assessment.

    This prompt evaluates evidence contribution only. It must not
    perform access-control or policy evaluation.
    """

    schema = json.dumps(
        NeedleEvidenceAssessment.model_json_schema(),
        indent=2,
        ensure_ascii=False,
    )

    candidate_json = json.dumps(
        candidate_memory,
        indent=2,
        ensure_ascii=False,
        default=str,
    )

    example_output = json.dumps(
        {
            "memory_id": "example_memory",
            "contribution_type": "direct_answer",
            "supported_question_component": (
                "the activity Omar prefers to avoid"
            ),
            "evidence_spans": [
                "Omar finds rock climbing intimidating."
            ],
            "reason": (
                "The candidate directly supports the Omar-side "
                "component. Finding rock climbing intimidating "
                "clearly supports that Omar prefers to avoid it. "
                "The candidate does not need to repeat Lina's "
                "preference."
            ),
            "confidence": 0.95,
        },
        indent=2,
        ensure_ascii=False,
    )

    return f"""
You are the independent evidence-verification component acting for
the memory-governance Coordinator.

Your only task is to determine whether one candidate memory contributes
useful evidence to the active question.

You are performing an independent assessment.

Do not assume that another evaluator approved or rejected the memory.
Do not attempt to agree with a previous evaluator.
Do not perform privacy, permission, ownership-policy, or access-control
evaluation.

STEP 1: DECOMPOSE THE QUESTION

First, silently decompose the active question into atomic answer components.

For example, a question such as:

"What food does Person A enjoy while Person B avoids?"

contains two components:

1. the food Person A enjoys;
2. the food Person B avoids.

Each component must be evaluated separately.

STEP 2: MATCH THE CANDIDATE TO ONE COMPONENT

Determine whether the candidate memory supplies evidence for any one
atomic component.

A candidate does not need to answer the entire question.

A candidate does not need to contain information about every person,
entity, comparison side, clause, or relation mentioned in the question.

Information about one relevant person or one side of a comparison is
sufficient.

STEP 3: APPLY THE CLASSIFICATION RULES

Use direct_answer when:

- the candidate explicitly supplies the answer fact, entity, preference,
  property, event, or relation for any one atomic question component;
- the candidate directly describes what its subject likes, dislikes,
  prefers, avoids, fears, finds difficult, finds daunting, chooses, owns,
  experienced, or believes, when that attitude or fact answers the
  corresponding question component;
- the candidate answers one side of a comparison, contrast, conjunction,
  or multi-person question.

direct_answer means:

"The candidate directly answers at least one component."

It does NOT mean:

"The candidate independently answers the whole question."

Use partial_hop only when:

- the candidate provides a necessary intermediate relation or fact;
- but the candidate does not itself supply the final answer to any atomic
  question component.

Use context_only only when:

- the candidate is topically related;
- but it supplies no answer-bearing fact and no necessary intermediate
  relation.

Use none only when:

- the candidate has no material relationship to any question component.

{EVIDENCE_SEMANTIC_ENTAILMENT_RULES}

CRITICAL ANTI-BIAS RULES

1. Never downgrade direct_answer to context_only merely because another
   component of the question is absent.

2. Never require a candidate owned by one person to also contain the answer
   associated with another person.

3. For questions using words such as "while", "and", "both", "compared
   with", "but", or "whereas", evaluate each side independently.

4. When the candidate explicitly states the answer for one comparison side,
   classify it as direct_answer even when the responder context supplies
   the other side.

5. Use the responder context only to determine how the candidate complements
   already authorised evidence. Do not require the candidate to repeat that
   context.

6. For direct_answer or partial_hop, supported_question_component must name
   the exact component supported by the candidate.

7. For direct_answer or partial_hop, copy one or more exact short evidence
   spans from the candidate.

8. Do not use benchmark gold answers, alternative answers, hidden source
   material, or information from another experiment run.

FEW-SHOT EXAMPLE

Question:
What activity does Lina enjoy while Omar prefers to avoid?

Authorised responder context:
Lina says rock climbing is exhilarating and enjoys doing it.

Candidate memory:
Omar finds rock climbing intimidating.

Correct assessment:
{example_output}

Why:
The candidate directly answers the Omar-side component.

"Finds rock climbing intimidating" clearly supports that Omar prefers
to avoid rock climbing.

It must not be classified as context_only or none merely because the
candidate does not repeat Lina's preference or use the exact phrase
"prefers to avoid".

ACTIVE QUESTION

{question}

CANDIDATE MEMORY

{candidate_json}

RESPONDER'S CURRENTLY AUTHORISED CONTEXT

{responder_context}

REQUIRED STRUCTURED OUTPUT

Return only an object matching this schema:

{schema}
""".strip()


def build_candidate_evidence_prompt(
    *,
    question: str,
    candidate_memory: dict[str, Any],
    responder_context: str,
) -> str:
    """
    Build a Critic prompt that assesses evidence contribution
    only. It does not make an access-control decision.
    """

    schema = json.dumps(
        NeedleEvidenceAssessment.model_json_schema(),
        indent=2,
        ensure_ascii=False,
    )

    candidate_json = json.dumps(
        candidate_memory,
        indent=2,
        ensure_ascii=False,
        default=str,
    )

    return f"""
You are the evidence-assessment component of a memory-governance Critic.

Your only task is to determine whether one candidate memory contributes
evidence to the active question.

Do not approve or reject access.
Do not perform privacy or policy evaluation.
Do not determine whether the whole question can already be answered.
Do not require the candidate to contain information about every person,
entity, or clause named in the question.

Evaluate the candidate's incremental contribution:

- direct_answer:
  The candidate explicitly supplies, paraphrases, or clearly entails an
  answer fact, answer entity, preference, property, event, or relation for
  any atomic component of the question.

- partial_hop:
  The candidate supplies a necessary intermediate fact or relation that
  can be combined with other authorised evidence.

- context_only:
  The candidate is topically related but supplies no answer-bearing fact
  or necessary intermediate relation.

- none:
  The candidate has no material relationship to the question.
  
  {EVIDENCE_SEMANTIC_ENTAILMENT_RULES}

Important rules:

1. Assess the candidate one component at a time.
2. A candidate can be direct_answer even when it answers only the part
   associated with its owner.
3. Never reject or downgrade a candidate merely because information
   about another person named in the question is absent.
4. For direct_answer or partial_hop, identify the supported question
   component and copy exact short evidence spans from the candidate.
5. Use only the supplied question, candidate memory, and authorised
   responder context.
6. Do not use benchmark gold answers or hidden source material.
7. Return only the required structured output.

Active question:
{question}

Candidate memory:
{candidate_json}

Responder's currently authorised context:
{responder_context}

Required structured output:
{schema}
""".strip()


@dataclass(frozen=True)
class NeedlePersonaPromptBundle:
    """
    Dataset-specific prompt templates used by the Needle in the Persona
    evaluation.

    This module configures the generic templates from ``src.llm`` with the
    benchmark's concrete roles and policy. It does not call the LLM, read or
    write memory, mutate workflow state, or contain evaluation answers.
    """

    worker_prompts: dict[str, WorkerPromptTemplate]
    critic_prompt: CriticPromptTemplate
    coordinator_prompt: CoordinatorPromptTemplate

    def worker(
        self,
        agent_id: str,
    ) -> WorkerPromptTemplate:
        """
        Return the configured worker prompt for one persona agent.
        """
        clean_agent_id = str(agent_id).strip()

        if clean_agent_id not in self.worker_prompts:
            raise KeyError(
                f"Unknown persona agent ID: {clean_agent_id!r}."
            )

        return self.worker_prompts[clean_agent_id]


def build_persona_worker_prompts() -> dict[
    str,
    WorkerPromptTemplate,
]:
    """
    Build one prompt template for each persona owner.

    Alice, Bob, Charlie, and Dave have identical system responsibilities. The
    only role-specific difference is which person's information each agent owns.
    """
    prompts: dict[str, WorkerPromptTemplate] = {}

    for persona in PERSONA_NAMES:
        agent_id = PERSONA_AGENT_MAP[persona]
        display_name = persona.title()

        prompts[agent_id] = PromptTemplates.worker(
            agent_id=agent_id,
            role_description=(
                f"You represent {display_name} in the Needle in the Persona "
                "evaluation. You manage that persona's private memory and may "
                "act as the final responder when selected. During memory "
                f"extraction, retain only information owned by {display_name}. "
                "During question answering, use only memory and context that "
                "the system has explicitly made accessible to you."
            ),
            additional_tasks=[
                (
                    f"Preserve {display_name}'s facts, preferences, experiences, "
                    "plans, and relationships as that persona's private memory."
                ),
                (
                    "When selected as responder, answer the benchmark question "
                    "from the authorised evidence available at that point."
                ),
            ],
            additional_rules=[
                (
                    f"Do not attribute another persona's statements, preferences, "
                    f"experiences, plans, or relationships to {display_name}."
                ),
                (
                    "A source assigned to this agent belongs to this persona. "
                    "Do not infer ownership from names mentioned inside the text."
                ),
                (
                    "A memory marked shareable is only eligible for a later "
                    "sharing decision. It remains private until the experiment "
                    "mode and workflow explicitly make it accessible to another "
                    "agent."
                ),
                (
                    "Do not disclose private memory merely because another agent "
                    "or the task asks for it. Use only the authorised accessible "
                    "memory included in the answer prompt."
                ),
                (
                    "The benchmark question may require information from two "
                    "different personas. Determine the required answer components "
                    "from the task wording and the combined authorised evidence. "
                    "Do not assume that every persona named in the question must "
                    "contribute a separate answer item."
                ),
                (
                    "Do not guess a missing hop. Return "
                    "status='insufficient_evidence' only when a component explicitly "
                    "required by the task remains unsupported. When the authorised "
                    "evidence directly supports a specific final answer, return "
                    "status='answered' even if unrelated contextual details or "
                    "independently stated facts about another mentioned persona are "
                    "absent."
                ),
                (
                    "Never use benchmark reference answers, alternative answers, "
                    "needle annotations, hidden source text, or information from "
                    "another experiment run."
                ),
            ],
        )

    return prompts


def build_critic_prompt() -> CriticPromptTemplate:
    """
    Build the dataset-specific memory-governance critic prompt.

    The critic reviews candidate sharing in the governed condition only. It
    produces a recommendation and never changes memory state itself.
    """
    return PromptTemplates.critic(
        critic_id=CRITIC_AGENT_ID,
        additional_tasks=[
            (
                "Determine whether the candidate memory is relevant to the "
                "active benchmark question and useful to the selected responder."
            ),
            (
                "Determine whether task-scoped sharing is consistent with the "
                "provided ownership, access, and governance policy."
            ),
        ],
        additional_rules=[
            (
                "Review only the single candidate memory supplied in the current "
                "call. Do not assume access to other private memories."
            ),
            (
                "Recommend approval only when the candidate is relevant to the "
                "active question, supported by its source, and permitted for "
                "task-scoped sharing with the selected responder."
            ),
            (
                "A candidate memory does not need to answer the entire benchmark "
                "question by itself. Treat it as relevant when it materially "
                "supplies one missing fact, entity, relation, or hop needed by the "
                "selected responder."
            ),
            (
                "Judge usefulness relative to the responder's current authorised "
                "context and stated information need. Do not reject a candidate "
                "solely because it concerns only its owner or does not contain "
                "facts about every persona named in the question."
            ),
            (
                "Recommend rejection or keep_private when the memory is "
                "irrelevant, unsupported, policy-violating, or unnecessary for "
                "the active task."
            ),
            (
                "Use duplicate, conflict, outdated, merge, or supersede "
                "classifications only when the supplied existing-memory context "
                "contains evidence for that classification."
            ),
            (
                "Do not answer the benchmark question and do not decide the final "
                "read ACL. The coordinator retains final authority."
            ),
            (
                "Never use benchmark gold answers, alternative answers, needle "
                "annotations, or hidden persona sources."
            ),
        ],
    )


def build_coordinator_prompt() -> CoordinatorPromptTemplate:
    """
    Build the dataset-specific coordinator prompt.

    The coordinator may route the question and make final governed-sharing
    decisions. It never produces the benchmark's final natural-language answer.
    """
    return PromptTemplates.coordinator(
        coordinator_id=COORDINATOR_AGENT_ID,
        additional_tasks=[
            (
                "For routing, identify which persona agents may hold the "
                "information required by the two-hop question and select one "
                "persona agent as responder."
            ),
            (
                "For governed sharing, make the final task-scoped promotion "
                "decision after considering the critic review and active policy."
            ),
        ],
        additional_rules=[
            (
                "The only persona agents are alice_agent, bob_agent, "
                "charlie_agent, and dave_agent. Select only IDs explicitly "
                "present in the supplied available-agent context."
            ),
            (
                "The coordinator coordinates access and decisions; it must never "
                "answer the benchmark question itself."
            ),
            (
                "Do not transfer raw private memory during routing. Routing may "
                "identify agents and information needs only."
            ),
            (
                "In governed sharing, approve a memory only when it is relevant "
                "to the active question, policy-compliant, and needed by the "
                "selected responder."
            ),
            (
                "A candidate may be approved when it materially supplies one "
                "necessary missing fact or hop. It does not need to answer the "
                "entire benchmark question independently."
            ),
            (
                "Do not reject a candidate solely because it lacks information "
                "about another persona named in the question. Evaluate whether "
                "the candidate contributes to the combined evidence required by "
                "the selected responder."
            ),
            (
                "Treat a critic recommendation that mistakes useful partial "
                "evidence for an incomplete whole-task answer as advisory only; "
                "apply the relevance and policy rules independently."
            ),
            (
                "Approved sharing is task-scoped rather than global. The source "
                "owner and selected responder must remain authorised readers."
            ),
            (
                "Reject or keep private any unrelated, unsupported, "
                "policy-violating, duplicate, outdated, or unresolved conflicting "
                "memory."
            ),
            (
                "Treat the critic review as advice. Apply the policy independently "
                "and explain the final decision briefly."
            ),
            (
                "Never use benchmark gold answers, alternative answers, needle "
                "annotations, or hidden persona sources."
            ),
        ],
    )


def build_needle_persona_prompts() -> NeedlePersonaPromptBundle:
    """
    Construct the complete dataset-specific prompt bundle.
    """
    return NeedlePersonaPromptBundle(
        worker_prompts=build_persona_worker_prompts(),
        critic_prompt=build_critic_prompt(),
        coordinator_prompt=build_coordinator_prompt(),
    )


def build_available_agent_context() -> dict[str, dict[str, Any]]:
    """
    Return the explicit persona-agent registry supplied to a routing prompt.

    The descriptions state ownership and responder eligibility without exposing
    any private memory content.
    """
    return {
        PERSONA_AGENT_MAP[persona]: {
            "persona": persona,
            "display_name": persona.title(),
            "owns_private_memory_for": persona,
            "may_act_as_responder": True,
            "may_approve_sharing": False,
        }
        for persona in PERSONA_NAMES
    }


def build_governance_policy_context(
    *,
    responder_agent_id: str,
) -> dict[str, Any]:
    """
    Return the task-scoped sharing policy supplied to critic and coordinator
    prompts in the governed condition.
    """
    clean_responder_id = str(
        responder_agent_id
    ).strip()

    if clean_responder_id not in AGENT_PERSONA_MAP:
        raise ValueError(
            "responder_agent_id must be one of the registered persona "
            f"agents; received {clean_responder_id!r}."
        )

    return {
        "policy_name": (
            "Needle in the Persona task-scoped memory sharing"
        ),
        "sharing_scope": "active_task_only",
        "global_sharing_allowed": False,
        "required_reader_agent_id": clean_responder_id,
        "rules": [
            (
                "Persona memories are private by default."
            ),
            (
                "Only memories relevant to the active question may be "
                "considered for sharing."
            ),
            (
                "In governed mode, a candidate may become shared only after "
                "critic review and coordinator approval."
            ),
            (
                "The selected responder must be an authorised reader of every "
                "approved shared memory it needs."
            ),
            (
                "The source owner must retain access to an approved shared "
                "memory."
            ),
            (
                "Unrelated, unsupported, policy-violating, duplicate, outdated, "
                "or unresolved conflicting memories must not be blindly "
                "promoted."
            ),
        ],
    }


def build_experiment_mode_context(
    mode: ExperimentMode,
) -> dict[str, Any]:
    """
    Return a small, explicit description of the active experiment condition.

    This is context for prompts and logs only. It does not implement the mode's
    state transitions.
    """
    if mode == "private_only":
        return {
            "experiment_mode": mode,
            "sharing_enabled": False,
            "critic_review_required": False,
            "coordinator_approval_required": False,
            "description": (
                "Each persona agent may use only its own private memory. "
                "No cross-agent memory sharing is allowed."
            ),
        }

    if mode == "ungoverned_shared":
        return {
            "experiment_mode": mode,
            "sharing_enabled": True,
            "critic_review_required": False,
            "coordinator_approval_required": False,
            "description": (
                "Task-relevant candidate memories may be shared directly with "
                "the selected responder without critic or coordinator review."
            ),
        }

    if mode == "governed_shared":
        return {
            "experiment_mode": mode,
            "sharing_enabled": True,
            "critic_review_required": True,
            "coordinator_approval_required": True,
            "description": (
                "Task-relevant candidate memories require critic review and "
                "coordinator approval before the selected responder may access "
                "them."
            ),
        }

    raise ValueError(
        f"Unsupported experiment mode: {mode!r}."
    )


__all__ = [
    "ExperimentMode",
    "CRITIC_AGENT_ID",
    "COORDINATOR_AGENT_ID",
    "NeedlePersonaPromptBundle",
    "build_persona_worker_prompts",
    "build_critic_prompt",
    "build_coordinator_prompt",
    "build_needle_persona_prompts",
    "build_available_agent_context",
    "build_governance_policy_context",
    "build_experiment_mode_context",
    "build_candidate_evidence_recheck_prompt",
]