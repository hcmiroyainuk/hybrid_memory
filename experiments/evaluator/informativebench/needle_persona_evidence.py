from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import cast

from .needle_persona_models import (
    AGENT_PERSONA_MAP,
    PERSONA_AGENT_MAP,
    PERSONA_NAMES,
    NeedlePersonaSample,
    PersonaName,
    PrivateMemoryIndex,
)
from .needle_persona_state import (
    NeedlePersonaEvaluationEvidence,
)


class NeedlePersonaEvidenceError(ValueError):
    """
    Raised when deterministic evaluation evidence cannot be constructed safely.
    """


# InformativeBench sample 47 refers to the persona represented by dave_agent as
# "Nancy" in the question. Keeping this alias in the benchmark adapter avoids
# changing the generic agent and memory models.
DEFAULT_PERSONA_ALIASES: dict[str, PersonaName] = {
    "nancy": "dave",
}


_PERSONA_NAME_PATTERN = re.compile(
    r"(?<![A-Za-z])"
    r"(alice|bob|charlie|dave)"
    r"(?![A-Za-z])",
    flags=re.IGNORECASE,
)


def _clean_unique_strings(
    values: Sequence[str],
) -> list[str]:
    """
    Strip, deduplicate, and preserve first-seen order.
    """
    cleaned: list[str] = []

    for value in values:
        item = str(value).strip()

        if item and item not in cleaned:
            cleaned.append(item)

    return cleaned


def _normalise_aliases(
    aliases: Mapping[str, PersonaName] | None,
) -> dict[str, PersonaName]:
    """
    Merge caller aliases with the benchmark's fixed alias mapping.
    """
    normalised: dict[str, PersonaName] = dict(
        DEFAULT_PERSONA_ALIASES
    )

    for alias, persona in (aliases or {}).items():
        clean_alias = str(alias).strip().casefold()

        if not clean_alias:
            raise NeedlePersonaEvidenceError(
                "Persona alias cannot be empty."
            )

        if persona not in PERSONA_NAMES:
            raise NeedlePersonaEvidenceError(
                f"Alias {alias!r} maps to an unknown persona: "
                f"{persona!r}."
            )

        normalised[clean_alias] = persona

    return normalised


def extract_target_personas(
    question: str,
    *,
    aliases: Mapping[str, PersonaName] | None = None,
) -> list[PersonaName]:
    """
    Extract benchmark personas explicitly referenced by a question.

    Detection is deterministic and preserves mention order. Registered aliases
    are resolved to their canonical persona. For example, ``Nancy`` maps to
    ``dave`` for the one InformativeBench record that uses that display name.

    Raises:
        NeedlePersonaEvidenceError:
            if the question is empty or no supported persona can be resolved.
    """
    text = str(question).strip()

    if not text:
        raise NeedlePersonaEvidenceError(
            "question cannot be empty."
        )

    alias_map = _normalise_aliases(aliases)

    token_to_persona: dict[str, PersonaName] = {
        persona: persona
        for persona in PERSONA_NAMES
    }
    token_to_persona.update(alias_map)

    token_pattern = re.compile(
        r"(?<![A-Za-z])("
        + "|".join(
            re.escape(token)
            for token in sorted(
                token_to_persona,
                key=len,
                reverse=True,
            )
        )
        + r")(?![A-Za-z])",
        flags=re.IGNORECASE,
    )

    targets: list[PersonaName] = []

    for match in token_pattern.finditer(text):
        token = match.group(1).casefold()
        persona = token_to_persona[token]

        if persona not in targets:
            targets.append(persona)

    if not targets:
        raise NeedlePersonaEvidenceError(
            "No supported persona name or alias was found "
            f"in question: {question!r}."
        )

    return targets


def build_source_id_to_persona(
    sample: NeedlePersonaSample,
) -> dict[str, PersonaName]:
    """
    Build a source-owner mapping from the validated benchmark sample.
    """
    source_id_to_persona: dict[str, PersonaName] = {}

    for persona in PERSONA_NAMES:
        for source in sample.get_sources(persona):
            source_id = source.source_id.strip()

            if source_id in source_id_to_persona:
                raise NeedlePersonaEvidenceError(
                    f"Duplicate source ID in sample: "
                    f"{source_id!r}."
                )

            if source.persona != persona:
                raise NeedlePersonaEvidenceError(
                    f"Source {source_id!r} is stored under "
                    f"{persona!r} but belongs to "
                    f"{source.persona!r}."
                )

            source_id_to_persona[source_id] = persona

    if not source_id_to_persona:
        raise NeedlePersonaEvidenceError(
            f"Sample {sample.sample_id!r} contains no "
            "persona sources."
        )

    return source_id_to_persona


def build_expected_source_ids(
    sample: NeedlePersonaSample,
    target_personas: Sequence[PersonaName],
) -> list[str]:
    """
    Return all source IDs owned by the personas required by the question.

    The final experiment uses ``persona_document`` granularity, where each
    persona's benchmark dialogue is represented as one owned source document.
    """
    targets: list[PersonaName] = []

    for persona in target_personas:
        if persona not in PERSONA_NAMES:
            raise NeedlePersonaEvidenceError(
                f"Unknown target persona: {persona!r}."
            )

        if persona not in targets:
            targets.append(persona)

    if not targets:
        raise NeedlePersonaEvidenceError(
            "target_personas cannot be empty."
        )

    expected_source_ids: list[str] = []

    for persona in targets:
        sources = sample.get_sources(persona)

        if not sources:
            raise NeedlePersonaEvidenceError(
                f"Target persona {persona!r} has no source "
                f"in sample {sample.sample_id!r}."
            )

        expected_source_ids.extend(
            source.source_id
            for source in sources
        )

    return _clean_unique_strings(
        expected_source_ids
    )


def build_memory_id_to_owner_agent_id(
    memory_index: PrivateMemoryIndex,
) -> dict[str, str]:
    """
    Reverse ``private_memory_ids`` into memory ID -> owner agent ID.

    A memory ID must belong to exactly one persona agent.
    """
    result: dict[str, str] = {}

    for agent_id, memory_ids in (
        memory_index.private_memory_ids.items()
    ):
        if agent_id not in AGENT_PERSONA_MAP:
            raise NeedlePersonaEvidenceError(
                f"Unknown persona agent in memory index: "
                f"{agent_id!r}."
            )

        for memory_id in memory_ids:
            clean_memory_id = str(memory_id).strip()

            if not clean_memory_id:
                raise NeedlePersonaEvidenceError(
                    "Memory IDs cannot be empty."
                )

            previous_owner = result.get(
                clean_memory_id
            )

            if (
                previous_owner is not None
                and previous_owner != agent_id
            ):
                raise NeedlePersonaEvidenceError(
                    f"Memory {clean_memory_id!r} is assigned "
                    f"to multiple owners: {previous_owner!r} "
                    f"and {agent_id!r}."
                )

            result[clean_memory_id] = agent_id

    if not result:
        raise NeedlePersonaEvidenceError(
            "PrivateMemoryIndex contains no memory IDs."
        )

    return result


def build_memory_id_to_source_ids(
    memory_index: PrivateMemoryIndex,
    *,
    known_memory_ids: Sequence[str] | None = None,
) -> dict[str, list[str]]:
    """
    Reverse ``source_to_memory_ids`` into memory ID -> supporting source IDs.

    Multiple source IDs may map to the same extracted memory; source order is
    preserved and duplicates are removed.
    """
    allowed_memory_ids = (
        set(_clean_unique_strings(known_memory_ids))
        if known_memory_ids is not None
        else None
    )

    result: dict[str, list[str]] = {}

    for source_id, memory_ids in (
        memory_index.source_to_memory_ids.items()
    ):
        clean_source_id = str(source_id).strip()

        if not clean_source_id:
            raise NeedlePersonaEvidenceError(
                "Source IDs in the memory index cannot be empty."
            )

        for memory_id in memory_ids:
            clean_memory_id = str(memory_id).strip()

            if not clean_memory_id:
                raise NeedlePersonaEvidenceError(
                    f"Source {clean_source_id!r} references "
                    "an empty memory ID."
                )

            if (
                allowed_memory_ids is not None
                and clean_memory_id
                not in allowed_memory_ids
            ):
                raise NeedlePersonaEvidenceError(
                    f"Source {clean_source_id!r} references "
                    f"unknown memory {clean_memory_id!r}."
                )

            result.setdefault(
                clean_memory_id,
                [],
            )

            if (
                clean_source_id
                not in result[clean_memory_id]
            ):
                result[clean_memory_id].append(
                    clean_source_id
                )

    return result


def map_memory_ids_to_source_ids(
    memory_ids: Sequence[str],
    memory_id_to_source_ids: Mapping[
        str,
        Sequence[str],
    ],
    *,
    strict: bool = True,
) -> list[str]:
    """
    Convert runtime memory IDs into deduplicated source IDs.

    ``strict=True`` prevents missing mappings from silently improving or
    degrading retrieval and governance metrics.
    """
    source_ids: list[str] = []
    unknown_memory_ids: list[str] = []

    for memory_id in _clean_unique_strings(
        memory_ids
    ):
        mapped_sources = (
            memory_id_to_source_ids.get(memory_id)
        )

        if mapped_sources is None:
            unknown_memory_ids.append(memory_id)
            continue

        for source_id in mapped_sources:
            clean_source_id = str(
                source_id
            ).strip()

            if (
                clean_source_id
                and clean_source_id
                not in source_ids
            ):
                source_ids.append(clean_source_id)

    if strict and unknown_memory_ids:
        raise NeedlePersonaEvidenceError(
            "No source mapping exists for memory IDs: "
            f"{unknown_memory_ids}."
        )

    return source_ids


def map_memory_ids_to_owner_agent_ids(
    memory_ids: Sequence[str],
    memory_id_to_owner_agent_id: Mapping[
        str,
        str,
    ],
    *,
    strict: bool = True,
) -> list[str]:
    """
    Convert runtime memory IDs into deduplicated owner-agent IDs.
    """
    owner_agent_ids: list[str] = []
    unknown_memory_ids: list[str] = []

    for memory_id in _clean_unique_strings(
        memory_ids
    ):
        owner_agent_id = (
            memory_id_to_owner_agent_id.get(
                memory_id
            )
        )

        if owner_agent_id is None:
            unknown_memory_ids.append(memory_id)
            continue

        clean_owner = str(owner_agent_id).strip()

        if (
            clean_owner
            and clean_owner
            not in owner_agent_ids
        ):
            owner_agent_ids.append(clean_owner)

    if strict and unknown_memory_ids:
        raise NeedlePersonaEvidenceError(
            "No owner mapping exists for memory IDs: "
            f"{unknown_memory_ids}."
        )

    return owner_agent_ids


def validate_needle_persona_evidence(
    evidence: NeedlePersonaEvaluationEvidence,
) -> None:
    """
    Validate the partitioning and mapping invariants of an evidence object.
    """
    targets = list(
        evidence["target_personas"]
    )

    if not targets:
        raise NeedlePersonaEvidenceError(
            "Evidence target_personas cannot be empty."
        )

    if any(
        persona not in PERSONA_NAMES
        for persona in targets
    ):
        raise NeedlePersonaEvidenceError(
            "Evidence contains an unknown target persona."
        )

    expected = set(
        evidence["expected_source_ids"]
    )
    local = set(
        evidence["expected_local_source_ids"]
    )
    cross_agent = set(
        evidence[
            "expected_cross_agent_source_ids"
        ]
    )

    if not expected:
        raise NeedlePersonaEvidenceError(
            "Evidence expected_source_ids cannot be empty."
        )

    if local & cross_agent:
        raise NeedlePersonaEvidenceError(
            "Local and cross-agent expected sources "
            "must be disjoint."
        )

    if local | cross_agent != expected:
        raise NeedlePersonaEvidenceError(
            "Local and cross-agent expected sources "
            "must partition expected_source_ids."
        )

    memory_to_sources = evidence[
        "memory_id_to_source_ids"
    ]
    memory_to_owner = evidence[
        "memory_id_to_owner_agent_id"
    ]

    unknown_source_mapping_memories = (
        set(memory_to_sources)
        - set(memory_to_owner)
    )

    if unknown_source_mapping_memories:
        raise NeedlePersonaEvidenceError(
            "Source mappings exist for memories without "
            "an owner mapping: "
            f"{sorted(unknown_source_mapping_memories)}."
        )

    mapped_source_ids = {
        source_id
        for source_ids in memory_to_sources.values()
        for source_id in source_ids
    }

    missing_expected_sources = (
        expected - mapped_source_ids
    )

    if missing_expected_sources:
        raise NeedlePersonaEvidenceError(
            "Expected sources were not converted into "
            "memories during ingestion: "
            f"{sorted(missing_expected_sources)}."
        )


def build_needle_persona_evidence(
    *,
    sample: NeedlePersonaSample,
    memory_index: PrivateMemoryIndex,
    responder_agent_id: str,
    aliases: Mapping[str, PersonaName] | None = None,
) -> NeedlePersonaEvaluationEvidence:
    """
    Construct evaluator-only evidence after the workflow has completed.

    Final experiment assumptions:
    - the benchmark sample uses persona-document source granularity;
    - target personas are identified from the question, including the fixed
      Nancy -> Dave benchmark alias;
    - all sources owned by a target persona are required evidence;
    - the responder's sources are local evidence and all remaining target
      sources are cross-agent evidence.

    This function performs no retrieval and does not read or modify MemoryStore.
    It uses only the held-out sample and the ingestion trace.
    """
    if memory_index.sample_id != sample.sample_id:
        raise NeedlePersonaEvidenceError(
            "memory_index.sample_id does not match "
            "sample.sample_id: "
            f"{memory_index.sample_id!r} != "
            f"{sample.sample_id!r}."
        )

    responder_id = str(
        responder_agent_id
    ).strip()

    responder_persona = AGENT_PERSONA_MAP.get(
        responder_id
    )

    if responder_persona is None:
        raise NeedlePersonaEvidenceError(
            f"Unknown responder agent: "
            f"{responder_agent_id!r}."
        )

    source_granularity = str(
        sample.metadata.get(
            "source_granularity",
            "",
        )
    ).strip()

    if (
        source_granularity
        and source_granularity
        != "persona_document"
    ):
        raise NeedlePersonaEvidenceError(
            "The final Needle evaluation protocol requires "
            "source_granularity='persona_document'; "
            f"received {source_granularity!r}."
        )

    target_personas = extract_target_personas(
        sample.question,
        aliases=aliases,
    )

    if responder_persona not in target_personas:
        raise NeedlePersonaEvidenceError(
            f"Responder {responder_id!r} represents "
            f"{responder_persona!r}, which is not one of "
            f"the question targets {target_personas}."
        )

    source_id_to_persona = (
        build_source_id_to_persona(sample)
    )

    expected_source_ids = (
        build_expected_source_ids(
            sample,
            target_personas,
        )
    )

    expected_local_source_ids = [
        source_id
        for source_id in expected_source_ids
        if source_id_to_persona[source_id]
        == responder_persona
    ]

    expected_cross_agent_source_ids = [
        source_id
        for source_id in expected_source_ids
        if source_id_to_persona[source_id]
        != responder_persona
    ]

    memory_id_to_owner_agent_id = (
        build_memory_id_to_owner_agent_id(
            memory_index
        )
    )

    memory_id_to_source_ids = (
        build_memory_id_to_source_ids(
            memory_index,
            known_memory_ids=list(
                memory_id_to_owner_agent_id
            ),
        )
    )

    sample_source_ids = set(
        source_id_to_persona
    )
    indexed_source_ids = set(
        memory_index.source_to_memory_ids
    )

    unknown_indexed_source_ids = (
        indexed_source_ids - sample_source_ids
    )

    if unknown_indexed_source_ids:
        raise NeedlePersonaEvidenceError(
            "PrivateMemoryIndex contains source IDs "
            "that do not belong to the sample: "
            f"{sorted(unknown_indexed_source_ids)}."
        )

    # Confirm that every memory created from a source is owned by that source's
    # persona agent. This catches accidental cross-persona ingestion leakage.
    for memory_id, source_ids in (
        memory_id_to_source_ids.items()
    ):
        owner_agent_id = (
            memory_id_to_owner_agent_id[memory_id]
        )

        for source_id in source_ids:
            source_persona = (
                source_id_to_persona[source_id]
            )
            expected_owner = (
                PERSONA_AGENT_MAP[source_persona]
            )

            if owner_agent_id != expected_owner:
                raise NeedlePersonaEvidenceError(
                    f"Memory {memory_id!r} is owned by "
                    f"{owner_agent_id!r} but source "
                    f"{source_id!r} belongs to "
                    f"{expected_owner!r}."
                )

    evidence = NeedlePersonaEvaluationEvidence(
        target_personas=list(
            target_personas
        ),
        expected_source_ids=list(
            expected_source_ids
        ),
        expected_local_source_ids=list(
            expected_local_source_ids
        ),
        expected_cross_agent_source_ids=list(
            expected_cross_agent_source_ids
        ),
        memory_id_to_source_ids={
            memory_id: list(source_ids)
            for memory_id, source_ids
            in memory_id_to_source_ids.items()
        },
        memory_id_to_owner_agent_id=dict(
            memory_id_to_owner_agent_id
        ),
    )

    validate_needle_persona_evidence(
        evidence
    )

    return evidence


__all__ = [
    "NeedlePersonaEvidenceError",
    "DEFAULT_PERSONA_ALIASES",
    "extract_target_personas",
    "build_source_id_to_persona",
    "build_expected_source_ids",
    "build_memory_id_to_owner_agent_id",
    "build_memory_id_to_source_ids",
    "map_memory_ids_to_source_ids",
    "map_memory_ids_to_owner_agent_ids",
    "validate_needle_persona_evidence",
    "build_needle_persona_evidence",
]