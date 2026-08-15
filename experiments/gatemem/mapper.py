from __future__ import annotations

"""
GateMem-to-system mapping layer.

This module only normalises GateMem data. It does not classify memory events,
assign ACLs, retrieve memories, or generate answers.

Checkpoint evaluation labels are intentionally excluded from MappedQuery so
they cannot leak into the evaluated system.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


GATEMEM_EVALUATION_ONLY_CHECKPOINT_FIELDS = frozenset(
    {
        "query_type",
        "attack_type",
        "expected_action",
        "judge_spec",
        "leak_targets",
    }
)

_PRINCIPAL_FIELDS = {"principal_id", "role", "display_name"}
_RELATIONSHIP_FIELDS = {
    "type",
    "principal_id",
    "for_principal_id",
    "access_scope",
    "expires",
}
_TURN_FIELDS = {"turn_id", "timestamp", "speaker", "turn_kind", "text"}
_EPISODE_FIELDS = {"episode_id", "domain", "entities", "turns"}


class GateMemMappingError(ValueError):
    """Base error raised when GateMem data cannot be mapped safely."""


class InvalidGateMemEpisodeError(GateMemMappingError):
    """Raised when an episode is malformed or internally inconsistent."""


class InvalidGateMemCheckpointError(GateMemMappingError):
    """Raised when a checkpoint is malformed or inconsistent."""


class GateMemReferenceError(GateMemMappingError):
    """Raised when a principal, turn, or episode reference cannot be resolved."""


class MappedPrincipal(BaseModel):
    """Principal identity used to create one Worker."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    agent_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_worker_kwargs(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "display_name": self.display_name,
            "metadata": dict(self.metadata),
        }


class MappedRelationship(BaseModel):
    """
    Relationship used as Coordinator/Critic policy context.

    Resource-specific IDs such as project_id, program_id, course_id,
    building_id, and case_id are kept in resource_refs.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    relationship_type: str = Field(min_length=1)
    principal_agent_id: str = Field(min_length=1)
    target_agent_id: str | None = None
    access_scope: str | None = None
    resource_refs: dict[str, str] = Field(default_factory=dict)
    expires: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)

    def to_policy_record(self) -> dict[str, Any]:
        return self.model_dump(mode="python", exclude_none=True)


class MappedTurn(BaseModel):
    """
    One GateMem turn prepared for memory ingestion.

    Event interpretation such as CREATE, UPDATE, DELETE, or POLICY_CHANGE
    belongs in a separate event classifier.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    episode_id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    sequence_index: int = Field(ge=0)

    speaker_agent_id: str = Field(min_length=1)
    speaker_role: str = Field(min_length=1)

    text: str = Field(min_length=1)
    timestamp: str | None = None
    turn_kind: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def memory_id(self) -> str:
        """Deterministic memory ID scoped to one episode."""
        return f"{self.episode_id}:{self.turn_id}"

    def to_memory_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "source": "gatemem",
            "domain": self.domain,
            "episode_id": self.episode_id,
            "turn_id": self.turn_id,
            "sequence_index": self.sequence_index,
            "speaker_agent_id": self.speaker_agent_id,
            "speaker_role": self.speaker_role,
        }
        if self.timestamp is not None:
            metadata["timestamp"] = self.timestamp
        if self.turn_kind is not None:
            metadata["turn_kind"] = self.turn_kind
        metadata.update(self.metadata)
        return metadata


class MappedQuery(BaseModel):
    """
    Runtime-safe GateMem checkpoint query.

    It deliberately excludes query_type, attack_type, expected_action,
    judge_spec, and leak_targets.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    checkpoint_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    as_of_turn_id: str = Field(min_length=1)

    requester_agent_id: str = Field(min_length=1)
    requester_role: str = Field(min_length=1)
    query_text: str = Field(min_length=1)

    def to_task_context(self) -> dict[str, Any]:
        return self.model_dump(mode="python")


class MappedEpisode(BaseModel):
    """Mapped episode and its isolated runtime namespace."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    episode_id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    principals: tuple[MappedPrincipal, ...]
    relationships: tuple[MappedRelationship, ...]
    turns: tuple[MappedTurn, ...]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def namespace_id(self) -> str:
        return self.episode_id

    @property
    def agent_ids(self) -> tuple[str, ...]:
        return tuple(principal.agent_id for principal in self.principals)

    def principal_index(self) -> dict[str, MappedPrincipal]:
        return {principal.agent_id: principal for principal in self.principals}

    def turn_index(self) -> dict[str, MappedTurn]:
        return {turn.turn_id: turn for turn in self.turns}

    def require_principal(self, agent_id: str) -> MappedPrincipal:
        clean_id = _required_text(agent_id, "agent_id", GateMemReferenceError)
        try:
            return self.principal_index()[clean_id]
        except KeyError as error:
            raise GateMemReferenceError(
                f"Principal {clean_id!r} is not registered in "
                f"episode {self.episode_id!r}."
            ) from error

    def require_turn(self, turn_id: str) -> MappedTurn:
        clean_id = _required_text(turn_id, "turn_id", GateMemReferenceError)
        try:
            return self.turn_index()[clean_id]
        except KeyError as error:
            raise GateMemReferenceError(
                f"Turn {clean_id!r} does not exist in "
                f"episode {self.episode_id!r}."
            ) from error

    def turns_up_to(self, as_of_turn_id: str) -> tuple[MappedTurn, ...]:
        """Return turns from the start through the checkpoint boundary."""
        boundary = self.require_turn(as_of_turn_id)
        return self.turns[: boundary.sequence_index + 1]

    def to_policy_context(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "domain": self.domain,
            "principals": [
                principal.model_dump(mode="python")
                for principal in self.principals
            ],
            "relationships": [
                relationship.to_policy_record()
                for relationship in self.relationships
            ],
        }


class GateMemMapper:
    """
    Domain-neutral mapper for education, medical, office, and household data.
    """

    def map_principal(
        self,
        payload: Mapping[str, Any],
        *,
        path: str = "principal",
    ) -> MappedPrincipal:
        raw = _required_mapping(payload, path, InvalidGateMemEpisodeError)
        return MappedPrincipal(
            agent_id=_required_text(
                raw.get("principal_id"),
                f"{path}.principal_id",
                InvalidGateMemEpisodeError,
            ),
            role=_required_text(
                raw.get("role"),
                f"{path}.role",
                InvalidGateMemEpisodeError,
            ),
            display_name=_required_text(
                raw.get("display_name"),
                f"{path}.display_name",
                InvalidGateMemEpisodeError,
            ),
            metadata={
                key: value
                for key, value in raw.items()
                if key not in _PRINCIPAL_FIELDS
            },
        )

    def map_relationship(
        self,
        payload: Mapping[str, Any],
        *,
        path: str = "relationship",
    ) -> MappedRelationship:
        raw = _required_mapping(payload, path, InvalidGateMemEpisodeError)
        resource_refs: dict[str, str] = {}
        attributes: dict[str, Any] = {}

        for key, value in raw.items():
            if key in _RELATIONSHIP_FIELDS:
                continue
            if key.endswith("_id"):
                resource_refs[key] = _required_text(
                    value,
                    f"{path}.{key}",
                    InvalidGateMemEpisodeError,
                )
            else:
                attributes[key] = value

        return MappedRelationship(
            relationship_type=_required_text(
                raw.get("type"),
                f"{path}.type",
                InvalidGateMemEpisodeError,
            ),
            principal_agent_id=_required_text(
                raw.get("principal_id"),
                f"{path}.principal_id",
                InvalidGateMemEpisodeError,
            ),
            target_agent_id=_optional_text(
                raw.get("for_principal_id"),
                f"{path}.for_principal_id",
                InvalidGateMemEpisodeError,
            ),
            access_scope=_optional_text(
                raw.get("access_scope"),
                f"{path}.access_scope",
                InvalidGateMemEpisodeError,
            ),
            resource_refs=resource_refs,
            expires=_optional_text(
                raw.get("expires"),
                f"{path}.expires",
                InvalidGateMemEpisodeError,
            ),
            attributes=attributes,
        )

    def map_turn(
        self,
        payload: Mapping[str, Any],
        *,
        episode_id: str,
        domain: str,
        sequence_index: int,
        path: str = "turn",
    ) -> MappedTurn:
        raw = _required_mapping(payload, path, InvalidGateMemEpisodeError)
        speaker = _required_mapping(
            raw.get("speaker"),
            f"{path}.speaker",
            InvalidGateMemEpisodeError,
        )
        if not isinstance(sequence_index, int) or sequence_index < 0:
            raise InvalidGateMemEpisodeError(
                f"{path}.sequence_index must be a non-negative integer."
            )

        return MappedTurn(
            episode_id=_required_text(
                episode_id,
                "episode_id",
                InvalidGateMemEpisodeError,
            ),
            domain=_required_text(
                domain,
                "domain",
                InvalidGateMemEpisodeError,
            ),
            turn_id=_required_text(
                raw.get("turn_id"),
                f"{path}.turn_id",
                InvalidGateMemEpisodeError,
            ),
            sequence_index=sequence_index,
            speaker_agent_id=_required_text(
                speaker.get("principal_id"),
                f"{path}.speaker.principal_id",
                InvalidGateMemEpisodeError,
            ),
            speaker_role=_required_text(
                speaker.get("role"),
                f"{path}.speaker.role",
                InvalidGateMemEpisodeError,
            ),
            text=_required_text(
                raw.get("text"),
                f"{path}.text",
                InvalidGateMemEpisodeError,
            ),
            timestamp=_optional_text(
                raw.get("timestamp"),
                f"{path}.timestamp",
                InvalidGateMemEpisodeError,
            ),
            turn_kind=_optional_text(
                raw.get("turn_kind"),
                f"{path}.turn_kind",
                InvalidGateMemEpisodeError,
            ),
            metadata={
                key: value
                for key, value in raw.items()
                if key not in _TURN_FIELDS
            },
        )

    def map_episode(self, payload: Mapping[str, Any]) -> MappedEpisode:
        raw = _required_mapping(
            payload,
            "episode",
            InvalidGateMemEpisodeError,
        )
        episode_id = _required_text(
            raw.get("episode_id"),
            "episode.episode_id",
            InvalidGateMemEpisodeError,
        )
        domain = _required_text(
            raw.get("domain"),
            "episode.domain",
            InvalidGateMemEpisodeError,
        )
        entities = _required_mapping(
            raw.get("entities"),
            "episode.entities",
            InvalidGateMemEpisodeError,
        )
        raw_principals = _required_sequence(
            entities.get("principals"),
            "episode.entities.principals",
            InvalidGateMemEpisodeError,
        )
        raw_relationship_payload = entities.get("relationships")

        if domain == "medical" and raw_relationship_payload is None:
            raw_relationships = []
        else:
            raw_relationships = _required_sequence(
                raw_relationship_payload,
                "episode.entities.relationships",
                InvalidGateMemEpisodeError,
            )
        raw_turns = _required_sequence(
            raw.get("turns"),
            "episode.turns",
            InvalidGateMemEpisodeError,
        )

        principals = tuple(
            self.map_principal(
                principal,
                path=f"episode.entities.principals[{index}]",
            )
            for index, principal in enumerate(raw_principals)
        )
        principal_index = self._validate_principals(principals, episode_id)

        if domain == "medical":
            normalised_relationships = []

            for index, relationship in enumerate(raw_relationships):
                relationship = dict(relationship)

                # Medical relationships use domain-specific principal ID fields,
                # e.g. clinician_id / family_id + patient_id.
                if not relationship.get("principal_id"):
                    source_principal_id = None

                    for key in (
                            "clinician_id",
                            "family_id",
                    ):
                        value = relationship.get(key)

                        if isinstance(value, str) and value.strip():
                            source_principal_id = value.strip()
                            break

                    if source_principal_id is None:
                        raise InvalidGateMemEpisodeError(
                            "Could not determine source principal for "
                            f"episode.entities.relationships[{index}]: "
                            f"{relationship!r}"
                        )

                    relationship["principal_id"] = (
                        source_principal_id
                    )

                if not relationship.get("for_principal_id"):
                    patient_id = relationship.get("patient_id")

                    if (
                            isinstance(patient_id, str)
                            and patient_id.strip()
                    ):
                        relationship["for_principal_id"] = (
                            patient_id.strip()
                        )

                # Remove Medical-specific aliases after converting
                # them to the canonical GateMem mapper fields.
                relationship.pop("clinician_id", None)
                relationship.pop("family_id", None)
                relationship.pop("patient_id", None)

                normalised_relationships.append(
                    self.map_relationship(
                        relationship,
                        path=(
                            "episode.entities.relationships"
                            f"[{index}]"
                        ),
                    )
                )

            relationships = tuple(
                normalised_relationships
            )

        else:
            relationships = tuple(
                self.map_relationship(
                    relationship,
                    path=(
                        "episode.entities.relationships"
                        f"[{index}]"
                    ),
                )
                for index, relationship
                in enumerate(raw_relationships)
            )

        turns = tuple(
            self.map_turn(
                turn,
                episode_id=episode_id,
                domain=domain,
                sequence_index=index,
                path=f"episode.turns[{index}]",
            )
            for index, turn in enumerate(raw_turns)
        )
        self._validate_turns(turns, principal_index, episode_id)

        metadata = {
            key: value
            for key, value in raw.items()
            if key not in _EPISODE_FIELDS
        }
        entity_metadata = {
            key: value
            for key, value in entities.items()
            if key not in {"principals", "relationships"}
        }
        if entity_metadata:
            metadata["entity_metadata"] = entity_metadata

        return MappedEpisode(
            episode_id=episode_id,
            domain=domain,
            principals=principals,
            relationships=relationships,
            turns=turns,
            metadata=metadata,
        )

    def map_checkpoint(
        self,
        payload: Mapping[str, Any],
        *,
        episode: MappedEpisode | None = None,
    ) -> MappedQuery:
        """
        Map only runtime-safe checkpoint fields.

        Evaluation labels present in the raw JSON are silently discarded.
        """
        raw = _required_mapping(
            payload,
            "checkpoint",
            InvalidGateMemCheckpointError,
        )
        asker = _required_mapping(
            raw.get("asker"),
            "checkpoint.asker",
            InvalidGateMemCheckpointError,
        )

        mapped = MappedQuery(
            checkpoint_id=_required_text(
                raw.get("checkpoint_id"),
                "checkpoint.checkpoint_id",
                InvalidGateMemCheckpointError,
            ),
            episode_id=_required_text(
                raw.get("episode_id"),
                "checkpoint.episode_id",
                InvalidGateMemCheckpointError,
            ),
            as_of_turn_id=_required_text(
                raw.get("as_of_turn_id"),
                "checkpoint.as_of_turn_id",
                InvalidGateMemCheckpointError,
            ),
            requester_agent_id=_required_text(
                asker.get("principal_id"),
                "checkpoint.asker.principal_id",
                InvalidGateMemCheckpointError,
            ),
            requester_role=_required_text(
                asker.get("role"),
                "checkpoint.asker.role",
                InvalidGateMemCheckpointError,
            ),
            query_text=_required_text(
                raw.get("query_text"),
                "checkpoint.query_text",
                InvalidGateMemCheckpointError,
            ),
        )

        if episode is not None:
            self._validate_checkpoint(mapped, episode)
        return mapped

    def map_episodes(
        self,
        payloads: Iterable[Mapping[str, Any]],
    ) -> tuple[MappedEpisode, ...]:
        episodes = tuple(self.map_episode(payload) for payload in payloads)
        seen: set[str] = set()
        for episode in episodes:
            if episode.episode_id in seen:
                raise InvalidGateMemEpisodeError(
                    f"Duplicate episode_id: {episode.episode_id!r}."
                )
            seen.add(episode.episode_id)
        return episodes

    def map_checkpoints(
        self,
        payloads: Iterable[Mapping[str, Any]],
        *,
        episodes: (
            Mapping[str, MappedEpisode]
            | Sequence[MappedEpisode]
            | None
        ) = None,
    ) -> tuple[MappedQuery, ...]:
        episode_index = _normalise_episode_index(episodes)
        checkpoints: list[MappedQuery] = []
        seen: set[str] = set()

        for payload in payloads:
            raw = _required_mapping(
                payload,
                "checkpoint",
                InvalidGateMemCheckpointError,
            )
            episode_id = _required_text(
                raw.get("episode_id"),
                "checkpoint.episode_id",
                InvalidGateMemCheckpointError,
            )
            episode = None
            if episode_index is not None:
                try:
                    episode = episode_index[episode_id]
                except KeyError as error:
                    raise GateMemReferenceError(
                        f"Checkpoint references unknown episode "
                        f"{episode_id!r}."
                    ) from error

            checkpoint = self.map_checkpoint(payload, episode=episode)
            if checkpoint.checkpoint_id in seen:
                raise InvalidGateMemCheckpointError(
                    f"Duplicate checkpoint_id: "
                    f"{checkpoint.checkpoint_id!r}."
                )
            seen.add(checkpoint.checkpoint_id)
            checkpoints.append(checkpoint)

        return tuple(checkpoints)

    def load_episodes_jsonl(
        self,
        path: str | Path,
    ) -> tuple[MappedEpisode, ...]:
        return self.map_episodes(_read_jsonl(path))

    def load_checkpoints_jsonl(
        self,
        path: str | Path,
        *,
        episodes: (
            Mapping[str, MappedEpisode]
            | Sequence[MappedEpisode]
            | None
        ) = None,
    ) -> tuple[MappedQuery, ...]:
        return self.map_checkpoints(_read_jsonl(path), episodes=episodes)

    @staticmethod
    def runtime_checkpoint_payload(
        checkpoint: MappedQuery,
    ) -> dict[str, Any]:
        """Return the exact checkpoint payload permitted into the system."""
        if not isinstance(checkpoint, MappedQuery):
            raise TypeError("checkpoint must be a MappedQuery instance.")
        return checkpoint.model_dump(mode="python")

    @staticmethod
    def _validate_principals(
        principals: Sequence[MappedPrincipal],
        episode_id: str,
    ) -> dict[str, MappedPrincipal]:
        if not principals:
            raise InvalidGateMemEpisodeError(
                f"Episode {episode_id!r} must contain at least one principal."
            )

        index: dict[str, MappedPrincipal] = {}
        for principal in principals:
            if principal.agent_id in index:
                raise InvalidGateMemEpisodeError(
                    f"Episode {episode_id!r} contains duplicate principal_id "
                    f"{principal.agent_id!r}."
                )
            index[principal.agent_id] = principal
        return index

    @staticmethod
    def _validate_relationships(
        relationships: Sequence[MappedRelationship],
        principals: Mapping[str, MappedPrincipal],
        episode_id: str,
    ) -> None:
        for relationship in relationships:
            if relationship.principal_agent_id not in principals:
                raise GateMemReferenceError(
                    f"Episode {episode_id!r} relationship references unknown "
                    f"principal {relationship.principal_agent_id!r}."
                )
            if (
                relationship.target_agent_id is not None
                and relationship.target_agent_id not in principals
            ):
                raise GateMemReferenceError(
                    f"Episode {episode_id!r} relationship references unknown "
                    f"target principal {relationship.target_agent_id!r}."
                )

    @staticmethod
    def _validate_turns(
        turns: Sequence[MappedTurn],
        principals: Mapping[str, MappedPrincipal],
        episode_id: str,
    ) -> None:
        seen: set[str] = set()
        for expected_index, turn in enumerate(turns):
            if turn.sequence_index != expected_index:
                raise InvalidGateMemEpisodeError(
                    f"Episode {episode_id!r} has a non-contiguous turn order."
                )
            if turn.turn_id in seen:
                raise InvalidGateMemEpisodeError(
                    f"Episode {episode_id!r} contains duplicate turn_id "
                    f"{turn.turn_id!r}."
                )
            seen.add(turn.turn_id)

            try:
                principal = principals[turn.speaker_agent_id]
            except KeyError as error:
                raise GateMemReferenceError(
                    f"Turn {turn.turn_id!r} references unknown speaker "
                    f"{turn.speaker_agent_id!r}."
                ) from error

            if principal.role != turn.speaker_role:
                raise InvalidGateMemEpisodeError(
                    f"Turn {turn.turn_id!r} speaker role "
                    f"{turn.speaker_role!r} does not match registered role "
                    f"{principal.role!r}."
                )

    @staticmethod
    def _validate_checkpoint(
        checkpoint: MappedQuery,
        episode: MappedEpisode,
    ) -> None:
        if checkpoint.episode_id != episode.episode_id:
            raise InvalidGateMemCheckpointError(
                "Checkpoint episode_id does not match the supplied episode: "
                f"{checkpoint.episode_id!r} != {episode.episode_id!r}."
            )

        principal = episode.require_principal(checkpoint.requester_agent_id)
        if principal.role != checkpoint.requester_role:
            raise InvalidGateMemCheckpointError(
                "Checkpoint asker role does not match the principal registry: "
                f"{checkpoint.requester_role!r} != {principal.role!r}."
            )
        episode.require_turn(checkpoint.as_of_turn_id)


def _read_jsonl(path: str | Path) -> Iterable[Mapping[str, Any]]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"GateMem JSONL file not found: {file_path}.")

    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise GateMemMappingError(
                    f"Invalid JSON on line {line_number} of {file_path}."
                ) from error
            if not isinstance(payload, Mapping):
                raise GateMemMappingError(
                    f"Line {line_number} of {file_path} must be a JSON object."
                )
            yield payload


def _normalise_episode_index(
    episodes: (
        Mapping[str, MappedEpisode]
        | Sequence[MappedEpisode]
        | None
    ),
) -> dict[str, MappedEpisode] | None:
    if episodes is None:
        return None

    if isinstance(episodes, Mapping):
        result: dict[str, MappedEpisode] = {}
        for key, episode in episodes.items():
            if not isinstance(episode, MappedEpisode):
                raise TypeError(
                    "episodes mapping values must be MappedEpisode instances."
                )
            clean_key = _required_text(key, "episode index key", TypeError)
            if clean_key != episode.episode_id:
                raise ValueError(
                    "Episode index key does not match episode_id: "
                    f"{clean_key!r} != {episode.episode_id!r}."
                )
            result[clean_key] = episode
        return result

    if isinstance(episodes, (str, bytes, bytearray)):
        raise TypeError(
            "episodes must be a mapping or sequence of MappedEpisode objects."
        )

    result = {}
    for episode in episodes:
        if not isinstance(episode, MappedEpisode):
            raise TypeError(
                "episodes sequence values must be MappedEpisode instances."
            )
        if episode.episode_id in result:
            raise ValueError(f"Duplicate episode_id: {episode.episode_id!r}.")
        result[episode.episode_id] = episode
    return result


def _required_mapping(
    value: Any,
    field_name: str,
    error_type: type[Exception],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error_type(f"{field_name} must be an object.")
    return value


def _required_sequence(
    value: Any,
    field_name: str,
    error_type: type[Exception],
) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value,
        Sequence,
    ):
        raise error_type(f"{field_name} must be an array.")
    return value


def _required_text(
    value: Any,
    field_name: str,
    error_type: type[Exception],
) -> str:
    if not isinstance(value, str):
        raise error_type(f"{field_name} must be a string.")
    cleaned = value.strip()
    if not cleaned:
        raise error_type(f"{field_name} cannot be empty.")
    return cleaned


def _optional_text(
    value: Any,
    field_name: str,
    error_type: type[Exception],
) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name, error_type)


__all__ = [
    "GATEMEM_EVALUATION_ONLY_CHECKPOINT_FIELDS",
    "GateMemMappingError",
    "InvalidGateMemEpisodeError",
    "InvalidGateMemCheckpointError",
    "GateMemReferenceError",
    "MappedPrincipal",
    "MappedRelationship",
    "MappedTurn",
    "MappedQuery",
    "MappedEpisode",
    "GateMemMapper",
]