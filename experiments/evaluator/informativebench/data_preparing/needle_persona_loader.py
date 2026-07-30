from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from experiments.evaluator.informativebench.data_preparing.needle_persona_models import (
    PERSONA_NAMES,
    NeedlePersonaSample,
    PersonaName,
    PersonaSource,
)


SourceGranularity = Literal["persona_document", "dialogue_turn"]


class NeedlePersonaDatasetError(Exception):
    """
    Base exception for Needle in the Persona dataset loading errors.
    """


class NeedlePersonaRecordError(NeedlePersonaDatasetError):
    """
    Raised when one JSONL record cannot be converted into a valid sample.
    """

    def __init__(
        self,
        message: str,
        *,
        line_number: int | None = None,
        sample_id: str | None = None,
    ) -> None:
        location_parts: list[str] = []

        if line_number is not None:
            location_parts.append(f"line={line_number}")

        if sample_id is not None:
            location_parts.append(f"sample_id={sample_id!r}")

        location = (
            f" ({', '.join(location_parts)})"
            if location_parts
            else ""
        )

        super().__init__(f"{message}{location}")
        self.line_number = line_number
        self.sample_id = sample_id


@dataclass(frozen=True)
class DialogueTurn:
    """
    One parsed dialogue turn before conversion to PersonaSource.
    """

    speaker: PersonaName
    content: str
    turn_index: int


@dataclass(frozen=True)
class ParsedConversation:
    """
    Parsed dialogue plus non-speaker narration/stage directions.
    """

    turns: tuple[DialogueTurn, ...]
    narration: tuple[str, ...]


class NeedlePersonaLoader:
    """
    Loader for the generated InformativeBench Needle in the Persona 2-hop JSONL.

    Expected record fields:
    - id
    - modified_alice_bob_conversation
    - modified_charlie_dave_conversation
    - chat_bob_charlie
    - task_prompt
    - answer
    - needle_detail

    Safe defaults:
    - chat_bob_charlie is excluded because it already contains the generated
      Bob-Charlie collaboration that the governed workflow is intended to
      replace;
    - needle_detail is excluded because it explicitly describes the injected
      needle and may reveal the answer;
    - only each persona's own utterances are assigned to that persona agent;
    - persona_document granularity produces one source per persona per source
      conversation, reducing LLM ingestion calls compared with turn-level
      sources.

    The loader converts raw dataset records into NeedlePersonaSample objects.
    Workflow and ingestion code should depend on NeedlePersonaSample rather
    than on the raw JSON keys.
    """

    _REQUIRED_FIELDS = (
        "id",
        "modified_alice_bob_conversation",
        "modified_charlie_dave_conversation",
        "task_prompt",
        "answer",
    )

    _PLAIN_TURN_PATTERN = re.compile(
        r"^\s*(alice|bob|charlie|dave)\s*:\s*(.*?)\s*$",
        flags=re.IGNORECASE,
    )

    _MARKDOWN_TURN_PATTERN = re.compile(
        r"^\s*\*\*(alice|bob|charlie|dave)\s*:\*\*\s*(.*?)\s*$",
        flags=re.IGNORECASE,
    )

    _STAGE_DIRECTION_PATTERN = re.compile(
        r"""
        ^\s*
        (?:
            \[.*\]
            |
            \(.*\)
            |
            \*[^*].*\*
            |
            \*\s*\*\s*\*
        )
        \s*$
        """,
        flags=re.VERBOSE,
    )

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        source_granularity: SourceGranularity = "persona_document",
        include_collaborative_chat: bool = False,
        retain_raw_fields_in_metadata: bool = False,
        strict: bool = True,
        encoding: str = "utf-8",
    ) -> None:
        """
        Args:
            dataset_path:
                Path to dataset_2hop.jsonl.
            source_granularity:
                "persona_document" creates one grouped source for each persona
                in each conversation. "dialogue_turn" creates one source for
                every parsed utterance.
            include_collaborative_chat:
                Whether to ingest chat_bob_charlie into Bob and Charlie's
                sources. Keep False for the governed-memory workflow.
            retain_raw_fields_in_metadata:
                Whether to preserve raw conversations, chat_bob_charlie, and
                needle_detail in sample.metadata. Keep False during evaluation
                to prevent accidental context leakage.
            strict:
                When True, malformed records, unexpected speakers, empty
                conversations, and missing expected personas raise errors.
            encoding:
                File encoding.
        """
        self.dataset_path = Path(dataset_path)
        self.source_granularity = source_granularity
        self.include_collaborative_chat = include_collaborative_chat
        self.retain_raw_fields_in_metadata = retain_raw_fields_in_metadata
        self.strict = strict
        self.encoding = encoding

        if source_granularity not in {
            "persona_document",
            "dialogue_turn",
        }:
            raise ValueError(
                "source_granularity must be "
                "'persona_document' or 'dialogue_turn'."
            )

    # ------------------------------------------------------------------
    # Public loading API
    # ------------------------------------------------------------------

    def iter_samples(
        self,
        *,
        limit: int | None = None,
    ) -> Iterator[NeedlePersonaSample]:
        """
        Lazily read and convert JSONL records.

        Duplicate sample IDs are rejected so source IDs remain globally stable
        within one dataset file.
        """
        self._validate_dataset_path()

        if limit is not None and limit < 0:
            raise ValueError("limit cannot be negative.")

        seen_sample_ids: set[str] = set()
        yielded = 0

        with self.dataset_path.open(
            "r",
            encoding=self.encoding,
        ) as file:
            for line_number, raw_line in enumerate(file, start=1):
                if limit is not None and yielded >= limit:
                    break

                line = raw_line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise NeedlePersonaRecordError(
                        f"Invalid JSON: {error}",
                        line_number=line_number,
                    ) from error

                if not isinstance(record, Mapping):
                    raise NeedlePersonaRecordError(
                        "Each JSONL line must contain a JSON object.",
                        line_number=line_number,
                    )

                sample = self.parse_record(
                    record,
                    line_number=line_number,
                )

                if sample.sample_id in seen_sample_ids:
                    raise NeedlePersonaRecordError(
                        "Duplicate sample ID.",
                        line_number=line_number,
                        sample_id=sample.sample_id,
                    )

                seen_sample_ids.add(sample.sample_id)
                yielded += 1
                yield sample

    def load_all(
        self,
        *,
        limit: int | None = None,
    ) -> list[NeedlePersonaSample]:
        """
        Load all samples, or the first limit samples, into memory.
        """
        return list(self.iter_samples(limit=limit))

    def load_by_id(
        self,
        sample_id: str,
    ) -> NeedlePersonaSample:
        """
        Load one sample by its dataset ID.
        """
        target_id = str(sample_id).strip()

        if not target_id:
            raise ValueError("sample_id cannot be empty.")

        for sample in self.iter_samples():
            if sample.sample_id == target_id:
                return sample

        raise KeyError(
            f"Sample ID {target_id!r} was not found in "
            f"{self.dataset_path}."
        )

    def count_records(self) -> int:
        """
        Count non-empty JSONL records without parsing them.
        """
        self._validate_dataset_path()

        with self.dataset_path.open(
            "r",
            encoding=self.encoding,
        ) as file:
            return sum(1 for line in file if line.strip())

    # ------------------------------------------------------------------
    # Record conversion
    # ------------------------------------------------------------------

    def parse_record(
        self,
        record: Mapping[str, Any],
        *,
        line_number: int | None = None,
    ) -> NeedlePersonaSample:
        """
        Convert one raw JSON object into NeedlePersonaSample.
        """
        missing_fields = [
            field
            for field in self._REQUIRED_FIELDS
            if field not in record
        ]

        raw_sample_id = record.get("id")
        sample_id = (
            str(raw_sample_id).strip()
            if raw_sample_id is not None
            else None
        )

        if missing_fields:
            raise NeedlePersonaRecordError(
                "Missing required fields: "
                f"{', '.join(missing_fields)}.",
                line_number=line_number,
                sample_id=sample_id,
            )

        if not sample_id:
            raise NeedlePersonaRecordError(
                "Record ID cannot be empty.",
                line_number=line_number,
            )

        question = self._required_text(
            record.get("task_prompt"),
            field_name="task_prompt",
            line_number=line_number,
            sample_id=sample_id,
        )

        primary_answer, alternative_answers = self._parse_answers(
            record.get("answer"),
            line_number=line_number,
            sample_id=sample_id,
        )

        alice_bob = self.parse_conversation(
            record.get("modified_alice_bob_conversation"),
            expected_speakers={"alice", "bob"},
            field_name="modified_alice_bob_conversation",
            line_number=line_number,
            sample_id=sample_id,
        )

        charlie_dave = self.parse_conversation(
            record.get("modified_charlie_dave_conversation"),
            expected_speakers={"charlie", "dave"},
            field_name="modified_charlie_dave_conversation",
            line_number=line_number,
            sample_id=sample_id,
        )

        conversation_inputs: list[
            tuple[str, str, ParsedConversation]
        ] = [
            (
                f"{sample_id}_alice_bob",
                "modified_alice_bob_conversation",
                alice_bob,
            ),
            (
                f"{sample_id}_charlie_dave",
                "modified_charlie_dave_conversation",
                charlie_dave,
            ),
        ]

        if self.include_collaborative_chat:
            collaborative_chat = self.parse_conversation(
                record.get("chat_bob_charlie"),
                expected_speakers={"bob", "charlie"},
                field_name="chat_bob_charlie",
                line_number=line_number,
                sample_id=sample_id,
            )

            conversation_inputs.append(
                (
                    f"{sample_id}_bob_charlie",
                    "chat_bob_charlie",
                    collaborative_chat,
                )
            )

        persona_sources: dict[
            PersonaName,
            list[PersonaSource],
        ] = {
            persona: []
            for persona in PERSONA_NAMES
        }

        for (
            conversation_id,
            source_field,
            parsed_conversation,
        ) in conversation_inputs:
            created_sources = self._create_sources(
                sample_id=sample_id,
                conversation_id=conversation_id,
                source_field=source_field,
                parsed_conversation=parsed_conversation,
            )

            for persona, sources in created_sources.items():
                persona_sources[persona].extend(sources)

        if self.strict:
            missing_personas = [
                persona
                for persona in PERSONA_NAMES
                if not persona_sources[persona]
            ]

            if missing_personas:
                raise NeedlePersonaRecordError(
                    "No owned source content was produced for personas: "
                    f"{', '.join(missing_personas)}.",
                    line_number=line_number,
                    sample_id=sample_id,
                )

        metadata: dict[str, Any] = {
            "benchmark": "InformativeBench",
            "subset": "Needle_in_the_Persona",
            "dataset_file": self.dataset_path.name,
            "source_record_id": sample_id,
            "source_line_number": line_number,
            "source_granularity": self.source_granularity,
            "collaborative_chat_available": bool(
                self._optional_text(record.get("chat_bob_charlie"))
            ),
            "collaborative_chat_included": (
                self.include_collaborative_chat
            ),
            "raw_fields_retained": self.retain_raw_fields_in_metadata,
        }

        if self.retain_raw_fields_in_metadata:
            metadata["raw_fields"] = {
                "modified_alice_bob_conversation": record.get(
                    "modified_alice_bob_conversation"
                ),
                "modified_charlie_dave_conversation": record.get(
                    "modified_charlie_dave_conversation"
                ),
                "chat_bob_charlie": record.get("chat_bob_charlie"),
                "needle_detail": record.get("needle_detail"),
            }

        try:
            return NeedlePersonaSample(
                sample_id=sample_id,
                question=question,
                gold_answer=primary_answer,
                alternative_answers=alternative_answers,
                hop=2,
                persona_sources=persona_sources,
                metadata=metadata,
            )
        except Exception as error:
            raise NeedlePersonaRecordError(
                f"Failed to validate converted sample: {error}",
                line_number=line_number,
                sample_id=sample_id,
            ) from error

    # ------------------------------------------------------------------
    # Conversation parsing
    # ------------------------------------------------------------------

    def parse_conversation(
        self,
        raw_conversation: Any,
        *,
        expected_speakers: set[PersonaName],
        field_name: str,
        line_number: int | None = None,
        sample_id: str | None = None,
    ) -> ParsedConversation:
        """
        Parse plain ``alice: ...`` dialogue and Markdown ``**Bob:** ...``
        dialogue into speaker-owned turns.

        Non-speaker stage directions are retained separately as narration and
        are not assigned to a persona agent.
        """
        text = self._required_text(
            raw_conversation,
            field_name=field_name,
            line_number=line_number,
            sample_id=sample_id,
        )

        mutable_turns: list[dict[str, Any]] = []
        narration: list[str] = []

        for raw_line in text.splitlines():
            line = raw_line.strip()

            if not line:
                continue

            match = (
                self._MARKDOWN_TURN_PATTERN.match(line)
                or self._PLAIN_TURN_PATTERN.match(line)
            )

            if match:
                speaker = cast(
                    PersonaName,
                    match.group(1).strip().lower(),
                )
                content = match.group(2).strip()

                if speaker not in expected_speakers:
                    if self.strict:
                        raise NeedlePersonaRecordError(
                            f"Unexpected speaker {speaker!r} in "
                            f"{field_name}; expected "
                            f"{sorted(expected_speakers)}.",
                            line_number=line_number,
                            sample_id=sample_id,
                        )

                    narration.append(line)
                    continue

                if not content:
                    if self.strict:
                        raise NeedlePersonaRecordError(
                            f"Empty dialogue turn for {speaker!r} in "
                            f"{field_name}.",
                            line_number=line_number,
                            sample_id=sample_id,
                        )

                    continue

                mutable_turns.append(
                    {
                        "speaker": speaker,
                        "content": content,
                    }
                )
                continue

            if self._looks_like_stage_direction(line):
                narration.append(line)
                continue

            # Support wrapped/multiline content by attaching unlabelled text
            # to the preceding speaker turn.
            if mutable_turns:
                previous_content = mutable_turns[-1]["content"]
                mutable_turns[-1]["content"] = (
                    f"{previous_content}\n{line}"
                ).strip()
            elif self.strict:
                raise NeedlePersonaRecordError(
                    f"Could not parse dialogue line in {field_name}: "
                    f"{line!r}.",
                    line_number=line_number,
                    sample_id=sample_id,
                )
            else:
                narration.append(line)

        turns = tuple(
            DialogueTurn(
                speaker=cast(PersonaName, item["speaker"]),
                content=str(item["content"]).strip(),
                turn_index=index,
            )
            for index, item in enumerate(mutable_turns)
            if str(item["content"]).strip()
        )

        if not turns:
            raise NeedlePersonaRecordError(
                f"No dialogue turns were parsed from {field_name}.",
                line_number=line_number,
                sample_id=sample_id,
            )

        if self.strict:
            parsed_speakers = {
                turn.speaker
                for turn in turns
            }
            missing_speakers = expected_speakers - parsed_speakers

            if missing_speakers:
                raise NeedlePersonaRecordError(
                    f"Missing expected speakers in {field_name}: "
                    f"{sorted(missing_speakers)}.",
                    line_number=line_number,
                    sample_id=sample_id,
                )

        return ParsedConversation(
            turns=turns,
            narration=tuple(narration),
        )

    # ------------------------------------------------------------------
    # Source creation
    # ------------------------------------------------------------------

    def _create_sources(
        self,
        *,
        sample_id: str,
        conversation_id: str,
        source_field: str,
        parsed_conversation: ParsedConversation,
    ) -> dict[PersonaName, list[PersonaSource]]:
        if self.source_granularity == "dialogue_turn":
            return self._create_turn_sources(
                sample_id=sample_id,
                conversation_id=conversation_id,
                source_field=source_field,
                parsed_conversation=parsed_conversation,
            )

        return self._create_persona_document_sources(
            sample_id=sample_id,
            conversation_id=conversation_id,
            source_field=source_field,
            parsed_conversation=parsed_conversation,
        )

    def _create_turn_sources(
        self,
        *,
        sample_id: str,
        conversation_id: str,
        source_field: str,
        parsed_conversation: ParsedConversation,
    ) -> dict[PersonaName, list[PersonaSource]]:
        result: dict[PersonaName, list[PersonaSource]] = {
            persona: []
            for persona in PERSONA_NAMES
        }

        safe_sample_id = self._safe_identifier(sample_id)

        for turn in parsed_conversation.turns:
            source_id = (
                f"{safe_sample_id}__"
                f"{source_field}__"
                f"{turn.speaker}__"
                f"turn_{turn.turn_index:03d}"
            )

            result[turn.speaker].append(
                PersonaSource(
                    source_id=source_id,
                    persona=turn.speaker,
                    content=turn.content,
                    source_type="dialogue_turn",
                    conversation_id=conversation_id,
                    turn_index=turn.turn_index,
                    speaker=turn.speaker,
                    metadata={
                        "source_field": source_field,
                        "granularity": "dialogue_turn",
                    },
                )
            )

        return result

    def _create_persona_document_sources(
        self,
        *,
        sample_id: str,
        conversation_id: str,
        source_field: str,
        parsed_conversation: ParsedConversation,
    ) -> dict[PersonaName, list[PersonaSource]]:
        result: dict[PersonaName, list[PersonaSource]] = {
            persona: []
            for persona in PERSONA_NAMES
        }

        grouped_turns: dict[PersonaName, list[DialogueTurn]] = {
            persona: []
            for persona in PERSONA_NAMES
        }

        for turn in parsed_conversation.turns:
            grouped_turns[turn.speaker].append(turn)

        safe_sample_id = self._safe_identifier(sample_id)

        for persona, turns in grouped_turns.items():
            if not turns:
                continue

            # Preserve order and speaker labels while exposing only this
            # persona's own utterances.
            content = "\n".join(
                f"{persona}: {turn.content}"
                for turn in turns
            )

            source_id = (
                f"{safe_sample_id}__"
                f"{source_field}__"
                f"{persona}__document"
            )

            result[persona].append(
                PersonaSource(
                    source_id=source_id,
                    persona=persona,
                    content=content,
                    source_type="persona",
                    conversation_id=conversation_id,
                    speaker=persona,
                    metadata={
                        "source_field": source_field,
                        "granularity": "persona_document",
                        "turn_indices": [
                            turn.turn_index
                            for turn in turns
                        ],
                        "turn_count": len(turns),
                        "unassigned_narration_count": len(
                            parsed_conversation.narration
                        ),
                    },
                )
            )

        return result

    # ------------------------------------------------------------------
    # Field and file helpers
    # ------------------------------------------------------------------

    def _validate_dataset_path(self) -> None:
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset file does not exist: "
                f"{self.dataset_path}"
            )

        if not self.dataset_path.is_file():
            raise NeedlePersonaDatasetError(
                f"Dataset path is not a file: "
                f"{self.dataset_path}"
            )

    @staticmethod
    def _required_text(
        value: Any,
        *,
        field_name: str,
        line_number: int | None,
        sample_id: str | None,
    ) -> str:
        if value is None:
            raise NeedlePersonaRecordError(
                f"{field_name} cannot be null.",
                line_number=line_number,
                sample_id=sample_id,
            )

        text = str(value).strip()

        if not text:
            raise NeedlePersonaRecordError(
                f"{field_name} cannot be empty.",
                line_number=line_number,
                sample_id=sample_id,
            )

        return text

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        if value is None:
            return None

        text = str(value).strip()
        return text or None

    @classmethod
    def _parse_answers(
        cls,
        value: Any,
        *,
        line_number: int | None,
        sample_id: str,
    ) -> tuple[str, list[str]]:
        """
        Support the generated string answer field and tolerate a future list
        format without changing downstream models.
        """
        if isinstance(value, list):
            answers = [
                str(item).strip()
                for item in value
                if str(item).strip()
            ]

            if not answers:
                raise NeedlePersonaRecordError(
                    "answer list cannot be empty.",
                    line_number=line_number,
                    sample_id=sample_id,
                )

            return answers[0], cls._deduplicate(answers[1:])

        answer = cls._required_text(
            value,
            field_name="answer",
            line_number=line_number,
            sample_id=sample_id,
        )

        return answer, []

    @classmethod
    def _looks_like_stage_direction(
        cls,
        line: str,
    ) -> bool:
        return bool(cls._STAGE_DIRECTION_PATTERN.match(line))

    @staticmethod
    def _safe_identifier(value: str) -> str:
        cleaned = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            value.strip(),
        ).strip("_")

        return cleaned or "sample"

    @staticmethod
    def _deduplicate(values: list[str]) -> list[str]:
        cleaned: list[str] = []

        for value in values:
            item = str(value).strip()

            if item and item not in cleaned:
                cleaned.append(item)

        return cleaned


def load_needle_persona_dataset(
    dataset_path: str | Path,
    *,
    source_granularity: SourceGranularity = "persona_document",
    include_collaborative_chat: bool = False,
    retain_raw_fields_in_metadata: bool = False,
    strict: bool = True,
    limit: int | None = None,
) -> list[NeedlePersonaSample]:
    """
    Functional convenience wrapper around NeedlePersonaLoader.
    """
    loader = NeedlePersonaLoader(
        dataset_path,
        source_granularity=source_granularity,
        include_collaborative_chat=include_collaborative_chat,
        retain_raw_fields_in_metadata=retain_raw_fields_in_metadata,
        strict=strict,
    )

    return loader.load_all(limit=limit)


__all__ = [
    "SourceGranularity",
    "DialogueTurn",
    "ParsedConversation",
    "NeedlePersonaDatasetError",
    "NeedlePersonaRecordError",
    "NeedlePersonaLoader",
    "load_needle_persona_dataset",
]
