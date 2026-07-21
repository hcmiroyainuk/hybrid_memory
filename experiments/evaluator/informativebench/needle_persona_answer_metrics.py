from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection, Sequence
from dataclasses import dataclass


_STRONG_SEPARATOR_PATTERN = re.compile(r"\s*[,，;；、|\n]+\s*")
_CONJUNCTION_PATTERN = re.compile(r"\s+(?:and|&)\s+", flags=re.IGNORECASE)

_ANSWER_PREFIX_PATTERNS = (
    re.compile(r"^\s*answer\s*:\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*final\s+answer\s*:\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*prediction\s*:\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*the\s+answer\s+is\s+", flags=re.IGNORECASE),
)

_SURROUNDING_PUNCTUATION = (
    " \t\r\n"
    "\"'“”‘’`"
    "[](){}<>"
    ".,;:!?，。；：！？"
)

_BULLET_PREFIX_PATTERN = re.compile(
    r"(?m)^\s*(?:[-*•]|\d+[.)])\s+"
)

_LEADING_CONJUNCTION_PATTERN = re.compile(
    r"^\s*(?:and|&)\s+",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class AnswerMetricResult:
    """
    Deterministic answer-level and entity-level metrics for one prediction.
    """

    predicted_answer: str
    matched_reference_answer: str | None

    predicted_entities: tuple[str, ...]
    matched_reference_entities: tuple[str, ...]
    matched_entities: tuple[str, ...]

    answer_accuracy: float
    entity_precision: float
    entity_recall: float
    entity_f1: float

    def as_dict(self) -> dict[str, object]:
        """
        Return a JSON-serialisable representation.
        """
        return {
            "predicted_answer": self.predicted_answer,
            "matched_reference_answer": self.matched_reference_answer,
            "predicted_entities": list(self.predicted_entities),
            "matched_reference_entities": list(
                self.matched_reference_entities
            ),
            "matched_entities": list(self.matched_entities),
            "answer_accuracy": self.answer_accuracy,
            "entity_precision": self.entity_precision,
            "entity_recall": self.entity_recall,
            "entity_f1": self.entity_f1,
        }


def normalize_answer_text(value: str | None) -> str:
    """
    Apply conservative normalisation before entity splitting.

    Separators are preserved because commas, semicolons, conjunctions and
    line breaks may separate answer entities.
    """
    if value is None:
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00A0", " ")
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = _BULLET_PREFIX_PATTERN.sub("", text)

    for pattern in _ANSWER_PREFIX_PATTERNS:
        updated = pattern.sub("", text, count=1)
        if updated != text:
            text = updated
            break

    lines = [
        re.sub(r"[ \t\f\v]+", " ", line).strip()
        for line in text.split("\n")
    ]
    text = "\n".join(lines).strip()

    wrapping_pairs = {
        '"': '"',
        "'": "'",
        "“": "”",
        "‘": "’",
        "[": "]",
        "(": ")",
        "{": "}",
    }

    if len(text) >= 2:
        closing = wrapping_pairs.get(text[0])
        if closing is not None and text[-1] == closing:
            text = text[1:-1].strip()

    return text


def normalize_entity(value: str | None) -> str:
    """
    Normalise one answer entity for deterministic comparison.

    Internal wording and punctuation are preserved. Therefore ``McQueen`` and
    ``Alexander McQueen`` remain different entities.
    """
    if value is None:
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u00A0", " ")
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = " ".join(text.split())
    text = _LEADING_CONJUNCTION_PATTERN.sub("", text)
    text = text.strip(_SURROUNDING_PUNCTUATION)
    text = " ".join(text.split())

    return text.casefold()


def split_answer_entities(
    answer: str | None,
    *,
    split_conjunctions: bool = True,
) -> list[str]:
    """
    Split an answer into normalised, deduplicated entities.

    Predictions should use comma-space separation. Conjunction splitting is
    retained for benchmark references and legacy outputs.
    """
    text = normalize_answer_text(answer)

    if not text:
        return []

    strong_parts = [
        part.strip()
        for part in _STRONG_SEPARATOR_PATTERN.split(text)
        if part.strip()
    ]

    if not strong_parts:
        return []

    raw_parts: list[str] = []

    if not split_conjunctions:
        raw_parts = strong_parts
    elif len(strong_parts) == 1:
        raw_parts = [
            part.strip()
            for part in _CONJUNCTION_PATTERN.split(strong_parts[0])
            if part.strip()
        ]
    else:
        raw_parts.extend(strong_parts[:-1])
        final_part = strong_parts[-1]

        if _CONJUNCTION_PATTERN.search(final_part):
            raw_parts.extend(
                part.strip()
                for part in _CONJUNCTION_PATTERN.split(final_part)
                if part.strip()
            )
        else:
            raw_parts.append(final_part)

    entities: list[str] = []

    for part in raw_parts:
        entity = normalize_entity(part)

        if entity and entity not in entities:
            entities.append(entity)

    return entities


def extract_entity_set(
    answer: str | None,
    *,
    split_conjunctions: bool = True,
) -> set[str]:
    """
    Return the normalised entity set for an answer.
    """
    return set(
        split_answer_entities(
            answer,
            split_conjunctions=split_conjunctions,
        )
    )


def calculate_answer_accuracy(
    predicted_entities: Collection[str],
    reference_entities: Collection[str],
) -> float:
    """
    Return 1.0 only when the entity sets match exactly.
    """
    return float(
        set(predicted_entities)
        == set(reference_entities)
    )


def calculate_entity_precision(
    predicted_entities: Collection[str],
    reference_entities: Collection[str],
) -> float:
    """
    Calculate entity-level precision.
    """
    predicted = set(predicted_entities)
    reference = set(reference_entities)

    if not predicted:
        return 0.0

    return len(predicted & reference) / len(predicted)


def calculate_entity_recall(
    predicted_entities: Collection[str],
    reference_entities: Collection[str],
) -> float:
    """
    Calculate entity-level recall.
    """
    predicted = set(predicted_entities)
    reference = set(reference_entities)

    if not reference:
        return 0.0

    return len(predicted & reference) / len(reference)


def calculate_entity_f1(
    precision: float,
    recall: float,
) -> float:
    """
    Calculate the harmonic mean of precision and recall.
    """
    if precision + recall == 0.0:
        return 0.0

    return 2.0 * precision * recall / (precision + recall)


def _build_result(
    *,
    predicted_answer: str,
    reference_answer: str,
    predicted_entities: Sequence[str],
    reference_entities: Sequence[str],
) -> AnswerMetricResult:
    predicted_set = set(predicted_entities)
    reference_set = set(reference_entities)

    precision = calculate_entity_precision(
        predicted_set,
        reference_set,
    )
    recall = calculate_entity_recall(
        predicted_set,
        reference_set,
    )
    f1 = calculate_entity_f1(
        precision,
        recall,
    )

    matched = tuple(
        entity
        for entity in reference_entities
        if entity in predicted_set
    )

    return AnswerMetricResult(
        predicted_answer=predicted_answer,
        matched_reference_answer=reference_answer,
        predicted_entities=tuple(predicted_entities),
        matched_reference_entities=tuple(reference_entities),
        matched_entities=matched,
        answer_accuracy=calculate_answer_accuracy(
            predicted_set,
            reference_set,
        ),
        entity_precision=precision,
        entity_recall=recall,
        entity_f1=f1,
    )


def calculate_entity_metrics(
    predicted_answer: str | None,
    reference_answer: str | None,
) -> AnswerMetricResult:
    """
    Calculate all answer metrics against one reference.
    """
    prediction_text = normalize_answer_text(
        predicted_answer
    )
    reference_text = normalize_answer_text(
        reference_answer
    )

    predicted_entities = split_answer_entities(
        prediction_text
    )
    reference_entities = split_answer_entities(
        reference_text
    )

    if not reference_entities:
        raise ValueError(
            "reference_answer must contain at least one "
            "parseable entity."
        )

    return _build_result(
        predicted_answer=prediction_text,
        reference_answer=reference_text,
        predicted_entities=predicted_entities,
        reference_entities=reference_entities,
    )


def find_best_reference_match(
    predicted_answer: str | None,
    accepted_answers: Sequence[str],
) -> AnswerMetricResult:
    """
    Compare a prediction with all accepted answers and return the best match.

    Ranking order:
    1. entity F1;
    2. answer accuracy;
    3. entity recall;
    4. entity precision;
    5. earliest accepted answer when all metrics tie.
    """
    if not accepted_answers:
        raise ValueError(
            "accepted_answers must contain at least one "
            "reference answer."
        )

    prediction_text = normalize_answer_text(
        predicted_answer
    )
    predicted_entities = split_answer_entities(
        prediction_text
    )

    best_result: AnswerMetricResult | None = None
    best_rank: tuple[float, float, float, float] | None = None
    invalid_reference_count = 0

    for reference_answer in accepted_answers:
        reference_text = normalize_answer_text(
            reference_answer
        )
        reference_entities = split_answer_entities(
            reference_text
        )

        if not reference_entities:
            invalid_reference_count += 1
            continue

        result = _build_result(
            predicted_answer=prediction_text,
            reference_answer=reference_text,
            predicted_entities=predicted_entities,
            reference_entities=reference_entities,
        )

        rank = (
            result.entity_f1,
            result.answer_accuracy,
            result.entity_recall,
            result.entity_precision,
        )

        if best_rank is None or rank > best_rank:
            best_result = result
            best_rank = rank

    if best_result is None:
        raise ValueError(
            "accepted_answers contains no parseable "
            f"reference answer ({invalid_reference_count} invalid)."
        )

    return best_result


def evaluate_answer(
    predicted_answer: str | None,
    accepted_answers: Sequence[str],
) -> AnswerMetricResult:
    """
    Public convenience wrapper used by NeedlePersonaEvaluator.
    """
    return find_best_reference_match(
        predicted_answer,
        accepted_answers,
    )


__all__ = [
    "AnswerMetricResult",
    "normalize_answer_text",
    "normalize_entity",
    "split_answer_entities",
    "extract_entity_set",
    "calculate_answer_accuracy",
    "calculate_entity_precision",
    "calculate_entity_recall",
    "calculate_entity_f1",
    "calculate_entity_metrics",
    "find_best_reference_match",
    "evaluate_answer",
]