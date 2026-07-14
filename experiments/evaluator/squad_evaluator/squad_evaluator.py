from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass


@dataclass
class SquadEvaluationResult:
    """
    Evaluation result for one prediction.
    """

    prediction: str
    gold_answers: list[str]
    exact_match: float
    f1: float


class SquadEvaluator:
    """
    SQuAD-style evaluator.

    Metrics:
    - Exact Match, EM
    - token-level F1

    This evaluator follows the common SQuAD normalization style:
    - lowercase
    - remove punctuation
    - remove articles
    - normalize whitespace
    """

    @staticmethod
    def normalize_answer(text: str) -> str:
        """
        Normalize answer text for SQuAD-style EM/F1.
        """

        if text is None:
            return ""

        text = str(text)

        def lower(s: str) -> str:
            return s.lower()

        def remove_punctuation(s: str) -> str:
            exclude = set(string.punctuation)
            return "".join(ch for ch in s if ch not in exclude)

        def remove_articles(s: str) -> str:
            return re.sub(r"\b(a|an|the)\b", " ", s)

        def white_space_fix(s: str) -> str:
            return " ".join(s.split())

        return white_space_fix(
            remove_articles(
                remove_punctuation(
                    lower(text)
                )
            )
        )

    @classmethod
    def exact_match_score(
        cls,
        prediction: str,
        gold_answer: str,
    ) -> float:
        """
        Return 1.0 if normalized prediction exactly matches normalized gold answer.
        """

        normalized_prediction = cls.normalize_answer(prediction)
        normalized_gold = cls.normalize_answer(gold_answer)

        return float(normalized_prediction == normalized_gold)

    @classmethod
    def f1_score(
        cls,
        prediction: str,
        gold_answer: str,
    ) -> float:
        """
        Compute token-level F1 between prediction and one gold answer.
        """

        prediction_tokens = cls.normalize_answer(prediction).split()
        gold_tokens = cls.normalize_answer(gold_answer).split()

        if len(prediction_tokens) == 0 and len(gold_tokens) == 0:
            return 1.0

        if len(prediction_tokens) == 0 or len(gold_tokens) == 0:
            return 0.0

        common = Counter(prediction_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = num_same / len(prediction_tokens)
        recall = num_same / len(gold_tokens)

        return 2 * precision * recall / (precision + recall)

    @classmethod
    def metric_max_over_ground_truths(
        cls,
        metric_fn,
        prediction: str,
        gold_answers: list[str],
    ) -> float:
        """
        For SQuAD, each question may have multiple gold answers.
        The score is the maximum score over all gold answers.
        """

        if not gold_answers:
            gold_answers = [""]

        return max(
            metric_fn(prediction, gold_answer)
            for gold_answer in gold_answers
        )

    @classmethod
    def evaluate_prediction(
        cls,
        prediction: str,
        gold_answers: list[str],
    ) -> SquadEvaluationResult:
        """
        Evaluate one prediction against all gold answers.
        """

        prediction = prediction or ""
        gold_answers = gold_answers or [""]

        exact_match = cls.metric_max_over_ground_truths(
            cls.exact_match_score,
            prediction,
            gold_answers,
        )

        f1 = cls.metric_max_over_ground_truths(
            cls.f1_score,
            prediction,
            gold_answers,
        )

        return SquadEvaluationResult(
            prediction=prediction,
            gold_answers=gold_answers,
            exact_match=exact_match,
            f1=f1,
        )

    @classmethod
    def evaluate_many(
        cls,
        predictions: list[str],
        gold_answers_list: list[list[str]],
    ) -> dict:
        """
        Evaluate multiple predictions and return average EM/F1.
        """

        if len(predictions) != len(gold_answers_list):
            raise ValueError(
                "predictions and gold_answers_list must have the same length."
            )

        if not predictions:
            return {
                "count": 0,
                "exact_match": 0.0,
                "f1": 0.0,
            }

        results = [
            cls.evaluate_prediction(prediction, gold_answers)
            for prediction, gold_answers in zip(predictions, gold_answers_list)
        ]

        average_em = sum(result.exact_match for result in results) / len(results)
        average_f1 = sum(result.f1 for result in results) / len(results)

        return {
            "count": len(results),
            "exact_match": average_em,
            "f1": average_f1,
        }