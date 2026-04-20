"""Evaluation metrics for reasoning completions."""

from __future__ import annotations

from dataclasses import dataclass

from grpo_reasoning.evaluation.answer_extraction import (
    extract_predicted_answer,
    normalize_numeric_answer,
)


@dataclass(frozen=True)
class EvaluationResult:
    total: int
    correct: int
    invalid: int
    accuracy: float
    invalid_rate: float


def evaluate_completions(completions: list[str], gold_answers: list[str]) -> EvaluationResult:
    """Compute exact-match accuracy and invalid-answer rate."""
    if len(completions) != len(gold_answers):
        raise ValueError("completions and gold_answers must have the same length")

    total = len(completions)
    correct = 0
    invalid = 0

    for completion, gold_answer in zip(completions, gold_answers, strict=True):
        prediction = extract_predicted_answer(completion)
        gold = normalize_numeric_answer(gold_answer)
        if prediction is None:
            invalid += 1
            continue
        if prediction == gold:
            correct += 1

    accuracy = correct / total if total else 0.0
    invalid_rate = invalid / total if total else 0.0
    return EvaluationResult(
        total=total,
        correct=correct,
        invalid=invalid,
        accuracy=accuracy,
        invalid_rate=invalid_rate,
    )

