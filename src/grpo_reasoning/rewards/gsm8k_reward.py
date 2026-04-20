"""Reward functions for GSM8K-style math reasoning."""

from __future__ import annotations

from grpo_reasoning.evaluation.answer_extraction import (
    extract_predicted_answer,
    normalize_numeric_answer,
)


def correctness_reward(completion: str, gold_answer: str | int | float | None) -> float:
    """Reward exact numeric correctness."""
    prediction = extract_predicted_answer(completion)
    gold = normalize_numeric_answer(gold_answer)
    if prediction is None or gold is None:
        return 0.0
    return 1.0 if prediction == gold else 0.0


def format_reward(completion: str) -> float:
    """Small reward for using the requested final-answer format."""
    return 0.1 if "final answer" in completion.lower() else 0.0

