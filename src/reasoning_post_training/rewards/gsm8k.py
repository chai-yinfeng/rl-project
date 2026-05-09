"""Reward functions for GSM8K-style math reasoning."""

from __future__ import annotations

import re

from reasoning_post_training.evaluation.answer_extraction import (
    extract_predicted_answer,
    normalize_numeric_answer,
    truncate_completion,
)

_FINAL_ANSWER_LINE_RE = re.compile(
    r"(?im)^\s*final answer\s*[:=]\s*[-+]?(?:\d[\d,]*)(?:\.\d+)?\s*$"
)
_CHAT_LEAK_MARKERS = ("\nHuman:", "\nAssistant:", "\nUser:", "\nSystem:", "<|im_start|>")


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


def grpo_shaped_reward(completion: str, gold_answer: str | int | float | None) -> float:
    """Dense but conservative reward for GRPO math training."""
    truncated = truncate_completion(completion)
    prediction = extract_predicted_answer(truncated)
    reward = correctness_reward(truncated, gold_answer)

    if _FINAL_ANSWER_LINE_RE.search(truncated):
        reward += 0.2
    elif "final answer" in truncated.lower():
        reward += 0.1

    if prediction is None:
        reward -= 0.2
    else:
        reward += 0.05

    if any(marker in completion for marker in _CHAT_LEAK_MARKERS):
        reward -= 0.1

    completion_chars = len(truncated)
    if completion_chars < 40:
        reward -= 0.05
    elif completion_chars > 1600:
        reward -= 0.05

    return reward
