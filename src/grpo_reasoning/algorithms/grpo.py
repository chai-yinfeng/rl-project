"""Small GRPO utilities shared by training code and tests."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class GRPOConfig:
    group_size: int = 4
    clip_range: float = 0.2
    kl_beta: float = 0.04
    advantage_epsilon: float = 1e-8


def compute_group_relative_advantages(
    rewards: list[float],
    group_size: int,
    epsilon: float = 1e-8,
) -> list[float]:
    """Normalize rewards within each sampled group."""
    if group_size < 1:
        raise ValueError("group_size must be at least 1")
    if len(rewards) % group_size != 0:
        raise ValueError("number of rewards must be divisible by group_size")

    advantages: list[float] = []
    for start in range(0, len(rewards), group_size):
        group = rewards[start : start + group_size]
        mean = sum(group) / group_size
        variance = sum((reward - mean) ** 2 for reward in group) / group_size
        std = sqrt(variance)
        if std <= epsilon:
            advantages.extend([0.0] * group_size)
        else:
            advantages.extend((reward - mean) / (std + epsilon) for reward in group)
    return advantages

