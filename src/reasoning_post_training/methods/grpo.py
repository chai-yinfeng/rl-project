"""GRPO training helpers for GSM8K."""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from math import sqrt
from typing import Any

from reasoning_post_training.rewards.gsm8k import grpo_shaped_reward


@dataclass(frozen=True)
class SimpleGRPOConfig:
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


def build_gsm8k_grpo_dataset(split: str, subset: str, max_examples: int | None = None):
    """Load GSM8K and keep the columns expected by TRL's GRPOTrainer."""
    from reasoning_post_training.datasets.gsm8k import load_gsm8k_split

    dataset = load_gsm8k_split(split=split, subset=subset)
    keep_columns = {"prompt", "gold_answer", "question", "answer"}
    drop_columns = [column for column in dataset.column_names if column not in keep_columns]
    if drop_columns:
        dataset = dataset.remove_columns(drop_columns)
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    return dataset


def gsm8k_grpo_reward_func(
    prompts: list[str],
    completions: list[str],
    gold_answer: list[str],
    **_: Any,
) -> list[float]:
    """Reward function signature compatible with TRL GRPOTrainer."""
    rewards: list[float] = []
    for completion, gold in zip(completions, gold_answer, strict=True):
        rewards.append(grpo_shaped_reward(completion, gold))
    return rewards


def build_grpo_config(config: dict[str, Any]):
    """Create a TRL GRPOConfig from a JSON-compatible dictionary."""
    try:
        from trl import GRPOConfig
    except ImportError as exc:
        raise RuntimeError("Install trl before running GRPO training.") from exc

    allowed_keys = set(inspect.signature(GRPOConfig.__init__).parameters)
    kwargs = {key: value for key, value in config.items() if key in allowed_keys}
    if "generation_batch_size" in allowed_keys and "num_generations" in kwargs:
        num_generations = int(kwargs["num_generations"])
        generation_batch_size = kwargs.get("generation_batch_size")
        if generation_batch_size is None:
            per_device_batch_size = int(kwargs.get("per_device_train_batch_size", 1))
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            global_batch_size = per_device_batch_size * max(world_size, 1)
            generation_batch_size = max(num_generations, global_batch_size)
            remainder = generation_batch_size % num_generations
            if remainder:
                generation_batch_size += num_generations - remainder
            kwargs["generation_batch_size"] = generation_batch_size
        elif int(generation_batch_size) % num_generations != 0:
            raise ValueError(
                "generation_batch_size must be divisible by num_generations "
                f"(got {generation_batch_size} and {num_generations})."
            )
    return GRPOConfig(**kwargs)
