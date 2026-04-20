"""GRPO training helpers for GSM8K."""

from __future__ import annotations

from typing import Any

from grpo_reasoning.evaluation.answer_extraction import extract_predicted_answer
from grpo_reasoning.rewards.gsm8k_reward import correctness_reward, format_reward


def build_gsm8k_grpo_dataset(split: str, subset: str, max_examples: int | None = None):
    """Load GSM8K and keep the columns expected by TRL's GRPOTrainer."""
    from grpo_reasoning.data.gsm8k import load_gsm8k_split

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
        reward = correctness_reward(completion, gold)
        reward += format_reward(completion)
        if extract_predicted_answer(completion) is None:
            reward -= 0.1
        rewards.append(reward)
    return rewards


def build_grpo_config(config: dict[str, Any]):
    """Create a TRL GRPOConfig from a JSON-compatible dictionary."""
    try:
        from trl import GRPOConfig
    except ImportError as exc:
        raise RuntimeError("Install trl before running GRPO training.") from exc

    allowed_keys = {
        "output_dir",
        "seed",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "learning_rate",
        "num_generations",
        "max_prompt_length",
        "max_completion_length",
        "temperature",
        "top_p",
        "beta",
        "epsilon",
        "optim",
        "fp16",
        "bf16",
        "gradient_checkpointing",
        "save_steps",
        "logging_steps",
        "report_to",
        "model_init_kwargs",
    }
    kwargs = {key: value for key, value in config.items() if key in allowed_keys}
    return GRPOConfig(**kwargs)
