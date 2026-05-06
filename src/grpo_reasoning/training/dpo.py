"""DPO data and training helpers for GSM8K."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from grpo_reasoning.evaluation.answer_extraction import extract_predicted_answer
from grpo_reasoning.rewards.gsm8k_reward import correctness_reward, format_reward


def score_gsm8k_completion(completion: str, gold_answer: str | int | float | None) -> float:
    """Score a completion with the same rule reward used by GRPO."""
    score = correctness_reward(completion, gold_answer)
    score += format_reward(completion)
    if extract_predicted_answer(completion) is None:
        score -= 0.1
    return score


def choose_preference_pair(
    completions: list[str],
    gold_answer: str | int | float | None,
    *,
    require_distinct_scores: bool = True,
) -> dict[str, Any] | None:
    """Choose the best/worst completion as a DPO pair for one prompt."""
    if len(completions) < 2:
        return None

    scored = [
        {
            "completion": completion,
            "score": score_gsm8k_completion(completion, gold_answer),
            "predicted_answer": extract_predicted_answer(completion),
        }
        for completion in completions
    ]
    scored.sort(key=lambda item: item["score"], reverse=True)
    chosen = scored[0]
    rejected = scored[-1]

    if require_distinct_scores and chosen["score"] <= rejected["score"]:
        return None

    return {
        "chosen": chosen["completion"],
        "rejected": rejected["completion"],
        "chosen_score": chosen["score"],
        "rejected_score": rejected["score"],
        "chosen_predicted_answer": chosen["predicted_answer"],
        "rejected_predicted_answer": rejected["predicted_answer"],
    }


def load_dpo_jsonl(path: Path):
    """Load prompt/chosen/rejected JSONL data as a datasets.Dataset."""
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets before loading DPO data.") from exc

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = {"prompt", "chosen", "rejected"} - set(record)
            if missing:
                raise ValueError(f"Line {line_number} is missing DPO keys: {sorted(missing)}")
            records.append(
                {
                    "prompt": str(record["prompt"]),
                    "chosen": str(record["chosen"]),
                    "rejected": str(record["rejected"]),
                }
            )

    return Dataset.from_list(records)


def build_dpo_config(config: dict[str, Any]):
    """Create a TRL DPOConfig from a JSON-compatible dictionary."""
    try:
        from trl import DPOConfig
    except ImportError as exc:
        raise RuntimeError("Install trl before running DPO training.") from exc

    allowed_keys = {
        "output_dir",
        "seed",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "num_train_epochs",
        "max_steps",
        "learning_rate",
        "lr_scheduler_type",
        "warmup_steps",
        "weight_decay",
        "max_grad_norm",
        "optim",
        "fp16",
        "bf16",
        "gradient_checkpointing",
        "save_steps",
        "save_strategy",
        "logging_steps",
        "report_to",
        "remove_unused_columns",
        "model_init_kwargs",
        "max_length",
        "beta",
        "precompute_ref_log_probs",
        "precompute_ref_batch_size",
    }
    kwargs = {key: value for key, value in config.items() if key in allowed_keys}
    return DPOConfig(**kwargs)
