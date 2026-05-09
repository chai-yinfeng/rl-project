"""DPO data and training helpers for GSM8K."""

from __future__ import annotations

import json
import inspect
from pathlib import Path
from typing import Any

from reasoning_post_training.evaluation.answer_extraction import extract_predicted_answer
from reasoning_post_training.rewards.gsm8k import correctness_reward, grpo_shaped_reward


def score_gsm8k_completion(completion: str, gold_answer: str | int | float | None) -> float:
    """Score a completion with the same shaped rule reward used by GRPO."""
    return grpo_shaped_reward(completion, gold_answer)


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


def build_gold_chosen_pair(
    rejected_completion: str,
    answer: str,
    gold_answer: str | int | float | None,
) -> dict[str, Any]:
    """Build a reliable DPO pair using the dataset solution as chosen."""
    chosen = f"{str(answer).strip()}\nFinal answer: {gold_answer}"
    return {
        "chosen": chosen,
        "rejected": rejected_completion,
        "chosen_score": score_gsm8k_completion(chosen, gold_answer),
        "rejected_score": score_gsm8k_completion(rejected_completion, gold_answer),
        "chosen_predicted_answer": extract_predicted_answer(chosen),
        "rejected_predicted_answer": extract_predicted_answer(rejected_completion),
    }


def build_gold_chosen_pair_from_completions(
    completions: list[str],
    answer: str,
    gold_answer: str | int | float | None,
) -> dict[str, Any] | None:
    """Build a gold-chosen pair only when an actually incorrect rejection exists."""
    scored = [
        {
            "completion": completion,
            "score": score_gsm8k_completion(completion, gold_answer),
            "predicted_answer": extract_predicted_answer(completion),
            "correctness": correctness_reward(completion, gold_answer),
        }
        for completion in completions
    ]
    incorrect = [item for item in scored if item["correctness"] <= 0.0]
    if not incorrect:
        return None

    rejected = min(incorrect, key=lambda item: item["score"])
    return build_gold_chosen_pair(rejected["completion"], answer, gold_answer)


def normalize_dpo_completion(completion: str) -> str:
    """Keep prompt/response tokenization stable at the concatenation boundary."""
    completion = completion.strip()
    return completion if completion.startswith(" ") else f" {completion}"


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
                    "chosen": normalize_dpo_completion(str(record["chosen"])),
                    "rejected": normalize_dpo_completion(str(record["rejected"])),
                }
            )

    return Dataset.from_list(records)


def build_dpo_config(config: dict[str, Any]):
    """Create a TRL DPOConfig from a JSON-compatible dictionary."""
    try:
        from trl import DPOConfig
    except ImportError as exc:
        raise RuntimeError("Install trl before running DPO training.") from exc

    allowed_keys = set(inspect.signature(DPOConfig.__init__).parameters)
    kwargs = {key: value for key, value in config.items() if key in allowed_keys}
    return DPOConfig(**kwargs)
