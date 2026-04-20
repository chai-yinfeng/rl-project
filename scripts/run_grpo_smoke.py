#!/usr/bin/env python3
"""Run a minimal GRPO smoke training job on GSM8K."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from grpo_reasoning.training.grpo import (
    build_grpo_config,
    build_gsm8k_grpo_dataset,
    gsm8k_grpo_reward_func,
)
from grpo_reasoning.training.runtime import set_seed


DEFAULT_CONFIG = Path("configs/train/grpo_smoke_3070.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.model is not None:
        config["model_name_or_path"] = args.model
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.max_train_examples is not None:
        config["max_train_examples"] = args.max_train_examples
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    set_seed(int(config.get("seed", 42)))
    train_dataset = build_gsm8k_grpo_dataset(
        split=config.get("split", "train"),
        subset=config.get("subset", "main"),
        max_examples=config.get("max_train_examples"),
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "model_name_or_path": config["model_name_or_path"],
                    "train_examples": len(train_dataset),
                    "output_dir": config["output_dir"],
                    "max_steps": config["max_steps"],
                    "num_generations": config["num_generations"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    try:
        from transformers import AutoTokenizer
        from trl import GRPOTrainer
    except ImportError as exc:
        raise RuntimeError("Install transformers and trl before running GRPO training.") from exc

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    grpo_config = build_grpo_config(config)
    trainer = GRPOTrainer(
        model=config["model_name_or_path"],
        reward_funcs=gsm8k_grpo_reward_func,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(config["output_dir"])


if __name__ == "__main__":
    main()

