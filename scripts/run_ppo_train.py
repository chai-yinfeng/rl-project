#!/usr/bin/env python3
"""Run TRL PPO rule-reward training on GSM8K."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reasoning_post_training.methods.ppo import build_ppo_dataset, train_ppo_style, train_ppo_trl
from reasoning_post_training.runtime import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
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
    dataset = build_ppo_dataset(
        split=config.get("split", "train"),
        subset=config.get("subset", "main"),
        max_examples=config.get("max_train_examples"),
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "method": "ppo",
                    "trainer": config.get("trainer", "trl"),
                    "model_name_or_path": config["model_name_or_path"],
                    "train_examples": len(dataset),
                    "output_dir": config["output_dir"],
                    "max_steps": config["max_steps"],
                    "kl_coef": config.get("kl_coef", config.get("kl_beta", 0.0)),
                    "use_peft": config.get("use_peft", True),
                    "load_in_4bit": config.get("load_in_4bit", False),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if config.get("trainer", "trl") == "legacy":
        summary = train_ppo_style(config)
    else:
        summary = train_ppo_trl(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
