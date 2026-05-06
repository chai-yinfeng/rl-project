#!/usr/bin/env python3
"""Run DPO training on synthetic GSM8K preference pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reasoning_post_training.experiments import write_json
from reasoning_post_training.methods.dpo import build_dpo_config, load_dpo_jsonl
from reasoning_post_training.runtime import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_peft_config(config: dict[str, Any]):
    if not config.get("use_peft", False):
        return None

    try:
        from peft import LoraConfig
    except ImportError as exc:
        raise RuntimeError("Install peft before running DPO with use_peft=true.") from exc

    peft_kwargs = {
        "r": config.get("lora_r", 16),
        "lora_alpha": config.get("lora_alpha", 32),
        "lora_dropout": config.get("lora_dropout", 0.05),
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    if config.get("lora_target_modules"):
        peft_kwargs["target_modules"] = config["lora_target_modules"]
    return LoraConfig(**peft_kwargs)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.model is not None:
        config["model_name_or_path"] = args.model
    if args.train_file is not None:
        config["train_file"] = str(args.train_file)
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    set_seed(int(config.get("seed", 42)))
    if args.dry_run:
        train_file = Path(config["train_file"])
        train_pairs = len(load_dpo_jsonl(train_file)) if train_file.exists() else None
        print(
            json.dumps(
                {
                    "model_name_or_path": config["model_name_or_path"],
                    "train_file": str(train_file),
                    "train_pairs": train_pairs,
                    "output_dir": config["output_dir"],
                    "max_steps": config.get("max_steps", -1),
                    "num_train_epochs": config.get("num_train_epochs", 1),
                    "use_peft": config.get("use_peft", False),
                    "load_in_4bit": config.get("load_in_4bit", False),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    train_file = Path(config["train_file"])
    train_dataset = load_dpo_jsonl(train_file)

    try:
        from transformers import AutoTokenizer
        from trl import DPOTrainer
    except ImportError as exc:
        raise RuntimeError("Install transformers and trl before running DPO training.") from exc

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_init_kwargs = dict(config.get("model_init_kwargs", {}))
    if config.get("load_in_4bit", False):
        model_init_kwargs["load_in_4bit"] = True
    config["model_init_kwargs"] = model_init_kwargs

    dpo_config = build_dpo_config(config)
    trainer = DPOTrainer(
        model=config["model_name_or_path"],
        args=dpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=build_peft_config(config),
    )
    train_result = trainer.train()
    trainer.save_model(config["output_dir"])
    trainer.save_state()
    write_json(
        Path(config["output_dir"]) / "metrics" / "train.json",
        {key: float(value) for key, value in train_result.metrics.items()},
    )


if __name__ == "__main__":
    main()
