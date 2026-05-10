#!/usr/bin/env python3
"""Run GRPO training on GSM8K."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reasoning_post_training.datasets.gsm8k import format_gsm8k_chat_prompt, load_gsm8k_split
from reasoning_post_training.evaluation.answer_extraction import truncate_completion
from reasoning_post_training.evaluation.metrics import evaluate_completions
from reasoning_post_training.experiments import append_jsonl, cuda_memory_summary, write_json
from reasoning_post_training.methods.grpo import (
    build_grpo_config,
    build_gsm8k_grpo_dataset,
    gsm8k_grpo_reward_func,
)
from reasoning_post_training.runtime import set_seed

try:
    from transformers import TrainerCallback
except ImportError:
    class TrainerCallback:  # type: ignore[no-redef]
        pass


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


def build_peft_config(config: dict[str, Any]):
    if not config.get("use_peft", False):
        return None

    try:
        from peft import LoraConfig
    except ImportError as exc:
        raise RuntimeError("Install peft before running GRPO with use_peft=true.") from exc

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


class TemperatureZeroEvalCallback(TrainerCallback):
    """Periodically evaluate the current policy with deterministic generation."""

    def __init__(self, tokenizer, eval_records: list[dict[str, Any]], config: dict[str, Any]) -> None:
        self.tokenizer = tokenizer
        self.eval_records = eval_records
        self.eval_steps = int(config.get("train_eval_steps", 0) or 0)
        self.batch_size = int(config.get("train_eval_batch_size", 4))
        self.max_new_tokens = int(config.get("train_eval_max_new_tokens", 256))
        self.max_prompt_length = int(config.get("max_prompt_length", 512))
        self.use_chat_template = bool(config.get("use_chat_template", True))
        self.output_path = Path(config["output_dir"]) / "metrics" / "train_eval.jsonl"
        self.completed_steps: set[int] = set()
        self.trainer = None

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        if not self.eval_records or self.eval_steps <= 0:
            return control
        step = int(state.global_step)
        if step <= 0 or step in self.completed_steps or step % self.eval_steps != 0:
            return control

        model = kwargs.get("model")
        if model is None:
            return control
        metrics = self.evaluate(model, step)
        if self.trainer is not None:
            self.trainer.log(metrics)
        self.completed_steps.add(step)
        return control

    def evaluate(self, model, step: int) -> dict[str, float]:  # noqa: ANN001
        import torch

        was_training = model.training
        model.eval()
        completions: list[str] = []
        gold_answers: list[str] = []

        with torch.no_grad():
            for batch_start in range(0, len(self.eval_records), self.batch_size):
                batch = self.eval_records[batch_start : batch_start + self.batch_size]
                prompts = [
                    format_gsm8k_chat_prompt(self.tokenizer, str(record["prompt"]))
                    if self.use_chat_template
                    else str(record["prompt"])
                    for record in batch
                ]
                inputs = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_prompt_length,
                )
                device = next(model.parameters()).device
                inputs = {key: value.to(device) for key, value in inputs.items()}
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
                prompt_width = inputs["input_ids"].shape[1]
                completions.extend(
                    truncate_completion(
                        self.tokenizer.decode(output[prompt_width:], skip_special_tokens=True)
                    )
                    for output in outputs
                )
                gold_answers.extend(str(record["gold_answer"]) for record in batch)

        if was_training:
            model.train()

        result = evaluate_completions(completions, gold_answers)
        final_answer_count = sum("final answer" in completion.lower() for completion in completions)
        record = {
            "step": step,
            "total": result.total,
            "correct": result.correct,
            "invalid": result.invalid,
            "accuracy": result.accuracy,
            "invalid_rate": result.invalid_rate,
            "final_answer_rate": final_answer_count / result.total if result.total else 0.0,
        }
        append_jsonl(self.output_path, record)
        return {
            "train_eval/accuracy": result.accuracy,
            "train_eval/invalid_rate": result.invalid_rate,
            "train_eval/final_answer_rate": record["final_answer_rate"],
        }


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
                    "train_eval_steps": config.get("train_eval_steps", 0),
                    "train_eval_limit": config.get("train_eval_limit", 0),
                    "use_peft": config.get("use_peft", False),
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

    if config.get("use_chat_template", True):
        train_dataset = train_dataset.map(
            lambda record: {
                "prompt": format_gsm8k_chat_prompt(tokenizer, str(record["prompt"]))
            }
        )

    grpo_config = build_grpo_config(config)
    peft_config = build_peft_config(config)
    trainer = GRPOTrainer(
        model=config["model_name_or_path"],
        reward_funcs=gsm8k_grpo_reward_func,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    train_eval_limit = int(config.get("train_eval_limit", 0) or 0)
    if train_eval_limit > 0 and int(config.get("train_eval_steps", 0) or 0) > 0:
        eval_split = config.get("train_eval_split", "test")
        eval_start = int(config.get("train_eval_start", 0))
        eval_dataset = load_gsm8k_split(split=eval_split, subset=config.get("subset", "main"))
        eval_end = min(eval_start + train_eval_limit, len(eval_dataset))
        eval_records = [dict(eval_dataset[index]) for index in range(eval_start, eval_end)]
        eval_callback = TemperatureZeroEvalCallback(tokenizer, eval_records, config)
        eval_callback.trainer = trainer
        trainer.add_callback(eval_callback)
    train_result = trainer.train()
    trainer.save_model(config["output_dir"])
    trainer.save_state()
    effective_batch_size = int(config.get("per_device_train_batch_size", 1)) * int(
        config.get("gradient_accumulation_steps", 1)
    )
    write_json(
        Path(config["output_dir"]) / "metrics" / "train.json",
        {
            "method": "grpo",
            "model_name_or_path": config["model_name_or_path"],
            "train_examples": len(train_dataset),
            "group_size": int(config.get("num_generations", 1)),
            "num_generations": int(config.get("num_generations", 1)),
            "generation_batch_size": int(config.get("generation_batch_size", 0) or 0),
            "effective_batch_size": effective_batch_size,
            "per_device_train_batch_size": int(config.get("per_device_train_batch_size", 1)),
            "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps", 1)),
            "learning_rate": float(config.get("learning_rate", 0.0)),
            "lr_scheduler_type": config.get("lr_scheduler_type", ""),
            "warmup_ratio": float(config.get("warmup_ratio", 0.0) or 0.0),
            "max_steps": int(config.get("max_steps", -1)),
            "beta": float(config.get("beta", 0.0)),
            "epsilon": float(config.get("epsilon", 0.0) or 0.0),
            "epsilon_high": float(config.get("epsilon_high", 0.0) or 0.0),
            "loss_type": config.get("loss_type", ""),
            "scale_rewards": config.get("scale_rewards", ""),
            "temperature": float(config.get("temperature", 0.0) or 0.0),
            "top_p": float(config.get("top_p", 0.0) or 0.0),
            "max_completion_length": int(config.get("max_completion_length", 0) or 0),
            "train_eval_limit": int(config.get("train_eval_limit", 0) or 0),
            "train_eval_steps": int(config.get("train_eval_steps", 0) or 0),
            "use_peft": bool(config.get("use_peft", False)),
            **{key: float(value) for key, value in train_result.metrics.items()},
            **cuda_memory_summary(),
        },
    )


if __name__ == "__main__":
    main()
