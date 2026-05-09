#!/usr/bin/env python3
"""Unified experiment launcher for baseline, DPO, GRPO, and PPO-style runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from reasoning_post_training.experiments import append_jsonl, prepare_run_dir, timestamp, write_json


STAGE_ORDER = [
    "baseline",
    "dpo-pairs",
    "dpo-train",
    "dpo-eval",
    "grpo-train",
    "grpo-eval",
    "ppo-train",
    "ppo-eval",
]

STAGE_GROUPS = {
    "all": STAGE_ORDER,
    "run": ["baseline", "dpo-pairs", "dpo-train", "grpo-train", "ppo-train"],
    "eval": ["baseline", "dpo-eval", "grpo-eval", "ppo-eval"],
    "math": ["math-baseline"],
    "math-baseline": ["math-baseline"],
    "math-eval": ["math-baseline"],
    "dpo": ["dpo-pairs", "dpo-train", "dpo-eval"],
    "dpo-run": ["dpo-pairs", "dpo-train"],
    "grpo": ["grpo-train", "grpo-eval"],
    "grpo-run": ["grpo-train"],
    "ppo": ["ppo-train", "ppo-eval"],
    "ppo-run": ["ppo-train"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=[*STAGE_ORDER, *STAGE_GROUPS], default="all")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_command(command: list[str], *, dry_run: bool, run_dir: Path, stage: str) -> None:
    print(" ".join(command), flush=True)
    started_at = time.time()
    record = {
        "stage": stage,
        "command": command,
        "dry_run": dry_run,
        "started_at": started_at,
    }
    log_path = run_dir / "logs" / "stage_stdout" / f"{stage}.log"
    if dry_run:
        record.update({"returncode": 0, "elapsed_seconds": 0.0, "log_path": None})
        append_jsonl(run_dir / "logs" / "run_commands.jsonl", record)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
            log_handle.flush()
        returncode = process.wait()

    record.update(
        {
            "returncode": returncode,
            "elapsed_seconds": time.time() - started_at,
            "log_path": str(log_path),
        }
    )
    append_jsonl(run_dir / "logs" / "run_commands.jsonl", record)
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


def write_stage_config(run_dir: Path, name: str, config: dict[str, Any]) -> Path:
    path = run_dir / "configs" / f"{name}.json"
    write_json(path, config)
    return path


def build_eval_command(
    *,
    model: str,
    adapter: str | None,
    eval_config: dict[str, Any],
    output: Path,
    metrics_output: Path,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/eval/evaluate_model.py",
        "--model",
        model,
        "--split",
        eval_config.get("split", "test"),
        "--subset",
        eval_config.get("subset", "main"),
        "--limit",
        str(eval_config.get("limit", 100)),
        "--batch-size",
        str(eval_config.get("batch_size", 1)),
        "--max-new-tokens",
        str(eval_config.get("max_new_tokens", 256)),
        "--temperature",
        str(eval_config.get("temperature", 0.0)),
        "--top-p",
        str(eval_config.get("top_p", 0.95)),
        "--torch-dtype",
        eval_config.get("torch_dtype", "float16"),
        "--output",
        str(output),
        "--metrics-output",
        str(metrics_output),
        "--resume",
    ]
    if adapter:
        command.extend(["--adapter", adapter])
    if eval_config.get("load_in_4bit", False):
        command.append("--load-in-4bit")
    if eval_config.get("use_chat_template", True) is False:
        command.append("--no-chat-template")
    return command


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_name = config.get("run_name") or f"{Path(args.config).stem}_{timestamp()}"
    output_root = Path(config.get("output_root", "experiments"))
    run_dir = prepare_run_dir(output_root, run_name, config)

    model = config["model_name_or_path"]
    eval_config = config.get("eval", {})
    stages = STAGE_GROUPS.get(args.stage, [args.stage])

    dpo_pairs_path = run_dir / "data" / "dpo_pairs.jsonl"
    dpo_model_dir = run_dir / "models" / "dpo"
    grpo_model_dir = run_dir / "models" / "grpo"
    ppo_model_dir = run_dir / "models" / "ppo"

    for stage in stages:
        if stage == "baseline":
            run_command(
                build_eval_command(
                    model=model,
                    adapter=None,
                    eval_config=eval_config,
                    output=run_dir / "data" / "baseline_predictions.jsonl",
                    metrics_output=run_dir / "metrics" / "baseline.json",
                ),
                dry_run=args.dry_run,
                run_dir=run_dir,
                stage=stage,
            )

        elif stage == "math-baseline":
            math_model = config.get("math_model_name_or_path")
            if not math_model:
                raise ValueError("math-baseline stage requires math_model_name_or_path in config.")
            math_eval_config = {**eval_config, **config.get("math_eval", {})}
            run_command(
                build_eval_command(
                    model=math_model,
                    adapter=None,
                    eval_config=math_eval_config,
                    output=run_dir / "data" / "math_baseline_predictions.jsonl",
                    metrics_output=run_dir / "metrics" / "math_baseline.json",
                ),
                dry_run=args.dry_run,
                run_dir=run_dir,
                stage=stage,
            )

        elif stage == "dpo-pairs":
            dpo_pair_config = config.get("dpo_pairs", {})
            command = [
                sys.executable,
                "scripts/train/build_dpo_pairs.py",
                "--model",
                model,
                "--split",
                dpo_pair_config.get("split", "train"),
                "--subset",
                dpo_pair_config.get("subset", "main"),
                "--limit",
                str(dpo_pair_config.get("limit", 200)),
                "--num-completions",
                str(dpo_pair_config.get("num_completions", 4)),
                "--batch-size",
                str(dpo_pair_config.get("batch_size", 1)),
                "--max-new-tokens",
                str(dpo_pair_config.get("max_new_tokens", 128)),
                "--temperature",
                str(dpo_pair_config.get("temperature", 0.7)),
                "--top-p",
                str(dpo_pair_config.get("top_p", 0.95)),
                "--torch-dtype",
                dpo_pair_config.get("torch_dtype", "float16"),
                "--output",
                str(dpo_pairs_path),
                "--resume",
            ]
            if dpo_pair_config.get("load_in_4bit", False):
                command.append("--load-in-4bit")
            if dpo_pair_config.get("use_chat_template", True) is False:
                command.append("--no-chat-template")
            if dpo_pair_config.get("allow_ties", False):
                command.append("--allow-ties")
            if dpo_pair_config.get("include_gold_chosen", False):
                command.append("--include-gold-chosen")
            if dpo_pair_config.get("gold_fallback", False):
                command.append("--gold-fallback")
            run_command(command, dry_run=args.dry_run, run_dir=run_dir, stage=stage)

        elif stage == "dpo-train":
            dpo_config = dict(config.get("dpo_train", {}))
            dpo_config.setdefault("model_name_or_path", model)
            dpo_config["train_file"] = str(dpo_pairs_path)
            dpo_config["output_dir"] = str(dpo_model_dir)
            dpo_config_path = write_stage_config(run_dir, "dpo_train", dpo_config)
            command = [sys.executable, "scripts/train/run_dpo_train.py", "--config", str(dpo_config_path)]
            if args.dry_run:
                command.append("--dry-run")
            run_command(command, dry_run=args.dry_run, run_dir=run_dir, stage=stage)

        elif stage == "dpo-eval":
            run_command(
                build_eval_command(
                    model=model,
                    adapter=str(dpo_model_dir),
                    eval_config=eval_config,
                    output=run_dir / "data" / "dpo_predictions.jsonl",
                    metrics_output=run_dir / "metrics" / "dpo.json",
                ),
                dry_run=args.dry_run,
                run_dir=run_dir,
                stage=stage,
            )

        elif stage == "grpo-train":
            grpo_config = dict(config.get("grpo_train", {}))
            grpo_config.setdefault("model_name_or_path", model)
            grpo_config["output_dir"] = str(grpo_model_dir)
            grpo_config_path = write_stage_config(run_dir, "grpo_train", grpo_config)
            command = [sys.executable, "scripts/train/run_grpo_train.py", "--config", str(grpo_config_path)]
            if args.dry_run:
                command.append("--dry-run")
            run_command(command, dry_run=args.dry_run, run_dir=run_dir, stage=stage)

        elif stage == "grpo-eval":
            run_command(
                build_eval_command(
                    model=model,
                    adapter=str(grpo_model_dir),
                    eval_config=eval_config,
                    output=run_dir / "data" / "grpo_predictions.jsonl",
                    metrics_output=run_dir / "metrics" / "grpo.json",
                ),
                dry_run=args.dry_run,
                run_dir=run_dir,
                stage=stage,
            )

        elif stage == "ppo-train":
            ppo_config = dict(config.get("ppo_train", {}))
            ppo_config.setdefault("model_name_or_path", model)
            ppo_config["output_dir"] = str(ppo_model_dir)
            ppo_config_path = write_stage_config(run_dir, "ppo_train", ppo_config)
            command = [sys.executable, "scripts/train/run_ppo_train.py", "--config", str(ppo_config_path)]
            if args.dry_run:
                command.append("--dry-run")
            run_command(command, dry_run=args.dry_run, run_dir=run_dir, stage=stage)

        elif stage == "ppo-eval":
            run_command(
                build_eval_command(
                    model=model,
                    adapter=str(ppo_model_dir),
                    eval_config=eval_config,
                    output=run_dir / "data" / "ppo_predictions.jsonl",
                    metrics_output=run_dir / "metrics" / "ppo.json",
                ),
                dry_run=args.dry_run,
                run_dir=run_dir,
                stage=stage,
            )


if __name__ == "__main__":
    main()
