#!/usr/bin/env python3
"""Summarize experiment directories into paper-ready tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METHODS = ("baseline", "math_baseline", "dpo", "grpo", "ppo")
SUMMARY_COLUMNS = [
    "run",
    "method",
    "train_examples",
    "group_size",
    "accuracy",
    "delta_vs_baseline",
    "invalid_rate",
    "avg_completion_chars",
    "final_answer_rate",
    "train_eval_best_acc",
    "reward_final",
    "frac_reward_zero_std_final",
    "train_time_sec",
    "eval_time_sec",
    "peak_vram_gb",
    "effective_batch_size",
    "total_episodes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--csv-output", type=Path, default=Path("experiments/summary.csv"))
    parser.add_argument("--md-output", type=Path, default=Path("experiments/summary.md"))
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def nested_get(record: dict[str, Any], keys: list[str], default: Any = "") -> Any:
    current: Any = record
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def peak_vram_gb(*records: dict[str, Any]) -> str:
    values = [
        int(record.get(key, 0) or 0)
        for record in records
        for key in ("cuda_max_memory_allocated", "cuda_max_memory_reserved")
    ]
    peak = max(values, default=0)
    return f"{peak / (1024 ** 3):.2f}" if peak else ""


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def load_train_eval_best(run_dir: Path, method: str) -> str:
    records = read_jsonl(run_dir / "models" / method / "metrics" / "train_eval.jsonl")
    accuracies = [
        float(record["accuracy"])
        for record in records
        if "accuracy" in record
    ]
    return f"{max(accuracies):.4f}" if accuracies else ""


def load_trainer_tail(run_dir: Path, method: str) -> dict[str, Any]:
    state = read_json(run_dir / "models" / method / "trainer_state.json")
    history = state.get("log_history", [])
    for record in reversed(history):
        if "reward" in record or "frac_reward_zero_std" in record:
            return record
    return {}


def summarize_run(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_metrics = read_json(run_dir / "metrics" / "baseline.json")
    baseline_accuracy = baseline_metrics.get("accuracy")

    for method in METHODS:
        eval_metrics = read_json(run_dir / "metrics" / f"{method}.json")
        if not eval_metrics:
            continue

        train_metrics = {}
        if method == "ppo":
            train_metrics = read_json(run_dir / "models" / method / "metrics" / "summary.json")
        elif method != "baseline":
            train_metrics = read_json(run_dir / "models" / method / "metrics" / "train.json")

        diagnostics = read_json(run_dir / "logs" / "diagnostics" / f"{method}_eval.json")
        trainer_tail = load_trainer_tail(run_dir, method)
        accuracy = eval_metrics.get("accuracy")
        delta = (
            float(accuracy) - float(baseline_accuracy)
            if accuracy is not None and baseline_accuracy is not None
            else None
        )

        rows.append(
            {
                "run": run_dir.name,
                "method": method,
                "train_examples": train_metrics.get(
                    "train_examples",
                    train_metrics.get("train_pairs", ""),
                ),
                "group_size": train_metrics.get("group_size", ""),
                "accuracy": accuracy,
                "delta_vs_baseline": delta,
                "invalid_rate": eval_metrics.get("invalid_rate", ""),
                "avg_completion_chars": eval_metrics.get("average_completion_chars", ""),
                "final_answer_rate": diagnostics.get("final_answer_rate", ""),
                "train_eval_best_acc": load_train_eval_best(run_dir, method),
                "reward_final": trainer_tail.get("reward", ""),
                "frac_reward_zero_std_final": trainer_tail.get("frac_reward_zero_std", ""),
                "train_time_sec": train_metrics.get(
                    "train_runtime",
                    train_metrics.get("elapsed_seconds", ""),
                ),
                "eval_time_sec": eval_metrics.get("elapsed_seconds", ""),
                "peak_vram_gb": peak_vram_gb(train_metrics, eval_metrics),
                "effective_batch_size": train_metrics.get("effective_batch_size", ""),
                "total_episodes": train_metrics.get("total_episodes", ""),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in SUMMARY_COLUMNS})


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(SUMMARY_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in SUMMARY_COLUMNS) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(format_value(row.get(key, "")) for key in SUMMARY_COLUMNS)
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for run_dir in args.run_dirs:
        rows.extend(summarize_run(run_dir))
    if not rows:
        raise SystemExit("No result metrics found.")
    write_csv(args.csv_output, rows)
    write_markdown(args.md_output, rows)
    print(f"Wrote {args.csv_output}")
    print(f"Wrote {args.md_output}")


if __name__ == "__main__":
    main()
