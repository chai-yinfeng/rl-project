#!/usr/bin/env python3
"""Plot method-specific training dashboards from TRL trainer_state logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_METRICS = {
    "grpo": [
        "reward",
        "reward_std",
        "kl",
        "loss",
        "frac_reward_zero_std",
        "completions/mean_length",
        "completions/clipped_ratio",
        "train_eval/accuracy",
    ],
    "ppo": [
        "objective/rlhf_reward",
        "objective/scores",
        "objective/kl",
        "objective/non_score_reward",
        "objective/entropy",
        "policy/approxkl_avg",
        "policy/clipfrac_avg",
        "loss/policy_avg",
        "loss/value_avg",
        "val/clipfrac_avg",
        "policy/entropy_avg",
        "val/ratio",
        "val/num_eos_tokens",
        "lr",
    ],
    "dpo": [
        "loss",
        "rewards/accuracies",
        "rewards/margins",
        "rewards/chosen",
        "rewards/rejected",
        "mean_token_accuracy",
        "entropy",
        "grad_norm",
        "learning_rate",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--method", choices=["grpo", "ppo", "dpo"], default="grpo")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--metric", action="append", default=None)
    parser.add_argument("--rolling-window", type=int, default=1)
    parser.add_argument("--cols", type=int, default=4)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def trainer_state_path(run_dir: Path, method: str) -> Path:
    final = run_dir / "models" / method / "trainer_state.json"
    if final.exists():
        return final
    checkpoints = sorted((run_dir / "models" / method).glob("checkpoint-*/trainer_state.json"))
    if checkpoints:
        return checkpoints[-1]
    raise FileNotFoundError(f"No trainer_state.json found for {method} in {run_dir}")


def numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def x_value(record: dict[str, Any]) -> float | None:
    for key in ("episode", "step", "num_tokens"):
        value = numeric_value(record.get(key))
        if value is not None:
            return value
    return None


def rolling_average(points: list[tuple[float, float]], window: int) -> list[tuple[float, float]]:
    if window <= 1:
        return points
    smoothed: list[tuple[float, float]] = []
    values: list[float] = []
    for step, value in points:
        values.append(value)
        window_values = values[-window:]
        smoothed.append((step, sum(window_values) / len(window_values)))
    return smoothed


def load_points(path: Path, metrics: list[str]) -> dict[str, list[tuple[float, float]]]:
    history = read_json(path).get("log_history", [])
    series = {metric: [] for metric in metrics}
    for record in history:
        step = x_value(record)
        if step is None:
            continue
        for metric in metrics:
            value = numeric_value(record.get(metric))
            if value is not None:
                series[metric].append((step, value))
    return {metric: points for metric, points in series.items() if points}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "method", "metric", "x", "value"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    metrics = args.metric or DEFAULT_METRICS[args.method]
    output = args.output or Path(f"experiments/{args.method}_training_dashboard.png")
    csv_output = args.csv_output or output.with_suffix(".csv")

    all_series: dict[str, dict[str, list[tuple[float, float]]]] = {}
    csv_rows: list[dict[str, Any]] = []
    for run_dir in args.run_dirs:
        state_path = trainer_state_path(run_dir, args.method)
        label = run_dir.name
        series = load_points(state_path, metrics)
        all_series[label] = series
        for metric, points in series.items():
            for step, value in points:
                csv_rows.append(
                    {
                        "run": label,
                        "method": args.method,
                        "metric": metric,
                        "x": step,
                        "value": value,
                    }
                )

    present_metrics = [metric for metric in metrics if any(metric in s for s in all_series.values())]
    if not present_metrics:
        raise SystemExit("No requested metrics found.")

    write_csv(csv_output, csv_rows)

    cache_dir = tempfile.mkdtemp(prefix="matplotlib-")
    os.environ.setdefault("MPLCONFIGDIR", cache_dir)
    os.environ.setdefault("XDG_CACHE_HOME", cache_dir)
    import matplotlib.pyplot as plt

    cols = max(1, args.cols)
    rows = math.ceil(len(present_metrics) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)
    axes_flat = list(axes.ravel())
    for axis, metric in zip(axes_flat, present_metrics, strict=False):
        for label, series in all_series.items():
            points = series.get(metric)
            if not points:
                continue
            points = rolling_average(points, args.rolling_window)
            axis.plot(
                [step for step, _ in points],
                [value for _, value in points],
                linewidth=1.8,
                label=label,
            )
        axis.set_title(metric)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)

    for axis in axes_flat[len(present_metrics) :]:
        axis.axis("off")

    x_label = "episode" if args.method == "ppo" else "step"
    for axis in axes[-1, :]:
        if axis.has_data():
            axis.set_xlabel(x_label)
    fig.suptitle(f"{args.method.upper()} Training Metrics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Wrote {output}")
    print(f"Wrote {csv_output}")


if __name__ == "__main__":
    main()
