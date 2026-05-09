#!/usr/bin/env python3
"""Plot reward curves from TRL trainer_state.json files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any


SERIES_KEYS = (
    "reward",
    "rewards/gsm8k_grpo_reward_func/mean",
    "train_eval/accuracy",
    "train_eval/invalid_rate",
    "train_eval/final_answer_rate",
    "objective/scores",
    "objective/rlhf_reward",
    "objective/kl",
    "frac_reward_zero_std",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="Experiment run directories or direct trainer_state.json paths.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/training_rewards.png"),
        help="Figure output path, for example .png, .pdf, or .svg.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV output path. Defaults to the figure path with .csv suffix.",
    )
    parser.add_argument(
        "--series",
        action="append",
        choices=SERIES_KEYS,
        default=None,
        help="Metric to plot. Repeat to plot multiple metrics. Defaults to reward-like metrics.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=1,
        help="Optional moving-average window over logged points.",
    )
    return parser.parse_args()


def trainer_state_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    final_states = [
        path / "models" / method / "trainer_state.json"
        for method in ("grpo", "ppo", "dpo")
    ]
    existing_final_states = [candidate for candidate in final_states if candidate.exists()]
    if existing_final_states:
        return existing_final_states
    return sorted(path.glob("models/*/checkpoint-*/trainer_state.json"))


def resolve_label(path: Path) -> str:
    for parent in path.parents:
        if parent.name == "models":
            return f"{parent.parent.name}/{path.parent.name}"
    return path.parent.name


def load_series(path: Path, wanted_keys: list[str]) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)

    history = state.get("log_history", [])
    label = resolve_label(path)
    rows: list[dict[str, Any]] = []
    for key in wanted_keys:
        points: list[tuple[float, float]] = []
        for index, record in enumerate(history, start=1):
            if key not in record:
                continue
            x_value = record.get("step", index)
            try:
                points.append((float(x_value), float(record[key])))
            except (TypeError, ValueError):
                continue
        if points:
            rows.append({"label": label, "metric": key, "points": points})
    return rows


def load_train_eval_series(run_dir: Path, wanted_keys: list[str]) -> list[dict[str, Any]]:
    path = run_dir / "models" / "grpo" / "metrics" / "train_eval.jsonl"
    if not path.exists():
        return []

    metric_map = {
        "train_eval/accuracy": "accuracy",
        "train_eval/invalid_rate": "invalid_rate",
        "train_eval/final_answer_rate": "final_answer_rate",
    }
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows: list[dict[str, Any]] = []
    for metric, source_key in metric_map.items():
        if metric not in wanted_keys:
            continue
        points: list[tuple[float, float]] = []
        for record in records:
            if source_key not in record:
                continue
            points.append((float(record["step"]), float(record[source_key])))
        if points:
            rows.append({"label": f"{run_dir.name}/grpo", "metric": metric, "points": points})
    return rows


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


def write_csv(path: Path, series_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "series", "step", "value"])
        writer.writeheader()
        for row in series_rows:
            for step, value in row["points"]:
                writer.writerow(
                    {
                        "run": row["label"],
                        "series": row["metric"],
                        "step": step,
                        "value": value,
                    }
                )


def plot(series_rows: list[dict[str, Any]], output: Path, rolling_window: int) -> None:
    cache_dir = tempfile.mkdtemp(prefix="matplotlib-")
    os.environ.setdefault("MPLCONFIGDIR", cache_dir)
    os.environ.setdefault("XDG_CACHE_HOME", cache_dir)

    import matplotlib.pyplot as plt

    metrics = list(dict.fromkeys(row["metric"] for row in series_rows))
    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(10, max(3.6, 3.0 * len(metrics))),
        sharex=True,
        squeeze=False,
    )

    for axis, metric in zip(axes[:, 0], metrics, strict=True):
        metric_rows = [row for row in series_rows if row["metric"] == metric]
        for row in metric_rows:
            points = rolling_average(row["points"], rolling_window)
            xs = [step for step, _ in points]
            ys = [value for _, value in points]
            axis.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.8, label=row["label"])
        axis.set_title(metric)
        axis.set_ylabel("value")
        axis.grid(True, alpha=0.28)
        axis.legend(loc="best", fontsize=8)

    axes[-1, 0].set_xlabel("step")
    fig.suptitle("Training Metrics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    wanted_keys = args.series or [
        "reward",
        "rewards/gsm8k_grpo_reward_func/mean",
        "train_eval/accuracy",
        "objective/scores",
    ]
    series_rows: list[dict[str, Any]] = []
    for run_dir in args.run_dirs:
        for state_path in trainer_state_paths(run_dir):
            series_rows.extend(load_series(state_path, wanted_keys))
        if run_dir.is_dir():
            existing = {(row["label"], row["metric"]) for row in series_rows}
            for row in load_train_eval_series(run_dir, wanted_keys):
                if (row["label"], row["metric"]) not in existing:
                    series_rows.append(row)

    if not series_rows:
        raise SystemExit("No matching trainer_state metrics found.")

    plot(series_rows, args.output, args.rolling_window)
    csv_output = args.csv_output or args.output.with_suffix(".csv")
    write_csv(csv_output, series_rows)
    print(f"Wrote {args.output}")
    print(f"Wrote {csv_output}")


if __name__ == "__main__":
    main()
