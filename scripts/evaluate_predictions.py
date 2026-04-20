#!/usr/bin/env python3
"""Evaluate GSM8K prediction JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grpo_reasoning.evaluation.metrics import evaluate_completions


PREDICTION_KEYS = ("completion", "prediction", "response", "generated_text")
GOLD_KEYS = ("gold_answer", "answer", "target", "reference")


def _first_present(record: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return str(value)
    return None


def read_prediction_jsonl(path: Path) -> tuple[list[str], list[str]]:
    completions: list[str] = []
    gold_answers: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            completion = _first_present(record, PREDICTION_KEYS)
            gold_answer = _first_present(record, GOLD_KEYS)
            if completion is None or gold_answer is None:
                raise ValueError(
                    f"Line {line_number} must include one prediction key {PREDICTION_KEYS} "
                    f"and one gold key {GOLD_KEYS}."
                )
            completions.append(completion)
            gold_answers.append(gold_answer)

    return completions, gold_answers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path, help="JSONL file with predictions and answers.")
    args = parser.parse_args()

    completions, gold_answers = read_prediction_jsonl(args.predictions)
    result = evaluate_completions(completions, gold_answers)
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

