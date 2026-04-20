#!/usr/bin/env python3
"""Load a small GSM8K slice and print normalized examples."""

from __future__ import annotations

import argparse

from grpo_reasoning.data.gsm8k import load_gsm8k_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    dataset = load_gsm8k_split(split=args.split)
    for record in dataset.select(range(min(args.limit, len(dataset)))):
        print({"question": record["question"], "gold_answer": record["gold_answer"]})


if __name__ == "__main__":
    main()

