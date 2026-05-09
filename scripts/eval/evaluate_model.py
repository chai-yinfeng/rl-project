#!/usr/bin/env python3
"""Evaluate a base model or PEFT adapter on GSM8K."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reasoning_post_training.methods.baseline import evaluate_gsm8k_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base model or full checkpoint path.")
    parser.add_argument("--adapter", default=None, help="Optional PEFT adapter path.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--subset", default="main")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_gsm8k_model(
        model_name_or_path=args.model,
        adapter=args.adapter,
        split=args.split,
        subset=args.subset,
        limit=args.limit,
        start=args.start,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
        output_path=args.output,
        metrics_output_path=args.metrics_output or args.output.with_suffix(".metrics.json"),
        resume=args.resume,
        use_chat_template=not args.no_chat_template,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
