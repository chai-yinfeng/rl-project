#!/usr/bin/env python3
"""Build synthetic GSM8K DPO preference pairs from sampled model completions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from reasoning_post_training.datasets.gsm8k import format_gsm8k_chat_prompt, load_gsm8k_split
from reasoning_post_training.evaluation.answer_extraction import truncate_completion
from reasoning_post_training.models.loading import load_causal_lm_and_tokenizer
from reasoning_post_training.methods.dpo import (
    build_gold_chosen_pair_from_completions,
    choose_preference_pair,
)
from reasoning_post_training.runtime import set_seed


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split", default="train")
    parser.add_argument("--subset", default="main")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num-completions", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/processed/gsm8k_dpo_pairs.jsonl"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-ties", action="store_true")
    parser.add_argument("--include-gold-chosen", action="store_true")
    parser.add_argument("--gold-fallback", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_completed_indices(
    output_path: Path,
    *,
    expected_prompt_hashes: dict[int, str] | None = None,
    expected_generation_config: dict[str, Any] | None = None,
) -> set[int]:
    if not output_path.exists():
        return set()

    completed: set[int] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if "example_index" in record:
                example_index = int(record["example_index"])
                if expected_prompt_hashes is not None:
                    expected = expected_prompt_hashes.get(example_index)
                    if record.get("prompt_hash") != expected:
                        continue
                if expected_generation_config is not None:
                    if any(record.get(key) != value for key, value in expected_generation_config.items()):
                        continue
                completed.add(example_index)
    return completed


def generate_completion_batches(
    model,
    tokenizer,
    prompts: list[str],
    args: argparse.Namespace,
) -> list[list[str]]:
    model_inputs = [
        format_gsm8k_chat_prompt(tokenizer, prompt) if not args.no_chat_template else prompt
        for prompt in prompts
    ]
    inputs = tokenizer(model_inputs, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_return_sequences": args.num_completions,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    outputs = model.generate(**inputs, **generation_kwargs)
    prompt_width = inputs["input_ids"].shape[1]
    completions = [
        truncate_completion(tokenizer.decode(output[prompt_width:], skip_special_tokens=True))
        for output in outputs
    ]
    return [
        completions[start : start + args.num_completions]
        for start in range(0, len(completions), args.num_completions)
    ]


def main() -> None:
    args = parse_args()
    if args.num_completions < 2:
        raise ValueError("--num-completions must be at least 2")
    if args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    set_seed(args.seed)
    dataset = load_gsm8k_split(split=args.split, subset=args.subset)
    end = min(args.start + args.limit, len(dataset))
    indices = list(range(args.start, end))
    expected_prompt_hashes = {
        index: prompt_hash(str(dataset[index]["prompt"]))
        for index in indices
    }
    expected_generation_config = {
        "model": args.model,
        "split": args.split,
        "num_completions": args.num_completions,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "use_chat_template": not args.no_chat_template,
        "allow_ties": args.allow_ties,
        "include_gold_chosen": args.include_gold_chosen,
        "gold_fallback": args.gold_fallback,
    }
    completed = (
        read_completed_indices(
            args.output,
            expected_prompt_hashes=expected_prompt_hashes,
            expected_generation_config=expected_generation_config,
        )
        if args.resume
        else set()
    )
    pending_indices = [idx for idx in indices if idx not in completed]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "split": args.split,
                    "candidate_examples": len(indices),
                    "pending_examples": len(pending_indices),
                    "batch_size": args.batch_size,
                    "num_completions": args.num_completions,
                    "include_gold_chosen": args.include_gold_chosen,
                    "gold_fallback": args.gold_fallback,
                    "use_chat_template": not args.no_chat_template,
                    "output": str(args.output),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_causal_lm_and_tokenizer(
        args.model,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.padding_side = "left"

    kept = 0
    skipped = 0
    with args.output.open("a", encoding="utf-8") as handle:
        for batch_start in tqdm(range(0, len(pending_indices), args.batch_size)):
            batch_indices = pending_indices[batch_start : batch_start + args.batch_size]
            records = [dataset[example_index] for example_index in batch_indices]
            prompt_completions = generate_completion_batches(
                model,
                tokenizer,
                [record["prompt"] for record in records],
                args,
            )
            for example_index, record, completions in zip(
                batch_indices, records, prompt_completions, strict=True
            ):
                if args.include_gold_chosen:
                    pair = build_gold_chosen_pair_from_completions(
                        completions,
                        record["answer"],
                        record["gold_answer"],
                    )
                else:
                    pair = choose_preference_pair(
                        completions,
                        record["gold_answer"],
                        require_distinct_scores=not args.allow_ties,
                    )
                    if pair is None and args.gold_fallback:
                        pair = build_gold_chosen_pair_from_completions(
                            completions,
                            record["answer"],
                            record["gold_answer"],
                        )
                if pair is None:
                    skipped += 1
                    continue

                output_record = {
                    "example_index": example_index,
                    "model": args.model,
                    "split": args.split,
                    "question": record["question"],
                    "prompt": record["prompt"],
                    "prompt_hash": prompt_hash(record["prompt"]),
                    **expected_generation_config,
                    "gold_answer": record["gold_answer"],
                    "answer": record["answer"],
                    **pair,
                }
                handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                handle.flush()
                kept += 1

    print(json.dumps({"kept_pairs": kept, "skipped_examples": skipped}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
