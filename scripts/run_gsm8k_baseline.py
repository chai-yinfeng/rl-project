#!/usr/bin/env python3
"""Run baseline inference on GSM8K and write prediction JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from grpo_reasoning.data.gsm8k import DEFAULT_SYSTEM_PROMPT, load_gsm8k_split
from grpo_reasoning.evaluation.answer_extraction import extract_predicted_answer
from grpo_reasoning.models.loading import load_causal_lm_and_tokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split", default="test")
    parser.add_argument("--subset", default="main")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/processed/gsm8k_baseline.jsonl"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_model_and_tokenizer(args: argparse.Namespace):
    return load_causal_lm_and_tokenizer(
        args.model,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
    )


def build_chat_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem:\n{question.strip()}"},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{DEFAULT_SYSTEM_PROMPT}\n\nProblem:\n{question.strip()}\n\nSolution:"


def read_completed_indices(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()

    completed: set[int] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if "example_index" in record:
                completed.add(int(record["example_index"]))
    return completed


def generate_batch(model, tokenizer, prompts: list[str], args: argparse.Namespace) -> list[str]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
        )
    else:
        generation_kwargs["do_sample"] = False

    outputs = model.generate(**inputs, **generation_kwargs)
    prompt_width = inputs["input_ids"].shape[1]

    completions = []
    for output in outputs:
        generated_tokens = output[prompt_width:]
        completions.append(tokenizer.decode(generated_tokens, skip_special_tokens=True).strip())
    return completions


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    dataset = load_gsm8k_split(split=args.split, subset=args.subset)
    end = min(args.start + args.limit, len(dataset)) if args.limit else len(dataset)
    indices = list(range(args.start, end))

    completed = read_completed_indices(args.output) if args.resume else set()
    pending_indices = [idx for idx in indices if idx not in completed]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer(args)

    with args.output.open("a", encoding="utf-8") as handle:
        for batch_start in tqdm(range(0, len(pending_indices), args.batch_size)):
            batch_indices = pending_indices[batch_start : batch_start + args.batch_size]
            batch = [dataset[idx] for idx in batch_indices]
            prompts = [build_chat_prompt(tokenizer, record["question"]) for record in batch]
            completions = generate_batch(model, tokenizer, prompts, args)

            for example_index, record, prompt, completion in zip(
                batch_indices, batch, prompts, completions, strict=True
            ):
                output_record = {
                    "example_index": example_index,
                    "model": args.model,
                    "split": args.split,
                    "question": record["question"],
                    "prompt": prompt,
                    "completion": completion,
                    "predicted_answer": extract_predicted_answer(completion),
                    "gold_answer": record["gold_answer"],
                    "answer": record["answer"],
                }
                handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                handle.flush()


if __name__ == "__main__":
    main()
