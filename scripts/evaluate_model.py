#!/usr/bin/env python3
"""Evaluate a base model or PEFT adapter on GSM8K and write predictions plus metrics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from grpo_reasoning.data.gsm8k import load_gsm8k_split
from grpo_reasoning.evaluation.answer_extraction import extract_predicted_answer
from grpo_reasoning.evaluation.metrics import evaluate_completions
from grpo_reasoning.experiments import cuda_memory_summary, write_json
from grpo_reasoning.models.loading import resolve_torch_dtype


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


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


def load_model_and_tokenizer(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_kwargs: dict[str, Any] = {
        "device_map": args.device_map,
        "torch_dtype": resolve_torch_dtype(args.torch_dtype),
        "low_cpu_mem_usage": True,
    }
    if args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    return model, tokenizer


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
        generation_kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})
    else:
        generation_kwargs["do_sample"] = False

    outputs = model.generate(**inputs, **generation_kwargs)
    prompt_width = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(output[prompt_width:], skip_special_tokens=True).strip()
        for output in outputs
    ]


def main() -> None:
    args = parse_args()
    dataset = load_gsm8k_split(split=args.split, subset=args.subset)
    end = min(args.start + args.limit, len(dataset)) if args.limit else len(dataset)
    indices = list(range(args.start, end))
    completed = read_completed_indices(args.output) if args.resume else set()
    pending_indices = [index for index in indices if index not in completed]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output = args.metrics_output or args.output.with_suffix(".metrics.json")
    model, tokenizer = load_model_and_tokenizer(args)

    started_at = time.time()
    with args.output.open("a", encoding="utf-8") as handle:
        for batch_start in tqdm(range(0, len(pending_indices), args.batch_size)):
            batch_indices = pending_indices[batch_start : batch_start + args.batch_size]
            batch = [dataset[index] for index in batch_indices]
            prompts = [record["prompt"] for record in batch]
            completions = generate_batch(model, tokenizer, prompts, args)
            for example_index, record, prompt, completion in zip(
                batch_indices, batch, prompts, completions, strict=True
            ):
                handle.write(
                    json.dumps(
                        {
                            "example_index": example_index,
                            "model": args.model,
                            "adapter": args.adapter,
                            "split": args.split,
                            "question": record["question"],
                            "prompt": prompt,
                            "completion": completion,
                            "predicted_answer": extract_predicted_answer(completion),
                            "gold_answer": record["gold_answer"],
                            "answer": record["answer"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                handle.flush()

    completions: list[str] = []
    gold_answers: list[str] = []
    lengths: list[int] = []
    with args.output.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record["example_index"]) not in indices:
                continue
            completion = str(record["completion"])
            completions.append(completion)
            gold_answers.append(str(record["gold_answer"]))
            lengths.append(len(completion))

    result = evaluate_completions(completions, gold_answers)
    metrics = {
        **result.__dict__,
        "model": args.model,
        "adapter": args.adapter,
        "split": args.split,
        "limit": args.limit,
        "start": args.start,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "average_completion_chars": sum(lengths) / len(lengths) if lengths else 0.0,
        "elapsed_seconds": time.time() - started_at,
        **cuda_memory_summary(),
    }
    write_json(metrics_output, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
