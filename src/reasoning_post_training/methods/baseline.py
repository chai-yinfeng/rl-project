"""Baseline model inference and GSM8K evaluation helpers."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from reasoning_post_training.datasets.gsm8k import format_gsm8k_chat_prompt, load_gsm8k_split
from reasoning_post_training.evaluation.answer_extraction import extract_predicted_answer, truncate_completion
from reasoning_post_training.evaluation.metrics import evaluate_completions
from reasoning_post_training.experiments import cuda_memory_summary, write_json
from reasoning_post_training.models.loading import resolve_torch_dtype


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


def load_model_and_tokenizer(
    model_name_or_path: str,
    *,
    adapter: str | None = None,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    load_in_4bit: bool = False,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_kwargs: dict[str, Any] = {
        "device_map": device_map,
        "dtype": resolve_torch_dtype(torch_dtype),
        "low_cpu_mem_usage": True,
    }
    if load_in_4bit:
        model_kwargs["load_in_4bit"] = True

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    use_chat_template: bool = True,
) -> list[str]:
    model_inputs = [
        format_gsm8k_chat_prompt(tokenizer, prompt) if use_chat_template else prompt
        for prompt in prompts
    ]
    inputs = tokenizer(model_inputs, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
    else:
        generation_kwargs.update(
            {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
        )

    outputs = model.generate(**inputs, **generation_kwargs)
    prompt_width = inputs["input_ids"].shape[1]
    return [
        truncate_completion(tokenizer.decode(output[prompt_width:], skip_special_tokens=True))
        for output in outputs
    ]


def write_eval_diagnostics(
    metrics_output_path: Path,
    completions: list[str],
    gold_answers: list[str],
) -> None:
    """Write lightweight generation diagnostics next to run logs."""
    if not completions:
        return

    number_re = re.compile(r"[-+]?(?:\d[\d,]*)(?:\.\d+)?")
    final_answer_re = re.compile(r"final answer\s*[:=]?\s*[-+]?(?:\d[\d,]*)(?:\.\d+)?", re.I)
    exact = [
        extract_predicted_answer(completion) == str(gold)
        for completion, gold in zip(completions, gold_answers, strict=True)
    ]
    diagnostics = {
        "total": len(completions),
        "accuracy": sum(exact) / len(exact),
        "final_answer_rate": sum(bool(final_answer_re.search(text)) for text in completions)
        / len(completions),
        "chat_leak_rate": sum(
            any(marker in text for marker in ("Human:", "Assistant:", "User:", "System:"))
            for text in completions
        )
        / len(completions),
        "average_completion_chars": sum(len(text) for text in completions) / len(completions),
        "max_completion_chars": max(len(text) for text in completions),
        "average_number_count": sum(len(number_re.findall(text)) for text in completions)
        / len(completions),
    }
    run_dir = metrics_output_path.parent.parent
    write_json(
        run_dir / "logs" / "diagnostics" / f"{metrics_output_path.stem}_eval.json",
        diagnostics,
    )


def evaluate_gsm8k_model(
    *,
    model_name_or_path: str,
    adapter: str | None,
    split: str,
    subset: str,
    limit: int,
    start: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    torch_dtype: str,
    device_map: str,
    load_in_4bit: bool,
    output_path: Path,
    metrics_output_path: Path,
    resume: bool,
    use_chat_template: bool = True,
) -> dict[str, Any]:
    dataset = load_gsm8k_split(split=split, subset=subset)
    end = min(start + limit, len(dataset)) if limit else len(dataset)
    indices = list(range(start, end))
    completed = read_completed_indices(output_path) if resume else set()
    pending_indices = [index for index in indices if index not in completed]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path,
        adapter=adapter,
        torch_dtype=torch_dtype,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
    )

    started_at = time.time()
    with output_path.open("a", encoding="utf-8") as handle:
        for batch_start in tqdm(range(0, len(pending_indices), batch_size)):
            batch_indices = pending_indices[batch_start : batch_start + batch_size]
            batch = [dataset[index] for index in batch_indices]
            prompts = [record["prompt"] for record in batch]
            completions = generate_batch(
                model,
                tokenizer,
                prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                use_chat_template=use_chat_template,
            )
            for example_index, record, prompt, completion in zip(
                batch_indices, batch, prompts, completions, strict=True
            ):
                handle.write(
                    json.dumps(
                        {
                            "example_index": example_index,
                            "model": model_name_or_path,
                            "adapter": adapter,
                            "split": split,
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
    with output_path.open("r", encoding="utf-8") as handle:
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
    write_eval_diagnostics(metrics_output_path, completions, gold_answers)
    metrics = {
        **result.__dict__,
        "model": model_name_or_path,
        "adapter": adapter,
        "split": split,
        "limit": limit,
        "start": start,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "use_chat_template": use_chat_template,
        "average_completion_chars": sum(lengths) / len(lengths) if lengths else 0.0,
        "elapsed_seconds": time.time() - started_at,
        **cuda_memory_summary(),
    }
    write_json(metrics_output_path, metrics)
    return metrics
