"""A compact PPO-style trainer for rule-reward GSM8K experiments.

This is intentionally small and transparent. It uses sampled rollouts, frozen
old-policy log probabilities, optional reference-model KL control, and a clipped
policy-gradient objective. It does not train a learned value model; advantages are
batch-normalized rewards. That makes it useful as a compute-aware PPO-style
engineering baseline for the course project.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from reasoning_post_training.experiments import append_jsonl, cuda_memory_summary, write_json
from reasoning_post_training.rewards.gsm8k import correctness_reward, format_reward
from reasoning_post_training.evaluation.answer_extraction import extract_predicted_answer


def build_ppo_dataset(split: str, subset: str, max_examples: int | None = None):
    from reasoning_post_training.datasets.gsm8k import load_gsm8k_split

    dataset = load_gsm8k_split(split=split, subset=subset)
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    return dataset


def ppo_rule_reward(completion: str, gold_answer: str | int | float | None) -> float:
    reward = correctness_reward(completion, gold_answer)
    reward += format_reward(completion)
    if extract_predicted_answer(completion) is None:
        reward -= 0.1
    return reward


def _resolve_torch_dtype(dtype_name: str):
    import torch

    if dtype_name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def _load_model(model_name_or_path: str, config: dict[str, Any], *, trainable: bool):
    import torch
    from transformers import AutoModelForCausalLM

    model_kwargs: dict[str, Any] = {
        "device_map": config.get("device_map", "auto"),
        "dtype": _resolve_torch_dtype(config.get("torch_dtype", "float16")),
        "low_cpu_mem_usage": True,
    }
    if config.get("load_in_4bit", False):
        model_kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    if trainable and config.get("use_peft", True):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if config.get("load_in_4bit", False):
            model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=config.get("lora_target_modules") or None,
        )
        model = get_peft_model(model, peft_config)
    if not trainable:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    elif hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def _sequence_logprobs(model, tokenizer, prompts: list[str], completions: list[str]):
    import torch
    import torch.nn.functional as F

    texts = [prompt + completion for prompt, completion in zip(prompts, completions, strict=True)]
    encoded = tokenizer(texts, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    prompt_lengths = [
        tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].shape[1]
        for prompt in prompts
    ]
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    token_logprobs = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    response_mask = torch.zeros_like(labels, dtype=torch.bool)
    for row, prompt_length in enumerate(prompt_lengths):
        response_mask[row, max(prompt_length - 1, 0) :] = True
    response_mask = response_mask & attention_mask[:, 1:].bool()

    lengths = response_mask.sum(dim=1).clamp_min(1)
    sequence_logprobs = (token_logprobs * response_mask).sum(dim=1) / lengths
    return sequence_logprobs, lengths


def _generate_batch(model, tokenizer, prompts: list[str], config: dict[str, Any]) -> list[str]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=config.get("max_new_tokens", 128),
        do_sample=True,
        temperature=config.get("temperature", 0.7),
        top_p=config.get("top_p", 0.95),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_width = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(output[prompt_width:], skip_special_tokens=True).strip()
        for output in outputs
    ]


def train_ppo_style(config: dict[str, Any]) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    model_name = config["model_name_or_path"]
    output_dir = Path(config["output_dir"])
    log_path = output_dir / "logs" / "train_metrics.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = _load_model(model_name, config, trainable=True)
    reference = None
    if config.get("kl_beta", 0.0) > 0:
        reference = _load_model(model_name, config, trainable=False)

    dataset = build_ppo_dataset(
        split=config.get("split", "train"),
        subset=config.get("subset", "main"),
        max_examples=config.get("max_train_examples"),
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=config.get("learning_rate", 1e-6),
    )

    batch_size = int(config.get("per_device_train_batch_size", 1))
    max_steps = int(config.get("max_steps", 20))
    clip_range = float(config.get("clip_range", 0.2))
    kl_beta = float(config.get("kl_beta", 0.02))
    save_steps = int(config.get("save_steps", max_steps))
    start_time = time.time()

    policy.train()
    for step in range(1, max_steps + 1):
        batch_indices = [((step - 1) * batch_size + offset) % len(dataset) for offset in range(batch_size)]
        batch = [dataset[index] for index in batch_indices]
        prompts = [record["prompt"] for record in batch]
        gold_answers = [record["gold_answer"] for record in batch]

        policy.eval()
        with torch.no_grad():
            completions = _generate_batch(policy, tokenizer, prompts, config)
            old_logprobs, lengths = _sequence_logprobs(policy, tokenizer, prompts, completions)
            ref_logprobs = None
            if reference is not None:
                ref_logprobs, _ = _sequence_logprobs(reference, tokenizer, prompts, completions)
        policy.train()

        rewards = torch.tensor(
            [ppo_rule_reward(completion, gold) for completion, gold in zip(completions, gold_answers, strict=True)],
            dtype=torch.float32,
            device=old_logprobs.device,
        )
        if rewards.numel() > 1 and rewards.std(unbiased=False) > 1e-6:
            advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-6)
        else:
            advantages = rewards - rewards.mean()
            if torch.allclose(advantages, torch.zeros_like(advantages)):
                advantages = rewards

        new_logprobs, _ = _sequence_logprobs(policy, tokenizer, prompts, completions)
        ratios = torch.exp(new_logprobs - old_logprobs)
        unclipped = ratios * advantages
        clipped = torch.clamp(ratios, 1.0 - clip_range, 1.0 + clip_range) * advantages
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        approx_kl = new_logprobs - old_logprobs
        ref_kl = torch.zeros_like(approx_kl)
        if ref_logprobs is not None:
            ref_kl = new_logprobs - ref_logprobs
        loss = policy_loss + kl_beta * ref_kl.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), config.get("max_grad_norm", 1.0))
        optimizer.step()

        invalid = sum(extract_predicted_answer(completion) is None for completion in completions)
        record = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "reward_mean": float(rewards.mean().detach().cpu()),
            "reward_std": float(rewards.std(unbiased=False).detach().cpu()) if rewards.numel() > 1 else 0.0,
            "invalid_rate": invalid / len(completions),
            "completion_length_mean": float(lengths.float().mean().detach().cpu()),
            "approx_kl_mean": float(approx_kl.mean().detach().cpu()),
            "ref_kl_mean": float(ref_kl.mean().detach().cpu()),
            "elapsed_seconds": time.time() - start_time,
            **cuda_memory_summary(),
        }
        append_jsonl(log_path, record)

        if save_steps > 0 and step % save_steps == 0:
            checkpoint_dir = output_dir / f"checkpoint-{step}"
            policy.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    summary = {
        "method": "ppo_style",
        "model_name_or_path": model_name,
        "output_dir": str(output_dir),
        "max_steps": max_steps,
        "elapsed_seconds": time.time() - start_time,
        **cuda_memory_summary(),
    }
    write_json(output_dir / "metrics" / "summary.json", summary)
    if not math.isfinite(summary["elapsed_seconds"]):
        raise RuntimeError("Invalid PPO-style elapsed time.")
    return summary
