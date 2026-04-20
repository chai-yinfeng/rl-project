"""Centralized model and tokenizer loading."""

from __future__ import annotations

from typing import Any


def resolve_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install torch before loading models.") from exc

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def load_causal_lm_and_tokenizer(
    model_name_or_path: str,
    *,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    load_in_4bit: bool = False,
    trust_remote_code: bool = False,
) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers before loading models.") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "device_map": device_map,
        "torch_dtype": resolve_torch_dtype(torch_dtype),
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
    }
    if load_in_4bit:
        model_kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.eval()
    return model, tokenizer

