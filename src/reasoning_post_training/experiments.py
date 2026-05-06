"""Shared experiment filesystem helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def prepare_run_dir(output_root: Path, run_name: str, config: dict[str, Any]) -> Path:
    run_dir = output_root / run_name
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    return run_dir


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)


def cuda_memory_summary() -> dict[str, int]:
    try:
        import torch
    except ImportError:
        return {}
    if not torch.cuda.is_available():
        return {}
    return {
        "cuda_max_memory_allocated": int(torch.cuda.max_memory_allocated()),
        "cuda_max_memory_reserved": int(torch.cuda.max_memory_reserved()),
    }
