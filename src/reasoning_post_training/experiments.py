"""Shared experiment filesystem helpers."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from importlib import metadata
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
    write_json(run_dir / "logs" / "env.json", environment_summary())
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


def environment_summary() -> dict[str, Any]:
    """Collect reproducibility metadata for an experiment run."""
    packages = {}
    for package_name in ("torch", "transformers", "trl", "peft", "datasets", "accelerate"):
        try:
            packages[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            packages[package_name] = None

    summary: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "git": git_summary(),
    }

    try:
        import torch
    except ImportError:
        return summary

    cuda: dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "torch_cuda": getattr(torch.version, "cuda", None),
    }
    if torch.cuda.is_available():
        cuda["device_count"] = torch.cuda.device_count()
        cuda["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
    summary["cuda"] = cuda
    return summary


def git_summary() -> dict[str, Any]:
    """Return git commit and dirty-state metadata when available."""
    def run_git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "status_short": run_git(["status", "--short"]),
    }
