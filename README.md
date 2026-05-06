# EECS E6892 RL Project: GRPO/DPO/PPO-Style Reasoning

This repository contains a course project for comparing post-training methods on
GSM8K-style mathematical reasoning with small open language models.

## Scope

The project compares:

- baseline inference
- DPO from synthetic GSM8K preference pairs
- GRPO with online grouped rule rewards
- PPO-style rule-reward training with rollout logprobs, clipped policy updates, and optional reference KL

The main models are:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`

## Setup

Preferred `uv` setup:

```bash
uv sync --extra dev --extra quantization
uv run python -m pytest
```

Standard pip/venv setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,quantization]"
python -m pytest
```

Makefile commands default to `uv run python`. With a plain venv, use:

```bash
make PYTHON=python test
```

## Primary Entry Points

Use Makefile targets rather than calling individual stage scripts directly.

Full model groups:

```bash
make qwen0_5b-dry-run
make qwen0_5b-test
make qwen0_5b-run
make qwen0_5b-eval
make qwen0_5b-all

make qwen1_5b-dry-run
make qwen1_5b-test
make qwen1_5b-run
make qwen1_5b-eval
make qwen1_5b-all
```

Per-method targets:

```bash
make dpo-1_5b-dry-run
make dpo-1_5b-test
make dpo-1_5b-run
make dpo-1_5b-eval

make grpo-1_5b-dry-run
make grpo-1_5b-test
make grpo-1_5b-run
make grpo-1_5b-eval

make ppo-1_5b-dry-run
make ppo-1_5b-test
make ppo-1_5b-run
make ppo-1_5b-eval
```

Replace `1_5b` with `0_5b` for the smaller model.

## Output Layout

All experiment artifacts are written under:

```text
experiments/<run_name>/
  config.json
  configs/
  data/
  logs/
  metrics/
  models/
```

Package results for download:

```bash
make package-results
```

See [docs/experiment_design.md](docs/experiment_design.md) for the experiment
matrix and metrics, and [docs/server_setup.md](docs/server_setup.md) for RunPod
setup notes.
