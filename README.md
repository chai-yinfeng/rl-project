# EECS E6892 RL Project: GRPO for LLM Reasoning

This repository contains the course project for EECS E6892 Reinforcement Learning.

## Goal

Implement and evaluate Group Relative Policy Optimization (GRPO) for small language
model reasoning tasks, with GSM8K as the first benchmark. The project will compare
GRPO against PPO-style RLHF and DPO-style preference optimization where feasible.

## Core Questions

- Does group-relative advantage estimation improve training stability for reasoning?
- How does group size affect variance, sample efficiency, and final accuracy?
- What are the practical tradeoffs between GRPO, PPO, and DPO for small open models?

## Repository Layout

- `configs/`: experiment and model configuration files.
- `data/`: local dataset cache and processed artifacts.
- `docs/`: project planning notes and proposal-derived design decisions.
- `notebooks/`: exploratory analysis and result inspection.
- `reports/`: figures, tables, and final report assets.
- `scripts/`: runnable training, evaluation, and data preparation entrypoints.
- `src/grpo_reasoning/`: reusable project source code.
- `tests/`: focused unit tests for data processing, rewards, and algorithm helpers.

## Initial Milestones

1. Build a supervised/evaluation baseline on GSM8K.
2. Implement reward extraction for math answers.
3. Implement GRPO training with configurable group size.
4. Add PPO/DPO baselines or use well-scoped library baselines.
5. Run controlled experiments and summarize stability, accuracy, and cost.

## Quick Start

Install the project in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Inspect a few GSM8K examples:

```bash
python scripts/inspect_gsm8k.py --split test --limit 3
```

Run a small baseline inference job:

```bash
python scripts/run_gsm8k_baseline.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --limit 100 \
  --batch-size 1 \
  --max-new-tokens 256 \
  --output data/processed/gsm8k_baseline.jsonl \
  --resume
```

Or use the 3070-friendly launcher:

```bash
bash scripts/run_baseline_3070.sh
```

The same flow is available through `make`:

```bash
make baseline-smoke
make baseline-100
make evaluate-baseline
```

Evaluate a JSONL prediction file:

```bash
python scripts/evaluate_predictions.py data/processed/gsm8k_baseline.jsonl
```

Each prediction record should include one prediction field, such as `completion`,
`prediction`, `response`, or `generated_text`, and one answer field, such as
`gold_answer`, `answer`, `target`, or `reference`.

Run local tests:

```bash
pytest
```

## Pipeline

The dataset is downloaded automatically by Hugging Face `datasets` when a script
first calls GSM8K. The repository intentionally does not commit dataset files,
prediction JSONL files, checkpoints, or logs. Local artifacts are written under
`data/processed/`, `outputs/`, `runs/`, or `wandb/`.

Run a minimal end-to-end smoke flow on the 3070 machine:

```bash
bash scripts/run_pipeline_smoke_3070.sh
```

Or run individual stages:

```bash
make inspect-gsm8k
make baseline-smoke
python scripts/evaluate_predictions.py data/processed/gsm8k_baseline_smoke.jsonl
make grpo-dry-run
make grpo-smoke
```

See `docs/pipeline.md` for the full project flow and module responsibilities.
