# Pipeline

## Data

GSM8K is loaded by the scripts through Hugging Face `datasets`. The repository does
not commit dataset files, model outputs, checkpoints, or run logs. Those artifacts
are created locally under:

- `data/raw/` for optional local raw data.
- `data/processed/` for generated prediction JSONL files.
- `outputs/` for GRPO checkpoints and trainer outputs.
- `runs/` or `wandb/` for optional experiment tracking.

The first run on a new machine will download GSM8K into the Hugging Face cache.

## Minimal Flow

1. Install dependencies:

```bash
python -m pip install -e ".[dev]"
```

2. Confirm GSM8K can be loaded:

```bash
make inspect-gsm8k
```

3. Run baseline inference:

```bash
make baseline-smoke
```

4. Evaluate baseline predictions:

```bash
python scripts/evaluate_predictions.py data/processed/gsm8k_baseline_smoke.jsonl
```

5. Dry-run the GRPO smoke configuration:

```bash
python scripts/run_grpo_smoke.py --dry-run
```

6. Run the smallest GRPO smoke job:

```bash
python scripts/run_grpo_smoke.py --max-train-examples 8 --max-steps 1
```

## 3070 Smoke Flow

On the 8GB RTX 3070 machine, start with the smoke pipeline:

```bash
bash scripts/run_pipeline_smoke_3070.sh
```

If that completes, increase gradually:

```bash
GRPO_EXAMPLES=32 GRPO_STEPS=5 bash scripts/run_pipeline_smoke_3070.sh
```

For 8GB VRAM, keep the first GRPO experiments conservative:

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- batch size: `1`
- num generations: `2`
- max completion length: `128`
- max steps: `1` to `5` for smoke tests

## Module Responsibilities

- `src/grpo_reasoning/data/`: dataset loading and prompt formatting.
- `src/grpo_reasoning/evaluation/`: answer extraction and metrics.
- `src/grpo_reasoning/rewards/`: reward functions used by baseline analysis and GRPO.
- `src/grpo_reasoning/models/`: shared model/tokenizer loading helpers.
- `src/grpo_reasoning/algorithms/`: GRPO algorithm utilities that can be tested without GPUs.
- `src/grpo_reasoning/training/`: trainer adapters and runtime setup.
- `scripts/`: runnable project entrypoints.
- `configs/`: reproducible JSON configs for model, evaluation, and training.

## Current Scope

The project currently supports a minimal GRPO smoke training path through TRL's
`GRPOTrainer`. This validates that data loading, reward computation, generation,
training configuration, and checkpoint writing are connected. It is not yet a final
experiment; final runs should increase sample count, max steps, logging, and
evaluation after each checkpoint.

