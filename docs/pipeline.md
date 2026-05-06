# Pipeline

## Entry Design

The user-facing entrypoint is the Makefile. It calls:

```text
scripts/run_experiment.py
```

The stage scripts are implementation details:

- `scripts/build_dpo_pairs.py`
- `scripts/run_dpo_train.py`
- `scripts/run_grpo_train.py`
- `scripts/run_ppo_train.py`
- `scripts/evaluate_model.py`
- `scripts/package_results.sh`

## Source Layout

The Python package is named `reasoning_post_training` to reflect the full project
scope rather than one algorithm.

```text
src/reasoning_post_training/
  datasets/       # GSM8K loading and prompt formatting
  evaluation/     # answer extraction and exact-match metrics
  rewards/        # reusable rule rewards
  models/         # model/tokenizer loading helpers
  methods/        # baseline, DPO, GRPO, PPO-style method code
  experiments.py  # run directory, JSON, and CUDA metric helpers
  runtime.py      # reproducibility helpers
```

Method code is split by algorithm:

```text
methods/baseline.py       # inference and model evaluation
methods/dpo.py            # pair scoring, pair loading, DPOConfig adapter
methods/grpo.py           # GRPO dataset/reward/config adapter
methods/grpo_algorithm.py # standalone group-relative advantage utility
methods/ppo.py            # compact PPO-style rule-reward trainer
```

## Stages

The unified launcher supports these primitive stages:

- `baseline`
- `dpo-pairs`
- `dpo-train`
- `dpo-eval`
- `grpo-train`
- `grpo-eval`
- `ppo-train`
- `ppo-eval`

It also supports grouped stages:

- `dpo`: pairs, train, eval
- `dpo-run`: pairs, train
- `grpo`: train, eval
- `grpo-run`: train
- `ppo`: train, eval
- `ppo-run`: train
- `run`: baseline plus all training stages
- `eval`: baseline plus all method eval stages
- `all`: complete run and eval

## Model Configs

The maintained experiment configs are:

```text
configs/experiments/gsm8k_qwen0_5b_test.json
configs/experiments/gsm8k_qwen0_5b_full.json
configs/experiments/gsm8k_qwen1_5b_test.json
configs/experiments/gsm8k_qwen1_5b_full.json
```

The test configs are small enough to validate that the server environment,
generation, training, evaluation, and logging paths work. The full configs are the
course-project runs.

## Artifacts

Each run writes:

```text
experiments/<run_name>/
  config.json
  configs/
    dpo_train.json
    grpo_train.json
    ppo_train.json
  data/
    baseline_predictions.jsonl
    dpo_pairs.jsonl
    dpo_predictions.jsonl
    grpo_predictions.jsonl
    ppo_predictions.jsonl
  logs/
  metrics/
    baseline.json
    dpo.json
    grpo.json
    ppo.json
  models/
    dpo/
    grpo/
    ppo/
```
