#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
BASELINE_LIMIT="${BASELINE_LIMIT:-10}"
GRPO_EXAMPLES="${GRPO_EXAMPLES:-8}"
GRPO_STEPS="${GRPO_STEPS:-1}"
BASELINE_OUTPUT="${BASELINE_OUTPUT:-data/processed/gsm8k_baseline_smoke.jsonl}"
GRPO_OUTPUT="${GRPO_OUTPUT:-outputs/grpo_smoke_qwen_0_5b}"

python scripts/inspect_gsm8k.py --split test --limit 3

python scripts/run_gsm8k_baseline.py \
  --model "${MODEL}" \
  --limit "${BASELINE_LIMIT}" \
  --batch-size 1 \
  --max-new-tokens 128 \
  --torch-dtype float16 \
  --output "${BASELINE_OUTPUT}" \
  --resume

python scripts/evaluate_predictions.py "${BASELINE_OUTPUT}"

python scripts/run_grpo_smoke.py \
  --model "${MODEL}" \
  --max-train-examples "${GRPO_EXAMPLES}" \
  --max-steps "${GRPO_STEPS}" \
  --output-dir "${GRPO_OUTPUT}"

