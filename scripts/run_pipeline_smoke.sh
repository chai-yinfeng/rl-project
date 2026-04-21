#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
BASELINE_LIMIT="${BASELINE_LIMIT:-10}"
BASELINE_MAX_NEW_TOKENS="${BASELINE_MAX_NEW_TOKENS:-512}"
GRPO_EXAMPLES="${GRPO_EXAMPLES:-8}"
GRPO_STEPS="${GRPO_STEPS:-1}"
BASELINE_OUTPUT="${BASELINE_OUTPUT:-data/processed/gsm8k_baseline_smoke.jsonl}"
GRPO_OUTPUT="${GRPO_OUTPUT:-outputs/grpo_smoke_qwen_0_5b}"

${PYTHON_CMD} scripts/inspect_gsm8k.py --split test --limit 3

${PYTHON_CMD} scripts/run_gsm8k_baseline.py \
  --model "${MODEL}" \
  --limit "${BASELINE_LIMIT}" \
  --batch-size 1 \
  --max-new-tokens "${BASELINE_MAX_NEW_TOKENS}" \
  --torch-dtype float16 \
  --output "${BASELINE_OUTPUT}" \
  --resume

${PYTHON_CMD} scripts/evaluate_predictions.py "${BASELINE_OUTPUT}"

${PYTHON_CMD} scripts/run_grpo_smoke.py \
  --model "${MODEL}" \
  --max-train-examples "${GRPO_EXAMPLES}" \
  --max-steps "${GRPO_STEPS}" \
  --output-dir "${GRPO_OUTPUT}"
