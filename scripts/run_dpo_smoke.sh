#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
LIMIT="${LIMIT:-20}"
NUM_COMPLETIONS="${NUM_COMPLETIONS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
PAIRS_OUTPUT="${PAIRS_OUTPUT:-data/processed/gsm8k_dpo_pairs_smoke.jsonl}"
DPO_CONFIG="${DPO_CONFIG:-configs/train/dpo_small.json}"
DPO_OUTPUT="${DPO_OUTPUT:-outputs/dpo_smoke_qwen_0_5b}"
DPO_MAX_STEPS="${DPO_MAX_STEPS:-1}"

${PYTHON_CMD} scripts/build_dpo_pairs.py \
  --model "${MODEL}" \
  --limit "${LIMIT}" \
  --num-completions "${NUM_COMPLETIONS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature 0.7 \
  --top-p 0.95 \
  --torch-dtype float16 \
  --output "${PAIRS_OUTPUT}" \
  --resume

${PYTHON_CMD} scripts/run_dpo_train.py \
  --config "${DPO_CONFIG}" \
  --model "${MODEL}" \
  --train-file "${PAIRS_OUTPUT}" \
  --max-steps "${DPO_MAX_STEPS}" \
  --output-dir "${DPO_OUTPUT}"
