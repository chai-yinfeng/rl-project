#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
LIMIT="${LIMIT:-100}"
OUTPUT="${OUTPUT:-data/processed/gsm8k_baseline_qwen_0_5b_${LIMIT}.jsonl}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

python scripts/run_gsm8k_baseline.py \
  --model "${MODEL}" \
  --limit "${LIMIT}" \
  --batch-size 1 \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --torch-dtype float16 \
  --output "${OUTPUT}" \
  --resume

python scripts/evaluate_predictions.py "${OUTPUT}"

