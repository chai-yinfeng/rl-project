#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-experiments}"
ARCHIVE="${2:-results_$(date +%Y%m%d_%H%M%S).tar.gz}"

tar \
  --exclude='*/checkpoint-*' \
  --exclude='*/models/*/optimizer.pt' \
  --exclude='*/models/*/scheduler.pt' \
  -czf "${ARCHIVE}" \
  "${RUN_DIR}"

echo "${ARCHIVE}"
