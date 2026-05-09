#!/usr/bin/env bash
set -uo pipefail

targets=("$@")
if [ "${#targets[@]}" -eq 0 ]; then
  targets=(
    grpo-gs2-1_5b-all
    grpo-gs4-1_5b-all
    grpo-gs8-1_5b-all
  )
fi

log_root="${LOG_ROOT:-logs/overnight/$(date +%Y%m%d_%H%M%S)}"
keep_going="${KEEP_GOING:-1}"
mkdir -p "$log_root"

echo "Logging to $log_root"
printf "%s\n" "${targets[@]}" > "$log_root/targets.txt"

failed=()
for target in "${targets[@]}"; do
  started_at="$(date -Iseconds)"
  log_file="$log_root/${target}.log"
  echo "[$started_at] START $target"
  if make "$target" 2>&1 | tee "$log_file"; then
    echo "[$(date -Iseconds)] OK $target"
  else
    status="${PIPESTATUS[0]}"
    echo "[$(date -Iseconds)] FAILED $target status=$status"
    failed+=("$target")
    if [ "$keep_going" != "1" ]; then
      break
    fi
  fi
done

if [ "${#failed[@]}" -gt 0 ]; then
  printf "Failed targets:\n" | tee "$log_root/failed.txt"
  printf "%s\n" "${failed[@]}" | tee -a "$log_root/failed.txt"
  exit 1
fi

echo "All targets completed."
