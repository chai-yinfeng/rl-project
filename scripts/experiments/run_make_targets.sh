#!/usr/bin/env bash
set -u
set -o pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 TARGET [TARGET ...]" >&2
  exit 2
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
log_dir="logs/make_targets/${timestamp}"
mkdir -p "${log_dir}"

summary="${log_dir}/summary.tsv"
printf "target\tstatus\tstarted_at\tended_at\telapsed_seconds\tlog\n" > "${summary}"

overall_status=0
for target in "$@"; do
  started_epoch="$(date +%s)"
  started_at="$(date -Iseconds)"
  safe_target="$(printf "%s" "${target}" | tr -c 'A-Za-z0-9_.-' '_')"
  log_path="${log_dir}/${safe_target}.log"

  echo "[run_make_targets] starting ${target}"
  if PYTHONUNBUFFERED=1 make "${target}" 2>&1 | tee "${log_path}"; then
    status=0
  else
    status=$?
    if [ "${overall_status}" -eq 0 ]; then
      overall_status="${status}"
    fi
    echo "[run_make_targets] target failed: ${target} (status ${status})" >&2
  fi

  ended_epoch="$(date +%s)"
  ended_at="$(date -Iseconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${target}" "${status}" "${started_at}" "${ended_at}" \
    "$((ended_epoch - started_epoch))" "${log_path}" >> "${summary}"
done

echo "[run_make_targets] summary: ${summary}"
exit "${overall_status}"
