#!/usr/bin/env bash

UV_RUN_FLAGS="${UV_RUN_FLAGS:-}"

if [[ -z "${PYTHON_CMD:-}" ]]; then
  if command -v uv >/dev/null 2>&1; then
    PYTHON_CMD="uv run ${UV_RUN_FLAGS} python"
  else
    PYTHON_CMD="python"
  fi
fi
