#!/usr/bin/env bash

if [[ -z "${PYTHON_CMD:-}" ]]; then
  if command -v uv >/dev/null 2>&1; then
    PYTHON_CMD="uv run python"
  else
    PYTHON_CMD="python"
  fi
fi

