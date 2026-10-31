#!/usr/bin/env bash
# Manage the local vLLM Metal server; the chat UI uses the stable askiisc alias.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${ASKIISC_PYTHON:-}" ]]; then
  APP_PYTHON="$ASKIISC_PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  APP_PYTHON="$ROOT/.venv/bin/python"
elif [[ -f "$ROOT/.local/python" ]]; then
  APP_PYTHON="$(cat "$ROOT/.local/python")"
else
  APP_PYTHON=python3
fi
exec "$APP_PYTHON" "$ROOT/tools/model.py" "$@"
