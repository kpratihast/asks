#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -z "${ASKIISC_PYTHON:-}" && -x .venv/bin/python ]]; then
  ASKIISC_PYTHON="$PWD/.venv/bin/python"
elif [[ -z "${ASKIISC_PYTHON:-}" && -f .local/python ]]; then
  ASKIISC_PYTHON="$(cat .local/python)"
fi
exec "${ASKIISC_PYTHON:-python3}" tools/ingest_document.py "$@"
