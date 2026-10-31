#!/usr/bin/env bash
set -euo pipefail
project="$(cd "$(dirname "$0")" && pwd)"
python="${ASKIISC_PYTHON:-$project/.venv/bin/python}"
if [[ ! -x "$python" ]]; then python=python3; fi
exec "$python" "$project/deploy/aws/deploy_latest.py" \
  --root "$project" --bundles "$project/.local/aws-releases" "$@"
