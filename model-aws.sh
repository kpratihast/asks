#!/usr/bin/env bash
set -euo pipefail
project="$(cd "$(dirname "$0")" && pwd)"
python="${ASKIISC_PYTHON:-$project/.venv/bin/python}"
if [[ ! -x "$python" ]]; then python=python3; fi
exec "$python" "$project/deploy/aws/model_profiles.py" \
  --config "$project/config/aws-models.json" --bundles "$project/.local/aws-releases" "$@"
