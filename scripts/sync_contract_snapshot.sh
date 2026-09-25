#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [ "$#" -ne 1 ]; then
    echo "usage: bash scripts/sync_contract_snapshot.sh ARTIFACT_DIR" >&2
    exit 2
fi
python3 "${repo_dir}/scripts/sync_contract_snapshot.py" \
    --artifact-dir "$1" --target-dir "${repo_dir}/proto" --profile task-maze
