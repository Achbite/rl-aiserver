#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="${RL_TRAINING_WORKSPACE:-$(cd "${repo_dir}/.." && pwd)}"
contracts_version="$(tr -d '[:space:]' < "${workspace_root}/rl-contracts/VERSION")"
artifact_dir="${workspace_root}/.workspace/artifacts/rl-contracts/${contracts_version}/training"

python3 "${repo_dir}/scripts/sync_contract_snapshot.py" \
    --artifact-dir "${artifact_dir}" \
    --target-dir "${repo_dir}/proto" \
    --profile training
