#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="${RL_TRAINING_WORKSPACE:-$(cd "${repo_dir}/.." && pwd)}"
source "${repo_dir}/artifact_versions.env"

: "${RL_TASK_CONTRACTS_VERSION:?RL_TASK_CONTRACTS_VERSION is required}"

artifact_dir="${workspace_root}/.workspace/artifacts/rl-contracts/${RL_TASK_CONTRACTS_VERSION}/task-maze"

python3 "${repo_dir}/scripts/sync_contract_snapshot.py" \
    --artifact-dir "${artifact_dir}" \
    --target-dir "${repo_dir}/proto" \
    --version "${RL_TASK_CONTRACTS_VERSION}"
