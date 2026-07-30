#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${repo_dir}/build"
source "${repo_dir}/artifact_versions.env"

python3 "${repo_dir}/scripts/verify_contract_snapshot.py" \
    "${repo_dir}/proto" "${RL_CONTRACTS_VERSION}"

cmake -S "${repo_dir}" -B "${build_dir}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --parallel
ctest --test-dir "${build_dir}" --output-on-failure
