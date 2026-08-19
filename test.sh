#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/rl-aiserver-test.XXXXXX")"
trap 'rm -rf "${build_dir}"' EXIT

if [ "$#" -ne 0 ]; then
    echo "usage: bash ./test.sh" >&2
    exit 2
fi

source "${repo_dir}/artifact_versions.env"
bash "${repo_dir}/scripts/verify_source_inventory.sh"

contract_cpp_dir=""
if [ -n "${RL_CONTRACT_DEV_ARTIFACT_DIR:-}" ]; then
    python3 "${repo_dir}/scripts/verify_contract_snapshot.py" \
        "${RL_CONTRACT_DEV_ARTIFACT_DIR}" \
        "${RL_CONTRACTS_VERSION}" \
        "${RL_CONTRACTS_PLATFORM}" \
        --artifact-layout
    contract_cpp_dir="${RL_CONTRACT_DEV_ARTIFACT_DIR}/cpp"
else
    python3 "${repo_dir}/scripts/verify_contract_snapshot.py" \
        "${repo_dir}/proto" \
        "${RL_CONTRACTS_VERSION}" \
        "${RL_CONTRACTS_PLATFORM}"
fi

cmake_args=(
    -S "${repo_dir}"
    -B "${build_dir}"
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_TESTING=ON
)
if [ -n "${contract_cpp_dir}" ]; then
    cmake_args+=("-DCONTRACT_CPP_DIR=${contract_cpp_dir}")
fi
if command -v ccache >/dev/null 2>&1; then
    cmake_args+=("-DCMAKE_CXX_COMPILER_LAUNCHER=$(command -v ccache)")
fi

cmake "${cmake_args[@]}"
cmake --build "${build_dir}" --parallel
ctest --test-dir "${build_dir}" --output-on-failure
