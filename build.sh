#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${repo_dir}/build"
action="${1:-build}"
source "${repo_dir}/artifact_versions.env"

case "${action}" in
    build|test|verify)
        ;;
    *)
        echo "usage: $0 [build|test|verify]" >&2
        exit 2
        ;;
esac

bash "${repo_dir}/scripts/verify_source_inventory.sh"

python3 "${repo_dir}/scripts/verify_contract_snapshot.py" \
    "${repo_dir}/proto" \
    "${RL_CONTRACTS_VERSION}" \
    "${RL_CONTRACTS_PLATFORM}"

cmake_args=(
    -S "${repo_dir}"
    -B "${build_dir}"
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
)
if command -v ccache >/dev/null 2>&1; then
    cmake_args+=("-DCMAKE_CXX_COMPILER_LAUNCHER=$(command -v ccache)")
fi
cmake "${cmake_args[@]}"

if [ "${action}" = "build" ]; then
    cmake --build "${build_dir}" --parallel --target maze_aiserver
    exit 0
fi

cmake --build "${build_dir}" --parallel
ctest --test-dir "${build_dir}" --output-on-failure
