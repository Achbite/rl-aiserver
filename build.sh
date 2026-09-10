#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${repo_dir}/build"

if [ "$#" -ne 0 ]; then
    echo "usage: $0" >&2
    echo "tests are run only through: bash ./test.sh" >&2
    exit 2
fi

cmake_args=(
    -S "${repo_dir}"
    -B "${build_dir}"
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_TESTING=OFF
)
if command -v ccache >/dev/null 2>&1; then
    cmake_args+=("-DCMAKE_CXX_COMPILER_LAUNCHER=$(command -v ccache)")
fi
cmake "${cmake_args[@]}"
cmake --build "${build_dir}" --parallel --target rl_aiserver
