#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/rl-aiserver-test.XXXXXX")"
trap 'rm -rf "${build_dir}"' EXIT

if [ "$#" -ne 0 ]; then
    echo "usage: bash ./test.sh" >&2
    exit 2
fi

cmake_args=(
    -S "${repo_dir}"
    -B "${build_dir}"
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_TESTING=ON
)
if command -v ccache >/dev/null 2>&1; then
    cmake_args+=("-DCMAKE_CXX_COMPILER_LAUNCHER=$(command -v ccache)")
fi

cmake "${cmake_args[@]}"
cmake --build "${build_dir}" --parallel --target \
    aiserver_model_update_development_test \
    aiserver_gae_sample_delivery_development_test
ctest --test-dir "${build_dir}" --output-on-failure \
    -R '^(aiserver_model_update_data_path|aiserver_gae_sample_delivery_data_path)$'
