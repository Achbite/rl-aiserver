#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
required_sources=(
    "src/log/logger.h"
)

for relative_path in "${required_sources[@]}"; do
    source_path="${repo_dir}/${relative_path}"
    if ! test -f "${source_path}"; then
        echo "Required source file is missing: ${source_path}" >&2
        exit 1
    fi

done
