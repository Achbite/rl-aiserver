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

    if git -C "${repo_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1 &&
       ! git -C "${repo_dir}" cat-file -e "HEAD:${relative_path}" 2>/dev/null; then
        echo "Required source file is not committed in HEAD: ${relative_path}" >&2
        echo "A clean clone would be unable to build this repository." >&2
        exit 1
    fi
done
