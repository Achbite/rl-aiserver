#!/usr/bin/env bash

set -euo pipefail

image_tag="${RL_PROJECT_IMAGE_TAG:-maze-tag-001}"
image_name="rl-training/aiserver"

if [[ ! "${image_tag}" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]]; then
    echo "RL_PROJECT_IMAGE_TAG is not a valid Docker tag" >&2
    exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
image_ref="${image_name}:${image_tag}"
docker build --provenance=false \
    --label "org.rl-training.component=aiserver" \
    --label "org.rl-training.project-image-tag=${image_tag}" \
    --tag "${image_ref}" \
    "${repo_dir}"

printf '%s\n' "${image_ref}"
