#!/usr/bin/env bash

set -euo pipefail

action="${1:-shell}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
workspace_root="${RL_TRAINING_WORKSPACE:-$(cd "${repo_dir}/.." && pwd -P)}"
container_name="aiserver-dev"
network_name="rl-training-dev"
tag="${AISERVER_DEV_IMAGE_TAG:-test-001}"
dev_image="rl-training/aiserver-dev:${tag}"
ccache_volume="rl-training-aiserver-ccache"
ccache_dir="/var/cache/ccache"
cpp_dev_base_image="${AISERVER_DEV_BASE_IMAGE:-python@sha256:b27df5841f3355e9473f9a516d38a6783b6c8dfeacaf2d14a240f443b368ddb6}"
onnxruntime_version="${AISERVER_DEV_ONNXRUNTIME_VERSION:-1.17.0}"

if [ -f "/.dockerenv" ]; then
    echo "make shell is a host-side Docker entrypoint; leave the component container first" >&2
    exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
    echo "make shell is a host-side Docker entrypoint; Docker CLI is unavailable" >&2
    exit 1
fi
if ! platform="$(docker version --format '{{.Server.Os}}/{{.Server.Arch}}' 2>/dev/null)" ||
   [ -z "${platform}" ]; then
    echo "make shell cannot reach the host Docker daemon" >&2
    exit 1
fi

dev_image_input_digest() {
    python3 - \
        "${repo_dir}/Dockerfile.dev" \
        "${repo_dir}/artifact_versions.env" \
        "${repo_dir}/scripts/dev_container.sh" \
        "${platform}" \
        "${cpp_dev_base_image}" \
        "${onnxruntime_version}" <<'PY'
import hashlib
import sys
from pathlib import Path

digest = hashlib.sha256()
for raw in sys.argv[1:4]:
    path = Path(raw)
    digest.update(path.name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")
for value in sys.argv[4:]:
    digest.update(value.encode("utf-8"))
    digest.update(b"\0")
print(digest.hexdigest())
PY
}

dev_input_digest="$(dev_image_input_digest)"

build_image() {
    docker build \
        --file "${repo_dir}/Dockerfile.dev" \
        --build-arg "CPP_DEV_BASE_IMAGE=${cpp_dev_base_image}" \
        --build-arg "ONNXRUNTIME_VERSION=${onnxruntime_version}" \
        --label "org.rl-training.component=aiserver-dev" \
        --label "org.rl-training.dev-input-digest=${dev_input_digest}" \
        --label "org.rl-training.dev-platform=${platform}" \
        --tag "${dev_image}" \
        "${repo_dir}"
}

ensure_dev_image() {
    local actual_digest=""
    if docker image inspect "${dev_image}" >/dev/null 2>&1; then
        actual_digest="$(
            docker image inspect \
                --format '{{index .Config.Labels "org.rl-training.dev-input-digest"}}' \
                "${dev_image}"
        )"
    fi
    if [ "${actual_digest}" != "${dev_input_digest}" ]; then
        echo "Building AIServer development image for input ${dev_input_digest:0:12}" >&2
        build_image
    fi
}

prepare_contract_artifact() {
    contract_dir="$(
        RL_TRAINING_WORKSPACE="${workspace_root}" \
            bash "${workspace_root}/rl-contracts/build_dev_artifact.sh"
    )"
    if [ ! -f "${contract_dir}/manifest.json" ] ||
       [ ! -f "${contract_dir}/cpp/training.pb.cc" ]; then
        echo "development Contracts artifact is incomplete: ${contract_dir}" >&2
        return 1
    fi
}

container_mount_source() {
    local destination="$1"
    docker inspect \
        --format "{{range .Mounts}}{{if eq .Destination \"${destination}\"}}{{.Source}}{{end}}{{end}}" \
        "${container_name}"
}

container_mount_name() {
    local destination="$1"
    docker inspect \
        --format "{{range .Mounts}}{{if eq .Destination \"${destination}\"}}{{.Name}}{{end}}{{end}}" \
        "${container_name}"
}

container_has_business_processes() {
    local process_status
    [ "$(docker inspect --format '{{.State.Running}}' "${container_name}")" = "true" ] ||
        return 1
    set +e
    docker exec "${container_name}" sh -lc \
        "pgrep -f '[m]aze_aiserver|[/]run.sh|[c]make --build|[c]test' >/dev/null"
    process_status=$?
    set -e
    if [ "${process_status}" -eq 0 ]; then
        return 0
    fi
    if [ "${process_status}" -eq 1 ]; then
        return 1
    fi
    echo "Unable to inspect aiserver-dev business processes" >&2
    return 2
}

container_matches_inputs() {
    [ "$(docker inspect --format '{{.Image}}' "${container_name}")" = \
      "$(docker image inspect --format '{{.Id}}' "${dev_image}")" ] &&
    [ "$(container_mount_name "/var/cache/ccache")" = "${ccache_volume}" ] &&
    [ "$(container_mount_source "/workspace/dev-artifacts/rl-contracts")" = \
      "${contract_dir}" ]
}

create_container() {
    docker run --detach \
        --name "${container_name}" \
        --network "${network_name}" \
        --network-alias "${container_name}" \
        --network-alias "maze-aiserver" \
        --env "CCACHE_DIR=${ccache_dir}" \
        --env "RL_CONTRACT_DEV_ARTIFACT_DIR=/workspace/dev-artifacts/rl-contracts" \
        --volume "${repo_dir}:/workspace/rl-aiserver" \
        --volume "${ccache_volume}:${ccache_dir}" \
        --volume "${contract_dir}:/workspace/dev-artifacts/rl-contracts:ro" \
        "${dev_image}" >/dev/null
}

ensure_container() {
    local process_state
    ensure_dev_image
    prepare_contract_artifact
    if ! docker network inspect "${network_name}" >/dev/null 2>&1; then
        docker network create "${network_name}" >/dev/null
    fi
    if ! docker volume inspect "${ccache_volume}" >/dev/null 2>&1; then
        docker volume create "${ccache_volume}" >/dev/null
    fi
    if docker container inspect "${container_name}" >/dev/null 2>&1 &&
       ! container_matches_inputs; then
        process_state=0
        container_has_business_processes || process_state=$?
        if [ "${process_state}" -eq 0 ]; then
            echo "aiserver-dev inputs changed while AIServer/build processes are active" >&2
            echo "Stop the active process before recreating aiserver-dev" >&2
            exit 1
        elif [ "${process_state}" -ne 1 ]; then
            exit 1
        fi
        echo "Recreating idle aiserver-dev for current development inputs" >&2
        docker rm --force "${container_name}" >/dev/null
    fi
    if ! docker container inspect "${container_name}" >/dev/null 2>&1; then
        create_container
    elif [ "$(docker inspect --format '{{.State.Running}}' "${container_name}")" != "true" ]; then
        docker start "${container_name}" >/dev/null
    fi
}

case "${action}" in
    image)
        build_image
        ;;
    shell)
        ensure_container
        exec docker exec -it "${container_name}" bash
        ;;
    build)
        ensure_container
        docker exec "${container_name}" sh -lc \
            "cd /workspace/rl-aiserver && ./build.sh"
        ;;
    clean)
        if docker container inspect "${container_name}" >/dev/null 2>&1; then
            if [ "$(docker inspect --format '{{.State.Running}}' "${container_name}")" = "true" ]; then
                process_state=0
                container_has_business_processes || process_state=$?
                if [ "${process_state}" -eq 0 ]; then
                    echo "aiserver-dev has active AIServer/build processes" >&2
                    echo "Stop the active process before make dev-clean" >&2
                    exit 1
                elif [ "${process_state}" -ne 1 ]; then
                    exit 1
                fi
                docker stop --time 5 "${container_name}" >/dev/null
            fi
            docker rm "${container_name}" >/dev/null
        fi
        ;;
    *)
        echo "unknown dev action: ${action}" >&2
        exit 2
        ;;
esac
