#!/usr/bin/env bash

set -euo pipefail

action="${1:-shell}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
container_name="aiserver-dev"
network_name="rl-training-dev"
tag="${AISERVER_DEV_IMAGE_TAG:-test-001}"
dev_image="rl-training/aiserver-dev:${tag}"
ccache_volume="rl-training-aiserver-ccache"
ccache_dir="/var/cache/ccache"
cpp_dev_base_image="${AISERVER_DEV_BASE_IMAGE:-python:3.11-slim}"
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

build_image() {
    docker build \
        --file "${repo_dir}/Dockerfile.dev" \
        --build-arg "CPP_DEV_BASE_IMAGE=${cpp_dev_base_image}" \
        --build-arg "ONNXRUNTIME_VERSION=${onnxruntime_version}" \
        --label "org.rl-training.component=aiserver-dev" \
        --label "org.rl-training.dev-platform=${platform}" \
        --tag "${dev_image}" \
        "${repo_dir}"
}

ensure_dev_image() {
    echo "Building AIServer development image" >&2
    build_image
}

container_exists() {
    docker container inspect "${container_name}" >/dev/null 2>&1
}

container_running() {
    [ "$(docker inspect --format '{{.State.Running}}' "${container_name}")" = "true" ]
}

container_uses_current_image() {
    docker image inspect "${dev_image}" >/dev/null 2>&1 || return 1
    [ "$(docker inspect --format '{{.Image}}' "${container_name}")" = \
      "$(docker image inspect --format '{{.Id}}' "${dev_image}")" ]
}

container_has_legacy_contract_mount() {
    [ "$(docker inspect \
        --format '{{range .Mounts}}{{if eq .Destination "/workspace/dev-artifacts/rl-contracts"}}yes{{end}}{{end}}' \
        "${container_name}")" = "yes" ]
}

warn_container_drift() {
    if docker image inspect "${dev_image}" >/dev/null 2>&1 &&
       ! container_uses_current_image; then
        echo "aiserver-dev uses an older local image; run make dev-refresh when ready" >&2
    fi
    if container_has_legacy_contract_mount; then
        echo "aiserver-dev still has the retired Contracts mount; run make dev-refresh to remove it" >&2
    fi
}

container_has_business_processes() {
    local process_status
    container_running || return 1
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

ensure_container_resources() {
    if ! docker network inspect "${network_name}" >/dev/null 2>&1; then
        docker network create "${network_name}" >/dev/null
    fi
    if ! docker volume inspect "${ccache_volume}" >/dev/null 2>&1; then
        docker volume create "${ccache_volume}" >/dev/null
    fi
}

create_container() {
    docker run --detach \
        --name "${container_name}" \
        --network "${network_name}" \
        --network-alias "${container_name}" \
        --network-alias "maze-aiserver" \
        --env "CCACHE_DIR=${ccache_dir}" \
        --volume "${repo_dir}:/workspace/rl-aiserver" \
        --volume "${ccache_volume}:${ccache_dir}" \
        "${dev_image}" >/dev/null
}

ensure_container() {
    if container_exists; then
        if ! container_running; then
            docker start "${container_name}" >/dev/null
        fi
        warn_container_drift
        return
    fi

    ensure_dev_image
    ensure_container_resources
    create_container
}

refresh_container() {
    local process_state
    if container_exists && container_running; then
        process_state=0
        container_has_business_processes || process_state=$?
        if [ "${process_state}" -eq 0 ]; then
            echo "aiserver-dev has an active AIServer, test, or build process" >&2
            echo "Stop it before refreshing the development container" >&2
            exit 1
        elif [ "${process_state}" -ne 1 ]; then
            exit 1
        fi
    fi

    ensure_dev_image
    ensure_container_resources
    if container_exists; then
        if container_running; then
            docker stop --time 5 "${container_name}" >/dev/null
        fi
        docker rm "${container_name}" >/dev/null
    fi
    create_container
    echo "AIServer development container refreshed: ${container_name}"
}

case "${action}" in
    image)
        build_image
        ;;
    refresh)
        refresh_container
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
        if container_exists; then
            if container_running; then
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
