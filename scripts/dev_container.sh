#!/usr/bin/env bash

set -euo pipefail

action="${1:-shell}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
container_name="aiserver-dev"
network_name="rl-training-dev"
tag="${AISERVER_DEV_IMAGE_TAG:-test-001}"
runtime_image="rl-training/aiserver:${AISERVER_IMAGE_TAG:-test-001}"
dev_image="rl-training/aiserver-dev:${tag}"
ccache_volume="rl-training-aiserver-ccache"
ccache_dir="/var/cache/ccache"

build_image() {
    docker build \
        --file "${repo_dir}/Dockerfile.dev" \
        --build-arg "AISERVER_RUNTIME_IMAGE=${runtime_image}" \
        --tag "${dev_image}" \
        "${repo_dir}"
}

ensure_container() {
    if ! docker image inspect "${dev_image}" >/dev/null 2>&1; then
        build_image
    fi
    if ! docker network inspect "${network_name}" >/dev/null 2>&1; then
        docker network create "${network_name}" >/dev/null
    fi
    if ! docker volume inspect "${ccache_volume}" >/dev/null 2>&1; then
        docker volume create "${ccache_volume}" >/dev/null
    fi
    if docker container inspect "${container_name}" >/dev/null 2>&1; then
        expected_image_id="$(docker image inspect --format '{{.Id}}' "${dev_image}")"
        actual_image_id="$(docker inspect --format '{{.Image}}' "${container_name}")"
        cache_mount="$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/var/cache/ccache"}}{{.Name}}{{end}}{{end}}' "${container_name}")"
        if [ "${actual_image_id}" != "${expected_image_id}" ] || \
           [ "${cache_mount}" != "${ccache_volume}" ]; then
            docker rm --force "${container_name}" >/dev/null
        fi
    fi
    if ! docker container inspect "${container_name}" >/dev/null 2>&1; then
        docker run --detach \
            --name "${container_name}" \
            --network "${network_name}" \
            --network-alias "${container_name}" \
            --network-alias "maze-aiserver" \
            --env "CCACHE_DIR=${ccache_dir}" \
            --volume "${repo_dir}:/workspace/rl-aiserver" \
            --volume "${ccache_volume}:${ccache_dir}" \
            "${dev_image}" >/dev/null
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
            "cd /workspace/rl-aiserver && ./build.sh build"
        ;;
    test)
        ensure_container
        docker exec "${container_name}" sh -lc \
            "cd /workspace/rl-aiserver && ./build.sh test"
        ;;
    clean)
        if docker container inspect "${container_name}" >/dev/null 2>&1; then
            if [ "$(docker inspect --format '{{.State.Running}}' "${container_name}")" = "true" ]; then
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
