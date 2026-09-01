#!/usr/bin/env bash

set -u

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
default_aiserver_bin="${repo_dir}/build/maze_aiserver"
if [ -x "${repo_dir}/bin/maze_aiserver" ]; then
    default_aiserver_bin="${repo_dir}/bin/maze_aiserver"
fi
aiserver_bin="${AISERVER_BIN:-${default_aiserver_bin}}"
managed=0
if [ "${RL_INFRA_MANAGED:-}" = "true" ]; then
    managed=1
    rm -f /run/rl/readiness.json /run/rl/aiserver-managed-ready
elif [ -n "${RL_INFRA_MANAGED:-}" ]; then
    echo "RL_INFRA_MANAGED must be exactly true when supplied" >&2
    exit 2
fi

runtime_arguments=("$@")
if [ "${managed}" -eq 1 ]; then
    required_platform_values=(
        RL_INFRA_ENDPOINT_AISERVER_TASK_PORT
        RL_INFRA_ENDPOINT_SAMPLE_POOL_HOST
        RL_INFRA_ENDPOINT_SAMPLE_POOL_PORT
        RL_INFRA_ENDPOINT_MODEL_DISTRIBUTOR_HOST
        RL_INFRA_ENDPOINT_MODEL_DISTRIBUTOR_PORT
        RL_INFRA_DATA_ROOT
        RL_INFRA_POD_ID
    )
    for name in "${required_platform_values[@]}"; do
        if [ -z "${!name:-}" ]; then
            echo "AIServer managed runtime fact is missing: ${name}" >&2
            exit 2
        fi
    done
    runtime_arguments+=(
        --workload training
        --listen-port "${RL_INFRA_ENDPOINT_AISERVER_TASK_PORT}"
        --sample-distributor "${RL_INFRA_ENDPOINT_SAMPLE_POOL_HOST}:${RL_INFRA_ENDPOINT_SAMPLE_POOL_PORT}"
        --model-distributor "${RL_INFRA_ENDPOINT_MODEL_DISTRIBUTOR_HOST}:${RL_INFRA_ENDPOINT_MODEL_DISTRIBUTOR_PORT}"
    )
fi

aiserver_pid=""
aiserver_child_status=""
stopping=0
quiesced=0
marker_scope="$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
quiesce_marker="${RL_AISERVER_QUIESCE_MARKER:-/tmp/rl-training-${marker_scope}-quiesced}"
quiesce_failure_marker="${RL_AISERVER_QUIESCE_FAILURE_MARKER:-/tmp/rl-training-${marker_scope}-quiesce-failed}"
quiesce_timeout_seconds="${RL_AISERVER_QUIESCE_TIMEOUT_SECONDS:-45}"
rm -f "${quiesce_marker}" "${quiesce_failure_marker}"
case "${quiesce_timeout_seconds}" in
    ''|*[!0-9]*)
        echo "RL_AISERVER_QUIESCE_TIMEOUT_SECONDS must be a positive integer" >&2
        exit 2
        ;;
esac
if [ "${quiesce_timeout_seconds}" -le 0 ]; then
    echo "RL_AISERVER_QUIESCE_TIMEOUT_SECONDS must be a positive integer" >&2
    exit 2
fi

terminate_process() {
    local pid="$1"
    local timeout_seconds="$2"
    if [ -z "${pid}" ]; then
        return 125
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
        wait "${pid}" 2>/dev/null
        return $?
    fi
    kill -TERM "${pid}" 2>/dev/null || true
    local waited=0
    while kill -0 "${pid}" 2>/dev/null &&
          [ "${waited}" -lt "${timeout_seconds}" ]; do
        sleep 1
        waited=$((waited + 1))
    done
    if kill -0 "${pid}" 2>/dev/null; then
        kill -KILL "${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null
}

shutdown() {
    if [ "${stopping}" -eq 1 ]; then
        return
    fi
    stopping=1
    terminate_process "${aiserver_pid}" "${quiesce_timeout_seconds}" || true
    aiserver_pid=""
    if [ "${managed}" -eq 1 ]; then
        rm -f /run/rl/readiness.json /run/rl/aiserver-managed-ready
    fi
}

quiesce() {
    if [ "${quiesced}" -eq 1 ] || [ "${stopping}" -eq 1 ]; then
        return
    fi
    quiesced=1
    local child_status="${aiserver_child_status}"
    if [ -n "${aiserver_pid}" ]; then
        if terminate_process "${aiserver_pid}" "${quiesce_timeout_seconds}"; then
            child_status=0
        else
            child_status=$?
        fi
    elif [ -z "${child_status}" ]; then
        child_status=125
    fi
    if [ -n "${aiserver_pid}" ]; then
        aiserver_child_status="${child_status}"
    fi
    aiserver_pid=""
    if [ "${child_status}" -eq 0 ]; then
        : > "${quiesce_marker}"
    else
        printf '%s\n' "${child_status}" > "${quiesce_failure_marker}"
    fi
}

trap quiesce USR1
trap shutdown EXIT TERM INT

if [ ! -x "${aiserver_bin}" ]; then
    echo "AIServer executable is missing: ${aiserver_bin}" >&2
    exit 1
fi

cd "${repo_dir}"

"${aiserver_bin}" "${runtime_arguments[@]}" &
aiserver_pid=$!

if [ "${managed}" -eq 1 ]; then
    managed_ready=0
    for _ in $(seq 1 3000); do
        if [ -s /run/rl/aiserver-managed-ready ]; then
            managed_ready=1
            break
        fi
        if ! kill -0 "${aiserver_pid}" 2>/dev/null; then
            break
        fi
        sleep 0.1
    done
    if [ "${managed_ready}" -ne 1 ]; then
        echo "AIServer managed readiness timeout" >&2
        exit 1
    fi
    marker_value() {
        awk -v key="$1" \
            'index($0, key "=") == 1 { print substr($0, length(key) + 2); exit }' \
            /run/rl/aiserver-managed-ready
    }
    metric_component="$(marker_value component)"
    metric_instance_id="$(marker_value instance_id)"
    metric_lifecycle_epoch="$(marker_value lifecycle_epoch)"
    metric_container_port="$(marker_value container_port)"
    metric_schema_id="$(marker_value schema_id)"
    metric_schema_version="$(marker_value schema_version)"
    metric_schema_digest="$(marker_value schema_digest)"
    contract_package="$(marker_value contract_package)"
    contract_version="$(marker_value contract_version)"
    contract_platform="$(marker_value contract_platform)"
    if [ "${metric_component}" != "rl-aiserver" ] ||
       [ "${metric_container_port}" != "9002" ] ||
       [ -z "${metric_instance_id}" ] ||
       [[ ! "${metric_lifecycle_epoch}" =~ ^[1-9][0-9]*$ ]] ||
       [ "${metric_schema_id}" != "maze.episode.metrics" ] ||
       [ "${metric_schema_version}" != "1" ] ||
       [[ ! "${metric_schema_digest}" =~ ^[0-9a-f]{64}$ ]] ||
       [ "${contract_package}" != "rl-contracts" ] ||
       [ -z "${contract_version}" ] ||
       [ -z "${contract_platform}" ]; then
        echo "AIServer managed metric source identity is invalid" >&2
        exit 1
    fi
    python3 scripts/publish_readiness.py \
        --component aiserver \
        --fact grpc=serving \
        --fact metric_service=serving \
        --fact metric_component="${metric_component}" \
        --fact metric_instance_id="${metric_instance_id}" \
        --fact metric_lifecycle_epoch="${metric_lifecycle_epoch}" \
        --fact metric_container_port="${metric_container_port}" \
        --fact metric_schema_id="${metric_schema_id}" \
        --fact metric_schema_version="${metric_schema_version}" \
        --fact metric_schema_digest="${metric_schema_digest}" \
        --fact contract_package="${contract_package}" \
        --fact contract_version="${contract_version}" \
        --fact contract_platform="${contract_platform}"
fi

while [ "${stopping}" -eq 0 ]; do
    if [ -n "${aiserver_pid}" ] &&
       ! kill -0 "${aiserver_pid}" 2>/dev/null; then
        if wait "${aiserver_pid}"; then
            status=0
        else
            status=$?
        fi
        aiserver_child_status="${status}"
        aiserver_pid=""
        shutdown
        exit "${status}"
    fi
    sleep 0.2
done

shutdown
exit 0
