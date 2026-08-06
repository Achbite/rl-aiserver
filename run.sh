#!/usr/bin/env bash

set -u

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
default_aiserver_bin="${repo_dir}/build/maze_aiserver"
if [ -x "${repo_dir}/bin/maze_aiserver" ]; then
    default_aiserver_bin="${repo_dir}/bin/maze_aiserver"
fi
aiserver_bin="${AISERVER_BIN:-${default_aiserver_bin}}"
aiserver_config="${AISERVER_CONFIG:-${repo_dir}/configs/server_config.yaml}"
local_train_root="${RL_LOCAL_TRAIN_ROOT:-${repo_dir}/models/local-train}"

canonical_workload() {
    case "$1" in
        1|train|training)
            printf '%s\n' "training"
            ;;
        2|local-test|inference-smoke)
            printf '%s\n' "local-test"
            ;;
        3|model-evaluation)
            printf '%s\n' "model-evaluation"
            ;;
        4|map-validation)
            printf '%s\n' "map-validation"
            ;;
        *)
            return 1
            ;;
    esac
}

read_config_mode() {
    awk '
        /^[^[:space:]#][^:]*:[[:space:]]*($|#)/ {
            section = $0
            sub(/:.*/, "", section)
            next
        }
        section == "server" &&
        /^[[:space:]]+run_mode:[[:space:]]*/ {
            value = $0
            sub(/^[^:]*:[[:space:]]*/, "", value)
            sub(/[[:space:]]*#.*/, "", value)
            gsub(/^[[:space:]"]+|[[:space:]"]+$/, "", value)
            print value
            exit
        }
    ' "$1"
}

runtime_arguments=("$@")
forward_arguments=()
workload_override=""
argument_index=0
while [ "${argument_index}" -lt "${#runtime_arguments[@]}" ]; do
    argument="${runtime_arguments[${argument_index}]}"
    case "${argument}" in
        --config)
            value_index=$((argument_index + 1))
            if [ "${value_index}" -ge "${#runtime_arguments[@]}" ] ||
               [ -z "${runtime_arguments[${value_index}]}" ]; then
                echo "--config requires a value" >&2
                exit 2
            fi
            aiserver_config="${runtime_arguments[${value_index}]}"
            argument_index=$((argument_index + 2))
            ;;
        --workload)
            value_index=$((argument_index + 1))
            if [ "${value_index}" -ge "${#runtime_arguments[@]}" ] ||
               [ -z "${runtime_arguments[${value_index}]}" ]; then
                echo "--workload requires a value" >&2
                exit 2
            fi
            workload_override="${runtime_arguments[${value_index}]}"
            argument_index=$((argument_index + 2))
            ;;
        --local-train-dir)
            value_index=$((argument_index + 1))
            if [ "${value_index}" -ge "${#runtime_arguments[@]}" ] ||
               [ -z "${runtime_arguments[${value_index}]}" ]; then
                echo "--local-train-dir requires a value" >&2
                exit 2
            fi
            local_train_root="${runtime_arguments[${value_index}]}"
            argument_index=$((argument_index + 2))
            ;;
        --sample-distributor)
            value_index=$((argument_index + 1))
            if [ "${value_index}" -ge "${#runtime_arguments[@]}" ]; then
                echo "--sample-distributor requires host:port" >&2
                exit 2
            fi
            address="${runtime_arguments[${value_index}]}"
            if [[ "${address}" != *:* ]]; then
                echo "--sample-distributor requires host:port" >&2
                exit 2
            fi
            sample_port="${address##*:}"
            if [[ ! "${sample_port}" =~ ^[0-9]+$ ]] ||
               [ "${sample_port}" -le 0 ] ||
               [ "${sample_port}" -gt 65535 ]; then
                echo "--sample-distributor requires a valid TCP port" >&2
                exit 2
            fi
            export RL_SAMPLE_DISTRIBUTOR_HOST="${address%:*}"
            export RL_SAMPLE_DISTRIBUTOR_PORT="${sample_port}"
            forward_arguments+=("${argument}" "${address}")
            argument_index=$((argument_index + 2))
            ;;
        --listen-port|--model-distributor|--local-test-model-dir|\
        --training-sample-budget|\
        --smoke-model-dir)
            value_index=$((argument_index + 1))
            if [ "${value_index}" -ge "${#runtime_arguments[@]}" ] ||
               [ -z "${runtime_arguments[${value_index}]}" ]; then
                echo "${argument} requires a value" >&2
                exit 2
            fi
            forward_arguments+=(
                "${argument}" "${runtime_arguments[${value_index}]}")
            argument_index=$((argument_index + 2))
            ;;
        --train)
            workload_override="training"
            argument_index=$((argument_index + 1))
            ;;
        *)
            if [ "${argument_index}" -eq 0 ] &&
               [ -z "${workload_override}" ]; then
                if candidate_workload="$(canonical_workload "${argument}")"; then
                    workload_override="${candidate_workload}"
                    argument_index=$((argument_index + 1))
                    continue
                fi
            fi
            forward_arguments+=("${argument}")
            argument_index=$((argument_index + 1))
            ;;
    esac
done

if [ ! -f "${aiserver_config}" ]; then
    echo "AIServer config is missing: ${aiserver_config}" >&2
    exit 1
fi

configured_mode="$(read_config_mode "${aiserver_config}")"
selected_mode="${workload_override}"
if [ -z "${selected_mode}" ]; then
    selected_mode="${RL_AISERVER_RUN_MODE:-}"
fi
if [ -z "${selected_mode}" ]; then
    selected_mode="${configured_mode}"
fi
if [ -z "${selected_mode}" ]; then
    echo "server.run_mode is required in ${aiserver_config}" >&2
    exit 2
fi
if ! workload="$(canonical_workload "${selected_mode}")"; then
    echo "unknown AIServer run mode: ${selected_mode}" >&2
    exit 2
fi
export RL_AISERVER_RUN_MODE="${workload}"
printf 'AIServer run mode: %s (%s)\n' "${selected_mode}" "${workload}"

aiserver_pid=""
stopping=0
quiesced=0
quiesce_marker="${RL_AISERVER_QUIESCE_MARKER:-/tmp/rl-training-quiesced}"
training_lock=""
rm -f "${quiesce_marker}"

terminate_process() {
    local pid="$1"
    local timeout_seconds="$2"
    if [ -z "${pid}" ] || ! kill -0 "${pid}" 2>/dev/null; then
        return
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
    wait "${pid}" 2>/dev/null || true
}

shutdown() {
    if [ "${stopping}" -eq 1 ]; then
        return
    fi
    stopping=1
    terminate_process "${aiserver_pid}" 15
    aiserver_pid=""
    if [ -n "${training_lock}" ]; then
        rm -rf -- "${training_lock}"
        training_lock=""
    fi
}

quiesce() {
    if [ "${quiesced}" -eq 1 ] || [ "${stopping}" -eq 1 ]; then
        return
    fi
    quiesced=1
    terminate_process "${aiserver_pid}" 15
    aiserver_pid=""
    : > "${quiesce_marker}"
}

trap quiesce USR1
trap shutdown EXIT TERM INT

case "${workload}" in
    local-test|model-evaluation|map-validation)
        ;;
    training)
        if [ "$(basename "${local_train_root}")" != "local-train" ]; then
            echo "AIServer local-train path must end with /local-train" >&2
            exit 1
        fi
        if [ -L "${local_train_root}" ]; then
            echo "AIServer local-train path must not be a symbolic link" >&2
            exit 1
        fi
        mkdir -p "$(dirname "${local_train_root}")"
        local_train_parent="$(
            cd "$(dirname "${local_train_root}")" && pwd -P
        )"
        local_train_root="${local_train_parent}/local-train"
        expected_local_train_root="${repo_dir}/models/local-train"
        if [ "${local_train_root}" != "${expected_local_train_root}" ]; then
            echo "Unsafe AIServer local-train path: ${local_train_root}" >&2
            exit 1
        fi
        training_lock="${RL_AISERVER_TRAIN_LOCK_DIR:-${local_train_parent}/.aiserver-local-train.lock}"
        if ! mkdir "${training_lock}" 2>/dev/null; then
            echo "AIServer training is already active or its lock remains: ${training_lock}" >&2
            exit 1
        fi
        printf '%s\n' "$$" > "${training_lock}/pid"
        if [ -d "${local_train_root}" ]; then
            find "${local_train_root}" -mindepth 1 -maxdepth 1 \
                -exec rm -rf -- {} +
        else
            mkdir -p "${local_train_root}"
        fi
        mkdir -p \
            "${local_train_root}/incoming" \
            "${local_train_root}/active" \
            "${local_train_root}/previous"
        export RL_LOCAL_TRAIN_ROOT="${local_train_root}"
        export RL_SAMPLE_DISTRIBUTOR_HOST="${RL_SAMPLE_DISTRIBUTOR_HOST:-maze-learner}"
        export RL_SAMPLE_DISTRIBUTOR_PORT="${RL_SAMPLE_DISTRIBUTOR_PORT:-9100}"
        ;;
    *)
        echo "unknown workload: ${workload}" >&2
        exit 2
        ;;
esac

if [ ! -x "${aiserver_bin}" ]; then
    echo "AIServer executable is missing: ${aiserver_bin}" >&2
    exit 1
fi

cd "${repo_dir}"

if [ "${#forward_arguments[@]}" -gt 0 ]; then
    "${aiserver_bin}" \
        --config "${aiserver_config}" \
        --workload "${workload}" \
        "${forward_arguments[@]}" &
else
    "${aiserver_bin}" \
        --config "${aiserver_config}" \
        --workload "${workload}" &
fi
aiserver_pid=$!

while [ "${stopping}" -eq 0 ]; do
    if [ -n "${aiserver_pid}" ] &&
       ! kill -0 "${aiserver_pid}" 2>/dev/null; then
        wait "${aiserver_pid}"
        status=$?
        aiserver_pid=""
        shutdown
        exit "${status}"
    fi
    sleep 0.2
done

shutdown
exit 0
