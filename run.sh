#!/usr/bin/env bash

set -u

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
default_aiserver_bin="${repo_dir}/build/maze_aiserver"
if [ -x "${repo_dir}/bin/maze_aiserver" ]; then
    default_aiserver_bin="${repo_dir}/bin/maze_aiserver"
fi
aiserver_bin="${AISERVER_BIN:-${default_aiserver_bin}}"

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

"${aiserver_bin}" "$@" &
aiserver_pid=$!

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
