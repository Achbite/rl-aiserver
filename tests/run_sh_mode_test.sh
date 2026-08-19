#!/usr/bin/env bash

set -euo pipefail

repo_dir="$1"
test_root="$(mktemp -d)"
trap 'rm -rf "${test_root}"' EXIT

fake_aiserver="${test_root}/maze_aiserver"
cat >"${fake_aiserver}" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$@" >"${FAKE_ARGS_FILE}"
if [ "${FAKE_HOLD_OPEN:-0}" = "1" ]; then
    trap 'printf "stopped\n" >"${FAKE_STOP_FILE}"; exit 0' TERM INT
    while true; do sleep 0.05; done
fi
exit "${FAKE_EXIT_STATUS:-0}"
SH
chmod +x "${fake_aiserver}"

export AISERVER_BIN="${fake_aiserver}"
export FAKE_ARGS_FILE="${test_root}/arguments"

expected_arguments=(
    --config "${test_root}/server_config.yaml"
    --workload evaluation
    --evaluation-model "${test_root}/SaveModel.onnx"
    --listen-port 19002
)
bash "${repo_dir}/run.sh" "${expected_arguments[@]}" >/dev/null
printf '%s\n' "${expected_arguments[@]}" >"${test_root}/expected"
cmp -s "${FAKE_ARGS_FILE}" "${test_root}/expected"

if FAKE_EXIT_STATUS=23 bash "${repo_dir}/run.sh" --config invalid \
   >"${test_root}/child-exit.out" 2>&1; then
    echo "run.sh hid a non-zero child exit status" >&2
    exit 1
else
    child_status=$?
fi
[ "${child_status}" -eq 23 ]

export FAKE_HOLD_OPEN=1
export FAKE_STOP_FILE="${test_root}/stopped"
rm -f "${FAKE_ARGS_FILE}"
bash "${repo_dir}/run.sh" --workload training \
    >"${test_root}/stop.out" 2>&1 &
launcher_pid=$!
for _ in $(seq 1 100); do
    [ -s "${FAKE_ARGS_FILE}" ] && break
    sleep 0.02
done
kill -TERM "${launcher_pid}"
wait "${launcher_pid}"
[ "$(cat "${FAKE_STOP_FILE}")" = "stopped" ]
