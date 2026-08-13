#!/usr/bin/env bash

set -euo pipefail

repo_dir="$1"
test_root="$(mktemp -d)"
trap 'rm -rf "${test_root}"' EXIT

grep -Fq \
    "COPY proto/manifest.json /opt/rl/aiserver/proto/manifest.json" \
    "${repo_dir}/Dockerfile"
grep -Fq "COPY proto/schemas /opt/rl/aiserver/proto/schemas" \
    "${repo_dir}/Dockerfile"

fake_aiserver="${test_root}/maze_aiserver"
cat >"${fake_aiserver}" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$@" >"${FAKE_ARGS_FILE}"
if [ "${FAKE_HOLD_OPEN:-0}" = "1" ]; then
    trap 'exit "${FAKE_TERM_STATUS:-0}"' TERM
    while true; do
        sleep 0.05
    done
fi
SH
chmod +x "${fake_aiserver}"

config="${test_root}/server_config.yaml"
cat >"${config}" <<'YAML'
server:
  listen_port: 9002
  run_mode: 2
YAML

assert_workload() {
    local expected="$1"
    local actual
    actual="$(awk '$0 == "--workload" { getline; print; exit }' \
        "${FAKE_ARGS_FILE}")"
    if [ "${actual}" != "${expected}" ]; then
        echo "expected workload ${expected}, got ${actual}" >&2
        exit 1
    fi
}

export AISERVER_BIN="${fake_aiserver}"
export AISERVER_CONFIG="${config}"
export FAKE_ARGS_FILE="${test_root}/arguments"

bash "${repo_dir}/run.sh" >/dev/null
assert_workload "local-test"

RL_AISERVER_RUN_MODE=3 bash "${repo_dir}/run.sh" >/dev/null
assert_workload "model-evaluation"

bash "${repo_dir}/run.sh" map-validation >/dev/null
assert_workload "map-validation"

launcher_repo="${test_root}/launcher"
mkdir -p \
    "${launcher_repo}/configs" \
    "${launcher_repo}/models/local-train/cache/000000"
cp "${repo_dir}/run.sh" "${launcher_repo}/run.sh"
cp "${config}" "${launcher_repo}/configs/server_config.yaml"
printf '%s\n' "keep" >"${launcher_repo}/models/local-train/active-model"
printf '%s\n' "cached-model" >\
    "${launcher_repo}/models/local-train/cache/000000/SaveModel.onnx"
printf '%s\n' '{"model_version":"0"}' >\
    "${launcher_repo}/models/local-train/cache/000000/manifest.json"

if AISERVER_BIN="${fake_aiserver}" \
   AISERVER_CONFIG="${launcher_repo}/configs/server_config.yaml" \
   bash "${launcher_repo}/run.sh" training \
   --training-sample-budget 3072 \
   --sample-distributor maze-learner:9100 \
   >"${test_root}/retired-budget.out" 2>&1; then
    echo "training unexpectedly accepted a retired sample budget" >&2
    exit 1
fi
grep -q "training has no hard cap" "${test_root}/retired-budget.out"
test -f "${launcher_repo}/models/local-train/active-model"
test -f "${launcher_repo}/models/local-train/cache/000000/SaveModel.onnx"
test -f "${launcher_repo}/models/local-train/cache/000000/manifest.json"

if ! AISERVER_BIN="${fake_aiserver}" \
   AISERVER_CONFIG="${launcher_repo}/configs/server_config.yaml" \
   bash "${launcher_repo}/run.sh" training \
   --sample-distributor maze-learner:9100 \
   >"${test_root}/training.out" 2>&1; then
    echo "unbounded training launcher failed" >&2
    exit 1
fi
assert_workload "training"
test -f "${launcher_repo}/models/local-train/active-model"
test -d "${launcher_repo}/models/local-train/cache"
test -f "${launcher_repo}/models/local-train/cache/000000/SaveModel.onnx"
test -f "${launcher_repo}/models/local-train/cache/000000/manifest.json"
grep -qx -- "--sample-distributor" "${FAKE_ARGS_FILE}"
grep -qx -- "maze-learner:9100" "${FAKE_ARGS_FILE}"
grep -q 'return 125' "${repo_dir}/run.sh"
grep -q 'aiserver_child_status="${status}"' "${repo_dir}/run.sh"
if grep -q -- "--training-sample-budget" "${FAKE_ARGS_FILE}"; then
    echo "retired sample budget reached the AIServer process" >&2
    exit 1
fi

printf '%s\n' "remove-on-new-run" >\
    "${launcher_repo}/models/local-train/cache/000000/old-model"
if ! AISERVER_BIN="${fake_aiserver}" \
   AISERVER_CONFIG="${launcher_repo}/configs/server_config.yaml" \
   bash "${launcher_repo}/run.sh" training --new-run \
   --sample-distributor maze-learner:9100 \
   >"${test_root}/new-run.out" 2>&1; then
    echo "explicit AIServer new-run reset failed" >&2
    exit 1
fi
test ! -e "${launcher_repo}/models/local-train/active-model"
test ! -e "${launcher_repo}/models/local-train/cache/000000"
test -d "${launcher_repo}/models/local-train/cache"
grep -q "AIServer new Run reset" "${test_root}/new-run.out"
if grep -q -- "--new-run" "${FAKE_ARGS_FILE}"; then
    echo "new-run control flag reached the AIServer process" >&2
    exit 1
fi

printf '%s\n' "preserve" >\
    "${launcher_repo}/models/local-train/preserve-nontraining"
if AISERVER_BIN="${fake_aiserver}" \
   AISERVER_CONFIG="${launcher_repo}/configs/server_config.yaml" \
   bash "${launcher_repo}/run.sh" local-test --new-run \
   >"${test_root}/new-run-nontraining.out" 2>&1; then
    echo "non-training workload unexpectedly accepted --new-run" >&2
    exit 1
fi
grep -q "only valid for the training workload" \
    "${test_root}/new-run-nontraining.out"
test -f "${launcher_repo}/models/local-train/preserve-nontraining"

quiesce_launcher="${test_root}/quiesce-launcher"
mkdir -p \
    "${quiesce_launcher}/configs" \
    "${quiesce_launcher}/models/local-train"
cp "${repo_dir}/run.sh" "${quiesce_launcher}/run.sh"
cp "${config}" "${quiesce_launcher}/configs/server_config.yaml"
quiesce_success="${test_root}/quiesced"
quiesce_failure="${test_root}/quiesce-failed"
: >"${FAKE_ARGS_FILE}"
RL_AISERVER_QUIESCE_MARKER="${quiesce_success}" \
RL_AISERVER_QUIESCE_FAILURE_MARKER="${quiesce_failure}" \
FAKE_HOLD_OPEN=1 \
FAKE_TERM_STATUS=17 \
AISERVER_BIN="${fake_aiserver}" \
AISERVER_CONFIG="${quiesce_launcher}/configs/server_config.yaml" \
bash "${quiesce_launcher}/run.sh" training \
    >"${test_root}/quiesce.out" 2>&1 &
launcher_pid=$!
for _ in $(seq 1 100); do
    if [ -s "${FAKE_ARGS_FILE}" ]; then
        break
    fi
    sleep 0.02
done
kill -USR1 "${launcher_pid}"
for _ in $(seq 1 100); do
    if [ -e "${quiesce_failure}" ]; then
        break
    fi
    sleep 0.02
done
if [ -e "${quiesce_success}" ]; then
    echo "non-zero AIServer shutdown published quiesce success" >&2
    exit 1
fi
if [ "$(cat "${quiesce_failure}")" != "17" ]; then
    echo "AIServer quiesce failure did not preserve child status" >&2
    exit 1
fi
kill -TERM "${launcher_pid}"
wait "${launcher_pid}"

outside_root="${test_root}/outside/local-train"
mkdir -p "${outside_root}"
printf '%s\n' "keep" >"${outside_root}/active-model"
if RL_LOCAL_TRAIN_ROOT="${outside_root}" \
   AISERVER_BIN="${fake_aiserver}" \
   AISERVER_CONFIG="${launcher_repo}/configs/server_config.yaml" \
   bash "${launcher_repo}/run.sh" training \
   >"${test_root}/unsafe-path.out" 2>&1; then
    echo "training unexpectedly accepted an uncontrolled local-train path" >&2
    exit 1
fi
grep -q "Unsafe AIServer local-train path" "${test_root}/unsafe-path.out"
test -f "${outside_root}/active-model"

mkdir -p "${launcher_repo}/models/.aiserver-local-train.lock"
printf '%s\n' "keep" >"${launcher_repo}/models/local-train/active-model"
if AISERVER_BIN="${fake_aiserver}" \
   AISERVER_CONFIG="${launcher_repo}/configs/server_config.yaml" \
   bash "${launcher_repo}/run.sh" training \
   >"${test_root}/active-lock.out" 2>&1; then
    echo "training unexpectedly accepted an existing lock" >&2
    exit 1
fi
grep -q "training is already active" "${test_root}/active-lock.out"
test -f "${launcher_repo}/models/local-train/active-model"
