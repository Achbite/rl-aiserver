#!/usr/bin/env bash

set -euo pipefail

repo_dir="$1"
test_root="$(mktemp -d)"
trap 'rm -rf "${test_root}"' EXIT

fake_aiserver="${test_root}/maze_aiserver"
cat >"${fake_aiserver}" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$@" >"${FAKE_ARGS_FILE}"
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

MAZE_RUN_MODE=3 bash "${repo_dir}/run.sh" >/dev/null
assert_workload "model-evaluation"

bash "${repo_dir}/run.sh" astar-test >/dev/null
assert_workload "astar-test"

launcher_repo="${test_root}/launcher"
mkdir -p \
    "${launcher_repo}/configs" \
    "${launcher_repo}/models/local-train"
cp "${repo_dir}/run.sh" "${launcher_repo}/run.sh"
cp "${config}" "${launcher_repo}/configs/server_config.yaml"
printf '%s\n' "keep" >"${launcher_repo}/models/local-train/active-model"

if ! AISERVER_BIN="${fake_aiserver}" \
   AISERVER_CONFIG="${launcher_repo}/configs/server_config.yaml" \
   bash "${launcher_repo}/run.sh" training \
   --sample-distributor maze-learner:9100 \
   >"${test_root}/training.out" 2>&1; then
    echo "training launcher failed without a bundled SampleDistributor" >&2
    exit 1
fi
assert_workload "training"
if [ -e "${launcher_repo}/models/local-train/active-model" ]; then
    echo "training did not clear the AIServer local-train directory" >&2
    exit 1
fi
grep -qx -- "--sample-distributor" "${FAKE_ARGS_FILE}"
grep -qx -- "maze-learner:9100" "${FAKE_ARGS_FILE}"

outside_root="${test_root}/outside/local-train"
mkdir -p "${outside_root}"
printf '%s\n' "keep" >"${outside_root}/active-model"
if MAZE_LOCAL_TRAIN_ROOT="${outside_root}" \
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
