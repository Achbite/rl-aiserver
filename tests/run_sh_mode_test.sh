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
assert_workload "inference-smoke"

MAZE_RUN_MODE=3 bash "${repo_dir}/run.sh" >/dev/null
assert_workload "model-evaluation"

bash "${repo_dir}/run.sh" astar-test >/dev/null
assert_workload "astar-test"
