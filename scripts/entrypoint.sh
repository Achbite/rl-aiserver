#!/usr/bin/env bash

set -euo pipefail

cd /opt/rl/aiserver
if [ -n "${MAZE_WORKLOAD:-}" ]; then
    exec ./run.sh "${MAZE_WORKLOAD}" --config configs/server_config.yaml
fi
if [ -n "${MAZE_RUN_MODE:-}" ]; then
    exec ./run.sh "${MAZE_RUN_MODE}" --config configs/server_config.yaml
fi
exec ./run.sh --config configs/server_config.yaml
