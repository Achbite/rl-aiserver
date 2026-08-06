#!/usr/bin/env bash

set -euo pipefail

cd /opt/rl/aiserver
if [ -n "${RL_AISERVER_RUN_MODE:-}" ]; then
    exec ./run.sh "${RL_AISERVER_RUN_MODE}" --config configs/server_config.yaml
fi
exec ./run.sh --config configs/server_config.yaml
