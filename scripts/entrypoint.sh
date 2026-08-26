#!/usr/bin/env bash

set -euo pipefail

cd /opt/rl/aiserver
if [ -n "${RL_CONFIG_PATH:-}" ]; then
    if [ "$#" -ne 0 ] &&
       { [ "$#" -ne 2 ] || [ "$1" != "--config" ] ||
         [ "$2" != "configs/server_config.yaml" ]; }; then
        echo "managed AIServer accepts only RL_CONFIG_PATH" >&2
        exit 2
    fi
    exec ./run.sh --config "${RL_CONFIG_PATH}"
fi
exec ./run.sh "$@"
