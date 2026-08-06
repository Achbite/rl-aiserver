#!/usr/bin/env bash

set -euo pipefail

export RL_AISERVER_RUN_MODE="${RL_AISERVER_RUN_MODE:-model-evaluation}"
exec /opt/rl/aiserver/scripts/entrypoint.sh
