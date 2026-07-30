#!/usr/bin/env bash

set -euo pipefail

export MAZE_WORKLOAD="${MAZE_WORKLOAD:-model-evaluation}"
exec /opt/rl/aiserver/scripts/entrypoint.sh
