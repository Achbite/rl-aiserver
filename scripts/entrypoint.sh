#!/usr/bin/env bash

set -euo pipefail

cd /opt/rl/aiserver
exec ./run.sh "$@"
