#!/bin/bash
# Config wrapper for split server/client runs.
#
# Usage:
#   bash deploy/run.sh server <config>
#   bash deploy/run.sh client <config> "task" [duration_s]
set -eu
cd "$(dirname "$0")/.."
MODE="${1:?usage: run.sh server <config> | run.sh client <config> <task> [duration_s]}"
CONFIG="${2:?usage: run.sh server <config> | run.sh client <config> <task> [duration_s]}"

case "$MODE" in
  server)
    exec python3 -m deploy.server --config "$CONFIG"
    ;;
  client)
    TASK="${3:?usage: run.sh client <config> <task> [duration_s]}"
    ARGS=(--config "$CONFIG" --task="$TASK")
    if [ -n "${4-}" ]; then ARGS+=(--duration_s="$4"); fi
    exec python3 -m deploy.client "${ARGS[@]}"
    ;;
  *)
    echo "usage: run.sh server <config> | run.sh client <config> <task> [duration_s]" >&2
    exit 2
    ;;
esac
