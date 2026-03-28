#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${1:-ensta@raspberrypi}"
RUN_ID="${2:-latest}"
REMOTE_ROOT="${3:-/home/ensta/AiAtonomousRc/APEX/ros2_ws/debug_runs}"
LOCAL_ROOT="${4:-$(pwd)/debug_runs}"

if [ "${RUN_ID}" = "latest" ]; then
  RUN_ID="$(ssh "${REMOTE_HOST}" "ls -1t '${REMOTE_ROOT}' | head -n 1")"
fi

if [ -z "${RUN_ID}" ]; then
  echo "No run_id resolved from ${REMOTE_HOST}:${REMOTE_ROOT}" >&2
  exit 1
fi

mkdir -p "${LOCAL_ROOT}"
echo "[APEX] Fetching ${RUN_ID} from ${REMOTE_HOST}:${REMOTE_ROOT}"
rsync -av "${REMOTE_HOST}:${REMOTE_ROOT%/}/${RUN_ID}/" "${LOCAL_ROOT%/}/${RUN_ID}/"
echo "[APEX] Local bundle: ${LOCAL_ROOT%/}/${RUN_ID}"
