#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${1:-ensta@raspberrypi}"
RUN_ID="${2:-latest}"
REMOTE_ROOT="${3:-/home/ensta/AiAtonomousRc/APEX/ros2_ws/apex_rect_sensorfus}"
LOCAL_ROOT="${4:-$(pwd)/APEX/data/apex_rect_sensorfus}"

if [[ "${RUN_ID}" == "latest" ]]; then
  RUN_ID="$(ssh "${REMOTE_HOST}" "ls -1dt '${REMOTE_ROOT}'/* 2>/dev/null | head -n 1 | xargs -r basename")"
fi

if [[ -z "${RUN_ID}" ]]; then
  echo "[APEX][ERROR] No rect sensor capture found in ${REMOTE_HOST}:${REMOTE_ROOT}" >&2
  exit 1
fi

mkdir -p "${LOCAL_ROOT}"
echo "[APEX] Fetching rect sensor capture ${RUN_ID}"
rsync -av "${REMOTE_HOST}:${REMOTE_ROOT%/}/${RUN_ID}/" "${LOCAL_ROOT%/}/${RUN_ID}/"
echo "[APEX] Local capture: ${LOCAL_ROOT%/}/${RUN_ID}"
