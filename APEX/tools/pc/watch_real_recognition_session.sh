#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REMOTE_HOST="${APEX_REMOTE_HOST:-ensta@raspberrypi}"
REMOTE_STATUS_PATH="${APEX_REMOTE_STATUS_PATH:-/home/ensta/AiAtonomousRc/APEX/.apex_runtime/real_session/status.json}"
REMOTE_RUN_ROOT="${APEX_REMOTE_RECOGNITION_ROOT:-/home/ensta/AiAtonomousRc/APEX/ros2_ws/apex_recognition_tour}"
LOCAL_RUN_ROOT="${APEX_LOCAL_RECOGNITION_ROOT:-${APEX_ROOT}/apex_recognition_tour}"
POLL_S="${APEX_REAL_SESSION_POLL_S:-2}"
STATE_DIR="${APEX_ROOT}/.apex_runtime/pc_real_session"
STATE_FILE="${STATE_DIR}/last_fetched_run_id.txt"

mkdir -p "${STATE_DIR}" "${LOCAL_RUN_ROOT}"
LAST_FETCHED_RUN_ID="$(cat "${STATE_FILE}" 2>/dev/null || true)"

echo "[APEX][watch] remote host: ${REMOTE_HOST}"
echo "[APEX][watch] remote status: ${REMOTE_STATUS_PATH}"
echo "[APEX][watch] local root: ${LOCAL_RUN_ROOT}"

while true; do
  STATUS_JSON="$(ssh "${REMOTE_HOST}" "cat '${REMOTE_STATUS_PATH}' 2>/dev/null || true")"
  if [[ -n "${STATUS_JSON}" ]]; then
    COMPLETED_RUN_ID="$(python3 -c 'import json,sys
try:
    payload=json.loads(sys.stdin.read())
except Exception:
    payload={}
print(payload.get("completed_run_id",""))' <<< "${STATUS_JSON}")"
    SESSION_STATE="$(python3 -c 'import json,sys
try:
    payload=json.loads(sys.stdin.read())
except Exception:
    payload={}
print(payload.get("state",""))' <<< "${STATUS_JSON}")"

    if [[ -n "${COMPLETED_RUN_ID}" && "${COMPLETED_RUN_ID}" != "${LAST_FETCHED_RUN_ID}" ]]; then
      echo "[APEX][watch] fetching completed run ${COMPLETED_RUN_ID} (state=${SESSION_STATE})"
      if "${APEX_ROOT}/tools/capture/fetch_recognition_tour_capture.sh" \
        "${REMOTE_HOST}" \
        "${COMPLETED_RUN_ID}" \
        "${REMOTE_RUN_ROOT}" \
        "${LOCAL_RUN_ROOT}"; then
        LAST_FETCHED_RUN_ID="${COMPLETED_RUN_ID}"
        printf '%s\n' "${LAST_FETCHED_RUN_ID}" > "${STATE_FILE}"
      else
        echo "[APEX][watch][WARN] fetch failed for ${COMPLETED_RUN_ID}; will retry on next poll" >&2
      fi
    fi
  fi
  sleep "${POLL_S}"
done
