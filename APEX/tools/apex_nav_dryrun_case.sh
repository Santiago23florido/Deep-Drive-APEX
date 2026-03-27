#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <run_id> [timeout_s]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONTAINER_NAME="${APEX_CONTAINER_NAME:-apex_pipeline}"
SESSION_SCRIPT="/work/ros2_ws/scripts/apex_recon_session.sh"
RUN_ID="$1"
TIMEOUT_S="${2:-15}"
WAIT_SLACK_S="${APEX_RECON_WAIT_SLACK_S:-5}"

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "[APEX][ERROR] ${CONTAINER_NAME} is not running. Start it with tools/apex_core_up.sh" >&2
  exit 1
fi

cd "${APEX_ROOT}"
"${SCRIPT_DIR}/apex_recon_start.sh" \
  --run-id "${RUN_ID}" \
  --mode nav_dryrun \
  --record-debug 1 \
  --timeout-s "${TIMEOUT_S}"

deadline_s="$(python3 - "${TIMEOUT_S}" "${WAIT_SLACK_S}" <<'PY'
import sys

timeout_s = float(sys.argv[1])
slack_s = float(sys.argv[2])
print(int(timeout_s + slack_s + 2))
PY
)"

while [ "${deadline_s}" -gt 0 ]; do
  status_output="$(docker exec "${CONTAINER_NAME}" /bin/bash "${SESSION_SCRIPT}" status || true)"
  if ! printf '%s\n' "${status_output}" | grep -q "recon session running"; then
    break
  fi
  sleep 1
  deadline_s=$((deadline_s - 1))
done

status_output="$(docker exec "${CONTAINER_NAME}" /bin/bash "${SESSION_SCRIPT}" status || true)"
printf '%s\n' "${status_output}"

if [ -d "${APEX_ROOT}/ros2_ws/debug_runs" ]; then
  latest_bundle="$(ls -1t "${APEX_ROOT}/ros2_ws/debug_runs" 2>/dev/null | head -n 1 || true)"
  if [ -n "${latest_bundle}" ]; then
    echo "[APEX] Latest bundle: ${APEX_ROOT}/ros2_ws/debug_runs/${latest_bundle}"
  fi
fi
