#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="${APEX_CONTAINER_NAME:-apex_pipeline}"

cleanup() {
  "${SCRIPT_DIR}/apex_core_down.sh" >/dev/null 2>&1 || true
}

trap cleanup INT TERM EXIT

"${SCRIPT_DIR}/apex_real_ready_up.sh"

while true; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "[APEX][service-runner] ${CONTAINER_NAME} stopped unexpectedly" >&2
    exit 1
  fi
  sleep 5
done
