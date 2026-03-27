#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONTAINER_NAME="${APEX_CONTAINER_NAME:-apex_pipeline}"
SESSION_SCRIPT="/work/ros2_ws/scripts/apex_recon_session.sh"

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "[APEX][ERROR] ${CONTAINER_NAME} is not running. Start it with tools/apex_core_up.sh" >&2
  exit 1
fi

ACTION="start"
if [ "${1:-}" = "--restart" ]; then
  ACTION="restart"
  shift
fi

cd "${APEX_ROOT}"
docker exec "${CONTAINER_NAME}" /bin/bash "${SESSION_SCRIPT}" "${ACTION}" "$@"
