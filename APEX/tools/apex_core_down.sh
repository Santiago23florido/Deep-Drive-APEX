#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONTAINER_NAME="${APEX_CONTAINER_NAME:-apex_pipeline}"
SESSION_SCRIPT="/work/ros2_ws/scripts/apex_recon_session.sh"

cd "${APEX_ROOT}"
if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  docker exec "${CONTAINER_NAME}" /bin/bash "${SESSION_SCRIPT}" stop >/dev/null 2>&1 || true
fi
docker stop -t 20 "${CONTAINER_NAME}" 2>/dev/null || true
docker compose -f docker/docker-compose.yml down --remove-orphans

echo "[APEX] ${CONTAINER_NAME} stopped"
