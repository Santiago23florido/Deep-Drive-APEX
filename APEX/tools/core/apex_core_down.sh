#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTAINER_NAME="${APEX_CONTAINER_NAME:-apex_pipeline}"

cd "${APEX_ROOT}"
docker stop -t 20 "${CONTAINER_NAME}" 2>/dev/null || true
docker compose -f docker/docker-compose.yml down --remove-orphans
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

echo "[APEX] ${CONTAINER_NAME} stopped"
