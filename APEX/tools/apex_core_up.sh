#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONTAINER_NAME="${APEX_CONTAINER_NAME:-apex_pipeline}"

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "[APEX] ${CONTAINER_NAME} is already running"
  exit 0
fi

cd "${APEX_ROOT}"
export APEX_ENABLE_RECON_MAPPING=0
export APEX_ENABLE_SLAM_TOOLBOX=0
export APEX_RECORD_DEBUG=0
export APEX_SKIP_BUILD="${APEX_SKIP_BUILD:-1}"

./run_apex.sh -d

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "[APEX][ERROR] Failed to start ${CONTAINER_NAME}" >&2
  exit 1
fi

echo "[APEX] ${CONTAINER_NAME} is running in core mode"
