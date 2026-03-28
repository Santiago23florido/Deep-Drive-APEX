#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONTAINER_NAME="${APEX_CONTAINER_NAME:-apex_pipeline}"
NANO_PREFLIGHT_SCRIPT="${APEX_ROOT}/tools/ensure_nano33_stream.sh"
ROS_SETUP_SCRIPT="${APEX_ROS_SETUP_SCRIPT:-/opt/ros/jazzy/setup.bash}"
NANO_POSTCHECK_ENABLED="${APEX_NANO_POSTCHECK:-1}"
NANO_POSTCHECK_TOPIC="${APEX_NANO_POSTCHECK_TOPIC:-/apex/imu/data_raw}"
NANO_POSTCHECK_TIMEOUT_S="${APEX_NANO_POSTCHECK_TIMEOUT_S:-8}"
NANO_POSTCHECK_LOG_LINES="${APEX_NANO_POSTCHECK_LOG_LINES:-60}"

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "[APEX] ${CONTAINER_NAME} is already running"
  exit 0
fi

cd "${APEX_ROOT}"
export APEX_ENABLE_RECON_MAPPING=0
export APEX_ENABLE_SLAM_TOOLBOX=0
export APEX_RECORD_DEBUG=0
export APEX_SKIP_BUILD="${APEX_SKIP_BUILD:-1}"

if [[ "${APEX_NANO_PREFLIGHT:-1}" == "1" ]]; then
  "${NANO_PREFLIGHT_SCRIPT}"
fi

./run_apex.sh -d

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "[APEX][ERROR] Failed to start ${CONTAINER_NAME}" >&2
  exit 1
fi

echo "[APEX] ${CONTAINER_NAME} is running in core mode"

if [[ "${NANO_POSTCHECK_ENABLED}" == "1" ]]; then
  echo "[APEX] Nano/ROS postcheck logs:"
  docker logs --tail "${NANO_POSTCHECK_LOG_LINES}" "${CONTAINER_NAME}" \
    | egrep 'nano_accel_serial_node|Connected to Arduino|First valid Arduino|No serial IMU' \
    || true
  echo "[APEX] Nano/ROS postcheck topic sample: ${NANO_POSTCHECK_TOPIC}"
  if ! docker exec "${CONTAINER_NAME}" /bin/bash -lc \
    "source '${ROS_SETUP_SCRIPT}' && timeout ${NANO_POSTCHECK_TIMEOUT_S}s ros2 topic echo '${NANO_POSTCHECK_TOPIC}' --once"; then
    echo "[APEX][WARN] Could not read ${NANO_POSTCHECK_TOPIC} within ${NANO_POSTCHECK_TIMEOUT_S}s" >&2
  fi
fi
