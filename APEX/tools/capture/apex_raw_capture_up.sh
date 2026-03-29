#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTAINER_NAME="${APEX_CONTAINER_NAME:-apex_pipeline}"
NANO_PREFLIGHT_SCRIPT="${APEX_ROOT}/tools/firmware/ensure_nano33_stream.sh"
ROS_SETUP_SCRIPT="${APEX_ROS_SETUP_SCRIPT:-/opt/ros/jazzy/setup.bash}"
POSTCHECK_TIMEOUT_S="${APEX_RAW_POSTCHECK_TIMEOUT_S:-5}"
POSTCHECK_READY_DELAY_S="${APEX_RAW_POSTCHECK_READY_DELAY_S:-2}"

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "[APEX] ${CONTAINER_NAME} is already running"
  exit 0
fi

cd "${APEX_ROOT}"
export APEX_ENABLE_RECON_MAPPING=0
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

echo "[APEX] ${CONTAINER_NAME} is running in raw-capture mode"
sleep "${POSTCHECK_READY_DELAY_S}"

sample_topic() {
  local topic="$1"
  docker exec "${CONTAINER_NAME}" /bin/bash -lc \
    "source '${ROS_SETUP_SCRIPT}' && timeout ${POSTCHECK_TIMEOUT_S}s ros2 topic echo '${topic}' --once"
}

echo "[APEX] Raw postcheck topic sample: /apex/imu/data_raw"
sample_topic /apex/imu/data_raw || {
  echo "[APEX][ERROR] Missing /apex/imu/data_raw in raw-capture mode" >&2
  ./tools/core/apex_core_down.sh
  exit 1
}

echo "[APEX] Raw postcheck topic sample: /lidar/scan_localization"
sample_topic /lidar/scan_localization || {
  echo "[APEX][ERROR] Missing /lidar/scan_localization in raw-capture mode" >&2
  ./tools/core/apex_core_down.sh
  exit 1
}

echo "[APEX] Raw postcheck topic sample: /apex/odometry/imu_raw"
sample_topic /apex/odometry/imu_raw || {
  echo "[APEX][ERROR] Missing /apex/odometry/imu_raw in raw-capture mode" >&2
  ./tools/core/apex_core_down.sh
  exit 1
}
