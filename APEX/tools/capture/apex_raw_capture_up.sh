#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTAINER_NAME="${APEX_CONTAINER_NAME:-apex_pipeline}"
NANO_PREFLIGHT_SCRIPT="${APEX_ROOT}/tools/firmware/ensure_nano33_stream.sh"
ROS_SETUP_SCRIPT="${APEX_ROS_SETUP_SCRIPT:-/opt/ros/jazzy/setup.bash}"
POSTCHECK_TIMEOUT_S="${APEX_RAW_POSTCHECK_TIMEOUT_S:-5}"
POSTCHECK_READY_DELAY_S="${APEX_RAW_POSTCHECK_READY_DELAY_S:-4}"
POSTCHECK_RETRIES="${APEX_RAW_POSTCHECK_RETRIES:-3}"
PREFLIGHT_SOFT_FAIL="${APEX_NANO_PREFLIGHT_SOFT_FAIL:-1}"
AUTO_DOWN_ON_POSTCHECK_FAIL="${APEX_RAW_POSTCHECK_AUTO_DOWN_ON_FAIL:-0}"

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "[APEX] ${CONTAINER_NAME} is already running"
  exit 0
fi

cd "${APEX_ROOT}"
export APEX_ENABLE_RECON_MAPPING=0
export APEX_RECORD_DEBUG=0
export APEX_SKIP_BUILD="${APEX_SKIP_BUILD:-1}"
export APEX_ENABLE_KINEMATICS="${APEX_ENABLE_KINEMATICS:-1}"

if [[ "${APEX_NANO_PREFLIGHT:-1}" == "1" ]]; then
  if ! "${NANO_PREFLIGHT_SCRIPT}"; then
    if [[ "${PREFLIGHT_SOFT_FAIL}" == "1" ]]; then
      echo "[APEX][WARN] Nano preflight failed, continuing to ROS postcheck because the serial node may recover it"
    else
      exit 1
    fi
  fi
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

require_topic() {
  local topic="$1"
  local attempt=1
  while [[ "${attempt}" -le "${POSTCHECK_RETRIES}" ]]; do
    echo "[APEX] Raw postcheck topic sample: ${topic} (attempt ${attempt}/${POSTCHECK_RETRIES})"
    if sample_topic "${topic}"; then
      return 0
    fi
    sleep 1
    attempt=$((attempt + 1))
  done
  return 1
}

postcheck_fail() {
  local message="$1"
  echo "[APEX][ERROR] ${message}" >&2
  echo "[APEX] apex_pipeline state after failure:" >&2
  docker ps -a --format '{{.Names}} {{.Status}}' | grep "^${CONTAINER_NAME} " >&2 || true
  if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "[APEX] Last ${CONTAINER_NAME} logs:" >&2
    docker logs --tail 120 "${CONTAINER_NAME}" >&2 || true
  fi
  if [[ "${AUTO_DOWN_ON_POSTCHECK_FAIL}" == "1" ]]; then
    ./tools/core/apex_core_down.sh
  else
    echo "[APEX][WARN] Keeping failed container for inspection. Stop it manually with ./tools/core/apex_core_down.sh" >&2
  fi
  exit 1
}

require_topic /apex/imu/data_raw || {
  postcheck_fail "Missing /apex/imu/data_raw in raw-capture mode"
}

require_topic /lidar/scan_localization || {
  postcheck_fail "Missing /lidar/scan_localization in raw-capture mode"
}

if [[ "${APEX_ENABLE_KINEMATICS}" == "1" ]]; then
  require_topic /apex/odometry/imu_raw || {
    postcheck_fail "Missing /apex/odometry/imu_raw in raw-capture mode"
  }
fi
