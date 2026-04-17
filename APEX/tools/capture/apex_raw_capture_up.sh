#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTAINER_NAME="${APEX_CONTAINER_NAME:-apex_pipeline}"
NANO_PREFLIGHT_SCRIPT="${APEX_ROOT}/tools/firmware/ensure_nano33_stream.sh"
NANO_PROFILE_ENV_FILE="${APEX_NANO_PROFILE_ENV_FILE:-${APEX_ROOT}/.apex_runtime/nano_serial_profile.env}"
ROS_SETUP_SCRIPT="${APEX_ROS_SETUP_SCRIPT:-/opt/ros/jazzy/setup.bash}"
POSTCHECK_TIMEOUT_S="${APEX_RAW_POSTCHECK_TIMEOUT_S:-5}"
POSTCHECK_READY_DELAY_S="${APEX_RAW_POSTCHECK_READY_DELAY_S:-4}"
POSTCHECK_RETRIES="${APEX_RAW_POSTCHECK_RETRIES:-3}"
POSTCHECK_TOPIC_GAP_S="${APEX_RAW_POSTCHECK_TOPIC_GAP_S:-0.5}"
PREFLIGHT_SOFT_FAIL="${APEX_NANO_PREFLIGHT_SOFT_FAIL:-1}"
AUTO_DOWN_ON_POSTCHECK_FAIL="${APEX_RAW_POSTCHECK_AUTO_DOWN_ON_FAIL:-0}"
STARTUP_COMPAT="${APEX_STARTUP_COMPAT:-modern}"

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "[APEX] ${CONTAINER_NAME} is already running"
  exit 0
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "[APEX][WARN] Removing stale container ${CONTAINER_NAME} before startup"
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

cd "${APEX_ROOT}"
export APEX_ENABLE_RECON_MAPPING=0
export APEX_RECORD_DEBUG=0
export APEX_SKIP_BUILD="${APEX_SKIP_BUILD:-1}"
export APEX_ENABLE_KINEMATICS="${APEX_ENABLE_KINEMATICS:-1}"
export APEX_ENABLE_IMU_LIDAR_FUSION="${APEX_ENABLE_IMU_LIDAR_FUSION:-0}"
export APEX_ENABLE_CURVE_ENTRY_PLANNER="${APEX_ENABLE_CURVE_ENTRY_PLANNER:-0}"
export APEX_ENABLE_PATH_TRACKER="${APEX_ENABLE_PATH_TRACKER:-0}"
export APEX_ENABLE_RECOGNITION_TOUR_PLANNER="${APEX_ENABLE_RECOGNITION_TOUR_PLANNER:-0}"
export APEX_ENABLE_RECOGNITION_TOUR_TRACKER="${APEX_ENABLE_RECOGNITION_TOUR_TRACKER:-0}"
export APEX_ENABLE_CMDVEL_ACTUATION_BRIDGE="${APEX_ENABLE_CMDVEL_ACTUATION_BRIDGE:-0}"
export APEX_ENABLE_MANUAL_CONTROL_BRIDGE="${APEX_ENABLE_MANUAL_CONTROL_BRIDGE:-0}"
export APEX_ENABLE_RECOGNITION_SESSION_MANAGER="${APEX_ENABLE_RECOGNITION_SESSION_MANAGER:-0}"

if [[ "${STARTUP_COMPAT}" == "safe" ]]; then
  POSTCHECK_READY_DELAY_S="${APEX_RAW_POSTCHECK_READY_DELAY_S:-8}"
  POSTCHECK_RETRIES="${APEX_RAW_POSTCHECK_RETRIES:-2}"
  POSTCHECK_TOPIC_GAP_S="${APEX_RAW_POSTCHECK_TOPIC_GAP_S:-1.0}"
fi

if [[ "${APEX_NANO_PREFLIGHT:-1}" == "1" ]]; then
  if ! "${NANO_PREFLIGHT_SCRIPT}"; then
    if [[ "${PREFLIGHT_SOFT_FAIL}" == "1" ]]; then
      echo "[APEX][WARN] Nano preflight failed, continuing to ROS postcheck because the serial node may recover it"
    else
      exit 1
    fi
  fi
fi

echo "[APEX] Startup compat: ${STARTUP_COMPAT}"
if [[ "${STARTUP_COMPAT}" != "legacy" && -f "${NANO_PROFILE_ENV_FILE}" ]]; then
  # Reuse the serial-open profile that actually produced bytes during host-side
  # preflight so the ROS node does not reopen /dev/ttyACM0 with a mismatched policy.
  # shellcheck disable=SC1090
  source "${NANO_PROFILE_ENV_FILE}"
  echo "[APEX] Loaded Nano runtime profile from ${NANO_PROFILE_ENV_FILE}: ${APEX_SERIAL_CONNECT_PROFILE_NAME:-unknown}"
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
      if python3 - "${POSTCHECK_TOPIC_GAP_S}" <<'PY'
import sys
try:
    value = float(sys.argv[1])
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if value > 1.0e-6 else 1)
PY
      then
        sleep "${POSTCHECK_TOPIC_GAP_S}"
      fi
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

if [[ "${APEX_ENABLE_IMU_LIDAR_FUSION}" == "1" ]]; then
  require_topic /apex/estimation/status || {
    postcheck_fail "Missing /apex/estimation/status in raw-capture mode"
  }
fi

if [[ "${APEX_ENABLE_CURVE_ENTRY_PLANNER}" == "1" ]]; then
  require_topic /apex/planning/curve_entry_status || {
    postcheck_fail "Missing /apex/planning/curve_entry_status in raw-capture mode"
  }
fi

if [[ "${APEX_ENABLE_PATH_TRACKER}" == "1" ]]; then
  require_topic /apex/tracking/status || {
    postcheck_fail "Missing /apex/tracking/status in raw-capture mode"
  }
fi

if [[ "${APEX_ENABLE_RECOGNITION_TOUR_PLANNER}" == "1" ]]; then
  require_topic /apex/planning/recognition_tour_status || {
    postcheck_fail "Missing /apex/planning/recognition_tour_status in raw-capture mode"
  }
fi

if [[ "${APEX_ENABLE_RECOGNITION_TOUR_TRACKER}" == "1" ]]; then
  require_topic /apex/tracking/recognition_tour_status || {
    postcheck_fail "Missing /apex/tracking/recognition_tour_status in raw-capture mode"
  }
fi

if [[ "${APEX_ENABLE_CMDVEL_ACTUATION_BRIDGE}" == "1" ]]; then
  require_topic /apex/vehicle/drive_bridge_status || {
    postcheck_fail "Missing /apex/vehicle/drive_bridge_status in raw-capture mode"
  }
fi

if [[ "${APEX_ENABLE_MANUAL_CONTROL_BRIDGE}" == "1" ]]; then
  require_topic /apex/manual_control/status || {
    postcheck_fail "Missing /apex/manual_control/status in real-ready mode"
  }
fi

if [[ "${APEX_ENABLE_RECOGNITION_SESSION_MANAGER}" == "1" ]]; then
  require_topic /apex/recognition_session/status || {
    postcheck_fail "Missing /apex/recognition_session/status in real-ready mode"
  }
fi
