#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONTAINER_NAME="${APEX_CONTAINER_NAME:-apex_pipeline}"
NANO_PREFLIGHT_SCRIPT="${APEX_ROOT}/tools/ensure_nano33_stream.sh"
ROS_SETUP_SCRIPT="${APEX_ROS_SETUP_SCRIPT:-/opt/ros/jazzy/setup.bash}"
NANO_POSTCHECK_ENABLED="${APEX_NANO_POSTCHECK:-1}"
NANO_POSTCHECK_TOPICS="${APEX_NANO_POSTCHECK_TOPICS:-/apex/imu/data_raw,/apex/odometry/fusion_status,/apex/imu/angular_velocity/raw,/apex/kinematics/acceleration,/apex/kinematics/angular_velocity}"
NANO_POSTCHECK_REQUIRED_TOPICS="${APEX_NANO_POSTCHECK_REQUIRED_TOPICS:-/apex/imu/data_raw,/apex/kinematics/acceleration,/apex/kinematics/angular_velocity,/apex/odometry/fusion_status}"
NANO_POSTCHECK_TOPIC="${APEX_NANO_POSTCHECK_TOPIC:-}"
NANO_POSTCHECK_TIMEOUT_S="${APEX_NANO_POSTCHECK_TIMEOUT_S:-5}"
NANO_POSTCHECK_LOG_LINES="${APEX_NANO_POSTCHECK_LOG_LINES:-60}"
NANO_POSTCHECK_READY_DELAY_S="${APEX_NANO_POSTCHECK_READY_DELAY_S:-2}"

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
  postcheck_failed=0
  declare -A postcheck_topic_success=()
  echo "[APEX] Nano/ROS postcheck logs:"
  docker logs --tail "${NANO_POSTCHECK_LOG_LINES}" "${CONTAINER_NAME}" \
    | egrep 'nano_accel_serial_node|Connected to Arduino|First valid Arduino|No serial IMU' \
    || true

  sleep "${NANO_POSTCHECK_READY_DELAY_S}"

  IFS=',' read -r -a topic_list <<< "${NANO_POSTCHECK_TOPICS}"
  IFS=',' read -r -a required_topic_list <<< "${NANO_POSTCHECK_REQUIRED_TOPICS}"

  if [[ -n "${NANO_POSTCHECK_TOPIC}" ]]; then
    topic_list+=("${NANO_POSTCHECK_TOPIC}")
  fi

  sample_topic() {
    local topic="$1"
    docker exec "${CONTAINER_NAME}" /bin/bash -lc \
      "source '${ROS_SETUP_SCRIPT}' && timeout ${NANO_POSTCHECK_TIMEOUT_S}s ros2 topic echo '${topic}' --once"
  }

  for topic in "${topic_list[@]}"; do
    topic="$(echo "${topic}" | xargs)"
    [[ -z "${topic}" ]] && continue
    echo "[APEX] Nano/ROS postcheck topic sample: ${topic} (waiting up to ${NANO_POSTCHECK_TIMEOUT_S}s)"
    if sample_topic "${topic}"; then
      postcheck_topic_success["${topic}"]=1
    else
      postcheck_topic_success["${topic}"]=0
      echo "[APEX][WARN] Could not read ${topic} within ${NANO_POSTCHECK_TIMEOUT_S}s" >&2
    fi
  done

  for topic in "${required_topic_list[@]}"; do
    topic="$(echo "${topic}" | xargs)"
    [[ -z "${topic}" ]] && continue
    if [[ "${postcheck_topic_success["${topic}"]:-0}" == "1" ]]; then
      continue
    fi
    echo "[APEX] Nano/ROS postcheck retry: ${topic} (waiting up to ${NANO_POSTCHECK_TIMEOUT_S}s)"
    if sample_topic "${topic}" >/dev/null; then
      postcheck_topic_success["${topic}"]=1
      continue
    fi
    if [[ "${topic}" == "/apex/kinematics/angular_velocity" ]] && [[ "${postcheck_topic_success["/apex/imu/angular_velocity/raw"]:-0}" == "1" ]]; then
      echo "[APEX][WARN] Using raw gyro availability as fallback for ${topic}" >&2
      continue
    fi
    if [[ "${topic}" == "/apex/kinematics/acceleration" ]] && [[ "${postcheck_topic_success["/apex/imu/data_raw"]:-0}" == "1" ]]; then
      echo "[APEX][WARN] Using raw IMU availability as fallback for ${topic}" >&2
      continue
    fi
    if [[ "${topic}" == "/apex/imu/data_raw" ]] && [[ "${postcheck_topic_success["/apex/imu/angular_velocity/raw"]:-0}" == "1" ]]; then
      echo "[APEX][WARN] Using raw gyro availability as fallback for ${topic}" >&2
      continue
    fi
      echo "[APEX][ERROR] Required Nano/ROS topic missing: ${topic}" >&2
      postcheck_failed=1
  done

  if [[ "${postcheck_failed}" == "1" ]]; then
    echo "[APEX][ERROR] Nano/ROS postcheck failed; stopping core before any trial" >&2
    ./tools/apex_core_down.sh
    exit 1
  fi
fi
