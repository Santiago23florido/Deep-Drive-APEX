#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="${APEX_LOCAL_RECOGNITION_ROOT:-${SCRIPT_DIR}/apex_recognition_tour}"
RUN_ID="${1:-latest}"
RVIZ_CONFIG="${SCRIPT_DIR}/rviz/apex_manual_mapping_offline.rviz"
PUBLISHER_SCRIPT="${SCRIPT_DIR}/tools/sim/apex_general_map_publisher_up.sh"

if [[ "${RUN_ID}" == "latest" ]]; then
  RUN_ID="$(ls -1dt "${LOCAL_ROOT}"/* 2>/dev/null | head -n 1 | xargs -r basename)"
fi

if [[ -z "${RUN_ID}" ]]; then
  echo "[APEX][ERROR] No local recognition run found in ${LOCAL_ROOT}" >&2
  exit 1
fi

RUN_DIR="${LOCAL_ROOT%/}/${RUN_ID}"
FIXED_MAP_DIR="${RUN_DIR}/fixed_map"
FIXED_MAP_YAML="${FIXED_MAP_DIR}/fixed_map.yaml"
MAPPING_SUMMARY_JSON="${FIXED_MAP_DIR}/mapping_summary.json"

if [[ ! -f "${FIXED_MAP_YAML}" ]]; then
  echo "[APEX][ERROR] Missing fixed map yaml: ${FIXED_MAP_YAML}" >&2
  exit 1
fi
if [[ ! -f "${RVIZ_CONFIG}" ]]; then
  echo "[APEX][ERROR] Missing RViz config: ${RVIZ_CONFIG}" >&2
  exit 1
fi

export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"
set +u
source /opt/ros/jazzy/setup.bash
if [[ -f "${SCRIPT_DIR}/install/setup.bash" ]]; then
  source "${SCRIPT_DIR}/install/setup.bash"
fi
set -u

unset ROS_STATIC_PEERS FASTRTPS_DEFAULT_PROFILES_FILE ROS_LOCALHOST_ONLY
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-30}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-SUBNET}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export APEX_RVIZ_SOFTWARE_GL="${APEX_RVIZ_SOFTWARE_GL:-1}"
if [[ "${APEX_RVIZ_SOFTWARE_GL}" == "1" ]]; then
  export LIBGL_ALWAYS_SOFTWARE=1
  export QT_XCB_FORCE_SOFTWARE_OPENGL=1
fi

"${PUBLISHER_SCRIPT}" \
  "${FIXED_MAP_YAML}" \
  "${MAPPING_SUMMARY_JSON}" \
  "odom_imu_lidar_fused" \
  "/apex/sim/fixed_map/grid" \
  "/apex/sim/fixed_map/visual_points" \
  "/apex/sim/fixed_map/path" \
  "/apex/sim/fixed_map/status" \
  "false" \
  "false" &
PUBLISHER_PID=$!
trap 'kill "${PUBLISHER_PID}" 2>/dev/null || true' EXIT

echo "[APEX] Opening fixed map run: ${RUN_ID}"
exec rviz2 -d "${RVIZ_CONFIG}"
