#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RVIZ_CONFIG="${SCRIPT_DIR}/rviz/apex_recognition_live.rviz"
MONITOR_SCRIPT="${SCRIPT_DIR}/tools/analysis/apex_recognition_live_monitor.py"

if [ ! -f "${RVIZ_CONFIG}" ]; then
  echo "[APEX] RViz config not found: ${RVIZ_CONFIG}" >&2
  exit 1
fi
if [ ! -f "${MONITOR_SCRIPT}" ]; then
  echo "[APEX] Live monitor script not found: ${MONITOR_SCRIPT}" >&2
  exit 1
fi

export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"
set +u
source /opt/ros/jazzy/setup.bash
set -u

unset ROS_STATIC_PEERS FASTRTPS_DEFAULT_PROFILES_FILE ROS_LOCALHOST_ONLY
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-30}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-SUBNET}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export APEX_RVIZ_SOFTWARE_GL="${APEX_RVIZ_SOFTWARE_GL:-1}"
if [ "${APEX_RVIZ_SOFTWARE_GL}" = "1" ]; then
  export LIBGL_ALWAYS_SOFTWARE=1
  export QT_XCB_FORCE_SOFTWARE_OPENGL=1
fi

echo "[APEX] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "[APEX] RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"
echo "[APEX] Probing live recognition topics..."
timeout 5 ros2 topic list | egrep "/apex/estimation/full_map_points|/apex/estimation/live_map_points|/apex/estimation/current_pose|/apex/estimation/path|/apex/odometry/imu_lidar_fused|/apex/planning/recognition_tour_local_path|/apex/planning/recognition_tour_route" || true

python3 "${MONITOR_SCRIPT}" &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT
echo "[APEX] Live monitor PID=${MONITOR_PID}"

rviz2 -d "${RVIZ_CONFIG}"
