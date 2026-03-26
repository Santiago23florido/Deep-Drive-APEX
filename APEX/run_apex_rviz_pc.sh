#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RVIZ_CONFIG="${SCRIPT_DIR}/rviz/apex_slam_auto.rviz"

if [ ! -f "${RVIZ_CONFIG}" ]; then
  echo "[APEX] RViz config not found: ${RVIZ_CONFIG}" >&2
  exit 1
fi

# Prevent unbound-variable failures inside ROS setup scripts.
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"
# ROS setup scripts are not nounset-safe with `set -u`.
set +u
source /opt/ros/jazzy/setup.bash
set -u

unset ROS_STATIC_PEERS FASTRTPS_DEFAULT_PROFILES_FILE ROS_LOCALHOST_ONLY
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-30}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-SUBNET}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
# WSL/OpenGL stacks can fail linking RViz shaders. Keep software rendering as default.
export APEX_RVIZ_SOFTWARE_GL="${APEX_RVIZ_SOFTWARE_GL:-1}"
if [ "${APEX_RVIZ_SOFTWARE_GL}" = "1" ]; then
  export LIBGL_ALWAYS_SOFTWARE=1
  export QT_XCB_FORCE_SOFTWARE_OPENGL=1
fi

echo "[APEX] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "[APEX] RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"
if [ "${APEX_RVIZ_SOFTWARE_GL}" = "1" ]; then
  echo "[APEX] RViz software OpenGL enabled"
fi
echo "[APEX] Probing topics (expected: /lidar/scan, /map, /map_metadata, /odom)..."
timeout 5 ros2 topic list | egrep "/lidar/scan|/map|/map_metadata|/odom" || true

exec rviz2 -d "${RVIZ_CONFIG}"
