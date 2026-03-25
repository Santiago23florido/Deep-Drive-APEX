#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <bundle_dir>" >&2
  exit 1
fi

BUNDLE_DIR="$(cd "$1" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BAG_DIR="${BUNDLE_DIR}/bag/raw_debug_run"

if [ ! -f "${BAG_DIR}/metadata.yaml" ]; then
  echo "rosbag2 directory not found at ${BAG_DIR}" >&2
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

ros2 bag play "${BAG_DIR}" --clock 20 &
PLAY_PID=$!
trap 'kill ${PLAY_PID} 2>/dev/null || true' EXIT INT TERM

rviz2 \
  --ros-args -p use_sim_time:=true \
  -d "${APEX_DIR}/rviz/apex_slam_auto.rviz"
