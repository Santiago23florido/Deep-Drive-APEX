#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: apex_live_offline_map_publisher_up.sh <run_dir>" >&2
  exit 1
fi

RUN_DIR="$(cd "$1" && pwd)"
ROS_SETUP_SCRIPT="${APEX_ROS_SETUP_SCRIPT:-/opt/ros/jazzy/setup.bash}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_SIM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SIM_WS_ROOT="${APEX_SIM_ROOT}/ros2_ws"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  if [[ "${PATH}" == "${VIRTUAL_ENV}/bin:"* ]]; then
    PATH="${PATH#${VIRTUAL_ENV}/bin:}"
  fi
  unset VIRTUAL_ENV
fi
unset PYTHONHOME || true

set +u
source "${ROS_SETUP_SCRIPT}"
if [[ -f "${SIM_WS_ROOT}/install/setup.bash" ]]; then
  source "${SIM_WS_ROOT}/install/setup.bash"
fi
set -u

exec ros2 run rc_sim_description apex_offline_sensorfusion_map_publisher.py --ros-args \
  -p run_dir:="${RUN_DIR}" \
  -p trajectory_csv:="${RUN_DIR}/analysis_sensor_fusion_live/sensor_fusion_trajectory.csv" \
  -p summary_json:="${RUN_DIR}/analysis_sensor_fusion_live/sensor_fusion_summary.json" \
  -p lidar_points_csv:="${RUN_DIR}/analysis_sensor_fusion_live/sensor_fusion_lidar_points.csv" \
  -p frame_id:=odom_imu_lidar_fused \
  -p child_frame_id:=offline_live_base_link \
  -p map_topic:=/apex/sim/live_offline_map_points \
  -p path_topic:=/apex/sim/live_offline_map_path \
  -p odom_topic:=/apex/sim/live_offline_map_odom \
  -p status_topic:=/apex/sim/live_offline_map_status \
  -p point_stride:=1 \
  -p grid_resolution_m:=0.03 \
  -p reload_on_change:=true \
  -p allow_missing_inputs:=true \
  -p reload_period_s:=1.0
