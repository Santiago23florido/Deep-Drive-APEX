#!/usr/bin/env bash
set -eo pipefail

# Some ROS setup scripts reference this variable even when it is unset.
# Keep it defined to avoid unbound-variable failures in strict shells.
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"

source /opt/ros/jazzy/setup.bash

# Runtime python path:
# - local APEX package source (no colcon build required)
# - venv site-packages containing rplidar-roboticia
export PYTHONPATH="/work/ros2_ws/src/apex_telemetry:${PYTHONPATH:-}"
APEX_VENV_SITEPKG="$(echo /opt/apex_venv/lib/python*/site-packages)"
if [ -d "${APEX_VENV_SITEPKG}" ]; then
  export PYTHONPATH="${APEX_VENV_SITEPKG}:${PYTHONPATH}"
fi
unset ROS_STATIC_PEERS FASTRTPS_DEFAULT_PROFILES_FILE ROS_LOCALHOST_ONLY

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-30}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-SUBNET}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

PARAMS_FILE="/work/ros2_ws/src/apex_telemetry/config/apex_params.yaml"
SLAM_PARAMS_FILE="/work/ros2_ws/src/apex_telemetry/config/apex_slam_toolbox.yaml"

PIDS=()

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait || true
}

trap cleanup INT TERM

python3 -m apex_telemetry.nano_accel_serial_node \
  --ros-args \
  --params-file "${PARAMS_FILE}" \
  -p serial_port:="${APEX_SERIAL_PORT:-/dev/ttyACM0}" \
  -p baudrate:="${APEX_BAUDRATE:-115200}" &
PIDS+=("$!")

python3 -m apex_telemetry.kinematics_estimator_node \
  --ros-args \
  --params-file "${PARAMS_FILE}" &
PIDS+=("$!")

python3 -m apex_telemetry.kinematics_odometry_node \
  --ros-args \
  --params-file "${PARAMS_FILE}" &
PIDS+=("$!")

python3 -m apex_telemetry.rplidar_publisher_node \
  --ros-args \
  --params-file "${PARAMS_FILE}" \
  -p port:="${APEX_LIDAR_PORT:-/dev/ttyUSB0}" \
  -p baudrate:="${APEX_LIDAR_BAUDRATE:-115200}" &
PIDS+=("$!")

ros2 run tf2_ros static_transform_publisher \
  --x "${APEX_LIDAR_X_M:-0.18}" \
  --y "${APEX_LIDAR_Y_M:-0.0}" \
  --z "${APEX_LIDAR_Z_M:-0.12}" \
  --roll "${APEX_LIDAR_ROLL_RAD:-0.0}" \
  --pitch "${APEX_LIDAR_PITCH_RAD:-0.0}" \
  --yaw "${APEX_LIDAR_YAW_RAD:-0.0}" \
  --frame-id "${APEX_BASE_FRAME:-base_link}" \
  --child-frame-id "${APEX_LIDAR_FRAME:-laser}" &
PIDS+=("$!")

ros2 launch slam_toolbox online_async_launch.py \
  use_sim_time:=false \
  slam_params_file:="${SLAM_PARAMS_FILE}" &
PIDS+=("$!")

wait -n
cleanup
