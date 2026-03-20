#!/usr/bin/env bash
set -eo pipefail

# Some ROS setup scripts reference this variable even when it is unset.
# Keep it defined to avoid unbound-variable failures in strict shells.
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

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

if [ "${APEX_ENABLE_RECON_MAPPING:-0}" = "1" ]; then
  RECON_ARGS=(
    --ros-args
    --params-file "${PARAMS_FILE}"
  )
  if [ -n "${APEX_RECON_DIAGNOSTIC_MODE:-}" ]; then
    RECON_ARGS+=(-p "diagnostic_mode:=${APEX_RECON_DIAGNOSTIC_MODE}")
  fi
  if [ -n "${APEX_RECON_FIXED_SPEED_PCT:-}" ]; then
    RECON_ARGS+=(-p "diagnostic_fixed_speed_pct:=${APEX_RECON_FIXED_SPEED_PCT}")
  fi
  if [ -n "${APEX_RECON_STEP_DURATION_S:-}" ]; then
    RECON_ARGS+=(-p "diagnostic_step_duration_s:=${APEX_RECON_STEP_DURATION_S}")
  fi
  if [ -n "${APEX_RECON_LOG_PATH:-}" ]; then
    RECON_ARGS+=(-p "diagnostic_log_path:=${APEX_RECON_LOG_PATH}")
  fi
  if [ -n "${APEX_RECON_LOG_FLUSH_EVERY:-}" ]; then
    RECON_ARGS+=(-p "diagnostic_file_flush_every_n_records:=${APEX_RECON_LOG_FLUSH_EVERY}")
  fi
  if [ -n "${APEX_RECON_LOG_OVERWRITE:-}" ]; then
    RECON_ARGS+=(-p "diagnostic_overwrite_log_on_start:=${APEX_RECON_LOG_OVERWRITE}")
  fi
  if [ -n "${APEX_RECON_TIMEOUT_S:-}" ]; then
    RECON_ARGS+=(-p "diagnostic_recon_timeout_s:=${APEX_RECON_TIMEOUT_S}")
  fi
  if [ -n "${APEX_RECON_MAX_RECOVERIES:-}" ]; then
    RECON_ARGS+=(-p "diagnostic_max_recoveries:=${APEX_RECON_MAX_RECOVERIES}")
  fi
  if [ -n "${APEX_RECON_MIN_PROGRESS_M:-}" ]; then
    RECON_ARGS+=(-p "diagnostic_min_progress_m:=${APEX_RECON_MIN_PROGRESS_M}")
  fi
  if [ -n "${APEX_RECON_LOG_LEVEL:-}" ]; then
    RECON_ARGS+=(--log-level "${APEX_RECON_LOG_LEVEL}")
  fi
  python3 -m apex_telemetry.recon_mapping_node "${RECON_ARGS[@]}" &
  PIDS+=("$!")
fi

wait -n
cleanup
