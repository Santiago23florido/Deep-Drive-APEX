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
APEX_DEBUG_RUN_NAME=""
APEX_DEBUG_BUNDLE_DIR=""
APEX_DOCKER_TAIL_LOG=""
APEX_RAW_BAG_DIR=""
APEX_DEBUG_FINALIZED=0
APEX_ROSBAG_PID=""

normalize_double_env() {
  local raw_value="${1:-}"
  if [ -z "${raw_value}" ]; then
    return 0
  fi
  if [[ "${raw_value}" =~ ^-?[0-9]+$ ]]; then
    printf '%s.0' "${raw_value}"
    return 0
  fi
  printf '%s' "${raw_value}"
}

write_pwm_snapshot() {
  if [ "${APEX_RECORD_DEBUG:-0}" != "1" ] || [ -z "${APEX_DEBUG_BUNDLE_DIR}" ]; then
    return
  fi

  local phase="$1"
  local output_path="${APEX_DEBUG_BUNDLE_DIR}/pwm_snapshot_${phase}.txt"
  {
    echo "# pwm_snapshot ${phase}"
    echo "# timestamp_utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [ ! -d /sys/class/pwm ]; then
      echo "/sys/class/pwm is not available"
      exit 0
    fi

    find /sys/class/pwm -maxdepth 2 -type f \
      \( -name enable -o -name period -o -name duty_cycle \) \
      | sort \
      | while read -r path; do
          echo "## ${path}"
          cat "${path}" 2>/dev/null || echo "<unreadable>"
          echo
        done
  } > "${output_path}"
}

write_debug_metadata() {
  if [ "${APEX_RECORD_DEBUG:-0}" != "1" ] || [ -z "${APEX_DEBUG_BUNDLE_DIR}" ]; then
    return
  fi

  export APEX_METADATA_PATH="${APEX_DEBUG_BUNDLE_DIR}/run_metadata.json"
  export APEX_DOCKER_ENV_PATH="${APEX_DEBUG_BUNDLE_DIR}/docker_env.json"
  python3 <<'PY'
import json
import os
from pathlib import Path

metadata_path = Path(os.environ["APEX_METADATA_PATH"])
docker_env_path = Path(os.environ["APEX_DOCKER_ENV_PATH"])

selected_env = {}
for key in sorted(os.environ):
    if key.startswith("APEX_") or key in {
        "ROS_DOMAIN_ID",
        "ROS_AUTOMATIC_DISCOVERY_RANGE",
        "RMW_IMPLEMENTATION",
    }:
        selected_env[key] = os.environ[key]

docker_env_path.write_text(
    json.dumps(selected_env, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

metadata = {
    "run_id": os.environ.get("APEX_DEBUG_RUN_NAME", ""),
    "git_commit": os.environ.get("APEX_GIT_COMMIT", "unknown"),
    "git_dirty_count": os.environ.get("APEX_GIT_DIRTY", "unknown"),
    "bundle_dir": os.environ.get("APEX_DEBUG_BUNDLE_DIR", ""),
    "bag_dir": os.environ.get("APEX_RAW_BAG_DIR", ""),
    "bag_mcap_path": os.environ.get("APEX_FINAL_MCAP_PATH", ""),
    "bag_metadata_path": os.environ.get("APEX_FINAL_BAG_METADATA_PATH", ""),
    "docker_tail_log": os.environ.get("APEX_DOCKER_TAIL_LOG", ""),
    "recon_diagnostic_log": os.environ.get("APEX_RECON_LOG_PATH", ""),
    "params_snapshot": os.environ.get("APEX_PARAMS_SNAPSHOT_PATH", ""),
    "slam_params_snapshot": os.environ.get("APEX_SLAM_PARAMS_SNAPSHOT_PATH", ""),
    "record_debug_enabled": os.environ.get("APEX_RECORD_DEBUG", "0"),
    "diagnostic_mode_env": os.environ.get("APEX_RECON_DIAGNOSTIC_MODE", ""),
    "steering_direction_sign_env": os.environ.get("APEX_STEERING_DIRECTION_SIGN", ""),
    "lidar_heading_offset_env": os.environ.get("APEX_LIDAR_HEADING_OFFSET_DEG", ""),
    "timestamp_utc": os.environ.get("APEX_RUN_TIMESTAMP_UTC", ""),
    "env_overrides": selected_env,
}
metadata_path.write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

finalize_debug_artifacts() {
  if [ "${APEX_RECORD_DEBUG:-0}" != "1" ] || [ -z "${APEX_DEBUG_BUNDLE_DIR}" ]; then
    return
  fi
  if [ "${APEX_DEBUG_FINALIZED}" = "1" ]; then
    return
  fi

  export APEX_FINAL_MCAP_PATH=""
  export APEX_FINAL_BAG_METADATA_PATH=""
  if [ -n "${APEX_RAW_BAG_DIR}" ] && [ -d "${APEX_RAW_BAG_DIR}" ]; then
    local mcap_path metadata_path
    mcap_path="$(find "${APEX_RAW_BAG_DIR}" -maxdepth 1 -name '*.mcap' | sort | head -n 1 || true)"
    metadata_path="${APEX_RAW_BAG_DIR}/metadata.yaml"
    if [ -n "${mcap_path}" ]; then
      cp "${mcap_path}" "${APEX_DEBUG_BUNDLE_DIR}/bag/debug_run.mcap"
      export APEX_FINAL_MCAP_PATH="${APEX_DEBUG_BUNDLE_DIR}/bag/debug_run.mcap"
    fi
    if [ -f "${metadata_path}" ]; then
      cp "${metadata_path}" "${APEX_DEBUG_BUNDLE_DIR}/bag/metadata.yaml"
      export APEX_FINAL_BAG_METADATA_PATH="${APEX_DEBUG_BUNDLE_DIR}/bag/metadata.yaml"
    fi
  fi

  write_debug_metadata
  APEX_DEBUG_FINALIZED=1
}

cleanup() {
  write_pwm_snapshot "after"
  if [ -n "${APEX_ROSBAG_PID}" ]; then
    kill -INT "${APEX_ROSBAG_PID}" 2>/dev/null || true
    wait "${APEX_ROSBAG_PID}" 2>/dev/null || true
  fi
  for pid in "${PIDS[@]:-}"; do
    if [ -n "${APEX_ROSBAG_PID}" ] && [ "${pid}" = "${APEX_ROSBAG_PID}" ]; then
      continue
    fi
    kill -INT "$pid" 2>/dev/null || true
  done
  sleep 1
  for pid in "${PIDS[@]:-}"; do
    if [ -n "${APEX_ROSBAG_PID}" ] && [ "${pid}" = "${APEX_ROSBAG_PID}" ]; then
      continue
    fi
    kill "$pid" 2>/dev/null || true
  done
  wait || true
  finalize_debug_artifacts
}

trap cleanup INT TERM EXIT

setup_debug_run() {
  if [ "${APEX_RECORD_DEBUG:-0}" != "1" ]; then
    return
  fi

  local timestamp_utc run_name
  timestamp_utc="$(date -u +%Y%m%dT%H%M%SZ)"
  export APEX_RUN_TIMESTAMP_UTC="${timestamp_utc}"
  if [ -n "${APEX_DEBUG_RUN_ID:-}" ]; then
    run_name="${APEX_DEBUG_RUN_ID}_${timestamp_utc}"
  else
    run_name="debug_${timestamp_utc}"
  fi

  export APEX_DEBUG_RUN_NAME="${run_name}"
  export APEX_DEBUG_BUNDLE_DIR="${APEX_DEBUG_OUTPUT_DIR%/}/${run_name}"
  APEX_DEBUG_BUNDLE_DIR="${APEX_DEBUG_BUNDLE_DIR}"
  mkdir -p "${APEX_DEBUG_BUNDLE_DIR}/bag" "${APEX_DEBUG_BUNDLE_DIR}/config"

  export APEX_DOCKER_TAIL_LOG="${APEX_DEBUG_BUNDLE_DIR}/docker_tail.log"
  APEX_DOCKER_TAIL_LOG="${APEX_DOCKER_TAIL_LOG}"
  export APEX_RECON_LOG_PATH="${APEX_RECON_LOG_PATH:-${APEX_DEBUG_BUNDLE_DIR}/recon_diagnostic.log}"
  mkdir -p "$(dirname "${APEX_RECON_LOG_PATH}")"

  export APEX_PARAMS_SNAPSHOT_PATH="${APEX_DEBUG_BUNDLE_DIR}/config/apex_params.yaml"
  export APEX_SLAM_PARAMS_SNAPSHOT_PATH="${APEX_DEBUG_BUNDLE_DIR}/config/apex_slam_toolbox.yaml"
  cp "${PARAMS_FILE}" "${APEX_PARAMS_SNAPSHOT_PATH}"
  cp "${SLAM_PARAMS_FILE}" "${APEX_SLAM_PARAMS_SNAPSHOT_PATH}"

  export APEX_RAW_BAG_DIR="${APEX_DEBUG_BUNDLE_DIR}/bag/raw_debug_run"
  APEX_RAW_BAG_DIR="${APEX_RAW_BAG_DIR}"
  write_debug_metadata
  write_pwm_snapshot "before"

  exec > >(tee -a "${APEX_DOCKER_TAIL_LOG}") 2>&1
  echo "[APEX] Debug run bundle: ${APEX_DEBUG_BUNDLE_DIR}"
}

start_debug_bag_recording() {
  if [ "${APEX_RECORD_DEBUG:-0}" != "1" ]; then
    return
  fi

  rm -rf "${APEX_RAW_BAG_DIR}"
  ros2 bag record \
    --storage mcap \
    --output "${APEX_RAW_BAG_DIR}" \
    --topics \
    /lidar/scan \
    /odom \
    /tf \
    /tf_static \
    /map \
    /map_metadata \
    /apex/kinematics/acceleration \
    /apex/kinematics/velocity \
    /apex/kinematics/position \
    /apex/kinematics/angular_velocity \
    /apex/kinematics/heading &
  APEX_ROSBAG_PID="$!"
  PIDS+=("${APEX_ROSBAG_PID}")
}

setup_debug_run

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

LIDAR_ARGS=(
  --ros-args
  --params-file "${PARAMS_FILE}"
  -p "port:=${APEX_LIDAR_PORT:-/dev/ttyUSB0}"
  -p "baudrate:=${APEX_LIDAR_BAUDRATE:-115200}"
)
if [ -n "${APEX_LIDAR_HEADING_OFFSET_DEG:-}" ]; then
  LIDAR_ARGS+=(-p "heading_offset_deg:=${APEX_LIDAR_HEADING_OFFSET_DEG}")
fi
python3 -m apex_telemetry.rplidar_publisher_node "${LIDAR_ARGS[@]}" &
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

if [ "${APEX_ENABLE_SLAM_TOOLBOX:-1}" = "1" ]; then
  ros2 launch slam_toolbox online_async_launch.py \
    use_sim_time:=false \
    slam_params_file:="${SLAM_PARAMS_FILE}" &
  PIDS+=("$!")
else
  echo "[APEX] slam_toolbox disabled for this run"
fi

start_debug_bag_recording

if [ "${APEX_ENABLE_RECON_MAPPING:-0}" = "1" ]; then
  RECON_ARGS=(
    --ros-args
    --params-file "${PARAMS_FILE}"
  )
  if [ "${APEX_ENABLE_SLAM_TOOLBOX:-1}" != "1" ]; then
    RECON_ARGS+=(-p "reset_map_on_start:=false")
    RECON_ARGS+=(-p "save_map_on_completion:=false")
  fi
  if [ -n "${APEX_RECON_DIAGNOSTIC_MODE:-}" ]; then
    RECON_ARGS+=(-p "diagnostic_mode:=${APEX_RECON_DIAGNOSTIC_MODE}")
  fi
  if [ -n "${APEX_RESET_MAP_ON_START:-}" ]; then
    RECON_ARGS+=(-p "reset_map_on_start:=${APEX_RESET_MAP_ON_START}")
  fi
  if [ -n "${APEX_SAVE_MAP_ON_COMPLETION:-}" ]; then
    RECON_ARGS+=(-p "save_map_on_completion:=${APEX_SAVE_MAP_ON_COMPLETION}")
  fi
  if [ -n "${APEX_STEERING_CENTER_TRIM_DC:-}" ]; then
    RECON_ARGS+=(-p "steering_center_trim_dc:=$(normalize_double_env "${APEX_STEERING_CENTER_TRIM_DC}")")
  fi
  if [ -n "${APEX_STEERING_DIRECTION_SIGN:-}" ]; then
    RECON_ARGS+=(-p "steering_direction_sign:=$(normalize_double_env "${APEX_STEERING_DIRECTION_SIGN}")")
  fi
  if [ -n "${APEX_RECON_FIXED_SPEED_PCT:-}" ]; then
    RECON_ARGS+=(-p "diagnostic_fixed_speed_pct:=$(normalize_double_env "${APEX_RECON_FIXED_SPEED_PCT}")")
  fi
  if [ -n "${APEX_EXPLORE_MIN_SPEED_PCT:-}" ]; then
    RECON_ARGS+=(-p "explore_min_speed_pct:=$(normalize_double_env "${APEX_EXPLORE_MIN_SPEED_PCT}")")
  fi
  if [ -n "${APEX_EXPLORE_MAX_SPEED_PCT:-}" ]; then
    RECON_ARGS+=(-p "explore_max_speed_pct:=$(normalize_double_env "${APEX_EXPLORE_MAX_SPEED_PCT}")")
  fi
  if [ -n "${APEX_STEERING_GAIN:-}" ]; then
    RECON_ARGS+=(-p "steering_gain:=$(normalize_double_env "${APEX_STEERING_GAIN}")")
  fi
  if [ -n "${APEX_FRONT_WINDOW_DEG:-}" ]; then
    RECON_ARGS+=(-p "front_window_deg:=${APEX_FRONT_WINDOW_DEG}")
  fi
  if [ -n "${APEX_SIDE_WINDOW_DEG:-}" ]; then
    RECON_ARGS+=(-p "side_window_deg:=${APEX_SIDE_WINDOW_DEG}")
  fi
  if [ -n "${APEX_FOV_HALF_ANGLE_DEG:-}" ]; then
    RECON_ARGS+=(-p "fov_half_angle_deg:=$(normalize_double_env "${APEX_FOV_HALF_ANGLE_DEG}")")
  fi
  if [ -n "${APEX_CENTER_ANGLE_PENALTY_PER_DEG:-}" ]; then
    RECON_ARGS+=(-p "center_angle_penalty_per_deg:=$(normalize_double_env "${APEX_CENTER_ANGLE_PENALTY_PER_DEG}")")
  fi
  if [ -n "${APEX_WALL_AVOID_DISTANCE_M:-}" ]; then
    RECON_ARGS+=(-p "wall_avoid_distance_m:=$(normalize_double_env "${APEX_WALL_AVOID_DISTANCE_M}")")
  fi
  if [ -n "${APEX_WALL_AVOID_GAIN_DEG_PER_M:-}" ]; then
    RECON_ARGS+=(-p "wall_avoid_gain_deg_per_m:=$(normalize_double_env "${APEX_WALL_AVOID_GAIN_DEG_PER_M}")")
  fi
  if [ -n "${APEX_GAP_ESCAPE_HEADING_THRESHOLD_DEG:-}" ]; then
    RECON_ARGS+=(-p "gap_escape_heading_threshold_deg:=$(normalize_double_env "${APEX_GAP_ESCAPE_HEADING_THRESHOLD_DEG}")")
  fi
  if [ -n "${APEX_GAP_ESCAPE_RELEASE_DISTANCE_M:-}" ]; then
    RECON_ARGS+=(-p "gap_escape_release_distance_m:=$(normalize_double_env "${APEX_GAP_ESCAPE_RELEASE_DISTANCE_M}")")
  fi
  if [ -n "${APEX_GAP_ESCAPE_WEIGHT:-}" ]; then
    RECON_ARGS+=(-p "gap_escape_weight:=$(normalize_double_env "${APEX_GAP_ESCAPE_WEIGHT}")")
  fi
  if [ -n "${APEX_CORRIDOR_BALANCE_RATIO_THRESHOLD:-}" ]; then
    RECON_ARGS+=(-p "corridor_balance_ratio_threshold:=$(normalize_double_env "${APEX_CORRIDOR_BALANCE_RATIO_THRESHOLD}")")
  fi
  if [ -n "${APEX_CORRIDOR_FRONT_MIN_CLEARANCE_M:-}" ]; then
    RECON_ARGS+=(-p "corridor_front_min_clearance_m:=$(normalize_double_env "${APEX_CORRIDOR_FRONT_MIN_CLEARANCE_M}")")
  fi
  if [ -n "${APEX_CORRIDOR_SIDE_MIN_CLEARANCE_M:-}" ]; then
    RECON_ARGS+=(-p "corridor_side_min_clearance_m:=$(normalize_double_env "${APEX_CORRIDOR_SIDE_MIN_CLEARANCE_M}")")
  fi
  if [ -n "${APEX_CORRIDOR_FRONT_TURN_WEIGHT:-}" ]; then
    RECON_ARGS+=(-p "corridor_front_turn_weight:=$(normalize_double_env "${APEX_CORRIDOR_FRONT_TURN_WEIGHT}")")
  fi
  if [ -n "${APEX_CORRIDOR_OVERRIDE_MARGIN_DEG:-}" ]; then
    RECON_ARGS+=(-p "corridor_override_margin_deg:=$(normalize_double_env "${APEX_CORRIDOR_OVERRIDE_MARGIN_DEG}")")
  fi
  if [ -n "${APEX_CORRIDOR_MIN_HEADING_DEG:-}" ]; then
    RECON_ARGS+=(-p "corridor_min_heading_deg:=$(normalize_double_env "${APEX_CORRIDOR_MIN_HEADING_DEG}")")
  fi
  if [ -n "${APEX_CORRIDOR_WALL_START_DEG:-}" ]; then
    RECON_ARGS+=(-p "corridor_wall_start_deg:=${APEX_CORRIDOR_WALL_START_DEG}")
  fi
  if [ -n "${APEX_CORRIDOR_WALL_END_DEG:-}" ]; then
    RECON_ARGS+=(-p "corridor_wall_end_deg:=${APEX_CORRIDOR_WALL_END_DEG}")
  fi
  if [ -n "${APEX_CORRIDOR_WALL_MIN_POINTS:-}" ]; then
    RECON_ARGS+=(-p "corridor_wall_min_points:=${APEX_CORRIDOR_WALL_MIN_POINTS}")
  fi
  if [ -n "${APEX_WALL_FOLLOW_TARGET_DISTANCE_M:-}" ]; then
    RECON_ARGS+=(-p "wall_follow_target_distance_m:=$(normalize_double_env "${APEX_WALL_FOLLOW_TARGET_DISTANCE_M}")")
  fi
  if [ -n "${APEX_WALL_FOLLOW_GAIN_DEG_PER_M:-}" ]; then
    RECON_ARGS+=(-p "wall_follow_gain_deg_per_m:=$(normalize_double_env "${APEX_WALL_FOLLOW_GAIN_DEG_PER_M}")")
  fi
  if [ -n "${APEX_WALL_FOLLOW_LIMIT_DEG:-}" ]; then
    RECON_ARGS+=(-p "wall_follow_limit_deg:=$(normalize_double_env "${APEX_WALL_FOLLOW_LIMIT_DEG}")")
  fi
  if [ -n "${APEX_WALL_FOLLOW_ACTIVATION_HEADING_DEG:-}" ]; then
    RECON_ARGS+=(-p "wall_follow_activation_heading_deg:=$(normalize_double_env "${APEX_WALL_FOLLOW_ACTIVATION_HEADING_DEG}")")
  fi
  if [ -n "${APEX_WALL_FOLLOW_RELEASE_BALANCE_RATIO:-}" ]; then
    RECON_ARGS+=(-p "wall_follow_release_balance_ratio:=$(normalize_double_env "${APEX_WALL_FOLLOW_RELEASE_BALANCE_RATIO}")")
  fi
  if [ -n "${APEX_WALL_FOLLOW_MIN_CYCLES:-}" ]; then
    RECON_ARGS+=(-p "wall_follow_min_cycles:=${APEX_WALL_FOLLOW_MIN_CYCLES}")
  fi
  if [ -n "${APEX_WALL_FOLLOW_MAX_CLEARANCE_M:-}" ]; then
    RECON_ARGS+=(-p "wall_follow_max_clearance_m:=$(normalize_double_env "${APEX_WALL_FOLLOW_MAX_CLEARANCE_M}")")
  fi
  if [ -n "${APEX_WALL_FOLLOW_FRONT_TURN_WEIGHT:-}" ]; then
    RECON_ARGS+=(-p "wall_follow_front_turn_weight:=$(normalize_double_env "${APEX_WALL_FOLLOW_FRONT_TURN_WEIGHT}")")
  fi
  if [ -n "${APEX_STARTUP_CONSENSUS_MIN_HEADING_DEG:-}" ]; then
    RECON_ARGS+=(-p "startup_consensus_min_heading_deg:=$(normalize_double_env "${APEX_STARTUP_CONSENSUS_MIN_HEADING_DEG}")")
  fi
  if [ -n "${APEX_STARTUP_VALID_CYCLES_REQUIRED:-}" ]; then
    RECON_ARGS+=(-p "startup_valid_cycles_required:=${APEX_STARTUP_VALID_CYCLES_REQUIRED}")
  fi
  if [ -n "${APEX_STARTUP_GAP_LOCKOUT_CYCLES:-}" ]; then
    RECON_ARGS+=(-p "startup_gap_lockout_cycles:=${APEX_STARTUP_GAP_LOCKOUT_CYCLES}")
  fi
  if [ -n "${APEX_STARTUP_LATCH_CYCLES:-}" ]; then
    RECON_ARGS+=(-p "startup_latch_cycles:=${APEX_STARTUP_LATCH_CYCLES}")
  fi
  if [ -n "${APEX_AMBIGUITY_PROBE_SPEED_PCT:-}" ]; then
    RECON_ARGS+=(-p "ambiguity_probe_speed_pct:=$(normalize_double_env "${APEX_AMBIGUITY_PROBE_SPEED_PCT}")")
  fi
  if [ -n "${APEX_TURN_SPEED_REDUCTION:-}" ]; then
    RECON_ARGS+=(-p "turn_speed_reduction:=$(normalize_double_env "${APEX_TURN_SPEED_REDUCTION}")")
  fi
  if [ -n "${APEX_MIN_TURN_SPEED_FACTOR:-}" ]; then
    RECON_ARGS+=(-p "min_turn_speed_factor:=$(normalize_double_env "${APEX_MIN_TURN_SPEED_FACTOR}")")
  fi
  if [ -n "${APEX_RECON_STEP_DURATION_S:-}" ]; then
    RECON_ARGS+=(-p "diagnostic_step_duration_s:=$(normalize_double_env "${APEX_RECON_STEP_DURATION_S}")")
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
    RECON_ARGS+=(-p "diagnostic_recon_timeout_s:=$(normalize_double_env "${APEX_RECON_TIMEOUT_S}")")
  fi
  if [ -n "${APEX_RECON_MAX_RECOVERIES:-}" ]; then
    RECON_ARGS+=(-p "diagnostic_max_recoveries:=${APEX_RECON_MAX_RECOVERIES}")
  fi
  if [ -n "${APEX_RECON_MIN_PROGRESS_M:-}" ]; then
    RECON_ARGS+=(-p "diagnostic_min_progress_m:=$(normalize_double_env "${APEX_RECON_MIN_PROGRESS_M}")")
  fi
  if [ -n "${APEX_RECON_LOG_LEVEL:-}" ]; then
    RECON_ARGS+=(--log-level "${APEX_RECON_LOG_LEVEL}")
  fi
  python3 -m apex_telemetry.recon_mapping_node "${RECON_ARGS[@]}" &
  PIDS+=("$!")
fi

wait -n
