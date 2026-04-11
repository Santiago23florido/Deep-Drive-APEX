#!/usr/bin/env bash
set -euo pipefail

export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

set +u
source /opt/ros/jazzy/setup.bash
set -u

export PYTHONPATH="/work/ros2_ws/src/apex_telemetry:${PYTHONPATH:-}"
APEX_VENV_SITEPKG="$(echo /opt/apex_venv/lib/python*/site-packages)"
if [ -d "${APEX_VENV_SITEPKG}" ]; then
  export PYTHONPATH="${APEX_VENV_SITEPKG}:${PYTHONPATH}"
fi

PARAMS_FILE="${APEX_PARAMS_FILE:-/work/ros2_ws/src/apex_telemetry/config/apex_params.yaml}"
PIDS=()
PID_NAMES=()
ENABLE_KINEMATICS="${APEX_ENABLE_KINEMATICS:-1}"
ENABLE_IMU_LIDAR_FUSION="${APEX_ENABLE_IMU_LIDAR_FUSION:-0}"
ENABLE_CURVE_ENTRY_PLANNER="${APEX_ENABLE_CURVE_ENTRY_PLANNER:-0}"
ENABLE_PATH_TRACKER="${APEX_ENABLE_PATH_TRACKER:-0}"
ENABLE_RECOGNITION_TOUR_PLANNER="${APEX_ENABLE_RECOGNITION_TOUR_PLANNER:-0}"
ENABLE_RECOGNITION_TOUR_TRACKER="${APEX_ENABLE_RECOGNITION_TOUR_TRACKER:-0}"
ENABLE_CMDVEL_ACTUATION_BRIDGE="${APEX_ENABLE_CMDVEL_ACTUATION_BRIDGE:-0}"
ENABLE_MANUAL_CONTROL_BRIDGE="${APEX_ENABLE_MANUAL_CONTROL_BRIDGE:-0}"
ENABLE_RECOGNITION_SESSION_MANAGER="${APEX_ENABLE_RECOGNITION_SESSION_MANAGER:-0}"
ENABLE_OFFLINE_SUBMAP_REFINER="${APEX_ENABLE_OFFLINE_SUBMAP_REFINER:-0}"
ENABLE_FIXED_MAP_ROUTE_PLANNER="${APEX_ENABLE_FIXED_MAP_ROUTE_PLANNER:-0}"
ENABLE_FIXED_MAP_PUBLISHER="${APEX_ENABLE_FIXED_MAP_PUBLISHER:-0}"
STARTUP_COMPAT="${APEX_STARTUP_COMPAT:-modern}"
STAGGERED_STARTUP="${APEX_STAGGERED_STARTUP:-1}"
SERIAL_WARMUP_S="${APEX_SERIAL_WARMUP_S:-1.0}"
LIDAR_STARTUP_SETTLE_S="${APEX_LIDAR_STARTUP_SETTLE_S:-2.5}"
NODE_STARTUP_STAGGER_S="${APEX_NODE_STARTUP_STAGGER_S:-0.0}"
SERIAL_CONNECT_TOGGLE_DTR="${APEX_SERIAL_CONNECT_TOGGLE_DTR:-}"
SERIAL_CONNECT_DTR_LOW_S="${APEX_SERIAL_CONNECT_DTR_LOW_S:-}"
SERIAL_CONNECT_SETTLE_S="${APEX_SERIAL_CONNECT_SETTLE_S:-}"
SERIAL_FLUSH_INPUT_ON_CONNECT="${APEX_SERIAL_FLUSH_INPUT_ON_CONNECT:-}"
SERIAL_CONNECT_PROFILE_NAME="${APEX_SERIAL_CONNECT_PROFILE_NAME:-}"
SERIAL_NO_DATA_RECONNECT_S="${APEX_SERIAL_NO_DATA_RECONNECT_S:-}"
BRIDGE_MIN_EFFECTIVE_SPEED_PCT="${APEX_BRIDGE_MIN_EFFECTIVE_SPEED_PCT:-}"
BRIDGE_MAX_SPEED_PCT="${APEX_BRIDGE_MAX_SPEED_PCT:-}"
BRIDGE_MIN_EFFECTIVE_REVERSE_SPEED_PCT="${APEX_BRIDGE_MIN_EFFECTIVE_REVERSE_SPEED_PCT:-}"
BRIDGE_MAX_REVERSE_SPEED_PCT="${APEX_BRIDGE_MAX_REVERSE_SPEED_PCT:-}"
BRIDGE_LAUNCH_BOOST_SPEED_PCT="${APEX_BRIDGE_LAUNCH_BOOST_SPEED_PCT:-}"
BRIDGE_LAUNCH_BOOST_HOLD_S="${APEX_BRIDGE_LAUNCH_BOOST_HOLD_S:-}"
BRIDGE_ACTIVE_BRAKE_ON_ZERO="${APEX_BRIDGE_ACTIVE_BRAKE_ON_ZERO:-}"
BRIDGE_ACTUATION_BACKEND="${APEX_BRIDGE_ACTUATION_BACKEND:-${APEX_ACTUATION_BACKEND:-}}"
RECOGNITION_TRACKER_PUBLISH_UNARMED_ZERO_CMD="${APEX_RECOGNITION_TRACKER_PUBLISH_UNARMED_ZERO_CMD:-false}"
ESTIMATION_BACKEND="${APEX_ESTIMATION_BACKEND:-}"
FIXED_MAP_RUN_DIR="${APEX_FIXED_MAP_RUN_DIR:-}"
FIXED_MAP_DIR="${APEX_FIXED_MAP_DIR:-}"
if [[ -z "${FIXED_MAP_DIR}" && -n "${FIXED_MAP_RUN_DIR}" ]]; then
  FIXED_MAP_DIR="${FIXED_MAP_RUN_DIR%/}/fixed_map"
fi
if [[ -z "${FIXED_MAP_RUN_DIR}" && -n "${FIXED_MAP_DIR}" ]]; then
  FIXED_MAP_RUN_DIR="$(dirname "${FIXED_MAP_DIR}")"
fi
FIXED_MAP_YAML="${APEX_FIXED_MAP_YAML:-}"
FIXED_MAP_DISTANCE_NPY="${APEX_FIXED_MAP_DISTANCE_NPY:-}"
FIXED_MAP_VISUAL_POINTS_CSV="${APEX_FIXED_MAP_VISUAL_POINTS_CSV:-}"
FIXED_MAP_ROUTE_CSV="${APEX_FIXED_MAP_ROUTE_CSV:-}"
FIXED_MAP_BUILD_STATUS_JSON="${APEX_FIXED_MAP_BUILD_STATUS_JSON:-}"
if [[ -n "${FIXED_MAP_DIR}" ]]; then
  FIXED_MAP_YAML="${FIXED_MAP_YAML:-${FIXED_MAP_DIR%/}/fixed_map.yaml}"
  FIXED_MAP_DISTANCE_NPY="${FIXED_MAP_DISTANCE_NPY:-${FIXED_MAP_DIR%/}/fixed_map_distance.npy}"
  FIXED_MAP_VISUAL_POINTS_CSV="${FIXED_MAP_VISUAL_POINTS_CSV:-${FIXED_MAP_DIR%/}/fixed_map_visual_points.csv}"
  FIXED_MAP_ROUTE_CSV="${FIXED_MAP_ROUTE_CSV:-${FIXED_MAP_DIR%/}/fixed_route_path.csv}"
  FIXED_MAP_BUILD_STATUS_JSON="${FIXED_MAP_BUILD_STATUS_JSON:-${FIXED_MAP_DIR%/}/fixed_map_build_status.json}"
fi
FIXED_MAP_AUTOBUILD="${APEX_FIXED_MAP_AUTOBUILD:-0}"
FIXED_MAP_BUILDER_SCRIPT="${APEX_FIXED_MAP_BUILDER_SCRIPT:-/work/repo/APEX/tools/core/build_fixed_map_from_offline_refined.py}"
FUSION_ODOM_TOPIC="${APEX_FUSION_ODOM_TOPIC:-}"
FUSION_PATH_TOPIC="${APEX_FUSION_PATH_TOPIC:-}"
FUSION_POSE_TOPIC="${APEX_FUSION_POSE_TOPIC:-}"
FUSION_LIVE_MAP_TOPIC="${APEX_FUSION_LIVE_MAP_TOPIC:-}"
FUSION_FULL_MAP_TOPIC="${APEX_FUSION_FULL_MAP_TOPIC:-}"
FUSION_STATUS_TOPIC="${APEX_FUSION_STATUS_TOPIC:-}"
FUSION_ODOM_FRAME_ID="${APEX_FUSION_ODOM_FRAME_ID:-}"
RECOGNITION_TRACKER_PATH_TOPIC="${APEX_RECOGNITION_TRACKER_PATH_TOPIC:-}"
RECOGNITION_TRACKER_PLANNING_STATUS_TOPIC="${APEX_RECOGNITION_TRACKER_PLANNING_STATUS_TOPIC:-}"
RECOGNITION_TRACKER_ODOM_TOPIC="${APEX_RECOGNITION_TRACKER_ODOM_TOPIC:-}"
RECOGNITION_TRACKER_FUSION_STATUS_TOPIC="${APEX_RECOGNITION_TRACKER_FUSION_STATUS_TOPIC:-}"
RECOGNITION_TRACKER_STATUS_TOPIC="${APEX_RECOGNITION_TRACKER_STATUS_TOPIC:-}"
RECOGNITION_TRACKER_REQUIRE_ARM="${APEX_RECOGNITION_TRACKER_REQUIRE_ARM:-}"
RECOGNITION_TRACKER_DEFAULT_ARMED="${APEX_RECOGNITION_TRACKER_DEFAULT_ARMED:-}"
RECOGNITION_TRACKER_PATH_STALE_MAX_AGE_S="${APEX_RECOGNITION_TRACKER_PATH_STALE_MAX_AGE_S:-}"
RECOGNITION_TRACKER_PATH_STALE_ABORT_HOLD_S="${APEX_RECOGNITION_TRACKER_PATH_STALE_ABORT_HOLD_S:-}"
RECOGNITION_TRACKER_REAR_AXLE_OFFSET_X_M="${APEX_RECOGNITION_TRACKER_REAR_AXLE_OFFSET_X_M:-}"
RECOGNITION_TRACKER_REAR_AXLE_OFFSET_Y_M="${APEX_RECOGNITION_TRACKER_REAR_AXLE_OFFSET_Y_M:-}"
RECOGNITION_TRACKER_ODOM_TIMEOUT_S="${APEX_RECOGNITION_TRACKER_ODOM_TIMEOUT_S:-}"
RECOGNITION_TRACKER_MIN_LOOKAHEAD_M="${APEX_RECOGNITION_TRACKER_MIN_LOOKAHEAD_M:-}"
RECOGNITION_TRACKER_MAX_LOOKAHEAD_M="${APEX_RECOGNITION_TRACKER_MAX_LOOKAHEAD_M:-}"
RECOGNITION_TRACKER_SHARP_TURN_LOOKAHEAD_MIN_M="${APEX_RECOGNITION_TRACKER_SHARP_TURN_LOOKAHEAD_MIN_M:-}"
RECOGNITION_TRACKER_ANGULAR_CMD_EMA_ALPHA="${APEX_RECOGNITION_TRACKER_ANGULAR_CMD_EMA_ALPHA:-}"
if [[ -z "${ESTIMATION_BACKEND}" && "${ENABLE_FIXED_MAP_ROUTE_PLANNER}" == "1" ]]; then
  ESTIMATION_BACKEND="fixed_map"
fi
if [[ "${ESTIMATION_BACKEND}" == "fixed_map" ]]; then
  FUSION_ODOM_TOPIC="${FUSION_ODOM_TOPIC:-/apex/odometry/fixed_map_localized}"
  FUSION_PATH_TOPIC="${FUSION_PATH_TOPIC:-/apex/localization/fixed_map_path}"
  FUSION_POSE_TOPIC="${FUSION_POSE_TOPIC:-/apex/localization/fixed_map_pose}"
  FUSION_LIVE_MAP_TOPIC="${FUSION_LIVE_MAP_TOPIC:-/apex/localization/fixed_map_live_points}"
  FUSION_FULL_MAP_TOPIC="${FUSION_FULL_MAP_TOPIC:-/apex/localization/fixed_map_points}"
  FUSION_STATUS_TOPIC="${FUSION_STATUS_TOPIC:-/apex/localization/fixed_map_status}"
  FUSION_ODOM_FRAME_ID="${FUSION_ODOM_FRAME_ID:-odom_imu_lidar_fused}"
fi

read_param_value() {
  local key="$1"
  python3 - "${PARAMS_FILE}" "${key}" <<'PY'
import re
import sys

params_path = sys.argv[1]
target_key = sys.argv[2]
pattern = re.compile(r"^\s*" + re.escape(target_key) + r"\s*:\s*(.*?)\s*$")

with open(params_path, "r", encoding="utf-8") as handle:
    for raw_line in handle:
        line = raw_line.split("#", 1)[0].rstrip()
        match = pattern.match(line)
        if match:
            print(match.group(1).strip().strip('"').strip("'"))
            break
PY
}

normalize_bool_override() {
  local raw_value="$1"
  local normalized
  normalized="$(printf '%s' "${raw_value}" | tr '[:upper:]' '[:lower:]')"
  case "${normalized}" in
    1|true|yes|on)
      printf 'true\n'
      ;;
    0|false|no|off)
      printf 'false\n'
      ;;
    *)
      printf '%s\n' "${raw_value}"
      ;;
  esac
}

normalize_float_override() {
  python3 - "$1" <<'PY'
import sys

try:
    print(repr(float(sys.argv[1])))
except Exception:
    print(sys.argv[1])
PY
}

first_pwm_chip_path() {
  find /sys/class/pwm -maxdepth 1 -type d -name 'pwmchip*' | sort | head -n 1
}

ensure_pwm_channel_dir() {
  local chip_path="$1"
  local channel="$2"
  local pwm_dir="${chip_path}/pwm${channel}"
  if [ ! -d "${pwm_dir}" ]; then
    echo "${channel}" > "${chip_path}/export" 2>/dev/null || true
    local waited=0
    while [ ! -d "${pwm_dir}" ] && [ "${waited}" -lt 50 ]; do
      sleep 0.05
      waited=$((waited + 1))
    done
  fi
  if [ -d "${pwm_dir}" ]; then
    printf '%s\n' "${pwm_dir}"
  fi
}

set_pwm_output_pct() {
  local pwm_dir="$1"
  local frequency_hz="$2"
  local duty_cycle_pct="$3"
  if [ -z "${pwm_dir}" ] || [ ! -d "${pwm_dir}" ]; then
    return
  fi
  python3 - "${pwm_dir}" "${frequency_hz}" "${duty_cycle_pct}" <<'PY'
import os
import sys

pwm_dir = sys.argv[1]
frequency_hz = max(1.0, float(sys.argv[2]))
duty_cycle_pct = max(0.0, min(100.0, float(sys.argv[3])))

period_ns = int(round(1.0e9 / frequency_hz))
duty_cycle_ns = int(round(period_ns * duty_cycle_pct / 100.0))

with open(os.path.join(pwm_dir, "period"), "w", encoding="utf-8") as handle:
    handle.write(f"{period_ns}\n")
with open(os.path.join(pwm_dir, "duty_cycle"), "w", encoding="utf-8") as handle:
    handle.write(f"{duty_cycle_ns}\n")
with open(os.path.join(pwm_dir, "enable"), "w", encoding="utf-8") as handle:
    handle.write("1\n")
PY
}

force_motor_neutral_from_params() {
  if [ ! -d /sys/class/pwm ]; then
    return
  fi
  local chip_path channel frequency_hz neutral_dc pwm_dir
  chip_path="$(first_pwm_chip_path)"
  channel="$(read_param_value "motor_channel")"
  frequency_hz="$(read_param_value "motor_frequency_hz")"
  neutral_dc="$(read_param_value "motor_neutral_dc")"
  pwm_dir="$(ensure_pwm_channel_dir "${chip_path}" "${channel}")"
  if [ -n "${pwm_dir}" ]; then
    set_pwm_output_pct "${pwm_dir}" "${frequency_hz}" "${neutral_dc}"
    echo "[APEX] Forced motor neutral on ${pwm_dir} at ${neutral_dc}%"
  fi
}

force_steering_center_from_params() {
  if [ ! -d /sys/class/pwm ]; then
    return
  fi
  local chip_path channel frequency_hz dc_min dc_max trim_dc min_authority_ratio center_dc pwm_dir
  chip_path="$(first_pwm_chip_path)"
  channel="$(read_param_value "steering_channel")"
  frequency_hz="$(read_param_value "steering_frequency_hz")"
  dc_min="$(read_param_value "steering_dc_min")"
  dc_max="$(read_param_value "steering_dc_max")"
  trim_dc="$(read_param_value "steering_center_trim_dc")"
  min_authority_ratio="$(read_param_value "steering_min_authority_ratio")"
  center_dc="$(python3 - "${dc_min}" "${dc_max}" "${trim_dc}" "${min_authority_ratio}" <<'PY'
import sys
dc_min = float(sys.argv[1])
dc_max = float(sys.argv[2])
trim_dc = float(sys.argv[3])
min_authority_ratio = float(sys.argv[4])
raw_dc_center = 0.5 * (dc_min + dc_max) + trim_dc
half_span = 0.5 * (dc_max - dc_min)
required_half_span = max(0.0, min(1.0, min_authority_ratio)) * half_span
dc_center_min = dc_min + required_half_span
dc_center_max = dc_max - required_half_span
if dc_center_min <= dc_center_max:
    dc_center = min(max(raw_dc_center, dc_center_min), dc_center_max)
else:
    dc_center = 0.5 * (dc_min + dc_max)
print(dc_center)
PY
)"
  pwm_dir="$(ensure_pwm_channel_dir "${chip_path}" "${channel}")"
  if [ -n "${pwm_dir}" ]; then
    set_pwm_output_pct "${pwm_dir}" "${frequency_hz}" "${center_dc}"
    echo "[APEX] Forced steering center on ${pwm_dir} at ${center_dc}%"
  fi
}

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill -INT "${pid}" 2>/dev/null || true
  done
  sleep 1
  for pid in "${PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait || true
  force_motor_neutral_from_params
  force_steering_center_from_params
}

trap cleanup INT TERM EXIT

echo "[APEX] Startup compat: ${STARTUP_COMPAT}"

is_legacy_like() {
  [[ "${STARTUP_COMPAT}" == "legacy" || "${STARTUP_COMPAT}" == "safe" ]]
}

is_safe_like() {
  [[ "${STARTUP_COMPAT}" == "safe" ]]
}

startup_stage_sleep() {
  local label="$1"
  if [[ "${STAGGERED_STARTUP}" != "1" ]]; then
    return
  fi
  if [[ -z "${NODE_STARTUP_STAGGER_S}" ]]; then
    return
  fi
  if python3 - "${NODE_STARTUP_STAGGER_S}" <<'PY'
import sys
try:
    value = float(sys.argv[1])
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if value > 1.0e-6 else 1)
PY
  then
    echo "[APEX] Startup staging: waiting ${NODE_STARTUP_STAGGER_S}s after ${label}"
    sleep "${NODE_STARTUP_STAGGER_S}"
  fi
}

track_pid() {
  local name="$1"
  local pid="$2"
  PIDS+=("${pid}")
  PID_NAMES+=("${name}")
  echo "[APEX] Started ${name} (pid=${pid})"
}

wait_for_first_exit() {
  local idx name status exited_pid
  exited_pid=""
  set +e
  wait -n -p exited_pid "${PIDS[@]}"
  status="$?"
  set -e
  name="unknown"
  for idx in "${!PIDS[@]}"; do
    if [[ "${PIDS[${idx}]}" == "${exited_pid}" ]]; then
      name="${PID_NAMES[${idx}]:-unknown}"
      break
    fi
  done
  echo "[APEX][ERROR] Child process exited: ${name} (pid=${exited_pid:-unknown}, status=${status})" >&2
  return "${status}"
}

ensure_fixed_map_assets() {
  if [[ "${FIXED_MAP_AUTOBUILD}" != "1" ]]; then
    return
  fi
  if [[ -z "${FIXED_MAP_RUN_DIR}" || -z "${FIXED_MAP_DIR}" ]]; then
    echo "[APEX][WARN] Fixed-map autobuild requested but APEX_FIXED_MAP_RUN_DIR/APEX_FIXED_MAP_DIR is missing"
    return 1
  fi
  if [[ -f "${FIXED_MAP_YAML}" && -f "${FIXED_MAP_DISTANCE_NPY}" && -f "${FIXED_MAP_VISUAL_POINTS_CSV}" && -f "${FIXED_MAP_ROUTE_CSV}" ]]; then
    echo "[APEX] Fixed-map assets already present (${FIXED_MAP_DIR})"
    return
  fi
  if [[ ! -f "${FIXED_MAP_BUILDER_SCRIPT}" ]]; then
    echo "[APEX][WARN] Fixed-map builder not found: ${FIXED_MAP_BUILDER_SCRIPT}"
    return 1
  fi
  echo "[APEX] Building fixed-map assets from ${FIXED_MAP_RUN_DIR}"
  python3 "${FIXED_MAP_BUILDER_SCRIPT}" "${FIXED_MAP_RUN_DIR}" \
    --output-dir "${FIXED_MAP_DIR}" \
    --resolution-m "${APEX_FIXED_MAP_RESOLUTION_M:-0.05}" \
    --margin-m "${APEX_FIXED_MAP_MARGIN_M:-0.75}" \
    --clearance-m "${APEX_FIXED_MAP_CLEARANCE_M:-0.15}" \
    --clearance-soft-m "${APEX_FIXED_MAP_CLEARANCE_SOFT_M:-0.45}" \
    --corridor-half-width-m "${APEX_FIXED_MAP_CORRIDOR_HALF_WIDTH_M:-0.75}" \
    --corridor-retry-half-widths-m "${APEX_FIXED_MAP_CORRIDOR_RETRY_HALF_WIDTHS_M:-0.90,1.10}" \
    --route-step-m "${APEX_FIXED_MAP_ROUTE_STEP_M:-0.05}"
}

if is_safe_like && [[ -z "${SERIAL_CONNECT_PROFILE_NAME}" ]]; then
  SERIAL_CONNECT_PROFILE_NAME="safe_passive_default"
  SERIAL_CONNECT_TOGGLE_DTR="${SERIAL_CONNECT_TOGGLE_DTR:-false}"
  SERIAL_CONNECT_DTR_LOW_S="${SERIAL_CONNECT_DTR_LOW_S:-0.0}"
  SERIAL_CONNECT_SETTLE_S="${SERIAL_CONNECT_SETTLE_S:-0.0}"
  SERIAL_FLUSH_INPUT_ON_CONNECT="${SERIAL_FLUSH_INPUT_ON_CONNECT:-false}"
fi

NANO_CMD=(python3 -m apex_telemetry.imu.nano_accel_serial_node
  --ros-args
  --params-file "${PARAMS_FILE}"
  -p "serial_port:=${APEX_SERIAL_PORT:-/dev/ttyACM0}"
  -p "baudrate:=${APEX_BAUDRATE:-115200}")
if ! is_legacy_like || is_safe_like; then
  if [[ -n "${SERIAL_CONNECT_TOGGLE_DTR}" ]]; then
    NANO_CMD+=(-p "connect_toggle_dtr:=$(normalize_bool_override "${SERIAL_CONNECT_TOGGLE_DTR}")")
  fi
  if [[ -n "${SERIAL_CONNECT_DTR_LOW_S}" ]]; then
    NANO_CMD+=(-p "connect_dtr_low_s:=${SERIAL_CONNECT_DTR_LOW_S}")
  fi
  if [[ -n "${SERIAL_CONNECT_SETTLE_S}" ]]; then
    NANO_CMD+=(-p "connect_settle_s:=${SERIAL_CONNECT_SETTLE_S}")
  fi
  if [[ -n "${SERIAL_FLUSH_INPUT_ON_CONNECT}" ]]; then
    NANO_CMD+=(-p "flush_input_on_connect:=$(normalize_bool_override "${SERIAL_FLUSH_INPUT_ON_CONNECT}")")
  fi
  if [[ -n "${SERIAL_CONNECT_PROFILE_NAME}" ]]; then
    NANO_CMD+=(-p "connection_profile_name:=${SERIAL_CONNECT_PROFILE_NAME}")
  fi
  if [[ -n "${SERIAL_NO_DATA_RECONNECT_S}" ]]; then
    NANO_CMD+=(-p "no_data_reconnect_s:=${SERIAL_NO_DATA_RECONNECT_S}")
  fi
fi
"${NANO_CMD[@]}" &
track_pid "nano serial node" "$!"
if [[ "${STAGGERED_STARTUP}" == "1" ]]; then
  echo "[APEX] Startup staging: waiting ${SERIAL_WARMUP_S}s after Nano serial node"
  sleep "${SERIAL_WARMUP_S}"
fi

if [[ "${ENABLE_KINEMATICS}" == "1" ]]; then
  python3 -m apex_telemetry.odometry.kinematics_estimator_node \
    --ros-args \
    --params-file "${PARAMS_FILE}" &
  track_pid "kinematics estimator" "$!"

  python3 -m apex_telemetry.odometry.kinematics_odometry_node \
    --ros-args \
    --params-file "${PARAMS_FILE}" &
  track_pid "kinematics odometry" "$!"
else
  echo "[APEX] Kinematics/raw odometry disabled for this run"
fi

python3 -m apex_telemetry.perception.rplidar_publisher_node \
  --ros-args \
  --params-file "${PARAMS_FILE}" \
  -p "port:=${APEX_LIDAR_PORT:-/dev/ttyUSB0}" \
  -p "baudrate:=${APEX_LIDAR_BAUDRATE:-115200}" &
track_pid "RPLidar publisher" "$!"
if [[ "${STAGGERED_STARTUP}" == "1" ]]; then
  echo "[APEX] Startup staging: waiting ${LIDAR_STARTUP_SETTLE_S}s for LiDAR motor spin-up"
  sleep "${LIDAR_STARTUP_SETTLE_S}"
fi

if [[ "${ENABLE_IMU_LIDAR_FUSION}" == "1" ]]; then
  if [[ "${ESTIMATION_BACKEND}" == "fixed_map" || "${ENABLE_FIXED_MAP_ROUTE_PLANNER}" == "1" || "${ENABLE_FIXED_MAP_PUBLISHER}" == "1" ]]; then
    ensure_fixed_map_assets
  fi
  FUSION_CMD=(python3 -m apex_telemetry.estimation.imu_lidar_planar_fusion_node
    --ros-args
    --params-file "${PARAMS_FILE}")
  if [[ -n "${ESTIMATION_BACKEND}" ]]; then
    FUSION_CMD+=(-p "estimation_backend:=${ESTIMATION_BACKEND}")
  fi
  if [[ -n "${FUSION_ODOM_TOPIC}" ]]; then
    FUSION_CMD+=(-p "odom_topic:=${FUSION_ODOM_TOPIC}")
  fi
  if [[ -n "${FUSION_PATH_TOPIC}" ]]; then
    FUSION_CMD+=(-p "path_topic:=${FUSION_PATH_TOPIC}")
  fi
  if [[ -n "${FUSION_POSE_TOPIC}" ]]; then
    FUSION_CMD+=(-p "pose_topic:=${FUSION_POSE_TOPIC}")
  fi
  if [[ -n "${FUSION_LIVE_MAP_TOPIC}" ]]; then
    FUSION_CMD+=(-p "live_map_topic:=${FUSION_LIVE_MAP_TOPIC}")
  fi
  if [[ -n "${FUSION_FULL_MAP_TOPIC}" ]]; then
    FUSION_CMD+=(-p "full_map_topic:=${FUSION_FULL_MAP_TOPIC}")
  fi
  if [[ -n "${FUSION_STATUS_TOPIC}" ]]; then
    FUSION_CMD+=(-p "status_topic:=${FUSION_STATUS_TOPIC}")
  fi
  if [[ -n "${FUSION_ODOM_FRAME_ID}" ]]; then
    FUSION_CMD+=(-p "odom_frame_id:=${FUSION_ODOM_FRAME_ID}")
  fi
  if [[ "${ESTIMATION_BACKEND}" == "fixed_map" ]]; then
    FUSION_CMD+=(
      -p "fixed_map_yaml:=${FIXED_MAP_YAML}"
      -p "fixed_map_distance_npy:=${FIXED_MAP_DISTANCE_NPY}"
      -p "fixed_map_visual_points_csv:=${FIXED_MAP_VISUAL_POINTS_CSV}"
      -p "fixed_map_route_csv:=${FIXED_MAP_ROUTE_CSV}"
    )
  fi
  "${FUSION_CMD[@]}" &
  track_pid "IMU+LiDAR fusion" "$!"
  startup_stage_sleep "IMU+LiDAR fusion"
else
  echo "[APEX] Online IMU+LiDAR fusion disabled for this run"
fi

if [[ "${ENABLE_OFFLINE_SUBMAP_REFINER}" == "1" ]]; then
  OFFLINE_REFINER_SCRIPT="${APEX_OFFLINE_REFINER_SCRIPT:-/work/repo/src/rc_sim_description/scripts/offline_submap_refiner.py}"
  if [[ -f "${OFFLINE_REFINER_SCRIPT}" ]]; then
    echo "[APEX] Offline submap refiner enabled (script=${OFFLINE_REFINER_SCRIPT})"
    python3 "${OFFLINE_REFINER_SCRIPT}" \
      --ros-args \
      -p "use_sim_time:=false" \
      -p "replay_mode:=live_buffer" \
      -p "scan_topic:=${APEX_OFFLINE_REFINER_SCAN_TOPIC:-/lidar/scan_slam}" \
      -p "imu_topic:=${APEX_OFFLINE_REFINER_IMU_TOPIC:-/apex/imu/data_raw}" \
      -p "seed_odom_topic:=${APEX_OFFLINE_REFINER_SEED_ODOM_TOPIC:-/apex/odometry/imu_lidar_fused}" \
      -p "seed_status_topic:=${APEX_OFFLINE_REFINER_SEED_STATUS_TOPIC:-/apex/estimation/status}" \
      -p "frame_id:=${APEX_OFFLINE_REFINER_FRAME_ID:-odom_imu_lidar_fused}" \
      -p "child_frame_id:=${APEX_OFFLINE_REFINER_CHILD_FRAME_ID:-offline_refined_base_link}" \
      -p "map_topic:=${APEX_OFFLINE_REFINER_MAP_TOPIC:-/apex/real/offline_refined_map}" \
      -p "grid_topic:=${APEX_OFFLINE_REFINER_GRID_TOPIC:-/apex/real/offline_refined_grid}" \
      -p "path_topic:=${APEX_OFFLINE_REFINER_PATH_TOPIC:-/apex/real/offline_refined_path}" \
      -p "submap_topic:=${APEX_OFFLINE_REFINER_SUBMAP_TOPIC:-/apex/real/offline_current_submap}" \
      -p "status_topic:=${APEX_OFFLINE_REFINER_STATUS_TOPIC:-/apex/real/offline_refined_status}" \
      -p "odom_topic:=${APEX_OFFLINE_REFINER_ODOM_TOPIC:-/apex/real/offline_refined_odom}" \
      -p "publish_global_correction:=$(normalize_bool_override "${APEX_OFFLINE_REFINER_PUBLISH_GLOBAL_CORRECTION:-true}")" \
      -p "global_correction_topic:=${APEX_OFFLINE_REFINER_GLOBAL_CORRECTION_TOPIC:-/apex/real/offline_global_correction}" \
      -p "anchor_pose_topic:=${APEX_OFFLINE_REFINER_ANCHOR_POSE_TOPIC:-/apex/real/offline_anchor_pose}" \
      -p "seed_odom_frame_id:=${APEX_OFFLINE_REFINER_SEED_ODOM_FRAME_ID:-odom_imu_lidar_fused}" \
      -p "window_scan_count:=${APEX_OFFLINE_REFINER_WINDOW_SCAN_COUNT:-48}" \
      -p "window_overlap_count:=${APEX_OFFLINE_REFINER_WINDOW_OVERLAP_COUNT:-24}" \
      -p "initial_scan_count:=${APEX_OFFLINE_REFINER_INITIAL_SCAN_COUNT:-24}" \
      -p "submap_window_scans:=${APEX_OFFLINE_REFINER_SUBMAP_WINDOW_SCANS:-10}" \
      -p "point_stride:=${APEX_OFFLINE_REFINER_POINT_STRIDE:-2}" \
      -p "max_correspondence_m:=${APEX_OFFLINE_REFINER_MAX_CORRESPONDENCE_M:-0.35}" \
      -p "offline_update_period_sec:=${APEX_OFFLINE_REFINER_UPDATE_PERIOD_S:-0.5}" \
      -p "seed_status_timeout_sec:=${APEX_OFFLINE_REFINER_SEED_STATUS_TIMEOUT_S:-2.0}" \
      -p "seed_status_max_median_submap_residual_m:=${APEX_OFFLINE_REFINER_SEED_MAX_MEDIAN_RESIDUAL_M:-0.18}" \
      -p "enable_inter_window_alignment:=$(normalize_bool_override "${APEX_OFFLINE_REFINER_ENABLE_INTER_WINDOW_ALIGNMENT:-true}")" \
      -p "inter_window_alignment_gain:=${APEX_OFFLINE_REFINER_INTER_WINDOW_ALIGNMENT_GAIN:-0.85}" \
      -p "inter_window_min_overlap_scans:=${APEX_OFFLINE_REFINER_INTER_WINDOW_MIN_OVERLAP_SCANS:-4}" \
      -p "inter_window_min_points:=${APEX_OFFLINE_REFINER_INTER_WINDOW_MIN_POINTS:-80}" \
      -p "inter_window_max_points:=${APEX_OFFLINE_REFINER_INTER_WINDOW_MAX_POINTS:-2500}" \
      -p "inter_window_max_translation_m:=${APEX_OFFLINE_REFINER_INTER_WINDOW_MAX_TRANSLATION_M:-0.25}" \
      -p "inter_window_max_yaw_deg:=${APEX_OFFLINE_REFINER_INTER_WINDOW_MAX_YAW_DEG:-10.0}" \
      -p "enable_manual_idle_finalize:=$(normalize_bool_override "${APEX_OFFLINE_REFINER_ENABLE_MANUAL_IDLE_FINALIZE:-true}")" \
      -p "manual_status_topic:=${APEX_OFFLINE_REFINER_MANUAL_STATUS_TOPIC:-/apex/manual_control/status}" \
      -p "manual_idle_timeout_s:=${APEX_OFFLINE_REFINER_MANUAL_IDLE_TIMEOUT_S:-4.0}" \
      -p "manual_motion_linear_deadband_mps:=${APEX_OFFLINE_REFINER_MANUAL_MOTION_LINEAR_DEADBAND_MPS:-0.02}" \
      -p "final_min_scan_count:=${APEX_OFFLINE_REFINER_FINAL_MIN_SCAN_COUNT:-8}" \
      -p "save_on_finalize:=$(normalize_bool_override "${APEX_OFFLINE_REFINER_SAVE_ON_FINALIZE:-true}")" \
      -p "save_output_dir:=${APEX_OFFLINE_REFINER_SAVE_OUTPUT_DIR:-/work/repo/APEX/.apex_runtime/offline_refined_maps}" &
    track_pid "offline submap refiner" "$!"
    startup_stage_sleep "offline submap refiner"
  else
    echo "[APEX][WARN] Offline submap refiner enabled but script not found: ${OFFLINE_REFINER_SCRIPT}"
  fi
else
  echo "[APEX] Offline submap refiner disabled for this run"
fi

if [[ "${ENABLE_CURVE_ENTRY_PLANNER}" == "1" ]]; then
  python3 -m apex_telemetry.perception.curve_entry_path_planner_node \
    --ros-args \
    --params-file "${PARAMS_FILE}" &
  track_pid "curve entry planner" "$!"
  startup_stage_sleep "curve planner"
else
  echo "[APEX] Curve-entry planner disabled for this run"
fi

if [[ "${ENABLE_PATH_TRACKER}" == "1" ]]; then
  python3 -m apex_telemetry.control.curve_path_tracker_node \
    --ros-args \
    --params-file "${PARAMS_FILE}" &
  track_pid "curve path tracker" "$!"
  startup_stage_sleep "path tracker"
else
  echo "[APEX] Curve path tracker disabled for this run"
fi

if [[ "${ENABLE_RECOGNITION_TOUR_PLANNER}" == "1" ]]; then
  python3 -m apex_telemetry.perception.recognition_tour_planner_node \
    --ros-args \
    --params-file "${PARAMS_FILE}" &
  track_pid "recognition tour planner" "$!"
  startup_stage_sleep "recognition planner"
else
  echo "[APEX] Recognition tour planner disabled for this run"
fi

if [[ "${ENABLE_FIXED_MAP_PUBLISHER}" == "1" ]]; then
  if [[ "${ENABLE_IMU_LIDAR_FUSION}" != "1" ]]; then
    ensure_fixed_map_assets
  fi
  FIXED_MAP_PUBLISHER_SCRIPT="${APEX_FIXED_MAP_PUBLISHER_SCRIPT:-/work/repo/APEX/tools/core/apex_fixed_map_publisher.py}"
  if [[ -f "${FIXED_MAP_PUBLISHER_SCRIPT}" ]]; then
    python3 "${FIXED_MAP_PUBLISHER_SCRIPT}" \
      --ros-args \
      -p "map_yaml:=${FIXED_MAP_YAML}" \
      -p "summary_json:=${FIXED_MAP_BUILD_STATUS_JSON}" \
      -p "frame_id:=${APEX_FIXED_MAP_FRAME_ID:-odom_imu_lidar_fused}" \
      -p "grid_topic:=${APEX_FIXED_MAP_GRID_TOPIC:-/apex/fixed_map/grid}" \
      -p "visual_points_topic:=${APEX_FIXED_MAP_VISUAL_POINTS_TOPIC:-/apex/fixed_map/visual_points}" \
      -p "path_topic:=${APEX_FIXED_MAP_RECORDED_PATH_TOPIC:-/apex/fixed_map/route_preview_path}" \
      -p "status_topic:=${APEX_FIXED_MAP_STATUS_TOPIC:-/apex/fixed_map/status}" \
      -p "reload_on_change:=false" \
      -p "allow_missing_inputs:=false" &
    track_pid "fixed map publisher" "$!"
    startup_stage_sleep "fixed map publisher"
  else
    echo "[APEX][WARN] Fixed map publisher script not found: ${FIXED_MAP_PUBLISHER_SCRIPT}"
  fi
else
  echo "[APEX] Fixed map publisher disabled for this run"
fi

if [[ "${ENABLE_FIXED_MAP_ROUTE_PLANNER}" == "1" ]]; then
  if [[ "${ENABLE_IMU_LIDAR_FUSION}" != "1" && "${ENABLE_FIXED_MAP_PUBLISHER}" != "1" ]]; then
    ensure_fixed_map_assets
  fi
  python3 -m apex_telemetry.perception.fixed_map_route_planner_node \
    --ros-args \
    --params-file "${PARAMS_FILE}" \
    -p "fixed_map_dir:=${FIXED_MAP_DIR}" \
    -p "route_csv:=${FIXED_MAP_ROUTE_CSV}" \
    -p "build_status_json:=${FIXED_MAP_BUILD_STATUS_JSON}" \
    -p "frame_id:=${APEX_FIXED_MAP_FRAME_ID:-odom_imu_lidar_fused}" \
    -p "path_topic:=${APEX_FIXED_MAP_ROUTE_PATH_TOPIC:-/apex/planning/fixed_map_path}" \
    -p "status_topic:=${APEX_FIXED_MAP_ROUTE_STATUS_TOPIC:-/apex/planning/fixed_map_status}" \
    -p "clearance_min_m:=${APEX_FIXED_MAP_CLEARANCE_M:-0.15}" &
  track_pid "fixed map route planner" "$!"
  startup_stage_sleep "fixed map route planner"
else
  echo "[APEX] Fixed map route planner disabled for this run"
fi

if [[ "${ENABLE_RECOGNITION_TOUR_TRACKER}" == "1" ]]; then
  TRACKER_CMD=(python3 -m apex_telemetry.control.recognition_tour_tracker_node
    --ros-args
    --params-file "${PARAMS_FILE}"
    -p "publish_unarmed_zero_cmd:=$(normalize_bool_override "${RECOGNITION_TRACKER_PUBLISH_UNARMED_ZERO_CMD}")")
  if [[ -n "${RECOGNITION_TRACKER_PATH_TOPIC}" ]]; then
    TRACKER_CMD+=(-p "path_topic:=${RECOGNITION_TRACKER_PATH_TOPIC}")
  fi
  if [[ -n "${RECOGNITION_TRACKER_PLANNING_STATUS_TOPIC}" ]]; then
    TRACKER_CMD+=(-p "planning_status_topic:=${RECOGNITION_TRACKER_PLANNING_STATUS_TOPIC}")
  fi
  if [[ -n "${RECOGNITION_TRACKER_ODOM_TOPIC}" ]]; then
    TRACKER_CMD+=(-p "odom_topic:=${RECOGNITION_TRACKER_ODOM_TOPIC}")
  fi
  if [[ -n "${RECOGNITION_TRACKER_FUSION_STATUS_TOPIC}" ]]; then
    TRACKER_CMD+=(-p "fusion_status_topic:=${RECOGNITION_TRACKER_FUSION_STATUS_TOPIC}")
  fi
  if [[ -n "${RECOGNITION_TRACKER_STATUS_TOPIC}" ]]; then
    TRACKER_CMD+=(-p "status_topic:=${RECOGNITION_TRACKER_STATUS_TOPIC}")
  fi
  if [[ -n "${RECOGNITION_TRACKER_REQUIRE_ARM}" ]]; then
    TRACKER_CMD+=(-p "require_arm:=$(normalize_bool_override "${RECOGNITION_TRACKER_REQUIRE_ARM}")")
  fi
  if [[ -n "${RECOGNITION_TRACKER_DEFAULT_ARMED}" ]]; then
    TRACKER_CMD+=(-p "default_armed:=$(normalize_bool_override "${RECOGNITION_TRACKER_DEFAULT_ARMED}")")
  fi
  if [[ -n "${RECOGNITION_TRACKER_PATH_STALE_MAX_AGE_S}" ]]; then
    TRACKER_CMD+=(-p "path_stale_max_age_s:=${RECOGNITION_TRACKER_PATH_STALE_MAX_AGE_S}")
  fi
  if [[ -n "${RECOGNITION_TRACKER_PATH_STALE_ABORT_HOLD_S}" ]]; then
    TRACKER_CMD+=(-p "path_stale_abort_hold_s:=${RECOGNITION_TRACKER_PATH_STALE_ABORT_HOLD_S}")
  fi
  if [[ -n "${RECOGNITION_TRACKER_REAR_AXLE_OFFSET_X_M}" ]]; then
    TRACKER_CMD+=(-p "rear_axle_offset_x_m:=${RECOGNITION_TRACKER_REAR_AXLE_OFFSET_X_M}")
  fi
  if [[ -n "${RECOGNITION_TRACKER_REAR_AXLE_OFFSET_Y_M}" ]]; then
    TRACKER_CMD+=(-p "rear_axle_offset_y_m:=${RECOGNITION_TRACKER_REAR_AXLE_OFFSET_Y_M}")
  fi
  if [[ -n "${RECOGNITION_TRACKER_ODOM_TIMEOUT_S}" ]]; then
    TRACKER_CMD+=(-p "odom_timeout_s:=${RECOGNITION_TRACKER_ODOM_TIMEOUT_S}")
  fi
  if [[ -n "${RECOGNITION_TRACKER_MIN_LOOKAHEAD_M}" ]]; then
    TRACKER_CMD+=(-p "min_lookahead_m:=${RECOGNITION_TRACKER_MIN_LOOKAHEAD_M}")
  fi
  if [[ -n "${RECOGNITION_TRACKER_MAX_LOOKAHEAD_M}" ]]; then
    TRACKER_CMD+=(-p "max_lookahead_m:=${RECOGNITION_TRACKER_MAX_LOOKAHEAD_M}")
  fi
  if [[ -n "${RECOGNITION_TRACKER_SHARP_TURN_LOOKAHEAD_MIN_M}" ]]; then
    TRACKER_CMD+=(-p "sharp_turn_lookahead_min_m:=${RECOGNITION_TRACKER_SHARP_TURN_LOOKAHEAD_MIN_M}")
  fi
  if [[ -n "${RECOGNITION_TRACKER_ANGULAR_CMD_EMA_ALPHA}" ]]; then
    TRACKER_CMD+=(-p "angular_cmd_ema_alpha:=${RECOGNITION_TRACKER_ANGULAR_CMD_EMA_ALPHA}")
  fi
  "${TRACKER_CMD[@]}" &
  track_pid "recognition tracker" "$!"
  startup_stage_sleep "recognition tracker"
else
  echo "[APEX] Recognition tour tracker disabled for this run"
fi

if [[ "${ENABLE_CMDVEL_ACTUATION_BRIDGE}" == "1" ]]; then
  echo "[APEX] CmdVel bridge launch config: min=${BRIDGE_MIN_EFFECTIVE_SPEED_PCT:-yaml}% max=${BRIDGE_MAX_SPEED_PCT:-yaml}% rev_min=${BRIDGE_MIN_EFFECTIVE_REVERSE_SPEED_PCT:-yaml}% rev_max=${BRIDGE_MAX_REVERSE_SPEED_PCT:-yaml}% launch_boost=${BRIDGE_LAUNCH_BOOST_SPEED_PCT:-yaml}% hold=${BRIDGE_LAUNCH_BOOST_HOLD_S:-yaml}s active_brake=${BRIDGE_ACTIVE_BRAKE_ON_ZERO:-yaml}"
  CMD=(python3 -m apex_telemetry.actuation.cmd_vel_to_apex_actuation_node
    --ros-args
    --params-file "${PARAMS_FILE}")
  if [[ -n "${BRIDGE_MIN_EFFECTIVE_SPEED_PCT}" ]]; then
    CMD+=(-p "min_effective_speed_pct:=$(normalize_float_override "${BRIDGE_MIN_EFFECTIVE_SPEED_PCT}")")
  fi
  if [[ -n "${BRIDGE_MAX_SPEED_PCT}" ]]; then
    CMD+=(-p "max_speed_pct:=$(normalize_float_override "${BRIDGE_MAX_SPEED_PCT}")")
  fi
  if [[ -n "${BRIDGE_MIN_EFFECTIVE_REVERSE_SPEED_PCT}" ]]; then
    CMD+=(-p "min_effective_reverse_speed_pct:=$(normalize_float_override "${BRIDGE_MIN_EFFECTIVE_REVERSE_SPEED_PCT}")")
  fi
  if [[ -n "${BRIDGE_MAX_REVERSE_SPEED_PCT}" ]]; then
    CMD+=(-p "max_reverse_speed_pct:=$(normalize_float_override "${BRIDGE_MAX_REVERSE_SPEED_PCT}")")
  fi
  if [[ -n "${BRIDGE_LAUNCH_BOOST_SPEED_PCT}" ]]; then
    CMD+=(-p "launch_boost_speed_pct:=$(normalize_float_override "${BRIDGE_LAUNCH_BOOST_SPEED_PCT}")")
  fi
  if [[ -n "${BRIDGE_LAUNCH_BOOST_HOLD_S}" ]]; then
    CMD+=(-p "launch_boost_hold_s:=$(normalize_float_override "${BRIDGE_LAUNCH_BOOST_HOLD_S}")")
  fi
  if [[ -n "${BRIDGE_ACTIVE_BRAKE_ON_ZERO}" ]]; then
    if is_legacy_like; then
      CMD+=(-p "active_brake_on_zero:=${BRIDGE_ACTIVE_BRAKE_ON_ZERO}")
    else
      CMD+=(-p "active_brake_on_zero:=$(normalize_bool_override "${BRIDGE_ACTIVE_BRAKE_ON_ZERO}")")
    fi
  fi
  if [[ -n "${BRIDGE_ACTUATION_BACKEND}" ]]; then
    CMD+=(-p "actuation_backend:=${BRIDGE_ACTUATION_BACKEND}")
  fi
  "${CMD[@]}" &
  track_pid "cmd_vel actuation bridge" "$!"
  startup_stage_sleep "cmd_vel bridge"
else
  echo "[APEX] CmdVel actuation bridge disabled for this run"
fi

if [[ "${ENABLE_MANUAL_CONTROL_BRIDGE}" == "1" ]]; then
  python3 -m apex_telemetry.control.windows_gamepad_bridge_node \
    --ros-args \
    --params-file "${PARAMS_FILE}" &
  track_pid "manual control bridge" "$!"
  startup_stage_sleep "manual control bridge"
else
  echo "[APEX] Manual control bridge disabled for this run"
fi

if [[ "${ENABLE_RECOGNITION_SESSION_MANAGER}" == "1" ]]; then
  python3 -m apex_telemetry.control.recognition_session_manager_node \
    --ros-args \
    --params-file "${PARAMS_FILE}" &
  track_pid "recognition session manager" "$!"
  startup_stage_sleep "recognition session manager"
else
  echo "[APEX] Recognition session manager disabled for this run"
fi

ros2 run tf2_ros static_transform_publisher \
  --x "${APEX_LIDAR_X_M:-0.18}" \
  --y "${APEX_LIDAR_Y_M:-0.0}" \
  --z "${APEX_LIDAR_Z_M:-0.12}" \
  --roll "${APEX_LIDAR_ROLL_RAD:-0.0}" \
  --pitch "${APEX_LIDAR_PITCH_RAD:-0.0}" \
  --yaw "${APEX_LIDAR_YAW_RAD:-0.0}" \
  --frame-id "${APEX_BASE_FRAME:-base_link}" \
  --child-frame-id "${APEX_LIDAR_FRAME:-laser}" &
track_pid "laser static transform publisher" "$!"

PIPELINE_FEATURES=("Nano raw" "LiDAR")
if [[ "${ENABLE_KINEMATICS}" == "1" ]]; then
  PIPELINE_FEATURES+=("raw odometry")
fi
if [[ "${ENABLE_IMU_LIDAR_FUSION}" == "1" ]]; then
  if [[ "${ESTIMATION_BACKEND}" == "fixed_map" ]]; then
    PIPELINE_FEATURES+=("fixed map localization")
  else
    PIPELINE_FEATURES+=("online fusion")
  fi
fi
if [[ "${ENABLE_CURVE_ENTRY_PLANNER}" == "1" ]]; then
  PIPELINE_FEATURES+=("curve planner")
fi
if [[ "${ENABLE_PATH_TRACKER}" == "1" ]]; then
  PIPELINE_FEATURES+=("path tracker")
fi
if [[ "${ENABLE_RECOGNITION_TOUR_PLANNER}" == "1" ]]; then
  PIPELINE_FEATURES+=("recognition planner")
fi
if [[ "${ENABLE_FIXED_MAP_PUBLISHER}" == "1" ]]; then
  PIPELINE_FEATURES+=("fixed map publisher")
fi
if [[ "${ENABLE_FIXED_MAP_ROUTE_PLANNER}" == "1" ]]; then
  PIPELINE_FEATURES+=("fixed map route planner")
fi
if [[ "${ENABLE_RECOGNITION_TOUR_TRACKER}" == "1" ]]; then
  PIPELINE_FEATURES+=("recognition tracker")
fi
if [[ "${ENABLE_CMDVEL_ACTUATION_BRIDGE}" == "1" ]]; then
  PIPELINE_FEATURES+=("cmd_vel bridge")
fi
if [[ "${ENABLE_MANUAL_CONTROL_BRIDGE}" == "1" ]]; then
  PIPELINE_FEATURES+=("manual control bridge")
fi
if [[ "${ENABLE_RECOGNITION_SESSION_MANAGER}" == "1" ]]; then
  PIPELINE_FEATURES+=("session manager")
fi
if [[ "${ENABLE_OFFLINE_SUBMAP_REFINER}" == "1" ]]; then
  PIPELINE_FEATURES+=("offline submap refiner")
fi
PIPELINE_FEATURE_SUMMARY=""
for FEATURE in "${PIPELINE_FEATURES[@]}"; do
  if [[ -z "${PIPELINE_FEATURE_SUMMARY}" ]]; then
    PIPELINE_FEATURE_SUMMARY="${FEATURE}"
  else
    PIPELINE_FEATURE_SUMMARY="${PIPELINE_FEATURE_SUMMARY} + ${FEATURE}"
  fi
done
echo "[APEX] Minimal raw pipeline started (${PIPELINE_FEATURE_SUMMARY})"

set +e
wait_for_first_exit
EXIT_STATUS="$?"
set -e
exit "${EXIT_STATUS}"
