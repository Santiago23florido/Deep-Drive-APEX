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
ENABLE_KINEMATICS="${APEX_ENABLE_KINEMATICS:-1}"
ENABLE_IMU_LIDAR_FUSION="${APEX_ENABLE_IMU_LIDAR_FUSION:-0}"
ENABLE_CURVE_ENTRY_PLANNER="${APEX_ENABLE_CURVE_ENTRY_PLANNER:-0}"
ENABLE_PATH_TRACKER="${APEX_ENABLE_PATH_TRACKER:-0}"
ENABLE_RECOGNITION_TOUR_PLANNER="${APEX_ENABLE_RECOGNITION_TOUR_PLANNER:-0}"
ENABLE_RECOGNITION_TOUR_TRACKER="${APEX_ENABLE_RECOGNITION_TOUR_TRACKER:-0}"
ENABLE_CMDVEL_ACTUATION_BRIDGE="${APEX_ENABLE_CMDVEL_ACTUATION_BRIDGE:-0}"
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
BRIDGE_LAUNCH_BOOST_SPEED_PCT="${APEX_BRIDGE_LAUNCH_BOOST_SPEED_PCT:-}"
BRIDGE_LAUNCH_BOOST_HOLD_S="${APEX_BRIDGE_LAUNCH_BOOST_HOLD_S:-}"
BRIDGE_ACTIVE_BRAKE_ON_ZERO="${APEX_BRIDGE_ACTIVE_BRAKE_ON_ZERO:-}"

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
  local chip_path channel frequency_hz dc_min dc_max trim_dc center_dc pwm_dir
  chip_path="$(first_pwm_chip_path)"
  channel="$(read_param_value "steering_channel")"
  frequency_hz="$(read_param_value "steering_frequency_hz")"
  dc_min="$(read_param_value "steering_dc_min")"
  dc_max="$(read_param_value "steering_dc_max")"
  trim_dc="$(read_param_value "steering_center_trim_dc")"
  center_dc="$(python3 - "${dc_min}" "${dc_max}" "${trim_dc}" <<'PY'
import sys
dc_min = float(sys.argv[1])
dc_max = float(sys.argv[2])
trim_dc = float(sys.argv[3])
print((0.5 * (dc_min + dc_max)) + trim_dc)
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
PIDS+=("$!")
if [[ "${STAGGERED_STARTUP}" == "1" ]]; then
  echo "[APEX] Startup staging: waiting ${SERIAL_WARMUP_S}s after Nano serial node"
  sleep "${SERIAL_WARMUP_S}"
fi

if [[ "${ENABLE_KINEMATICS}" == "1" ]]; then
  python3 -m apex_telemetry.odometry.kinematics_estimator_node \
    --ros-args \
    --params-file "${PARAMS_FILE}" &
  PIDS+=("$!")

  python3 -m apex_telemetry.odometry.kinematics_odometry_node \
    --ros-args \
    --params-file "${PARAMS_FILE}" &
  PIDS+=("$!")
else
  echo "[APEX] Kinematics/raw odometry disabled for this run"
fi

python3 -m apex_telemetry.perception.rplidar_publisher_node \
  --ros-args \
  --params-file "${PARAMS_FILE}" \
  -p "port:=${APEX_LIDAR_PORT:-/dev/ttyUSB0}" \
  -p "baudrate:=${APEX_LIDAR_BAUDRATE:-115200}" &
PIDS+=("$!")
if [[ "${STAGGERED_STARTUP}" == "1" ]]; then
  echo "[APEX] Startup staging: waiting ${LIDAR_STARTUP_SETTLE_S}s for LiDAR motor spin-up"
  sleep "${LIDAR_STARTUP_SETTLE_S}"
fi

if [[ "${ENABLE_IMU_LIDAR_FUSION}" == "1" ]]; then
  python3 -m apex_telemetry.estimation.imu_lidar_planar_fusion_node \
    --ros-args \
    --params-file "${PARAMS_FILE}" &
  PIDS+=("$!")
  startup_stage_sleep "online fusion"
else
  echo "[APEX] Online IMU+LiDAR fusion disabled for this run"
fi

if [[ "${ENABLE_CURVE_ENTRY_PLANNER}" == "1" ]]; then
  python3 -m apex_telemetry.perception.curve_entry_path_planner_node \
    --ros-args \
    --params-file "${PARAMS_FILE}" &
  PIDS+=("$!")
  startup_stage_sleep "curve planner"
else
  echo "[APEX] Curve-entry planner disabled for this run"
fi

if [[ "${ENABLE_PATH_TRACKER}" == "1" ]]; then
  python3 -m apex_telemetry.control.curve_path_tracker_node \
    --ros-args \
    --params-file "${PARAMS_FILE}" &
  PIDS+=("$!")
  startup_stage_sleep "path tracker"
else
  echo "[APEX] Curve path tracker disabled for this run"
fi

if [[ "${ENABLE_RECOGNITION_TOUR_PLANNER}" == "1" ]]; then
  python3 -m apex_telemetry.perception.recognition_tour_planner_node \
    --ros-args \
    --params-file "${PARAMS_FILE}" &
  PIDS+=("$!")
  startup_stage_sleep "recognition planner"
else
  echo "[APEX] Recognition tour planner disabled for this run"
fi

if [[ "${ENABLE_RECOGNITION_TOUR_TRACKER}" == "1" ]]; then
  python3 -m apex_telemetry.control.recognition_tour_tracker_node \
    --ros-args \
    --params-file "${PARAMS_FILE}" &
  PIDS+=("$!")
  startup_stage_sleep "recognition tracker"
else
  echo "[APEX] Recognition tour tracker disabled for this run"
fi

if [[ "${ENABLE_CMDVEL_ACTUATION_BRIDGE}" == "1" ]]; then
  echo "[APEX] CmdVel bridge launch config: min=${BRIDGE_MIN_EFFECTIVE_SPEED_PCT:-yaml}% max=${BRIDGE_MAX_SPEED_PCT:-yaml}% launch_boost=${BRIDGE_LAUNCH_BOOST_SPEED_PCT:-yaml}% hold=${BRIDGE_LAUNCH_BOOST_HOLD_S:-yaml}s active_brake=${BRIDGE_ACTIVE_BRAKE_ON_ZERO:-yaml}"
  CMD=(python3 -m apex_telemetry.actuation.cmd_vel_to_apex_actuation_node
    --ros-args
    --params-file "${PARAMS_FILE}")
  if [[ -n "${BRIDGE_MIN_EFFECTIVE_SPEED_PCT}" ]]; then
    CMD+=(-p "min_effective_speed_pct:=${BRIDGE_MIN_EFFECTIVE_SPEED_PCT}")
  fi
  if [[ -n "${BRIDGE_MAX_SPEED_PCT}" ]]; then
    CMD+=(-p "max_speed_pct:=${BRIDGE_MAX_SPEED_PCT}")
  fi
  if [[ -n "${BRIDGE_LAUNCH_BOOST_SPEED_PCT}" ]]; then
    CMD+=(-p "launch_boost_speed_pct:=${BRIDGE_LAUNCH_BOOST_SPEED_PCT}")
  fi
  if [[ -n "${BRIDGE_LAUNCH_BOOST_HOLD_S}" ]]; then
    CMD+=(-p "launch_boost_hold_s:=${BRIDGE_LAUNCH_BOOST_HOLD_S}")
  fi
  if [[ -n "${BRIDGE_ACTIVE_BRAKE_ON_ZERO}" ]]; then
    if is_legacy_like; then
      CMD+=(-p "active_brake_on_zero:=${BRIDGE_ACTIVE_BRAKE_ON_ZERO}")
    else
      CMD+=(-p "active_brake_on_zero:=$(normalize_bool_override "${BRIDGE_ACTIVE_BRAKE_ON_ZERO}")")
    fi
  fi
  "${CMD[@]}" &
  PIDS+=("$!")
  startup_stage_sleep "cmd_vel bridge"
else
  echo "[APEX] CmdVel actuation bridge disabled for this run"
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
PIDS+=("$!")

PIPELINE_FEATURES=("Nano raw" "LiDAR")
if [[ "${ENABLE_KINEMATICS}" == "1" ]]; then
  PIPELINE_FEATURES+=("raw odometry")
fi
if [[ "${ENABLE_IMU_LIDAR_FUSION}" == "1" ]]; then
  PIPELINE_FEATURES+=("online fusion")
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
if [[ "${ENABLE_RECOGNITION_TOUR_TRACKER}" == "1" ]]; then
  PIPELINE_FEATURES+=("recognition tracker")
fi
if [[ "${ENABLE_CMDVEL_ACTUATION_BRIDGE}" == "1" ]]; then
  PIPELINE_FEATURES+=("cmd_vel bridge")
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

wait -n "${PIDS[@]}"
