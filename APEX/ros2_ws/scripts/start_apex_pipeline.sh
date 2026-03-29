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

PARAMS_FILE="/work/ros2_ws/src/apex_telemetry/config/apex_params.yaml"
PIDS=()
ENABLE_KINEMATICS="${APEX_ENABLE_KINEMATICS:-1}"
ENABLE_IMU_LIDAR_FUSION="${APEX_ENABLE_IMU_LIDAR_FUSION:-0}"

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

python3 -m apex_telemetry.imu.nano_accel_serial_node \
  --ros-args \
  --params-file "${PARAMS_FILE}" \
  -p serial_port:="${APEX_SERIAL_PORT:-/dev/ttyACM0}" \
  -p baudrate:="${APEX_BAUDRATE:-115200}" &
PIDS+=("$!")

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

if [[ "${ENABLE_IMU_LIDAR_FUSION}" == "1" ]]; then
  python3 -m apex_telemetry.estimation.imu_lidar_planar_fusion_node \
    --ros-args \
    --params-file "${PARAMS_FILE}" &
  PIDS+=("$!")
else
  echo "[APEX] Online IMU+LiDAR fusion disabled for this run"
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

if [[ "${ENABLE_KINEMATICS}" == "1" && "${ENABLE_IMU_LIDAR_FUSION}" == "1" ]]; then
  echo "[APEX] Minimal raw pipeline started (Nano raw + LiDAR + raw odometry + online fusion)"
elif [[ "${ENABLE_KINEMATICS}" == "1" ]]; then
  echo "[APEX] Minimal raw pipeline started (Nano raw + LiDAR + raw odometry)"
elif [[ "${ENABLE_IMU_LIDAR_FUSION}" == "1" ]]; then
  echo "[APEX] Minimal raw pipeline started (Nano raw + LiDAR + online fusion)"
else
  echo "[APEX] Minimal raw pipeline started (Nano raw + LiDAR)"
fi

wait -n "${PIDS[@]}"
