#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTAINER_NAME="${APEX_CONTAINER_NAME:-apex_pipeline}"
ROS_SETUP_SCRIPT="${APEX_ROS_SETUP_SCRIPT:-/opt/ros/jazzy/setup.bash}"
REMOTE_OUTPUT_ROOT="${APEX_RECT_SENSORFUS_ROOT:-${APEX_ROOT}/ros2_ws/apex_rect_sensorfus}"
CONTAINER_OUTPUT_ROOT="${APEX_RECT_SENSORFUS_CONTAINER_ROOT:-/work/ros2_ws/apex_rect_sensorfus}"
CAPTURE_DURATION_S="${APEX_RECT_SENSORFUS_CAPTURE_DURATION_S:-5.0}"
DRIVE_DELAY_S="${APEX_RECT_SENSORFUS_DRIVE_DELAY_S:-1.0}"
DRIVE_DURATION_S="${APEX_RECT_SENSORFUS_DRIVE_DURATION_S:-1.0}"
DRIVE_SPEED_PCT="${APEX_RECT_SENSORFUS_SPEED_PCT:-20.0}"
LAUNCH_SPEED_PCT="${APEX_RECT_SENSORFUS_LAUNCH_SPEED_PCT:-35.0}"
LAUNCH_DURATION_S="${APEX_RECT_SENSORFUS_LAUNCH_DURATION_S:-0.35}"
STEERING_DEG="${APEX_RECT_SENSORFUS_STEERING_DEG:-0.0}"
RUN_ID="${APEX_RECT_SENSORFUS_RUN_ID:-rect_capture}"

usage() {
  cat <<'EOF'
Usage: apex_rect_sensorfus_capture.sh [options]

Options:
  --run-id <id>
  --capture-duration-s <seconds>
  --drive-delay-s <seconds>
  --drive-duration-s <seconds>
  --speed-pct <pct>
  --launch-speed-pct <pct>
  --launch-duration-s <seconds>
  --steering-deg <deg>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --capture-duration-s)
      CAPTURE_DURATION_S="${2:-}"
      shift 2
      ;;
    --drive-delay-s)
      DRIVE_DELAY_S="${2:-}"
      shift 2
      ;;
    --drive-duration-s)
      DRIVE_DURATION_S="${2:-}"
      shift 2
      ;;
    --speed-pct)
      DRIVE_SPEED_PCT="${2:-}"
      shift 2
      ;;
    --launch-speed-pct)
      LAUNCH_SPEED_PCT="${2:-}"
      shift 2
      ;;
    --launch-duration-s)
      LAUNCH_DURATION_S="${2:-}"
      shift 2
      ;;
    --steering-deg)
      STEERING_DEG="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[APEX][ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "[APEX][ERROR] ${CONTAINER_NAME} is not running. Start it with tools/core/apex_core_up.sh" >&2
  exit 1
fi

read_param_value() {
  local key="$1"
  python3 - "${APEX_ROOT}/ros2_ws/src/apex_telemetry/config/apex_params.yaml" "${key}" <<'PY'
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

set_pwm_output_pct_in_container() {
  local channel="$1"
  local frequency_hz="$2"
  local duty_cycle_pct="$3"
  local keep_enabled="${4:-1}"
  docker exec "${CONTAINER_NAME}" /bin/bash -lc "python3 - '${channel}' '${frequency_hz}' '${duty_cycle_pct}' '${keep_enabled}' <<'PY'
import glob
import os
import sys
import time

channel = int(sys.argv[1])
frequency_hz = max(1.0, float(sys.argv[2]))
duty_cycle_pct = max(0.0, min(100.0, float(sys.argv[3])))
keep_enabled = int(sys.argv[4]) != 0

chips = sorted(glob.glob('/sys/class/pwm/pwmchip*'))
if not chips:
    raise SystemExit('No PWM chip found under /sys/class/pwm')
chip_path = chips[0]
pwm_dir = os.path.join(chip_path, f'pwm{channel}')
if not os.path.isdir(pwm_dir):
    try:
        with open(os.path.join(chip_path, 'export'), 'w', encoding='utf-8') as handle:
            handle.write(f'{channel}\n')
    except OSError:
        pass
    deadline = time.time() + 5.0
    while not os.path.isdir(pwm_dir):
        if time.time() >= deadline:
            raise SystemExit(f'Timed out waiting for {pwm_dir}')
        time.sleep(0.05)

period_ns = int(round(1.0e9 / frequency_hz))
duty_cycle_ns = int(round(period_ns * duty_cycle_pct / 100.0))
with open(os.path.join(pwm_dir, 'period'), 'w', encoding='utf-8') as handle:
    handle.write(f'{period_ns}\n')
with open(os.path.join(pwm_dir, 'duty_cycle'), 'w', encoding='utf-8') as handle:
    handle.write(f'{duty_cycle_ns}\n')
with open(os.path.join(pwm_dir, 'enable'), 'w', encoding='utf-8') as handle:
    handle.write('1\n' if keep_enabled else '0\n')
print(f'[APEX] PWM channel {channel} -> {duty_cycle_pct:.3f}%')
PY"
}

set_steering_from_params() {
  local steering_deg="$1"
  local channel frequency_hz dc_min dc_max trim_dc direction_sign duty_cycle
  channel="$(read_param_value "steering_channel")"
  frequency_hz="$(read_param_value "steering_frequency_hz")"
  dc_min="$(read_param_value "steering_dc_min")"
  dc_max="$(read_param_value "steering_dc_max")"
  trim_dc="$(read_param_value "steering_center_trim_dc")"
  direction_sign="$(read_param_value "steering_direction_sign")"
  duty_cycle="$(python3 - "${steering_deg}" "${dc_min}" "${dc_max}" "${trim_dc}" "${direction_sign}" <<'PY'
import sys
steering_deg = float(sys.argv[1])
dc_min = float(sys.argv[2])
dc_max = float(sys.argv[3])
trim_dc = float(sys.argv[4])
direction_sign = -1.0 if float(sys.argv[5]) < 0.0 else 1.0
limit_deg = 18.0
center = 0.5 * (dc_min + dc_max) + trim_dc
variation_per_deg = 0.5 * (dc_max - dc_min) / limit_deg
bounded = max(-limit_deg, min(limit_deg, steering_deg))
signed = bounded * direction_sign
print(center + signed * variation_per_deg)
PY
)"
  set_pwm_output_pct_in_container "${channel}" "${frequency_hz}" "${duty_cycle}" 1
}

set_motor_speed_from_params() {
  local speed_pct="$1"
  local channel frequency_hz dc_min dc_max neutral_dc duty_cycle
  channel="$(read_param_value "motor_channel")"
  frequency_hz="$(read_param_value "motor_frequency_hz")"
  dc_min="$(read_param_value "motor_dc_min")"
  dc_max="$(read_param_value "motor_dc_max")"
  neutral_dc="$(read_param_value "motor_neutral_dc")"
  duty_cycle="$(python3 - "${speed_pct}" "${dc_min}" "${dc_max}" "${neutral_dc}" <<'PY'
import sys
speed_pct = max(-100.0, min(100.0, float(sys.argv[1])))
dc_min = float(sys.argv[2])
dc_max = float(sys.argv[3])
neutral_dc = float(sys.argv[4])
if abs(speed_pct) < 1.0:
    duty = neutral_dc
elif speed_pct > 0.0:
    duty = neutral_dc + ((speed_pct / 100.0) * (dc_max - neutral_dc))
else:
    duty = neutral_dc + ((speed_pct / 100.0) * (neutral_dc - dc_min))
print(duty)
PY
)"
  set_pwm_output_pct_in_container "${channel}" "${frequency_hz}" "${duty_cycle}" 1
}

hold_motor_neutral() {
  set_motor_speed_from_params 0.0
}

center_steering() {
  set_steering_from_params 0.0
}

CAPTURE_NAME="${RUN_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
HOST_RUN_DIR="${REMOTE_OUTPUT_ROOT}/${CAPTURE_NAME}"
CONTAINER_RUN_DIR="${CONTAINER_OUTPUT_ROOT}/${CAPTURE_NAME}"
CONTAINER_PARAMS_FILE="/work/ros2_ws/src/apex_telemetry/config/apex_params.yaml"
mkdir -p "${HOST_RUN_DIR}"

cleanup() {
  hold_motor_neutral || true
  center_steering || true
  if [[ -n "${CAPTURE_PID:-}" ]]; then
    wait "${CAPTURE_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

cat > "${HOST_RUN_DIR}/capture_meta.json" <<EOF
{
  "run_id": "${CAPTURE_NAME}",
  "capture_duration_s": ${CAPTURE_DURATION_S},
  "drive_delay_s": ${DRIVE_DELAY_S},
  "drive_duration_s": ${DRIVE_DURATION_S},
  "speed_pct": ${DRIVE_SPEED_PCT},
  "launch_speed_pct": ${LAUNCH_SPEED_PCT},
  "launch_duration_s": ${LAUNCH_DURATION_S},
  "steering_deg": ${STEERING_DEG}
}
EOF

docker exec "${CONTAINER_NAME}" /bin/bash -lc \
  "source '${ROS_SETUP_SCRIPT}' && python3 /work/ros2_ws/scripts/capture/record_rect_sensorfus_capture.py \
    --imu-output '${CONTAINER_RUN_DIR}/imu_raw.csv' \
    --lidar-output '${CONTAINER_RUN_DIR}/lidar_points.csv' \
    --duration-s '${CAPTURE_DURATION_S}'" \
  > "${HOST_RUN_DIR}/capture.log" 2>&1 &
CAPTURE_PID=$!

if ! docker exec "${CONTAINER_NAME}" /bin/bash -lc \
  "source '${ROS_SETUP_SCRIPT}' && python3 /work/ros2_ws/scripts/capture/rect_sensorfus_actuation.py \
    --params-file '${CONTAINER_PARAMS_FILE}' \
    --trace-output '${CONTAINER_RUN_DIR}/pwm_trace.csv' \
    --drive-delay-s '${DRIVE_DELAY_S}' \
    --drive-duration-s '${DRIVE_DURATION_S}' \
    --speed-pct '${DRIVE_SPEED_PCT}' \
    --launch-speed-pct '${LAUNCH_SPEED_PCT}' \
    --launch-duration-s '${LAUNCH_DURATION_S}' \
    --steering-deg '${STEERING_DEG}'" \
  > "${HOST_RUN_DIR}/actuation.log" 2>&1
then
  echo "[APEX][ERROR] Actuation process failed. Log: ${HOST_RUN_DIR}/actuation.log" >&2
  if [[ -f "${HOST_RUN_DIR}/actuation.log" ]]; then
    tail -n 80 "${HOST_RUN_DIR}/actuation.log" >&2 || true
  fi
  exit 1
fi

if ! wait "${CAPTURE_PID}"; then
  echo "[APEX][ERROR] Sensor capture process failed. Log: ${HOST_RUN_DIR}/capture.log" >&2
  if [[ -f "${HOST_RUN_DIR}/capture.log" ]]; then
    tail -n 80 "${HOST_RUN_DIR}/capture.log" >&2 || true
  fi
  exit 1
fi
echo "[APEX] Rect sensor capture ready: ${HOST_RUN_DIR}"
echo "[APEX] Files:"
echo "  ${HOST_RUN_DIR}/imu_raw.csv"
echo "  ${HOST_RUN_DIR}/lidar_points.csv"
echo "  ${HOST_RUN_DIR}/pwm_trace.csv"
