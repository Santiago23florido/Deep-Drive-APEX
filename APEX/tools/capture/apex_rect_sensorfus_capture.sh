#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTAINER_NAME="${APEX_CONTAINER_NAME:-apex_pipeline}"
ROS_SETUP_SCRIPT="${APEX_ROS_SETUP_SCRIPT:-/opt/ros/jazzy/setup.bash}"
REMOTE_OUTPUT_ROOT="${APEX_RECT_SENSORFUS_ROOT:-${APEX_ROOT}/ros2_ws/apex_rect_sensorfus}"
CONTAINER_OUTPUT_ROOT="${APEX_RECT_SENSORFUS_CONTAINER_ROOT:-/work/ros2_ws/apex_rect_sensorfus}"
READY_TIMEOUT_S="${APEX_RECT_SENSORFUS_READY_TIMEOUT_S:-8.0}"
READY_MIN_IMU_MESSAGES="${APEX_RECT_SENSORFUS_READY_MIN_IMU_MESSAGES:-5}"
READY_MIN_SCAN_MESSAGES="${APEX_RECT_SENSORFUS_READY_MIN_SCAN_MESSAGES:-3}"
ARM_BEFORE_READY="${APEX_RECT_SENSORFUS_ARM_BEFORE_READY:-1}"
PRE_READY_NEUTRAL_HOLD_S="${APEX_RECT_SENSORFUS_PRE_READY_NEUTRAL_HOLD_S:-0.0}"
CAPTURE_WARMUP_S="${APEX_RECT_SENSORFUS_CAPTURE_WARMUP_S:-0.30}"
CAPTURE_DURATION_S="${APEX_RECT_SENSORFUS_CAPTURE_DURATION_S:-5.0}"
DRIVE_DELAY_S="${APEX_RECT_SENSORFUS_DRIVE_DELAY_S:-1.0}"
DRIVE_DURATION_S="${APEX_RECT_SENSORFUS_DRIVE_DURATION_S:-1.0}"
DRIVE_SPEED_PCT="${APEX_RECT_SENSORFUS_SPEED_PCT:-20.0}"
LAUNCH_SPEED_PCT="${APEX_RECT_SENSORFUS_LAUNCH_SPEED_PCT:-35.0}"
LAUNCH_DURATION_S="${APEX_RECT_SENSORFUS_LAUNCH_DURATION_S:-0.35}"
STEERING_DEG="${APEX_RECT_SENSORFUS_STEERING_DEG:-0.0}"
PRE_ARM_NEUTRAL_S="${APEX_RECT_SENSORFUS_PRE_ARM_NEUTRAL_S:-0.8}"
SYSFS_MONITOR_SAMPLE_DT_S="${APEX_RECT_SENSORFUS_SYSFS_MONITOR_SAMPLE_DT_S:-0.02}"
RUN_ID="${APEX_RECT_SENSORFUS_RUN_ID:-rect_capture}"
RECORD_ONLINE_FUSION="${APEX_RECT_SENSORFUS_RECORD_ONLINE_FUSION:-0}"
COMPARE_ONLINE_FUSION="${APEX_RECT_SENSORFUS_COMPARE_ONLINE_FUSION:-0}"

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
  --pre-arm-neutral-s <seconds>
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
    --pre-arm-neutral-s)
      PRE_ARM_NEUTRAL_S="${2:-}"
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
  local keep_enabled="${2:-1}"
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
  set_pwm_output_pct_in_container "${channel}" "${frequency_hz}" "${duty_cycle}" "${keep_enabled}"
}

set_motor_speed_from_params() {
  local speed_pct="$1"
  local keep_enabled="${2:-1}"
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
  set_pwm_output_pct_in_container "${channel}" "${frequency_hz}" "${duty_cycle}" "${keep_enabled}"
}

hold_motor_neutral() {
  local keep_enabled="${1:-0}"
  set_motor_speed_from_params 0.0 "${keep_enabled}"
}

center_steering() {
  local keep_enabled="${1:-0}"
  set_steering_from_params 0.0 "${keep_enabled}"
}

write_pwm_snapshot() {
  local output_path="$1"
  docker exec "${CONTAINER_NAME}" /bin/bash -lc "
    echo '# pwm_snapshot'
    echo '# timestamp_utc '\"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
    if [ ! -d /sys/class/pwm ]; then
      echo '/sys/class/pwm is not available'
      exit 0
    fi
    find /sys/class/pwm -maxdepth 4 -type f \\( -name enable -o -name period -o -name duty_cycle \\) | sort | while read -r path; do
      echo \"## \${path}\"
      cat \"\${path}\" 2>/dev/null || echo '<unreadable>'
      echo
    done
  " > "${output_path}" 2>&1 || true
}

CAPTURE_NAME="${RUN_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
HOST_RUN_DIR="${REMOTE_OUTPUT_ROOT}/${CAPTURE_NAME}"
CONTAINER_RUN_DIR="${CONTAINER_OUTPUT_ROOT}/${CAPTURE_NAME}"
CONTAINER_PARAMS_FILE="/work/ros2_ws/src/apex_telemetry/config/apex_params.yaml"
mkdir -p "${HOST_RUN_DIR}"

MONITOR_DURATION_S="$(python3 - "${READY_TIMEOUT_S}" "${PRE_READY_NEUTRAL_HOLD_S}" "${CAPTURE_DURATION_S}" "${CAPTURE_WARMUP_S}" "${PRE_ARM_NEUTRAL_S}" "${DRIVE_DELAY_S}" "${LAUNCH_DURATION_S}" "${DRIVE_DURATION_S}" <<'PY'
import sys

ready_timeout_s = float(sys.argv[1])
pre_ready_neutral_hold_s = float(sys.argv[2])
capture_duration_s = float(sys.argv[3])
capture_warmup_s = float(sys.argv[4])
pre_arm_neutral_s = float(sys.argv[5])
drive_delay_s = float(sys.argv[6])
launch_duration_s = float(sys.argv[7])
drive_duration_s = float(sys.argv[8])
monitor_duration_s = max(
    pre_ready_neutral_hold_s + ready_timeout_s + capture_duration_s + 1.50,
    pre_ready_neutral_hold_s
    + ready_timeout_s
    + capture_warmup_s
    + pre_arm_neutral_s
    + drive_delay_s
    + launch_duration_s
    + drive_duration_s
    + 1.50,
)
print(f"{monitor_duration_s:.3f}")
PY
)"
DEFAULT_ONLINE_FUSION_RECORDER_DURATION_S="$(python3 - "${CAPTURE_DURATION_S}" "${CAPTURE_WARMUP_S}" "${PRE_ARM_NEUTRAL_S}" "${DRIVE_DELAY_S}" "${LAUNCH_DURATION_S}" "${DRIVE_DURATION_S}" <<'PY'
import sys

capture_duration_s = float(sys.argv[1])
capture_warmup_s = float(sys.argv[2])
pre_arm_neutral_s = float(sys.argv[3])
drive_delay_s = float(sys.argv[4])
launch_duration_s = float(sys.argv[5])
drive_duration_s = float(sys.argv[6])
minimum_window_s = (
    capture_warmup_s
    + pre_arm_neutral_s
    + drive_delay_s
    + launch_duration_s
    + drive_duration_s
    + 1.50
)
print(f"{max(capture_duration_s + 1.50, minimum_window_s):.3f}")
PY
)"
ONLINE_FUSION_RECORDER_DURATION_S="${APEX_RECT_SENSORFUS_ONLINE_FUSION_RECORDER_DURATION_S:-${DEFAULT_ONLINE_FUSION_RECORDER_DURATION_S}}"

finalize_bundle_artifacts() {
  if [[ -n "${ONLINE_FUSION_RECORDER_PID:-}" ]]; then
    wait "${ONLINE_FUSION_RECORDER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SYSFS_MONITOR_PID:-}" ]]; then
    wait "${SYSFS_MONITOR_PID}" 2>/dev/null || true
  fi
  write_pwm_snapshot "${HOST_RUN_DIR}/pwm_snapshot_after.txt"
  docker logs --tail 120 "${CONTAINER_NAME}" > "${HOST_RUN_DIR}/docker_tail.log" 2>&1 || true
  if [[ -f "${APEX_ROOT}/tools/analysis/analyze_forward_raw_capture.py" && -f "${HOST_RUN_DIR}/pwm_trace.csv" ]]; then
    python3 "${APEX_ROOT}/tools/analysis/analyze_forward_raw_capture.py" \
      --run-dir "${HOST_RUN_DIR}" \
      > "${HOST_RUN_DIR}/debug_summary.txt" 2>&1 || true
  fi
  if [[ "${COMPARE_ONLINE_FUSION}" == "1" && -f "${APEX_ROOT}/tools/analysis/compare_online_offline_fusion.py" ]]; then
    python3 "${APEX_ROOT}/tools/analysis/compare_online_offline_fusion.py" \
      --run-dir "${HOST_RUN_DIR}" \
      > "${HOST_RUN_DIR}/analysis_sensor_fusion_comparison.log" 2>&1 || true
  fi
}

run_actuation_process() {
  docker exec "${CONTAINER_NAME}" /bin/bash -lc \
    "source '${ROS_SETUP_SCRIPT}' && python3 /work/ros2_ws/scripts/capture/rect_sensorfus_actuation.py \
      --params-file '${CONTAINER_PARAMS_FILE}' \
      --trace-output '${CONTAINER_RUN_DIR}/pwm_trace.csv' \
      --events-output '${CONTAINER_RUN_DIR}/actuation_events.csv' \
      --pre-arm-neutral-s '${PRE_ARM_NEUTRAL_S}' \
      --drive-delay-s '${DRIVE_DELAY_S}' \
      --drive-duration-s '${DRIVE_DURATION_S}' \
      --speed-pct '${DRIVE_SPEED_PCT}' \
      --launch-speed-pct '${LAUNCH_SPEED_PCT}' \
      --launch-duration-s '${LAUNCH_DURATION_S}' \
      --steering-deg '${STEERING_DEG}'" \
    > "${HOST_RUN_DIR}/actuation.log" 2>&1
}

cleanup() {
  hold_motor_neutral 0 || true
  center_steering 0 || true
  if [[ -n "${CAPTURE_PID:-}" ]]; then
    wait "${CAPTURE_PID}" 2>/dev/null || true
  fi
  if [[ -n "${ONLINE_FUSION_RECORDER_PID:-}" ]]; then
    wait "${ONLINE_FUSION_RECORDER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

cat > "${HOST_RUN_DIR}/capture_meta.json" <<EOF
{
  "run_id": "${CAPTURE_NAME}",
  "actuation_mode": "forward_raw",
  "ready_timeout_s": ${READY_TIMEOUT_S},
  "ready_min_imu_messages": ${READY_MIN_IMU_MESSAGES},
  "ready_min_scan_messages": ${READY_MIN_SCAN_MESSAGES},
  "arm_before_ready": ${ARM_BEFORE_READY},
  "pre_ready_neutral_hold_s": ${PRE_READY_NEUTRAL_HOLD_S},
  "capture_warmup_s": ${CAPTURE_WARMUP_S},
  "pre_arm_neutral_s": ${PRE_ARM_NEUTRAL_S},
  "sysfs_monitor_sample_dt_s": ${SYSFS_MONITOR_SAMPLE_DT_S},
  "sysfs_monitor_duration_s": ${MONITOR_DURATION_S},
  "record_online_fusion": ${RECORD_ONLINE_FUSION},
  "compare_online_fusion": ${COMPARE_ONLINE_FUSION},
  "online_fusion_recorder_duration_s": ${ONLINE_FUSION_RECORDER_DURATION_S},
  "capture_duration_s": ${CAPTURE_DURATION_S},
  "drive_delay_s": ${DRIVE_DELAY_S},
  "drive_duration_s": ${DRIVE_DURATION_S},
  "speed_pct": ${DRIVE_SPEED_PCT},
  "launch_speed_pct": ${LAUNCH_SPEED_PCT},
  "launch_duration_s": ${LAUNCH_DURATION_S},
  "steering_deg": ${STEERING_DEG}
}
EOF

if [[ "${ARM_BEFORE_READY}" == "1" ]]; then
  hold_motor_neutral 1
  center_steering 1
  if python3 - "${PRE_READY_NEUTRAL_HOLD_S}" <<'PY'
import sys
sys.exit(0 if float(sys.argv[1]) > 0.0 else 1)
PY
  then
    sleep "${PRE_READY_NEUTRAL_HOLD_S}"
  fi
fi

write_pwm_snapshot "${HOST_RUN_DIR}/pwm_snapshot_before.txt"

docker exec "${CONTAINER_NAME}" /bin/bash -lc \
  "source '${ROS_SETUP_SCRIPT}' && python3 /work/ros2_ws/scripts/capture/monitor_pwm_sysfs.py \
    --params-file '${CONTAINER_PARAMS_FILE}' \
    --output '${CONTAINER_RUN_DIR}/sysfs_pwm_monitor.csv' \
    --duration-s '${MONITOR_DURATION_S}' \
    --sample-dt-s '${SYSFS_MONITOR_SAMPLE_DT_S}'" \
  > "${HOST_RUN_DIR}/sysfs_monitor.log" 2>&1 &
SYSFS_MONITOR_PID=$!

if ! docker exec "${CONTAINER_NAME}" /bin/bash -lc \
  "source '${ROS_SETUP_SCRIPT}' && python3 /work/ros2_ws/scripts/capture/wait_raw_pipeline_ready.py \
    --imu-topic /apex/imu/data_raw \
    --scan-topic /lidar/scan_localization \
    --timeout-s '${READY_TIMEOUT_S}' \
    --min-imu-messages '${READY_MIN_IMU_MESSAGES}' \
    --min-scan-messages '${READY_MIN_SCAN_MESSAGES}' \
    --json-output '${CONTAINER_RUN_DIR}/readiness.json'" \
  > "${HOST_RUN_DIR}/readiness.log" 2>&1
then
  echo "[APEX][ERROR] Raw pipeline readiness check failed. Log: ${HOST_RUN_DIR}/readiness.log" >&2
  if [[ -f "${HOST_RUN_DIR}/readiness.log" ]]; then
    tail -n 80 "${HOST_RUN_DIR}/readiness.log" >&2 || true
  fi
  write_pwm_snapshot "${HOST_RUN_DIR}/pwm_snapshot_after.txt"
  exit 1
fi

if [[ "${RECORD_ONLINE_FUSION}" == "1" ]]; then
  if ! docker exec "${CONTAINER_NAME}" /bin/bash -lc \
    "source '${ROS_SETUP_SCRIPT}' && timeout '${READY_TIMEOUT_S}'s ros2 topic echo /apex/estimation/status --once" \
    > "${HOST_RUN_DIR}/online_fusion_ready.log" 2>&1
  then
    echo "[APEX][ERROR] Online fusion topic /apex/estimation/status is not available. Restart apex_pipeline with APEX_ENABLE_IMU_LIDAR_FUSION=1." >&2
    if [[ -f "${HOST_RUN_DIR}/online_fusion_ready.log" ]]; then
      tail -n 80 "${HOST_RUN_DIR}/online_fusion_ready.log" >&2 || true
    fi
    write_pwm_snapshot "${HOST_RUN_DIR}/pwm_snapshot_after.txt"
    exit 1
  fi
fi

if [[ "${RECORD_ONLINE_FUSION}" == "1" ]]; then
  docker exec "${CONTAINER_NAME}" /bin/bash -lc \
    "source '${ROS_SETUP_SCRIPT}' && python3 /work/ros2_ws/scripts/capture/record_online_fusion_estimate.py \
      --odom-topic /apex/odometry/imu_lidar_fused \
      --status-topic /apex/estimation/status \
      --output-dir '${CONTAINER_RUN_DIR}/analysis_sensor_fusion_online' \
      --duration-s '${ONLINE_FUSION_RECORDER_DURATION_S}'" \
    > "${HOST_RUN_DIR}/online_fusion_record.log" 2>&1 &
  ONLINE_FUSION_RECORDER_PID=$!
fi

docker exec "${CONTAINER_NAME}" /bin/bash -lc \
  "source '${ROS_SETUP_SCRIPT}' && python3 /work/ros2_ws/scripts/capture/record_rect_sensorfus_capture.py \
    --imu-output '${CONTAINER_RUN_DIR}/imu_raw.csv' \
    --lidar-output '${CONTAINER_RUN_DIR}/lidar_points.csv' \
    --duration-s '${CAPTURE_DURATION_S}'" \
  > "${HOST_RUN_DIR}/capture.log" 2>&1 &
CAPTURE_PID=$!

sleep "${CAPTURE_WARMUP_S}"

if ! run_actuation_process; then
  echo "[APEX][ERROR] Actuation process failed. Log: ${HOST_RUN_DIR}/actuation.log" >&2
  if [[ -f "${HOST_RUN_DIR}/actuation.log" ]]; then
    tail -n 80 "${HOST_RUN_DIR}/actuation.log" >&2 || true
  fi
  finalize_bundle_artifacts
  exit 1
fi

if ! wait "${CAPTURE_PID}"; then
  echo "[APEX][ERROR] Sensor capture process failed. Log: ${HOST_RUN_DIR}/capture.log" >&2
  if [[ -f "${HOST_RUN_DIR}/capture.log" ]]; then
    tail -n 80 "${HOST_RUN_DIR}/capture.log" >&2 || true
  fi
  finalize_bundle_artifacts
  exit 1
fi

finalize_bundle_artifacts

echo "[APEX] Movement capture ready: ${HOST_RUN_DIR}"
echo "[APEX] Files:"
echo "  ${HOST_RUN_DIR}/imu_raw.csv"
echo "  ${HOST_RUN_DIR}/lidar_points.csv"
echo "  ${HOST_RUN_DIR}/pwm_trace.csv"
echo "  ${HOST_RUN_DIR}/actuation_events.csv"
echo "  ${HOST_RUN_DIR}/sysfs_pwm_monitor.csv"
echo "  ${HOST_RUN_DIR}/readiness.log"
echo "  ${HOST_RUN_DIR}/readiness.json"
echo "  ${HOST_RUN_DIR}/pwm_snapshot_before.txt"
echo "  ${HOST_RUN_DIR}/pwm_snapshot_after.txt"
if [[ -f "${HOST_RUN_DIR}/debug_summary.txt" ]]; then
  echo "  ${HOST_RUN_DIR}/debug_summary.txt"
fi
if [[ -f "${HOST_RUN_DIR}/analysis/summary.md" ]]; then
  echo "  ${HOST_RUN_DIR}/analysis/summary.md"
  echo "  ${HOST_RUN_DIR}/analysis/flags.json"
  echo "  ${HOST_RUN_DIR}/analysis/pwm_timeline.csv"
fi
if [[ -f "${HOST_RUN_DIR}/analysis_sensor_fusion_online/online_fusion_trajectory.csv" ]]; then
  echo "  ${HOST_RUN_DIR}/analysis_sensor_fusion_online/online_fusion_trajectory.csv"
  echo "  ${HOST_RUN_DIR}/analysis_sensor_fusion_online/online_fusion_summary.json"
fi
if [[ -f "${HOST_RUN_DIR}/analysis_sensor_fusion_comparison/online_vs_offline_summary.json" ]]; then
  echo "  ${HOST_RUN_DIR}/analysis_sensor_fusion_comparison/online_vs_offline_summary.json"
fi
