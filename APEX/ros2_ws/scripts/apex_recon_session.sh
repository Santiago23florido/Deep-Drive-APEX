#!/usr/bin/env bash
set -eo pipefail

export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"
export AMENT_PYTHON_EXECUTABLE="${AMENT_PYTHON_EXECUTABLE:-/usr/bin/python3}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

source /opt/ros/jazzy/setup.bash

export PYTHONPATH="/work/ros2_ws/src/apex_telemetry:${PYTHONPATH:-}"
APEX_VENV_SITEPKG="$(echo /opt/apex_venv/lib/python*/site-packages)"
if [ -d "${APEX_VENV_SITEPKG}" ]; then
  export PYTHONPATH="${APEX_VENV_SITEPKG}:${PYTHONPATH}"
fi
unset ROS_STATIC_PEERS FASTRTPS_DEFAULT_PROFILES_FILE ROS_LOCALHOST_ONLY

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-30}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-SUBNET}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

APEX_REPO_ROOT="/work"
PARAMS_FILE="/work/ros2_ws/src/apex_telemetry/config/apex_params.yaml"
SLAM_PARAMS_FILE="/work/ros2_ws/src/apex_telemetry/config/apex_slam_toolbox.yaml"
LOCAL_SLAM_PARAMS_FILE="/work/ros2_ws/src/apex_telemetry/config/apex_local_slam_toolbox.yaml"
EKF_PARAMS_FILE="/work/ros2_ws/src/apex_telemetry/config/apex_local_ekf.yaml"
IMU_FILTER_PARAMS_FILE="/work/ros2_ws/src/apex_telemetry/config/apex_local_imu_filter.yaml"
STATE_DIR="${APEX_RECON_SESSION_DIR:-/tmp/apex_recon_session}"
RUNNER_PID_FILE="${STATE_DIR}/runner.pid"
RECON_PID_FILE="${STATE_DIR}/recon.pid"
ROSBAG_PID_FILE="${STATE_DIR}/rosbag.pid"
SESSION_ENV_FILE="${STATE_DIR}/session.env"
CURRENT_BUNDLE_FILE="${STATE_DIR}/current_bundle_dir"
CURRENT_RUN_ID_FILE="${STATE_DIR}/current_run_id"
CURRENT_MODE_FILE="${STATE_DIR}/current_mode"
LAST_STATUS_FILE="${STATE_DIR}/last_status"
LAST_METADATA_FILE="${STATE_DIR}/last_metadata_path"
LAST_INVALID_FILE="${STATE_DIR}/last_invalid_bundle"
START_LOG_FILE="${STATE_DIR}/start.log"

mkdir -p "${STATE_DIR}"

APEX_DEBUG_RUN_NAME=""
APEX_DEBUG_BUNDLE_DIR=""
APEX_DOCKER_TAIL_LOG=""
APEX_RAW_BAG_DIR=""
APEX_ROSBAG_LOG_PATH=""
APEX_DEBUG_FINALIZED=0
APEX_ROSBAG_PID=""
RECON_PID=""
WATCHDOG_PID=""
RUNNER_CLEANUP_DONE=0
RECON_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  apex_recon_session.sh start --run-id <id> --mode <mode> [options]
  apex_recon_session.sh stop
  apex_recon_session.sh status
  apex_recon_session.sh restart --run-id <id> --mode <mode> [options]

Options:
  --run-id <id>
  --mode <nav_dryrun|recon_debug|curve_static_probe|curve_entry_probe|steering_static|steering_sign_check|straight_open_loop>
  --record-debug <0|1>
  --min-speed-pct <pct>
  --max-speed-pct <pct>
  --fixed-speed-pct <pct>
  --step-duration-s <seconds>
  --timeout-s <seconds>
EOF
}

pid_is_live() {
  local pid="${1:-}"
  [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null
}

read_file_trimmed() {
  local path="$1"
  if [ -f "${path}" ]; then
    tr -d '[:space:]' < "${path}"
  fi
}

remove_live_state() {
  rm -f \
    "${RUNNER_PID_FILE}" \
    "${RECON_PID_FILE}" \
    "${ROSBAG_PID_FILE}" \
    "${CURRENT_BUNDLE_FILE}" \
    "${CURRENT_RUN_ID_FILE}" \
    "${CURRENT_MODE_FILE}" \
    "${SESSION_ENV_FILE}"
}

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

normalize_bool_env() {
  local raw_value="${1:-}"
  if [ -z "${raw_value}" ]; then
    return 0
  fi
  case "${raw_value}" in
    1|true|TRUE|True|yes|YES|on|ON)
      printf 'true'
      ;;
    0|false|FALSE|False|no|NO|off|OFF)
      printf 'false'
      ;;
    *)
      printf '%s' "${raw_value}"
      ;;
  esac
}

refresh_git_metadata() {
  export APEX_GIT_COMMIT="${APEX_GIT_COMMIT:-$(git -C "${APEX_REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)}"
  export APEX_GIT_DIRTY="${APEX_GIT_DIRTY:-$(git -C "${APEX_REPO_ROOT}" status --porcelain 2>/dev/null | wc -l | tr -d ' ')}"
}

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
  local keep_enabled="${4:-1}"
  if [ -z "${pwm_dir}" ] || [ ! -d "${pwm_dir}" ]; then
    return
  fi
  python3 - "${pwm_dir}" "${frequency_hz}" "${duty_cycle_pct}" "${keep_enabled}" <<'PY'
import os
import sys

pwm_dir = sys.argv[1]
frequency_hz = max(1.0, float(sys.argv[2]))
duty_cycle_pct = max(0.0, min(100.0, float(sys.argv[3])))
keep_enabled = int(sys.argv[4]) != 0

period_ns = int(round(1.0e9 / frequency_hz))
duty_cycle_ns = int(round(period_ns * duty_cycle_pct / 100.0))

with open(os.path.join(pwm_dir, "period"), "w", encoding="utf-8") as handle:
    handle.write(f"{period_ns}\n")
with open(os.path.join(pwm_dir, "duty_cycle"), "w", encoding="utf-8") as handle:
    handle.write(f"{duty_cycle_ns}\n")
with open(os.path.join(pwm_dir, "enable"), "w", encoding="utf-8") as handle:
    handle.write("1\n" if keep_enabled else "0\n")
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
    set_pwm_output_pct "${pwm_dir}" "${frequency_hz}" "${neutral_dc}" 1
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
    set_pwm_output_pct "${pwm_dir}" "${frequency_hz}" "${center_dc}" 1
    echo "[APEX] Forced steering center on ${pwm_dir} at ${center_dc}%"
  fi
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

    find /sys/class/pwm -maxdepth 4 -type f \
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

record_debug_enabled = os.environ.get("APEX_RECORD_DEBUG", "0") == "1"
bundle_dir = Path(os.environ.get("APEX_DEBUG_BUNDLE_DIR", ""))
bag_dir_raw = os.environ.get("APEX_RAW_BAG_DIR", "")
bag_dir = Path(bag_dir_raw) if bag_dir_raw else None
bag_expected = record_debug_enabled and bool(bag_dir_raw)
bag_dir_present = bool(bag_dir and bag_dir.exists())
bag_mcap_path_raw = os.environ.get("APEX_FINAL_MCAP_PATH", "")
bag_metadata_path_raw = os.environ.get("APEX_FINAL_BAG_METADATA_PATH", "")
if bag_dir and not bag_mcap_path_raw:
    raw_mcap_candidates = sorted(bag_dir.glob("*.mcap"))
    if raw_mcap_candidates:
        bag_mcap_path_raw = str(raw_mcap_candidates[0])
if bag_dir and not bag_metadata_path_raw:
    raw_metadata_path = bag_dir / "metadata.yaml"
    if raw_metadata_path.is_file():
        bag_metadata_path_raw = str(raw_metadata_path)
bag_mcap_path = Path(bag_mcap_path_raw) if bag_mcap_path_raw else None
bag_metadata_path = Path(bag_metadata_path_raw) if bag_metadata_path_raw else None
docker_tail_log = Path(os.environ.get("APEX_DOCKER_TAIL_LOG", ""))
rosbag_log_path = Path(os.environ.get("APEX_ROSBAG_LOG_PATH", ""))
recon_log_path = Path(os.environ.get("APEX_RECON_LOG_PATH", ""))
params_snapshot = Path(os.environ.get("APEX_PARAMS_SNAPSHOT_PATH", ""))
slam_params_snapshot = Path(os.environ.get("APEX_SLAM_PARAMS_SNAPSHOT_PATH", ""))
local_slam_params_snapshot = Path(os.environ.get("APEX_LOCAL_SLAM_PARAMS_SNAPSHOT_PATH", ""))
ekf_params_snapshot = Path(os.environ.get("APEX_EKF_PARAMS_SNAPSHOT_PATH", ""))
imu_filter_params_snapshot = Path(os.environ.get("APEX_IMU_FILTER_PARAMS_SNAPSHOT_PATH", ""))
pwm_snapshot_before = bundle_dir / "pwm_snapshot_before.txt"
pwm_snapshot_after = bundle_dir / "pwm_snapshot_after.txt"

shutdown_diag_begin_present = False
shutdown_diag_final_present = False
if recon_log_path.is_file():
    with recon_log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            if not raw_line.startswith("DIAG_STOP "):
                continue
            try:
                payload = json.loads(raw_line.split(" ", 1)[1])
            except Exception:
                continue
            stage = payload.get("stage")
            if stage == "begin":
                shutdown_diag_begin_present = True
            elif stage == "final":
                shutdown_diag_final_present = True

pwm_snapshot_before_present = pwm_snapshot_before.is_file()
pwm_snapshot_after_present = pwm_snapshot_after.is_file()
bag_mcap_present = bool(bag_mcap_path and bag_mcap_path.is_file())
bag_metadata_present = bool(bag_metadata_path and bag_metadata_path.is_file())
docker_tail_present = docker_tail_log.is_file()
rosbag_log_present = rosbag_log_path.is_file()
recon_log_present = recon_log_path.is_file()
params_snapshot_present = params_snapshot.is_file()
slam_params_snapshot_present = slam_params_snapshot.is_file()
local_slam_params_snapshot_present = local_slam_params_snapshot.is_file()
ekf_params_snapshot_present = ekf_params_snapshot.is_file()
imu_filter_params_snapshot_present = imu_filter_params_snapshot.is_file()

shutdown_clean = (
    shutdown_diag_begin_present
    and shutdown_diag_final_present
    and pwm_snapshot_after_present
)

bundle_missing_artifacts = []
if not pwm_snapshot_before_present:
    bundle_missing_artifacts.append("pwm_snapshot_before.txt")
if not pwm_snapshot_after_present:
    bundle_missing_artifacts.append("pwm_snapshot_after.txt")
if not shutdown_diag_begin_present:
    bundle_missing_artifacts.append("DIAG_STOP begin")
if not shutdown_diag_final_present:
    bundle_missing_artifacts.append("DIAG_STOP final")
if not docker_tail_present:
    bundle_missing_artifacts.append("docker_tail.log")
if bag_expected and not rosbag_log_present:
    bundle_missing_artifacts.append("rosbag_record.log")
if not recon_log_present:
    bundle_missing_artifacts.append("recon_diagnostic.log")
if not params_snapshot_present:
    bundle_missing_artifacts.append("config/apex_params.yaml")
if not slam_params_snapshot_present:
    bundle_missing_artifacts.append("config/apex_slam_toolbox.yaml")
if not local_slam_params_snapshot_present:
    bundle_missing_artifacts.append("config/apex_local_slam_toolbox.yaml")
if not ekf_params_snapshot_present:
    bundle_missing_artifacts.append("config/apex_local_ekf.yaml")
if not imu_filter_params_snapshot_present:
    bundle_missing_artifacts.append("config/apex_local_imu_filter.yaml")
if bag_expected and not bag_dir_present:
    bundle_missing_artifacts.append("bag/raw_debug_run")
if bag_expected and not bag_mcap_present:
    bundle_missing_artifacts.append("bag/debug_run.mcap")

bundle_complete = (
    shutdown_clean
    and docker_tail_present
    and recon_log_present
    and params_snapshot_present
    and slam_params_snapshot_present
    and local_slam_params_snapshot_present
    and ekf_params_snapshot_present
    and imu_filter_params_snapshot_present
    and (not bag_expected or (bag_dir_present and bag_mcap_present))
)

metadata = {
    "run_id": os.environ.get("APEX_DEBUG_RUN_NAME", ""),
    "git_commit": os.environ.get("APEX_GIT_COMMIT", "unknown"),
    "git_dirty_count": os.environ.get("APEX_GIT_DIRTY", "unknown"),
    "bundle_dir": str(bundle_dir),
    "bag_dir": bag_dir_raw,
    "bag_expected": bag_expected,
    "bag_mcap_path": bag_mcap_path_raw,
    "bag_mcap_present": bag_mcap_present,
    "bag_metadata_path": bag_metadata_path_raw,
    "bag_metadata_present": bag_metadata_present,
    "docker_tail_log": str(docker_tail_log),
    "docker_tail_present": docker_tail_present,
    "rosbag_log_path": str(rosbag_log_path),
    "rosbag_log_present": rosbag_log_present,
    "recon_diagnostic_log": str(recon_log_path),
    "recon_diagnostic_present": recon_log_present,
    "params_snapshot": str(params_snapshot),
    "params_snapshot_present": params_snapshot_present,
    "slam_params_snapshot": str(slam_params_snapshot),
    "slam_params_snapshot_present": slam_params_snapshot_present,
    "local_slam_params_snapshot": str(local_slam_params_snapshot),
    "local_slam_params_snapshot_present": local_slam_params_snapshot_present,
    "ekf_params_snapshot": str(ekf_params_snapshot),
    "ekf_params_snapshot_present": ekf_params_snapshot_present,
    "imu_filter_params_snapshot": str(imu_filter_params_snapshot),
    "imu_filter_params_snapshot_present": imu_filter_params_snapshot_present,
    "record_debug_enabled": os.environ.get("APEX_RECORD_DEBUG", "0"),
    "diagnostic_mode_env": os.environ.get("APEX_RECON_DIAGNOSTIC_MODE", ""),
    "steering_direction_sign_env": os.environ.get("APEX_STEERING_DIRECTION_SIGN", ""),
    "lidar_heading_offset_env": os.environ.get("APEX_LIDAR_HEADING_OFFSET_DEG", ""),
    "timestamp_utc": os.environ.get("APEX_RUN_TIMESTAMP_UTC", ""),
    "pwm_snapshot_before_present": pwm_snapshot_before_present,
    "pwm_snapshot_after_present": pwm_snapshot_after_present,
    "shutdown_diag_begin_present": shutdown_diag_begin_present,
    "shutdown_diag_final_present": shutdown_diag_final_present,
    "shutdown_clean": shutdown_clean,
    "bundle_complete": bundle_complete,
    "bundle_missing_artifacts": bundle_missing_artifacts,
    "env_overrides": selected_env,
}
metadata_path.write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

validate_debug_artifacts() {
  if [ "${APEX_RECORD_DEBUG:-0}" != "1" ] || [ -z "${APEX_DEBUG_BUNDLE_DIR}" ]; then
    return 0
  fi

  local metadata_path="${APEX_DEBUG_BUNDLE_DIR}/run_metadata.json"
  if [ ! -f "${metadata_path}" ]; then
    echo "[APEX][ERROR] Missing run_metadata.json in ${APEX_DEBUG_BUNDLE_DIR}"
    return 1
  fi

  python3 - "${metadata_path}" <<'PY'
import json
import sys
from pathlib import Path

metadata_path = Path(sys.argv[1])
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
errors = []

if not metadata.get("shutdown_clean", False):
    errors.append("shutdown_clean=false")
if not metadata.get("pwm_snapshot_after_present", False):
    errors.append("pwm_snapshot_after.txt missing")
if metadata.get("bag_expected", False) and not metadata.get("bag_mcap_present", False):
    errors.append("bag/debug_run.mcap missing")
if not metadata.get("bundle_complete", False):
    missing = metadata.get("bundle_missing_artifacts", [])
    if missing:
        errors.append("bundle_missing_artifacts=" + ", ".join(str(item) for item in missing))
    else:
        errors.append("bundle_complete=false")

if errors:
    for error in errors:
        print(f"[APEX][ERROR] {error}")
    sys.exit(1)

print(f"[APEX] Debug bundle complete: {metadata_path.parent}")
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
    local waited=0
    while [ "${waited}" -lt 150 ]; do
      mcap_path="$(find "${APEX_RAW_BAG_DIR}" -maxdepth 1 -name '*.mcap' | sort | head -n 1 || true)"
      metadata_path="${APEX_RAW_BAG_DIR}/metadata.yaml"
      if [ -n "${mcap_path}" ] && [ -f "${metadata_path}" ]; then
        break
      fi
      sleep 0.1
      waited=$((waited + 1))
    done
    if [ -n "${mcap_path}" ] && [ ! -f "${metadata_path}" ]; then
      ros2 bag reindex "${APEX_RAW_BAG_DIR}" >/dev/null 2>&1 || true
      local reindex_waited=0
      while [ "${reindex_waited}" -lt 50 ] && [ ! -f "${metadata_path}" ]; do
        sleep 0.1
        reindex_waited=$((reindex_waited + 1))
      done
    fi
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
  export APEX_ROSBAG_LOG_PATH="${APEX_DEBUG_BUNDLE_DIR}/rosbag_record.log"
  APEX_ROSBAG_LOG_PATH="${APEX_ROSBAG_LOG_PATH}"
  export APEX_RECON_LOG_PATH="${APEX_RECON_LOG_PATH:-${APEX_DEBUG_BUNDLE_DIR}/recon_diagnostic.log}"
  mkdir -p "$(dirname "${APEX_RECON_LOG_PATH}")"

  export APEX_PARAMS_SNAPSHOT_PATH="${APEX_DEBUG_BUNDLE_DIR}/config/apex_params.yaml"
  export APEX_SLAM_PARAMS_SNAPSHOT_PATH="${APEX_DEBUG_BUNDLE_DIR}/config/apex_slam_toolbox.yaml"
  export APEX_LOCAL_SLAM_PARAMS_SNAPSHOT_PATH="${APEX_DEBUG_BUNDLE_DIR}/config/apex_local_slam_toolbox.yaml"
  export APEX_EKF_PARAMS_SNAPSHOT_PATH="${APEX_DEBUG_BUNDLE_DIR}/config/apex_local_ekf.yaml"
  export APEX_IMU_FILTER_PARAMS_SNAPSHOT_PATH="${APEX_DEBUG_BUNDLE_DIR}/config/apex_local_imu_filter.yaml"
  cp "${PARAMS_FILE}" "${APEX_PARAMS_SNAPSHOT_PATH}"
  cp "${SLAM_PARAMS_FILE}" "${APEX_SLAM_PARAMS_SNAPSHOT_PATH}"
  cp "${LOCAL_SLAM_PARAMS_FILE}" "${APEX_LOCAL_SLAM_PARAMS_SNAPSHOT_PATH}"
  cp "${EKF_PARAMS_FILE}" "${APEX_EKF_PARAMS_SNAPSHOT_PATH}"
  cp "${IMU_FILTER_PARAMS_FILE}" "${APEX_IMU_FILTER_PARAMS_SNAPSHOT_PATH}"

  export APEX_RAW_BAG_DIR="${APEX_DEBUG_BUNDLE_DIR}/bag/raw_debug_run"
  APEX_RAW_BAG_DIR="${APEX_RAW_BAG_DIR}"
  printf '%s\n' "${APEX_DEBUG_BUNDLE_DIR}" > "${CURRENT_BUNDLE_FILE}"

  write_pwm_snapshot "before"
  write_debug_metadata

  exec > >(tee -a "${APEX_DOCKER_TAIL_LOG}") 2>&1
  echo "[APEX] Recon debug bundle: ${APEX_DEBUG_BUNDLE_DIR}"
}

start_debug_bag_recording() {
  if [ "${APEX_RECORD_DEBUG:-0}" != "1" ]; then
    return
  fi

  rm -rf "${APEX_RAW_BAG_DIR}"
  : > "${APEX_ROSBAG_LOG_PATH}"
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
    /apex/imu/data_raw \
    /apex/imu/data_filtered \
    /apex/imu/acceleration/raw \
    /apex/imu/angular_velocity/raw \
    /apex/kinematics/acceleration \
    /apex/kinematics/velocity \
    /apex/kinematics/position \
    /apex/kinematics/angular_velocity \
    /apex/kinematics/heading \
    /apex/kinematics/status \
    /apex/odometry/imu_raw \
    /apex/odometry/fusion_status \
    /apex/lidar/pose_local \
    /odometry/filtered \
    > "${APEX_ROSBAG_LOG_PATH}" 2>&1 &
  APEX_ROSBAG_PID="$!"
  printf '%s\n' "${APEX_ROSBAG_PID}" > "${ROSBAG_PID_FILE}"
}

stop_debug_bag_recording() {
  local bag_pid="${APEX_ROSBAG_PID:-}"
  if [ -z "${bag_pid}" ] && [ -f "${ROSBAG_PID_FILE}" ]; then
    bag_pid="$(read_file_trimmed "${ROSBAG_PID_FILE}")"
  fi
  if [ -z "${bag_pid}" ] || ! pid_is_live "${bag_pid}"; then
    rm -f "${ROSBAG_PID_FILE}"
    return
  fi
  kill -INT "${bag_pid}" 2>/dev/null || true
  local waited=0
  while pid_is_live "${bag_pid}" && [ "${waited}" -lt 200 ]; do
    sleep 0.05
    waited=$((waited + 1))
  done
  if pid_is_live "${bag_pid}"; then
    kill -TERM "${bag_pid}" 2>/dev/null || true
    sleep 1
  fi
  if pid_is_live "${bag_pid}"; then
    kill -KILL "${bag_pid}" 2>/dev/null || true
  fi
  rm -f "${ROSBAG_PID_FILE}"

  if [ -n "${APEX_RAW_BAG_DIR:-}" ]; then
    local waited=0
    local mcap_path=""
    local metadata_path="${APEX_RAW_BAG_DIR}/metadata.yaml"
    while [ "${waited}" -lt 150 ]; do
      mcap_path="$(find "${APEX_RAW_BAG_DIR}" -maxdepth 1 -name '*.mcap' | sort | head -n 1 || true)"
      if [ -n "${mcap_path}" ] && [ -f "${metadata_path}" ]; then
        break
      fi
      sleep 0.1
      waited=$((waited + 1))
    done
    if [ -d "${APEX_RAW_BAG_DIR}" ] && [ ! -f "${metadata_path}" ]; then
      ros2 bag reindex "${APEX_RAW_BAG_DIR}" >/dev/null 2>&1 || true
    fi
  fi
}

build_recon_args() {
  RECON_ARGS=(
    --ros-args
    --params-file "${PARAMS_FILE}"
    -p "reset_map_on_start:=false"
    -p "save_map_on_completion:=false"
    -p "diagnostic_mode:=${APEX_RECON_DIAGNOSTIC_MODE}"
  )

  if [ -n "${APEX_RECON_FIXED_SPEED_PCT:-}" ]; then
    RECON_ARGS+=(-p "diagnostic_fixed_speed_pct:=$(normalize_double_env "${APEX_RECON_FIXED_SPEED_PCT}")")
  fi
  if [ -n "${APEX_RECON_STEP_DURATION_S:-}" ]; then
    RECON_ARGS+=(-p "diagnostic_step_duration_s:=$(normalize_double_env "${APEX_RECON_STEP_DURATION_S}")")
  fi
  if [ -n "${APEX_EXPLORE_MIN_SPEED_PCT:-}" ]; then
    RECON_ARGS+=(-p "explore_min_speed_pct:=$(normalize_double_env "${APEX_EXPLORE_MIN_SPEED_PCT}")")
  fi
  if [ -n "${APEX_EXPLORE_MAX_SPEED_PCT:-}" ]; then
    RECON_ARGS+=(-p "explore_max_speed_pct:=$(normalize_double_env "${APEX_EXPLORE_MAX_SPEED_PCT}")")
  fi
  if [ -n "${APEX_RECON_TIMEOUT_S:-}" ]; then
    RECON_ARGS+=(-p "diagnostic_recon_timeout_s:=$(normalize_double_env "${APEX_RECON_TIMEOUT_S}")")
  fi
  if [ -n "${APEX_RECON_LOG_PATH:-}" ]; then
    RECON_ARGS+=(-p "diagnostic_log_path:=${APEX_RECON_LOG_PATH}")
  fi
  if [ -n "${APEX_RECON_LOG_FLUSH_EVERY:-}" ]; then
    RECON_ARGS+=(-p "diagnostic_file_flush_every_n_records:=${APEX_RECON_LOG_FLUSH_EVERY}")
  fi
  if [ -n "${APEX_RECON_LOG_OVERWRITE:-}" ]; then
    RECON_ARGS+=(-p "diagnostic_overwrite_log_on_start:=$(normalize_bool_env "${APEX_RECON_LOG_OVERWRITE}")")
  fi
  if [ -n "${APEX_RECON_LOG_LEVEL:-}" ]; then
    RECON_ARGS+=(--log-level "${APEX_RECON_LOG_LEVEL}")
  fi
}

write_session_env() {
  local run_id="$1"
  local mode="$2"
  local record_debug="$3"
  local min_speed_pct="$4"
  local max_speed_pct="$5"
  local fixed_speed_pct="$6"
  local step_duration_s="$7"
  local timeout_s="$8"

  : > "${SESSION_ENV_FILE}"
  {
    printf 'APEX_DEBUG_RUN_ID=%q\n' "${run_id}"
    printf 'APEX_RECON_DIAGNOSTIC_MODE=%q\n' "${mode}"
    printf 'APEX_RECORD_DEBUG=%q\n' "${record_debug}"
    printf 'APEX_EXPLORE_MIN_SPEED_PCT=%q\n' "${min_speed_pct}"
    printf 'APEX_EXPLORE_MAX_SPEED_PCT=%q\n' "${max_speed_pct}"
    printf 'APEX_RECON_FIXED_SPEED_PCT=%q\n' "${fixed_speed_pct}"
    printf 'APEX_RECON_STEP_DURATION_S=%q\n' "${step_duration_s}"
    printf 'APEX_RECON_TIMEOUT_S=%q\n' "${timeout_s}"
    printf 'APEX_DEBUG_OUTPUT_DIR=%q\n' "${APEX_DEBUG_OUTPUT_DIR:-/work/ros2_ws/debug_runs}"
    printf 'APEX_RECON_LOG_FLUSH_EVERY=%q\n' "${APEX_RECON_LOG_FLUSH_EVERY:-}"
    printf 'APEX_RECON_LOG_OVERWRITE=%q\n' "$(normalize_bool_env "${APEX_RECON_LOG_OVERWRITE:-true}")"
    printf 'APEX_RECON_LOG_LEVEL=%q\n' "${APEX_RECON_LOG_LEVEL:-}"
    printf 'APEX_GIT_COMMIT=%q\n' "$(git -C "${APEX_REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
    printf 'APEX_GIT_DIRTY=%q\n' "$(git -C "${APEX_REPO_ROOT}" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  } >> "${SESSION_ENV_FILE}"
}

load_session_env() {
  if [ ! -f "${SESSION_ENV_FILE}" ]; then
    echo "[APEX][ERROR] Missing active session env file" >&2
    return 1
  fi
  # shellcheck disable=SC1090
  source "${SESSION_ENV_FILE}"
}

mark_last_run_validity() {
  local status_value="$1"
  printf '%s\n' "${status_value}" > "${LAST_STATUS_FILE}"
  if [ -n "${APEX_DEBUG_BUNDLE_DIR:-}" ]; then
    printf '%s\n' "${APEX_DEBUG_BUNDLE_DIR}" > "${LAST_INVALID_FILE}"
  fi
  if [ -f "${APEX_DEBUG_BUNDLE_DIR:-}/run_metadata.json" ]; then
    printf '%s\n' "${APEX_DEBUG_BUNDLE_DIR}/run_metadata.json" > "${LAST_METADATA_FILE}"
  fi
  if [ "${status_value}" = "valid" ]; then
    rm -f "${LAST_INVALID_FILE}"
  fi
}

load_last_bundle_context() {
  if ! load_session_env; then
    return 1
  fi
  if [ -f "${CURRENT_BUNDLE_FILE}" ]; then
    export APEX_DEBUG_BUNDLE_DIR="$(cat "${CURRENT_BUNDLE_FILE}")"
    export APEX_DOCKER_TAIL_LOG="${APEX_DEBUG_BUNDLE_DIR}/docker_tail.log"
    export APEX_ROSBAG_LOG_PATH="${APEX_DEBUG_BUNDLE_DIR}/rosbag_record.log"
    export APEX_RECON_LOG_PATH="${APEX_DEBUG_BUNDLE_DIR}/recon_diagnostic.log"
    export APEX_PARAMS_SNAPSHOT_PATH="${APEX_DEBUG_BUNDLE_DIR}/config/apex_params.yaml"
    export APEX_SLAM_PARAMS_SNAPSHOT_PATH="${APEX_DEBUG_BUNDLE_DIR}/config/apex_slam_toolbox.yaml"
    export APEX_RAW_BAG_DIR="${APEX_DEBUG_BUNDLE_DIR}/bag/raw_debug_run"
  fi
}

enforce_speed_floor() {
  local label="$1"
  local value="${2:-}"
  if [ -z "${value}" ]; then
    return 0
  fi
  python3 - "${label}" "${value}" <<'PY'
import sys

label = sys.argv[1]
value = float(sys.argv[2])
if 0.0 < abs(value) < 20.0:
    raise SystemExit(f"{label}={value:g} violates the 20% PWM floor")
PY
}

assert_allowed_mode() {
  local mode="$1"
  case "${mode}" in
    nav_dryrun|recon_debug|curve_static_probe|curve_entry_probe|steering_static|steering_sign_check|straight_open_loop)
      ;;
    *)
      echo "[APEX][ERROR] Unsupported mode: ${mode}" >&2
      usage
      exit 1
      ;;
  esac
}

clear_stale_session() {
  local runner_pid=""
  runner_pid="$(read_file_trimmed "${RUNNER_PID_FILE}")"
  if [ -z "${runner_pid}" ]; then
    return
  fi
  if pid_is_live "${runner_pid}"; then
    return
  fi
  echo "[APEX][WARN] Found stale recon session state for pid ${runner_pid}"
  load_last_bundle_context || true
  force_motor_neutral_from_params
  force_steering_center_from_params
  if [ "${APEX_RECORD_DEBUG:-0}" = "1" ] && [ -n "${APEX_DEBUG_BUNDLE_DIR:-}" ]; then
    write_pwm_snapshot "after"
    finalize_debug_artifacts || true
    write_debug_metadata || true
  fi
  if bundle_metadata_is_complete "${APEX_DEBUG_BUNDLE_DIR:-}/run_metadata.json"; then
    mark_last_run_validity "valid"
  else
    printf '%s\n' "invalid" > "${LAST_STATUS_FILE}"
    if [ -n "${APEX_DEBUG_BUNDLE_DIR:-}" ]; then
      printf '%s\n' "${APEX_DEBUG_BUNDLE_DIR}" > "${LAST_INVALID_FILE}"
    fi
  fi
  remove_live_state
}

assert_previous_run_clean() {
  if [ ! -f "${LAST_STATUS_FILE}" ]; then
    return
  fi
  local last_status
  last_status="$(read_file_trimmed "${LAST_STATUS_FILE}")"
  if [ "${last_status}" = "invalid" ]; then
    local invalid_bundle=""
    invalid_bundle="$(read_file_trimmed "${LAST_INVALID_FILE}")"
    echo "[APEX][ERROR] Previous recon session did not close cleanly." >&2
    if [ -n "${invalid_bundle}" ]; then
      echo "[APEX][ERROR] Invalid bundle: ${invalid_bundle}" >&2
    fi
    echo "[APEX][ERROR] Restart the core container with tools/apex_core_down.sh and tools/apex_core_up.sh before the next run." >&2
    exit 1
  fi
}

wait_for_pid_exit() {
  local pid="$1"
  local timeout_s="$2"
  local waited_steps=0
  local max_steps
  max_steps="$(python3 - "${timeout_s}" <<'PY'
import math
import sys
print(max(1, int(math.ceil(float(sys.argv[1]) / 0.1))))
PY
)"
  while pid_is_live "${pid}" && [ "${waited_steps}" -lt "${max_steps}" ]; do
    sleep 0.1
    waited_steps=$((waited_steps + 1))
  done
  ! pid_is_live "${pid}"
}

bundle_metadata_is_complete() {
  local metadata_path="${1:-}"
  if [ -z "${metadata_path}" ] || [ ! -f "${metadata_path}" ]; then
    return 1
  fi
  python3 - "${metadata_path}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    metadata = json.load(handle)

raise SystemExit(0 if metadata.get("bundle_complete", False) else 1)
PY
}

runner_cleanup() {
  local exit_code=$?
  if [ "${RUNNER_CLEANUP_DONE}" = "1" ]; then
    return
  fi
  RUNNER_CLEANUP_DONE=1
  trap - INT TERM EXIT

  if [ -n "${RECON_PID:-}" ] && pid_is_live "${RECON_PID}"; then
    kill -INT "${RECON_PID}" 2>/dev/null || true
    wait_for_pid_exit "${RECON_PID}" 8 || true
    if pid_is_live "${RECON_PID}"; then
      kill -TERM "${RECON_PID}" 2>/dev/null || true
      sleep 1
    fi
    if pid_is_live "${RECON_PID}"; then
      kill -KILL "${RECON_PID}" 2>/dev/null || true
    fi
  fi

  if [ -n "${WATCHDOG_PID:-}" ] && pid_is_live "${WATCHDOG_PID}"; then
    kill -TERM "${WATCHDOG_PID}" 2>/dev/null || true
    wait_for_pid_exit "${WATCHDOG_PID}" 2 || true
  fi

  stop_debug_bag_recording
  force_motor_neutral_from_params
  force_steering_center_from_params
  write_pwm_snapshot "after"
  finalize_debug_artifacts

  local validation_ok=0
  if validate_debug_artifacts; then
    validation_ok=1
    mark_last_run_validity "valid"
  else
    validation_ok=0
    mark_last_run_validity "invalid"
  fi

  rm -f "${RUNNER_PID_FILE}" "${RECON_PID_FILE}" "${ROSBAG_PID_FILE}" "${CURRENT_BUNDLE_FILE}" "${CURRENT_RUN_ID_FILE}" "${CURRENT_MODE_FILE}" "${SESSION_ENV_FILE}"

  if [ "${validation_ok}" -ne 1 ]; then
    echo "[APEX][ERROR] Recon session cleanup finished with invalid artifacts"
  fi

  return "${exit_code}"
}

cmd_run() {
  local env_file="${1:-}"
  if [ -z "${env_file}" ] || [ ! -f "${env_file}" ]; then
    echo "[APEX][ERROR] Missing session env for run action" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${env_file}"
  refresh_git_metadata

  trap runner_cleanup INT TERM EXIT

  if [ "${APEX_RECORD_DEBUG:-0}" = "1" ]; then
    setup_debug_run
  fi

  printf '%s\n' "$$" > "${RUNNER_PID_FILE}"
  printf '%s\n' "${APEX_DEBUG_RUN_ID}" > "${CURRENT_RUN_ID_FILE}"
  printf '%s\n' "${APEX_RECON_DIAGNOSTIC_MODE}" > "${CURRENT_MODE_FILE}"

  start_debug_bag_recording
  build_recon_args

  echo "[APEX] Starting recon session run_id=${APEX_DEBUG_RUN_ID} mode=${APEX_RECON_DIAGNOSTIC_MODE}"
  python3 -m apex_telemetry.recon_mapping_node "${RECON_ARGS[@]}" &
  RECON_PID="$!"
  printf '%s\n' "${RECON_PID}" > "${RECON_PID_FILE}"

  if [ -n "${APEX_RECON_TIMEOUT_S:-}" ]; then
    (
      sleep "${APEX_RECON_TIMEOUT_S}"
      if kill -0 "${RECON_PID}" 2>/dev/null; then
        echo "[APEX][WATCHDOG] Recon session exceeded timeout ${APEX_RECON_TIMEOUT_S}s; sending SIGINT" >&2
        kill -INT "${RECON_PID}" 2>/dev/null || true
        sleep 2
      fi
      if kill -0 "${RECON_PID}" 2>/dev/null; then
        echo "[APEX][WATCHDOG] Recon session still alive; sending SIGTERM" >&2
        kill -TERM "${RECON_PID}" 2>/dev/null || true
        sleep 2
      fi
      if kill -0 "${RECON_PID}" 2>/dev/null; then
        echo "[APEX][WATCHDOG] Recon session still alive; sending SIGKILL" >&2
        kill -KILL "${RECON_PID}" 2>/dev/null || true
      fi
    ) &
    WATCHDOG_PID="$!"
  fi

  set +e
  wait "${RECON_PID}"
  local recon_status=$?
  set -e
  exit "${recon_status}"
}

cmd_status() {
  clear_stale_session
  local runner_pid
  runner_pid="$(read_file_trimmed "${RUNNER_PID_FILE}")"
  if [ -n "${runner_pid}" ] && pid_is_live "${runner_pid}"; then
    local run_id mode bundle_dir
    run_id="$(read_file_trimmed "${CURRENT_RUN_ID_FILE}")"
    mode="$(read_file_trimmed "${CURRENT_MODE_FILE}")"
    bundle_dir="$(read_file_trimmed "${CURRENT_BUNDLE_FILE}")"
    echo "[APEX] recon session running"
    echo "runner_pid=${runner_pid}"
    echo "run_id=${run_id:-unknown}"
    echo "mode=${mode:-unknown}"
    echo "bundle_dir=${bundle_dir:-none}"
    return 0
  fi

  local last_status last_metadata
  last_status="$(read_file_trimmed "${LAST_STATUS_FILE}")"
  last_metadata="$(read_file_trimmed "${LAST_METADATA_FILE}")"
  echo "[APEX] recon session not running"
  if [ -n "${last_status}" ]; then
    echo "last_status=${last_status}"
  fi
  if [ -n "${last_metadata}" ]; then
    echo "last_metadata=${last_metadata}"
  fi
  [ "${last_status:-}" != "invalid" ]
}

cmd_start() {
  clear_stale_session
  assert_previous_run_clean

  local run_id=""
  local mode=""
  local record_debug="0"
  local min_speed_pct=""
  local max_speed_pct=""
  local fixed_speed_pct=""
  local step_duration_s="${APEX_RECON_STEP_DURATION_S:-}"
  local timeout_s=""

  while [ "$#" -gt 0 ]; do
    case "$1" in
      --run-id)
        run_id="${2:-}"
        shift 2
        ;;
      --mode)
        mode="${2:-}"
        shift 2
        ;;
      --record-debug)
        record_debug="${2:-}"
        shift 2
        ;;
      --min-speed-pct)
        min_speed_pct="${2:-}"
        shift 2
        ;;
      --max-speed-pct)
        max_speed_pct="${2:-}"
        shift 2
        ;;
      --fixed-speed-pct)
        fixed_speed_pct="${2:-}"
        shift 2
        ;;
      --step-duration-s)
        step_duration_s="${2:-}"
        shift 2
        ;;
      --timeout-s)
        timeout_s="${2:-}"
        shift 2
        ;;
      *)
        echo "[APEX][ERROR] Unknown argument: $1" >&2
        usage
        exit 1
        ;;
    esac
  done

  if [ -z "${run_id}" ] || [ -z "${mode}" ]; then
    echo "[APEX][ERROR] --run-id and --mode are required" >&2
    usage
    exit 1
  fi

  case "${record_debug}" in
    0|1) ;;
    *)
      echo "[APEX][ERROR] --record-debug must be 0 or 1" >&2
      exit 1
      ;;
  esac

  assert_allowed_mode "${mode}"
  enforce_speed_floor "--min-speed-pct" "${min_speed_pct}"
  enforce_speed_floor "--max-speed-pct" "${max_speed_pct}"
  enforce_speed_floor "--fixed-speed-pct" "${fixed_speed_pct}"

  local runner_pid
  runner_pid="$(read_file_trimmed "${RUNNER_PID_FILE}")"
  if [ -n "${runner_pid}" ] && pid_is_live "${runner_pid}"; then
    echo "[APEX][ERROR] Recon session already running (pid=${runner_pid})" >&2
    exit 1
  fi

  write_session_env \
    "${run_id}" \
    "${mode}" \
    "${record_debug}" \
    "${min_speed_pct}" \
    "${max_speed_pct}" \
    "${fixed_speed_pct}" \
    "${step_duration_s}" \
    "${timeout_s}"

  rm -f "${START_LOG_FILE}"
  nohup /bin/bash "$0" run "${SESSION_ENV_FILE}" > "${START_LOG_FILE}" 2>&1 &
  local launch_pid="$!"

  local waited=0
  while [ "${waited}" -lt 30 ]; do
    sleep 0.1
    waited=$((waited + 1))
    runner_pid="$(read_file_trimmed "${RUNNER_PID_FILE}")"
    if [ -n "${runner_pid}" ] && pid_is_live "${runner_pid}"; then
      echo "[APEX] Recon session started (runner_pid=${runner_pid}, launch_pid=${launch_pid})"
      return 0
    fi
    if ! pid_is_live "${launch_pid}"; then
      break
    fi
  done

  echo "[APEX][ERROR] Recon session failed to start" >&2
  if [ -f "${START_LOG_FILE}" ]; then
    tail -n 40 "${START_LOG_FILE}" >&2 || true
  fi
  exit 1
}

cmd_stop() {
  clear_stale_session

  local runner_pid recon_pid
  runner_pid="$(read_file_trimmed "${RUNNER_PID_FILE}")"
  recon_pid="$(read_file_trimmed "${RECON_PID_FILE}")"
  if [ -z "${runner_pid}" ]; then
    echo "[APEX] No active recon session"
    if [ -f "${LAST_METADATA_FILE}" ]; then
      echo "[APEX] Last metadata: $(cat "${LAST_METADATA_FILE}")"
    fi
    if [ -f "${LAST_STATUS_FILE}" ] && [ "$(cat "${LAST_STATUS_FILE}")" = "invalid" ]; then
      return 1
    fi
    return 0
  fi

  echo "[APEX] Stopping recon session (runner_pid=${runner_pid}${recon_pid:+, recon_pid=${recon_pid}})"
  if [ -n "${recon_pid}" ] && pid_is_live "${recon_pid}"; then
    kill -INT "${recon_pid}" 2>/dev/null || true
  else
    kill -INT "${runner_pid}" 2>/dev/null || true
  fi

  if ! wait_for_pid_exit "${runner_pid}" 12; then
    if [ -n "${recon_pid}" ] && pid_is_live "${recon_pid}"; then
      echo "[APEX][WARN] Recon child still alive after SIGINT; escalating child stop" >&2
      kill -TERM "${recon_pid}" 2>/dev/null || true
      wait_for_pid_exit "${recon_pid}" 4 || true
      if pid_is_live "${recon_pid}"; then
        kill -KILL "${recon_pid}" 2>/dev/null || true
      fi
    fi
  fi

  if ! wait_for_pid_exit "${runner_pid}" 10; then
    echo "[APEX][ERROR] Recon session did not stop cleanly; escalating runner" >&2
    kill -TERM "${runner_pid}" 2>/dev/null || true
    if ! wait_for_pid_exit "${runner_pid}" 5; then
      kill -KILL "${runner_pid}" 2>/dev/null || true
    fi
  fi

  if pid_is_live "${runner_pid}"; then
    load_last_bundle_context || true
    stop_debug_bag_recording || true
    force_motor_neutral_from_params
    force_steering_center_from_params
    if [ "${APEX_RECORD_DEBUG:-0}" = "1" ] && [ -n "${APEX_DEBUG_BUNDLE_DIR:-}" ]; then
      write_pwm_snapshot "after"
      finalize_debug_artifacts || true
      write_debug_metadata || true
    fi
    if validate_debug_artifacts; then
      mark_last_run_validity "valid"
    else
      mark_last_run_validity "invalid"
    fi
    remove_live_state
  fi

  local last_status
  last_status="$(read_file_trimmed "${LAST_STATUS_FILE}")"
  if [ "${last_status}" = "invalid" ]; then
    local invalid_bundle=""
    invalid_bundle="$(read_file_trimmed "${LAST_INVALID_FILE}")"
    echo "[APEX][ERROR] Recon session stopped, but artifacts are invalid" >&2
    if [ -n "${invalid_bundle}" ]; then
      echo "[APEX][ERROR] Invalid bundle: ${invalid_bundle}" >&2
    fi
    return 1
  fi

  if [ -f "${LAST_METADATA_FILE}" ]; then
    echo "[APEX] Last metadata: $(cat "${LAST_METADATA_FILE}")"
  fi
  return 0
}

cmd_restart() {
  cmd_stop || true
  cmd_start "$@"
}

main() {
  local action="${1:-}"
  shift || true

  case "${action}" in
    start)
      cmd_start "$@"
      ;;
    stop)
      cmd_stop
      ;;
    status)
      cmd_status
      ;;
    restart)
      cmd_restart "$@"
      ;;
    run)
      cmd_run "$@"
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
