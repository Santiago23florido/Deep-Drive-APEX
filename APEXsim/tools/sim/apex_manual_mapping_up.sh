#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_SIM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SIM_WS_ROOT="${APEX_SIM_ROOT}/ros2_ws"
RUNTIME_DIR="${APEX_RUNTIME_DIR:-${APEX_SIM_ROOT}/.apex_runtime/manual_mapping}"
ROS_SETUP_SCRIPT="${APEX_ROS_SETUP_SCRIPT:-/opt/ros/jazzy/setup.bash}"
SCENARIO="precision_fusion"
RVIZ="false"
SKIP_BUILD=0
GENERAL_MAPPER=1
PREVIEW_INTERVAL_S="12"
EVALUATION_WORLD="${APEX_SIM_ROOT}/ros2_ws/src/rc_sim_description/worlds/basic_track.world"

usage() {
  cat <<'EOF'
Usage: apex_manual_mapping_up.sh [options]

Options:
  --scenario <name>   Simulation scenario. Default: precision_fusion
  --rviz              Open the live manual-mapping RViz layout
  --general-mapper    Use the new general offline mapper preview pipeline
  --preview-interval-s <sec>  Preview rebuild interval. Default: 12
  --skip-build        Do not run colcon build
  -h, --help          Show this help
EOF
}

sanitize_python_env() {
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    if [[ "${PATH}" == "${VIRTUAL_ENV}/bin:"* ]]; then
      PATH="${PATH#${VIRTUAL_ENV}/bin:}"
    fi
    unset VIRTUAL_ENV
  fi
  unset PYTHONHOME || true
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario)
      SCENARIO="${2:-}"
      shift 2
      ;;
    --rviz)
      RVIZ="true"
      shift
      ;;
    --general-mapper)
      GENERAL_MAPPER=1
      shift
      ;;
    --preview-interval-s)
      PREVIEW_INTERVAL_S="${2:-}"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
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

mkdir -p "${RUNTIME_DIR}"

if [[ -f "${RUNTIME_DIR}/capture_pid" ]]; then
  kill -INT "$(cat "${RUNTIME_DIR}/capture_pid" 2>/dev/null || true)" 2>/dev/null || true
fi
if [[ -f "${RUNTIME_DIR}/capture_helper_pid" ]]; then
  kill -TERM "$(cat "${RUNTIME_DIR}/capture_helper_pid" 2>/dev/null || true)" 2>/dev/null || true
fi
if [[ -f "${RUNTIME_DIR}/live_offline_worker_pid" ]]; then
  kill -TERM "$(cat "${RUNTIME_DIR}/live_offline_worker_pid" 2>/dev/null || true)" 2>/dev/null || true
fi
if [[ -f "${RUNTIME_DIR}/live_offline_publisher_pid" ]]; then
  kill -TERM "$(cat "${RUNTIME_DIR}/live_offline_publisher_pid" 2>/dev/null || true)" 2>/dev/null || true
fi
if [[ -f "${RUNTIME_DIR}/mapping_preview_worker_pid" ]]; then
  kill -TERM "$(cat "${RUNTIME_DIR}/mapping_preview_worker_pid" 2>/dev/null || true)" 2>/dev/null || true
fi
if [[ -f "${RUNTIME_DIR}/mapping_preview_publisher_pid" ]]; then
  kill -TERM "$(cat "${RUNTIME_DIR}/mapping_preview_publisher_pid" 2>/dev/null || true)" 2>/dev/null || true
fi

if [[ -f "${RUNTIME_DIR}/offline_map_publisher_pid" ]]; then
  OLD_PID="$(cat "${RUNTIME_DIR}/offline_map_publisher_pid" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" ]]; then
    kill "${OLD_PID}" 2>/dev/null || true
  fi
  rm -f "${RUNTIME_DIR}/offline_map_publisher_pid"
fi

RUN_ID="manual_xbox_$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${APEX_SIM_ROOT}/data/apex_forward_raw/${RUN_ID}"
mkdir -p "${RUN_DIR}"

cd "${SIM_WS_ROOT}"
export APEX_SIM_ROOT
sanitize_python_env
if [[ "${SKIP_BUILD}" != "1" ]]; then
  set +u
  source "${ROS_SETUP_SCRIPT}"
  set -u
  colcon build \
    --symlink-install \
    --base-paths "${SIM_WS_ROOT}/src" \
    --packages-select rc_sim_description apex_telemetry voiture_system
fi

cat > "${RUN_DIR}/capture_meta.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "actuation_mode": "manual_windows_bridge",
  "scenario": "${SCENARIO}",
  "created_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

printf '%s\n' "${RUN_ID}" > "${RUNTIME_DIR}/active_run_id"
printf '%s\n' "${RUN_DIR}" > "${RUNTIME_DIR}/active_run_dir"
rm -f \
  "${RUNTIME_DIR}/capture_pid" \
  "${RUNTIME_DIR}/capture_helper_pid" \
  "${RUNTIME_DIR}/live_offline_worker_pid" \
  "${RUNTIME_DIR}/live_offline_publisher_pid" \
  "${RUNTIME_DIR}/mapping_preview_worker_pid" \
  "${RUNTIME_DIR}/mapping_preview_publisher_pid"

WINDOWS_EXE_PATH="${APEX_SIM_ROOT}/tools/windows/dist/apex_xbox_bridge_sim.exe"
echo "[APEX] Sim manual lista para puente Xbox desde Windows."
echo "[APEX] Abre este .exe en Windows:"
echo "  ${WINDOWS_EXE_PATH}"
if [[ ! -f "${WINDOWS_EXE_PATH}" ]]; then
  echo "[APEX][WARN] El .exe todavía no existe. Genéralo con:"
  echo "  ./APEXsim/tools/windows/build_apex_xbox_bridge_sim.sh"
fi
if [[ -n "${WSL_DISTRO_NAME:-}" ]]; then
  echo "[APEX] Ruta Windows equivalente:"
  echo "  \\\\wsl$\\${WSL_DISTRO_NAME}\\home\\santiago\\AiAtonomousRc\\APEXsim\\tools\\windows\\dist\\apex_xbox_bridge_sim.exe"
fi

(
  set +u
  source "${ROS_SETUP_SCRIPT}"
  if [[ -f "${SIM_WS_ROOT}/install/setup.bash" ]]; then
    source "${SIM_WS_ROOT}/install/setup.bash"
  fi
  set -u

  python3 "${SIM_WS_ROOT}/scripts/capture/wait_raw_pipeline_ready.py" \
    --imu-topic /apex/imu/data_raw \
    --scan-topic /lidar/scan_localization \
    --timeout-s 30.0 \
    --min-imu-messages 10 \
    --min-scan-messages 5 \
    --json-output "${RUN_DIR}/readiness.json" \
    > "${RUN_DIR}/readiness.log" 2>&1

  python3 "${SIM_WS_ROOT}/scripts/capture/record_manual_sensorfusion_capture.py" \
    --imu-topic /apex/imu/data_raw \
    --scan-topic /lidar/scan_localization \
    --odom-topic /apex/odometry/imu_lidar_fused \
    --imu-output "${RUN_DIR}/imu_raw.csv" \
    --lidar-output "${RUN_DIR}/lidar_points.csv" \
    --odom-output "${RUN_DIR}/odom_fused.csv" \
    --summary-json "${RUN_DIR}/capture_summary.json" \
    --status-json "${RUN_DIR}/capture_status.json" \
    > "${RUN_DIR}/capture.log" 2>&1 &
  CAPTURE_PID=$!
  printf '%s\n' "${CAPTURE_PID}" > "${RUNTIME_DIR}/capture_pid"
  wait "${CAPTURE_PID}"
) &
printf '%s\n' "$!" > "${RUNTIME_DIR}/capture_helper_pid"

if [[ "${GENERAL_MAPPER}" == "1" ]]; then
  nohup "${APEX_SIM_ROOT}/tools/sim/apex_general_map_publisher_up.sh" \
    "${RUN_DIR}/mapping_preview/fixed_map.yaml" \
    "${RUN_DIR}/mapping_preview/mapping_summary.json" \
    "odom_imu_lidar_fused" \
    "/apex/sim/mapping_preview/grid" \
    "/apex/sim/mapping_preview/visual_points" \
    "/apex/sim/mapping_preview/path" \
    "/apex/sim/mapping_preview/status" \
    "true" \
    "true" \
    > "${RUN_DIR}/mapping_preview_publisher.log" 2>&1 &
  printf '%s\n' "$!" > "${RUNTIME_DIR}/mapping_preview_publisher_pid"

  python3 "${APEX_SIM_ROOT}/tools/sim/apex_general_mapping_preview_worker.py" \
    --run-dir "${RUN_DIR}" \
    --mapper-script "${APEX_SIM_ROOT}/ros2_ws/src/rc_sim_description/scripts/apex_general_track_mapper.py" \
    --capture-status-json "${RUN_DIR}/capture_status.json" \
    --output-dir "${RUN_DIR}/mapping_preview" \
    --status-json "${RUN_DIR}/mapping_preview_status.json" \
    --log-path "${RUN_DIR}/mapping_preview.log" \
    --interval-s "${PREVIEW_INTERVAL_S}" \
    --min-scans 60 \
    --min-new-scans 35 \
    --evaluation-world "${EVALUATION_WORLD}" \
    > "${RUN_DIR}/mapping_preview_worker.log" 2>&1 &
  printf '%s\n' "$!" > "${RUNTIME_DIR}/mapping_preview_worker_pid"
fi

echo "[APEX] Progreso live del mapper general:"
echo "  tail -f ${RUN_DIR}/mapping_preview_status.json"
echo "  tail -f ${RUN_DIR}/mapping_preview.log"

launch_cmd=(
  "${APEX_SIM_ROOT}/tools/sim/apex_sim_up.sh"
  --scenario "${SCENARIO}"
  --control-mode manual_windows_bridge
  --skip-build
)
if [[ "${RVIZ}" == "true" ]]; then
  launch_cmd+=(--rviz)
fi

exec "${launch_cmd[@]}"
