#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_SIM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SIM_WS_ROOT="${APEX_SIM_ROOT}/ros2_ws"
RUNTIME_DIR="${APEX_RUNTIME_DIR:-${APEX_SIM_ROOT}/.apex_runtime/manual_mapping}"
ROS_SETUP_SCRIPT="${APEX_ROS_SETUP_SCRIPT:-/opt/ros/jazzy/setup.bash}"
OPEN_RVIZ=1
BUILD_FIXED_MAP=1

usage() {
  cat <<'EOF'
Usage: apex_manual_mapping_finish.sh [options]

Options:
  --build-fixed-map  Build the new general fixed map pipeline (default)
  --no-rviz     Do not open the offline-map RViz layout
  -h, --help    Show this help
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
    --build-fixed-map)
      BUILD_FIXED_MAP=1
      shift
      ;;
    --no-rviz)
      OPEN_RVIZ=0
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

if [[ ! -f "${RUNTIME_DIR}/active_run_dir" ]]; then
  echo "[APEX][ERROR] No active manual mapping run found in ${RUNTIME_DIR}" >&2
  exit 1
fi

RUN_DIR="$(cat "${RUNTIME_DIR}/active_run_dir")"
if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "[APEX][ERROR] Invalid run_dir in ${RUNTIME_DIR}/active_run_dir" >&2
  exit 1
fi

sanitize_python_env
set +u
source "${ROS_SETUP_SCRIPT}"
source "${SIM_WS_ROOT}/install/setup.bash"
set -u

FIXED_MAP_DIR="${RUN_DIR}/fixed_map"
EVALUATION_WORLD="${APEX_SIM_ROOT}/ros2_ws/src/rc_sim_description/worlds/basic_track.world"
mkdir -p "${FIXED_MAP_DIR}"
python3 - <<'PY' "${FIXED_MAP_DIR}/build_status.json" "${RUN_DIR}"
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
run_dir = pathlib.Path(sys.argv[2])
path.write_text(json.dumps({
    "stage": "stopping_manual_drive",
    "run_dir": str(run_dir),
}, indent=2), encoding="utf-8")
PY

echo "[APEX] Stopping manual drive and closing raw capture..."

pkill -f "apex_xbox_manual_teleop_node.py" 2>/dev/null || true
pkill -f "apex_windows_gamepad_bridge_node.py" 2>/dev/null || true
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

timeout 3s ros2 topic pub --once /apex/cmd_vel_track geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" \
  >/dev/null 2>&1 || true
sleep 0.6

if [[ -f "${RUNTIME_DIR}/capture_pid" ]]; then
  CAPTURE_PID="$(cat "${RUNTIME_DIR}/capture_pid" 2>/dev/null || true)"
  if [[ -n "${CAPTURE_PID}" ]]; then
    kill -INT "${CAPTURE_PID}" 2>/dev/null || true
  fi
fi
if [[ -f "${RUNTIME_DIR}/capture_helper_pid" ]]; then
  HELPER_PID="$(cat "${RUNTIME_DIR}/capture_helper_pid" 2>/dev/null || true)"
  if [[ -n "${HELPER_PID}" ]]; then
    python3 - <<'PY' "${FIXED_MAP_DIR}/build_status.json" "${RUN_DIR}"
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
run_dir = pathlib.Path(sys.argv[2])
path.write_text(json.dumps({
    "stage": "waiting_capture_shutdown",
    "run_dir": str(run_dir),
}, indent=2), encoding="utf-8")
PY
    timeout 15s bash -lc "while kill -0 ${HELPER_PID} 2>/dev/null; do sleep 0.2; done" || true
  fi
fi

SCAN_COUNT="$(python3 - <<'PY' "${RUN_DIR}/capture_summary.json"
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
if not path.exists():
    print("unknown")
    raise SystemExit(0)
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("unknown")
    raise SystemExit(0)
print(payload.get("scan_count", "unknown"))
PY
)"
POINT_COUNT="$(python3 - <<'PY' "${RUN_DIR}/capture_summary.json"
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
if not path.exists():
    print("unknown")
    raise SystemExit(0)
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("unknown")
    raise SystemExit(0)
print(payload.get("lidar_point_count", payload.get("point_count", "unknown")))
PY
)"

echo "[APEX] Building fixed map..."
echo "[APEX] Run dir: ${RUN_DIR}"
echo "[APEX] Captured scans: ${SCAN_COUNT} | points: ${POINT_COUNT}"
echo "[APEX] This step can take several minutes, but now writes progress live."
echo "[APEX] Progress files:"
echo "  ${RUN_DIR}/fixed_map/build_status.json"
echo "  ${RUN_DIR}/mapping_build.log"
echo "[APEX] Puedes vigilarlo en otra terminal con:"
echo "  tail -f ${RUN_DIR}/fixed_map/build_status.json"
echo "  tail -f ${RUN_DIR}/mapping_build.log"

touch "${RUN_DIR}/mapping_build.log"
python3 -u "${APEX_SIM_ROOT}/ros2_ws/src/rc_sim_description/scripts/apex_general_track_mapper.py" \
  --run-dir "${RUN_DIR}" \
  --output-dir "${FIXED_MAP_DIR}" \
  --status-json "${FIXED_MAP_DIR}/build_status.json" \
  --evaluation-world "${EVALUATION_WORLD}" \
  --evaluation-json "${FIXED_MAP_DIR}/mapping_evaluation.json" \
  > "${RUN_DIR}/mapping_build.log" 2>&1 &
OFFLINE_FUSION_PID=$!
START_S="$(date +%s)"
while kill -0 "${OFFLINE_FUSION_PID}" 2>/dev/null; do
  NOW_S="$(date +%s)"
  ELAPSED_S="$((NOW_S - START_S))"
  STAGE="$(python3 - <<'PY' "${FIXED_MAP_DIR}/build_status.json"
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
if not path.exists():
    print("starting")
    raise SystemExit(0)
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("unknown")
    raise SystemExit(0)
print(payload.get("stage", "unknown"))
PY
)"
  KEYFRAMES="$(python3 - <<'PY' "${FIXED_MAP_DIR}/build_status.json"
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
if not path.exists():
    print("0")
    raise SystemExit(0)
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("0")
    raise SystemExit(0)
print(payload.get("keyframe_count", payload.get("progress_index", 0)))
PY
)"
  echo "[APEX] Fixed-map build running... elapsed=${ELAPSED_S}s stage=${STAGE} keyframes=${KEYFRAMES}"
  sleep 10
done
wait "${OFFLINE_FUSION_PID}"
echo "[APEX] Fixed-map build finished."

if [[ -f "${RUNTIME_DIR}/offline_map_publisher_pid" ]]; then
  OLD_PID="$(cat "${RUNTIME_DIR}/offline_map_publisher_pid" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" ]]; then
    kill "${OLD_PID}" 2>/dev/null || true
  fi
fi

nohup "${APEX_SIM_ROOT}/tools/sim/apex_general_map_publisher_up.sh" \
  "${FIXED_MAP_DIR}/fixed_map.yaml" \
  "${FIXED_MAP_DIR}/mapping_summary.json" \
  "odom_imu_lidar_fused" \
  "/apex/sim/fixed_map/grid" \
  "/apex/sim/fixed_map/visual_points" \
  "/apex/sim/fixed_map/path" \
  "/apex/sim/fixed_map/status" \
  "false" \
  "false" \
  > "${RUN_DIR}/offline_map_publisher.log" 2>&1 &
printf '%s\n' "$!" > "${RUNTIME_DIR}/offline_map_publisher_pid"
echo "[APEX] Fixed-map publisher started."

if [[ "${OPEN_RVIZ}" == "1" ]]; then
  nohup bash -lc "
    set -e
    set +u
    source '${ROS_SETUP_SCRIPT}'
    source '${SIM_WS_ROOT}/install/setup.bash'
    set -u
    exec rviz2 -d '${APEX_SIM_ROOT}/rviz/apex_manual_mapping_offline.rviz'
  " > "${RUN_DIR}/offline_map_rviz.log" 2>&1 &
  echo "[APEX] Fixed-map RViz launched."
fi

echo "[APEX] Fixed map ready: ${RUN_DIR}"
echo "[APEX] Files:"
echo "  ${RUN_DIR}/imu_raw.csv"
echo "  ${RUN_DIR}/lidar_points.csv"
echo "  ${FIXED_MAP_DIR}/fixed_map.yaml"
echo "  ${FIXED_MAP_DIR}/fixed_map.pgm"
echo "  ${FIXED_MAP_DIR}/fixed_map_distance.npy"
echo "  ${FIXED_MAP_DIR}/fixed_map_visual_points.csv"
echo "  ${FIXED_MAP_DIR}/optimized_keyframes.csv"
echo "  ${FIXED_MAP_DIR}/mapping_vs_gazebo_overlay.png"
echo "  ${FIXED_MAP_DIR}/mapping_vs_gazebo_metrics.csv"
echo "  ${FIXED_MAP_DIR}/submap_cache/"
echo "[APEX] RViz topics:"
echo "  /apex/sim/fixed_map/grid"
echo "  /apex/sim/fixed_map/visual_points"
echo "  /apex/sim/fixed_map/path"
echo "[APEX] Ver mapa final otra vez:"
echo "  ./APEXsim/tools/sim/apex_general_map_publisher_up.sh ${FIXED_MAP_DIR}/fixed_map.yaml ${FIXED_MAP_DIR}/mapping_summary.json"
echo "  rviz2 -d ${APEX_SIM_ROOT}/rviz/apex_manual_mapping_offline.rviz"
echo "[APEX] Fixed-map run command:"
echo "  ./APEXsim/tools/sim/apex_sim_up.sh --scenario baseline --rviz --fixed-map-run ${RUN_DIR}"
