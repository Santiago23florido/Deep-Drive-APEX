#!/usr/bin/env bash
# Headless SLAM baseline capture: good vs damaged sensors.
#
#   1. launches Gazebo headless + the fusion research stack with both SLAMs
#      and raw-measurement recording (apex_fusion_research_up.sh --slam ...)
#   2. arms the recognition tour once the INS alignments are done
#   3. waits for the end of the tour (loop closed, timeout or abort)
#   4. stops the stack gracefully so every recorder writes its final files
#   5. generates the SLAM map comparison and the INS drift figures
#
# See simulation/ros2_ws/src/apex_fusion_research/README.md (SLAM baseline).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_SIM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SIM_WS_ROOT="${APEX_SIM_ROOT}/ros2_ws"
ROS_SETUP_SCRIPT="${APEX_ROS_SETUP_SCRIPT:-/opt/ros/jazzy/setup.bash}"

OUTPUT_ROOT="${APEX_SIM_ROOT}/data/fusion_research"
RUN_NAME="slam_baseline_$(date +%Y%m%d_%H%M%S)"
MAX_DURATION_S="420"
SETTLE_S="5"
ABLATION=("--slam-ablation")
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage: apex_fusion_slam_capture.sh [options] [-- extra apex_fusion_research_up.sh options]

  --run-name <name>        Run directory name (default: slam_baseline_<timestamp>)
  --output-root <dir>      Parent directory (default: simulation/data/fusion_research)
  --max-duration-s <s>     Wall-clock limit for the whole capture (default: 420)
  --settle-s <s>           Wait after the tour ends before stopping (default: 5)
  --no-ablation            Skip the ideal LiDAR + ideal-IMU heading SLAM (good_imu)
  -h, --help               Show this help

Anything after "--" is forwarded to apex_fusion_research_up.sh, e.g.
  apex_fusion_slam_capture.sh -- --imu-seed 3 --lidar-config ideal --pipeline-slam
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --max-duration-s) MAX_DURATION_S="$2"; shift 2 ;;
    --settle-s) SETTLE_S="$2"; shift 2 ;;
    --no-ablation) ABLATION=(); shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "[capture][ERROR] Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

# Same Conda/venv sanitisation as the launcher (ROS 2 Jazzy needs system Python).
PATH="$(printf '%s' "${PATH}" | tr ':' '\n' | grep -v -E 'conda|anaconda|miniforge|mambaforge' | paste -sd: -)"
export PATH
unset VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PROMPT_MODIFIER CONDA_EXE CONDA_PYTHON_EXE PYTHONHOME PYTHONPATH || true
set +u
source "${ROS_SETUP_SCRIPT}"
set -u

RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${OUTPUT_ROOT}"
LOG="${OUTPUT_ROOT}/${RUN_NAME}.launch.log"
echo "[capture] run directory: ${RUN_DIR}"
echo "[capture] launch log:    ${LOG}"

# Job control: without it, background jobs of a non-interactive shell start
# with SIGINT ignored and `ros2 launch` could not be stopped gracefully.
set -m
"${SCRIPT_DIR}/apex_fusion_research_up.sh" --headless --no-rviz --slam --record-measurements --arm \
  "${ABLATION[@]}" --output-root "${OUTPUT_ROOT}" --run-name "${RUN_NAME}" "${EXTRA_ARGS[@]}" >"${LOG}" 2>&1 &
LAUNCHER_PID=$!
set +m

wait_launcher_exit() {
  local seconds="$1"
  for _ in $(seq 1 "${seconds}"); do
    kill -0 "${LAUNCHER_PID}" 2>/dev/null || return 0
    sleep 1
  done
  return 1
}

stop_stack() {
  # SIGINT to `ros2 launch` propagates to every node (clean shutdown hooks
  # write the final maps); escalate to SIGTERM and SIGKILL if needed. The
  # launcher's own trap then removes any leftover process.
  local launch_pid sig
  for sig in INT TERM KILL; do
    launch_pid="$(pgrep -f "ros2 launch apex_fusion_research fusion_research_sim.launch.py" | head -1 || true)"
    [[ -z "${launch_pid}" ]] && break
    echo "[capture] stopping ros2 launch (${sig})"
    kill "-${sig}" "${launch_pid}" 2>/dev/null || true
    wait_launcher_exit 40 && return 0
  done
  wait_launcher_exit 20 || echo "[capture][WARN] launcher still running (pid ${LAUNCHER_PID})" >&2
}
trap stop_stack INT TERM

# Wait for the workspace build/launch, then for the end of the tour.
set +u
for _ in $(seq 1 120); do
  [[ -f "${SIM_WS_ROOT}/install/setup.bash" ]] && grep -q "recording run to" "${LOG}" 2>/dev/null && break
  kill -0 "${LAUNCHER_PID}" 2>/dev/null || { echo "[capture][ERROR] launcher exited early, see ${LOG}" >&2; exit 1; }
  sleep 1
done
source "${SIM_WS_ROOT}/install/setup.bash"
set -u

if ros2 run apex_fusion_research wait_for tour-end --timeout "${MAX_DURATION_S}"; then
  echo "[capture] tour finished; settling ${SETTLE_S} s"
else
  echo "[capture] no terminal tour state before the time limit; stopping anyway"
fi
sleep "${SETTLE_S}"
stop_stack
trap - INT TERM

echo "[capture] generating figures"
ros2 run apex_fusion_research plot_slam_maps "${RUN_DIR}" || echo "[capture] plot_slam_maps failed" >&2
ros2 run apex_fusion_research plot_ins_drift "${RUN_DIR}" || echo "[capture] plot_ins_drift failed" >&2

leftover="$(pgrep -af "gz sim|async_slam_toolbox_node|lib/apex_fusion_research/|install/apex_telemetry/lib" \
  | grep -v -E "pgrep|shell-snapshots" || true)"
if [[ -n "${leftover}" ]]; then
  echo "[capture][WARN] processes still running:" >&2
  echo "${leftover}" >&2
fi
echo "[capture] done: ${RUN_DIR}"
