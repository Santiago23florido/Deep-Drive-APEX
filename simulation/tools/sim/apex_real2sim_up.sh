#!/usr/bin/env bash
# Real2sim closed-loop run: the car drives a pose-dataset format (track,
# motion, seed, laps) on its own learned LiDAR-inertial odometry + slam_toolbox,
# with Gazebo-native sensors made real by the realism layer (A2M8 + LSM6DS3).
# The run ends when the referee (ground truth) declares the lap completed or
# failed; the evaluation (odometry metrics, SLAM map metrics, latencies) is
# then written to the run directory.
# See simulation/ros2_ws/src/apex_fusion_research/README.md (real2sim).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_SIM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SIM_WS_ROOT="${APEX_SIM_ROOT}/ros2_ws"
ROS_SETUP_SCRIPT="${APEX_ROS_SETUP_SCRIPT:-/opt/ros/jazzy/setup.bash}"
VENV_PYTHON="${APEX_SIM_ROOT}/learning/.venv/bin/python"

TRACK="val_mixed"
MOTION="medium"
SEED="1"
LAPS="2"
SENSOR="APEX_real"
CHECKPOINT="${APEX_SIM_ROOT}/learning/outputs/real2sim_v2/hybrid_submap/best_model.pt"
DEVICE="cpu"
DRIVE="estimate"
START_POSE="true"
GUI="false"
RVIZ="true"
RUN_NAME=""
OUTPUT_ROOT="${APEX_SIM_ROOT}/data/real2sim"
TIMEOUT_S="900"
SKIP_BUILD=0

usage() {
  cat <<'USAGE'
Usage: apex_real2sim_up.sh [options]

Format (as in tools/pose_dataset/config/campaign.yaml):
  --track <name>       track (default: val_mixed, the validation track)
  --motion <name>      motion profile (default: medium; slow, slow_variable, ... )
  --seed <n>           seed of the format (start point, direction, gains, sensor draws) (default: 1)
  --laps <x>           laps to drive (default: 2)
  --sensor <profile>   sensor profile of sensors.yaml (default: APEX_real; APEX_real_compat = 2 kHz A2M8)
Estimator:
  --checkpoint <file>  streaming / hybrid checkpoint (default: real2sim_v2/hybrid_submap/best_model.pt, trained on APEX_real)
  --device <d>         cpu (default, ~20 ms per scan) | cuda
Run:
  --drive <src>        estimate (closed loop, default) | truth (diagnostic)
  --start-pose <s>     true (default: the path starts where the car is) | nominal (start line, placement error unknown)
  --gui                Gazebo GUI (default: headless server)
  --no-rviz            do not open RViz
  --run-name <name>    run directory name (default: <track>__<motion>__s<seed>__<sensor>__<timestamp>)
  --output-root <dir>  where runs are stored (default: simulation/data/real2sim)
  --timeout-s <s>      wall-clock limit of the run (default: 900)
  --skip-build         do not run colcon build
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --track) TRACK="$2"; shift 2 ;;
    --motion) MOTION="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --laps) LAPS="$2"; shift 2 ;;
    --sensor) SENSOR="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --drive) DRIVE="$2"; shift 2 ;;
    --start-pose) START_POSE="$2"; shift 2 ;;
    --gui) GUI="true"; shift ;;
    --no-rviz) RVIZ="false"; shift ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --timeout-s) TIMEOUT_S="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[real2sim][ERROR] Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

# ROS 2 Jazzy is built against the system Python 3.12: drop Conda / venvs.
sanitize_python_env() {
  PATH="$(printf '%s' "${PATH}" | tr ':' '\n' | grep -v -E 'conda|anaconda|miniforge|mambaforge' | paste -sd: -)"
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PATH="$(printf '%s' "${PATH}" | tr ':' '\n' | grep -v -F "${VIRTUAL_ENV}/bin" | paste -sd: -)"
  fi
  export PATH
  unset VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PROMPT_MODIFIER CONDA_EXE CONDA_PYTHON_EXE PYTHONHOME PYTHONPATH || true
}

LAUNCH_PID=""
cleanup() {
  if [[ -n "${LAUNCH_PID}" ]] && kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    kill -INT -- "-${LAUNCH_PID}" 2>/dev/null || true
    for _ in $(seq 1 40); do kill -0 "${LAUNCH_PID}" 2>/dev/null || break; sleep 0.5; done
    kill -KILL -- "-${LAUNCH_PID}" 2>/dev/null || true
  fi
  # Leftovers of this launch only (never the dataset generator workers).
  pkill -f "apex_real2sim.launch.py" 2>/dev/null || true
  pkill -f "learned_odometry_node.py" 2>/dev/null || true
  pkill -f "lib/apex_fusion_research/(real_sensor_node|sim_actuation_node|track_driver_node|run_referee_node|slam_map_recorder_node)" 2>/dev/null || true
  pkill -f "real2sim_(clock_bridge|laser_tf|imu_tf|world_map_tf|rviz)" 2>/dev/null || true
  pkill -f "gz sim -r --seed" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sanitize_python_env
export APEX_SIM_ROOT
cd "${SIM_WS_ROOT}"
set +u
source "${ROS_SETUP_SCRIPT}"
set -u
if [[ "${SKIP_BUILD}" != "1" ]]; then
  colcon build --symlink-install --base-paths "${SIM_WS_ROOT}/src" --packages-select rc_sim_description apex_fusion_research
fi
set +u
source "${SIM_WS_ROOT}/install/setup.bash"
set -u

RUN_NAME="${RUN_NAME:-${TRACK}__${MOTION}__s${SEED}__${SENSOR}__$(date +%Y%m%dT%H%M%S)}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
echo "[real2sim] run directory: ${RUN_DIR}"
WALL_START="$(date +%s.%N)"

setsid ros2 launch apex_fusion_research apex_real2sim.launch.py \
  "track:=${TRACK}" "motion:=${MOTION}" "seed:=${SEED}" "laps:=${LAPS}" "sensor_profile:=${SENSOR}" \
  "checkpoint:=${CHECKPOINT}" "device:=${DEVICE}" "drive:=${DRIVE}" "start_pose:=${START_POSE}" "gui:=${GUI}" "rviz:=${RVIZ}" "run_dir:=${RUN_DIR}" \
  > >(tee "${RUN_DIR}/launch.log") 2>&1 &
LAUNCH_PID=$!

END_REASON="timeout"
for _ in $(seq 1 "$((TIMEOUT_S * 2))"); do
  if [[ -f "${RUN_DIR}/run_result.json" ]]; then END_REASON="referee"; break; fi
  if ! kill -0 "${LAUNCH_PID}" 2>/dev/null; then END_REASON="launch exited"; break; fi
  sleep 0.5
done
sleep 2  # last SLAM snapshot / estimates
echo "[real2sim] stopping (${END_REASON})"
cleanup
LAUNCH_PID=""
WALL_END="$(date +%s.%N)"
printf '{"wall_start": %s, "wall_end": %s, "end_reason": "%s"}\n' "${WALL_START}" "${WALL_END}" "${END_REASON}" > "${RUN_DIR}/wall.json"

ros2 run apex_fusion_research plot_slam_maps "${RUN_DIR}" || echo "[real2sim] plot_slam_maps failed"
"${VENV_PYTHON}" "${APEX_SIM_ROOT}/tools/analysis/evaluate_real2sim_run.py" "${RUN_DIR}" || echo "[real2sim] evaluation failed"
