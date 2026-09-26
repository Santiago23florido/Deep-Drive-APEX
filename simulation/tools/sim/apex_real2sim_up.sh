#!/usr/bin/env bash
# Real2sim closed-loop run on the car's own learned LiDAR-inertial odometry +
# slam_toolbox, with Gazebo-native sensors made real by the realism layer
# (A2M8 + LSM6DS3).
#   --driver race (default): the car only knows its odometry model and its own
#     limits: lap 1 reactive from the LiDAR while the map is built, then a race
#     line planned on its map for the remaining laps.
#   --driver format: the dataset driver on the seeded reference path of the
#     format (validation of the estimator only).
# The run ends when the referee (ground truth) declares it completed or
# failed; the evaluation (odometry, SLAM map, race metrics, latencies) is then
# written to the run directory.
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
LAPS=""
SENSOR="APEX_real"
CHECKPOINT="${APEX_SIM_ROOT}/learning/outputs/real2sim_v2_fast/hybrid_submap_v3/best_model.pt"
DEVICE="cpu"
DRIVE="estimate"
START_POSE="true"
GUI="false"
RVIZ="true"
RVIZ_CONFIG=""
RUN_NAME=""
OUTPUT_ROOT="${APEX_SIM_ROOT}/data/real2sim"
TIMEOUT_S=""
DRIVER="race"
W_MIN="1.5"
V_EXPLORE="1.5"
RACE_LINE="min_curvature"
V_MAX_RACE="3.5"
A_LAT="3.0"
SKIP_BUILD=0

usage() {
  cat <<'USAGE'
Usage: apex_real2sim_up.sh [options]

Format (as in tools/pose_dataset/config/campaign.yaml):
  --track <name>       track (default: val_mixed, the validation track)
  --motion <name>      motion profile (default: medium; slow, slow_variable, ... )
  --seed <n>           seed of the format (start point, direction, gains, sensor draws) (default: 1)
  --laps <x>           laps to drive (default: 3 with --driver race, 2 with --driver format)
  --sensor <profile>   sensor profile of sensors.yaml (default: APEX_real; APEX_real_compat = 2 kHz A2M8)
Estimator:
  --checkpoint <file>  streaming / hybrid checkpoint (default: real2sim_v2_fast/hybrid_submap_v3/best_model.pt, the latest trained on APEX_real)
  --device <d>         cpu (default, ~20 ms per scan) | cuda
Driver:
  --driver <d>         race (default: reactive lap 1 + map + race line) | format (dataset path)
  --w-min <m>          race: minimum lane width the car assumes (default: 1.5)
  --v-explore <m/s>    race: top speed of the reactive lap 1 (default: 1.5)
  --race-line <mode>   race: min_curvature (default) | centre
  --v-max-race <m/s>   race: top speed on the race line (default: 3.5)
  --a-lat <m/s2>       race: lateral acceleration limit (default: 3.0)
Run:
  --drive <src>        format only: estimate (closed loop, default) | truth (diagnostic)
  --start-pose <s>     true (default: the path starts where the car is) | nominal (start line, placement error unknown)
  --gui                Gazebo GUI (default: headless server)
  --no-rviz            do not open RViz
  --rviz-config FILE   RViz config (default: rviz/real2sim.rviz)
  --run-name <name>    run directory name (default: <track>__<motion>__s<seed>__<sensor>__<timestamp>)
  --output-root <dir>  where runs are stored (default: simulation/data/real2sim)
  --timeout-s <s>      wall-clock limit of the run (default: 1800 race, 900 format)
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
    --driver) DRIVER="$2"; shift 2 ;;
    --w-min) W_MIN="$2"; shift 2 ;;
    --v-explore) V_EXPLORE="$2"; shift 2 ;;
    --race-line) RACE_LINE="$2"; shift 2 ;;
    --v-max-race) V_MAX_RACE="$2"; shift 2 ;;
    --a-lat) A_LAT="$2"; shift 2 ;;
    --start-pose) START_POSE="$2"; shift 2 ;;
    --gui) GUI="true"; shift ;;
    --no-rviz) RVIZ="false"; shift ;;
    --rviz-config) RVIZ_CONFIG="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --timeout-s) TIMEOUT_S="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[real2sim][ERROR] Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "${LAPS}" ]]; then LAPS="$([[ "${DRIVER}" == "race" ]] && echo 3 || echo 2)"; fi
if [[ -z "${TIMEOUT_S}" ]]; then TIMEOUT_S="$([[ "${DRIVER}" == "race" ]] && echo 1800 || echo 900)"; fi

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
  pkill -f "lib/apex_fusion_research/(real_sensor_node|sim_actuation_node|track_driver_node|race_driver_node|run_referee_node|slam_map_recorder_node)" 2>/dev/null || true
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
  "checkpoint:=${CHECKPOINT}" "device:=${DEVICE}" "drive:=${DRIVE}" "start_pose:=${START_POSE}" "gui:=${GUI}" "rviz:=${RVIZ}" ${RVIZ_CONFIG:+"rviz_config:=${RVIZ_CONFIG}"} "run_dir:=${RUN_DIR}" \
  "driver:=${DRIVER}" "w_min:=${W_MIN}" "v_explore:=${V_EXPLORE}" "race_line:=${RACE_LINE}" "v_max_race:=${V_MAX_RACE}" "a_lat:=${A_LAT}" \
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
"${VENV_PYTHON}" "${APEX_SIM_ROOT}/tools/analysis/plot_real2sim_sensors.py" "${RUN_DIR}" || echo "[real2sim] figures failed"
