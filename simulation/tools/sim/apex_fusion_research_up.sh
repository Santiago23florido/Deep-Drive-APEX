#!/usr/bin/env bash
# Launch Gazebo (GUI) + the LiDAR/IMU sensor-fusion research stack + RViz.
# See simulation/ros2_ws/src/apex_fusion_research/README.md for details.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_SIM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SIM_WS_ROOT="${APEX_SIM_ROOT}/ros2_ws"
ROS_SETUP_SCRIPT="${APEX_ROS_SETUP_SCRIPT:-/opt/ros/jazzy/setup.bash}"

SCENARIO="baseline"
CONTROL_MODE="recognition_tour"
IMU_CONFIG="consumer_mems"
LIDAR_CONFIG="rplidar_like"
INS_CONFIG="static_coarse"
IMU_SEED="-1"
LIDAR_SEED="-1"
GAZEBO_GUI="true"
RVIZ="true"
RECORD="true"
RUN_NAME=""
OUTPUT_ROOT=""
ARM="false"
ARM_DELAY_S="20"
SKIP_BUILD=0

usage() {
  cat <<'USAGE'
Usage: apex_fusion_research_up.sh [options]

Sensor models (preset name from config/<kind>/ or a path to a YAML file):
  --imu-config <p>     IMU error model        (default: consumer_mems | white_noise_only | ideal)
  --lidar-config <p>   LiDAR noise model      (default: rplidar_like | ideal)
  --ins-config <p>     INS alignment/mechan.  (default: static_coarse | truth_init)
  --imu-seed <n>       Override the IMU random seed   (-1 = keep YAML seed)
  --lidar-seed <n>     Override the LiDAR random seed (-1 = keep YAML seed)

Simulation:
  --scenario <name>    APEX scenario (default: baseline)
  --control-mode <m>   recognition_tour | manual_xbox | manual_windows_bridge
  --arm                Arm the autonomous recognition tour so the car drives
  --arm-delay-s <s>    Delay before arming (default: 20, after the INS alignment)
  --headless           No Gazebo GUI
  --no-rviz            Do not open RViz
  --no-record          Do not record CSV/metadata
  --run-name <name>    Run directory name (default: timestamp)
  --output-root <dir>  Where runs are stored (default: simulation/data/fusion_research)
  --skip-build         Do not run colcon build
  -h, --help           Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --imu-config) IMU_CONFIG="$2"; shift 2 ;;
    --lidar-config) LIDAR_CONFIG="$2"; shift 2 ;;
    --ins-config) INS_CONFIG="$2"; shift 2 ;;
    --imu-seed) IMU_SEED="$2"; shift 2 ;;
    --lidar-seed) LIDAR_SEED="$2"; shift 2 ;;
    --scenario) SCENARIO="$2"; shift 2 ;;
    --control-mode) CONTROL_MODE="$2"; shift 2 ;;
    --arm) ARM="true"; shift ;;
    --arm-delay-s) ARM_DELAY_S="$2"; shift 2 ;;
    --headless) GAZEBO_GUI="false"; shift ;;
    --no-rviz) RVIZ="false"; shift ;;
    --no-record) RECORD="false"; shift ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[fusion][ERROR] Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

# ROS 2 Jazzy is built against the system Python 3.12. An active Conda/venv
# Python shadows it and breaks rclpy (No module named 'rclpy._rclpy_pybind11').
sanitize_python_env() {
  PATH="$(printf '%s' "${PATH}" | tr ':' '\n' | grep -v -E 'conda|anaconda|miniforge|mambaforge' | paste -sd: -)"
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PATH="$(printf '%s' "${PATH}" | tr ':' '\n' | grep -v -F "${VIRTUAL_ENV}/bin" | paste -sd: -)"
  fi
  export PATH
  unset VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PROMPT_MODIFIER CONDA_EXE CONDA_PYTHON_EXE PYTHONHOME PYTHONPATH || true
}

cleanup() {
  pkill -f "ros2 launch apex_fusion_research" 2>/dev/null || true
  pkill -f "lib/apex_fusion_research/" 2>/dev/null || true
  pkill -f "rviz2 -d .*fusion_research.rviz" 2>/dev/null || true
  pkill -f "install/apex_telemetry/lib/apex_telemetry/" 2>/dev/null || true
  pkill -f "robot_state_publisher .*__node:=rc_car_state_publisher" 2>/dev/null || true
  pkill -f "apex_gz_vehicle_bridge.py" 2>/dev/null || true
  pkill -f "apex_ground_truth_node.py" 2>/dev/null || true
  pkill -f "parameter_bridge .*__node:=apex_sim_bridges" 2>/dev/null || true
  pkill -f "^gz sim" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sanitize_python_env
export APEX_SIM_ROOT
cd "${SIM_WS_ROOT}"
set +u
source "${ROS_SETUP_SCRIPT}"
set -u
cleanup
sleep 1

if [[ "${SKIP_BUILD}" != "1" ]]; then
  colcon build --symlink-install --base-paths "${SIM_WS_ROOT}/src" \
    --packages-select rc_sim_description apex_telemetry apex_fusion_research
fi
set +u
source "${SIM_WS_ROOT}/install/setup.bash"
set -u

if [[ "${ARM}" == "true" ]]; then
  ( sleep "${ARM_DELAY_S}"; "${APEX_SIM_ROOT}/tools/sim/apex_arm_recognition_tour.sh" ) &
fi

ros2 launch apex_fusion_research fusion_research_sim.launch.py \
  "scenario:=${SCENARIO}" \
  "control_mode:=${CONTROL_MODE}" \
  "gazebo_gui:=${GAZEBO_GUI}" \
  "rviz:=${RVIZ}" \
  "imu_config:=${IMU_CONFIG}" \
  "lidar_config:=${LIDAR_CONFIG}" \
  "ins_config:=${INS_CONFIG}" \
  "imu_seed:=${IMU_SEED}" \
  "lidar_seed:=${LIDAR_SEED}" \
  "record:=${RECORD}" \
  "run_name:=${RUN_NAME}" \
  "output_root:=${OUTPUT_ROOT}"
