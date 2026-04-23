#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_SIM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SIM_WS_ROOT="${APEX_SIM_ROOT}/ros2_ws"
ROS_SETUP_SCRIPT="${APEX_ROS_SETUP_SCRIPT:-/opt/ros/jazzy/setup.bash}"
KILL_STALE="${APEX_SIM_KILL_STALE:-1}"
SCENARIO="baseline"
RVIZ="false"
USE_SLAM="false"
USE_REFINED_VISUAL_MAP="false"
CONTROL_MODE="recognition_tour"
ARM_AFTER_START="false"
ARM_DELAY_S="12"
SKIP_BUILD=0
FIXED_MAP_RUN=""
SPAWN_X=""
SPAWN_Y=""
SPAWN_Z=""
SPAWN_YAW_DEG=""

usage() {
  cat <<'EOF'
Usage: apex_sim_up.sh [options]

Options:
  --scenario <name>    baseline | precision_fusion | tight_right_saturation | outer_long_inner_short | startup_pose_jump | narrowing_false_corridor
  --control-mode <m>   recognition_tour | manual_xbox | manual_windows_bridge
  --rviz               Launch RViz
  --slam               Launch slam_toolbox and open the SLAM RViz layout by default
  --refined-map        Launch the forward_raw-style refined visual map node and matching RViz layout
  --fixed-map-run <d>  Use fixed-map localization from a finished manual mapping run dir
  --arm                Arm recognition_tour automatically after startup
  --arm-delay-s <sec>  Delay before auto-arm, default: 12
  --x <meters>         Override spawn x
  --y <meters>         Override spawn y
  --z <meters>         Override spawn z
  --yaw-deg <deg>      Override spawn yaw in degrees
  --skip-build         Do not run colcon build before launch
  -h, --help           Show this help
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

cleanup_stale_sim() {
  if [[ "${KILL_STALE}" != "1" ]]; then
    return
  fi
  pkill -f "ros2 launch rc_sim_description apex_sim.launch.py" 2>/dev/null || true
  pkill -f "apex_gz_vehicle_bridge.py" 2>/dev/null || true
  pkill -f "apex_ground_truth_node.py" 2>/dev/null || true
  pkill -f "apex_xbox_manual_teleop_node.py" 2>/dev/null || true
  pkill -f "apex_windows_gamepad_bridge_node.py" 2>/dev/null || true
  pkill -f "apex_general_track_map_publisher.py" 2>/dev/null || true
  pkill -f "parameter_bridge .*__node:=apex_sim_bridges" 2>/dev/null || true
  pkill -f "async_slam_toolbox_node" 2>/dev/null || true
  pkill -f "^gz sim server$" 2>/dev/null || true
  pkill -f "^gz sim gui$" 2>/dev/null || true
  pkill -f "^gz sim -r " 2>/dev/null || true
  sleep 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario)
      SCENARIO="${2:-}"
      shift 2
      ;;
    --control-mode)
      CONTROL_MODE="${2:-}"
      shift 2
      ;;
    --rviz)
      RVIZ="true"
      shift
      ;;
    --slam)
      USE_SLAM="true"
      shift
      ;;
    --refined-map)
      USE_REFINED_VISUAL_MAP="true"
      shift
      ;;
    --fixed-map-run)
      FIXED_MAP_RUN="${2:-}"
      shift 2
      ;;
    --arm)
      ARM_AFTER_START="true"
      shift
      ;;
    --arm-delay-s)
      ARM_DELAY_S="${2:-}"
      shift 2
      ;;
    --x)
      SPAWN_X="${2:-}"
      shift 2
      ;;
    --y)
      SPAWN_Y="${2:-}"
      shift 2
      ;;
    --z)
      SPAWN_Z="${2:-}"
      shift 2
      ;;
    --yaw-deg)
      SPAWN_YAW_DEG="${2:-}"
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

trap cleanup_stale_sim EXIT INT TERM

cd "${SIM_WS_ROOT}"
export APEX_SIM_ROOT
sanitize_python_env
set +u
source "${ROS_SETUP_SCRIPT}"
set -u

cleanup_stale_sim

if [[ "${SKIP_BUILD}" != "1" ]]; then
  colcon build \
    --symlink-install \
    --base-paths "${SIM_WS_ROOT}/src" \
    --packages-select rc_sim_description apex_telemetry voiture_system
fi

set +u
source "${SIM_WS_ROOT}/install/setup.bash"
set -u

if [[ "${ARM_AFTER_START}" == "true" ]]; then
  (
    sleep "${ARM_DELAY_S}"
    set +u
    source "${ROS_SETUP_SCRIPT}"
    source "${SIM_WS_ROOT}/install/setup.bash"
    set -u
    "${APEX_SIM_ROOT}/tools/sim/apex_arm_recognition_tour.sh"
  ) &
fi

launch_cmd=(
  ros2 launch rc_sim_description apex_sim.launch.py
  "scenario:=${SCENARIO}"
  "control_mode:=${CONTROL_MODE}"
  "rviz:=${RVIZ}"
  "use_slam:=${USE_SLAM}"
  "use_refined_visual_map:=${USE_REFINED_VISUAL_MAP}"
)
if [[ -n "${FIXED_MAP_RUN}" ]]; then
  launch_cmd+=("fixed_map_run:=${FIXED_MAP_RUN}")
fi
if [[ -n "${SPAWN_X}" ]]; then
  launch_cmd+=("x:=${SPAWN_X}")
fi
if [[ -n "${SPAWN_Y}" ]]; then
  launch_cmd+=("y:=${SPAWN_Y}")
fi
if [[ -n "${SPAWN_Z}" ]]; then
  launch_cmd+=("z:=${SPAWN_Z}")
fi
if [[ -n "${SPAWN_YAW_DEG}" ]]; then
  launch_cmd+=("yaw_deg:=${SPAWN_YAW_DEG}")
fi

"${launch_cmd[@]}"
