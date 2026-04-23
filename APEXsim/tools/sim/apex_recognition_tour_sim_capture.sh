#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_SIM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SIM_WS_ROOT="${APEX_SIM_ROOT}/ros2_ws"
ROS_SETUP_SCRIPT="${APEX_ROS_SETUP_SCRIPT:-/opt/ros/jazzy/setup.bash}"
RUN_ROOT="${APEX_RECOGNITION_TOUR_ROOT:-${APEX_SIM_ROOT}/data/apex_recognition_tour}"
SCENARIO="baseline"
TIMEOUT_S="60"
RUN_ID="recognition_tour_sim"
REAL_RUN=""
SKIP_BUILD=0
SIM_PID=""

usage() {
  cat <<'EOF'
Usage: apex_recognition_tour_sim_capture.sh [options]

Options:
  --scenario <name>
  --timeout-s <seconds>
  --run-id <id>
  --real-run <path>
  --skip-build
  -h, --help
EOF
}

cleanup() {
  if [[ -n "${SIM_PID}" ]]; then
    kill "${SIM_PID}" 2>/dev/null || true
    wait "${SIM_PID}" 2>/dev/null || true
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario)
      SCENARIO="${2:-}"
      shift 2
      ;;
    --timeout-s)
      TIMEOUT_S="${2:-}"
      shift 2
      ;;
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --real-run)
      REAL_RUN="${2:-}"
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

trap cleanup EXIT INT TERM

RUN_NAME="${RUN_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

cd "${SIM_WS_ROOT}"
export APEX_SIM_ROOT
set +u
source "${ROS_SETUP_SCRIPT}"
source "${SIM_WS_ROOT}/install/setup.bash" 2>/dev/null || true
set -u

sim_up_cmd=("${SCRIPT_DIR}/apex_sim_up.sh" --scenario "${SCENARIO}")
if [[ "${SKIP_BUILD}" == "1" ]]; then
  sim_up_cmd+=(--skip-build)
fi

"${sim_up_cmd[@]}" > "${RUN_DIR}/apex_sim_stdout.log" 2>&1 &
SIM_PID=$!

required_nodes=(
  /nano_accel_serial_node
  /apex_rplidar_publisher
  /imu_lidar_planar_fusion_node
  /recognition_tour_planner_node
  /recognition_tour_tracker_node
  /cmd_vel_to_apex_actuation_node
  /apex_gz_vehicle_bridge
  /apex_ground_truth_node
)

for _ in $(seq 1 120); do
  set +e
  node_list="$(ros2 node list 2>/dev/null)"
  status=$?
  set -e
  if [[ ${status} -eq 0 ]]; then
    ready=1
    for node_name in "${required_nodes[@]}"; do
      if ! grep -qx "${node_name}" <<<"${node_list}"; then
        ready=0
        break
      fi
    done
    if [[ "${ready}" == "1" ]]; then
      break
    fi
  fi
  sleep 1
done

node_list="$(ros2 node list 2>/dev/null || true)"
for node_name in "${required_nodes[@]}"; do
  if ! grep -qx "${node_name}" <<<"${node_list}"; then
    echo "[APEX][ERROR] Simulation did not become ready; missing ${node_name}" >&2
    exit 1
  fi
done

python3 "${SIM_WS_ROOT}/scripts/capture/record_recognition_tour.py" \
  --path-topic /apex/planning/recognition_tour_local_path \
  --route-topic /apex/planning/recognition_tour_route \
  --fusion-status-topic /apex/estimation/status \
  --planner-status-topic /apex/planning/recognition_tour_status \
  --tracker-status-topic /apex/tracking/recognition_tour_status \
  --bridge-status-topic /apex/vehicle/drive_bridge_status \
  --odom-topic /apex/odometry/imu_lidar_fused \
  --scan-topic /lidar/scan_localization \
  --ground-truth-odom-topic /apex/sim/ground_truth/odom \
  --ground-truth-status-topic /apex/sim/ground_truth/status \
  --ground-truth-map-topic /apex/sim/ground_truth/perfect_map_points \
  --lidar-offset-x-m 0.18 \
  --lidar-offset-y-m 0.0 \
  --output-dir "${RUN_DIR}" \
  --timeout-s "${TIMEOUT_S}" \
  > "${RUN_DIR}/recognition_tour_record.log" 2>&1 &
RECORDER_PID=$!

sleep 2
"${SCRIPT_DIR}/apex_arm_recognition_tour.sh" --duration-s 2.0 --rate-hz 10

wait "${RECORDER_PID}"

python3 "${APEX_SIM_ROOT}/tools/analysis/plot_recognition_tour_run.py" \
  --run-dir "${RUN_DIR}" \
  >> "${RUN_DIR}/recognition_tour_record.log" 2>&1 || true

if [[ -n "${REAL_RUN}" ]]; then
  python3 "${APEX_SIM_ROOT}/tools/analysis/compare_recognition_tour_runs.py" \
    --real-run "${REAL_RUN}" \
    --sim-run "${RUN_DIR}" \
    >> "${RUN_DIR}/recognition_tour_record.log" 2>&1 || true
fi

echo "[APEX] Sim recognition_tour capture ready: ${RUN_DIR}"
