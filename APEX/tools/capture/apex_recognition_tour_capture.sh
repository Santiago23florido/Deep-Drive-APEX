#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTAINER_NAME="${APEX_CONTAINER_NAME:-apex_pipeline}"
ROS_SETUP_SCRIPT="${APEX_ROS_SETUP_SCRIPT:-/opt/ros/jazzy/setup.bash}"
HOST_OUTPUT_ROOT="${APEX_RECOGNITION_TOUR_ROOT:-${APEX_ROOT}/ros2_ws/apex_recognition_tour}"
CONTAINER_OUTPUT_ROOT="${APEX_RECOGNITION_TOUR_CONTAINER_ROOT:-/work/ros2_ws/apex_recognition_tour}"
RUN_ID="${APEX_RECOGNITION_TOUR_RUN_ID:-recognition_tour}"
RUN_TIMEOUT_S="${APEX_RECOGNITION_TOUR_TIMEOUT_S:-60.0}"
ARM_TOPIC="${APEX_RECOGNITION_TOUR_ARM_TOPIC:-/apex/tracking/arm}"
BRIDGE_MIN_SPEED_PCT="${APEX_RECOGNITION_TOUR_MIN_SPEED_PCT:-14.0}"
BRIDGE_MAX_SPEED_PCT="${APEX_RECOGNITION_TOUR_MAX_SPEED_PCT:-20.0}"

usage() {
  cat <<'EOF'
Usage: apex_recognition_tour_capture.sh [options]

Options:
  --run-id <id>
  --timeout-s <seconds>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --timeout-s)
      RUN_TIMEOUT_S="${2:-}"
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

CAPTURE_NAME="${RUN_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
HOST_RUN_DIR="${HOST_OUTPUT_ROOT}/${CAPTURE_NAME}"
CONTAINER_RUN_DIR="${CONTAINER_OUTPUT_ROOT}/${CAPTURE_NAME}"
mkdir -p "${HOST_RUN_DIR}"

cat > "${HOST_RUN_DIR}/capture_meta.json" <<EOF
{
  "run_id": "${CAPTURE_NAME}",
  "mode": "recognition_tour",
  "timeout_s": ${RUN_TIMEOUT_S},
  "enable_imu_lidar_fusion": 1,
  "enable_recognition_tour_planner": 1,
  "enable_recognition_tour_tracker": 1,
  "enable_cmdvel_actuation_bridge": 1,
  "bridge_min_effective_speed_pct": ${BRIDGE_MIN_SPEED_PCT},
  "bridge_max_speed_pct": ${BRIDGE_MAX_SPEED_PCT}
}
EOF

cd "${APEX_ROOT}"
export PATH="${HOME}/local/bin:${PATH}"
./tools/core/apex_core_down.sh
APEX_ENABLE_IMU_LIDAR_FUSION=1 \
APEX_ENABLE_CURVE_ENTRY_PLANNER=0 \
APEX_ENABLE_PATH_TRACKER=0 \
APEX_ENABLE_RECOGNITION_TOUR_PLANNER=1 \
APEX_ENABLE_RECOGNITION_TOUR_TRACKER=1 \
APEX_ENABLE_CMDVEL_ACTUATION_BRIDGE=1 \
APEX_ENABLE_KINEMATICS=0 \
APEX_BRIDGE_MIN_EFFECTIVE_SPEED_PCT="${BRIDGE_MIN_SPEED_PCT}" \
APEX_BRIDGE_MAX_SPEED_PCT="${BRIDGE_MAX_SPEED_PCT}" \
APEX_BRIDGE_ACTIVE_BRAKE_ON_ZERO=1 \
APEX_SKIP_BUILD="${APEX_SKIP_BUILD:-1}" \
APEX_NANO_PREFLIGHT="${APEX_NANO_PREFLIGHT:-1}" \
./tools/capture/apex_raw_capture_up.sh

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "[APEX][ERROR] ${CONTAINER_NAME} is not running after startup" >&2
  exit 1
fi

docker exec "${CONTAINER_NAME}" /bin/bash -lc \
  "source '${ROS_SETUP_SCRIPT}' && python3 /work/ros2_ws/scripts/capture/record_recognition_tour.py \
    --path-topic /apex/planning/recognition_tour_local_path \
    --route-topic /apex/planning/recognition_tour_route \
    --planner-status-topic /apex/planning/recognition_tour_status \
    --tracker-status-topic /apex/tracking/recognition_tour_status \
    --bridge-status-topic /apex/vehicle/drive_bridge_status \
    --odom-topic /apex/odometry/imu_lidar_fused \
    --scan-topic /lidar/scan_localization \
    --lidar-offset-x-m 0.18 \
    --lidar-offset-y-m 0.0 \
    --output-dir '${CONTAINER_RUN_DIR}' \
    --timeout-s '${RUN_TIMEOUT_S}'" \
  > "${HOST_RUN_DIR}/recognition_tour_record.log" 2>&1 &
RECORDER_PID=$!

sleep 1.0
if ! docker exec "${CONTAINER_NAME}" /bin/bash -lc \
  "source '${ROS_SETUP_SCRIPT}' && ros2 topic pub --once '${ARM_TOPIC}' std_msgs/msg/Bool '{data: true}'"
then
  kill "${RECORDER_PID}" 2>/dev/null || true
  wait "${RECORDER_PID}" 2>/dev/null || true
  echo "[APEX][ERROR] Failed to arm recognition tour tracker on ${ARM_TOPIC}" >&2
  docker logs --tail 200 "${CONTAINER_NAME}" > "${HOST_RUN_DIR}/docker_tail.log" 2>&1 || true
  exit 1
fi

if ! wait "${RECORDER_PID}"; then
  echo "[APEX][ERROR] Recognition tour recorder failed. Log: ${HOST_RUN_DIR}/recognition_tour_record.log" >&2
  tail -n 120 "${HOST_RUN_DIR}/recognition_tour_record.log" >&2 || true
  docker logs --tail 200 "${CONTAINER_NAME}" > "${HOST_RUN_DIR}/docker_tail.log" 2>&1 || true
  exit 1
fi

docker exec "${CONTAINER_NAME}" /bin/bash -lc \
  "python3 /work/ros2_ws/tools/analysis/plot_recognition_tour_run.py --run-dir '${CONTAINER_RUN_DIR}'" \
  >> "${HOST_RUN_DIR}/recognition_tour_record.log" 2>&1 || true

docker logs --tail 200 "${CONTAINER_NAME}" > "${HOST_RUN_DIR}/docker_tail.log" 2>&1 || true

echo "[APEX] Recognition tour capture ready: ${HOST_RUN_DIR}"
echo "[APEX] Files:"
echo "  ${HOST_RUN_DIR}/capture_meta.json"
echo "  ${HOST_RUN_DIR}/recognition_tour_record.log"
echo "  ${HOST_RUN_DIR}/docker_tail.log"
echo "  ${HOST_RUN_DIR}/lidar_points.csv"
echo "  ${HOST_RUN_DIR}/analysis_recognition_tour/recognition_tour_summary.json"
echo "  ${HOST_RUN_DIR}/analysis_recognition_tour/recognition_tour_status.log"
echo "  ${HOST_RUN_DIR}/analysis_recognition_tour/recognition_tour_local_path_history.jsonl"
echo "  ${HOST_RUN_DIR}/analysis_recognition_tour/recognition_tour_route.csv"
echo "  ${HOST_RUN_DIR}/analysis_recognition_tour/recognition_tour_route.json"
echo "  ${HOST_RUN_DIR}/analysis_recognition_tour/recognition_tour_trajectory.csv"
echo "  ${HOST_RUN_DIR}/analysis_recognition_tour/recognition_tour_overview.png"
echo "  ${HOST_RUN_DIR}/analysis_recognition_tour/drive_bridge_status.log"
