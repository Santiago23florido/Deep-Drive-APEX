#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTAINER_NAME="${APEX_CONTAINER_NAME:-apex_pipeline}"
ROS_SETUP_SCRIPT="${APEX_ROS_SETUP_SCRIPT:-/opt/ros/jazzy/setup.bash}"
REMOTE_OUTPUT_ROOT="${APEX_STATIC_CURVE_ROOT:-${APEX_ROOT}/ros2_ws/apex_static_curve}"
CONTAINER_OUTPUT_ROOT="${APEX_STATIC_CURVE_CONTAINER_ROOT:-/work/ros2_ws/apex_static_curve}"
CAPTURE_DURATION_S="${APEX_STATIC_CURVE_CAPTURE_DURATION_S:-4.0}"
RUN_ID="${APEX_STATIC_CURVE_RUN_ID:-curve_static}"

usage() {
  cat <<'EOF'
Usage: apex_static_curve_capture.sh [options]

Options:
  --run-id <id>
  --capture-duration-s <seconds>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --capture-duration-s)
      CAPTURE_DURATION_S="${2:-}"
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

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "[APEX][ERROR] ${CONTAINER_NAME} is not running. Start it with tools/core/apex_core_up.sh" >&2
  exit 1
fi

CAPTURE_NAME="${RUN_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
HOST_RUN_DIR="${REMOTE_OUTPUT_ROOT}/${CAPTURE_NAME}"
CONTAINER_RUN_DIR="${CONTAINER_OUTPUT_ROOT}/${CAPTURE_NAME}"
mkdir -p "${HOST_RUN_DIR}"

cat > "${HOST_RUN_DIR}/capture_meta.json" <<EOF
{
  "run_id": "${CAPTURE_NAME}",
  "capture_duration_s": ${CAPTURE_DURATION_S},
  "mode": "static_curve"
}
EOF

if ! docker exec "${CONTAINER_NAME}" /bin/bash -lc \
  "source '${ROS_SETUP_SCRIPT}' && python3 /work/ros2_ws/scripts/capture/record_static_lidar_capture.py \
    --points-output '${CONTAINER_RUN_DIR}/lidar_points.csv' \
    --snapshot-output '${CONTAINER_RUN_DIR}/lidar_snapshot.csv' \
    --duration-s '${CAPTURE_DURATION_S}'" \
  > "${HOST_RUN_DIR}/capture.log" 2>&1
then
  echo "[APEX][ERROR] Static curve capture failed. Log: ${HOST_RUN_DIR}/capture.log" >&2
  if [[ -f "${HOST_RUN_DIR}/capture.log" ]]; then
    tail -n 120 "${HOST_RUN_DIR}/capture.log" >&2 || true
  fi
  exit 1
fi

if [[ ! -s "${HOST_RUN_DIR}/lidar_points.csv" ]]; then
  echo "[APEX][ERROR] Static curve capture produced no LiDAR points. Log: ${HOST_RUN_DIR}/capture.log" >&2
  if [[ -f "${HOST_RUN_DIR}/capture.log" ]]; then
    tail -n 120 "${HOST_RUN_DIR}/capture.log" >&2 || true
  fi
  exit 1
fi

if [[ "$(wc -l < "${HOST_RUN_DIR}/lidar_points.csv")" -le 1 ]]; then
  echo "[APEX][ERROR] Static curve capture produced an empty LiDAR point table. Log: ${HOST_RUN_DIR}/capture.log" >&2
  if [[ -f "${HOST_RUN_DIR}/capture.log" ]]; then
    tail -n 120 "${HOST_RUN_DIR}/capture.log" >&2 || true
  fi
  exit 1
fi

if [[ ! -s "${HOST_RUN_DIR}/lidar_snapshot.csv" ]]; then
  echo "[APEX][ERROR] Static curve capture produced no LiDAR snapshot. Log: ${HOST_RUN_DIR}/capture.log" >&2
  if [[ -f "${HOST_RUN_DIR}/capture.log" ]]; then
    tail -n 120 "${HOST_RUN_DIR}/capture.log" >&2 || true
  fi
  exit 1
fi

if [[ "$(wc -l < "${HOST_RUN_DIR}/lidar_snapshot.csv")" -le 1 ]]; then
  echo "[APEX][ERROR] Static curve capture produced an empty LiDAR snapshot table. Log: ${HOST_RUN_DIR}/capture.log" >&2
  if [[ -f "${HOST_RUN_DIR}/capture.log" ]]; then
    tail -n 120 "${HOST_RUN_DIR}/capture.log" >&2 || true
  fi
  exit 1
fi

echo "[APEX] Static curve capture ready: ${HOST_RUN_DIR}"
echo "[APEX] Files:"
echo "  ${HOST_RUN_DIR}/lidar_points.csv"
echo "  ${HOST_RUN_DIR}/lidar_snapshot.csv"
echo "  ${HOST_RUN_DIR}/capture.log"
