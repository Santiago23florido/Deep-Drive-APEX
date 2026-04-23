#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${1:-ensta@raspberrypi}"
RUN_ID="${2:-latest}"
REMOTE_ROOT="${3:-/home/ensta/AiAtonomousRc/APEX/ros2_ws/apex_recognition_tour}"
LOCAL_ROOT="${4:-$(pwd)/APEX/data/apex_recognition_tour}"
ANALYSIS_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)/analysis/plot_recognition_tour_run.py"
FETCH_LIDAR_POINTS="${APEX_FETCH_LIDAR_POINTS:-0}"

if [[ "${RUN_ID}" == "latest" ]]; then
  RUN_ID="$(ssh "${REMOTE_HOST}" "ls -1dt '${REMOTE_ROOT}'/* 2>/dev/null | head -n 1 | xargs -r basename")"
fi

if [[ -z "${RUN_ID}" ]]; then
  echo "[APEX][ERROR] No recognition_tour capture found in ${REMOTE_HOST}:${REMOTE_ROOT}" >&2
  exit 1
fi

mkdir -p "${LOCAL_ROOT}"
echo "[APEX] Fetching recognition_tour capture ${RUN_ID}"
rsync -av \
  --exclude 'lidar_points.csv' \
  "${REMOTE_HOST}:${REMOTE_ROOT%/}/${RUN_ID}/" "${LOCAL_ROOT%/}/${RUN_ID}/"
echo "[APEX] Local capture: ${LOCAL_ROOT%/}/${RUN_ID}"

if [[ "${FETCH_LIDAR_POINTS}" == "1" ]]; then
  echo "[APEX] Fetching lidar_points.csv separately"
  rsync -av \
    "${REMOTE_HOST}:${REMOTE_ROOT%/}/${RUN_ID}/lidar_points.csv" \
    "${LOCAL_ROOT%/}/${RUN_ID}/lidar_points.csv"
else
  echo "[APEX] Skipping lidar_points.csv by default; set APEX_FETCH_LIDAR_POINTS=1 to fetch it"
fi

if [[ -f "${ANALYSIS_SCRIPT}" ]]; then
  echo "[APEX] Regenerating local recognition_tour analysis"
  python3 "${ANALYSIS_SCRIPT}" --run-dir "${LOCAL_ROOT%/}/${RUN_ID}"
  echo "[APEX] Local analysis files:"
  echo "  ${LOCAL_ROOT%/}/${RUN_ID}/analysis_recognition_tour/recognition_tour_overview.png"
  echo "  ${LOCAL_ROOT%/}/${RUN_ID}/analysis_recognition_tour/recognition_tour_diagnostics.png"
  echo "  ${LOCAL_ROOT%/}/${RUN_ID}/analysis_recognition_tour/recognition_tour_diagnostics.json"
fi
