#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REMOTE_TARGET="${1:-ensta@raspberrypi:/home/ensta/AiAtonomousRc/APEX/}"

shift || true

echo "[APEX] Syncing code-only payload to ${REMOTE_TARGET}"

rsync -av \
  --no-perms --no-owner --no-group \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.pyo' \
  --exclude '.pytest_cache/' \
  --exclude '.mypy_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '.apex_runtime/' \
  --exclude 'analysis/' \
  --exclude 'apex_corridor_chain/' \
  --exclude 'apex_curve_track/' \
  --exclude 'apex_forward_raw/' \
  --exclude 'apex_recognition_tour/' \
  --exclude 'apex_static_curve/' \
  --exclude 'ros2_ws/apex_curve_track/' \
  --exclude 'ros2_ws/apex_recognition_tour/' \
  --exclude 'ros2_ws/logs/' \
  --exclude 'docker_tail.log' \
  --exclude 'capture_meta.json' \
  --exclude 'recognition_tour_record.log' \
  --exclude 'lidar_points.csv' \
  "${APEX_ROOT}/" \
  "${REMOTE_TARGET}" \
  "$@"

echo "[APEX] Code-only sync complete."
