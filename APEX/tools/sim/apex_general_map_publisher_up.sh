#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 ]]; then
  cat >&2 <<'EOF'
Usage: apex_general_map_publisher_up.sh <map_yaml> <summary_json> <frame_id> <grid_topic> <visual_points_topic> <path_topic> <status_topic> [reload_on_change] [allow_missing_inputs]
EOF
  exit 1
fi

MAP_DIR="$(dirname "$1")"
SUMMARY_DIR="$(dirname "$2")"
mkdir -p "${MAP_DIR}" "${SUMMARY_DIR}"
MAP_YAML="$(cd "${MAP_DIR}" && pwd)/$(basename "$1")"
SUMMARY_JSON="$(cd "${SUMMARY_DIR}" && pwd)/$(basename "$2")"
FRAME_ID="$3"
GRID_TOPIC="$4"
VISUAL_POINTS_TOPIC="$5"
PATH_TOPIC="$6"
STATUS_TOPIC="$7"
RELOAD_ON_CHANGE="${8:-false}"
ALLOW_MISSING_INPUTS="${9:-false}"

ROS_SETUP_SCRIPT="${APEX_ROS_SETUP_SCRIPT:-/opt/ros/jazzy/setup.bash}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  if [[ "${PATH}" == "${VIRTUAL_ENV}/bin:"* ]]; then
    PATH="${PATH#${VIRTUAL_ENV}/bin:}"
  fi
  unset VIRTUAL_ENV
fi
unset PYTHONHOME || true

set +u
source "${ROS_SETUP_SCRIPT}"
if [[ -f "${REPO_ROOT}/install/setup.bash" ]]; then
  source "${REPO_ROOT}/install/setup.bash"
fi
set -u

exec ros2 run rc_sim_description apex_general_track_map_publisher.py --ros-args \
  -p map_yaml:="${MAP_YAML}" \
  -p summary_json:="${SUMMARY_JSON}" \
  -p frame_id:="${FRAME_ID}" \
  -p grid_topic:="${GRID_TOPIC}" \
  -p visual_points_topic:="${VISUAL_POINTS_TOPIC}" \
  -p path_topic:="${PATH_TOPIC}" \
  -p status_topic:="${STATUS_TOPIC}" \
  -p reload_on_change:="${RELOAD_ON_CHANGE}" \
  -p allow_missing_inputs:="${ALLOW_MISSING_INPUTS}"
