#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

RVIZ_CONFIG="${1:-${APEX_ROOT}/rviz/apex_recognition_live.rviz}"

if [[ ! -f "${RVIZ_CONFIG}" ]]; then
  echo "[APEX][ERROR] Missing RViz config: ${RVIZ_CONFIG}" >&2
  exit 1
fi

set +u
source /opt/ros/jazzy/setup.bash
if [[ -f "${REPO_ROOT}/install/setup.bash" ]]; then
  source "${REPO_ROOT}/install/setup.bash"
fi
set -u

unset ROS_STATIC_PEERS FASTRTPS_DEFAULT_PROFILES_FILE ROS_LOCALHOST_ONLY
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-30}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-SUBNET}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export APEX_RVIZ_SOFTWARE_GL="${APEX_RVIZ_SOFTWARE_GL:-1}"

if [[ "${APEX_RVIZ_SOFTWARE_GL}" == "1" ]]; then
  export LIBGL_ALWAYS_SOFTWARE=1
  export QT_XCB_FORCE_SOFTWARE_OPENGL=1
fi

echo "[APEX][pc-rviz] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "[APEX][pc-rviz] config=${RVIZ_CONFIG}"

exec rviz2 -d "${RVIZ_CONFIG}"
