#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PORT="${APEX_PC_GAMEPAD_BRIDGE_PORT:-8765}"
LISTEN_HOST="${APEX_PC_GAMEPAD_BRIDGE_HOST:-0.0.0.0}"

normalize_bool_override() {
  local raw_value="$1"
  local normalized
  normalized="$(printf '%s' "${raw_value}" | tr '[:upper:]' '[:lower:]')"
  case "${normalized}" in
    1|true|yes|on)
      printf 'true\n'
      ;;
    0|false|no|off)
      printf 'false\n'
      ;;
    *)
      printf '%s\n' "${raw_value}"
      ;;
  esac
}

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
export APEX_PC_ENABLE_SESSION_TOGGLE_INPUT="${APEX_PC_ENABLE_SESSION_TOGGLE_INPUT:-0}"
export APEX_PC_REQUIRE_SESSION_TOGGLE_TO_DRIVE="${APEX_PC_REQUIRE_SESSION_TOGGLE_TO_DRIVE:-0}"
export APEX_PC_PUBLISH_CMD_VEL="${APEX_PC_PUBLISH_CMD_VEL:-0}"

echo "[APEX][pc-bridge] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "[APEX][pc-bridge] listen=${LISTEN_HOST}:${PORT}"
if [[ "$(normalize_bool_override "${APEX_PC_PUBLISH_CMD_VEL}")" == "true" ]]; then
  echo "[APEX][pc-bridge] publishing /apex/cmd_vel_track directly"
else
  echo "[APEX][pc-bridge] direct /apex/cmd_vel_track publishing disabled; session manager relays manual commands"
fi
echo "[APEX][pc-bridge] publishing /apex/sim/manual_control/status"
echo "[APEX][pc-bridge] publishing /apex/manual_control/status"
echo "[APEX][pc-bridge] publishing /apex/manual_control/session_toggle"
echo "[APEX][pc-bridge] enable_session_toggle_input=${APEX_PC_ENABLE_SESSION_TOGGLE_INPUT}"
echo "[APEX][pc-bridge] require_session_toggle_to_drive=${APEX_PC_REQUIRE_SESSION_TOGGLE_TO_DRIVE}"
echo "[APEX][pc-bridge] publish_cmd_vel=${APEX_PC_PUBLISH_CMD_VEL}"

exec ros2 run rc_sim_description apex_windows_gamepad_bridge_node.py --ros-args \
  -p cmd_vel_topic:=/apex/cmd_vel_track \
  -p status_topic:=/apex/sim/manual_control/status \
  -p manual_status_topic:=/apex/manual_control/status \
  -p session_toggle_topic:=/apex/manual_control/session_toggle \
  -p enable_session_toggle_input:="$(normalize_bool_override "${APEX_PC_ENABLE_SESSION_TOGGLE_INPUT}")" \
  -p require_session_toggle_to_drive:="$(normalize_bool_override "${APEX_PC_REQUIRE_SESSION_TOGGLE_TO_DRIVE}")" \
  -p publish_cmd_vel:="$(normalize_bool_override "${APEX_PC_PUBLISH_CMD_VEL}")" \
  -p listen_host:="${LISTEN_HOST}" \
  -p listen_port:="${PORT}"
