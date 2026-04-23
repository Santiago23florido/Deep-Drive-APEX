#!/usr/bin/env bash
set -euo pipefail

APEX_SIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_WS_ROOT="${APEX_SIM_ROOT}/ros2_ws"
WORLD="${SIM_WS_ROOT}/install/rc_sim_description/share/rc_sim_description/worlds/basic_track.world"
GUI_CONFIG="${HOME}/.gz/sim/8/gui.config"

if [ -f "/opt/ros/jazzy/setup.bash" ]; then
  # Load ROS 2 environment for gz and ROS integrations
  source "/opt/ros/jazzy/setup.bash"
fi
if [ -f "${SIM_WS_ROOT}/install/setup.bash" ]; then
  # Load workspace overlays if present
  source "${SIM_WS_ROOT}/install/setup.bash"
fi

export GZ_SIM_RESOURCE_PATH="${SIM_WS_ROOT}/install/rc_sim_description/share:${GZ_SIM_RESOURCE_PATH:-}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="${SIM_WS_ROOT}/install/rc_sim_description/lib:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
export GZ_GUI_RESOURCE_PATH="${SIM_WS_ROOT}/install/rc_sim_description/share:${GZ_GUI_RESOURCE_PATH:-}"

# Run with verbose logs and capture output
"gz" sim -r -v4 \
  --gui-config "${GUI_CONFIG}" \
  --render-engine-gui ogre2 \
  --render-engine-gui-api-backend opengl \
  "${WORLD}" 2>&1 | tee /tmp/gz_world.log

# Filter FSAA / anti-aliasing warnings
grep -iE "Ogre2RenderTarget|FSAA|anti-alias" /tmp/gz_world.log || true
