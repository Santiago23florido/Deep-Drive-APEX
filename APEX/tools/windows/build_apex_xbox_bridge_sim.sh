#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${SCRIPT_DIR}/xbox_bridge"
DIST_DIR="${SCRIPT_DIR}/dist"

mkdir -p "${DIST_DIR}"

npx -y @yao-pkg/pkg \
  "${APP_DIR}/src/apex_xbox_bridge.js" \
  --targets node20-win-x64 \
  --output "${DIST_DIR}/apex_xbox_bridge_sim.exe"

echo "[APEX] Windows sim bridge ready: ${DIST_DIR}/apex_xbox_bridge_sim.exe"
