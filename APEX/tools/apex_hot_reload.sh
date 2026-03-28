#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REMOTE_HOST="${APEX_REMOTE_HOST:-ensta@raspberrypi}"
REMOTE_ROOT="${APEX_REMOTE_ROOT:-~/AiAtonomousRc/APEX}"
RUN_ID="${1:-hot_reload}"
MODE="${2:-nav_dryrun}"
shift $(( $# >= 2 ? 2 : $# ))
EXTRA_ARGS=("$@")

RSYNC_EXCLUDES=(
  --exclude ".git"
  --exclude "__pycache__"
  --exclude ".venv*"
  --exclude "debug_runs"
  --exclude "logs"
  --exclude "ros2_ws/debug_runs"
  --exclude "ros2_ws/logs"
)

cd "${APEX_ROOT}"
rsync -avz --delete "${RSYNC_EXCLUDES[@]}" "${APEX_ROOT}/" "${REMOTE_HOST}:${REMOTE_ROOT}/"

REMOTE_ARGS=()
for arg in "${EXTRA_ARGS[@]}"; do
  REMOTE_ARGS+=("$(printf '%q' "${arg}")")
done

REMOTE_CMD="cd ${REMOTE_ROOT} && APEX_SKIP_BUILD=1 ./tools/apex_core_up.sh >/dev/null && ./tools/apex_recon_start.sh --restart --run-id $(printf '%q' "${RUN_ID}") --mode $(printf '%q' "${MODE}")"
if [ "${#REMOTE_ARGS[@]}" -gt 0 ]; then
  REMOTE_CMD+=" ${REMOTE_ARGS[*]}"
fi

ssh "${REMOTE_HOST}" "${REMOTE_CMD}"
echo "[APEX] Hot reload complete on ${REMOTE_HOST} (${RUN_ID}, ${MODE})"
