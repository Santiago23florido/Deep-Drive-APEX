#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${1:-ensta@raspberrypi}"
RUN_ID="${2:-latest}"
REMOTE_ROOT="${3:-/home/ensta/AiAtonomousRc/APEX/ros2_ws/debug_runs}"
LOCAL_ROOT="${4:-$(pwd)/debug_runs}"
SSH_SOCKET="${TMPDIR:-/tmp}/apex_fetch_ssh_$(id -u)_$(date +%s).sock"
SSH_BASE_OPTS=(-o ControlMaster=auto -o ControlPersist=60 -o ControlPath="${SSH_SOCKET}")

cleanup_ssh_socket() {
  ssh "${SSH_BASE_OPTS[@]}" -O exit "${REMOTE_HOST}" >/dev/null 2>&1 || true
  rm -f "${SSH_SOCKET}"
}

trap cleanup_ssh_socket EXIT

refresh_remote_status() {
  ssh "${SSH_BASE_OPTS[@]}" "${REMOTE_HOST}" "\
    if docker ps --format '{{.Names}}' | grep -qx apex_pipeline; then \
      docker exec apex_pipeline /bin/bash /work/ros2_ws/scripts/apex_recon_session.sh status >/dev/null 2>&1 || true; \
    fi"
}

wait_for_remote_bundle() {
  local run_id="$1"
  local waited=0
  while [ "${waited}" -lt 30 ]; do
    refresh_remote_status >/dev/null 2>&1 || true
    local state
    state="$(ssh "${SSH_BASE_OPTS[@]}" "${REMOTE_HOST}" "python3 - <<'PY'
import json
from pathlib import Path

bundle = Path('${REMOTE_ROOT%/}/${run_id}')
metadata = bundle / 'run_metadata.json'
if not metadata.exists():
    print('metadata_missing')
    raise SystemExit(0)
try:
    payload = json.loads(metadata.read_text(encoding='utf-8'))
except Exception:
    print('metadata_invalid')
    raise SystemExit(0)
print('ready' if payload.get('bundle_complete', False) else 'pending')
PY")"
    if [ "${state}" = "ready" ]; then
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "[APEX][WARN] Remote bundle did not report bundle_complete=true within timeout; fetching current contents anyway." >&2
}

if [ "${RUN_ID}" = "latest" ]; then
  refresh_remote_status >/dev/null 2>&1 || true
  RUN_ID="$(ssh "${SSH_BASE_OPTS[@]}" "${REMOTE_HOST}" "ls -1t '${REMOTE_ROOT}' | head -n 1")"
fi

if [ -z "${RUN_ID}" ]; then
  echo "No run_id resolved from ${REMOTE_HOST}:${REMOTE_ROOT}" >&2
  exit 1
fi

wait_for_remote_bundle "${RUN_ID}"

mkdir -p "${LOCAL_ROOT}"
echo "[APEX] Fetching ${RUN_ID} from ${REMOTE_HOST}:${REMOTE_ROOT}"
rsync -av -e "ssh ${SSH_BASE_OPTS[*]}" "${REMOTE_HOST}:${REMOTE_ROOT%/}/${RUN_ID}/" "${LOCAL_ROOT%/}/${RUN_ID}/"
echo "[APEX] Local bundle: ${LOCAL_ROOT%/}/${RUN_ID}"
