#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <bundle_dir_or_run_id>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INPUT_PATH="$1"

if [ -d "${INPUT_PATH}" ]; then
  BUNDLE_DIR="$(cd "${INPUT_PATH}" && pwd)"
elif [ -d "${APEX_ROOT}/debug_runs/${INPUT_PATH}" ]; then
  BUNDLE_DIR="$(cd "${APEX_ROOT}/debug_runs/${INPUT_PATH}" && pwd)"
else
  echo "[APEX][ERROR] Bundle not found: ${INPUT_PATH}" >&2
  exit 1
fi

cd "${APEX_ROOT}"

python3 ./tools/analyze_debug_run.py "${BUNDLE_DIR}"
if ! python3 ./tools/replay_nav_from_bag.py "${BUNDLE_DIR}"; then
  echo "[APEX][WARN] Replay from bag failed for ${BUNDLE_DIR}. Continuing with DIAG log analysis only." >&2
fi
python3 ./tools/explain_recon_run.py "${BUNDLE_DIR}"

echo "[APEX] Analysis complete"
echo "[APEX] Summary: ${BUNDLE_DIR}/analysis/summary.md"
echo "[APEX] Flags: ${BUNDLE_DIR}/analysis/flags.json"
echo "[APEX] Timeline: ${BUNDLE_DIR}/analysis/decision_timeline.csv"
echo "[APEX] Replay: ${BUNDLE_DIR}/analysis/replay_nav.csv"
echo "[APEX] Explainer: ${BUNDLE_DIR}/analysis/trajectory_explainer.md"
