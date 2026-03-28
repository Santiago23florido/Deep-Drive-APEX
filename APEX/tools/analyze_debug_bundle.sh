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

if [ ! -f "${BUNDLE_DIR}/run_metadata.json" ] && [ ! -f "${BUNDLE_DIR}/recon_diagnostic.log" ]; then
  echo "[APEX][ERROR] ${BUNDLE_DIR} is not a debug bundle directory." >&2
  echo "[APEX][ERROR] Expected run_metadata.json or recon_diagnostic.log inside that folder." >&2
  exit 1
fi

cd "${APEX_ROOT}"

python3 ./tools/analyze_debug_run.py "${BUNDLE_DIR}"
if ! python3 ./tools/replay_nav_from_bag.py "${BUNDLE_DIR}"; then
  echo "[APEX][WARN] Replay from bag failed for ${BUNDLE_DIR}. Continuing with DIAG log analysis only." >&2
fi
if ! python3 ./tools/explain_recon_run.py "${BUNDLE_DIR}"; then
  echo "[APEX][WARN] Explainer failed for ${BUNDLE_DIR}. Continuing with curve/static analysis outputs." >&2
fi

echo "[APEX] Analysis complete"
echo "[APEX] Summary: ${BUNDLE_DIR}/analysis/summary.md"
echo "[APEX] Flags: ${BUNDLE_DIR}/analysis/flags.json"
echo "[APEX] Timeline: ${BUNDLE_DIR}/analysis/decision_timeline.csv"
echo "[APEX] Curve static: ${BUNDLE_DIR}/analysis/curve_probe_static.json"
echo "[APEX] Curve static image: ${BUNDLE_DIR}/analysis/curve_probe_static.png"
echo "[APEX] Curve motion: ${BUNDLE_DIR}/analysis/curve_probe_motion.json"
echo "[APEX] Curve motion image: ${BUNDLE_DIR}/analysis/curve_probe_motion.png"
echo "[APEX] Curve compare: ${BUNDLE_DIR}/analysis/curve_probe_motion_compare.json"
echo "[APEX] Curve compare image: ${BUNDLE_DIR}/analysis/curve_probe_motion_compare.png"
echo "[APEX] Odom drift: ${BUNDLE_DIR}/analysis/odom_drift.json"
echo "[APEX] Odom drift image: ${BUNDLE_DIR}/analysis/odom_drift.png"
echo "[APEX] Fusion compare: ${BUNDLE_DIR}/analysis/sensor_fusion_compare.json"
echo "[APEX] Fusion compare image: ${BUNDLE_DIR}/analysis/sensor_fusion_compare.png"
echo "[APEX] Replay: ${BUNDLE_DIR}/analysis/replay_nav.csv"
echo "[APEX] Explainer: ${BUNDLE_DIR}/analysis/trajectory_explainer.md"
