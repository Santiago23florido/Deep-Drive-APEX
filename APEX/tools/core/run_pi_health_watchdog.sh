#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_ROOT="${1:-${APEX_ROOT}/pi_watchdog}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_ROOT%/}/watchdog_${STAMP}"
mkdir -p "${OUT_DIR}"

SAMPLE_CSV="${OUT_DIR}/health_samples.csv"
KERNEL_LOG="${OUT_DIR}/kernel_follow.log"
META_TXT="${OUT_DIR}/meta.txt"

{
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "cwd=$(pwd)"
} > "${META_TXT}"

echo "utc_iso,uptime_s,load1,load5,load15,temp_c,throttled_raw" > "${SAMPLE_CSV}"

KERNEL_PID=""
if command -v journalctl >/dev/null 2>&1; then
  stdbuf -oL journalctl -kf --no-pager >> "${KERNEL_LOG}" 2>&1 &
  KERNEL_PID=$!
fi

cleanup() {
  if [[ -n "${KERNEL_PID}" ]]; then
    kill "${KERNEL_PID}" 2>/dev/null || true
    wait "${KERNEL_PID}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

get_temp_c() {
  if command -v vcgencmd >/dev/null 2>&1; then
    vcgencmd measure_temp 2>/dev/null | sed -n "s/^temp=\\([0-9.]*\\)'C$/\\1/p"
    return
  fi
  if [[ -r /sys/class/thermal/thermal_zone0/temp ]]; then
    python3 - <<'PY'
from pathlib import Path
raw = Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()
try:
    print(float(raw) / 1000.0)
except Exception:
    print("")
PY
    return
  fi
  printf '\n'
}

get_throttled_raw() {
  if command -v vcgencmd >/dev/null 2>&1; then
    vcgencmd get_throttled 2>/dev/null | sed -n 's/^throttled=//p'
  else
    printf '\n'
  fi
}

echo "[APEX] Pi health watchdog writing to ${OUT_DIR}"
echo "[APEX] Stop with Ctrl+C"

while true; do
  utc_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  uptime_s="$(cut -d' ' -f1 /proc/uptime 2>/dev/null || true)"
  load_values="$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || echo '  ')"
  load1="$(printf '%s' "${load_values}" | awk '{print $1}')"
  load5="$(printf '%s' "${load_values}" | awk '{print $2}')"
  load15="$(printf '%s' "${load_values}" | awk '{print $3}')"
  temp_c="$(get_temp_c)"
  throttled_raw="$(get_throttled_raw)"
  printf '%s,%s,%s,%s,%s,%s,%s\n' \
    "${utc_iso}" \
    "${uptime_s}" \
    "${load1}" \
    "${load5}" \
    "${load15}" \
    "${temp_c}" \
    "${throttled_raw}" >> "${SAMPLE_CSV}"
  sleep 1
done
