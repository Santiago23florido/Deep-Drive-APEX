#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PORT="${APEX_SERIAL_PORT:-/dev/ttyACM0}"
RUNTIME_DIR="${APEX_RUNTIME_DIR:-${APEX_ROOT}/.apex_runtime}"
PROFILE_ENV_FILE="${APEX_NANO_PROFILE_ENV_FILE:-${RUNTIME_DIR}/nano_serial_profile.env}"
PROFILE_ENV_BACKUP_FILE="${PROFILE_ENV_FILE}.last_good"
CHECK_TIMEOUT_S="${APEX_NANO_CHECK_TIMEOUT_S:-6}"
CONNECT_DTR_LOW_S="${APEX_NANO_CONNECT_DTR_LOW_S:-0.2}"
CONNECT_SETTLE_S="${APEX_NANO_CONNECT_SETTLE_S:-2.0}"
PASSIVE_CHECK_TIMEOUT_S="${APEX_NANO_PASSIVE_CHECK_TIMEOUT_S:-7}"
PASSIVE_HEAD_LINES="${APEX_NANO_PASSIVE_HEAD_LINES:-20}"
POST_FLASH_RECOVERY_S="${APEX_NANO_POST_FLASH_RECOVERY_S:-8.0}"
AUTOFLASH="${APEX_NANO_AUTOFLASH:-1}"
UPLOAD_SCRIPT="${APEX_ROOT}/tools/firmware/upload_nano33_iot.sh"

usage() {
  cat <<'EOF'
Usage:
  tools/ensure_nano33_stream.sh [--port /dev/ttyACM0] [--timeout-s 6] [--no-autoflash]

Behavior:
  - Checks whether the Nano is already streaming valid ax,ay,az,gx,gy,gz CSV.
  - If the stream is missing and autoflash is enabled, reflashes the Nano sketch.
  - Rechecks the stream after flashing.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="${2:?missing port value}"
      shift 2
      ;;
    --timeout-s)
      CHECK_TIMEOUT_S="${2:?missing timeout value}"
      shift 2
      ;;
    --no-autoflash)
      AUTOFLASH=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[APEX][ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

persist_runtime_profile() {
  local profile_name="$1"
  local toggle_dtr="$2"
  local dtr_low_s="$3"
  local settle_s="$4"
  local flush_input="$5"
  mkdir -p "${RUNTIME_DIR}"
  cat > "${PROFILE_ENV_FILE}" <<EOF
export APEX_SERIAL_CONNECT_PROFILE_NAME="${profile_name}"
export APEX_SERIAL_CONNECT_TOGGLE_DTR="${toggle_dtr}"
export APEX_SERIAL_CONNECT_DTR_LOW_S="${dtr_low_s}"
export APEX_SERIAL_CONNECT_SETTLE_S="${settle_s}"
export APEX_SERIAL_FLUSH_INPUT_ON_CONNECT="${flush_input}"
EOF
  cp -f "${PROFILE_ENV_FILE}" "${PROFILE_ENV_BACKUP_FILE}"
  echo "[APEX] Recorded Nano runtime profile '${profile_name}' in ${PROFILE_ENV_FILE}"
}

restore_last_good_profile() {
  if [[ -f "${PROFILE_ENV_BACKUP_FILE}" ]]; then
    cp -f "${PROFILE_ENV_BACKUP_FILE}" "${PROFILE_ENV_FILE}"
    echo "[APEX][WARN] Restored last known good Nano runtime profile from ${PROFILE_ENV_BACKUP_FILE}"
    return 0
  fi
  return 1
}

persist_runtime_attach_profile() {
  local probe_name="$1"
  # The probe may need to kick the ACM device with DTR, but once the Nano is
  # streaming we want the ROS runtime to attach as passively as possible to
  # avoid a second reset when the serial node opens the same port.
  persist_runtime_profile \
    "runtime_passive_after_${probe_name}" \
    "false" \
    "0.0" \
    "0.0" \
    "false"
}

check_stream_pyserial() {
  local toggle_dtr="${1}"
  local settle_s="${2}"
  local sample
  if [[ ! -e "${PORT}" ]]; then
    echo "[APEX][ERROR] Nano serial port not found: ${PORT}" >&2
    return 1
  fi

  if [[ "${toggle_dtr}" == "1" ]]; then
    echo "[APEX] Checking Nano stream via pyserial with DTR handshake on ${PORT}"
  else
    echo "[APEX] Checking Nano stream via pyserial without DTR handshake on ${PORT}"
  fi

  set +e
  sample="$(
    python3 - "${PORT}" "${CHECK_TIMEOUT_S}" "${CONNECT_DTR_LOW_S}" "${settle_s}" "${toggle_dtr}" <<'PY' \
      || true
import re
import serial
import sys
import time

port = sys.argv[1]
timeout_s = float(sys.argv[2])
dtr_low_s = float(sys.argv[3])
settle_s = float(sys.argv[4])
toggle_dtr = sys.argv[5] == "1"
num_re = re.compile(r"^[-+]?[0-9]*\.?[0-9]+$")

try:
    ser = serial.Serial(port, 115200, timeout=0.5)
except Exception:
    raise SystemExit(1)

try:
    if toggle_dtr:
        ser.setDTR(False)
        time.sleep(max(0.0, dtr_low_s))
        ser.reset_input_buffer()
        ser.setDTR(True)
    else:
        ser.reset_input_buffer()
    time.sleep(max(0.0, settle_s))
    deadline = time.time() + max(0.5, timeout_s)
    while time.time() < deadline:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line or line.startswith("INFO:") or line.startswith("ERROR:"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 6 and all(num_re.match(part) for part in parts):
            print(line)
            raise SystemExit(0)
    raise SystemExit(1)
finally:
    ser.close()
PY
  )"
  set -e

  [[ -n "${sample}" ]] || return 1
  if [[ "${toggle_dtr}" == "1" ]]; then
    persist_runtime_attach_profile "dtr_probe"
  else
    persist_runtime_attach_profile "pyserial_probe"
  fi
  echo "[APEX] Nano IMU stream detected on ${PORT}: ${sample}"
  return 0
}

check_stream_passive_cat() {
  local sample
  if [[ ! -e "${PORT}" ]]; then
    echo "[APEX][ERROR] Nano serial port not found: ${PORT}" >&2
    return 1
  fi

  echo "[APEX] Checking Nano stream via passive serial read on ${PORT}"

  set +e
  sample="$(
    stty -F "${PORT}" 115200 raw -echo -echoe -echok 2>/dev/null
    timeout -k 1s "${PASSIVE_CHECK_TIMEOUT_S}s" stdbuf -oL cat "${PORT}" 2>/dev/null \
      | awk '
          /^INFO:/ { next }
          /^ERROR:/ { next }
          /^[[:space:]]*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?,[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?,[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?,[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?,[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?,[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?[[:space:]]*$/ { print; exit }' \
      || true
  )"
  set -e

  [[ -n "${sample}" ]] || return 1
  persist_runtime_attach_profile "passive_cat_probe"
  echo "[APEX] Nano IMU stream detected on ${PORT}: ${sample}"
  return 0
}

check_stream_passive_head() {
  local sample
  if [[ ! -e "${PORT}" ]]; then
    echo "[APEX][ERROR] Nano serial port not found: ${PORT}" >&2
    return 1
  fi

  echo "[APEX] Checking Nano stream via passive cat|head read on ${PORT}"

  set +e
  sample="$(
    stty -F "${PORT}" 115200 raw -echo -echoe -echok 2>/dev/null
    timeout -k 1s "${PASSIVE_CHECK_TIMEOUT_S}s" stdbuf -oL cat "${PORT}" 2>/dev/null \
      | head -n "${PASSIVE_HEAD_LINES}" \
      | awk '
          /^INFO:/ { next }
          /^ERROR:/ { next }
          /^[[:space:]]*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?,[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?,[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?,[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?,[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?,[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?[[:space:]]*$/ { print; exit }' \
      || true
  )"
  set -e

  [[ -n "${sample}" ]] || return 1
  persist_runtime_attach_profile "passive_head_probe"
  echo "[APEX] Nano IMU stream detected on ${PORT}: ${sample}"
  return 0
}

dump_serial_preview() {
  if [[ ! -e "${PORT}" ]]; then
    return 0
  fi
  echo "[APEX] Nano serial preview on ${PORT} (best effort):"
  set +e
  stty -F "${PORT}" 115200 raw -echo -echoe -echok 2>/dev/null
  timeout 6s stdbuf -oL cat "${PORT}" 2>/dev/null | head -n 12 || true
  set -e
}

check_stream() {
  if check_stream_pyserial 1 "${CONNECT_SETTLE_S}"; then
    return 0
  fi
  if check_stream_pyserial 0 0.75; then
    return 0
  fi
  if check_stream_passive_head; then
    return 0
  fi
  if check_stream_passive_cat; then
    return 0
  fi

  echo "[APEX][WARN] No valid Nano IMU CSV sample detected on ${PORT} within ${CHECK_TIMEOUT_S}s" >&2
  return 1
}

if check_stream; then
  exit 0
fi

if [[ "${AUTOFLASH}" != "1" ]]; then
  echo "[APEX][ERROR] Nano stream missing and autoflash disabled" >&2
  exit 1
fi

if [[ ! -x "${UPLOAD_SCRIPT}" ]]; then
  echo "[APEX][ERROR] Upload script not found or not executable: ${UPLOAD_SCRIPT}" >&2
  exit 1
fi

echo "[APEX] Attempting Nano reflashing because the serial stream is missing"
"${UPLOAD_SCRIPT}" --port "${PORT}"
echo "[APEX] Waiting ${POST_FLASH_RECOVERY_S}s for Nano reboot after reflashing"
sleep "${POST_FLASH_RECOVERY_S}"

if check_stream; then
  exit 0
fi

dump_serial_preview
restore_last_good_profile || true

echo "[APEX][ERROR] Nano still not streaming after reflashing" >&2
exit 1
