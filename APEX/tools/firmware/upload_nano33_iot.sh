#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SKETCH_DIR="${APEX_NANO_SKETCH_DIR:-${REPO_ROOT}/arduino/wt61pc_uart_stream}"
PORT="/dev/ttyACM0"
FQBN="arduino:samd:nano_33_iot"
STOP_CORE=0

usage() {
  cat <<'EOF'
Usage:
  tools/upload_nano33_iot.sh [--port /dev/ttyACM0] [--fqbn arduino:samd:nano_33_iot] [--sketch-dir <dir>] [--stop-core]

Notes:
  - Requires arduino-cli installed on the Raspberry.
  - Requires the Arduino SAMD core.
  - Default sketch: arduino/wt61pc_uart_stream.
  - By default this script refuses to run if apex_pipeline is still using the serial port.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="${2:?missing port value}"
      shift 2
      ;;
    --fqbn)
      FQBN="${2:?missing fqbn value}"
      shift 2
      ;;
    --sketch-dir)
      SKETCH_DIR="${2:?missing sketch directory value}"
      shift 2
      ;;
    --stop-core)
      STOP_CORE=1
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

if ! command -v arduino-cli >/dev/null 2>&1; then
  cat >&2 <<'EOF'
[APEX][ERROR] arduino-cli is not installed.

Install example:
  sudo apt-get update
  sudo apt-get install -y arduino-cli
  arduino-cli core update-index
  arduino-cli core install arduino:samd
EOF
  exit 1
fi

if ! compgen -G "${SKETCH_DIR}"'/*.ino' >/dev/null; then
  echo "[APEX][ERROR] Sketch not found in: ${SKETCH_DIR}" >&2
  exit 1
fi

if docker ps --format '{{.Names}}' | grep -qx 'apex_pipeline'; then
  if [[ "${STOP_CORE}" == "1" ]]; then
    "${REPO_ROOT}/tools/core/apex_core_down.sh"
  else
    cat >&2 <<'EOF'
[APEX][ERROR] apex_pipeline is running and may be holding the Nano serial port.

Stop it first:
  ./tools/core/apex_core_down.sh

Or rerun this script with:
  --stop-core
EOF
    exit 1
  fi
fi

if [[ ! -e "${PORT}" ]]; then
  echo "[APEX][ERROR] Serial port not found: ${PORT}" >&2
  exit 1
fi

echo "[APEX] Compiling sketch ${SKETCH_DIR} for ${FQBN}"
arduino-cli compile --fqbn "${FQBN}" "${SKETCH_DIR}"

echo "[APEX] Uploading to ${PORT}"
arduino-cli upload -p "${PORT}" --fqbn "${FQBN}" "${SKETCH_DIR}"

cat <<EOF
[APEX] Upload complete.

Quick serial check:
  stty -F ${PORT} 115200 raw -echo -echoe -echok
  timeout 5s stdbuf -oL cat ${PORT} | head -n 10

Then restart the core:
  APEX_SKIP_BUILD=1 ./tools/apex_core_up.sh
EOF
