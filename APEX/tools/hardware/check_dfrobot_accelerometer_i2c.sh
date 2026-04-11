#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="${APEX_CONTAINER:-apex_pipeline}"
BUS_ARGS=()
if [[ -n "${APEX_I2C_BUS:-}" ]]; then
  BUS_ARGS=(--bus "${APEX_I2C_BUS}")
fi

TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then
  TTY_ARGS=(-it)
fi

if [[ -f /.dockerenv ]]; then
  exec python3 "${SCRIPT_DIR}/check_dfrobot_accelerometer_i2c.py" "${BUS_ARGS[@]}" "$@"
fi

if command -v docker >/dev/null 2>&1 && docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  exec docker exec "${TTY_ARGS[@]}" "${CONTAINER}" \
    python3 /work/repo/APEX/tools/hardware/check_dfrobot_accelerometer_i2c.py \
    "${BUS_ARGS[@]}" "$@"
fi

exec python3 "${SCRIPT_DIR}/check_dfrobot_accelerometer_i2c.py" "${BUS_ARGS[@]}" "$@"
