#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SERVICE_TEMPLATE="${APEX_ROOT}/systemd/apex-real-ready.service"
TARGET_SERVICE="/etc/systemd/system/apex-real-ready.service"
CURRENT_USER="${SUDO_USER:-${USER}}"
CURRENT_HOME="$(eval echo "~${CURRENT_USER}")"

if [[ ! -f "${SERVICE_TEMPLATE}" ]]; then
  echo "[APEX][ERROR] Missing service template: ${SERVICE_TEMPLATE}" >&2
  exit 1
fi

TMP_FILE="$(mktemp)"
trap 'rm -f "${TMP_FILE}"' EXIT

sed \
  -e "s|User=ensta|User=${CURRENT_USER}|g" \
  -e "s|Group=ensta|Group=${CURRENT_USER}|g" \
  -e "s|/home/ensta|${CURRENT_HOME}|g" \
  "${SERVICE_TEMPLATE}" > "${TMP_FILE}"

sudo install -m 0644 "${TMP_FILE}" "${TARGET_SERVICE}"
sudo systemctl daemon-reload
sudo systemctl enable --now apex-real-ready.service

echo "[APEX] Installed and started apex-real-ready.service for ${CURRENT_USER}"
