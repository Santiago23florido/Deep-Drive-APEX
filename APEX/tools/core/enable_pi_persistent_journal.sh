#!/usr/bin/env bash
set -euo pipefail

CONF_DIR="/etc/systemd/journald.conf.d"
CONF_FILE="${CONF_DIR}/apex-persistent.conf"

if ! command -v sudo >/dev/null 2>&1; then
  echo "[APEX][ERROR] sudo is required to enable persistent journald" >&2
  exit 1
fi

sudo mkdir -p /var/log/journal
sudo mkdir -p "${CONF_DIR}"
sudo tee "${CONF_FILE}" >/dev/null <<'EOF'
[Journal]
Storage=persistent
Compress=yes
SystemMaxUse=200M
RuntimeMaxUse=64M
EOF

sudo systemctl restart systemd-journald
sleep 1

echo "[APEX] Persistent journald enabled via ${CONF_FILE}"
echo "[APEX] Current boots:"
journalctl --list-boots | tail -n 10 || true
