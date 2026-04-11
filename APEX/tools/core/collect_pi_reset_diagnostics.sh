#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_ROOT="${1:-${APEX_ROOT}/pi_reset_diagnostics}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_ROOT%/}/pi_reset_diag_${STAMP}"
mkdir -p "${OUT_DIR}"

run_capture() {
  local name="$1"
  shift
  local out_file="${OUT_DIR}/${name}.txt"
  {
    echo "\$ $*"
    echo
    "$@" || true
  } > "${out_file}" 2>&1
}

run_shell_capture() {
  local name="$1"
  local cmd="$2"
  local out_file="${OUT_DIR}/${name}.txt"
  {
    echo "\$ ${cmd}"
    echo
    bash -lc "${cmd}" || true
  } > "${out_file}" 2>&1
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

THROTTLED_RAW="unavailable"
if have_cmd vcgencmd; then
  THROTTLED_RAW="$(vcgencmd get_throttled 2>/dev/null || echo unavailable)"
fi

run_capture "date_utc" date -u
run_capture "uname" uname -a
run_shell_capture "uptime" "uptime; echo; who -b || true; echo; cat /proc/uptime"
run_shell_capture "last_reboots" "last -x | head -n 30 || true"
run_shell_capture "memory_disk" "free -h; echo; df -h"
run_shell_capture "thermal" "vcgencmd measure_temp 2>/dev/null || true; echo; cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || true"
run_shell_capture "throttled" "vcgencmd get_throttled 2>/dev/null || true"
run_shell_capture "usb" "lsusb 2>/dev/null || true; echo; lsusb -t 2>/dev/null || true"
run_shell_capture "network" "ip -br address 2>/dev/null || true; echo; iw dev 2>/dev/null || true"
run_shell_capture "journal_prev_boot" "journalctl -b -1 --no-pager 2>/dev/null || true"
run_shell_capture "journal_prev_boot_kernel" "journalctl -k -b -1 --no-pager 2>/dev/null || true"
run_shell_capture "journal_curr_boot" "journalctl -b --no-pager 2>/dev/null || true"
run_shell_capture "journal_curr_boot_kernel" "journalctl -k -b --no-pager 2>/dev/null || true"
run_shell_capture "dmesg_current" "dmesg -T 2>/dev/null || true"
run_shell_capture "grep_prev_boot" "journalctl -b -1 --no-pager 2>/dev/null | grep -Ei 'under.?voltage|throttl|watchdog|oom|out of memory|panic|ext4|mmc|sdhci|usb .*disconnect|reset high-speed usb|reset super|brcmfmac|wifi|thermal|temperature|power|reboot|shutdown|segfault|I/O error|journald' || true"
run_shell_capture "grep_curr_boot" "journalctl -b --no-pager 2>/dev/null | grep -Ei 'under.?voltage|throttl|watchdog|oom|out of memory|panic|ext4|mmc|sdhci|usb .*disconnect|reset high-speed usb|reset super|brcmfmac|wifi|thermal|temperature|power|reboot|shutdown|segfault|I/O error|journald' || true"

python3 - "${OUT_DIR}" "${THROTTLED_RAW}" <<'PY'
import json
import re
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
throttled_raw = sys.argv[2].strip()

def read_text(name: str) -> str:
    path = out_dir / name
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""

def throttled_flags(raw: str) -> dict:
    flags = {
        "under_voltage_now": False,
        "arm_freq_capped_now": False,
        "throttled_now": False,
        "soft_temp_now": False,
        "under_voltage_has_occurred": False,
        "arm_freq_capped_has_occurred": False,
        "throttled_has_occurred": False,
        "soft_temp_has_occurred": False,
    }
    match = re.search(r"0x([0-9a-fA-F]+)", raw)
    if not match:
        return flags
    value = int(match.group(1), 16)
    bit_map = {
        0: "under_voltage_now",
        1: "arm_freq_capped_now",
        2: "throttled_now",
        3: "soft_temp_now",
        16: "under_voltage_has_occurred",
        17: "arm_freq_capped_has_occurred",
        18: "throttled_has_occurred",
        19: "soft_temp_has_occurred",
    }
    for bit, key in bit_map.items():
        flags[key] = bool(value & (1 << bit))
    return flags

prev_grep = read_text("grep_prev_boot.txt")
curr_grep = read_text("grep_curr_boot.txt")
last_reboots = read_text("last_reboots.txt")

flags = throttled_flags(throttled_raw)

evidence = {
    "power_or_brownout": [],
    "storage_or_filesystem": [],
    "usb_or_peripheral": [],
    "kernel_or_watchdog": [],
    "thermal": [],
}

combined = "\n".join([prev_grep, curr_grep])

if flags["under_voltage_now"] or flags["under_voltage_has_occurred"]:
    evidence["power_or_brownout"].append(f"vcgencmd throttled flags: {flags}")
if re.search(r"under.?voltage|throttl|power", combined, re.I):
    evidence["power_or_brownout"].append("journal contains power/undervoltage/throttle markers")
if re.search(r"mmc|sdhci|EXT4|I/O error|Buffer I/O error", combined, re.I):
    evidence["storage_or_filesystem"].append("journal contains mmc/ext4/I/O error markers")
if re.search(r"usb .*disconnect|reset high-speed usb|reset super|ttyACM|ttyUSB", combined, re.I):
    evidence["usb_or_peripheral"].append("journal contains USB disconnect/reset markers")
if re.search(r"watchdog|panic|Oops:|segfault|Out of memory|oom-killer", combined, re.I):
    evidence["kernel_or_watchdog"].append("journal contains watchdog/panic/OOM markers")
if flags["soft_temp_now"] or flags["soft_temp_has_occurred"] or re.search(r"thermal|temperature", combined, re.I):
    evidence["thermal"].append("vcgencmd or journal contains thermal markers")

scores = {
    key: len(value) for key, value in evidence.items()
}
ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
primary = ranking[0][0] if ranking and ranking[0][1] > 0 else "unknown"

summary = {
    "throttled_raw": throttled_raw,
    "throttled_flags": flags,
    "primary_suspect": primary,
    "scores": scores,
    "evidence": evidence,
    "has_previous_boot_journal": bool(prev_grep.strip() or read_text("journal_prev_boot.txt").strip()),
    "last_reboots_excerpt": last_reboots.splitlines()[:12],
}

(out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

lines = [
    f"Primary suspect: {primary}",
    f"throttled_raw: {throttled_raw}",
    "",
    "Scores:",
]
for key, value in ranking:
    lines.append(f"- {key}: {value}")
lines.append("")
lines.append("Evidence:")
for key, items in evidence.items():
    if items:
        lines.append(f"- {key}:")
        for item in items:
            lines.append(f"  {item}")
if not any(evidence.values()):
    lines.append("- none conclusive; inspect journal_prev_boot.txt and dmesg_current.txt manually")

(out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo "[APEX] Raspberry reset diagnostics written to ${OUT_DIR}"
echo "[APEX] Files:"
echo "  ${OUT_DIR}/summary.txt"
echo "  ${OUT_DIR}/summary.json"
echo "  ${OUT_DIR}/last_reboots.txt"
echo "  ${OUT_DIR}/throttled.txt"
echo "  ${OUT_DIR}/journal_prev_boot.txt"
echo "  ${OUT_DIR}/journal_prev_boot_kernel.txt"
echo "  ${OUT_DIR}/grep_prev_boot.txt"
