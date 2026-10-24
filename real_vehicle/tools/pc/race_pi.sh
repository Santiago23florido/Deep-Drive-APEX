#!/usr/bin/env bash
# PC entry point for the exact approved Pi helpers; no controller overrides.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: race_pi.sh [--host ensta@raspberrypi] [--dry-run] status|preflight|run

status     Read current runtime state over SSH.
preflight  Start a fresh disarmed stack, check sensors/SLAM/control/power,
           collect diagnostic data, then stop. Never publishes arm=true.
run        Start the approved RAM-backed two-lap recognition + SLAM trial.
           This mode moves the vehicle after its onboard preflight passes.
--dry-run  Print the selected SSH invocation and remote script without connecting.

Requires the deployed APEX tree and odometry environment on the Pi, the ensta
account, noninteractive SSH and passwordless sudo. Runtime root is fixed to
/home/ensta/AiAtonomousRc/APEX, as in the approved acquisition helpers.
USAGE
}

target="ensta@raspberrypi"
preview=0
mode=""
while (($#)); do
  case "$1" in
    --host) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; target="$2"; shift 2 ;;
    --dry-run) preview=1; shift ;;
    -h|--help) usage; exit 0 ;;
    status|preflight|run)
      [[ -z "$mode" ]] || { usage >&2; exit 2; }; mode="$1"; shift ;;
    *) usage >&2; exit 2 ;;
  esac
done
[[ -n "$mode" && "$target" =~ ^[a-zA-Z0-9_.@:-]+$ && "$target" != -* ]] || {
  usage >&2; exit 2;
}

remote_script=$(cat <<'REMOTE'
set -euo pipefail
mode="$1"
apex_root=/home/ensta/AiAtonomousRc/APEX
cd "$apex_root"
export PATH="$HOME/local/bin:$PATH"
if [[ "$mode" == status ]]; then
  exec ./tools/core/apex_race_up.sh status
fi
[[ "$(id -un)" == ensta ]] || { echo 'The recorded helpers require the ensta account.' >&2; exit 2; }
sudo -n true
containers=$(docker ps -a --format '{{.Names}}')
if [[ "$containers" == *apex_pipeline* ]]; then
  echo 'apex_pipeline already exists. Inspect status and stop the previous session before starting a fresh one.' >&2
  exit 2
fi
mkdir -p data/diagnostics/20261022_open_segment
cp tools/core/validated_trial_20261024_000432/*.py data/diagnostics/20261022_open_segment/
case "$mode" in
  preflight)
    exec python3 data/diagnostics/20261022_open_segment/run_oval.py --diagnose-only
    ;;
  run)
    exec sudo -n python3 tools/core/run_ram_capture.py --root "$apex_root" --execute
    ;;
esac
REMOTE
)

ssh_args=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=3)
if ((preview)); then
  printf '%q ' ssh "${ssh_args[@]}" "$target" bash -s -- "$mode"
  printf '\n%s\n' "$remote_script"
  exit 0
fi
printf '[race-pc] %s on %s\n' "$mode" "$target"
ssh "${ssh_args[@]}" "$target" bash -s -- "$mode" <<< "$remote_script"
