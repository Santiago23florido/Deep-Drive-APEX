#!/usr/bin/env bash
# PC entry point for the approved trial and opt-in forward hardware calibration.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: race_pi.sh [--host ensta@raspberrypi] [--profile approved|initialized|tracking] [--dry-run] status|preflight|run

status     Read current runtime state over SSH.
preflight  Start a fresh disarmed stack, check sensors/SLAM/control/power,
           collect diagnostic data, then stop. Never publishes arm=true.
run        Start the approved RAM-backed two-lap recognition + SLAM trial.
           This mode moves the vehicle after its onboard preflight passes.
--dry-run  Print the selected SSH invocation and remote script without connecting.
--profile  approved (default), initialized or tracking: deploy the explicit calibrated
           forward profile, verify hashes, then execute the same finite mission.

Requires the deployed APEX tree and odometry environment on the Pi, the ensta
account, noninteractive SSH and passwordless sudo. Runtime root is fixed to
/home/ensta/AiAtonomousRc/APEX, as in the approved acquisition helpers.
USAGE
}

target="ensta@raspberrypi"
preview=0
profile=approved
mode=""
while (($#)); do
  case "$1" in
    --host) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; target="$2"; shift 2 ;;
    --dry-run) preview=1; shift ;;
    --profile) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; profile="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    status|preflight|run)
      [[ -z "$mode" ]] || { usage >&2; exit 2; }; mode="$1"; shift ;;
    *) usage >&2; exit 2 ;;
  esac
done
[[ -n "$mode" && "$target" =~ ^[a-zA-Z0-9_.@:-]+$ && "$target" != -* ]] || {
  usage >&2; exit 2;
}
[[ "$profile" == approved || "$profile" == initialized || "$profile" == tracking ]] || { usage >&2; exit 2; }

remote_script=$(cat <<'REMOTE'
set -euo pipefail
mode="$1"
profile="$2"
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
    if [[ "$profile" != approved ]]; then
      cp tools/core/initialized_trial/run_oval.py data/diagnostics/20261022_open_segment/run_oval_initialized.py
      exec python3 data/diagnostics/20261022_open_segment/run_oval_initialized.py --diagnose-only --forward-profile "$profile"
    fi
    exec python3 data/diagnostics/20261022_open_segment/run_oval.py --diagnose-only
    ;;
  run)
    exec sudo -n python3 tools/core/run_ram_capture.py --root "$apex_root" --profile "$profile" --execute
    ;;
esac
REMOTE
)

ssh_args=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=3)
if ((preview)); then
  if [[ "$profile" != approved && "$mode" != status ]]; then
    "$(dirname "${BASH_SOURCE[0]}")/sync_initialized_to_pi.sh" --dry-run "$target"
  fi
  printf '%q ' ssh "${ssh_args[@]}" "$target" bash -s -- "$mode" "$profile"
  printf '\n%s\n' "$remote_script"
  exit 0
fi
if [[ "$profile" != approved && "$mode" != status ]]; then
  "$(dirname "${BASH_SOURCE[0]}")/sync_initialized_to_pi.sh" "$target"
fi
printf '[race-pc] %s on %s\n' "$mode" "$target"
ssh "${ssh_args[@]}" "$target" bash -s -- "$mode" "$profile" <<< "$remote_script"
