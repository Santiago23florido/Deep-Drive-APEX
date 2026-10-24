#!/usr/bin/env bash
# Deploy only the initialized profile's explicit source closure, with hashes.
set -euo pipefail
preview=0
if [[ "${1:-}" == --dry-run ]]; then preview=1; shift; fi
target="${1:-ensta@raspberrypi}"
[[ "$target" =~ ^[a-zA-Z0-9_.@:-]+$ && "$target" != -* ]] || exit 2
apex_local_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
payload=(
  ros2_ws/src/apex_race/apex_race/core/adaptive.py
  ros2_ws/src/apex_race/apex_race/core/plant_id.py
  ros2_ws/src/apex_race/apex_race/core/race_config.py
  ros2_ws/src/apex_race/apex_race/core/race_driver.py
  ros2_ws/src/apex_race/config/race_speed_initialized.yaml
  ros2_ws/src/apex_race/config/vehicle_speed_initialized.yaml
  ros2_ws/scripts/start_apex_pipeline.sh
  docker/docker-compose.yml
  tools/core/run_ram_capture.py
  tools/core/initialized_trial/run_oval.py
)
if ((preview)); then
  printf 'Would deploy and SHA256-verify %d files to %s (only with the stack stopped):\n' "${#payload[@]}" "$target"
  printf '  %s\n' "${payload[@]}"
  exit 0
fi
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=8)
ssh "${ssh_opts[@]}" "$target" bash -s <<'REMOTE'
set -euo pipefail
[[ "$(id -un)" == ensta ]]
cd /home/ensta/AiAtonomousRc/APEX
containers=$(docker ps -a --format '{{.Names}}')
if [[ "$containers" == *apex_pipeline* ]]; then
  echo 'Refusing deployment while the vehicle container exists; inspect and stop the previous session first.' >&2
  exit 2
fi
python3 - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, 'tools/core')
from race_fresh_start import verify_neutral
verify_neutral(Path('/sys/class/pwm/pwmchip0/pwm0'))
PY
REMOTE
cd "$apex_local_root"
for file in "${payload[@]}"; do [[ -f "$file" ]] || { echo "Missing $file" >&2; exit 2; }; done
printf '%s\n' "${payload[@]}" | rsync -av --checksum --files-from=- \
  --no-perms --no-owner --no-group ./ "$target:/home/ensta/AiAtonomousRc/APEX/"
sha256sum "${payload[@]}" | ssh "${ssh_opts[@]}" "$target" \
  'cd /home/ensta/AiAtonomousRc/APEX && sha256sum -c -'
