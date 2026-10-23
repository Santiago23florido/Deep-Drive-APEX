#!/usr/bin/env bash
# Bring up the race stack: learned odometry -> slam_toolbox -> race driver.
#
# The legacy autonomy (recognition tour, curve tracker, fixed map, session
# manager) is turned off in every mode, because three of those nodes publish
# `/apex/cmd_vel_track` and the car may have exactly one driver. The script
# asserts that afterwards.
#
# Modes, in the order the no-movement ladder uses them:
#
#   sensors-only  LiDAR + IMU only. For a capture and the timing audit (step C).
#   dry           The whole stack, but the driver publishes to
#                 /apex/race_driver/cmd_vel_dry. Nothing reaches the ESC, so the
#                 car can be on the floor with its drive battery unplugged while
#                 the estimator, the SLAM and the driver run for real (step D1).
#   replay        As `dry`, with the LiDAR and IMU nodes replaced by a recording
#                 of a past run. This is the only way the full state machine --
#                 explore, closing, planning, race -- runs on the Pi with the
#                 car switched off (step D2).
#   armed         The real thing: the driver drives the bridge. Requires an arm
#                 publisher; `arm` below is one. ANNOUNCE BEFORE USING.
#   arm           Publish /apex/tracking/arm at 10 Hz in the foreground.
#                 Ctrl-C disarms and the car stops within arm_timeout_s.
#   check         Run race_bringup_check.py against the running stack.
#   status        What is running and what the last run has written so far.
#                 Touches nothing, so it is the first thing to run after the
#                 car comes back on the network.
#   logs [N]      The container's last N lines, without the IMU heartbeat.
#   down          Stop everything.
#
# APEX_RACE_PERSIST=1 makes the container come back by itself after a reboot or
# a crash (Docker `unless-stopped`); `down` still stops it for good.
#
# Every run gets its own directory under ros2_ws/apex_race/<timestamp>, and the
# host samples the 5 V rail into power.csv beside the run's own artifacts. The
# stack is detached, so losing the SSH link does not stop a run or lose it:
# reconnect and use `status`.
#
#   ./tools/core/apex_race_up.sh dry
#   ./tools/core/apex_race_up.sh check --duration 180 --expect-phase explore
#   ./tools/core/apex_race_up.sh replay /work/data/replay/final_s13_sensors.npz
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${APEX_ROOT}"
export PATH="${HOME}/local/bin:${PATH}"

MODE="${1:-dry}"
shift || true
CONTAINER="${APEX_CONTAINER:-apex_pipeline}"
DRY_TOPIC="/apex/race_driver/cmd_vel_dry"
RUNS_HOST="${APEX_ROOT}/ros2_ws/apex_race"
LAST_RUN_FILE="${RUNS_HOST}/.last_run"

new_run_dir() {
  # Always restart the processes which own odometry/SLAM/adaptive state.
  # Send cleanup output to stderr: stdout is the new run id only.
  stop_power_log
  ./tools/core/apex_core_down.sh >&2
  sudo python3 "${SCRIPT_DIR}/race_fresh_start.py" --root "${APEX_ROOT}" \
    --container "${CONTAINER}" --uid "$(id -u)" --gid "$(id -g)"
}

start_power_log() {
  # What the kernel saw, sampled every second, next to the run it belongs to.
  # The car has already gone off the network mid-run and the only evidence was
  # in the journal; this puts it in the run directory, where the run is judged.
  # vcgencmd does not exist inside the container, so this runs on the host.
  local dir="$1" out="${1}/power.csv"
  [[ -f "${dir}/power.pid" ]] && return 0
  echo "utc_iso,lcrit_alarm,throttled,temp_c,freq_khz" > "${out}"
  setsid nohup bash -c "
    while true; do
      printf '%s,%s,%s,%s,%s\\n' \
        \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\" \
        \"\$(cat /sys/class/hwmon/hwmon2/in0_lcrit_alarm 2>/dev/null || echo ?)\" \
        \"\$(vcgencmd get_throttled 2>/dev/null | cut -d= -f2 || echo ?)\" \
        \"\$(vcgencmd measure_temp 2>/dev/null | tr -dc '0-9.' || echo ?)\" \
        \"\$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq 2>/dev/null || echo ?)\" >> '${out}'
      sleep 1
    done" >/dev/null 2>&1 &
  echo $! > "${dir}/power.pid"
}

stop_power_log() {
  local pid_file
  for pid_file in "${RUNS_HOST}"/*/power.pid; do
    [[ -f "${pid_file}" ]] || continue
    kill "$(cat "${pid_file}")" 2>/dev/null || true
    rm -f "${pid_file}"
  done
}

in_container() { docker exec "${CONTAINER}" bash -lc "source /opt/ros/jazzy/setup.bash >/dev/null 2>&1; $*"; }

require_topic() {
  local topic="$1" tries="${2:-20}"
  for ((i = 0; i < tries; i++)); do
    if in_container "timeout 3 ros2 topic echo ${topic} --once >/dev/null 2>&1"; then
      echo "[race]   ok   ${topic}"
      return 0
    fi
    sleep 1
  done
  echo "[race]   FAIL ${topic} never published" >&2
  return 1
}

assert_single_driver() {
  local n
  n="$(in_container "ros2 topic info /apex/cmd_vel_track" | sed -n 's/^Publisher count: //p' | tr -d ' ')"
  if [[ "${n}" != "1" ]]; then
    echo "[race] REFUSING: /apex/cmd_vel_track has ${n} publishers, must have exactly 1" >&2
    in_container "ros2 topic info -v /apex/cmd_vel_track" >&2 || true
    return 1
  fi
  echo "[race]   ok   /apex/cmd_vel_track has exactly one publisher"
}

start_stack() {
  # Survive a reboot and a dropped link, if asked. The car is on Wi-Fi to a
  # laptop and the link has gone away mid-run; the stack itself is detached and
  # keeps writing to disk, and with this it also comes back by itself after a
  # restart instead of waiting for someone to walk over.
  export APEX_RESTART_POLICY="${APEX_RESTART_POLICY:-$([[ "${APEX_RACE_PERSIST:-0}" == "1" ]] && echo unless-stopped || echo no)}"
  [[ "${APEX_RESTART_POLICY}" != "no" ]] && echo "[race] restart policy: ${APEX_RESTART_POLICY} (the session survives a reboot)"
  # Every legacy publisher of the command topic is off; so is the legacy
  # estimation chain, whose CPU the race stack needs.
  #
  # APEX_NANO_AUTOFLASH=0: the Nano now carries the WT61PC UART bridge, not the
  # old LSM6DS3 sketch. The preflight's recovery path reflashes with
  # upload_nano33_iot.sh, which would silently put the car back on the other
  # IMU -- and both sketches emit the same line format, so nothing downstream
  # would notice. Validation yes, automatic reflashing no.
  env APEX_ENABLE_RACE_STACK=1 \
  APEX_ENABLE_KINEMATICS="${APEX_ENABLE_KINEMATICS:-0}" \
  APEX_ENABLE_IMU_LIDAR_FUSION="${APEX_ENABLE_IMU_LIDAR_FUSION:-0}" \
  APEX_ENABLE_CURVE_ENTRY_PLANNER=0 \
  APEX_ENABLE_PATH_TRACKER=0 \
  APEX_ENABLE_RECOGNITION_TOUR_PLANNER=0 \
  APEX_ENABLE_RECOGNITION_TOUR_TRACKER=0 \
  APEX_ENABLE_RECOGNITION_SESSION_MANAGER=0 \
  APEX_ENABLE_MANUAL_CONTROL_BRIDGE=0 \
  APEX_ENABLE_OFFLINE_SUBMAP_REFINER=0 \
  APEX_ENABLE_CMDVEL_ACTUATION_BRIDGE="${APEX_ENABLE_CMDVEL_ACTUATION_BRIDGE:-1}" \
  APEX_STARTUP_COMPAT="${APEX_STARTUP_COMPAT:-safe}" \
  APEX_STAGGERED_STARTUP=1 \
  APEX_LIDAR_STARTUP_SETTLE_S="${APEX_LIDAR_STARTUP_SETTLE_S:-6.0}" \
  APEX_NODE_STARTUP_STAGGER_S="${APEX_NODE_STARTUP_STAGGER_S:-0.8}" \
  APEX_SKIP_BUILD="${APEX_SKIP_BUILD:-1}" \
  APEX_NANO_PREFLIGHT="${APEX_NANO_PREFLIGHT:-1}" \
  APEX_NANO_AUTOFLASH="${APEX_NANO_AUTOFLASH:-0}" \
  APEX_BRIDGE_PROFILE="${APEX_BRIDGE_PROFILE:-affine}" \
  FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}" \
  APEX_BRIDGE_MAX_SPEED_PCT="${APEX_BRIDGE_MAX_SPEED_PCT:-45.0}" \
  APEX_BRIDGE_MAX_REVERSE_SPEED_PCT="${APEX_BRIDGE_MAX_REVERSE_SPEED_PCT:-33.0}" \
  APEX_BRIDGE_MAX_STEERING_DEG="${APEX_BRIDGE_MAX_STEERING_DEG:-12.5}" \
  APEX_BRIDGE_LAUNCH_BOOST_SPEED_PCT="${APEX_BRIDGE_LAUNCH_BOOST_SPEED_PCT:-0.0}" \
  APEX_BRIDGE_LAUNCH_BOOST_HOLD_S="${APEX_BRIDGE_LAUNCH_BOOST_HOLD_S:-0.0}" \
  APEX_BRIDGE_ACTIVE_BRAKE_ON_ZERO="${APEX_BRIDGE_ACTIVE_BRAKE_ON_ZERO:-false}" \
  "$@" \
  ./run_apex.sh -d
}

case "${MODE}" in
  sensors-only)
    echo "[race] sensors only: LiDAR + IMU, no autonomy, no actuation"
    APEX_ENABLE_RACE_STACK=0 APEX_ENABLE_CMDVEL_ACTUATION_BRIDGE=0 \
      ./tools/capture/apex_raw_capture_up.sh
    ;;

  dry|replay)
    RUN_ID="$(new_run_dir)"
    start_power_log "${RUNS_HOST}/${RUN_ID}"
    echo "[race] run ${RUN_ID} -> ${RUNS_HOST}/${RUN_ID}"
    FIXTURE=""
    if [[ "${MODE}" == "replay" ]]; then
      FIXTURE="${1:-/work/data/replay/final_s13_sensors.npz}"
      echo "[race] REPLAY: the recorded sensors of a past run drive the stack; nothing on this car moves"
      # A replay is a finite job: it ends, the container exits, and a restart
      # policy would start it again over the same run directory. Measured:
      # RestartCount 1 and a second run on top of the first one's artifacts.
      if [[ "${APEX_RACE_PERSIST:-0}" == "1" ]]; then
        echo "[race] APEX_RACE_PERSIST is ignored in replay mode (the replay ends, and a restart would loop it)"
        APEX_RACE_PERSIST=0
      fi
    else
      echo "[race] DRY RUN: the stack runs on live sensors; the driver's commands go to ${DRY_TOPIC}"
    fi
    start_stack \
      APEX_RACE_RUN_DIR="/work/ros2_ws/apex_race/${RUN_ID}" \
      APEX_RACE_CMD_TOPIC="${DRY_TOPIC}" \
      APEX_ENABLE_CMDVEL_ACTUATION_BRIDGE=0 \
      APEX_RACE_REQUIRE_ARM="${APEX_RACE_REQUIRE_ARM:-false}" \
      APEX_RACE_LAPS="${APEX_RACE_LAPS:-3}" \
      ${FIXTURE:+APEX_REPLAY_FIXTURE="${FIXTURE}"} \
      ${FIXTURE:+APEX_ENABLE_LIDAR=0} \
      ${FIXTURE:+APEX_ENABLE_NANO=0}
    echo "[race] waiting for the stack to come up"
    sleep "${APEX_RACE_POSTCHECK_DELAY_S:-25}"
    require_topic /apex/imu/data_raw
    require_topic /lidar/scan_raw
    require_topic /apex/odometry/learned
    require_topic /apex/learned_odometry/scan_deskewed
    require_topic /apex/learned_odometry/status
    require_topic /map 40
    require_topic /apex/race_driver/status
    require_topic "${DRY_TOPIC}"
    echo "[race] up. Grade it with:  ./tools/core/apex_race_up.sh check --duration 180"
    ;;

  nudge)
    # Estimator + SLAM + the actuation bridge, and no race driver: the bounded
    # motion test is then the only publisher of /apex/cmd_vel_track, which is
    # the rule this stack enforces everywhere else too.
    RUN_ID="$(new_run_dir)"
    start_power_log "${RUNS_HOST}/${RUN_ID}"
    echo "[race] run ${RUN_ID} -> ${RUNS_HOST}/${RUN_ID}"
    echo "[race] MOTION TEST SETUP: the ESC will be driven by whatever publishes /apex/cmd_vel_track."
    echo "[race] Nothing moves until you run scripts/nudge.py."
    start_stack \
      APEX_RACE_RUN_DIR="/work/ros2_ws/apex_race/${RUN_ID}" \
      APEX_ENABLE_RACE_DRIVER=0 \
      APEX_ENABLE_CMDVEL_ACTUATION_BRIDGE=1
    sleep "${APEX_RACE_POSTCHECK_DELAY_S:-45}"
    require_topic /apex/imu/data_raw
    require_topic /lidar/scan_raw
    require_topic /apex/odometry/learned
    require_topic /apex/vehicle/drive_bridge_status
    echo "[race] ready. Move the car with:"
    echo "[race]   docker exec -it ${CONTAINER} bash -lc 'source /opt/ros/jazzy/setup.bash && python3 /work/ros2_ws/scripts/nudge.py --distance-m 0.20 --speed-mps 0.20'"
    ;;

  armed)
    echo "[race] ARMED: the driver will drive the ESC and the servo."
    echo "[race] The car must be on its stand or in a clear space, and someone must hold the arm."
    RUN_ID="$(new_run_dir)"
    start_power_log "${RUNS_HOST}/${RUN_ID}"
    echo "[race] run ${RUN_ID} -> ${RUNS_HOST}/${RUN_ID}"
    start_stack \
      APEX_RACE_RUN_DIR="/work/ros2_ws/apex_race/${RUN_ID}" \
      APEX_ENABLE_CMDVEL_ACTUATION_BRIDGE=1 \
      APEX_RACE_REQUIRE_ARM=true \
      APEX_RACE_LAPS="${APEX_RACE_LAPS:-1}"
    sleep "${APEX_RACE_POSTCHECK_DELAY_S:-25}"
    require_topic /apex/race_driver/status
    require_topic /apex/vehicle/drive_bridge_status
    assert_single_driver
    echo "[race] up and DISARMED. In another terminal:  ./tools/core/apex_race_up.sh arm"
    ;;

  arm)
    echo "[race] arming at 10 Hz. Ctrl-C disarms (the driver stops the car within arm_timeout_s)."
    trap 'echo; echo "[race] disarmed"; in_container "ros2 topic pub -1 /apex/tracking/arm std_msgs/msg/Bool \"{data: false}\"" >/dev/null 2>&1 || true' INT TERM EXIT
    in_container "ros2 topic pub -r 10 /apex/tracking/arm std_msgs/msg/Bool '{data: true}'"
    ;;

  check)
    in_container "python3 /work/ros2_ws/scripts/race_bringup_check.py --cmd-topic ${APEX_RACE_CMD_TOPIC:-${DRY_TOPIC}} $*"
    ;;

  status)
    # Safe to call after the link comes back: it touches nothing.
    RUN_ID="$(cat "${LAST_RUN_FILE}" 2>/dev/null || echo '')"
    echo "[race] container: $(docker ps --filter "name=${CONTAINER}" --format '{{.Status}}' 2>/dev/null | head -1)"
    echo "[race] last run : ${RUN_ID:-none}"
    if [[ -n "${RUN_ID}" && -d "${RUNS_HOST}/${RUN_ID}" ]]; then
      ls -la "${RUNS_HOST}/${RUN_ID}" | tail -n +2
      if [[ -f "${RUNS_HOST}/${RUN_ID}/power.csv" ]]; then
        echo "[race] power: $(( $(wc -l < "${RUNS_HOST}/${RUN_ID}/power.csv") - 1 )) samples, $(awk -F, 'NR>1 && $2 != "0" {n++} END {print n+0}' "${RUNS_HOST}/${RUN_ID}/power.csv") with the under-voltage alarm up"
      fi
    fi
    if docker ps --filter "name=${CONTAINER}" --format '{{.Names}}' | grep -q .; then
      in_container "timeout 4 ros2 topic echo /apex/race_driver/status --once" 2>/dev/null | tail -4 || true
    fi
    ;;

  logs)
    docker logs --tail "${1:-200}" "${CONTAINER}" 2>&1 | grep -v "IMU timing" | tail -n "${1:-80}"
    ;;

  down)
    stop_power_log
    ./tools/core/apex_core_down.sh
    ;;

  *)
    sed -n '2,30p' "${BASH_SOURCE[0]}" >&2
    exit 2
    ;;
esac
