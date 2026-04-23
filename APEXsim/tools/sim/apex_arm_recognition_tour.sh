#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_SIM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SIM_WS_ROOT="${APEX_SIM_ROOT}/ros2_ws"
ROS_SETUP_SCRIPT="${APEX_ROS_SETUP_SCRIPT:-/opt/ros/jazzy/setup.bash}"
RATE_HZ="10"
DURATION_S="2.0"
WAIT_TIMEOUT_S="45.0"
REQUIRED_SUBSCRIBERS="2"

usage() {
  cat <<'EOF'
Usage: apex_arm_recognition_tour.sh [options]

Options:
  --rate-hz <hz>
  --duration-s <seconds>
  --wait-timeout-s <seconds>
  --required-subscribers <count>
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rate-hz)
      RATE_HZ="${2:-}"
      shift 2
      ;;
    --duration-s)
      DURATION_S="${2:-}"
      shift 2
      ;;
    --wait-timeout-s)
      WAIT_TIMEOUT_S="${2:-}"
      shift 2
      ;;
    --required-subscribers)
      REQUIRED_SUBSCRIBERS="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[APEX][ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

cd "${SIM_WS_ROOT}"
export APEX_SIM_ROOT
set +u
source "${ROS_SETUP_SCRIPT}"
source "${SIM_WS_ROOT}/install/setup.bash" 2>/dev/null || true
set -u

python3 - <<PY
import json
import math
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool
from std_msgs.msg import String

rate_hz = max(1.0, float("${RATE_HZ}"))
duration_s = max(0.1, float("${DURATION_S}"))
wait_timeout_s = max(1.0, float("${WAIT_TIMEOUT_S}"))
required_subscribers = max(1, int("${REQUIRED_SUBSCRIBERS}"))
iterations = max(1, int(math.ceil(rate_hz * duration_s)))

rclpy.init()
node = Node("apex_arm_recognition_tour_cli")
qos = QoSProfile(
    depth=5,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)
publisher = node.create_publisher(Bool, "/apex/tracking/arm", qos)
latched_qos = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
message = Bool()
message.data = True
sleep_s = 1.0 / rate_hz
planner_status = {}
tracker_status = {}
kinematics_status = {}
estimation_status = {}


def _parse_status(msg: String) -> dict:
    try:
        payload = json.loads(msg.data)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _planner_cb(msg: String) -> None:
    global planner_status
    planner_status = _parse_status(msg)


def _tracker_cb(msg: String) -> None:
    global tracker_status
    tracker_status = _parse_status(msg)


def _kinematics_cb(msg: String) -> None:
    global kinematics_status
    kinematics_status = _parse_status(msg)


def _estimation_cb(msg: String) -> None:
    global estimation_status
    estimation_status = _parse_status(msg)


node.create_subscription(
    String,
    "/apex/planning/recognition_tour_status",
    _planner_cb,
    latched_qos,
)
node.create_subscription(
    String,
    "/apex/tracking/recognition_tour_status",
    _tracker_cb,
    latched_qos,
)
node.create_subscription(
    String,
    "/apex/kinematics/status",
    _kinematics_cb,
    20,
)
node.create_subscription(
    String,
    "/apex/estimation/status",
    _estimation_cb,
    20,
)

deadline = time.monotonic() + wait_timeout_s
last_log_monotonic = 0.0

while time.monotonic() < deadline and publisher.get_subscription_count() < required_subscribers:
    rclpy.spin_once(node, timeout_sec=0.2)
    now = time.monotonic()
    if (now - last_log_monotonic) >= 1.0:
        print(
            f"[APEX][arm] waiting subscribers on /apex/tracking/arm: "
            f"{publisher.get_subscription_count()}/{required_subscribers}",
            file=sys.stderr,
        )
        last_log_monotonic = now

if publisher.get_subscription_count() < required_subscribers:
    print(
        f"[APEX][arm][ERROR] only {publisher.get_subscription_count()} subscriber(s) on "
        f"/apex/tracking/arm after {wait_timeout_s:.1f}s",
        file=sys.stderr,
    )
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(1)

deadline = time.monotonic() + wait_timeout_s
last_log_monotonic = 0.0
while time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.2)
    calibration_complete = bool(kinematics_status.get("calibration_complete", False))
    calibration_active = bool(kinematics_status.get("calibration_active", False))
    alignment_ready = bool(estimation_status.get("alignment_ready", False))
    estimation_state = str(estimation_status.get("state", ""))
    planner_state = str(planner_status.get("state", ""))
    system_ready = (
        calibration_complete
        and (not calibration_active)
        and alignment_ready
        and estimation_state == "tracking"
    )
    if system_ready and planner_state in {"waiting_arm", "tracking"}:
        break
    now = time.monotonic()
    if (now - last_log_monotonic) >= 1.0:
        print(
            "[APEX][arm] waiting ready state: "
            f"calibration_complete={calibration_complete} "
            f"calibration_active={calibration_active} "
            f"alignment_ready={alignment_ready} "
            f"estimation_state={estimation_state or 'unknown'} "
            f"planner_state={planner_state or 'unknown'}",
            file=sys.stderr,
        )
        last_log_monotonic = now
else:
    print(
        f"[APEX][arm][ERROR] system not ready after {wait_timeout_s:.1f}s; "
        f"kinematics={kinematics_status} estimation={estimation_status} planner={planner_status}",
        file=sys.stderr,
    )
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(1)

deadline = time.monotonic() + wait_timeout_s
published = 0
while time.monotonic() < deadline:
    publisher.publish(message)
    published += 1
    rclpy.spin_once(node, timeout_sec=0.1)
    planner_armed = bool(planner_status.get("armed", False))
    tracker_armed = bool(tracker_status.get("armed", False))
    tracker_state = str(tracker_status.get("state", ""))
    if planner_armed and (tracker_armed or tracker_state == "tracking"):
        print(
            f"[APEX][arm] armed confirmed: planner_armed={planner_armed} "
            f"tracker_state={tracker_state or 'unknown'} tracker_armed={tracker_armed}",
            file=sys.stderr,
        )
        break
    if published >= iterations and planner_armed:
        print(
            f"[APEX][arm] planner armed confirmed; tracker_state={tracker_state or 'unknown'}",
            file=sys.stderr,
        )
        break
    time.sleep(sleep_s)
else:
    print(
        f"[APEX][arm][ERROR] arm not confirmed after {wait_timeout_s:.1f}s; "
        f"planner={planner_status} tracker={tracker_status}",
        file=sys.stderr,
    )
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(1)

node.destroy_node()
rclpy.shutdown()
PY
