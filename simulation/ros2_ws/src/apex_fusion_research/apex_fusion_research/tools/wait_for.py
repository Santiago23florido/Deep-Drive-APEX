"""Block until a condition of the running experiment is met (for scripts).

    ros2 run apex_fusion_research wait_for ins-navigating --timeout 180
    ros2 run apex_fusion_research wait_for tour-end --timeout 300

* ``ins-navigating``: ``/apex/fusion/ins/status`` reports ``state == navigating``
  (the INS finished its static alignment; it is now safe to move the car).
* ``tour-end``: ``/apex/tracking/recognition_tour_status`` (latched) reports a
  terminal state (loop closed, timeout or abort).

Exit code 0 when the condition is met, 1 on timeout (wall-clock seconds).
"""

from __future__ import annotations

import argparse
import json
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)

CONDITIONS = {
    "ins-navigating": ("/apex/fusion/ins/status", 10, lambda p: p.get("state") == "navigating"),
    "tour-end": ("/apex/tracking/recognition_tour_status", LATCHED, lambda p: bool(p.get("terminal", False))),
}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("condition", choices=sorted(CONDITIONS))
    parser.add_argument("--timeout", type=float, default=300.0, help="wall-clock seconds")
    parser.add_argument("--topic", default=None, help="override the status topic (e.g. /apex/fusion/ins_good/status)")
    args = parser.parse_args(argv)

    topic, qos, predicate = CONDITIONS[args.condition]
    topic = args.topic or topic
    rclpy.init()
    node = Node(f"wait_for_{args.condition.replace('-', '_')}_{abs(hash(topic)) % 10000}")
    state = {"done": False, "payload": {}}

    def on_msg(msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        state["payload"] = payload
        if predicate(payload):
            state["done"] = True

    node.create_subscription(String, topic, on_msg, qos)
    deadline = time.monotonic() + args.timeout
    last_print = 0.0
    try:
        while not state["done"] and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
            if time.monotonic() - last_print > 10.0:
                last_print = time.monotonic()
                summary = {k: state["payload"].get(k) for k in ("state", "cause", "terminal") if k in state["payload"]}
                print(f"[wait_for] {args.condition} on {topic}: waiting ({summary or 'no message yet'})", flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    if state["done"]:
        summary = {k: state["payload"].get(k) for k in ("state", "cause", "terminal") if k in state["payload"]}
        print(f"[wait_for] {args.condition}: reached {summary}", flush=True)
        raise SystemExit(0)
    print(f"[wait_for] {args.condition}: timeout after {args.timeout:.0f} s", flush=True)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
