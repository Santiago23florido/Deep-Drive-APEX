#!/usr/bin/env python3
"""Finite Pi-local arm publisher for an explicitly requested autonomous lap.

Run inside the ROS container, under an independent shell timeout. The driver
retains its own mission deadline, freshness guards and actuator watchdogs.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import time


def power_ok(path: Path) -> bool:
    try:
        if not 0 <= time.time() - path.stat().st_mtime <= 3:
            return False
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        row = rows[-1]
        temperature = float(row['temp_c'])
        return int(row['throttled'], 16) & 0xffff == 0 and math.isfinite(temperature) and temperature < 75
    except (OSError, IndexError, KeyError, TypeError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-seconds', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--power-file', type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error('max-seconds must be finite and positive')
    import rclpy
    from std_msgs.msg import Bool, String
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    rclpy.init()
    node = rclpy.create_node('apex_bounded_lap_arm')
    publisher = node.create_publisher(Bool, '/apex/tracking/arm', 1)
    args.output.mkdir(parents=True, exist_ok=True)
    status = {}
    bridge = {}
    reason = 'interrupted'
    started = time.monotonic()
    with (args.output / 'bridge.jsonl').open('w') as bridge_log, (args.output / 'status.jsonl').open('w') as status_log:
        def receive_status(message):
            nonlocal status
            status = json.loads(message.data)
            status_log.write(message.data + '\n')
            status_log.flush()

        def receive_bridge(message):
            nonlocal bridge
            bridge = json.loads(message.data)
            bridge_log.write(message.data + '\n')
            bridge_log.flush()

        # Observers must not back-pressure the actuator/status publishers.
        observer_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        node.create_subscription(String, '/apex/race_driver/status', receive_status, observer_qos)
        node.create_subscription(String, '/apex/vehicle/drive_bridge_status', receive_bridge, observer_qos)
        next_publish = next_power = next_report = started
        try:
            while rclpy.ok():
                now = time.monotonic()
                if status.get('phase') == 'done':
                    reason = 'driver_done'
                    break
                if bridge.get('cmd_gap_ms_max', 0) >= 100:
                    reason = 'actuator_command_gap'
                    break
                if now - started >= args.max_seconds:
                    reason = 'arm_deadline'
                    break
                if now >= next_power:
                    if not power_ok(args.power_file):
                        reason = 'host_power_fault_or_stale_log'
                        break
                    next_power = now + .5
                if now >= next_publish:
                    publisher.publish(Bool(data=True))
                    next_publish = now + .1
                if now >= next_report:
                    print(json.dumps({'elapsed_s': round(now-started, 1), 'phase': status.get('phase'),
                                      'mission': status.get('mission'), 'recover': status.get('recover'),
                                      'clearance_m': status.get('clearance_m')}), flush=True)
                    next_report = now + 5
                rclpy.spin_once(node, timeout_sec=.02)
        finally:
            for _ in range(3):
                if rclpy.ok():
                    publisher.publish(Bool(data=False))
                    rclpy.spin_once(node, timeout_sec=.1)
                time.sleep(.1)
            result = {'reason': reason, 'elapsed_s': time.monotonic()-started, 'last_status': status}
            (args.output / 'arm_result.json').write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(result), flush=True)
            node.destroy_node()
            rclpy.try_shutdown()
    return 0 if reason == 'driver_done' else 3


if __name__ == '__main__':
    raise SystemExit(main())
