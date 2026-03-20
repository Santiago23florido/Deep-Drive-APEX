#!/usr/bin/env python3
"""Recompute APEX navigation decisions from a recorded debug bag."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

try:
    import yaml
except Exception as exc:  # pragma: no cover - environment dependent
    raise SystemExit(f"PyYAML is required to replay bags: {exc}") from exc

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import LaserScan
except Exception as exc:  # pragma: no cover - environment dependent
    raise SystemExit(f"ROS 2 bag dependencies are required: {exc}") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
APEX_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(APEX_DIR / "ros2_ws" / "src" / "apex_telemetry"))

from apex_telemetry.recon_navigation import ReconNavigator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", help="Path to one debug bundle directory")
    parser.add_argument(
        "--output",
        help="Optional output CSV path (default: <bundle>/analysis/replay_nav.csv)",
    )
    return parser.parse_args()


def resolve_bag_dir(bundle_dir: Path) -> Path:
    raw_dir = bundle_dir / "bag" / "raw_debug_run"
    if (raw_dir / "metadata.yaml").exists():
        return raw_dir
    raise FileNotFoundError(f"Could not find rosbag2 directory under {bundle_dir / 'bag'}")


def build_navigator(config_path: Path) -> ReconNavigator:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    params = config["apex_recon_mapping_node"]["ros__parameters"]
    return ReconNavigator(
        steering_limit_deg=float(params["steering_limit_deg"]),
        steering_gain=float(params["steering_gain"]),
        fov_half_angle_deg=float(params["fov_half_angle_deg"]),
        smoothing_window=int(params["smoothing_window"]),
        stop_distance_m=float(params["stop_distance_m"]),
        slow_distance_m=float(params["slow_distance_m"]),
        min_speed_pct=float(params["explore_min_speed_pct"]),
        max_speed_pct=float(params["explore_max_speed_pct"]),
        front_window_deg=int(params["front_window_deg"]),
        side_window_deg=int(params["side_window_deg"]),
        center_angle_penalty_per_deg=float(params["center_angle_penalty_per_deg"]),
        wall_centering_gain_deg_per_m=float(params["wall_centering_gain_deg_per_m"]),
        wall_centering_limit_deg=float(params["wall_centering_limit_deg"]),
        wall_centering_base_weight=float(params["wall_centering_base_weight"]),
        wall_avoid_distance_m=float(params["wall_avoid_distance_m"]),
        wall_avoid_gain_deg_per_m=float(params["wall_avoid_gain_deg_per_m"]),
        wall_avoid_limit_deg=float(params["wall_avoid_limit_deg"]),
        turn_speed_reduction=float(params["turn_speed_reduction"]),
        min_turn_speed_factor=float(params["min_turn_speed_factor"]),
        vehicle_half_width_m=float(params["vehicle_half_width_m"]),
        vehicle_front_overhang_m=float(params["vehicle_front_overhang_m"]),
        vehicle_rear_overhang_m=float(params["vehicle_rear_overhang_m"]),
    )


def main() -> None:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else bundle_dir / "analysis" / "replay_nav.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bag_dir = resolve_bag_dir(bundle_dir)
    navigator = build_navigator(bundle_dir / "config" / "apex_params.yaml")

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    rows = []
    while reader.has_next():
        topic_name, data, timestamp = reader.read_next()
        if topic_name != "/lidar/scan":
            continue
        msg = deserialize_message(data, LaserScan)
        command = navigator.compute_command(np.asarray(msg.ranges, dtype=np.float32))
        rows.append(
            {
                "timestamp_ns": timestamp,
                "speed_pct": command.speed_pct,
                "steering_pre_servo_deg": command.steering_pre_servo_deg,
                "target_heading_deg": command.target_heading_deg,
                "gap_heading_deg": command.gap_heading_deg,
                "centering_heading_deg": command.centering_heading_deg,
                "avoidance_heading_deg": command.avoidance_heading_deg,
                "centering_weight": command.centering_weight,
                "front_clearance_m": command.front_clearance_m,
                "left_clearance_m": command.left_clearance_m,
                "right_clearance_m": command.right_clearance_m,
                "left_min_m": command.left_min_m,
                "right_min_m": command.right_min_m,
                "left_right_delta_m": command.left_right_delta_m,
                "active_heading_source": command.active_heading_source,
            }
        )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["timestamp_ns"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
