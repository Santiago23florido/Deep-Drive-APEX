#!/usr/bin/env python3
"""Recompute APEX navigation decisions from a recorded debug bag."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
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
    from geometry_msgs.msg import PointStamped, Vector3Stamped
    from sensor_msgs.msg import LaserScan
except Exception as exc:  # pragma: no cover - environment dependent
    raise SystemExit(f"ROS 2 bag dependencies are required: {exc}") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
APEX_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(APEX_DIR / "ros2_ws" / "src" / "apex_telemetry"))

from apex_telemetry.recon_navigation import ReconNavigator, ReconStateEstimate  # noqa: E402


def _normalize_angle_deg(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


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
    if raw_dir.is_dir() and any(raw_dir.glob("*.mcap")):
        try:
            subprocess.run(
                ["ros2", "bag", "reindex", str(raw_dir)],
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception:
            pass
        if (raw_dir / "metadata.yaml").exists():
            return raw_dir
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
        gap_escape_heading_threshold_deg=float(params["gap_escape_heading_threshold_deg"]),
        gap_escape_release_distance_m=float(params["gap_escape_release_distance_m"]),
        gap_escape_weight=float(params["gap_escape_weight"]),
        corridor_balance_ratio_threshold=float(params["corridor_balance_ratio_threshold"]),
        corridor_front_min_clearance_m=float(params["corridor_front_min_clearance_m"]),
        corridor_side_min_clearance_m=float(params["corridor_side_min_clearance_m"]),
        corridor_front_turn_weight=float(params["corridor_front_turn_weight"]),
        corridor_override_margin_deg=float(params["corridor_override_margin_deg"]),
        corridor_min_heading_deg=float(params["corridor_min_heading_deg"]),
        corridor_wall_start_deg=int(params["corridor_wall_start_deg"]),
        corridor_wall_end_deg=int(params["corridor_wall_end_deg"]),
        corridor_wall_min_points=int(params["corridor_wall_min_points"]),
        wall_follow_target_distance_m=float(params["wall_follow_target_distance_m"]),
        wall_follow_gain_deg_per_m=float(params["wall_follow_gain_deg_per_m"]),
        wall_follow_limit_deg=float(params["wall_follow_limit_deg"]),
        wall_follow_activation_heading_deg=float(params["wall_follow_activation_heading_deg"]),
        wall_follow_release_balance_ratio=float(params["wall_follow_release_balance_ratio"]),
        wall_follow_min_cycles=int(params["wall_follow_min_cycles"]),
        wall_follow_max_clearance_m=float(params["wall_follow_max_clearance_m"]),
        wall_follow_front_turn_weight=float(params["wall_follow_front_turn_weight"]),
        startup_consensus_min_heading_deg=float(params["startup_consensus_min_heading_deg"]),
        startup_valid_cycles_required=int(params["startup_valid_cycles_required"]),
        startup_gap_lockout_cycles=int(params["startup_gap_lockout_cycles"]),
        startup_latch_cycles=int(params["startup_latch_cycles"]),
        ambiguity_probe_speed_pct=float(params["ambiguity_probe_speed_pct"]),
        turn_speed_reduction=float(params["turn_speed_reduction"]),
        min_turn_speed_factor=float(params["min_turn_speed_factor"]),
        vehicle_half_width_m=float(params["vehicle_half_width_m"]),
        vehicle_front_overhang_m=float(params["vehicle_front_overhang_m"]),
        vehicle_rear_overhang_m=float(params["vehicle_rear_overhang_m"]),
        trajectory_horizon_m=float(params.get("trajectory_horizon_m", 0.95)),
        trajectory_lookahead_min_m=float(params.get("trajectory_lookahead_min_m", 0.35)),
        trajectory_lookahead_max_m=float(params.get("trajectory_lookahead_max_m", 0.85)),
        trajectory_curvature_slew_per_cycle=float(
            params.get("trajectory_curvature_slew_per_cycle", 0.22)
        ),
        trajectory_track_memory_alpha=float(params.get("trajectory_track_memory_alpha", 0.55)),
        trajectory_exit_heading_threshold_deg=float(
            params.get("trajectory_exit_heading_threshold_deg", 2.0)
        ),
        trajectory_curve_speed_gain=float(params.get("trajectory_curve_speed_gain", 1.2)),
        trajectory_state_aux_weight=float(params.get("trajectory_state_aux_weight", 0.30)),
        trajectory_min_confidence=float(params.get("trajectory_min_confidence", 0.28)),
        trajectory_flip_hold_cycles=int(params.get("trajectory_flip_hold_cycles", 8)),
        trajectory_entry_heading_threshold_deg=float(
            params.get("trajectory_entry_heading_threshold_deg", 4.0)
        ),
        trajectory_min_radius_m=float(params.get("trajectory_min_radius_m", 1.35)),
        trajectory_curve_heading_limit_deg=float(
            params.get("trajectory_curve_heading_limit_deg", 18.0)
        ),
    )


def build_state_estimate(
    caches: dict[str, tuple[float, ...] | None],
    baseline_pose: tuple[float, float, float] | None,
) -> tuple[ReconStateEstimate, tuple[float, float, float] | None]:
    position = caches.get("/apex/kinematics/position")
    heading = caches.get("/apex/kinematics/heading")
    velocity = caches.get("/apex/kinematics/velocity")
    acceleration = caches.get("/apex/kinematics/acceleration")
    angular_velocity = caches.get("/apex/kinematics/angular_velocity")

    pose = None
    if position is not None and heading is not None:
        pose = (float(position[0]), float(position[1]), float(heading[2]))
        if baseline_pose is None:
            baseline_pose = pose

    distance_from_start_m = 0.0
    lateral_drift_m = 0.0
    yaw_change_deg = 0.0
    if pose is not None and baseline_pose is not None:
        dx = pose[0] - baseline_pose[0]
        dy = pose[1] - baseline_pose[1]
        distance_from_start_m = math.hypot(dx, dy)
        lateral_drift_m = (-math.sin(baseline_pose[2]) * dx) + (math.cos(baseline_pose[2]) * dy)
        yaw_change_deg = _normalize_angle_deg(math.degrees(pose[2] - baseline_pose[2]))

    speed_mps = 0.0
    accel_mps2 = 0.0
    yaw_rate_rps = 0.0
    slip_proxy = 0.0
    if velocity is not None:
        speed_mps = math.hypot(float(velocity[0]), float(velocity[1]))
    if acceleration is not None:
        accel_mps2 = math.hypot(float(acceleration[0]), float(acceleration[1]))
    if angular_velocity is not None:
        yaw_rate_rps = float(angular_velocity[2])
    if pose is not None and velocity is not None and speed_mps >= 0.05:
        velocity_heading_deg = math.degrees(math.atan2(float(velocity[1]), float(velocity[0])))
        slip_proxy = _normalize_angle_deg(velocity_heading_deg - math.degrees(pose[2]))

    return (
        ReconStateEstimate(
            x_m=float(pose[0]) if pose is not None else 0.0,
            y_m=float(pose[1]) if pose is not None else 0.0,
            yaw_deg=math.degrees(pose[2]) if pose is not None else 0.0,
            yaw_rate_rps=float(yaw_rate_rps),
            speed_mps=float(speed_mps),
            accel_mps2=float(accel_mps2),
            lateral_drift_m=float(lateral_drift_m),
            slip_proxy=float(slip_proxy),
            pose_source="replay" if pose is not None else "none",
            distance_from_phase_start_m=float(distance_from_start_m),
            yaw_change_deg=float(yaw_change_deg),
            valid=pose is not None or velocity is not None or angular_velocity is not None,
        ),
        baseline_pose,
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
    caches: dict[str, tuple[float, ...] | None] = {
        "/apex/kinematics/position": None,
        "/apex/kinematics/heading": None,
        "/apex/kinematics/velocity": None,
        "/apex/kinematics/acceleration": None,
        "/apex/kinematics/angular_velocity": None,
    }
    baseline_pose = None
    while reader.has_next():
        topic_name, data, timestamp = reader.read_next()
        if topic_name == "/apex/kinematics/position":
            msg = deserialize_message(data, PointStamped)
            caches[topic_name] = (float(msg.point.x), float(msg.point.y), float(msg.point.z))
            continue
        if topic_name in {
            "/apex/kinematics/heading",
            "/apex/kinematics/velocity",
            "/apex/kinematics/acceleration",
            "/apex/kinematics/angular_velocity",
        }:
            msg = deserialize_message(data, Vector3Stamped)
            caches[topic_name] = (
                float(msg.vector.x),
                float(msg.vector.y),
                float(msg.vector.z),
            )
            continue
        if topic_name != "/lidar/scan":
            continue
        msg = deserialize_message(data, LaserScan)
        state_estimate, baseline_pose = build_state_estimate(caches, baseline_pose)
        command = navigator.compute_command(
            np.asarray(msg.ranges, dtype=np.float32),
            state_estimate=state_estimate,
        )
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
                "trajectory_phase": command.trajectory_phase,
                "lookahead_x_m": command.lookahead_x_m,
                "lookahead_y_m": command.lookahead_y_m,
                "signed_curvature": command.signed_curvature,
                "radius_m": command.radius_m,
                "target_speed_pct": command.target_speed_pct,
                "track_confidence": command.track_confidence,
                "state_speed_mps": command.state_speed_mps,
                "state_yaw_rate": command.state_yaw_rate,
                "slip_proxy": command.slip_proxy,
                "trajectory_flip_blocked": command.trajectory_flip_blocked,
            }
        )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["timestamp_ns"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
