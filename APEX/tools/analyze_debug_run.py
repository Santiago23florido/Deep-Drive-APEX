#!/usr/bin/env python3
"""Analyze one APEX debug bundle and produce artifacts for offline review."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Iterable

import matplotlib.pyplot as plt

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped, Vector3Stamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu
    from std_msgs.msg import String
    from tf2_msgs.msg import TFMessage
except Exception:  # pragma: no cover - ROS bag deps are environment specific
    rosbag2_py = None
    deserialize_message = None
    PointStamped = None
    PoseWithCovarianceStamped = None
    Vector3Stamped = None
    Odometry = None
    Imu = None
    String = None
    TFMessage = None


TAG_PATTERN = re.compile(r"^(DIAG_[A-Z_]+)\s+(\{.*\})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", help="Path to one debug bundle directory")
    parser.add_argument(
        "--output-dir",
        help="Optional analysis output directory (default: <bundle>/analysis)",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sign(value: float | None, threshold: float = 1e-6) -> int:
    if value is None:
        return 0
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _normalize_angle_rad(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _normalize_angle_deg(angle_deg: float) -> float:
    return math.degrees(_normalize_angle_rad(math.radians(angle_deg)))


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * ((w * z) + (x * y)), 1.0 - 2.0 * ((y * y) + (z * z)))


def resolve_raw_bag_dir(bundle_dir: Path) -> Path | None:
    raw_dir = bundle_dir / "bag" / "raw_debug_run"
    if (raw_dir / "metadata.yaml").exists() or any(raw_dir.glob("*.mcap")):
        return raw_dir
    return None


def prefix_payload(tag: str, payload: dict) -> dict:
    if tag == "DIAG_SCAN":
        return payload
    if tag == "DIAG_NAV":
        return payload
    if tag == "DIAG_POSE":
        return {f"pose_{key}": value for key, value in payload.items()}
    if tag == "DIAG_STEER":
        return {f"steer_{key}": value for key, value in payload.items()}
    if tag == "DIAG_MOTOR":
        return {f"motor_{key}": value for key, value in payload.items()}
    return payload


def parse_diag_log(log_path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    config_records: list[dict] = []
    summary_records: list[dict] = []
    timeline_rows: list[dict] = []
    grouped: dict[tuple[str, str, int], dict] = {}

    if not log_path.exists():
        raise FileNotFoundError(f"Diagnostic log not found: {log_path}")

    for line_no, raw_line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), start=1):
        match = TAG_PATTERN.match(raw_line.strip())
        if not match:
            continue

        tag, payload_json = match.groups()
        payload = json.loads(payload_json)
        payload["_tag"] = tag
        payload["_line_no"] = line_no

        if tag == "DIAG_CONFIG":
            config_records.append(payload)
            continue
        if tag == "DIAG_SUMMARY":
            summary_records.append(payload)
            continue

        key = (
            str(payload.get("phase")),
            str(payload.get("step")),
            int(payload.get("cycle", 0)),
        )
        row = grouped.get(key)
        if row is None:
            row = {
                "_sequence": len(timeline_rows),
                "phase": key[0],
                "step": key[1],
                "cycle": key[2],
            }
            grouped[key] = row
            timeline_rows.append(row)
        row.update(prefix_payload(tag, payload))

    return config_records, summary_records, timeline_rows


def infer_expected_direction(run_id: str) -> str | None:
    lower = run_id.lower()
    if "right_open" in lower:
        return "turn_right"
    if "left_open" in lower:
        return "turn_left"
    if "single_wall_right" in lower or "wall_right" in lower:
        return "away_from_right_wall"
    if "single_wall_left" in lower or "wall_left" in lower:
        return "away_from_left_wall"
    return None


def infer_orientation_expectation(run_id: str) -> str | None:
    lower = run_id.lower()
    if "orientation_right" in lower or "right_wall" in lower or "wall_right" in lower:
        return "right_wall"
    if "orientation_left" in lower or "left_wall" in lower or "wall_left" in lower:
        return "left_wall"
    if "orientation_front" in lower or "front_wall" in lower or "wall_front" in lower:
        return "front_wall"
    return None


def majority_direction(values: Iterable[int]) -> int:
    counts = {1: 0, -1: 0}
    for value in values:
        if value in counts:
            counts[value] += 1
    if counts[1] == counts[-1]:
        return 0
    return 1 if counts[1] > counts[-1] else -1


def analyze_flags(rows: list[dict], config: dict, run_metadata: dict) -> dict:
    run_id = str(run_metadata.get("run_id", ""))
    expected_direction = infer_expected_direction(run_id)
    orientation_expectation = infer_orientation_expectation(run_id)
    wall_avoid_distance = safe_float(config.get("wall_avoid_distance_m")) or 0.45
    slow_distance = safe_float(config.get("slow_distance_m")) or 0.90
    curve_nav_modes = {
        "curve_capture",
        "curve_entry",
        "curve_follow",
        "curve_exit",
        "fullsoft_follow",
    }
    trajectory_curve_phases = {
        "curve_entry",
        "curve_follow",
        "curve_exit",
        "fullsoft_follow",
    }

    nav_rows = [row for row in rows if safe_float(row.get("target_heading_deg")) is not None]
    direction_signs = [
        sign(safe_float(row.get("steering_pre_servo_deg")), threshold=1.0)
        for row in nav_rows
    ]
    dominant_direction = majority_direction(direction_signs)

    logic_sign_mismatch = False
    if expected_direction == "turn_right":
        logic_sign_mismatch = dominant_direction > 0
    elif expected_direction == "turn_left":
        logic_sign_mismatch = dominant_direction < 0
    elif expected_direction == "away_from_right_wall":
        logic_sign_mismatch = dominant_direction < 0
    elif expected_direction == "away_from_left_wall":
        logic_sign_mismatch = dominant_direction > 0

    centering_sign_mismatch = False
    for row in nav_rows:
        delta = safe_float(row.get("left_right_delta_m"))
        heading = safe_float(row.get("centering_heading_deg"))
        if delta is None or heading is None:
            continue
        if abs(delta) < 0.05 or abs(heading) < 1.0:
            continue
        if sign(delta) != sign(heading):
            centering_sign_mismatch = True
            break

    lidar_heading_offset_suspect = False
    if orientation_expectation == "right_wall":
        right_is_closer = [
            row for row in nav_rows
            if (safe_float(row.get("right_min_m")) or 0.0) > 0.0
            and (safe_float(row.get("left_min_m")) or 0.0) > 0.0
            and safe_float(row.get("right_min_m")) < safe_float(row.get("left_min_m"))
        ]
        lidar_heading_offset_suspect = len(right_is_closer) == 0
    elif orientation_expectation == "left_wall":
        left_is_closer = [
            row for row in nav_rows
            if (safe_float(row.get("right_min_m")) or 0.0) > 0.0
            and (safe_float(row.get("left_min_m")) or 0.0) > 0.0
            and safe_float(row.get("left_min_m")) < safe_float(row.get("right_min_m"))
        ]
        lidar_heading_offset_suspect = len(left_is_closer) == 0
    elif orientation_expectation == "front_wall":
        front_is_closer = [
            row for row in nav_rows
            if (safe_float(row.get("front_clearance_m")) or 0.0) > 0.0
            and (safe_float(row.get("left_clearance_m")) or 0.0) > 0.0
            and (safe_float(row.get("right_clearance_m")) or 0.0) > 0.0
            and safe_float(row.get("front_clearance_m"))
            < min(
                safe_float(row.get("left_clearance_m")),
                safe_float(row.get("right_clearance_m")),
            )
        ]
        lidar_heading_offset_suspect = len(front_is_closer) == 0

    near_wall_rows = []
    wrong_direction_rows = []
    avoidance_missing_rows = []
    for row in nav_rows:
        left_min_m = safe_float(row.get("left_min_m")) or 0.0
        right_min_m = safe_float(row.get("right_min_m")) or 0.0
        front_clearance_m = safe_float(row.get("front_clearance_m")) or 0.0
        steering_pre_servo_deg = safe_float(row.get("steering_pre_servo_deg")) or 0.0
        desired_sign = sign(left_min_m - right_min_m, threshold=0.03)
        actual_sign = sign(steering_pre_servo_deg, threshold=1.0)
        min_side_m = min(
            value for value in (left_min_m, right_min_m) if value > 0.0
        ) if any(value > 0.0 for value in (left_min_m, right_min_m)) else 0.0

        if 0.0 < min_side_m < wall_avoid_distance:
            near_wall_rows.append(row)
            source = str(row.get("active_heading_source") or "")
            if source != "avoidance" and not source.startswith("fullsoft_"):
                avoidance_missing_rows.append(row)
        if desired_sign != 0 and actual_sign != 0 and desired_sign != actual_sign and front_clearance_m <= slow_distance:
            wrong_direction_rows.append(row)

    wall_avoidance_inactive = len(avoidance_missing_rows) >= 3
    front_stop_ok_but_direction_wrong = len(wrong_direction_rows) >= 3

    gate_open_but_zero_intent_rows = [
        row
        for row in nav_rows
        if row.get("curve_gate_open")
        and abs(safe_float(row.get("curve_intent_score")) or 0.0) <= 1e-6
        and str(row.get("nav_mode")) not in curve_nav_modes
    ]
    wrong_sign_during_capture_rows = []
    curve_canceled_by_near_wall_rows = []
    steering_below_curve_floor_rows = []
    straight_through_curve_rows = []
    for row in nav_rows:
        nav_mode = str(row.get("nav_mode"))
        gate_curve_sign = sign(safe_float(row.get("gate_curve_sign")))
        steering_deg = abs(safe_float(row.get("steering_deg")) or 0.0)
        steering_floor_deg = safe_float(row.get("curve_steering_floor_deg")) or 0.0
        target_sign = sign(safe_float(row.get("target_heading_deg")), threshold=0.5)
        speed_pct = safe_float(row.get("speed_pct")) or 0.0

        if row.get("curve_capture_active") and gate_curve_sign != 0 and target_sign != 0 and target_sign != gate_curve_sign:
            wrong_sign_during_capture_rows.append(row)
        if str(row.get("sign_veto_reason")) == "near_wall_limit_curve_veto":
            curve_canceled_by_near_wall_rows.append(row)
        if (
            nav_mode in curve_nav_modes
            and steering_floor_deg > 0.0
            and steering_deg + 1e-6 < steering_floor_deg
        ):
            steering_below_curve_floor_rows.append(row)
        if (
            row.get("curve_gate_open")
            and gate_curve_sign != 0
            and nav_mode not in curve_nav_modes
            and speed_pct > 0.0
        ):
            straight_through_curve_rows.append(row)

    trajectory_rows = [
        row
        for row in nav_rows
        if row.get("trajectory_phase") not in {None, "", "None"}
    ]
    trajectory_curve_rows = [
        row
        for row in trajectory_rows
        if row.get("trajectory_phase") in trajectory_curve_phases
    ]
    trajectory_follow_rows = [
        row
        for row in trajectory_rows
        if row.get("trajectory_phase") in {"curve_follow", "fullsoft_follow"}
    ]
    trajectory_slew_limit = (
        safe_float(config.get("trajectory_curvature_slew_per_cycle")) or 0.22
    )
    trajectory_min_confidence = safe_float(config.get("trajectory_min_confidence")) or 0.28

    mid_curve_flip = False
    previous_follow_sign = 0
    opposite_sign_streak = 0
    for row in trajectory_follow_rows:
        curvature_sign = sign(safe_float(row.get("signed_curvature")), threshold=1e-3)
        if curvature_sign == 0:
            continue
        if previous_follow_sign == 0:
            previous_follow_sign = curvature_sign
            continue
        if curvature_sign != previous_follow_sign:
            opposite_sign_streak += 1
            if opposite_sign_streak >= 2:
                mid_curve_flip = True
                break
        else:
            opposite_sign_streak = 0

    trajectory_slew_excess_rows = []
    previous_curvature = None
    for row in trajectory_rows:
        curvature = safe_float(row.get("signed_curvature"))
        if curvature is None:
            continue
        if (
            previous_curvature is not None
            and abs(curvature - previous_curvature) > (trajectory_slew_limit * 1.15)
        ):
            trajectory_slew_excess_rows.append(row)
        previous_curvature = curvature

    low_confidence_curve_rows = [
        row
        for row in trajectory_curve_rows
        if (safe_float(row.get("track_confidence")) or 0.0) < trajectory_min_confidence
    ]

    straight_align_rows = [
        row
        for row in trajectory_rows
        if row.get("trajectory_phase") == "straight_align"
        and safe_float(row.get("pose_lateral_drift_m")) is not None
    ]
    straight_convergence_failure = False
    if len(straight_align_rows) >= 6:
        sample_size = min(5, max(3, len(straight_align_rows) // 4))
        first_drift = mean(
            abs(safe_float(row.get("pose_lateral_drift_m")) or 0.0)
            for row in straight_align_rows[:sample_size]
        )
        last_drift = mean(
            abs(safe_float(row.get("pose_lateral_drift_m")) or 0.0)
            for row in straight_align_rows[-sample_size:]
        )
        straight_convergence_failure = first_drift >= 0.08 and last_drift > max(0.05, first_drift * 0.80)

    servo_sign_mismatch = False
    servo_phase_rows = [
        row for row in rows
        if str(row.get("phase")) == "steering_sign_check"
        and safe_float(row.get("pose_yaw_change_deg")) is not None
        and abs(safe_float(row.get("pose_yaw_change_deg")) or 0.0) >= 5.0
        and abs(safe_float(row.get("steer_pre_sign_deg")) or 0.0) >= 3.0
    ]
    if servo_phase_rows:
        mismatches = 0
        for row in servo_phase_rows:
            if sign(safe_float(row.get("pose_yaw_change_deg")), threshold=2.0) != sign(
                safe_float(row.get("steer_pre_sign_deg")),
                threshold=2.0,
            ):
                mismatches += 1
        servo_sign_mismatch = mismatches >= max(1, math.ceil(len(servo_phase_rows) * 0.6))

    return {
        "logic_sign_mismatch": logic_sign_mismatch,
        "servo_sign_mismatch": servo_sign_mismatch,
        "lidar_heading_offset_suspect": lidar_heading_offset_suspect,
        "centering_sign_mismatch": centering_sign_mismatch,
        "front_stop_ok_but_direction_wrong": front_stop_ok_but_direction_wrong,
        "wall_avoidance_inactive": wall_avoidance_inactive,
        "gate_open_but_zero_intent": len(gate_open_but_zero_intent_rows) >= 1,
        "wrong_sign_during_capture": len(wrong_sign_during_capture_rows) >= 1,
        "curve_canceled_by_near_wall": len(curve_canceled_by_near_wall_rows) >= 1,
        "steering_below_curve_floor": len(steering_below_curve_floor_rows) >= 1,
        "straight_through_curve": len(straight_through_curve_rows) >= 2,
        "mid_curve_flip": mid_curve_flip,
        "trajectory_slew_excess": len(trajectory_slew_excess_rows) >= 1,
        "trajectory_low_confidence": len(low_confidence_curve_rows) >= 3,
        "straight_convergence_failure": straight_convergence_failure,
        "counts": {
            "timeline_rows": len(rows),
            "nav_rows": len(nav_rows),
            "near_wall_rows": len(near_wall_rows),
            "wrong_direction_rows": len(wrong_direction_rows),
            "avoidance_missing_rows": len(avoidance_missing_rows),
            "servo_phase_rows": len(servo_phase_rows),
            "gate_open_but_zero_intent_rows": len(gate_open_but_zero_intent_rows),
            "wrong_sign_during_capture_rows": len(wrong_sign_during_capture_rows),
            "curve_canceled_by_near_wall_rows": len(curve_canceled_by_near_wall_rows),
            "steering_below_curve_floor_rows": len(steering_below_curve_floor_rows),
            "straight_through_curve_rows": len(straight_through_curve_rows),
            "trajectory_rows": len(trajectory_rows),
            "trajectory_curve_rows": len(trajectory_curve_rows),
            "low_confidence_curve_rows": len(low_confidence_curve_rows),
            "trajectory_slew_excess_rows": len(trajectory_slew_excess_rows),
            "straight_align_rows": len(straight_align_rows),
        },
    }


def write_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_svg_plot(output_path: Path, rows: list[dict], series: list[tuple[str, str]], title: str) -> None:
    numeric_rows = []
    for index, row in enumerate(rows):
        numeric_row = {"x": index}
        for field, _ in series:
            numeric_row[field] = safe_float(row.get(field))
        numeric_rows.append(numeric_row)

    values = [
        value
        for numeric_row in numeric_rows
        for field, _ in series
        for value in [numeric_row[field]]
        if value is not None
    ]
    if not values:
        output_path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>\n", encoding="utf-8")
        return

    width = 1000
    height = 420
    margin_left = 70
    margin_right = 20
    margin_top = 40
    margin_bottom = 50
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    min_value = min(values)
    max_value = max(values)
    if math.isclose(min_value, max_value):
        min_value -= 1.0
        max_value += 1.0

    def scale_x(index: int) -> float:
        if len(numeric_rows) <= 1:
            return margin_left + (plot_width / 2.0)
        return margin_left + (index / float(len(numeric_rows) - 1)) * plot_width

    def scale_y(value: float) -> float:
        return margin_top + (1.0 - ((value - min_value) / (max_value - min_value))) * plot_height

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{margin_left}' y='24' font-size='18' font-family='monospace'>{title}</text>",
        f"<line x1='{margin_left}' y1='{margin_top + plot_height}' x2='{margin_left + plot_width}' y2='{margin_top + plot_height}' stroke='#555'/>",
        f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{margin_top + plot_height}' stroke='#555'/>",
    ]

    for field, color in series:
        points = []
        for row in numeric_rows:
            value = row[field]
            if value is None:
                continue
            points.append(f"{scale_x(int(row['x'])):.2f},{scale_y(value):.2f}")
        if points:
            parts.append(
                f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{' '.join(points)}'/>"
            )

    legend_y = height - 18
    legend_x = margin_left
    for field, color in series:
        parts.append(f"<rect x='{legend_x}' y='{legend_y - 10}' width='10' height='10' fill='{color}'/>")
        parts.append(
            f"<text x='{legend_x + 16}' y='{legend_y}' font-size='12' font-family='monospace'>{field}</text>"
        )
        legend_x += 190

    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_summary(
    summary_path: Path,
    run_metadata: dict,
    config: dict,
    flags: dict,
    rows: list[dict],
) -> None:
    run_id = run_metadata.get("run_id", summary_path.parent.parent.name)
    lines = [
        f"# Debug Summary: {run_id}",
        "",
        "## Flags",
    ]
    for key in (
        "logic_sign_mismatch",
        "servo_sign_mismatch",
        "lidar_heading_offset_suspect",
        "centering_sign_mismatch",
        "front_stop_ok_but_direction_wrong",
        "wall_avoidance_inactive",
        "gate_open_but_zero_intent",
        "wrong_sign_during_capture",
        "curve_canceled_by_near_wall",
        "steering_below_curve_floor",
        "straight_through_curve",
        "mid_curve_flip",
        "trajectory_slew_excess",
        "trajectory_low_confidence",
        "straight_convergence_failure",
    ):
        lines.append(f"- `{key}`: `{flags.get(key)}`")

    lines.extend(
        [
            "",
            "## Counts",
        ]
    )
    for key, value in flags.get("counts", {}).items():
        lines.append(f"- `{key}`: `{value}`")

    if config:
        lines.extend(
            [
                "",
                "## Config Snapshot",
                f"- `diagnostic_mode`: `{config.get('diagnostic_mode')}`",
                f"- `steering_direction_sign`: `{config.get('steering_direction_sign')}`",
                f"- `heading_offset_deg`: `{config.get('heading_offset_deg')}`",
                f"- `wall_centering_gain_deg_per_m`: `{config.get('wall_centering_gain_deg_per_m')}`",
                f"- `wall_centering_base_weight`: `{config.get('wall_centering_base_weight')}`",
                f"- `wall_avoid_distance_m`: `{config.get('wall_avoid_distance_m')}`",
                f"- `wall_avoid_gain_deg_per_m`: `{config.get('wall_avoid_gain_deg_per_m')}`",
                f"- `corridor_balance_ratio_threshold`: `{config.get('corridor_balance_ratio_threshold')}`",
                f"- `corridor_front_turn_weight`: `{config.get('corridor_front_turn_weight')}`",
                f"- `corridor_wall_start_deg`: `{config.get('corridor_wall_start_deg')}`",
                f"- `corridor_wall_end_deg`: `{config.get('corridor_wall_end_deg')}`",
                f"- `wall_follow_target_distance_m`: `{config.get('wall_follow_target_distance_m')}`",
                f"- `wall_follow_gain_deg_per_m`: `{config.get('wall_follow_gain_deg_per_m')}`",
                f"- `wall_follow_release_balance_ratio`: `{config.get('wall_follow_release_balance_ratio')}`",
                f"- `startup_consensus_min_heading_deg`: `{config.get('startup_consensus_min_heading_deg')}`",
                f"- `startup_valid_cycles_required`: `{config.get('startup_valid_cycles_required')}`",
                f"- `startup_gap_lockout_cycles`: `{config.get('startup_gap_lockout_cycles')}`",
                f"- `startup_latch_cycles`: `{config.get('startup_latch_cycles')}`",
                f"- `trajectory_horizon_m`: `{config.get('trajectory_horizon_m')}`",
                f"- `trajectory_lookahead_min_m`: `{config.get('trajectory_lookahead_min_m')}`",
                f"- `trajectory_lookahead_max_m`: `{config.get('trajectory_lookahead_max_m')}`",
                f"- `trajectory_curvature_slew_per_cycle`: `{config.get('trajectory_curvature_slew_per_cycle')}`",
                f"- `trajectory_track_memory_alpha`: `{config.get('trajectory_track_memory_alpha')}`",
                f"- `trajectory_exit_heading_threshold_deg`: `{config.get('trajectory_exit_heading_threshold_deg')}`",
                f"- `trajectory_curve_speed_gain`: `{config.get('trajectory_curve_speed_gain')}`",
                f"- `trajectory_state_aux_weight`: `{config.get('trajectory_state_aux_weight')}`",
                f"- `trajectory_min_confidence`: `{config.get('trajectory_min_confidence')}`",
            ]
        )

    sample_rows = rows[:5]
    if sample_rows:
        lines.extend(["", "## First Rows"])
        for row in sample_rows:
            lines.append(
                "- cycle={cycle} phase={phase} step={step} target={target:.2f} steer={steer:.2f} source={source}".format(
                    cycle=row.get("cycle"),
                    phase=row.get("phase"),
                    step=row.get("step"),
                    target=safe_float(row.get("target_heading_deg")) or 0.0,
                    steer=safe_float(row.get("steering_pre_servo_deg")) or 0.0,
                    source=row.get("active_heading_source"),
                )
            )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _coerce_xy_points(value) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        return []
    points: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            points.append((float(item[0]), float(item[1])))
        except Exception:
            continue
    return points


def _tracking_rows(rows: list[dict], phase_name: str) -> list[dict]:
    return [
        row
        for row in rows
        if row.get("phase") == phase_name
        and row.get("step") == "curve_entry_track"
        and safe_float(row.get("pose_x")) is not None
        and safe_float(row.get("pose_y")) is not None
        and safe_float(row.get("pose_yaw_deg")) is not None
    ]


def _localize_track_rows(track_rows: list[dict]) -> tuple[dict, list[dict]]:
    lock_row = track_rows[0]
    lock_x_m = safe_float(lock_row.get("pose_x")) or 0.0
    lock_y_m = safe_float(lock_row.get("pose_y")) or 0.0
    lock_yaw_rad = math.radians(safe_float(lock_row.get("pose_yaw_deg")) or 0.0)
    c = math.cos(lock_yaw_rad)
    s = math.sin(lock_yaw_rad)

    localized: list[dict] = []
    for row in track_rows:
        pose_x_m = safe_float(row.get("pose_x")) or 0.0
        pose_y_m = safe_float(row.get("pose_y")) or 0.0
        dx_m = pose_x_m - lock_x_m
        dy_m = pose_y_m - lock_y_m
        local_x_m = (c * dx_m) + (s * dy_m)
        local_y_m = (-s * dx_m) + (c * dy_m)
        lookahead_x_m = safe_float(row.get("lookahead_x_m")) or 0.0
        lookahead_y_m = safe_float(row.get("lookahead_y_m")) or 0.0
        localized.append(
            {
                "cycle": int(safe_float(row.get("cycle")) or 0.0),
                "local_x_m": float(local_x_m),
                "local_y_m": float(local_y_m),
                "pose_yaw_deg": float(safe_float(row.get("pose_yaw_deg")) or 0.0),
                "target_heading_deg": float(safe_float(row.get("target_heading_deg")) or 0.0),
                "steering_deg": float(safe_float(row.get("steering_deg")) or 0.0),
                "speed_pct": float(safe_float(row.get("speed_pct")) or 0.0),
                "probe_path_progress": float(safe_float(row.get("probe_path_progress")) or 0.0),
                "probe_goal_distance_m": float(safe_float(row.get("probe_goal_distance_m")) or 0.0),
                "lookahead_x_m": float(lookahead_x_m),
                "lookahead_y_m": float(lookahead_y_m),
                "lookahead_distance_m": float(math.hypot(lookahead_x_m, lookahead_y_m)),
                "effective_front_clearance_m": float(safe_float(row.get("effective_front_clearance_m")) or 0.0),
                "front_clearance_m": float(safe_float(row.get("front_clearance_m")) or 0.0),
            }
        )
    return lock_row, localized


def _nearest_path_metrics(
    local_x_m: float,
    local_y_m: float,
    path_points: list[tuple[float, float]],
) -> tuple[int, float]:
    if not path_points:
        return 0, 0.0
    best_idx = 0
    best_distance_m = float("inf")
    for idx, (path_x_m, path_y_m) in enumerate(path_points):
        distance_m = math.hypot(local_x_m - path_x_m, local_y_m - path_y_m)
        if distance_m < best_distance_m:
            best_idx = idx
            best_distance_m = distance_m
    return best_idx, float(best_distance_m)


def write_curve_tracking_comparison_artifact(output_dir: Path, rows: list[dict]) -> None:
    track_rows = _tracking_rows(rows, "curve_entry_probe")
    if len(track_rows) < 2:
        return

    lock_row, localized_rows = _localize_track_rows(track_rows)
    path_points = _coerce_xy_points(lock_row.get("curve_window_path_xy_m"))
    left_profile = _coerce_xy_points(lock_row.get("curve_window_left_profile_xy_m"))
    right_profile = _coerce_xy_points(lock_row.get("curve_window_right_profile_xy_m"))
    wall_points = _coerce_xy_points(lock_row.get("curve_window_points_xy_m"))
    anchor_points = _coerce_xy_points(lock_row.get("curve_window_anchor_points_xy_m"))

    if not path_points:
        return

    expected_side = sign(safe_float(lock_row.get("curve_window_target_y_m")) or 0.0, threshold=0.05)
    wrong_side_cycle = None
    tracking_errors_m: list[float] = []
    nearest_path_index: list[int] = []
    for record in localized_rows:
        idx, error_m = _nearest_path_metrics(
            record["local_x_m"],
            record["local_y_m"],
            path_points,
        )
        nearest_path_index.append(idx)
        tracking_errors_m.append(error_m)
        actual_side = sign(record["local_y_m"], threshold=0.05)
        if (
            wrong_side_cycle is None
            and expected_side != 0
            and actual_side != 0
            and actual_side != expected_side
        ):
            wrong_side_cycle = record["cycle"]

    first_record = localized_rows[0]
    final_record = localized_rows[-1]
    progress_delta = final_record["probe_path_progress"] - first_record["probe_path_progress"]
    progress_stalled = progress_delta <= 0.03
    all_points = path_points + left_profile + right_profile + wall_points + anchor_points + [
        (record["local_x_m"], record["local_y_m"]) for record in localized_rows
    ]
    axis_limit_m = max(1.0, max(math.hypot(x_m, y_m) for x_m, y_m in all_points) * 1.10)

    payload = {
        "lock_cycle": first_record["cycle"],
        "final_cycle": final_record["cycle"],
        "expected_side": "left" if expected_side > 0 else ("right" if expected_side < 0 else "center"),
        "wrong_side_cycle": wrong_side_cycle,
        "progress_stalled": progress_stalled,
        "path_progress_start": first_record["probe_path_progress"],
        "path_progress_final": final_record["probe_path_progress"],
        "path_progress_delta": progress_delta,
        "goal_distance_start_m": first_record["probe_goal_distance_m"],
        "goal_distance_final_m": final_record["probe_goal_distance_m"],
        "lookahead_distance_start_m": first_record["lookahead_distance_m"],
        "lookahead_distance_final_m": final_record["lookahead_distance_m"],
        "tracking_error_mean_m": float(mean(tracking_errors_m)),
        "tracking_error_max_m": float(max(tracking_errors_m)),
        "tracking_error_final_m": float(tracking_errors_m[-1]),
        "actual_local_final_xy_m": [final_record["local_x_m"], final_record["local_y_m"]],
        "target_local_xy_m": [
            float(safe_float(lock_row.get("curve_window_target_x_m")) or 0.0),
            float(safe_float(lock_row.get("curve_window_target_y_m")) or 0.0),
        ],
        "localized_track_xy_m": [
            [record["local_x_m"], record["local_y_m"]] for record in localized_rows
        ],
        "ideal_path_xy_m": [[x_m, y_m] for x_m, y_m in path_points],
        "nearest_path_index": nearest_path_index,
        "tracking_error_by_cycle_m": tracking_errors_m,
    }

    json_path = output_dir / "curve_probe_motion_compare.json"
    png_path = output_dir / "curve_probe_motion_compare.png"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    cycles = [record["cycle"] for record in localized_rows]
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), constrained_layout=True)

    ax_xy = axes[0]
    if wall_points:
        ax_xy.scatter(
            [point[0] for point in wall_points],
            [point[1] for point in wall_points],
            s=10,
            c="#27c1d9",
            alpha=0.45,
            label="Puntos LiDAR bloqueados",
        )
    if left_profile:
        ax_xy.plot(
            [point[0] for point in left_profile],
            [point[1] for point in left_profile],
            linewidth=1.2,
            color="#4c78a8",
            alpha=0.70,
            label="Pared izquierda",
        )
    if right_profile:
        ax_xy.plot(
            [point[0] for point in right_profile],
            [point[1] for point in right_profile],
            linewidth=1.2,
            color="#54a24b",
            alpha=0.70,
            label="Pared derecha",
        )
    ax_xy.plot(
        [point[0] for point in path_points],
        [point[1] for point in path_points],
        linewidth=2.8,
        color="#d81b60",
        label="Trayectoria ideal bloqueada",
    )
    ax_xy.plot(
        [record["local_x_m"] for record in localized_rows],
        [record["local_y_m"] for record in localized_rows],
        linewidth=2.3,
        color="#111111",
        label="Trayectoria real estimada",
    )
    ax_xy.scatter([0.0], [0.0], s=44, c="#d62728", label="Pose al bloquear")
    ax_xy.scatter(
        [final_record["local_x_m"]],
        [final_record["local_y_m"]],
        s=68,
        c="#ff7f0e",
        label="Ultima pose estimada",
    )
    target_x_m = float(safe_float(lock_row.get("curve_window_target_x_m")) or 0.0)
    target_y_m = float(safe_float(lock_row.get("curve_window_target_y_m")) or 0.0)
    ax_xy.scatter([target_x_m], [target_y_m], s=60, c="#9467bd", label="Objetivo ideal")
    ax_xy.set_xlim(-axis_limit_m, axis_limit_m)
    ax_xy.set_ylim(-axis_limit_m, axis_limit_m)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True, alpha=0.25)
    ax_xy.set_xlabel("x local [m] (frente+)")
    ax_xy.set_ylabel("y local [m] (izquierda+)")
    ax_xy.set_title("Ideal vs resultado estimado")
    ax_xy.legend(loc="upper right")

    ax_err = axes[1]
    ax_err.plot(cycles, tracking_errors_m, color="#d62728", lw=2.0, label="Error a trayectoria [m]")
    ax_err.plot(
        cycles,
        [record["lookahead_distance_m"] for record in localized_rows],
        color="#ff7f0e",
        lw=1.8,
        label="Lookahead local [m]",
    )
    ax_err.plot(
        cycles,
        [record["probe_goal_distance_m"] for record in localized_rows],
        color="#9467bd",
        lw=1.8,
        label="Distancia a meta [m]",
    )
    ax_err.set_xlabel("Cycle")
    ax_err.set_ylabel("Metros")
    ax_err.set_title("Error, lookahead y distancia a meta")
    ax_err.grid(True, alpha=0.25)
    ax_err.legend(loc="upper left")

    ax_prog = axes[2]
    ax_prog.plot(
        cycles,
        [record["probe_path_progress"] for record in localized_rows],
        color="#1f77b4",
        lw=2.0,
        label="Progreso reportado",
    )
    ax_prog.plot(
        cycles,
        [idx / max(1, len(path_points) - 1) for idx in nearest_path_index],
        color="#2ca02c",
        lw=1.8,
        label="Punto ideal mas cercano",
    )
    ax_prog_twin = ax_prog.twinx()
    ax_prog_twin.plot(
        cycles,
        [record["target_heading_deg"] for record in localized_rows],
        color="#8c564b",
        lw=1.8,
        label="Heading objetivo [deg]",
    )
    ax_prog_twin.plot(
        cycles,
        [record["steering_deg"] for record in localized_rows],
        color="#17becf",
        lw=1.8,
        label="Steering [deg]",
    )
    ax_prog.set_xlabel("Cycle")
    ax_prog.set_ylabel("Progreso normalizado")
    ax_prog_twin.set_ylabel("Grados")
    ax_prog.set_title("Progreso congelado vs heading exigido")
    ax_prog.grid(True, alpha=0.25)
    lines_a, labels_a = ax_prog.get_legend_handles_labels()
    lines_b, labels_b = ax_prog_twin.get_legend_handles_labels()
    ax_prog.legend(lines_a + lines_b, labels_a + labels_b, loc="upper left")

    fig.suptitle(
        "Curve entry debug: trayectoria ideal vs trayectoria real estimada\n"
        f"wrong_side_cycle={wrong_side_cycle or 'none'} | progress_stalled={progress_stalled}",
        fontsize=13,
    )
    fig.savefig(png_path, dpi=170)
    plt.close(fig)


def _curve_probe_row_candidates(rows: list[dict], phase_name: str) -> list[dict]:
    return [
        row
        for row in rows
        if row.get("phase") == phase_name and _coerce_xy_points(row.get("curve_window_points_xy_m"))
    ]


def _select_curve_probe_row(rows: list[dict], phase_name: str) -> dict | None:
    phase_rows = _curve_probe_row_candidates(rows, phase_name)
    if not phase_rows:
        return None

    if phase_name == "curve_static_probe":
        locked_rows = [row for row in phase_rows if row.get("probe_locked")]
        if locked_rows:
            return locked_rows[-1]
        valid_rows = [row for row in phase_rows if row.get("curve_window_valid")]
        if valid_rows:
            return valid_rows[-1]
        return phase_rows[-1]

    tracking_rows = [
        row
        for row in phase_rows
        if row.get("probe_locked") and row.get("probe_subphase") in {"tracking", "arrived"}
    ]
    if tracking_rows:
        return tracking_rows[-1]
    locked_rows = [row for row in phase_rows if row.get("probe_locked")]
    if locked_rows:
        return locked_rows[-1]
    return phase_rows[-1]


def _plot_curve_probe_row(output_path: Path, row: dict, title: str) -> None:
    points = _coerce_xy_points(row.get("curve_window_points_xy_m"))
    left_profile = _coerce_xy_points(row.get("curve_window_left_profile_xy_m"))
    right_profile = _coerce_xy_points(row.get("curve_window_right_profile_xy_m"))
    left_fit = _coerce_xy_points(row.get("curve_window_left_fit_xy_m"))
    right_fit = _coerce_xy_points(row.get("curve_window_right_fit_xy_m"))
    path_points = _coerce_xy_points(row.get("curve_window_path_xy_m"))
    anchor_points = _coerce_xy_points(row.get("curve_window_anchor_points_xy_m"))

    all_points = points + left_profile + right_profile + left_fit + right_fit + path_points + anchor_points
    if not all_points:
        return
    axis_limit_m = max(1.0, max(math.hypot(x_m, y_m) for x_m, y_m in all_points) * 1.20)

    fig, ax = plt.subplots(figsize=(9, 9))
    if points:
        ax.scatter(
            [point[0] for point in points],
            [point[1] for point in points],
            s=10,
            c="#27c1d9",
            alpha=0.85,
            label="Puntos LiDAR",
        )
    ax.scatter([0.0], [0.0], s=44, c="#d62728", label="LiDAR")

    if left_fit:
        ax.plot(
            [point[0] for point in left_fit],
            [point[1] for point in left_fit],
            linestyle="--",
            linewidth=1.5,
            color="#1f77b4",
            label="Recta base izquierda",
        )
    if right_fit:
        ax.plot(
            [point[0] for point in right_fit],
            [point[1] for point in right_fit],
            linestyle="--",
            linewidth=1.5,
            color="#2ca02c",
            label="Recta base derecha",
        )
    if left_profile:
        ax.plot(
            [point[0] for point in left_profile],
            [point[1] for point in left_profile],
            linewidth=1.2,
            color="#4c78a8",
            alpha=0.70,
        )
    if right_profile:
        ax.plot(
            [point[0] for point in right_profile],
            [point[1] for point in right_profile],
            linewidth=1.2,
            color="#54a24b",
            alpha=0.70,
        )
    if path_points:
        ax.plot(
            [point[0] for point in path_points],
            [point[1] for point in path_points],
            linewidth=2.8,
            color="#d81b60",
            label="Trayectoria bloqueada",
        )
    if anchor_points:
        ax.scatter(
            [point[0] for point in anchor_points],
            [point[1] for point in anchor_points],
            s=26,
            c="#d81b60",
            alpha=0.85,
        )

    entry_x_m = safe_float(row.get("curve_window_entry_x_m")) or 0.0
    entry_y_m = safe_float(row.get("curve_window_entry_y_m")) or 0.0
    straight_end_x_m = safe_float(row.get("curve_window_straight_end_x_m")) or 0.0
    far_wall_x_m = safe_float(row.get("curve_window_far_wall_x_m")) or 0.0
    window_width_m = safe_float(row.get("curve_window_width_m")) or 0.0
    target_x_m = safe_float(row.get("curve_window_target_x_m")) or 0.0
    target_y_m = safe_float(row.get("curve_window_target_y_m")) or 0.0
    side = str(row.get("curve_window_side") or "none")
    side_label = "izquierda" if side == "left" else ("derecha" if side == "right" else side)

    if safe_float(row.get("curve_window_entry_x_m")) is not None:
        ax.axvline(
            straight_end_x_m,
            linestyle=":",
            linewidth=1.4,
            color="#6b6b6b",
            alpha=0.90,
            label="Fin pared recta",
        )
        ax.scatter(
            [entry_x_m],
            [entry_y_m],
            s=70,
            c="#111111",
            marker="x",
            label="Inicio estimado",
        )
    if safe_float(row.get("curve_window_target_x_m")) is not None:
        ax.scatter(
            [target_x_m],
            [target_y_m],
            s=60,
            c="#9467bd",
            label="Objetivo interior",
        )
    if far_wall_x_m > entry_x_m and abs(window_width_m) > 1e-6:
        ax.plot(
            [entry_x_m, far_wall_x_m],
            [entry_y_m, entry_y_m],
            linestyle=":",
            linewidth=2.0,
            color="#111111",
        )
        ax.text(
            0.5 * (entry_x_m + far_wall_x_m),
            entry_y_m + 0.04 * axis_limit_m,
            f"ancho ventana\n{window_width_m:.3f} m",
            fontsize=9,
            ha="center",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.90, "edgecolor": "#555555"},
        )

    info_lines = [
        f"Curva visible: {side_label}",
        f"Inicio estimado: x={entry_x_m:.3f} m",
        f"Fin pared recta: x={straight_end_x_m:.3f} m",
        f"Pared visible despues de ventana: x={far_wall_x_m:.3f} m",
        f"Ancho ventana: {window_width_m:.3f} m",
        f"Objetivo: x={target_x_m:.3f} m, y={target_y_m:.3f} m",
        f"Locked: {row.get('probe_locked')}",
        f"Subfase: {row.get('probe_subphase')}",
        f"Progreso: {(safe_float(row.get('probe_path_progress')) or 0.0):.3f}",
        f"Distancia meta: {(safe_float(row.get('probe_goal_distance_m')) or 0.0):.3f} m",
        f"Score: {(safe_float(row.get('curve_window_score')) or 0.0):.3f}",
        f"Cycle: {row.get('cycle')}",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(info_lines),
        transform=ax.transAxes,
        fontsize=10,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.92, "edgecolor": "#666666"},
    )
    ax.set_xlim(-axis_limit_m, axis_limit_m)
    ax.set_ylim(-axis_limit_m, axis_limit_m)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x [m] (frente+)")
    ax.set_ylabel("y [m] (izquierda+)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def write_curve_probe_artifacts(output_dir: Path, rows: list[dict]) -> None:
    for phase_name, prefix, title in (
        ("curve_static_probe", "curve_probe_static", "Validacion estatica de curva"),
        ("curve_entry_probe", "curve_probe_motion", "Seguimiento al primer punto interior"),
    ):
        selected_row = _select_curve_probe_row(rows, phase_name)
        if selected_row is None:
            continue
        payload = {
            "phase": phase_name,
            "cycle": selected_row.get("cycle"),
            "step": selected_row.get("step"),
            "probe_subphase": selected_row.get("probe_subphase"),
            "probe_locked": selected_row.get("probe_locked"),
            "probe_goal_distance_m": safe_float(selected_row.get("probe_goal_distance_m")),
            "probe_path_progress": safe_float(selected_row.get("probe_path_progress")),
            "curve_window_valid": selected_row.get("curve_window_valid"),
            "curve_window_side": selected_row.get("curve_window_side"),
            "curve_window_entry_x_m": safe_float(selected_row.get("curve_window_entry_x_m")),
            "curve_window_entry_y_m": safe_float(selected_row.get("curve_window_entry_y_m")),
            "curve_window_width_m": safe_float(selected_row.get("curve_window_width_m")),
            "curve_window_far_wall_x_m": safe_float(selected_row.get("curve_window_far_wall_x_m")),
            "curve_window_target_x_m": safe_float(selected_row.get("curve_window_target_x_m")),
            "curve_window_target_y_m": safe_float(selected_row.get("curve_window_target_y_m")),
            "curve_window_score": safe_float(selected_row.get("curve_window_score")),
            "curve_window_points_xy_m": selected_row.get("curve_window_points_xy_m"),
            "curve_window_left_profile_xy_m": selected_row.get("curve_window_left_profile_xy_m"),
            "curve_window_right_profile_xy_m": selected_row.get("curve_window_right_profile_xy_m"),
            "curve_window_left_fit_xy_m": selected_row.get("curve_window_left_fit_xy_m"),
            "curve_window_right_fit_xy_m": selected_row.get("curve_window_right_fit_xy_m"),
            "curve_window_anchor_points_xy_m": selected_row.get("curve_window_anchor_points_xy_m"),
            "curve_window_path_xy_m": selected_row.get("curve_window_path_xy_m"),
        }
        json_path = output_dir / f"{prefix}.json"
        png_path = output_dir / f"{prefix}.png"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _plot_curve_probe_row(png_path, selected_row, title)


def _pose_metrics(samples: list[dict]) -> dict | None:
    if not samples:
        return None
    start = samples[0]
    end = samples[-1]
    dx_m = float(end["x_m"] - start["x_m"])
    dy_m = float(end["y_m"] - start["y_m"])
    return {
        "start_x_m": float(start["x_m"]),
        "start_y_m": float(start["y_m"]),
        "end_x_m": float(end["x_m"]),
        "end_y_m": float(end["y_m"]),
        "dx_m": dx_m,
        "dy_m": dy_m,
        "disp_m": float(math.hypot(dx_m, dy_m)),
        "yaw_change_deg": float(
            math.degrees(_normalize_angle_rad(float(end["yaw_rad"]) - float(start["yaw_rad"])))
        ),
    }


def _series_speed_stats(samples: list[dict], key: str) -> dict | None:
    if not samples:
        return None
    values = [float(sample[key]) for sample in samples]
    return {
        "start_mps": float(values[0]),
        "end_mps": float(values[-1]),
        "avg_mps": float(mean(values)),
        "max_mps": float(max(values)),
    }


def _series_xyz_means(samples: list[dict], keys: tuple[str, str, str]) -> dict | None:
    if not samples:
        return None
    return {
        "mean_x": float(mean(float(sample[keys[0]]) for sample in samples)),
        "mean_y": float(mean(float(sample[keys[1]]) for sample in samples)),
        "mean_z": float(mean(float(sample[keys[2]]) for sample in samples)),
    }


def _unwrapped_yaw_series(samples: list[dict]) -> list[dict]:
    if not samples:
        return []
    unwrapped: list[dict] = []
    previous_yaw = None
    yaw_accum = 0.0
    for sample in samples:
        yaw_rad = float(sample["yaw_rad"])
        if previous_yaw is None:
            yaw_accum = yaw_rad
        else:
            yaw_accum += _normalize_angle_rad(yaw_rad - previous_yaw)
        previous_yaw = yaw_rad
        unwrapped.append(
            {
                "time_s": float(sample["time_s"]),
                "yaw_deg": float(math.degrees(yaw_accum)),
            }
        )
    return unwrapped


def _series_endpoint_delta_m(samples_a: list[dict], samples_b: list[dict]) -> float | None:
    if not samples_a or not samples_b:
        return None
    end_a = samples_a[-1]
    end_b = samples_b[-1]
    return float(
        math.hypot(
            float(end_a["x_m"]) - float(end_b["x_m"]),
            float(end_a["y_m"]) - float(end_b["y_m"]),
        )
    )


def _phase_pose_metrics(rows: list[dict], phase_name: str) -> dict | None:
    phase_rows = [
        row
        for row in rows
        if row.get("phase") == phase_name
        and safe_float(row.get("pose_x")) is not None
        and safe_float(row.get("pose_y")) is not None
        and safe_float(row.get("pose_yaw_deg")) is not None
    ]
    if len(phase_rows) < 2:
        return None

    start = phase_rows[0]
    end = phase_rows[-1]
    start_x_m = float(safe_float(start.get("pose_x")) or 0.0)
    start_y_m = float(safe_float(start.get("pose_y")) or 0.0)
    end_x_m = float(safe_float(end.get("pose_x")) or 0.0)
    end_y_m = float(safe_float(end.get("pose_y")) or 0.0)
    dx_m = end_x_m - start_x_m
    dy_m = end_y_m - start_y_m
    yaw_delta_deg = _normalize_angle_deg(
        float(safe_float(end.get("pose_yaw_deg")) or 0.0)
        - float(safe_float(start.get("pose_yaw_deg")) or 0.0)
    )
    return {
        "phase": phase_name,
        "start_cycle": int(safe_float(start.get("cycle")) or 0.0),
        "end_cycle": int(safe_float(end.get("cycle")) or 0.0),
        "start_xy_m": [start_x_m, start_y_m],
        "end_xy_m": [end_x_m, end_y_m],
        "dx_m": float(dx_m),
        "dy_m": float(dy_m),
        "disp_m": float(math.hypot(dx_m, dy_m)),
        "lateral_error_m": float(abs(dy_m)),
        "yaw_delta_deg": float(yaw_delta_deg),
        "pose_source": start.get("pose_pose_source"),
    }


def _infer_odom_root_cause(
    kin_pose: dict | None,
    odom_pose: dict | None,
    tf_odom_base: dict | None,
    tf_map_odom: dict | None,
    tf_map_base: dict | None,
    kin_speed: dict | None,
    kin_heading_change_deg: float | None,
) -> tuple[str, str]:
    kin_disp_m = safe_float((kin_pose or {}).get("disp_m")) or 0.0
    odom_disp_m = safe_float((odom_pose or {}).get("disp_m")) or 0.0
    tf_odom_disp_m = safe_float((tf_odom_base or {}).get("disp_m")) or 0.0
    map_odom_disp_m = safe_float((tf_map_odom or {}).get("disp_m")) or 0.0
    map_base_disp_m = safe_float((tf_map_base or {}).get("disp_m")) or 0.0
    kin_speed_avg_mps = safe_float((kin_speed or {}).get("avg_mps")) or 0.0
    yaw_change_deg = abs(float(kin_heading_change_deg or 0.0))

    if kin_disp_m > 0.15 and abs(odom_disp_m - kin_disp_m) <= max(0.10, kin_disp_m * 0.25):
        if kin_speed_avg_mps > 0.03 and yaw_change_deg < 3.0:
            return (
                "kinematics_estimator_bias_or_gravity_compensation",
                "La deriva nace en /apex/kinematics/position y /odom solo la replica. "
                "Como la velocidad falsa aparece en estatico con yaw casi fijo, el culpable probable "
                "es sesgo residual de acelerometro, compensacion de gravedad o desalineacion del IMU.",
            )
        return (
            "kinematics_estimator_integration_drift",
            "La deriva nace en /apex/kinematics/position y /odom solo la replica. "
            "El origen esta aguas arriba del bridge de odometria.",
        )

    if odom_disp_m > (kin_disp_m + 0.10) and abs(tf_odom_disp_m - odom_disp_m) <= max(0.10, odom_disp_m * 0.25):
        return (
            "kinematics_odometry_bridge_or_tf_odom",
            "La cinemática deriva menos que /odom. El problema probable esta en el bridge de /odom o en TF odom->base_link.",
        )

    if map_base_disp_m > (odom_disp_m + 0.10) or map_odom_disp_m > 0.10:
        return (
            "slam_or_tf_map_layer",
            "La deriva adicional aparece en la capa map/SLAM, no en la odometria base.",
        )

    return (
        "inconclusive",
        "No hay separacion suficiente entre capas para aislar la causa solo con este bundle.",
    )


def _mean_true_ratio(samples: list[dict], predicate) -> float | None:
    if not samples:
        return None
    return float(mean(1.0 if predicate(sample) else 0.0 for sample in samples))


def _fusion_reacquire_metrics(
    fusion_status_samples: list[dict],
    fused_pose_samples: list[dict],
) -> tuple[int, float | None]:
    if len(fusion_status_samples) < 2:
        return 0, None

    reacquire_times_s: list[float] = []
    for prev, curr in zip(fusion_status_samples[:-1], fusion_status_samples[1:]):
        if (not prev["last_lidar_translation_observable"]) and curr["last_lidar_translation_observable"]:
            reacquire_times_s.append(float(curr["time_s"]))

    if not reacquire_times_s or len(fused_pose_samples) < 2:
        return len(reacquire_times_s), None

    max_jump_m = None
    for reacquire_time_s in reacquire_times_s:
        for prev, curr in zip(fused_pose_samples[:-1], fused_pose_samples[1:]):
            prev_time_s = float(prev["time_s"])
            curr_time_s = float(curr["time_s"])
            if prev_time_s <= reacquire_time_s <= curr_time_s and (curr_time_s - reacquire_time_s) <= 0.35:
                jump_m = float(
                    math.hypot(
                        float(curr["x_m"]) - float(prev["x_m"]),
                        float(curr["y_m"]) - float(prev["y_m"]),
                    )
                )
                if max_jump_m is None or jump_m > max_jump_m:
                    max_jump_m = jump_m
                break

    return len(reacquire_times_s), max_jump_m


def write_odom_drift_artifact(bundle_dir: Path, output_dir: Path) -> tuple[dict[str, list[dict]] | None, dict | None]:
    if rosbag2_py is None:
        return None, None

    bag_dir = resolve_raw_bag_dir(bundle_dir)
    if bag_dir is None:
        return None, None

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    series: dict[str, list[dict]] = {
        "kinematics_position": [],
        "kinematics_velocity": [],
        "kinematics_acceleration": [],
        "kinematics_heading": [],
        "kinematics_status": [],
        "imu_raw_odom_pose": [],
        "imu_raw_odom_speed": [],
        "lidar_local_pose": [],
        "fused_odom_pose": [],
        "fused_odom_speed": [],
        "fusion_status": [],
        "imu_filtered_heading": [],
        "odom_pose": [],
        "odom_speed": [],
        "tf_odom_base_link": [],
        "tf_map_odom": [],
        "tf_map_base_link": [],
        "tf_odom_imu_raw_base_link": [],
        "tf_odom_lidar_local_odom_imu_raw": [],
        "imu_accel_raw": [],
        "imu_gyro_raw": [],
    }
    start_ns = None

    while reader.has_next():
        topic_name, data, timestamp_ns = reader.read_next()
        if start_ns is None:
            start_ns = timestamp_ns
        time_s = float(timestamp_ns - start_ns) * 1e-9

        if topic_name == "/apex/kinematics/position":
            msg = deserialize_message(data, PointStamped)
            series["kinematics_position"].append(
                {"time_s": time_s, "x_m": float(msg.point.x), "y_m": float(msg.point.y), "yaw_rad": 0.0}
            )
            continue

        if topic_name == "/apex/kinematics/velocity":
            msg = deserialize_message(data, Vector3Stamped)
            speed_mps = math.hypot(float(msg.vector.x), float(msg.vector.y))
            series["kinematics_velocity"].append(
                {
                    "time_s": time_s,
                    "vx_mps": float(msg.vector.x),
                    "vy_mps": float(msg.vector.y),
                    "speed_mps": float(speed_mps),
                }
            )
            continue

        if topic_name == "/apex/kinematics/acceleration":
            msg = deserialize_message(data, Vector3Stamped)
            planar_mps2 = math.hypot(float(msg.vector.x), float(msg.vector.y))
            series["kinematics_acceleration"].append(
                {
                    "time_s": time_s,
                    "ax_mps2": float(msg.vector.x),
                    "ay_mps2": float(msg.vector.y),
                    "az_mps2": float(msg.vector.z),
                    "planar_mps2": float(planar_mps2),
                }
            )
            continue

        if topic_name == "/apex/kinematics/heading":
            msg = deserialize_message(data, Vector3Stamped)
            series["kinematics_heading"].append(
                {"time_s": time_s, "yaw_rad": float(msg.vector.z)}
            )
            continue

        if topic_name == "/apex/kinematics/status":
            msg = deserialize_message(data, String)
            try:
                payload = json.loads(msg.data)
            except Exception:
                payload = {}
            if isinstance(payload, dict):
                series["kinematics_status"].append(
                    {
                        "time_s": time_s,
                        "stationary_detected": bool(payload.get("stationary_detected", False)),
                        "zupt_applied": bool(payload.get("zupt_applied", False)),
                        "velocity_decay_active": bool(payload.get("velocity_decay_active", False)),
                        "odom_translation_confidence": float(
                            payload.get("odom_translation_confidence", 0.0) or 0.0
                        ),
                        "raw_accel_planar_mps2": float(
                            payload.get("raw_accel_planar_mps2", 0.0) or 0.0
                        ),
                        "corrected_accel_planar_mps2": float(
                            payload.get("corrected_accel_planar_mps2", 0.0) or 0.0
                        ),
                        "calibration_active": bool(payload.get("calibration_active", False)),
                        "calibration_complete": bool(payload.get("calibration_complete", False)),
                    }
                )
            continue

        if topic_name == "/apex/imu/acceleration/raw":
            msg = deserialize_message(data, Vector3Stamped)
            series["imu_accel_raw"].append(
                {
                    "time_s": time_s,
                    "ax_mps2": float(msg.vector.x),
                    "ay_mps2": float(msg.vector.y),
                    "az_mps2": float(msg.vector.z),
                }
            )
            continue

        if topic_name == "/apex/imu/angular_velocity/raw":
            msg = deserialize_message(data, Vector3Stamped)
            series["imu_gyro_raw"].append(
                {
                    "time_s": time_s,
                    "gx_rps": float(msg.vector.x),
                    "gy_rps": float(msg.vector.y),
                    "gz_rps": float(msg.vector.z),
                }
            )
            continue

        if topic_name == "/apex/imu/data_filtered":
            msg = deserialize_message(data, Imu)
            yaw_rad = _yaw_from_quat(
                float(msg.orientation.x),
                float(msg.orientation.y),
                float(msg.orientation.z),
                float(msg.orientation.w),
            )
            series["imu_filtered_heading"].append(
                {
                    "time_s": time_s,
                    "yaw_rad": float(yaw_rad),
                    "yaw_rate_rps": float(msg.angular_velocity.z),
                }
            )
            continue

        if topic_name == "/apex/lidar/pose_local":
            msg = deserialize_message(data, PoseWithCovarianceStamped)
            yaw_rad = _yaw_from_quat(
                float(msg.pose.pose.orientation.x),
                float(msg.pose.pose.orientation.y),
                float(msg.pose.pose.orientation.z),
                float(msg.pose.pose.orientation.w),
            )
            series["lidar_local_pose"].append(
                {
                    "time_s": time_s,
                    "x_m": float(msg.pose.pose.position.x),
                    "y_m": float(msg.pose.pose.position.y),
                    "yaw_rad": float(yaw_rad),
                    "cov_x_m2": float(msg.pose.covariance[0]),
                    "cov_y_m2": float(msg.pose.covariance[7]),
                    "cov_yaw_rad2": float(msg.pose.covariance[35]),
                }
            )
            continue

        if topic_name == "/apex/odometry/imu_raw":
            msg = deserialize_message(data, Odometry)
            yaw_rad = _yaw_from_quat(
                float(msg.pose.pose.orientation.x),
                float(msg.pose.pose.orientation.y),
                float(msg.pose.pose.orientation.z),
                float(msg.pose.pose.orientation.w),
            )
            speed_mps = math.hypot(float(msg.twist.twist.linear.x), float(msg.twist.twist.linear.y))
            series["imu_raw_odom_pose"].append(
                {
                    "time_s": time_s,
                    "x_m": float(msg.pose.pose.position.x),
                    "y_m": float(msg.pose.pose.position.y),
                    "yaw_rad": float(yaw_rad),
                }
            )
            series["imu_raw_odom_speed"].append(
                {
                    "time_s": time_s,
                    "vx_mps": float(msg.twist.twist.linear.x),
                    "vy_mps": float(msg.twist.twist.linear.y),
                    "speed_mps": float(speed_mps),
                    "yaw_rate_rps": float(msg.twist.twist.angular.z),
                }
            )
            continue

        if topic_name == "/odometry/filtered":
            msg = deserialize_message(data, Odometry)
            yaw_rad = _yaw_from_quat(
                float(msg.pose.pose.orientation.x),
                float(msg.pose.pose.orientation.y),
                float(msg.pose.pose.orientation.z),
                float(msg.pose.pose.orientation.w),
            )
            speed_mps = math.hypot(float(msg.twist.twist.linear.x), float(msg.twist.twist.linear.y))
            series["fused_odom_pose"].append(
                {
                    "time_s": time_s,
                    "x_m": float(msg.pose.pose.position.x),
                    "y_m": float(msg.pose.pose.position.y),
                    "yaw_rad": float(yaw_rad),
                }
            )
            series["fused_odom_speed"].append(
                {
                    "time_s": time_s,
                    "vx_mps": float(msg.twist.twist.linear.x),
                    "vy_mps": float(msg.twist.twist.linear.y),
                    "speed_mps": float(speed_mps),
                    "yaw_rate_rps": float(msg.twist.twist.angular.z),
                }
            )
            continue

        if topic_name == "/apex/odometry/fusion_status":
            msg = deserialize_message(data, String)
            try:
                payload = json.loads(msg.data)
            except Exception:
                payload = {}
            if isinstance(payload, dict):
                series["fusion_status"].append(
                    {
                        "time_s": time_s,
                        "lidar_pose_fresh": bool(payload.get("lidar_pose_fresh", False)),
                        "last_lidar_translation_observable": bool(
                            payload.get("last_lidar_translation_observable", False)
                        ),
                        "lidar_position_update_suppressed": bool(
                            payload.get("lidar_position_update_suppressed", False)
                        ),
                        "last_lidar_position_gain": float(
                            payload.get("last_lidar_position_gain", 0.0) or 0.0
                        ),
                        "last_lidar_velocity_gain": float(
                            payload.get("last_lidar_velocity_gain", 0.0) or 0.0
                        ),
                        "last_lidar_delta_m": float(payload.get("last_lidar_delta_m", 0.0) or 0.0),
                        "last_lidar_speed_mps": float(
                            payload.get("last_lidar_speed_mps", 0.0) or 0.0
                        ),
                    }
                )
            continue

        if topic_name == "/odom":
            msg = deserialize_message(data, Odometry)
            yaw_rad = _yaw_from_quat(
                float(msg.pose.pose.orientation.x),
                float(msg.pose.pose.orientation.y),
                float(msg.pose.pose.orientation.z),
                float(msg.pose.pose.orientation.w),
            )
            speed_mps = math.hypot(float(msg.twist.twist.linear.x), float(msg.twist.twist.linear.y))
            series["odom_pose"].append(
                {
                    "time_s": time_s,
                    "x_m": float(msg.pose.pose.position.x),
                    "y_m": float(msg.pose.pose.position.y),
                    "yaw_rad": float(yaw_rad),
                }
            )
            series["odom_speed"].append(
                {
                    "time_s": time_s,
                    "vx_mps": float(msg.twist.twist.linear.x),
                    "vy_mps": float(msg.twist.twist.linear.y),
                    "speed_mps": float(speed_mps),
                    "yaw_rate_rps": float(msg.twist.twist.angular.z),
                }
            )
            continue

        if topic_name != "/tf":
            continue

        msg = deserialize_message(data, TFMessage)
        for transform in msg.transforms:
            key = f"{transform.header.frame_id}->{transform.child_frame_id}"
            yaw_rad = _yaw_from_quat(
                float(transform.transform.rotation.x),
                float(transform.transform.rotation.y),
                float(transform.transform.rotation.z),
                float(transform.transform.rotation.w),
            )
            sample = {
                "time_s": time_s,
                "x_m": float(transform.transform.translation.x),
                "y_m": float(transform.transform.translation.y),
                "yaw_rad": float(yaw_rad),
            }
            if key == "odom->base_link":
                series["tf_odom_base_link"].append(sample)
            elif key == "odom_imu_raw->base_link":
                series["tf_odom_imu_raw_base_link"].append(sample)
            elif key == "map->odom":
                series["tf_map_odom"].append(sample)
            elif key == "map->base_link":
                series["tf_map_base_link"].append(sample)
            elif key == "odom_lidar_local->odom_imu_raw":
                series["tf_odom_lidar_local_odom_imu_raw"].append(sample)

    kin_pose_metrics = _pose_metrics(series["kinematics_position"])
    odom_pose_samples = series["odom_pose"] if series["odom_pose"] else series["imu_raw_odom_pose"]
    odom_speed_samples = series["odom_speed"] if series["odom_speed"] else series["imu_raw_odom_speed"]
    odom_pose_metrics = _pose_metrics(odom_pose_samples)
    tf_odom_metrics = _pose_metrics(series["tf_odom_base_link"])
    tf_map_odom_metrics = _pose_metrics(series["tf_map_odom"])
    tf_map_base_metrics = _pose_metrics(series["tf_map_base_link"])
    kin_speed_stats = _series_speed_stats(series["kinematics_velocity"], "speed_mps")
    odom_speed_stats = _series_speed_stats(odom_speed_samples, "speed_mps")
    imu_raw_odom_metrics = _pose_metrics(series["imu_raw_odom_pose"])
    lidar_local_metrics = _pose_metrics(series["lidar_local_pose"])
    fused_odom_metrics = _pose_metrics(series["fused_odom_pose"])
    fusion_status_samples = series["fusion_status"]
    kin_heading_change_deg = None
    if series["kinematics_heading"]:
        kin_heading_change_deg = math.degrees(
            _normalize_angle_rad(
                float(series["kinematics_heading"][-1]["yaw_rad"])
                - float(series["kinematics_heading"][0]["yaw_rad"])
            )
        )

    def _window_disp(samples: list[dict], window_s: float) -> float | None:
        if not samples:
            return None
        end_t = float(samples[-1]["time_s"])
        start_index = 0
        for index, sample in enumerate(samples):
            if end_t - float(sample["time_s"]) <= window_s:
                start_index = index
                break
        start_sample = samples[start_index]
        end_sample = samples[-1]
        return float(
            math.hypot(
                float(end_sample["x_m"]) - float(start_sample["x_m"]),
                float(end_sample["y_m"]) - float(start_sample["y_m"]),
            )
        )

    root_cause, interpretation = _infer_odom_root_cause(
        kin_pose_metrics,
        odom_pose_metrics,
        tf_odom_metrics,
        tf_map_odom_metrics,
        tf_map_base_metrics,
        kin_speed_stats,
        kin_heading_change_deg,
    )
    fusion_reacquire_count, fusion_reacquire_max_jump_m = _fusion_reacquire_metrics(
        fusion_status_samples, series["fused_odom_pose"]
    )

    payload = {
        "duration_s": max(
            [
                0.0,
                *[float(samples[-1]["time_s"]) for samples in series.values() if samples],
            ]
        ),
        "available_series": {name: len(samples) for name, samples in series.items()},
        "suspected_root_cause": root_cause,
        "interpretation": interpretation,
        "kinematics_position": kin_pose_metrics,
        "kinematics_velocity": kin_speed_stats,
        "kinematics_heading": {
            "yaw_change_deg": float(kin_heading_change_deg or 0.0),
        }
        if kin_heading_change_deg is not None
        else None,
        "odom_pose": odom_pose_metrics,
        "odom_speed": odom_speed_stats,
        "imu_raw_odom_pose": imu_raw_odom_metrics,
        "lidar_local_pose": lidar_local_metrics,
        "fused_odom_pose": fused_odom_metrics,
        "tf_odom_base_link": tf_odom_metrics,
        "tf_map_odom": tf_map_odom_metrics,
        "tf_map_base_link": tf_map_base_metrics,
        "imu_accel_raw_mean": _series_xyz_means(series["imu_accel_raw"], ("ax_mps2", "ay_mps2", "az_mps2")),
        "imu_gyro_raw_mean": _series_xyz_means(series["imu_gyro_raw"], ("gx_rps", "gy_rps", "gz_rps")),
        "lidar_translation_observable_ratio": _mean_true_ratio(
            fusion_status_samples,
            lambda sample: sample["last_lidar_translation_observable"],
        ),
        "lidar_fresh_but_unobservable_ratio": _mean_true_ratio(
            fusion_status_samples,
            lambda sample: sample["lidar_pose_fresh"] and not sample["last_lidar_translation_observable"],
        ),
        "fusion_prediction_only_ratio": _mean_true_ratio(
            fusion_status_samples,
            lambda sample: sample["lidar_position_update_suppressed"],
        ),
        "fusion_position_update_suppressed_ratio": _mean_true_ratio(
            fusion_status_samples,
            lambda sample: sample["lidar_position_update_suppressed"],
        ),
        "fusion_lidar_reacquire_count": fusion_reacquire_count,
        "fusion_lidar_reacquire_max_jump_m": fusion_reacquire_max_jump_m,
        "fusion_lidar_position_gain_mean": float(
            mean(sample["last_lidar_position_gain"] for sample in fusion_status_samples)
        )
        if fusion_status_samples
        else None,
        "fusion_lidar_velocity_gain_mean": float(
            mean(sample["last_lidar_velocity_gain"] for sample in fusion_status_samples)
        )
        if fusion_status_samples
        else None,
        "fusion_lidar_delta_mean_m": float(
            mean(sample["last_lidar_delta_m"] for sample in fusion_status_samples)
        )
        if fusion_status_samples
        else None,
        "fusion_lidar_speed_mean_mps": float(
            mean(sample["last_lidar_speed_mps"] for sample in fusion_status_samples)
        )
        if fusion_status_samples
        else None,
        "stationary_detected_ratio": float(
            mean(1.0 if sample["stationary_detected"] else 0.0 for sample in series["kinematics_status"])
        )
        if series["kinematics_status"]
        else None,
        "zupt_applied_ratio": float(
            mean(1.0 if sample["zupt_applied"] else 0.0 for sample in series["kinematics_status"])
        )
        if series["kinematics_status"]
        else None,
        "velocity_decay_ratio": float(
            mean(1.0 if sample["velocity_decay_active"] else 0.0 for sample in series["kinematics_status"])
        )
        if series["kinematics_status"]
        else None,
        "fused_tail_disp_last_5s_m": _window_disp(series["fused_odom_pose"], 5.0),
        "imu_raw_tail_disp_last_5s_m": _window_disp(series["imu_raw_odom_pose"], 5.0),
    }

    json_path = output_dir / "odom_drift.json"
    png_path = output_dir / "odom_drift.png"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)

    ax_xy = axes[0][0]
    for label, color, key in (
        ("Kinematics pos", "#1f77b4", "kinematics_position"),
        ("IMU raw odom", "#d62728", "imu_raw_odom_pose"),
        ("Fused odom", "#ff7f0e", "fused_odom_pose"),
        ("Legacy /odom", "#8c564b", "odom_pose"),
        ("TF odom->base", "#2ca02c", "tf_odom_base_link"),
        ("TF map->base", "#9467bd", "tf_map_base_link"),
    ):
        samples = series[key]
        if not samples:
            continue
        ax_xy.plot(
            [sample["x_m"] for sample in samples],
            [sample["y_m"] for sample in samples],
            lw=2.0,
            label=label,
            color=color,
        )
        ax_xy.scatter([samples[0]["x_m"]], [samples[0]["y_m"]], s=22, color=color)
        ax_xy.scatter([samples[-1]["x_m"]], [samples[-1]["y_m"]], s=36, color=color, marker="x")
    ax_xy.set_title("Drift XY por capa")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.grid(True, alpha=0.25)
    ax_xy.axis("equal")
    ax_xy.legend(loc="best")

    ax_speed = axes[0][1]
    if series["kinematics_velocity"]:
        ax_speed.plot(
            [sample["time_s"] for sample in series["kinematics_velocity"]],
            [sample["speed_mps"] for sample in series["kinematics_velocity"]],
            lw=2.0,
            color="#1f77b4",
            label="Velocidad kinematics [m/s]",
        )
    if series["imu_raw_odom_speed"]:
        ax_speed.plot(
            [sample["time_s"] for sample in series["imu_raw_odom_speed"]],
            [sample["speed_mps"] for sample in series["imu_raw_odom_speed"]],
            lw=1.8,
            color="#d62728",
            label="Velocidad imu_raw [m/s]",
        )
    if series["fused_odom_speed"]:
        ax_speed.plot(
            [sample["time_s"] for sample in series["fused_odom_speed"]],
            [sample["speed_mps"] for sample in series["fused_odom_speed"]],
            lw=1.8,
            color="#ff7f0e",
            label="Velocidad fused [m/s]",
        )
    ax_speed.set_title("Velocidad aparente en estatico")
    ax_speed.set_xlabel("Tiempo [s]")
    ax_speed.set_ylabel("m/s")
    ax_speed.grid(True, alpha=0.25)
    if series["kinematics_status"]:
        ax_speed_status = ax_speed.twinx()
        ax_speed_status.step(
            [sample["time_s"] for sample in series["kinematics_status"]],
            [1.0 if sample["stationary_detected"] else 0.0 for sample in series["kinematics_status"]],
            where="post",
            lw=1.6,
            color="#2ca02c",
            label="stationary",
        )
        ax_speed_status.step(
            [sample["time_s"] for sample in series["kinematics_status"]],
            [1.0 if sample["zupt_applied"] else 0.0 for sample in series["kinematics_status"]],
            where="post",
            lw=1.4,
            color="#9467bd",
            label="zupt",
        )
        ax_speed_status.set_ylim(-0.05, 1.05)
        ax_speed_status.set_ylabel("estado")
        lines_a, labels_a = ax_speed.get_legend_handles_labels()
        lines_b, labels_b = ax_speed_status.get_legend_handles_labels()
        ax_speed.legend(lines_a + lines_b, labels_a + labels_b, loc="best")
    else:
        ax_speed.legend(loc="best")

    ax_accel = axes[1][0]
    if series["kinematics_status"]:
        ax_accel.plot(
            [sample["time_s"] for sample in series["kinematics_status"]],
            [sample["raw_accel_planar_mps2"] for sample in series["kinematics_status"]],
            lw=1.6,
            color="#ff7f0e",
            label="Accel planar raw [m/s2]",
        )
        ax_accel.plot(
            [sample["time_s"] for sample in series["kinematics_status"]],
            [sample["corrected_accel_planar_mps2"] for sample in series["kinematics_status"]],
            lw=1.8,
            color="#1f77b4",
            label="Accel planar corregida [m/s2]",
        )
        ax_accel_conf = ax_accel.twinx()
        ax_accel_conf.plot(
            [sample["time_s"] for sample in series["kinematics_status"]],
            [sample["odom_translation_confidence"] for sample in series["kinematics_status"]],
            lw=1.6,
            color="#2ca02c",
            label="Confianza translacion",
        )
        ax_accel_conf.set_ylim(-0.05, 1.05)
        ax_accel_conf.set_ylabel("confianza")
        lines_a, labels_a = ax_accel.get_legend_handles_labels()
        lines_b, labels_b = ax_accel_conf.get_legend_handles_labels()
        ax_accel.legend(lines_a + lines_b, labels_a + labels_b, loc="best")
    else:
        if series["kinematics_heading"]:
            ax_accel.plot(
                [sample["time_s"] for sample in series["kinematics_heading"]],
                [math.degrees(sample["yaw_rad"]) for sample in series["kinematics_heading"]],
                lw=2.0,
                color="#1f77b4",
                label="Heading kinematics [deg]",
            )
        if series["odom_pose"]:
            ax_accel.plot(
                [sample["time_s"] for sample in series["odom_pose"]],
                [math.degrees(sample["yaw_rad"]) for sample in series["odom_pose"]],
                lw=1.8,
                color="#d62728",
                label="Yaw odom [deg]",
            )
        ax_accel.legend(loc="best")
    ax_accel.set_title("Aceleracion corregida y confianza")
    ax_accel.set_xlabel("Tiempo [s]")
    ax_accel.set_ylabel("m/s2")
    ax_accel.grid(True, alpha=0.25)

    ax_text = axes[1][1]
    ax_text.axis("off")
    text_lines = [
        f"Suspect: {root_cause}",
        "",
        interpretation,
        "",
        f"Kinematics drift: {(kin_pose_metrics or {}).get('disp_m', 0.0):.3f} m",
        f"Odom drift: {(odom_pose_metrics or {}).get('disp_m', 0.0):.3f} m",
        f"TF odom->base drift: {(tf_odom_metrics or {}).get('disp_m', 0.0):.3f} m",
        f"TF map->odom drift: {(tf_map_odom_metrics or {}).get('disp_m', 0.0):.3f} m",
        f"Avg kinematics speed: {(kin_speed_stats or {}).get('avg_mps', 0.0):.3f} m/s",
        f"Heading change: {float(kin_heading_change_deg or 0.0):.3f} deg",
    ]
    if payload.get("stationary_detected_ratio") is not None:
        text_lines.extend(
            [
                f"stationary ratio: {float(payload['stationary_detected_ratio']):.3f}",
                f"zupt ratio: {float(payload['zupt_applied_ratio']):.3f}",
                f"velocity decay ratio: {float(payload['velocity_decay_ratio']):.3f}",
            ]
        )
    if payload.get("lidar_fresh_but_unobservable_ratio") is not None:
        text_lines.extend(
            [
                f"lidar fresh+unobservable: {float(payload['lidar_fresh_but_unobservable_ratio']):.3f}",
                f"fusion prediction-only: {float(payload['fusion_prediction_only_ratio']):.3f}",
                f"lidar reacquire count: {int(payload['fusion_lidar_reacquire_count'] or 0)}",
                f"lidar reacquire max jump: {float(payload['fusion_lidar_reacquire_max_jump_m'] or 0.0):.3f} m",
            ]
        )
    accel_mean = payload.get("imu_accel_raw_mean") or {}
    gyro_mean = payload.get("imu_gyro_raw_mean") or {}
    if accel_mean:
        text_lines.extend(
            [
                "",
                "IMU raw mean accel [m/s2]:",
                f"  x={float(accel_mean.get('mean_x', 0.0)):.4f} y={float(accel_mean.get('mean_y', 0.0)):.4f} z={float(accel_mean.get('mean_z', 0.0)):.4f}",
            ]
        )
    if gyro_mean:
        text_lines.extend(
            [
                "IMU raw mean gyro [rad/s]:",
                f"  x={float(gyro_mean.get('mean_x', 0.0)):.4f} y={float(gyro_mean.get('mean_y', 0.0)):.4f} z={float(gyro_mean.get('mean_z', 0.0)):.4f}",
            ]
        )
    else:
        text_lines.extend(
            [
                "",
                "Raw IMU no estaba grabada en este bundle.",
                "En la siguiente corrida ya quedara incluida.",
            ]
        )
    ax_text.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.92, "edgecolor": "#666666"},
        transform=ax_text.transAxes,
    )

    fig.suptitle("Diagnostico de deriva de odometria", fontsize=14)
    fig.savefig(png_path, dpi=170)
    plt.close(fig)
    return series, payload


def write_sensor_fusion_compare_artifact(
    output_dir: Path,
    rows: list[dict],
    series: dict[str, list[dict]] | None,
    odom_payload: dict | None,
) -> None:
    if not series:
        return

    imu_raw_pose = series.get("imu_raw_odom_pose", [])
    lidar_local_pose = series.get("lidar_local_pose", [])
    fused_odom_pose = series.get("fused_odom_pose", [])
    kinematics_heading = series.get("kinematics_heading", [])
    imu_filtered_heading = series.get("imu_filtered_heading", [])
    kinematics_status = series.get("kinematics_status", [])
    imu_raw_speed = series.get("imu_raw_odom_speed", [])
    fused_speed = series.get("fused_odom_speed", [])

    if not any((imu_raw_pose, lidar_local_pose, fused_odom_pose)):
        return

    lidar_cov_base_x = min(
        (float(sample["cov_x_m2"]) for sample in lidar_local_pose if float(sample["cov_x_m2"]) > 0.0),
        default=None,
    )
    lidar_cov_base_yaw = min(
        (float(sample["cov_yaw_rad2"]) for sample in lidar_local_pose if float(sample["cov_yaw_rad2"]) > 0.0),
        default=None,
    )
    lidar_degraded_samples = []
    for sample in lidar_local_pose:
        degraded = False
        if lidar_cov_base_x is not None and float(sample["cov_x_m2"]) > (lidar_cov_base_x * 4.0):
            degraded = True
        if lidar_cov_base_yaw is not None and float(sample["cov_yaw_rad2"]) > (lidar_cov_base_yaw * 4.0):
            degraded = True
        lidar_degraded_samples.append(
            {
                "time_s": float(sample["time_s"]),
                "degraded": degraded,
                "cov_x_m2": float(sample["cov_x_m2"]),
                "cov_yaw_rad2": float(sample["cov_yaw_rad2"]),
            }
        )

    lidar_gap_max_s = None
    if len(lidar_local_pose) >= 2:
        lidar_gap_max_s = max(
            float(curr["time_s"]) - float(prev["time_s"])
            for prev, curr in zip(lidar_local_pose[:-1], lidar_local_pose[1:])
        )

    straight_phase_metrics = _phase_pose_metrics(rows, "straight_open_loop")
    curve_entry_metrics = _phase_pose_metrics(rows, "curve_entry_probe")
    payload = {
        "available_series": {
            "imu_raw_pose": len(imu_raw_pose),
            "lidar_local_pose": len(lidar_local_pose),
            "fused_odom_pose": len(fused_odom_pose),
            "imu_filtered_heading": len(imu_filtered_heading),
            "kinematics_heading": len(kinematics_heading),
            "fusion_status": 0 if odom_payload is None else int((odom_payload.get("available_series") or {}).get("fusion_status", 0)),
        },
        "straight_phase": straight_phase_metrics,
        "curve_entry_phase": curve_entry_metrics,
        "imu_raw_vs_fused_terminal_delta_m": _series_endpoint_delta_m(imu_raw_pose, fused_odom_pose),
        "lidar_vs_fused_terminal_delta_m": _series_endpoint_delta_m(lidar_local_pose, fused_odom_pose),
        "lidar_vs_imu_raw_terminal_delta_m": _series_endpoint_delta_m(lidar_local_pose, imu_raw_pose),
        "fused_tail_disp_last_5s_m": None if odom_payload is None else odom_payload.get("fused_tail_disp_last_5s_m"),
        "imu_raw_tail_disp_last_5s_m": None if odom_payload is None else odom_payload.get("imu_raw_tail_disp_last_5s_m"),
        "lidar_pose_covariance_base_x_m2": lidar_cov_base_x,
        "lidar_pose_covariance_base_yaw_rad2": lidar_cov_base_yaw,
        "lidar_pose_degraded_ratio": float(
            mean(1.0 if sample["degraded"] else 0.0 for sample in lidar_degraded_samples)
        )
        if lidar_degraded_samples
        else None,
        "lidar_pose_gap_max_s": lidar_gap_max_s,
        "lidar_fresh_but_unobservable_ratio": None
        if odom_payload is None
        else odom_payload.get("lidar_fresh_but_unobservable_ratio"),
        "fusion_prediction_only_ratio": None
        if odom_payload is None
        else odom_payload.get("fusion_prediction_only_ratio"),
        "fusion_position_update_suppressed_ratio": None
        if odom_payload is None
        else odom_payload.get("fusion_position_update_suppressed_ratio"),
        "fusion_lidar_reacquire_count": 0
        if odom_payload is None
        else int(odom_payload.get("fusion_lidar_reacquire_count", 0) or 0),
        "fusion_lidar_reacquire_max_jump_m": None
        if odom_payload is None
        else odom_payload.get("fusion_lidar_reacquire_max_jump_m"),
        "fusion_lidar_position_gain_mean": None
        if odom_payload is None
        else odom_payload.get("fusion_lidar_position_gain_mean"),
        "fusion_lidar_velocity_gain_mean": None
        if odom_payload is None
        else odom_payload.get("fusion_lidar_velocity_gain_mean"),
    }

    json_path = output_dir / "sensor_fusion_compare.json"
    png_path = output_dir / "sensor_fusion_compare.png"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)

    ax_xy = axes[0][0]
    for label, color, samples in (
        ("IMU raw odom", "#d62728", imu_raw_pose),
        ("LiDAR local", "#2ca02c", lidar_local_pose),
        ("Fused odom", "#ff7f0e", fused_odom_pose),
    ):
        if not samples:
            continue
        ax_xy.plot(
            [float(sample["x_m"]) for sample in samples],
            [float(sample["y_m"]) for sample in samples],
            lw=2.0,
            color=color,
            label=label,
        )
        ax_xy.scatter([float(samples[0]["x_m"])], [float(samples[0]["y_m"])], s=20, color=color)
        ax_xy.scatter([float(samples[-1]["x_m"])], [float(samples[-1]["y_m"])], s=36, color=color, marker="x")
    ax_xy.set_title("Comparacion XY: imu_raw vs lidar_local vs fused")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.grid(True, alpha=0.25)
    ax_xy.axis("equal")
    ax_xy.legend(loc="best")

    ax_yaw = axes[0][1]
    for label, color, samples in (
        ("Yaw kinematics", "#1f77b4", _unwrapped_yaw_series(kinematics_heading)),
        ("Yaw IMU filtered", "#8c564b", _unwrapped_yaw_series(imu_filtered_heading)),
        ("Yaw LiDAR local", "#2ca02c", _unwrapped_yaw_series(lidar_local_pose)),
        ("Yaw fused", "#ff7f0e", _unwrapped_yaw_series(fused_odom_pose)),
    ):
        if not samples:
            continue
        ax_yaw.plot(
            [float(sample["time_s"]) for sample in samples],
            [float(sample["yaw_deg"]) for sample in samples],
            lw=1.9,
            color=color,
            label=label,
        )
    ax_yaw.set_title("Comparacion de yaw")
    ax_yaw.set_xlabel("Tiempo [s]")
    ax_yaw.set_ylabel("Yaw [deg]")
    ax_yaw.grid(True, alpha=0.25)
    ax_yaw.legend(loc="best")

    ax_speed = axes[1][0]
    if imu_raw_speed:
        ax_speed.plot(
            [float(sample["time_s"]) for sample in imu_raw_speed],
            [float(sample["speed_mps"]) for sample in imu_raw_speed],
            lw=1.8,
            color="#d62728",
            label="Velocidad imu_raw",
        )
    if fused_speed:
        ax_speed.plot(
            [float(sample["time_s"]) for sample in fused_speed],
            [float(sample["speed_mps"]) for sample in fused_speed],
            lw=1.8,
            color="#ff7f0e",
            label="Velocidad fused",
        )
    if kinematics_status:
        ax_speed_state = ax_speed.twinx()
        ax_speed_state.step(
            [float(sample["time_s"]) for sample in kinematics_status],
            [1.0 if sample["stationary_detected"] else 0.0 for sample in kinematics_status],
            where="post",
            lw=1.4,
            color="#2ca02c",
            label="stationary",
        )
        ax_speed_state.step(
            [float(sample["time_s"]) for sample in kinematics_status],
            [1.0 if sample["zupt_applied"] else 0.0 for sample in kinematics_status],
            where="post",
            lw=1.4,
            color="#9467bd",
            label="zupt",
        )
        ax_speed_state.set_ylim(-0.05, 1.05)
        ax_speed_state.set_ylabel("estado")
        lines_a, labels_a = ax_speed.get_legend_handles_labels()
        lines_b, labels_b = ax_speed_state.get_legend_handles_labels()
        ax_speed.legend(lines_a + lines_b, labels_a + labels_b, loc="best")
    else:
        ax_speed.legend(loc="best")
    ax_speed.set_title("Velocidad y reentrada a reposo")
    ax_speed.set_xlabel("Tiempo [s]")
    ax_speed.set_ylabel("m/s")
    ax_speed.grid(True, alpha=0.25)

    ax_text = axes[1][1]
    ax_text.axis("off")
    text_lines = [
        "Resumen fusion local",
        "",
        f"imu_raw vs fused final: {float(payload['imu_raw_vs_fused_terminal_delta_m'] or 0.0):.3f} m",
        f"lidar vs fused final: {float(payload['lidar_vs_fused_terminal_delta_m'] or 0.0):.3f} m",
        f"lidar gap max: {float(payload['lidar_pose_gap_max_s'] or 0.0):.3f} s",
        f"lidar degraded ratio: {float(payload['lidar_pose_degraded_ratio'] or 0.0):.3f}",
        f"imu_raw tail drift 5s: {float(payload['imu_raw_tail_disp_last_5s_m'] or 0.0):.3f} m",
        f"fused tail drift 5s: {float(payload['fused_tail_disp_last_5s_m'] or 0.0):.3f} m",
        f"lidar fresh+unobservable: {float(payload['lidar_fresh_but_unobservable_ratio'] or 0.0):.3f}",
        f"fusion prediction-only: {float(payload['fusion_prediction_only_ratio'] or 0.0):.3f}",
        f"lidar reacquire count: {int(payload['fusion_lidar_reacquire_count'] or 0)}",
        f"lidar reacquire max jump: {float(payload['fusion_lidar_reacquire_max_jump_m'] or 0.0):.3f} m",
    ]
    if straight_phase_metrics is not None:
        text_lines.extend(
            [
                "",
                "Recta abierta",
                f"dx={float(straight_phase_metrics['dx_m']):.3f} m",
                f"dy={float(straight_phase_metrics['dy_m']):.3f} m",
                f"disp={float(straight_phase_metrics['disp_m']):.3f} m",
                f"lateral_abs={float(straight_phase_metrics['lateral_error_m']):.3f} m",
                f"yaw_delta={float(straight_phase_metrics['yaw_delta_deg']):.3f} deg",
            ]
        )
    if curve_entry_metrics is not None:
        text_lines.extend(
            [
                "",
                "Curve entry",
                f"dx={float(curve_entry_metrics['dx_m']):.3f} m",
                f"dy={float(curve_entry_metrics['dy_m']):.3f} m",
                f"disp={float(curve_entry_metrics['disp_m']):.3f} m",
                f"yaw_delta={float(curve_entry_metrics['yaw_delta_deg']):.3f} deg",
            ]
        )
    ax_text.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.92, "edgecolor": "#666666"},
        transform=ax_text.transAxes,
    )

    fig.suptitle("Fusion local LiDAR + IMU", fontsize=14)
    fig.savefig(png_path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else bundle_dir / "analysis"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    run_metadata = load_json(bundle_dir / "run_metadata.json")
    config_records, summary_records, rows = parse_diag_log(bundle_dir / "recon_diagnostic.log")
    config = config_records[0] if config_records else {}
    flags = analyze_flags(rows, config, run_metadata)

    write_csv(rows, output_dir / "decision_timeline.csv")
    (output_dir / "flags.json").write_text(
        json.dumps(flags, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary(output_dir / "summary.md", run_metadata, config, flags, rows)
    write_curve_probe_artifacts(output_dir, rows)
    write_curve_tracking_comparison_artifact(output_dir, rows)
    series, odom_payload = write_odom_drift_artifact(bundle_dir, output_dir)
    write_sensor_fusion_compare_artifact(output_dir, rows, series, odom_payload)
    write_svg_plot(
        plots_dir / "headings.svg",
        rows,
        [
            ("gap_heading_deg", "#1f77b4"),
            ("front_turn_heading_deg", "#8c564b"),
            ("corridor_axis_heading_deg", "#bcbd22"),
            ("corridor_center_heading_deg", "#17becf"),
            ("wall_follow_heading_deg", "#7f7f7f"),
            ("centering_heading_deg", "#ff7f0e"),
            ("avoidance_heading_deg", "#d62728"),
            ("target_heading_deg", "#2ca02c"),
            ("steering_pre_servo_deg", "#9467bd"),
        ],
        "Heading Decisions",
    )
    write_svg_plot(
        plots_dir / "clearances.svg",
        rows,
        [
            ("front_clearance_m", "#1f77b4"),
            ("left_clearance_m", "#2ca02c"),
            ("right_clearance_m", "#d62728"),
            ("left_min_m", "#17becf"),
            ("right_min_m", "#8c564b"),
        ],
        "Scan Clearances",
    )
    if summary_records:
        (output_dir / "phase_summaries.json").write_text(
            json.dumps(summary_records, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
