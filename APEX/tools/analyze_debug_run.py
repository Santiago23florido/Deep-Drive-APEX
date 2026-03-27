#!/usr/bin/env python3
"""Analyze one APEX debug bundle and produce artifacts for offline review."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable


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
            if row.get("active_heading_source") != "avoidance":
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
        and str(row.get("nav_mode")) not in {"curve_capture", "curve_entry", "curve_follow"}
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
            nav_mode in {"curve_capture", "curve_entry", "curve_follow"}
            and steering_floor_deg > 0.0
            and steering_deg + 1e-6 < steering_floor_deg
        ):
            steering_below_curve_floor_rows.append(row)
        if (
            row.get("curve_gate_open")
            and gate_curve_sign != 0
            and nav_mode not in {"curve_capture", "curve_entry", "curve_follow", "curve_exit"}
            and speed_pct > 0.0
        ):
            straight_through_curve_rows.append(row)

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
