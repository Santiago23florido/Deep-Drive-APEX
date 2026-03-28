#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


SOURCE_COLORS = {
    "front_turn": "#2ca02c",
    "turn_commit": "#9467bd",
    "gap": "#ff7f0e",
    "gap_escape": "#d62728",
    "avoidance": "#1f77b4",
    "corridor_center": "#17becf",
    "trajectory_planner": "#bcbd22",
    "fullsoft_max_space": "#bcbd22",
    "fullsoft_corner_adjust": "#8c564b",
    "fullsoft_bias_blend": "#17becf",
    "fullsoft_continuity_gate": "#ff7f0e",
    "fullsoft_curve_preview": "#2ca02c",
    "fullsoft_curve_ramp": "#9467bd",
    "fullsoft_forward_gate": "#1f77b4",
    "fullsoft_straight_gate": "#17becf",
    "fullsoft_stop": "#d62728",
    "fullsoft_curve": "#9467bd",
    "None": "#7f7f7f",
}


def safe_float(value: str | None) -> float | None:
    if value in (None, "", "None", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sign_label(value: float | None) -> str:
    if value is None:
        return "none"
    if value > 0.0:
        return "left"
    if value < 0.0:
        return "right"
    return "center"


def load_timeline(bundle_dir: Path) -> list[dict[str, str]]:
    timeline_path = bundle_dir / "analysis" / "decision_timeline.csv"
    if not timeline_path.exists():
        raise FileNotFoundError(f"Missing analysis timeline: {timeline_path}")
    with timeline_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def autonomous_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    allowed_steps = {
        "autonomous",
        "dryrun",
        "curve_static_detect",
        "curve_entry_detect",
        "curve_entry_track",
    }
    allowed_phases = {
        "curve_static_probe",
        "curve_entry_probe",
    }
    return [
        row
        for row in rows
        if (row.get("step") in allowed_steps) or (row.get("phase") in allowed_phases)
    ]


def find_first_opposite_sign_cycle(nav_rows: list[dict[str, str]]) -> int | None:
    initial_sign = None
    for row in nav_rows:
        target = safe_float(row.get("target_heading_deg"))
        if target is None or abs(target) < 1e-6:
            continue
        initial_sign = math.copysign(1.0, target)
        break
    if initial_sign is None:
        return None

    consecutive = []
    for row in nav_rows:
        target = safe_float(row.get("target_heading_deg"))
        if target is None or abs(target) < 1e-6:
            consecutive.clear()
            continue
        current_sign = math.copysign(1.0, target)
        if current_sign != initial_sign:
            consecutive.append(int(row["cycle"]))
            if len(consecutive) >= 3:
                return consecutive[0]
        else:
            consecutive.clear()
    return None


def find_first_sustained_source(
    nav_rows: list[dict[str, str]], source_name: str, min_consecutive: int = 3
) -> int | None:
    consecutive = []
    for row in nav_rows:
        if row.get("active_heading_source") == source_name:
            consecutive.append(int(row["cycle"]))
            if len(consecutive) >= min_consecutive:
                return consecutive[0]
        else:
            consecutive.clear()
    return None


def contiguous_source_segments(nav_rows: list[dict[str, str]]) -> list[tuple[int, int, str]]:
    if not nav_rows:
        return []
    segments: list[tuple[int, int, str]] = []
    current_source = nav_rows[0].get("active_heading_source") or "None"
    start_cycle = int(nav_rows[0]["cycle"])
    prev_cycle = start_cycle
    for row in nav_rows[1:]:
        cycle = int(row["cycle"])
        source = row.get("active_heading_source") or "None"
        if source != current_source:
            segments.append((start_cycle, prev_cycle, current_source))
            current_source = source
            start_cycle = cycle
        prev_cycle = cycle
    segments.append((start_cycle, prev_cycle, current_source))
    return segments


def column(nav_rows: list[dict[str, str]], key: str) -> list[float | None]:
    return [safe_float(row.get(key)) for row in nav_rows]


def explain_text(bundle_name: str, nav_rows: list[dict[str, str]]) -> str:
    source_counts = Counter((row.get("active_heading_source") or "None") for row in nav_rows)
    first_flip = find_first_opposite_sign_cycle(nav_rows)
    first_gap = find_first_sustained_source(nav_rows, "gap")
    corridor_present = any(
        safe_float(row.get("corridor_center_heading_deg")) is not None for row in nav_rows
    )

    def avg_for_range(start: int, end: int, key: str) -> float | None:
        values = [
            safe_float(row.get(key))
            for row in nav_rows
            if start <= int(row["cycle"]) <= end and safe_float(row.get(key)) is not None
        ]
        return mean(values) if values else None

    parts = [f"# Recon Explanation: {bundle_name}", "", "## Reading"]
    parts.append(
        f"- Source counts: {dict(source_counts)}"
    )
    parts.append(
        f"- First sustained sign flip cycle: {first_flip if first_flip is not None else 'none'}"
    )
    parts.append(
        f"- First sustained `gap` takeover cycle: {first_gap if first_gap is not None else 'none'}"
    )
    parts.append(
        f"- Corridor diagnostics present: {'yes' if corridor_present else 'no'}"
    )
    parts.extend(["", "## Interpretation"])

    if not nav_rows:
        parts.append("- No autonomous rows found.")
        return "\n".join(parts)

    early_start = int(nav_rows[0]["cycle"])
    early_end = min(early_start + 20, int(nav_rows[-1]["cycle"]))
    early_target = avg_for_range(early_start, early_end, "target_heading_deg")
    early_steer = avg_for_range(early_start, early_end, "steering_pre_servo_deg")
    if early_target is not None and early_steer is not None:
        parts.append(
            f"- Entry phase: average target `{early_target:.2f} deg` ({sign_label(early_target)}), "
            f"average steering `{early_steer:.2f} deg`."
        )

    if first_flip is not None:
        flip_target = avg_for_range(first_flip, first_flip + 8, "target_heading_deg")
        flip_front = avg_for_range(first_flip, first_flip + 8, "front_turn_heading_deg")
        flip_left = avg_for_range(first_flip, first_flip + 8, "front_left_clearance_m")
        flip_right = avg_for_range(first_flip, first_flip + 8, "front_right_clearance_m")
        parts.append(
            f"- Around cycle `{first_flip}`, the target changes sign. In that window, "
            f"`front_left_clearance_m` drops to about `{flip_left:.3f}` while "
            f"`front_right_clearance_m` rises to about `{flip_right:.3f}`, so the logic "
            f"starts reading the outside opening more strongly than the inside continuation."
        )
        if flip_target is not None and flip_front is not None:
            parts.append(
                f"- That produces `target_heading_deg ≈ {flip_target:.2f}` and "
                f"`front_turn_heading_deg ≈ {flip_front:.2f}`."
            )

    if first_gap is not None:
        gap_target = avg_for_range(first_gap, first_gap + 20, "target_heading_deg")
        gap_steer = avg_for_range(first_gap, first_gap + 20, "steering_pre_servo_deg")
        gap_heading = avg_for_range(first_gap, first_gap + 20, "gap_heading_deg")
        parts.append(
            f"- From cycle `{first_gap}`, `gap` dominates. Its average heading is "
            f"`{gap_heading:.2f} deg`, which pushes the car toward the open side instead of "
            f"holding the corridor axis. That is the main reason the real trajectory opens out."
        )
        parts.append(
            f"- In the same window, the steering average is `{gap_steer:.2f} deg` for a "
            f"`target_heading_deg` average of `{gap_target:.2f} deg`."
        )

    if not corridor_present:
        parts.append(
            "- This bundle does not contain `corridor_*` diagnostics, so it was not running "
            "the new corridor-axis logic yet."
        )

    return "\n".join(parts)


def plot_bundle(bundle_dir: Path, nav_rows: list[dict[str, str]]) -> Path:
    analysis_dir = bundle_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    output_path = analysis_dir / "trajectory_explainer.png"

    cycles = [int(row["cycle"]) for row in nav_rows]
    if not cycles:
        raise RuntimeError("No autonomous rows found in analysis timeline.")

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True, constrained_layout=True)

    segments = contiguous_source_segments(nav_rows)
    for ax in axes:
        for start_cycle, end_cycle, source in segments:
            ax.axvspan(
                start_cycle,
                end_cycle,
                color=SOURCE_COLORS.get(source, "#cccccc"),
                alpha=0.08,
                lw=0,
            )

    left = column(nav_rows, "left_clearance_m")
    right = column(nav_rows, "right_clearance_m")
    front_left = column(nav_rows, "front_left_clearance_m")
    front_right = column(nav_rows, "front_right_clearance_m")
    front = column(nav_rows, "effective_front_clearance_m")

    axes[0].plot(cycles, left, label="left_clearance", color="#2ca02c", lw=2.0)
    axes[0].plot(cycles, right, label="right_clearance", color="#d62728", lw=2.0)
    axes[0].plot(cycles, front_left, label="front_left", color="#1f77b4", lw=1.6)
    axes[0].plot(cycles, front_right, label="front_right", color="#9467bd", lw=1.6)
    axes[0].plot(cycles, front, label="effective_front", color="#7f7f7f", lw=1.4, ls="--")
    axes[0].set_ylabel("Clearance [m]")
    axes[0].set_title("Perception that drives the controller")
    axes[0].legend(loc="upper right", ncol=3)
    axes[0].grid(alpha=0.25)

    axes[1].plot(cycles, column(nav_rows, "avoidance_heading_deg"), label="avoidance", color="#1f77b4")
    axes[1].plot(cycles, column(nav_rows, "front_turn_heading_deg"), label="front_turn", color="#8c564b")
    axes[1].plot(cycles, column(nav_rows, "gap_heading_deg"), label="gap", color="#ff7f0e")
    if any(safe_float(row.get("corridor_center_heading_deg")) is not None for row in nav_rows):
        axes[1].plot(
            cycles,
            column(nav_rows, "corridor_center_heading_deg"),
            label="corridor_center",
            color="#17becf",
        )
    axes[1].plot(cycles, column(nav_rows, "target_heading_deg"), label="target", color="black", lw=2.0)
    axes[1].axhline(0.0, color="#555555", lw=1.0, ls=":")
    axes[1].set_ylabel("Heading [deg]")
    axes[1].set_title("Candidate headings vs chosen target")
    axes[1].legend(loc="upper right", ncol=4)
    axes[1].grid(alpha=0.25)

    steering = column(nav_rows, "steering_pre_servo_deg")
    speed = column(nav_rows, "speed_pct")
    axes[2].plot(cycles, steering, label="steering_pre_servo", color="#2ca02c", lw=2.0)
    ax2b = axes[2].twinx()
    ax2b.plot(cycles, speed, label="speed_pct", color="#d62728", lw=1.7)
    axes[2].axhline(0.0, color="#555555", lw=1.0, ls=":")
    axes[2].set_ylabel("Steering [deg]")
    ax2b.set_ylabel("Speed [%]")
    axes[2].set_title("Issued steering and speed")
    lines_a, labels_a = axes[2].get_legend_handles_labels()
    lines_b, labels_b = ax2b.get_legend_handles_labels()
    axes[2].legend(lines_a + lines_b, labels_a + labels_b, loc="upper right")
    axes[2].grid(alpha=0.25)

    front_bias = []
    side_bias = []
    for row in nav_rows:
        left_v = safe_float(row.get("left_clearance_m"))
        right_v = safe_float(row.get("right_clearance_m"))
        front_left_v = safe_float(row.get("front_left_clearance_m"))
        front_right_v = safe_float(row.get("front_right_clearance_m"))
        side_bias.append((left_v - right_v) if left_v is not None and right_v is not None else None)
        front_bias.append(
            (front_left_v - front_right_v)
            if front_left_v is not None and front_right_v is not None
            else None
        )
    axes[3].plot(cycles, side_bias, label="left-right clearance bias", color="#2ca02c", lw=2.0)
    axes[3].plot(cycles, front_bias, label="front-left minus front-right", color="#8c564b", lw=2.0)
    axes[3].axhline(0.0, color="#555555", lw=1.0, ls=":")
    axes[3].set_ylabel("Bias [m]")
    axes[3].set_xlabel("Cycle")
    axes[3].set_title("Why the sign changes: corridor bias seen by the logic")
    axes[3].legend(loc="upper right")
    axes[3].grid(alpha=0.25)

    first_flip = find_first_opposite_sign_cycle(nav_rows)
    first_gap = find_first_sustained_source(nav_rows, "gap")
    for ax in axes:
        if first_flip is not None:
            ax.axvline(first_flip, color="#d62728", ls="--", lw=1.5)
        if first_gap is not None:
            ax.axvline(first_gap, color="#ff7f0e", ls="--", lw=1.5)

    fig.suptitle(
        f"Recon behavior explainer: {bundle_dir.name}\n"
        "Shaded bands show active heading source. Positive heading = left, negative = right."
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def write_markdown(bundle_dir: Path, nav_rows: list[dict[str, str]], image_path: Path) -> Path:
    markdown_path = bundle_dir / "analysis" / "trajectory_explainer.md"
    markdown_path.write_text(
        explain_text(bundle_dir.name, nav_rows)
        + "\n\n## Output\n"
        + f"- Plot: `{image_path.name}`\n",
        encoding="utf-8",
    )
    return markdown_path


def write_empty_markdown(bundle_dir: Path, reason: str) -> Path:
    markdown_path = bundle_dir / "analysis" / "trajectory_explainer.md"
    markdown_path.write_text(
        "## No Autonomous Timeline\n"
        f"- {reason}\n",
        encoding="utf-8",
    )
    return markdown_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an explanation plot for a reconnaissance debug bundle."
    )
    parser.add_argument("bundle_dir", type=Path, help="Path to the debug run bundle directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_timeline(args.bundle_dir)
    nav_rows = autonomous_rows(rows)
    if not nav_rows:
        markdown_path = write_empty_markdown(
            args.bundle_dir,
            "No se encontraron filas compatibles para el explicador en este bundle.",
        )
        print(markdown_path)
        return
    image_path = plot_bundle(args.bundle_dir, nav_rows)
    markdown_path = write_markdown(args.bundle_dir, nav_rows, image_path)
    print(image_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
