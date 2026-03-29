#!/usr/bin/env python3
"""Detect a visible curve opening from a static LiDAR snapshot CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROS2_SRC_DIR = SCRIPT_DIR.parent.parent / "ros2_ws" / "src" / "apex_telemetry"
if str(ROS2_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(ROS2_SRC_DIR))

from apex_telemetry.perception.curve_window_detection import (  # noqa: E402
    CurveWindowDetectionConfig,
    CurveWindowDetectionResult,
    curve_window_result_summary,
    detect_curve_window_points,
)


def _load_snapshot_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angles_deg: list[float] = []
    ranges_m: list[float] = []
    counts: list[int] = []
    with csv_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            angle_deg = float(row["angle_deg"])
            try:
                range_m = float(row["range_m"])
            except ValueError:
                range_m = float("nan")
            count = int(row["count"])
            if not math.isfinite(range_m) or range_m <= 0.0 or count <= 0:
                continue
            angles_deg.append(angle_deg)
            ranges_m.append(range_m)
            counts.append(count)
    if not angles_deg:
        raise ValueError(f"No valid points found in {csv_path}")
    return (
        np.asarray(angles_deg, dtype=np.float64),
        np.asarray(ranges_m, dtype=np.float64),
        np.asarray(counts, dtype=np.int32),
    )


def _snapshot_display_xy_to_forward_left_xy(
    angles_deg: np.ndarray,
    ranges_m: np.ndarray,
    *,
    right_positive: bool,
) -> tuple[np.ndarray, np.ndarray]:
    angles_rad = np.radians(angles_deg)
    x_m = -ranges_m * np.cos(angles_rad)
    y_m = ranges_m * np.sin(angles_rad)
    if right_positive:
        y_m = -y_m
    return x_m, y_m


def _plot_analysis(
    *,
    output_path: Path,
    detection: CurveWindowDetectionResult,
) -> None:
    x_m = detection.points_x_m
    y_m = detection.points_y_m
    left_profile = detection.left_profile
    right_profile = detection.right_profile
    candidate = detection.candidate
    trajectory = detection.trajectory
    axis_limit_m = detection.axis_limit_m

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(x_m, y_m, s=10, c="#27c1d9", alpha=0.85, label="Puntos LiDAR")
    ax.scatter([0.0], [0.0], s=44, c="#d62728", label="LiDAR")

    info_lines = [
        "No se detecto una curva visible dominante.",
    ]

    if left_profile is not None:
        ax.plot(
            left_profile.x_m,
            left_profile.fit_y_m,
            linestyle="--",
            linewidth=1.5,
            color="#1f77b4",
            label="Recta base izquierda",
        )
        ax.plot(left_profile.x_m, left_profile.y_m, linewidth=1.2, color="#4c78a8", alpha=0.70)
    if right_profile is not None:
        ax.plot(
            right_profile.x_m,
            right_profile.fit_y_m,
            linestyle="--",
            linewidth=1.5,
            color="#2ca02c",
            label="Recta base derecha",
        )
        ax.plot(right_profile.x_m, right_profile.y_m, linewidth=1.2, color="#54a24b", alpha=0.70)

    if candidate is not None and trajectory is not None and left_profile is not None and right_profile is not None:
        profile = left_profile if candidate.side_name == "left" else right_profile
        opposite_profile = right_profile if candidate.side_name == "left" else left_profile
        side_label = "izquierda" if candidate.side_name == "left" else "derecha"
        cluster_slice = slice(candidate.cluster_start_idx, candidate.cluster_end_idx + 1)

        ax.axvline(
            candidate.straight_end_x_m,
            linestyle=":",
            linewidth=1.4,
            color="#6b6b6b",
            alpha=0.90,
            label="Fin pared recta",
        )
        ax.plot(
            profile.x_m[cluster_slice],
            profile.y_m[cluster_slice],
            linewidth=2.6,
            color="#ff7f0e",
            label=f"Borde curvo {side_label}",
        )
        ax.scatter(
            [candidate.first_curve_x_m],
            [candidate.first_curve_y_m],
            s=64,
            c="#9467bd",
            label="Primer punto curvo",
        )
        ax.scatter(
            [candidate.entry_x_m],
            [candidate.entry_y_m],
            s=70,
            c="#111111",
            marker="x",
            label="Inicio estimado",
        )
        ax.plot(
            trajectory.x_m,
            trajectory.y_m,
            linewidth=2.8,
            color="#d81b60",
            label="Trayectoria estimada",
        )
        ax.scatter(
            [point[0] for point in trajectory.anchor_points],
            [point[1] for point in trajectory.anchor_points],
            s=26,
            c="#d81b60",
            alpha=0.85,
        )
        ax.annotate(
            "",
            xy=(trajectory.x_m[-1], trajectory.y_m[-1]),
            xytext=(trajectory.x_m[-6], trajectory.y_m[-6]),
            arrowprops={"arrowstyle": "->", "color": "#d81b60", "linewidth": 2.4},
        )

        if candidate.gap_only_opening:
            ax.plot(
                [candidate.entry_x_m, candidate.window_far_wall_x_m],
                [candidate.entry_y_m, candidate.entry_y_m],
                linestyle=":",
                linewidth=2.0,
                color="#111111",
            )
            ax.text(
                0.5 * (candidate.entry_x_m + candidate.window_far_wall_x_m),
                candidate.entry_y_m + 0.04 * axis_limit_m,
                f"ancho ventana\n{candidate.window_width_m:.3f} m",
                fontsize=9,
                ha="center",
                va="bottom",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.90, "edgecolor": "#555555"},
            )
        else:
            opposite_y_m = float(np.polyval(opposite_profile.fit_coef, candidate.entry_x_m))
            ax.plot(
                [candidate.entry_x_m, candidate.entry_x_m],
                [candidate.entry_y_m, opposite_y_m],
                linestyle=":",
                linewidth=2.0,
                color="#111111",
            )
            ax.text(
                candidate.entry_x_m + 0.03,
                0.5 * (candidate.entry_y_m + opposite_y_m),
                f"ancho visible\n{candidate.entry_width_m:.3f} m",
                fontsize=9,
                ha="left",
                va="center",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.90, "edgecolor": "#555555"},
            )

        info_lines = [
            f"Curva visible: {side_label}",
            f"Inicio estimado: x={candidate.entry_x_m:.3f} m",
            f"Fin pared recta: x={candidate.straight_end_x_m:.3f} m",
            (
                f"Pared visible despues de ventana: x={candidate.window_far_wall_x_m:.3f} m"
                if candidate.gap_only_opening
                else f"Continuidad pared opuesta hasta x={candidate.opposite_wall_visible_until_x_m:.3f} m"
            ),
            f"Trayectoria apunta a x={trajectory.target_x_m:.3f} m, y={trajectory.target_y_m:.3f} m",
            f"Distancia radial al inicio: {candidate.start_radial_m:.3f} m",
            f"Ancho recto base: {candidate.straight_width_m:.3f} m",
            (
                f"Ancho ventana: {candidate.window_width_m:.3f} m"
                if candidate.gap_only_opening
                else f"Ancho visible en la entrada: {candidate.entry_width_m:.3f} m"
            ),
            f"Apertura lateral de curva: {candidate.curve_wall_shift_m:.3f} m",
            f"Ganancia de ancho en curva: {candidate.opening_width_gain_m:.3f} m",
            f"Gap misma pared: {candidate.same_side_gap_m:.3f} m",
            f"Cierre frontal interior: {candidate.front_closure_point_count} pts",
            f"Modo apertura: {'ventana sin puntos' if candidate.gap_only_opening else 'borde detectado'}",
            f"Sector angular visible: {candidate.angle_start_deg:.1f}..{candidate.angle_end_deg:.1f} deg",
            f"Puntos usados en curva: {candidate.curve_point_count}",
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
    ax.set_title("Analisis de curva visible desde snapshot LiDAR")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="",
        help="Snapshot CSV generated by lidar_subscriber_node.py",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path_flag",
        default="",
        help="Snapshot CSV generated by lidar_subscriber_node.py",
    )
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Base path for output files (.png and .json). Defaults next to the CSV.",
    )
    parser.add_argument(
        "--snapshot-right-positive",
        action="store_true",
        default=False,
        help="Interpret the snapshot display angles as derecha+ / izquierda-.",
    )
    parser.add_argument(
        "--snapshot-left-positive",
        action="store_true",
        help="Interpret the snapshot display angles as izquierda+ / derecha- (default).",
    )
    parser.add_argument("--x-bin-m", type=float, default=0.05, help="Bin width along x")
    parser.add_argument("--fit-x-min-m", type=float, default=-0.75, help="Start x of straight fit")
    parser.add_argument("--fit-x-max-m", type=float, default=0.05, help="End x of straight fit")
    parser.add_argument(
        "--search-x-min-m",
        type=float,
        default=0.15,
        help="Start x for curve search after the straight segment",
    )
    parser.add_argument(
        "--deviation-threshold-m",
        type=float,
        default=0.12,
        help="Minimum side-wall deviation required to tag a visible curve",
    )
    parser.add_argument(
        "--min-points-per-bin",
        type=int,
        default=2,
        help="Minimum points per x-bin on one side",
    )
    parser.add_argument(
        "--min-curve-bins",
        type=int,
        default=2,
        help="Minimum consecutive bins to accept a curve candidate",
    )
    parser.add_argument(
        "--gap-threshold-m",
        type=float,
        default=0.11,
        help="Minimum x gap in one side-wall profile to mark a possible opening",
    )
    parser.add_argument(
        "--opposite-continuation-min-m",
        type=float,
        default=0.10,
        help="Minimum extra visible length required on the opposite wall after the entry point",
    )
    parser.add_argument(
        "--front-closure-x-window-m",
        type=float,
        default=0.12,
        help="Half-window in x used to reject entries that actually look like a frontal closing wall",
    )
    parser.add_argument(
        "--front-closure-min-points",
        type=int,
        default=6,
        help="Minimum interior points near the mouth to reject the candidate as front closure",
    )
    args = parser.parse_args()
    if not args.csv_path and args.csv_path_flag:
        args.csv_path = args.csv_path_flag
    if not args.csv_path:
        parser.error("csv_path is required")
    if args.snapshot_left_positive:
        args.snapshot_right_positive = False
    return args


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path).expanduser().resolve()
    angles_deg, ranges_m, _counts = _load_snapshot_csv(csv_path)
    x_m, y_m = _snapshot_display_xy_to_forward_left_xy(
        angles_deg,
        ranges_m,
        right_positive=bool(args.snapshot_right_positive),
    )

    config = CurveWindowDetectionConfig(
        x_bin_m=float(args.x_bin_m),
        fit_x_min_m=float(args.fit_x_min_m),
        fit_x_max_m=float(args.fit_x_max_m),
        search_x_min_m=float(args.search_x_min_m),
        deviation_threshold_m=float(args.deviation_threshold_m),
        min_points_per_bin=int(args.min_points_per_bin),
        min_curve_bins=int(args.min_curve_bins),
        gap_threshold_m=float(args.gap_threshold_m),
        opposite_continuation_min_m=float(args.opposite_continuation_min_m),
        front_closure_x_window_m=float(args.front_closure_x_window_m),
        front_closure_min_points=int(args.front_closure_min_points),
    )
    detection = detect_curve_window_points(
        x_m,
        y_m,
        angles_deg=angles_deg,
        config=config,
    )

    if args.output_prefix:
        output_prefix = Path(args.output_prefix).expanduser().resolve()
    else:
        output_prefix = csv_path.with_suffix("")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_png = output_prefix.with_name(output_prefix.name + "_curve_analysis").with_suffix(".png")
    output_json = output_prefix.with_name(output_prefix.name + "_curve_analysis").with_suffix(".json")

    _plot_analysis(
        output_path=output_png,
        detection=detection,
    )

    result = {
        "csv_path": str(csv_path),
        "output_png": str(output_png),
        "output_json": str(output_json),
        "point_count": int(ranges_m.size),
        "input_convention": {
            "snapshot_right_positive": bool(args.snapshot_right_positive),
            "internal_frame": "x_forward_y_left",
        },
        "curve_candidate": None,
    }
    if detection.valid:
        summary = curve_window_result_summary(detection)
        summary.update(
            {
                "entry_width_m": float(detection.candidate.entry_width_m),
                "straight_width_m": float(detection.candidate.straight_width_m),
                "curve_wall_shift_m": float(detection.candidate.curve_wall_shift_m),
                "curve_lateral_opening_m": float(detection.candidate.curve_wall_shift_m),
                "curve_width_m": float(detection.candidate.curve_width_m),
                "opening_width_gain_m": float(detection.candidate.opening_width_gain_m),
                "front_closure_point_count": int(detection.candidate.front_closure_point_count),
                "front_closure_y_span_m": float(detection.candidate.front_closure_y_span_m),
                "start_forward_m": float(detection.candidate.start_forward_m),
                "start_radial_m": float(detection.candidate.start_radial_m),
                "angle_start_deg": float(detection.candidate.angle_start_deg),
                "angle_end_deg": float(detection.candidate.angle_end_deg),
                "angle_center_deg": float(detection.candidate.angle_center_deg),
                "curve_point_count": int(detection.candidate.curve_point_count),
                "first_curve_point_x_m": float(detection.candidate.first_curve_x_m),
                "first_curve_point_y_m": float(detection.candidate.first_curve_y_m),
            }
        )
        result["curve_candidate"] = summary

    output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[APEX] Saved curve analysis image: {output_png}")
    print(f"[APEX] Saved curve analysis metrics: {output_json}")
    if not detection.valid:
        print("[APEX] No dominant visible curve candidate was detected.")
    else:
        candidate = detection.candidate
        assert candidate is not None
        side_label = "izquierda" if candidate.side_name == "left" else "derecha"
        print(
            "[APEX] Curve candidate: side=%s start_forward=%.3fm window_width=%.3fm target=(%.3f, %.3f) heuristic=%s"
            % (
                side_label,
                candidate.start_forward_m,
                candidate.window_width_m,
                detection.trajectory.target_x_m,
                detection.trajectory.target_y_m,
                candidate.heuristic_name,
            )
        )


if __name__ == "__main__":
    main()
