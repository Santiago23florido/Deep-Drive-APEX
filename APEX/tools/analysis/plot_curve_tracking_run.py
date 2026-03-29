#!/usr/bin/env python3
"""Render one curve-tracking run as a single corridor/path/trajectory image."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_planned_path(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = _load_json(path)
    points = np.asarray(payload.get("path_xy_yaw", []), dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
        raise ValueError(f"Missing path points in {path}")
    return points[:, :2], points[:, 2]


def _load_tracking_csv(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                x_m = float(row["x_m"])
                y_m = float(row["y_m"])
            except (KeyError, ValueError):
                continue
            rows.append([x_m, y_m])
    if not rows:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _load_lidar_world_points(path: Path, *, max_points: int = 30000) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 2), dtype=np.float64)
    rows: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                x_m = float(row["x_world_m"])
                y_m = float(row["y_world_m"])
            except (KeyError, ValueError):
                continue
            if not (math.isfinite(x_m) and math.isfinite(y_m)):
                continue
            rows.append([x_m, y_m])
    if not rows:
        return np.empty((0, 2), dtype=np.float64)
    points = np.asarray(rows, dtype=np.float64)
    if points.shape[0] <= max_points:
        return points
    stride = max(1, int(math.ceil(points.shape[0] / max_points)))
    return points[::stride]


def _compute_path_yaw_from_xy(path_xy: np.ndarray) -> np.ndarray:
    if path_xy.shape[0] == 1:
        return np.zeros((1,), dtype=np.float64)
    yaw = np.zeros((path_xy.shape[0],), dtype=np.float64)
    diffs = np.diff(path_xy, axis=0)
    seg_yaw = np.arctan2(diffs[:, 1], diffs[:, 0])
    yaw[:-1] = seg_yaw
    yaw[-1] = seg_yaw[-1]
    return yaw


def _compute_corridor_polygon(path_xy: np.ndarray, path_yaw: np.ndarray, width_m: float) -> np.ndarray:
    half_width_m = max(0.05, 0.5 * float(width_m))
    left_points = []
    right_points = []
    for (x_m, y_m), yaw_rad in zip(path_xy, path_yaw):
        normal_left = np.asarray(
            [-math.sin(float(yaw_rad)), math.cos(float(yaw_rad))],
            dtype=np.float64,
        )
        left_points.append(np.asarray([x_m, y_m], dtype=np.float64) + (half_width_m * normal_left))
        right_points.append(np.asarray([x_m, y_m], dtype=np.float64) - (half_width_m * normal_left))
    return np.vstack([np.asarray(left_points), np.asarray(right_points)[::-1]])


def _compute_corridor_edges(
    path_xy: np.ndarray,
    path_yaw: np.ndarray,
    width_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    half_width_m = max(0.05, 0.5 * float(width_m))
    left_points = []
    right_points = []
    for (x_m, y_m), yaw_rad in zip(path_xy, path_yaw):
        normal_left = np.asarray(
            [-math.sin(float(yaw_rad)), math.cos(float(yaw_rad))],
            dtype=np.float64,
        )
        base_xy = np.asarray([x_m, y_m], dtype=np.float64)
        left_points.append(base_xy + (half_width_m * normal_left))
        right_points.append(base_xy - (half_width_m * normal_left))
    return np.asarray(left_points, dtype=np.float64), np.asarray(right_points, dtype=np.float64)


def _plot_run(
    *,
    run_dir: Path,
    planned_path_xy: np.ndarray,
    planned_path_yaw: np.ndarray,
    tracking_xy: np.ndarray,
    lidar_world_xy: np.ndarray,
    summary: dict,
    output_path: Path,
) -> None:
    planner_status = summary.get("planner_status") or {}
    curve_summary = planner_status.get("curve_summary") or {}
    planner_pose = planner_status.get("planner_pose") or {}
    target_pose = summary.get("target_pose") or planner_status.get("target_pose") or {}
    entry_line_center_pose = planner_status.get("entry_line_center_pose") or {}
    second_corridor_axis_xy = planner_status.get("second_corridor_axis_xy") or {}
    target_depth_into_second_corridor_m = planner_status.get(
        "target_depth_into_second_corridor_m"
    )

    corridor_width_m = float(curve_summary.get("window_width_m") or 0.8)
    corridor_width_m = max(0.25, min(2.5, corridor_width_m))
    path_yaw = (
        planned_path_yaw
        if planned_path_yaw.size == planned_path_xy.shape[0]
        else _compute_path_yaw_from_xy(planned_path_xy)
    )
    corridor_polygon = _compute_corridor_polygon(planned_path_xy, path_yaw, corridor_width_m)
    left_wall_xy, right_wall_xy = _compute_corridor_edges(planned_path_xy, path_yaw, corridor_width_m)

    fig, ax = plt.subplots(figsize=(10.5, 8.0))

    if lidar_world_xy.shape[0] > 0:
        ax.scatter(
            lidar_world_xy[:, 0],
            lidar_world_xy[:, 1],
            s=4,
            c="#8c8c8c",
            alpha=0.18,
            linewidths=0.0,
            label=f"LiDAR agregado ({lidar_world_xy.shape[0]} pts)",
            zorder=1,
        )

    ax.fill(
        corridor_polygon[:, 0],
        corridor_polygon[:, 1],
        color="#5ec2a5",
        alpha=0.14,
        label=f"Corredor estimado ({corridor_width_m:.2f} m)",
        zorder=2,
    )
    ax.plot(
        left_wall_xy[:, 0],
        left_wall_xy[:, 1],
        color="#2ca25f",
        linestyle="--",
        linewidth=1.7,
        label="Pared izq estimada",
        zorder=3,
    )
    ax.plot(
        right_wall_xy[:, 0],
        right_wall_xy[:, 1],
        color="#2ca25f",
        linestyle=":",
        linewidth=1.7,
        label="Pared der estimada",
        zorder=3,
    )
    ax.plot(
        planned_path_xy[:, 0],
        planned_path_xy[:, 1],
        color="#1f77b4",
        linewidth=2.4,
        label="Path planificado",
        zorder=4,
    )
    ax.scatter(
        [planned_path_xy[0, 0]],
        [planned_path_xy[0, 1]],
            c="#00a676",
            s=60,
            marker="o",
            label="Inicio path",
            zorder=5,
        )
    ax.scatter(
        [planned_path_xy[-1, 0]],
        [planned_path_xy[-1, 1]],
            c="#d81b60",
            s=70,
            marker="X",
            label="Fin path",
            zorder=5,
        )

    if tracking_xy.shape[0] > 0:
        ax.plot(
            tracking_xy[:, 0],
            tracking_xy[:, 1],
            color="#111111",
            linewidth=2.2,
            label="Trayectoria seguida",
            zorder=6,
        )
        ax.scatter(
            [tracking_xy[0, 0]],
            [tracking_xy[0, 1]],
            c="#111111",
            s=40,
            marker="o",
        )
        ax.scatter(
            [tracking_xy[-1, 0]],
            [tracking_xy[-1, 1]],
            c="#ff7f0e",
            s=55,
            marker="D",
            label="Fin seguimiento",
            zorder=7,
        )
    else:
        ax.text(
            0.02,
            0.14,
            "Este run no llego a registrar trayectoria seguida.\nEl tracker aborto antes de moverse o antes de publicar poses de tracking.",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#666666",
                "alpha": 0.92,
            },
        )

    if lidar_world_xy.shape[0] == 0:
        ax.text(
            0.02,
            0.03,
            "Este bundle no incluye lidar_points.csv.\nLas paredes visibles aqui son estimadas desde el corredor detectado, no la nube real del LiDAR.",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.30",
                "facecolor": "white",
                "edgecolor": "#999999",
                "alpha": 0.90,
            },
        )

    if planner_pose:
        ax.scatter(
            [float(planner_pose.get("x_m", 0.0))],
            [float(planner_pose.get("y_m", 0.0))],
            c="#7f3c8d",
            s=55,
            marker="P",
            label="Pose al planificar",
            zorder=7,
        )
    if entry_line_center_pose:
        entry_center_xy = np.asarray(
            [
                float(entry_line_center_pose.get("x_m", 0.0)),
                float(entry_line_center_pose.get("y_m", 0.0)),
            ],
            dtype=np.float64,
        )
        axis_xy = np.asarray(
            [
                float(second_corridor_axis_xy.get("x", 0.0)),
                float(second_corridor_axis_xy.get("y", 0.0)),
            ],
            dtype=np.float64,
        )
        axis_norm = float(np.linalg.norm(axis_xy))
        if axis_norm > 1.0e-9:
            axis_xy /= axis_norm
            entry_normal_xy = np.asarray([-axis_xy[1], axis_xy[0]], dtype=np.float64)
            line_half_width_m = max(0.18, 0.45 * corridor_width_m)
            line_xy = np.vstack(
                [
                    entry_center_xy - (line_half_width_m * entry_normal_xy),
                    entry_center_xy + (line_half_width_m * entry_normal_xy),
                ]
            )
            axis_xy_plot = np.vstack(
                [
                    entry_center_xy,
                    entry_center_xy + (0.32 * axis_xy),
                ]
            )
            ax.plot(
                line_xy[:, 0],
                line_xy[:, 1],
                color="#9467bd",
                linestyle="--",
                linewidth=1.6,
                label="Entry line",
                zorder=6,
            )
            ax.plot(
                axis_xy_plot[:, 0],
                axis_xy_plot[:, 1],
                color="#9467bd",
                linestyle="-.",
                linewidth=1.4,
                label="Eje segundo corredor",
                zorder=6,
            )
            ax.scatter(
                [entry_center_xy[0]],
                [entry_center_xy[1]],
                c="#9467bd",
                s=42,
                marker="s",
                label="Centro entry line",
                zorder=7,
            )
            if target_pose:
                target_xy = np.asarray(
                    [
                        float(target_pose.get("x_m", 0.0)),
                        float(target_pose.get("y_m", 0.0)),
                    ],
                    dtype=np.float64,
                )
                ax.plot(
                    [entry_center_xy[0], target_xy[0]],
                    [entry_center_xy[1], target_xy[1]],
                    color="#ff7f0e",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.85,
                    zorder=6,
                )
                if target_depth_into_second_corridor_m is not None:
                    text_xy = 0.5 * (entry_center_xy + target_xy)
                    ax.text(
                        float(text_xy[0]),
                        float(text_xy[1]),
                        f"depth={float(target_depth_into_second_corridor_m):.2f} m",
                        fontsize=8,
                        color="#7a4b00",
                        ha="left",
                        va="bottom",
                        bbox={
                            "boxstyle": "round,pad=0.20",
                            "facecolor": "white",
                            "edgecolor": "#d9a441",
                            "alpha": 0.8,
                        },
                        zorder=8,
                    )
    if target_pose:
        ax.scatter(
            [float(target_pose.get("x_m", 0.0))],
            [float(target_pose.get("y_m", 0.0))],
            c="#d62728",
            s=65,
            marker="*",
            label="Objetivo",
            zorder=7,
        )

    all_points = [planned_path_xy, corridor_polygon, left_wall_xy, right_wall_xy]
    if tracking_xy.shape[0] > 0:
        all_points.append(tracking_xy)
    if lidar_world_xy.shape[0] > 0:
        all_points.append(lidar_world_xy)
    if planner_pose:
        all_points.append(
            np.asarray([[float(planner_pose.get("x_m", 0.0)), float(planner_pose.get("y_m", 0.0))]])
        )
    if target_pose:
        all_points.append(
            np.asarray([[float(target_pose.get("x_m", 0.0)), float(target_pose.get("y_m", 0.0))]])
        )
    if entry_line_center_pose:
        all_points.append(
            np.asarray(
                [[float(entry_line_center_pose.get("x_m", 0.0)), float(entry_line_center_pose.get("y_m", 0.0))]]
            )
        )
    stack = np.vstack(all_points)
    pad_x = max(0.15, 0.08 * float(np.max(stack[:, 0]) - np.min(stack[:, 0]) + 1e-6))
    pad_y = max(0.15, 0.08 * float(np.max(stack[:, 1]) - np.min(stack[:, 1]) + 1e-6))
    ax.set_xlim(float(np.min(stack[:, 0]) - pad_x), float(np.max(stack[:, 0]) + pad_x))
    ax.set_ylim(float(np.min(stack[:, 1]) - pad_y), float(np.max(stack[:, 1]) + pad_y))

    info_lines = [
        f"run: {run_dir.name}",
        f"fin: {summary.get('end_cause', 'unknown')}",
        f"path points: {summary.get('path_point_count', 0)}",
        f"trayectoria points: {summary.get('trajectory_row_count', 0)}",
        f"lidar points: {summary.get('lidar_point_count', 0)}",
        f"curva: {curve_summary.get('side_label_es', curve_summary.get('side', 'n/a'))}",
        f"score curva: {curve_summary.get('score', 'n/a')}",
    ]
    goal_distance_m = summary.get("goal_distance_m")
    if goal_distance_m is not None:
        info_lines.append(f"distancia final a objetivo: {float(goal_distance_m):.3f} m")
    if target_depth_into_second_corridor_m is not None:
        info_lines.append(
            f"depth target 2do corredor: {float(target_depth_into_second_corridor_m):.2f} m"
        )
    final_error_yaw_deg = summary.get("final_error_yaw_deg")
    if final_error_yaw_deg is not None:
        info_lines.append(f"error final yaw: {float(final_error_yaw_deg):.2f} deg")

    ax.text(
        0.02,
        0.98,
        "\n".join(info_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#666666",
            "alpha": 0.92,
        },
    )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x [m] (frente+)")
    ax.set_ylabel("y [m] (izquierda+)")
    ax.set_title("Corredor estimado y seguimiento de trayectoria a curva")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--output",
        default="",
        help="Optional output image path. Default: <run-dir>/analysis_curve_tracking/curve_tracking_overview.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    analysis_dir = run_dir / "analysis_curve_tracking"
    planned_path_json = analysis_dir / "planned_path.json"
    tracking_summary_json = analysis_dir / "tracking_summary.json"
    tracking_csv = analysis_dir / "tracking_trajectory.csv"
    lidar_points_csv = run_dir / "lidar_points.csv"
    if not planned_path_json.exists():
        raise FileNotFoundError(f"Missing {planned_path_json}")
    if not tracking_summary_json.exists():
        raise FileNotFoundError(f"Missing {tracking_summary_json}")
    if not tracking_csv.exists():
        raise FileNotFoundError(f"Missing {tracking_csv}")

    planned_path_xy, planned_path_yaw = _load_planned_path(planned_path_json)
    tracking_xy = _load_tracking_csv(tracking_csv)
    lidar_world_xy = _load_lidar_world_points(lidar_points_csv)
    summary = _load_json(tracking_summary_json)
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else analysis_dir / "curve_tracking_overview.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _plot_run(
        run_dir=run_dir,
        planned_path_xy=planned_path_xy,
        planned_path_yaw=planned_path_yaw,
        tracking_xy=tracking_xy,
        lidar_world_xy=lidar_world_xy,
        summary=summary,
        output_path=output_path,
    )
    print(f"[APEX] Curve tracking plot: {output_path}")


if __name__ == "__main__":
    main()
