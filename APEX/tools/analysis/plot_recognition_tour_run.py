#!/usr/bin/env python3
"""Render one recognition_tour run as a single route/trajectory image."""

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


def _load_route_json(path: Path) -> np.ndarray:
    payload = _load_json(path)
    points = np.asarray(payload.get("path_xy_yaw", []), dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float64)
    return points[:, :3]


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


def _plot_run(
    *,
    route_xy_yaw: np.ndarray,
    tracking_xy: np.ndarray,
    lidar_world_xy: np.ndarray,
    summary: dict,
    output_path: Path,
) -> None:
    planner_status = summary.get("planner_status") or {}
    tracker_status = summary.get("tracker_status") or {}
    start_pose = planner_status.get("start_pose") or {}
    start_axis = planner_status.get("start_axis") or {}
    route_xy = route_xy_yaw[:, :2] if route_xy_yaw.shape[0] else np.empty((0, 2), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(11.0, 8.0))

    if lidar_world_xy.shape[0] > 0:
        ax.scatter(
            lidar_world_xy[:, 0],
            lidar_world_xy[:, 1],
            s=4,
            c="#8c8c8c",
            alpha=0.16,
            linewidths=0.0,
            label=f"LiDAR aggregated ({lidar_world_xy.shape[0]} pts)",
            zorder=1,
        )

    if route_xy.shape[0] > 0:
        ax.plot(
            route_xy[:, 0],
            route_xy[:, 1],
            color="#1f77b4",
            linewidth=2.6,
            label="Saved route",
            zorder=4,
        )
        ax.scatter(
            [route_xy[0, 0]],
            [route_xy[0, 1]],
            c="#00a676",
            s=70,
            marker="o",
            label="Route start",
            zorder=5,
        )
        ax.scatter(
            [route_xy[-1, 0]],
            [route_xy[-1, 1]],
            c="#d81b60",
            s=75,
            marker="X",
            label="Route end",
            zorder=5,
        )

    if tracking_xy.shape[0] > 0:
        ax.plot(
            tracking_xy[:, 0],
            tracking_xy[:, 1],
            color="#111111",
            linewidth=2.2,
            label="Driven trajectory",
            zorder=6,
        )
        ax.scatter(
            [tracking_xy[-1, 0]],
            [tracking_xy[-1, 1]],
            c="#ff7f0e",
            s=60,
            marker="D",
            label="Tracking end",
            zorder=7,
        )

    if start_pose and start_axis:
        normal_xy = np.asarray(start_axis.get("normal_xy") or [1.0, 0.0], dtype=np.float64)
        tangent_xy = np.asarray(start_axis.get("tangent_xy") or [0.0, 1.0], dtype=np.float64)
        line_center = np.asarray(
            [float(start_pose.get("x_m", 0.0)), float(start_pose.get("y_m", 0.0))],
            dtype=np.float64,
        )
        line_half_length_m = 1.2
        line_points = np.vstack(
            [
                line_center - (line_half_length_m * tangent_xy),
                line_center + (line_half_length_m * tangent_xy),
            ]
        )
        ax.plot(
            line_points[:, 0],
            line_points[:, 1],
            color="#7b3fbc",
            linestyle="--",
            linewidth=2.0,
            label="Start axis",
            zorder=5,
        )
        ax.arrow(
            float(line_center[0]),
            float(line_center[1]),
            0.35 * float(normal_xy[0]),
            0.35 * float(normal_xy[1]),
            width=0.01,
            head_width=0.07,
            head_length=0.08,
            color="#7b3fbc",
            length_includes_head=True,
            zorder=5,
        )

    info_lines = [
        f"run: {Path(summary.get('run_dir', '')).name}",
        f"end: {summary.get('end_cause')}",
        f"trajectory points: {summary.get('trajectory_row_count')}",
        f"route points: {summary.get('route_point_count')}",
        f"local path msgs: {summary.get('local_path_message_count')}",
        f"travel distance: {planner_status.get('travel_distance_m', 'n/a')}",
        f"planner state: {planner_status.get('state', 'n/a')}",
        f"tracker state: {tracker_status.get('state', 'n/a')}",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(info_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#666666",
            "alpha": 0.94,
        },
    )

    all_points = []
    if lidar_world_xy.shape[0]:
        all_points.append(lidar_world_xy)
    if route_xy.shape[0]:
        all_points.append(route_xy)
    if tracking_xy.shape[0]:
        all_points.append(tracking_xy)

    if all_points:
        combined = np.vstack(all_points)
        x_margin = max(0.25, 0.08 * float(np.max(combined[:, 0]) - np.min(combined[:, 0]) + 1.0))
        y_margin = max(0.25, 0.08 * float(np.max(combined[:, 1]) - np.min(combined[:, 1]) + 1.0))
        ax.set_xlim(float(np.min(combined[:, 0]) - x_margin), float(np.max(combined[:, 0]) + x_margin))
        ax.set_ylim(float(np.min(combined[:, 1]) - y_margin), float(np.max(combined[:, 1]) + y_margin))

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x [m] (forward+)")
    ax.set_ylabel("y [m] (left+)")
    ax.set_title("Recognition Tour: route and driven trajectory")
    ax.legend(loc="upper right", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    analysis_dir = run_dir / "analysis_recognition_tour"
    route_json = analysis_dir / "recognition_tour_route.json"
    tracking_csv = analysis_dir / "recognition_tour_trajectory.csv"
    summary_json = analysis_dir / "recognition_tour_summary.json"
    lidar_points_csv = run_dir / "lidar_points.csv"
    output_path = analysis_dir / "recognition_tour_overview.png"

    summary = _load_json(summary_json)
    route_xy_yaw = _load_route_json(route_json) if route_json.exists() else np.empty((0, 3), dtype=np.float64)
    tracking_xy = _load_tracking_csv(tracking_csv) if tracking_csv.exists() else np.empty((0, 2), dtype=np.float64)
    lidar_world_xy = _load_lidar_world_points(lidar_points_csv)
    _plot_run(
        route_xy_yaw=route_xy_yaw,
        tracking_xy=tracking_xy,
        lidar_world_xy=lidar_world_xy,
        summary=summary,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
