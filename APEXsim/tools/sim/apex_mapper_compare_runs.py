#!/usr/bin/env python3
"""Create a side-by-side PNG comparison for mapper iterations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def _load_summary(run_dir: Path) -> dict[str, object]:
    summary_path = run_dir / "mapping_summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _candidate_metrics(summary: dict[str, object], candidate_name: str) -> dict[str, float]:
    candidate_evaluations = summary.get("candidate_evaluations", {})
    payload = candidate_evaluations.get(candidate_name)
    if not isinstance(payload, dict):
        payload = summary.get("evaluation", {})
    return {
        "dilated_iou": float(payload.get("dilated_iou", 0.0) or 0.0),
        "wall_precision": float(payload.get("wall_precision", 0.0) or 0.0),
        "wall_recall": float(payload.get("wall_recall", 0.0) or 0.0),
        "chamfer_distance_m": float(payload.get("chamfer_distance_m", 0.0) or 0.0),
        "closure_gap_m": float(payload.get("closure_gap_m", 0.0) or 0.0),
        "runtime_s": float(summary.get("processing_elapsed_s", 0.0) or 0.0),
        "occupied_cells": float(summary.get("occupied_cell_count", 0.0) or 0.0),
        "visual_points": float(summary.get("visual_point_count", 0.0) or 0.0),
    }


def _overlay_path(summary: dict[str, object], run_dir: Path, candidate_name: str) -> Path:
    candidate_evaluations = summary.get("candidate_evaluations", {})
    payload = candidate_evaluations.get(candidate_name)
    if isinstance(payload, dict):
        files = payload.get("files", {})
        overlay = files.get("mapping_vs_gazebo_overlay_png")
        if overlay:
            return Path(str(overlay))
    return run_dir / f"candidate_{candidate_name}" / "mapping_vs_gazebo_overlay.png"


def _render_metrics(axis, title: str, metrics: dict[str, float], baseline_metrics: dict[str, float] | None) -> None:
    axis.axis("off")
    lines = [title, ""]
    ordered_keys = [
        ("dilated_iou", "{:.3f}"),
        ("wall_precision", "{:.3f}"),
        ("wall_recall", "{:.3f}"),
        ("chamfer_distance_m", "{:.3f} m"),
        ("closure_gap_m", "{:.3f} m"),
        ("runtime_s", "{:.1f} s"),
        ("occupied_cells", "{:.0f}"),
        ("visual_points", "{:.0f}"),
    ]
    for key, fmt in ordered_keys:
        value = metrics.get(key, 0.0)
        if baseline_metrics is not None:
            delta = value - baseline_metrics.get(key, 0.0)
            if key in {"chamfer_distance_m", "closure_gap_m", "runtime_s", "occupied_cells", "visual_points"}:
                delta_text = f"{delta:+.3f}"
            else:
                delta_text = f"{delta:+.3f}"
            lines.append(f"{key}: {fmt.format(value)}  ({delta_text})")
        else:
            lines.append(f"{key}: {fmt.format(value)}")
    axis.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--baseline-candidate", type=str, default="loop_pose_graph")
    parser.add_argument("--candidate-candidate", type=str, default="loop_pose_graph")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Mapper Iteration Comparison")
    args = parser.parse_args()

    baseline_summary = _load_summary(args.baseline_dir)
    candidate_summary = _load_summary(args.candidate_dir)
    baseline_overlay = _overlay_path(baseline_summary, args.baseline_dir, args.baseline_candidate)
    candidate_overlay = _overlay_path(candidate_summary, args.candidate_dir, args.candidate_candidate)
    baseline_image = mpimg.imread(str(baseline_overlay))
    candidate_image = mpimg.imread(str(candidate_overlay))
    baseline_metrics = _candidate_metrics(baseline_summary, args.baseline_candidate)
    candidate_metrics = _candidate_metrics(candidate_summary, args.candidate_candidate)

    figure = plt.figure(figsize=(15, 9))
    grid = figure.add_gridspec(2, 2, height_ratios=[3.2, 1.8], wspace=0.08, hspace=0.15)
    axis_baseline = figure.add_subplot(grid[0, 0])
    axis_candidate = figure.add_subplot(grid[0, 1])
    axis_baseline_metrics = figure.add_subplot(grid[1, 0])
    axis_candidate_metrics = figure.add_subplot(grid[1, 1])

    axis_baseline.imshow(baseline_image)
    axis_baseline.set_title(f"Baseline: {args.baseline_dir.name}")
    axis_baseline.axis("off")

    axis_candidate.imshow(candidate_image)
    axis_candidate.set_title(f"Current: {args.candidate_dir.name}")
    axis_candidate.axis("off")

    _render_metrics(axis_baseline_metrics, "Baseline Metrics", baseline_metrics, None)
    _render_metrics(axis_candidate_metrics, "Current Metrics", candidate_metrics, baseline_metrics)

    figure.suptitle(args.title)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
