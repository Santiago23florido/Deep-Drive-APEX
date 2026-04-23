#!/usr/bin/env python3
"""Compare one simulated recognition_tour run against a real run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _resolve_run_dir(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if path.name == "analysis_recognition_tour":
        return path.parent
    return path


def _trajectory_path(run_dir: Path) -> Path:
    return run_dir / "analysis_recognition_tour" / "recognition_tour_trajectory.csv"


def _summary_path(run_dir: Path) -> Path:
    return run_dir / "analysis_recognition_tour" / "recognition_tour_summary.json"


def _load_summary(run_dir: Path) -> dict[str, Any]:
    path = _summary_path(run_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_rows(run_dir: Path) -> list[dict[str, str]]:
    path = _trajectory_path(run_dir)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _column(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        raw = row.get(key, "")
        if raw in ("", None):
            values.append(float("nan"))
            continue
        try:
            value = float(raw)
        except Exception:
            value = float("nan")
        values.append(value if math.isfinite(value) else float("nan"))
    return np.asarray(values, dtype=np.float64)


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _finite_max(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def _finite_min(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.min(finite))


def _series_metrics(rows: list[dict[str, str]], summary: dict[str, Any]) -> dict[str, Any]:
    t = _column(rows, "t_monotonic_s")
    applied_speed = _column(rows, "applied_speed_pct")
    applied_steering = _column(rows, "applied_steering_deg")
    desired_steering = _column(rows, "desired_steering_deg")
    path_deviation = _column(rows, "path_deviation_m")
    return {
        "samples": len(rows),
        "duration_s": float(summary.get("time_total_s", float("nan"))),
        "end_cause": summary.get("end_cause", ""),
        "mean_applied_speed_pct": _finite_mean(applied_speed),
        "mean_abs_applied_steering_deg": _finite_mean(np.abs(applied_steering)),
        "mean_abs_desired_steering_deg": _finite_mean(np.abs(desired_steering)),
        "mean_path_deviation_m": _finite_mean(path_deviation),
        "max_path_deviation_m": _finite_max(path_deviation),
        "t_start_s": float(t[0]) if t.size else float("nan"),
        "t_end_s": float(t[-1]) if t.size else float("nan"),
    }


def _sim_ground_truth_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    gt_pose_error = _column(rows, "gt_pose_error_m")
    gt_yaw_error = _column(rows, "gt_yaw_error_rad")
    gt_local_path_error = _column(rows, "gt_local_path_error_m")
    gt_inner_clearance = _column(rows, "gt_clearance_inner_m")
    gt_outer_clearance = _column(rows, "gt_clearance_outer_m")
    gt_real_steering = _column(rows, "gt_steering_real_deg")
    desired_steering = _column(rows, "desired_steering_deg")
    applied_steering = _column(rows, "applied_steering_deg")
    return {
        "mean_pose_error_m": _finite_mean(gt_pose_error),
        "max_pose_error_m": _finite_max(gt_pose_error),
        "mean_yaw_error_rad": _finite_mean(np.abs(gt_yaw_error)),
        "max_yaw_error_rad": _finite_max(np.abs(gt_yaw_error)),
        "mean_local_path_error_m": _finite_mean(gt_local_path_error),
        "max_local_path_error_m": _finite_max(gt_local_path_error),
        "min_inner_clearance_m": _finite_min(gt_inner_clearance),
        "min_outer_clearance_m": _finite_min(gt_outer_clearance),
        "mean_abs_requested_vs_real_steering_deg": _finite_mean(
            np.abs(desired_steering - gt_real_steering)
        ),
        "mean_abs_applied_vs_real_steering_deg": _finite_mean(
            np.abs(applied_steering - gt_real_steering)
        ),
    }


def _write_plot(
    *,
    real_rows: list[dict[str, str]],
    sim_rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    real_t = _column(real_rows, "t_monotonic_s")
    sim_t = _column(sim_rows, "t_monotonic_s")

    axes[0].plot(real_t, _column(real_rows, "applied_speed_pct"), label="real applied speed %")
    axes[0].plot(sim_t, _column(sim_rows, "applied_speed_pct"), label="sim applied speed %")
    axes[0].set_ylabel("speed %")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(real_t, _column(real_rows, "desired_steering_deg"), label="real desired steer")
    axes[1].plot(real_t, _column(real_rows, "applied_steering_deg"), label="real applied steer")
    axes[1].plot(sim_t, _column(sim_rows, "desired_steering_deg"), label="sim desired steer")
    axes[1].plot(sim_t, _column(sim_rows, "applied_steering_deg"), label="sim applied steer")
    axes[1].plot(sim_t, _column(sim_rows, "gt_steering_real_deg"), label="sim real steer")
    axes[1].set_ylabel("steering deg")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(real_t, _column(real_rows, "path_deviation_m"), label="real path deviation")
    axes[2].plot(sim_t, _column(sim_rows, "path_deviation_m"), label="sim path deviation")
    axes[2].plot(sim_t, _column(sim_rows, "gt_local_path_error_m"), label="sim GT path error")
    axes[2].set_ylabel("meters")
    axes[2].set_xlabel("t [s]")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-run", required=True)
    parser.add_argument("--sim-run", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-figure", default="")
    args = parser.parse_args()

    real_run = _resolve_run_dir(args.real_run)
    sim_run = _resolve_run_dir(args.sim_run)
    real_rows = _load_rows(real_run)
    sim_rows = _load_rows(sim_run)
    real_summary = _load_summary(real_run)
    sim_summary = _load_summary(sim_run)

    if not real_rows:
        raise SystemExit(f"No trajectory CSV found for real run: {real_run}")
    if not sim_rows:
        raise SystemExit(f"No trajectory CSV found for sim run: {sim_run}")

    output_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else sim_run / "analysis_recognition_tour" / "recognition_tour_comparison.json"
    )
    output_figure = (
        Path(args.output_figure).expanduser().resolve()
        if args.output_figure
        else sim_run / "analysis_recognition_tour" / "recognition_tour_comparison.png"
    )

    payload = {
        "real_run": str(real_run),
        "sim_run": str(sim_run),
        "real_metrics": _series_metrics(real_rows, real_summary),
        "sim_metrics": _series_metrics(sim_rows, sim_summary),
        "sim_ground_truth_metrics": _sim_ground_truth_metrics(sim_rows),
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_plot(real_rows=real_rows, sim_rows=sim_rows, output_path=output_figure)
    print(str(output_json))
    print(str(output_figure))


if __name__ == "__main__":
    main()
