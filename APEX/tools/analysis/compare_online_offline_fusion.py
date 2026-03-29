#!/usr/bin/env python3
"""Compare online fusion outputs against the offline straight-run reference."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


PLANAR_ERROR_PASS_M = 0.30
YAW_ERROR_PASS_DEG = 12.0
HIGH_CONFIDENCE_PASS_PCT = 80.0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_last_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV has no rows: {path}")
    return rows[-1]


def _build_online_summary_from_trajectory(
    trajectory_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    with trajectory_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"CSV has no rows: {trajectory_path}")

    final_row = rows[-1]
    distance_m = 0.0
    for prev_row, next_row in zip(rows[:-1], rows[1:]):
        distance_m += math.hypot(
            float(next_row["x_m"]) - float(prev_row["x_m"]),
            float(next_row["y_m"]) - float(prev_row["y_m"]),
        )

    high_confidence_count = sum(
        1 for row in rows if str(row.get("confidence", "")).strip().lower() == "high"
    )
    alignment_ready = any(str(row.get("alignment_ready", "0")).strip() == "1" for row in rows)
    imu_initialized = any(str(row.get("imu_initialized", "0")).strip() == "1" for row in rows)
    best_effort_init = any(str(row.get("best_effort_init", "0")).strip() == "1" for row in rows)

    summary = {
        "trajectory_csv": str(trajectory_path),
        "trajectory_row_count": len(rows),
        "status_message_count": None,
        "final_pose": {
            "x_m": float(final_row["x_m"]),
            "y_m": float(final_row["y_m"]),
            "yaw_rad": float(final_row["yaw_rad"]),
        },
        "distance_m": distance_m,
        "high_confidence_pct": 100.0 * high_confidence_count / max(1, len(rows)),
        "static_initialization": None,
        "corridor_model": None,
        "quality": None,
        "parameters": None,
        "state": "reconstructed_from_trajectory",
        "alignment_ready": alignment_ready,
        "imu_initialized": imu_initialized,
        "best_effort_init": best_effort_init,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _run_offline_if_needed(run_dir: Path, apex_root: Path) -> tuple[Path, Path]:
    offline_dir = run_dir / "analysis_sensor_fusion"
    trajectory_path = offline_dir / "sensor_fusion_trajectory.csv"
    summary_path = offline_dir / "sensor_fusion_summary.json"
    if trajectory_path.exists() and summary_path.exists():
        return trajectory_path, summary_path

    script_path = apex_root / "apex_forward_raw" / "sensor_fusionn.py"
    subprocess.run(
        [sys.executable, str(script_path), "--run-dir", str(run_dir)],
        check=True,
    )
    return trajectory_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    apex_root = Path(__file__).resolve().parents[2]
    output_dir = run_dir / "analysis_sensor_fusion_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "online_vs_offline_summary.json"

    online_dir = run_dir / "analysis_sensor_fusion_online"
    online_trajectory_path = online_dir / "online_fusion_trajectory.csv"
    online_summary_path = online_dir / "online_fusion_summary.json"

    if not online_trajectory_path.exists():
        output_path.write_text(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "pass": False,
                    "error": "missing_online_outputs",
                    "online_trajectory_csv": str(online_trajectory_path),
                    "online_summary_json": str(online_summary_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"comparison_json: {output_path}")
        print("pass: False")
        return

    if online_summary_path.exists():
        online_summary = _load_json(online_summary_path)
    else:
        online_summary = _build_online_summary_from_trajectory(
            trajectory_path=online_trajectory_path,
            summary_path=online_summary_path,
        )

    offline_trajectory_path, offline_summary_path = _run_offline_if_needed(run_dir, apex_root)

    offline_summary = _load_json(offline_summary_path)
    online_last = _load_last_csv_row(online_trajectory_path)
    offline_last = _load_last_csv_row(offline_trajectory_path)

    online_x_m = float(online_last["x_m"])
    online_y_m = float(online_last["y_m"])
    online_yaw_rad = float(online_last["yaw_rad"])
    offline_x_m = float(offline_last["x_m"])
    offline_y_m = float(offline_last["y_m"])
    offline_yaw_rad = float(offline_last["yaw_rad"])

    final_planar_error_m = math.hypot(online_x_m - offline_x_m, online_y_m - offline_y_m)
    final_yaw_error_deg = abs(
        math.degrees(_normalize_angle(online_yaw_rad - offline_yaw_rad))
    )
    online_distance_m = float(online_summary.get("distance_m") or 0.0)
    offline_distance_m = 0.0
    try:
        offline_rows = list(csv.DictReader(offline_trajectory_path.open(newline="", encoding="utf-8")))
        for prev_row, next_row in zip(offline_rows[:-1], offline_rows[1:]):
            offline_distance_m += math.hypot(
                float(next_row["x_m"]) - float(prev_row["x_m"]),
                float(next_row["y_m"]) - float(prev_row["y_m"]),
            )
    except Exception:
        offline_distance_m = 0.0

    high_confidence_pct = float(online_summary.get("high_confidence_pct") or 0.0)
    result = {
        "run_dir": str(run_dir),
        "online_trajectory_csv": str(online_trajectory_path),
        "online_summary_json": str(online_summary_path),
        "offline_trajectory_csv": str(offline_trajectory_path),
        "offline_summary_json": str(offline_summary_path),
        "online_final_pose": {
            "x_m": online_x_m,
            "y_m": online_y_m,
            "yaw_rad": online_yaw_rad,
        },
        "offline_final_pose": {
            "x_m": offline_x_m,
            "y_m": offline_y_m,
            "yaw_rad": offline_yaw_rad,
        },
        "metrics": {
            "final_planar_error_m": final_planar_error_m,
            "final_yaw_error_deg": final_yaw_error_deg,
            "distance_delta_m": abs(online_distance_m - offline_distance_m),
            "online_high_confidence_pct": high_confidence_pct,
        },
        "thresholds": {
            "planar_error_m_max": PLANAR_ERROR_PASS_M,
            "yaw_error_deg_max": YAW_ERROR_PASS_DEG,
            "high_confidence_pct_min": HIGH_CONFIDENCE_PASS_PCT,
        },
        "pass": (
            final_planar_error_m <= PLANAR_ERROR_PASS_M
            and final_yaw_error_deg <= YAW_ERROR_PASS_DEG
            and high_confidence_pct >= HIGH_CONFIDENCE_PASS_PCT
        ),
    }

    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"comparison_json: {output_path}")
    print(f"pass: {result['pass']}")


if __name__ == "__main__":
    main()
