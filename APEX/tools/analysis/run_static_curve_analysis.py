#!/usr/bin/env python3
"""Build a LiDAR snapshot and run static curve analysis in one command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BUILD_SNAPSHOT_SCRIPT = SCRIPT_DIR / "build_lidar_snapshot.py"
ANALYZE_SCRIPT = SCRIPT_DIR / "analyze_lidar_curve_snapshot.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Run directory containing lidar_points.csv")
    parser.add_argument("--scan-index", type=int, default=None, help="Explicit scan_index to export")
    parser.add_argument(
        "--snapshot-left-positive",
        action="store_true",
        help="Interpret snapshot angles as izquierda+ instead of derecha+",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    snapshot_path = run_dir / "lidar_snapshot.csv"
    output_prefix = run_dir / "lidar_snapshot"

    build_cmd = [sys.executable, str(BUILD_SNAPSHOT_SCRIPT), "--run-dir", str(run_dir)]
    if args.scan_index is not None:
        build_cmd.extend(["--scan-index", str(args.scan_index)])
    subprocess.run(build_cmd, check=True)

    analyze_cmd = [
        sys.executable,
        str(ANALYZE_SCRIPT),
        str(snapshot_path),
        "--output-prefix",
        str(output_prefix),
    ]
    if args.snapshot_left_positive:
        analyze_cmd.append("--snapshot-left-positive")
    subprocess.run(analyze_cmd, check=True)

    print(f"[APEX] Snapshot CSV: {snapshot_path}")
    print(f"[APEX] Curve JSON: {output_prefix}_curve_analysis.json")
    print(f"[APEX] Curve PNG: {output_prefix}_curve_analysis.png")


if __name__ == "__main__":
    main()
