#!/usr/bin/env python3
"""Build a static LiDAR snapshot CSV from lidar_points.csv."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="", help="Run directory containing lidar_points.csv")
    parser.add_argument("--lidar-points", default="", help="Path to lidar_points.csv")
    parser.add_argument("--output", default="", help="Output snapshot CSV path")
    parser.add_argument("--scan-index", type=int, default=None, help="Explicit scan_index to export")
    args = parser.parse_args()
    if not args.run_dir and not args.lidar_points:
        parser.error("either --run-dir or --lidar-points is required")
    return args


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
    if args.lidar_points:
        lidar_points_path = Path(args.lidar_points).expanduser().resolve()
    else:
        lidar_points_path = run_dir / "lidar_points.csv"
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    elif run_dir is not None:
        output_path = run_dir / "lidar_snapshot.csv"
    else:
        output_path = lidar_points_path.with_name("lidar_snapshot.csv")
    return lidar_points_path, output_path


def build_snapshot(
    lidar_points_path: Path,
    output_path: Path,
    scan_index: int | None,
) -> tuple[int, int]:
    rows_by_scan: dict[int, list[dict[str, str]]] = defaultdict(list)
    with lidar_points_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows_by_scan[int(row["scan_index"])].append(row)

    if not rows_by_scan:
        raise ValueError(f"No scans found in {lidar_points_path}")

    chosen_scan = scan_index if scan_index is not None else max(rows_by_scan, key=lambda idx: len(rows_by_scan[idx]))
    if chosen_scan not in rows_by_scan:
        raise ValueError(f"scan_index={chosen_scan} not found in {lidar_points_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    point_count = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["angle_deg", "range_m", "count"])
        for row in rows_by_scan[chosen_scan]:
            angle_deg = math.degrees(float(row["angle_rad"]))
            range_m = float(row["range_m"])
            writer.writerow([f"{angle_deg:.6f}", f"{range_m:.6f}", 1])
            point_count += 1
    return chosen_scan, point_count


def main() -> None:
    args = parse_args()
    lidar_points_path, output_path = resolve_paths(args)
    chosen_scan, point_count = build_snapshot(
        lidar_points_path=lidar_points_path,
        output_path=output_path,
        scan_index=args.scan_index,
    )
    print(f"[APEX] Snapshot source: {lidar_points_path}")
    print(f"[APEX] Snapshot output: {output_path}")
    print(f"[APEX] scan_index={chosen_scan}")
    print(f"[APEX] point_count={point_count}")


if __name__ == "__main__":
    main()
