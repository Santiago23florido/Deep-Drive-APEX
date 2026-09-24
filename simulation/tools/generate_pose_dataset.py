#!/usr/bin/env python3
"""Generate the multi-scenario LiDAR-IMU pose dataset (headless Gazebo -> SQLite).

Examples (from simulation/):

    python3 tools/generate_pose_dataset.py --calibrate-speed
    python3 tools/generate_pose_dataset.py --preset smoke --headless \
        --database data/multiscenario_pose/smoke_pose_dataset.sqlite3
    python3 tools/generate_pose_dataset.py --preset full --estimate-only
    python3 tools/generate_pose_dataset.py --preset full --headless \
        --database data/multiscenario_pose/pose_dataset.sqlite3 --resume

The script re-executes itself with the system Python 3.12 and a clean
environment when needed: Conda / virtualenv Pythons and the ROS Gazebo vendor
libraries on LD_LIBRARY_PATH break the gz-sim 8 Python bindings.
See tools/pose_dataset/README.md.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

TOOLS_DIR = Path(__file__).resolve().parent
SIM_ROOT = TOOLS_DIR.parent
REEXEC_FLAG = "POSE_DATASET_REEXEC"


def _needs_reexec() -> bool:
    if os.environ.get(REEXEC_FLAG) == "1":
        return False
    bad_ld = any("/opt/ros/" in p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p)
    conda = "conda" in sys.executable or "CONDA_PREFIX" in os.environ
    return bad_ld or conda or sys.version_info[:2] != (3, 12)


def _reexec() -> None:
    venv = SIM_ROOT / "learning" / ".venv" / "bin" / "python"
    python = "/usr/bin/python3"
    if venv.exists():
        python = str(venv)  # system Python 3.12 + system site-packages (gz bindings) + torch
    env = {
        "HOME": os.environ.get("HOME", "/tmp"),
        "PATH": "/usr/bin:/bin",
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "PYTHONPATH": str(TOOLS_DIR),
        REEXEC_FLAG: "1",
        "PYTHONUNBUFFERED": "1",
    }
    os.execve(python, [python, str(Path(__file__).resolve()), *sys.argv[1:]], env)


def _resolve(path: str) -> Path:
    """Relative paths are resolved from the simulation/ directory, so the
    command behaves the same from any working directory."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (SIM_ROOT / p).resolve()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="smoke", help="smoke | full | zeroshot (config/campaign.yaml)")
    ap.add_argument("--database", default="data/multiscenario_pose/pose_dataset.sqlite3", help="SQLite file (relative paths are resolved from simulation/)")
    ap.add_argument("--headless", action="store_true", help="accepted for compatibility: the generator is always headless")
    ap.add_argument("--resume", action="store_true", help="continue an existing database, skipping completed runs")
    ap.add_argument("--retry-failed", action="store_true", help="with --resume, also retry runs marked failed")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 3), help="parallel Gazebo processes")
    ap.add_argument("--repeats", type=int, default=1, help="multiply every seed list (seed + 1000*k)")
    ap.add_argument("--laps", type=float, default=None, help="override the laps per trajectory of the preset")
    ap.add_argument("--cache-dir", default="data/multiscenario_pose/cache", help="trajectory logs (1 kHz exact state) and worker files")
    ap.add_argument("--report-dir", default=None, help="default: <database stem>_report next to the database")
    ap.add_argument("--compress-ranges", action="store_true", help="store ranges as f32le+zlib instead of raw f32le")
    ap.add_argument("--estimate-only", action="store_true", help="print the plan, time and size estimate, then exit")
    ap.add_argument("--report-only", action="store_true", help="rebuild the report of an existing database")
    ap.add_argument("--calibrate-speed", action="store_true", help="measure the maximum stable speed of every track")
    ap.add_argument("--tracks", nargs="*", default=None, help="tracks for --calibrate-speed / --install-gazebo-worlds (default: all)")
    ap.add_argument("--export-worlds", default=None, help="write world.sdf / track.json / preview of every track to this directory")
    ap.add_argument("--install-gazebo-worlds", action="store_true", help="write every track to rc_sim_description/worlds/pose_dataset/ and add pose_<track> scenarios")
    args = ap.parse_args()

    if _needs_reexec():
        _reexec()
    sys.path.insert(0, str(TOOLS_DIR))

    from pose_dataset.calibrate import CALIBRATION_FILE, calibrate
    from pose_dataset.campaign import Campaign, build_plan, estimate, plan_summary
    from pose_dataset.report import build_report

    cache_dir = _resolve(args.cache_dir)

    if args.export_worlds:
        from pose_dataset.tracks import export_track, load_track, load_track_specs, plot_tracks

        out = _resolve(args.export_worlds)
        tracks = [load_track(n) for n in load_track_specs()]
        for tr in tracks:
            export_track(tr, out / tr.name)
        plot_tracks(tracks, out / "tracks_overview.png")
        print(f"[worlds] {len(tracks)} tracks written to {out}")
        return 0

    if args.install_gazebo_worlds:
        from pose_dataset.gazebo_worlds import SCENARIOS, WORLD_DIR, export_gazebo_worlds

        index = export_gazebo_worlds(args.tracks)
        for name, info in index.items():
            print(f"[gazebo] {info['world']:42s} scenario {info['scenario']:30s} spawn {info['spawn']}")
        print(f"[gazebo] {len(index)} worlds in {WORLD_DIR}; scenarios added to {SCENARIOS}")
        print("[gazebo] rebuild once: colcon build --symlink-install --packages-select rc_sim_description")
        return 0

    if args.calibrate_speed:
        result = calibrate(args.tracks, cache_dir / "calibration", args.workers)
        if args.tracks and CALIBRATION_FILE.exists():
            old = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
            old["tracks"].update(result["tracks"])
            old["generated_utc"] = result["generated_utc"]
            result = old
        CALIBRATION_FILE.write_text(json.dumps(result, indent=1), encoding="utf-8")
        for name, tr in result["tracks"].items():
            print(f"[calibrate] {name:24s} v_max_stable = {tr['v_max_stable_mps']:.2f} m/s  (first unstable: {tr['first_unstable']})")
        return 0

    db_path = _resolve(args.database)
    report_dir = _resolve(args.report_dir) if args.report_dir else db_path.parent / f"{db_path.stem}_report"

    if args.report_only:
        rep = build_report(db_path, report_dir)
        print(json.dumps({k: rep[k] for k in ("runs_completed", "runs_failed", "row_counts", "integrity", "split_distribution")}, indent=1, default=float))
        return 0

    plan = build_plan(args.preset, repeats=args.repeats, laps=args.laps)
    summary = plan_summary(plan)
    est = estimate(plan, args.workers)
    print(f"[plan] preset={plan['preset']} role={plan['role']}: {plan['description']}")
    print(f"[plan] {summary['runs']} runs from {summary['trajectories']} simulated trajectories, runs per split {summary['runs_by_split']}")
    print(f"[plan] tracks per split {summary['tracks_by_split']}")
    print(f"[plan] sensor+motion combinations absent from train: {summary['sensor_motion_combos_unseen_in_train']}")
    print(f"[estimate] {est['simulated_h']:.2f} h of simulated time; ~{est['wall_min_estimate']:.0f} min wall with {args.workers} workers "
          f"(per-process real-time factor {est['per_process_rtf']:.1f}); database ~{est['database_gb_estimate']:.2f} GB")
    if args.estimate_only:
        return 0

    if db_path.exists() and not args.resume:
        print(f"[error] {db_path} exists: pass --resume to continue it, or choose another --database", file=sys.stderr)
        return 2
    campaign = Campaign(plan, db_path, cache_dir, args.workers, compress_ranges=args.compress_ranges, retry_failed=args.retry_failed)
    result = campaign.run()
    rep = build_report(db_path, report_dir, integrity={k: result[k] for k in ("integrity_check", "foreign_key_violations")}, extra={"plan": summary, "estimate": est, "execution": result})
    print(f"[done] completed runs {rep['runs_completed']}, failed {rep['runs_failed']}, other {rep['runs_other']}; "
          f"scans {rep['row_counts']['lidar_scans']}, IMU samples {rep['row_counts']['imu_samples']}, ground truth {rep['row_counts']['ground_truth']}")
    print(f"[done] integrity_check={result['integrity_check']} foreign_key_violations={result['foreign_key_violations']} "
          f"database {rep['database_bytes'] / 1e6:.1f} MB, wall {result['wall_s'] / 60:.1f} min{' (interrupted, rerun with --resume)' if result['interrupted'] else ''}")
    print(f"[done] report: {report_dir}")
    return 0 if result["integrity_check"] == "ok" and not result["interrupted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
