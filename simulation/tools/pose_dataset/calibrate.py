"""Experimental maximum stable speed of the car on every track.

For each track and direction the car drives one lap at constant target
speed with the dataset driver. A speed is *stable* when the lap is completed
without collision, the tracking error stays below the configured limit, the
rear axle does not slide and the chassis does not tip. ``v_max_stable`` is the
highest tested speed such that every lower tested speed is stable in both
directions; the motion profiles use fractions of it.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .parallel import JobPool
from .tracks import load_track_specs
from .vehicle import PACKAGE_DIR, load_vehicle_config

CALIBRATION_FILE = PACKAGE_DIR / "config" / "speed_calibration.json"


def load_calibration(path: Path | None = None) -> dict[str, Any]:
    path = path or CALIBRATION_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run generate_pose_dataset.py --calibrate-speed first")
    return json.loads(path.read_text(encoding="utf-8"))


def is_stable(summary: dict[str, Any] | None, crit: dict[str, Any]) -> tuple[bool, str]:
    if summary is None:
        return False, "worker failed"
    if summary["status"] != "completed" or not summary["lap_completed"]:
        return False, summary.get("failure_reason") or "lap not completed"
    if summary["max_lateral_error_m"] > crit["max_lateral_error_m"]:
        return False, f"lateral error {summary['max_lateral_error_m']:.2f} m"
    if summary["max_rear_slip_deg"] > crit["max_rear_slip_deg"]:
        return False, f"rear slip {summary['max_rear_slip_deg']:.1f} deg"
    if summary["max_roll_pitch_deg"] > crit["max_roll_pitch_deg"]:
        return False, f"roll/pitch {summary['max_roll_pitch_deg']:.1f} deg"
    return True, "ok"


def calibrate(tracks: list[str] | None, work_dir: Path, workers: int, log=print) -> dict[str, Any]:
    vcfg = load_vehicle_config()
    crit = vcfg["calibration"]
    specs = load_track_specs()
    tracks = tracks or list(specs)
    jobs = []
    for tr in tracks:
        for direction in crit["directions"]:
            for v in crit["speeds_mps"]:
                key = f"calib__{tr}__d{'ccw' if direction > 0 else 'cw'}__v{v:.2f}"
                jobs.append({
                    "trajectory_key": key, "track": tr, "motion": "calibration", "seed": 0,
                    "direction": direction, "laps": 1.0, "speed_override_mps": v, "calibration": True,
                    "out_npz": str(work_dir / f"{key}.npz"), "tmp_dir": str(work_dir),
                })
    results: dict[str, dict[str, Any]] = {}
    cached = []
    todo = []
    for job in jobs:
        summary_path = Path(job["out_npz"]).with_suffix(".summary.json")
        # Reuse a lap only if it was simulated with the current configuration and code.
        from .gz_runner import physics_hash

        if summary_path.exists() and json.loads(summary_path.read_text(encoding="utf-8")).get("physics_hash") == physics_hash(job):
            cached.append((job, json.loads(summary_path.read_text(encoding="utf-8")), ""))
        else:
            todo.append(job)
    log(f"[calibrate] {len(jobs)} constant-speed laps on {len(tracks)} tracks ({len(cached)} cached) with {workers} workers")
    pool = JobPool(workers)

    def outcomes():
        yield from cached
        yield from pool.run(todo, timeout_fn=lambda j: 600.0)

    for job, summary, err in outcomes():
        ok, why = is_stable(summary, crit)
        entry = {
            "track": job["track"], "direction": job["direction"], "speed_mps": job["speed_override_mps"], "stable": ok, "reason": why,
        }
        if summary:
            entry.update({k: summary[k] for k in ("max_lateral_error_m", "max_rear_slip_deg", "max_roll_pitch_deg", "simulated_s", "wall_s")})
        results[job["trajectory_key"]] = entry
        log(f"[calibrate] {job['trajectory_key']}: {'stable' if ok else 'UNSTABLE'} ({why})")
    out: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "criteria": crit,
        "esc_top_speed_mps": vcfg["esc"]["top_speed_mps"],
        "tracks": {},
    }
    speeds = sorted(crit["speeds_mps"])
    for tr in tracks:
        v_max = 0.0
        limiting = "none"
        for v in speeds:
            runs = [r for r in results.values() if r["track"] == tr and abs(r["speed_mps"] - v) < 1e-9]
            if runs and all(r["stable"] for r in runs):
                v_max = v
            else:
                limiting = "; ".join(f"{'ccw' if r['direction'] > 0 else 'cw'}: {r['reason']}" for r in runs if not r["stable"]) or "missing"
                break
        else:
            limiting = f"ESC top speed / largest tested speed ({speeds[-1]} m/s)"
        out["tracks"][tr] = {
            "v_max_stable_mps": v_max,
            "first_unstable": limiting,
            "runs": sorted(
                ({k: r[k] for k in r if k != "track"} for r in results.values() if r["track"] == tr),
                key=lambda r: (r["direction"], r["speed_mps"]),
            ),
        }
    return out
