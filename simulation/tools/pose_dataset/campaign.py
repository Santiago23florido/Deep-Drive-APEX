"""Campaign planning, execution, resume and clean shutdown."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import signal
import subprocess
import time
from typing import Any

import yaml

from . import GENERATOR_VERSION
from .calibrate import load_calibration
from .controller import load_motion_profiles
from .db import connect, delete_run, finalize, init_db, run_states, write_run
from .parallel import JobPool
from .sensors import load_sensor_profiles
from .tracks import load_track_specs
from .vehicle import PACKAGE_DIR, SIM_ROOT, load_vehicle_config

CAMPAIGN_FILE = PACKAGE_DIR / "config" / "campaign.yaml"


def load_campaign(path: Path | None = None) -> dict[str, Any]:
    return yaml.safe_load((path or CAMPAIGN_FILE).read_text(encoding="utf-8"))


def held_out_names() -> dict[str, list[str]]:
    return {
        "tracks": sorted(n for n, s in load_track_specs().items() if s.get("held_out")),
        "motions": sorted(n for n, s in load_motion_profiles().items() if s.get("held_out")),
        "sensors": sorted(n for n, s in load_sensor_profiles().items() if s.get("held_out")),
    }


# ---------------------------------------------------------------- planning
def build_plan(preset_name: str, repeats: int = 1, laps: float | None = None) -> dict[str, Any]:
    campaign = load_campaign()
    if preset_name not in campaign["presets"]:
        raise KeyError(f"unknown preset {preset_name!r}; available: {sorted(campaign['presets'])}")
    preset = campaign["presets"][preset_name]
    role = preset["role"]
    tracks = load_track_specs()
    motions = load_motion_profiles()
    sensors = load_sensor_profiles()
    calib = load_calibration()["tracks"]
    held = held_out_names()
    laps = float(laps if laps is not None else preset.get("laps", 1.0))
    # gazebo_native: Gazebo's own sensors + the real2sim realism layer
    # (native_runner); every sensor of the preset must be a native profile.
    backend = preset.get("sensor_backend", "synthesized")
    trajectories: dict[str, dict[str, Any]] = {}
    for group in preset["groups"]:
        excluded = {tuple(c) for c in group.get("exclude_combos", [])}
        seeds = [int(s) + 1000 * k for k in range(max(1, repeats)) for s in group["seeds"]]
        for tr in group["tracks"]:
            for mo in group["motions"]:
                for seed in seeds:
                    for se in group["sensors"]:
                        for kind, name, table in (("track", tr, tracks), ("motion", mo, motions), ("sensor", se, sensors)):
                            if name not in table:
                                raise KeyError(f"preset {preset_name}: unknown {kind} {name!r}")
                        if (sensors[se].get("backend", "synthesized") == "gazebo_native") != (backend == "gazebo_native"):
                            raise ValueError(f"preset {preset_name}: sensor {se} does not match the sensor_backend {backend}")
                        if role == "main" and (tr in held["tracks"] or mo in held["motions"] or se in held["sensors"]):
                            raise ValueError(f"preset {preset_name} (role main) uses a held-out zero-shot variant: {tr}/{mo}/{se}")
                        if (se, mo) in excluded:
                            continue
                        variant = group.get("variant")
                        tkey = f"{tr}__{mo}__s{seed}" + (f"__{variant}" if variant else "")
                        if tr not in calib:
                            raise KeyError(f"track {tr} has no speed calibration: run --calibrate-speed --tracks {tr}")
                        v_ref = float(calib[tr]["v_max_stable_mps"]) * float(group.get("v_ref_scale", 1.0))
                        traj = trajectories.setdefault(tkey, {
                            "trajectory_key": tkey, "track": tr, "track_family": tracks[tr].get("family", tr), "motion": mo,
                            "seed": seed, "direction": 1 if seed % 2 == 1 else -1, "laps": float(group.get("laps", laps)), "v_ref": v_ref,
                            "split": group["split"], "variant": variant, "track_overrides": group.get("track_overrides", {}),
                            "backend": backend, "runs": [],
                        })
                        if traj["split"] != group["split"]:
                            raise ValueError(f"trajectory {tkey} assigned to two splits ({traj['split']}, {group['split']})")
                        run_key = f"{tkey}__{se}"
                        if any(r["run_key"] == run_key for r in traj["runs"]):
                            continue
                        traj["runs"].append({"run_key": run_key, "sensor": se})
    # Whole tracks: validation and test must each contain at least one track
    # that never appears in any other split (unseen-track generalisation).
    splits_of: dict[str, set[str]] = {}
    for t in trajectories.values():
        splits_of.setdefault(t["track"], set()).add(t["split"])
    present = {t["split"] for t in trajectories.values()}
    if role == "main" and "train" in present:
        for split in ("validation", "test"):
            if split in present and not any(sp == {split} for sp in splits_of.values()):
                raise ValueError(f"preset {preset_name}: no track is exclusive to {split}")
    return {"preset": preset_name, "role": role, "description": preset.get("description", ""), "laps": laps, "repeats": repeats, "trajectories": list(trajectories.values())}


def plan_summary(plan: dict[str, Any]) -> dict[str, Any]:
    runs = [(t["split"], t["track"], r["sensor"], t["motion"]) for t in plan["trajectories"] for r in t["runs"]]
    by_split: dict[str, int] = {}
    for s, *_ in runs:
        by_split[s] = by_split.get(s, 0) + 1
    tracks_by_split: dict[str, list[str]] = {}
    for t in plan["trajectories"]:
        tracks_by_split.setdefault(t["split"], [])
        if t["track"] not in tracks_by_split[t["split"]]:
            tracks_by_split[t["split"]].append(t["track"])
    train_combos = {(se, mo) for s, _, se, mo in runs if s == "train"}
    unseen = {s: sorted({f"{se}+{mo}" for sp, _, se, mo in runs if sp == s and (se, mo) not in train_combos}) for s in ("validation", "test")}
    return {"runs": len(runs), "trajectories": len(plan["trajectories"]), "runs_by_split": by_split, "tracks_by_split": tracks_by_split, "sensor_motion_combos_unseen_in_train": unseen}


def estimate(plan: dict[str, Any], workers: int) -> dict[str, Any]:
    """Planned simulated time, wall time and database size of a plan."""
    from .gz_runner import build_run_setup
    from .tracks import load_track

    vcfg = load_vehicle_config()
    sensors = load_sensor_profiles()
    cache: dict[str, Any] = {}
    sim_s = 0.0
    size = 0.0
    overhead = float(vcfg["model"]["settle_s"]) + float(vcfg["run"]["static_start_s"]) + float(vcfg["run"]["static_end_s"])
    for t in plan["trajectories"]:
        track = cache.setdefault(t["track"], load_track(t["track"]))
        setup = build_run_setup(t, track, vcfg)
        dur = setup["plan"].planned_duration() + overhead + 1.0
        sim_s += dur
        for r in t["runs"]:
            p = sensors[r["sensor"]]
            scans = dur * p["lidar"]["rate_hz"]
            beams = p["lidar"]["beams"]
            imu = dur * p["imu"]["rate_hz"]
            gt = imu + 2 * scans
            size += scans * (4 * beams + beams / 8 + beams + 190) + imu * 160 + gt * 290
    rtf = _measured_rtf()
    synth_s = sim_s * 0.35 * (len([r for t in plan["trajectories"] for r in t["runs"]]) / max(1, len(plan["trajectories"])))
    wall = (sim_s / rtf + synth_s) / max(1, workers) * 1.15
    return {"simulated_s": sim_s, "simulated_h": sim_s / 3600.0, "per_process_rtf": rtf, "wall_s_estimate": wall, "wall_min_estimate": wall / 60.0, "database_bytes_estimate": size, "database_gb_estimate": size / 1e9}


def _measured_rtf() -> float:
    try:
        runs = [r for tr in load_calibration()["tracks"].values() for r in tr["runs"] if "simulated_s" in r and r.get("wall_s")]
        vals = sorted(r["simulated_s"] / r["wall_s"] for r in runs)
        return float(vals[len(vals) // 2])  # measured with the calibration running in parallel
    except Exception:
        return 2.0


# --------------------------------------------------------------- execution
def _git_commit() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=SIM_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "tools/pose_dataset", "tools/generate_pose_dataset.py"], cwd=SIM_ROOT, capture_output=True, text=True).stdout.strip()
        return commit + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def _simulator_version() -> str:
    try:
        out = subprocess.run(["dpkg-query", "-W", "-f=${Version}", "libgz-sim8"], capture_output=True, text=True).stdout.strip()
        return f"gz-sim {out} (python bindings, dartsim, 1 ms step)"
    except Exception:
        return "gz-sim 8"


def _config_hash() -> str:
    h = hashlib.sha1()
    for name in ("tracks.yaml", "motion.yaml", "sensors.yaml", "vehicle.yaml", "campaign.yaml", "speed_calibration.json"):
        h.update((PACKAGE_DIR / "config" / name).read_bytes())
    return h.hexdigest()


def _run_row(t: dict[str, Any], run: dict[str, Any], status: str, meta: dict[str, Any], physics: dict[str, Any] | None, reason: str = "") -> dict[str, Any]:
    cfg = {
        "trajectory": {k: v for k, v in t.items() if k != "runs"},
        "sensor_profile": run["sensor"],
        "physics": {k: v for k, v in (physics or {}).items() if k not in ("events", "runs", "job")},
        "config_hash": meta["config_hash"],
    }
    rec_start = int((physics or {}).get("record_start_ns") or 0)
    return {
        "run_key": run["run_key"], "trajectory_key": t["trajectory_key"], "scenario_name": f"{t['track']}/{t['motion']}",
        "track_name": t["track"], "track_family": t["track_family"], "split": t["split"], "motion_profile": t["motion"],
        "sensor_profile": run["sensor"], "seed": t["seed"], "direction": t["direction"],
        "simulator_version": meta["simulator_version"], "git_commit": meta["git_commit"], "generator_version": GENERATOR_VERSION,
        "start_time": meta.get("job_start"), "end_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "simulated_duration_s": (physics or {}).get("simulated_s", 0.0) - rec_start * 1e-9 if physics else None,
        "start_sim_time_ns": rec_start, "lap_completed": int(bool((physics or {}).get("lap_completed", False))),
        "laps_target": t["laps"], "distance_m": (physics or {}).get("distance_m"), "v_ref_mps": t["v_ref"],
        "status": status, "failure_reason": reason or None, "qa_passed": None, "configuration_json": json.dumps(cfg, default=float),
    }


class Campaign:
    def __init__(self, plan: dict[str, Any], db_path: Path, cache_dir: Path, workers: int, log=print, compress_ranges: bool = False, retry_failed: bool = False, keep_run_files: bool = False) -> None:
        self.plan = plan
        self.db_path = db_path
        self.cache_dir = cache_dir
        self.workers = workers
        self.log = log
        self.compress = compress_ranges
        self.retry_failed = retry_failed
        self.keep_run_files = keep_run_files
        self.stop = False
        self.pool = JobPool(workers)

    def _signal(self, signum, _frame) -> None:  # noqa: ANN001
        if not self.stop:
            self.log(f"\n[campaign] signal {signum}: stopping workers, finishing the current database transaction...")
        self.stop = True
        self.pool.stop_requested = True

    def run(self) -> dict[str, Any]:
        old_int = signal.signal(signal.SIGINT, self._signal)
        old_term = signal.signal(signal.SIGTERM, self._signal)
        try:
            return self._run()
        finally:
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)

    def _run(self) -> dict[str, Any]:
        conn = connect(self.db_path)
        meta = {"role": self.plan["role"], "last_preset": self.plan["preset"], "config_hash": _config_hash(), "generator_version": GENERATOR_VERSION, "held_out_variants": held_out_names()}
        init_db(conn, meta)
        meta.update(simulator_version=_simulator_version(), git_commit=_git_commit())
        states = run_states(conn)
        # Previous attempts that never finished are redone.
        for key, st in states.items():
            if st["status"] in ("running", "interrupted", "pending"):
                delete_run(conn, key)
        states = run_states(conn)
        jobs = []
        skipped = 0
        traj_dir = self.cache_dir / "trajectories"
        for t in self.plan["trajectories"]:
            pending = []
            for r in t["runs"]:
                st = states.get(r["run_key"])
                if st and st["status"] == "completed":
                    skipped += 1
                    continue
                if st and st["status"] == "failed" and not self.retry_failed:
                    skipped += 1
                    continue
                if st:
                    delete_run(conn, r["run_key"])
                pending.append(r)
            if not pending:
                continue
            job = {k: v for k, v in t.items() if k not in ("runs",)}
            job["runs"] = pending
            native = t.get("backend") == "gazebo_native"
            job_dir = traj_dir / "native" if native else traj_dir
            job["out_npz"] = str(job_dir / f"{t['trajectory_key']}.npz")
            job["tmp_dir"] = str(job_dir)
            if native:
                job["runner"] = "pose_dataset.native_runner"
            jobs.append(job)
        total_runs = sum(len(j["runs"]) for j in jobs)
        self.log(f"[campaign] preset={self.plan['preset']} database={self.db_path} runs pending={total_runs} (skipped {skipped} already done) trajectories={len(jobs)} workers={self.workers}")
        # Placeholders: visible 'running' status while the job is in flight.
        placeholder_meta = dict(meta, job_start=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        conn.execute("BEGIN IMMEDIATE")
        for j in jobs:
            for r in j["runs"]:
                row = _run_row(j, r, "running", placeholder_meta, None)
                cols = list(row)
                conn.execute(f"INSERT INTO runs({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})", [row[c] for c in cols])
        conn.execute("COMMIT")

        stats = {"completed": 0, "failed": 0, "qa_failed": 0, "sim_s": 0.0, "wall_start": time.time(), "done_traj": 0}
        timeout_fn = lambda j: 900.0 + 60.0 * float(j.get("laps", 1.0)) * 10  # noqa: E731
        for job, summary, err in self.pool.run(jobs, timeout_fn):
            stats["done_traj"] += 1
            job_meta = dict(meta, job_start=placeholder_meta["job_start"])
            if summary is None or summary.get("status") != "completed":
                reason = err.strip().splitlines()[-1] if summary is None and err.strip() else (summary or {}).get("failure_reason", "unknown")
                for r in job["runs"]:
                    row = _run_row(job, r, "failed", job_meta, summary, reason)
                    conn.execute("DELETE FROM runs WHERE run_key = ? AND status = 'running'", (r["run_key"],))
                    write_run(conn, row, None, (summary or {}).get("events", []), None)
                    stats["failed"] += 1
                self.log(f"[campaign] {stats['done_traj']}/{len(jobs)} {job['trajectory_key']}: FAILED ({reason[:160]})")
                continue
            stats["sim_s"] += summary["simulated_s"]
            files = {r["run_key"]: r for r in summary.get("runs", [])}
            for r in job["runs"]:
                info = files.get(r["run_key"])
                conn.execute("DELETE FROM runs WHERE run_key = ? AND status = 'running'", (r["run_key"],))
                if info is None:
                    write_run(conn, _run_row(job, r, "failed", job_meta, summary, "sensor synthesis missing"), None, summary.get("events", []), None)
                    stats["failed"] += 1
                    continue
                with open(info["file"], "rb") as fh:
                    payload = pickle.load(fh)
                qa = payload["qa"]
                status = "completed" if qa["passed"] else "failed"
                row = _run_row(job, r, status, job_meta, summary, "" if qa["passed"] else "QA failed: " + ", ".join(k for k, v in qa["checks"].items() if not v))
                row["qa_passed"] = int(bool(qa["passed"]))
                qa_store = dict(qa, body_params=payload["synth"].get("body_params"))
                write_run(conn, row, payload["synth"], summary.get("events", []), qa_store, compress_ranges=self.compress)
                if not self.keep_run_files:
                    Path(info["file"]).unlink(missing_ok=True)
                stats["completed" if qa["passed"] else "qa_failed"] += 1
                m = qa["metrics"]
                self.log(
                    f"[campaign] {stats['done_traj']}/{len(jobs)} {r['run_key']}: {status} | {summary['simulated_s']:.1f} s sim, "
                    f"RTF {summary['real_time_factor']:.1f}x | LiDAR {m['lidar_rate_hz_median']:.2f} Hz, IMU {m['imu_rate_hz_median']:.1f} Hz, "
                    f"IMU/scan {m['imu_per_interval_min']}-{m['imu_per_interval_max']} | valid {m['valid_range_pct']:.1f} % | "
                    f"DB {self._db_size() / 1e6:.0f} MB"
                )
            elapsed = time.time() - stats["wall_start"]
            done_frac = stats["done_traj"] / max(1, len(jobs))
            self.log(f"[campaign] progress {100 * done_frac:.0f} % | simulated {stats['sim_s'] / 60:.1f} min in {elapsed / 60:.1f} min wall "
                     f"({stats['sim_s'] / max(elapsed, 1e-9):.1f}x real time) | ETA {elapsed / max(done_frac, 1e-9) * (1 - done_frac) / 60:.1f} min")
        if self.stop:
            conn.execute("UPDATE runs SET status = 'interrupted' WHERE status = 'running'")
        result = finalize(conn)
        conn.close()
        result.update({k: v for k, v in stats.items() if k != "wall_start"})
        result["wall_s"] = time.time() - stats["wall_start"]
        result["interrupted"] = self.stop
        result["database_bytes"] = self._db_size()
        return result

    def _db_size(self) -> int:
        total = 0
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(self.db_path) + suffix)
            if p.exists():
                total += p.stat().st_size
        return total
