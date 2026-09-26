#!/usr/bin/env python3
"""Fast 2D closed loop of the race driver (development only, uses the truth to simulate).

    python3 tools/analysis/race_harness_2d.py --track test_unseen --seeds 6 7 8 9 --laps 3

Runs the car-side logic of the live race driver (``RaceDriverCore``: reactive
lap 1, lap closure, planning on the map, race laps) against a simplified
world, in seconds instead of Gazebo at a real-time factor of 0.5:

* **LiDAR**: exact ray casting of the track geometry at the scan plane
  (curbs below it are not seen), 360 beams at 13 Hz, the rear sector
  145-205 deg occluded, range noise and dropouts; delivered with the
  estimator latency.
* **Odometry**: the true motion with a scale error and a heading drift (the
  magnitudes of the learned odometry), plus noise.
* **SLAM**: ``map -> odom`` brings the odometry back to the truth with a few
  centimetres of noise, at 5 Hz; the occupancy grid is ray-traced from the
  scans at the SLAM poses (with isolated phantom cells); the graph vertices
  are the SLAM poses every 0.25 m.
* **Car**: the dataset actuators (ESC / servo) and a kinematic bicycle.

The spawn pose, the actuator perturbation and the geometry come from the run
format (track, motion, seed) exactly as in the live simulation. The verdict
(collision with walls, curbs or pillars; lap times; clearances) uses the
truth. It is a tuning tool: the live run in Gazebo is the validation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

SIM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM / "ros2_ws" / "src" / "apex_fusion_research"))
sys.path.insert(0, str(SIM / "tools"))

from apex_fusion_research.core.map_metrics import se2_apply, se2_compose, se2_inverse  # noqa: E402
from apex_fusion_research.core.race_config import RaceDriverConfig, vehicle_limits  # noqa: E402
from apex_fusion_research.core.race_driver import DONE, LOG_COLUMNS, RaceDriverCore  # noqa: E402
from apex_fusion_research.core.race_eval import TrueTrack, plan_truth_metrics  # noqa: E402
from apex_fusion_research.core.race_planner import Grid  # noqa: E402
from apex_fusion_research.nodes._pose_tools import run_setup  # noqa: E402
from pose_dataset.controller import PurePursuit  # noqa: E402
from pose_dataset.geometry import FootprintChecker, rect_corners  # noqa: E402
from pose_dataset.raycast import RayCaster  # noqa: E402
from pose_dataset.vehicle import Actuators  # noqa: E402

BEAMS, SCAN_HZ, LIDAR_Z = 360, 13.03, 0.12


class MapBuilder:
    """Occupancy grid ray-traced from the scans at the SLAM poses."""

    def __init__(self, lo: np.ndarray, hi: np.ndarray, res: float = 0.05) -> None:
        self.res, self.ox, self.oy = res, float(lo[0]), float(lo[1])
        self.w, self.h = int((hi[0] - lo[0]) / res) + 1, int((hi[1] - lo[1]) / res) + 1
        self.hit = np.zeros((self.h, self.w), dtype=np.int32)
        self.free = np.zeros((self.h, self.w), dtype=np.int32)

    def _cells(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = np.floor((xy[:, 0] - self.ox) / self.res).astype(int)
        r = np.floor((xy[:, 1] - self.oy) / self.res).astype(int)
        ok = (r >= 0) & (r < self.h) & (c >= 0) & (c < self.w)
        return r[ok], c[ok], ok

    def add(self, pose: tuple[float, float, float], ranges: np.ndarray, angles: np.ndarray, free_max: float = 5.0) -> None:
        ok = np.isfinite(ranges)
        c, s = math.cos(pose[2]), math.sin(pose[2])
        a = angles + pose[2]
        hits = np.column_stack((pose[0] + ranges[ok] * np.cos(a[ok]), pose[1] + ranges[ok] * np.sin(a[ok])))
        r, cc, _ = self._cells(hits)
        np.add.at(self.hit, (r, cc), 1)
        rr = np.where(ok, np.minimum(ranges, free_max), free_max) - 0.5 * self.res
        steps = np.arange(0.1, free_max, 0.5 * self.res)
        m = steps[None, :] < rr[:, None]
        d = np.broadcast_to(steps[None, :], m.shape)[m]
        ang = np.broadcast_to(a[:, None], m.shape)[m]
        pts = np.column_stack((pose[0] + d * np.cos(ang), pose[1] + d * np.sin(ang)))
        r, cc, _ = self._cells(pts)
        np.add.at(self.free, (r, cc), 1)
        del c, s

    def grid(self, rng: np.random.Generator, phantoms: int = 60) -> Grid:
        data = np.full((self.h, self.w), -1, dtype=np.int16)
        data[self.free > 0] = 0
        data[(self.hit >= 1) & (self.hit >= 0.3 * self.free)] = 100
        free_r, free_c = np.nonzero(data == 0)
        if phantoms and len(free_r):  # isolated false returns (scan plane grazing the floor)
            k = rng.choice(len(free_r), size=min(phantoms, len(free_r)), replace=False)
            data[free_r[k], free_c[k]] = 100
        return Grid(data, self.res, self.ox, self.oy)


def run_one(track_name: str, motion: str, seed: int, laps: int, cfg: RaceDriverConfig, out_dir: Path | None, max_sim_s: float) -> dict:
    rs = run_setup(track_name, motion, seed, laps)
    track, vcfg = rs["track"], rs["vcfg"]
    rng = np.random.default_rng(1000 + seed)
    veh = cfg.vehicle
    caster = RayCaster(track.geometry, max_range=12.0)
    checker = FootprintChecker(track.geometry)
    truth_tt = TrueTrack(track, veh)
    act = Actuators(vcfg, rs["setup"]["actuator_perturbation"])
    ref_locator = PurePursuit(rs["setup"]["path"], veh.wheelbase_m, vcfg["controller"], 1.0)
    lap_len = rs["setup"]["path"].length
    # truth: base_link pose
    bx, by, byaw = rs["spawn"]
    t_world_map = (bx, by, byaw)  # the map frame is the car's pose at its first scan (it stands still)
    # odometry errors of the learned model (magnitudes of the closed-loop runs)
    scale_err = float(rng.choice([-1, 1])) * 0.008
    drift = float(rng.choice([-1, 1])) * math.radians(0.06)
    odom = (0.0, 0.0, 0.0)
    angles = -math.pi + 2 * math.pi / BEAMS * np.arange(BEAMS)
    occluded = (np.degrees(angles) % 360 > 145) & (np.degrees(angles) % 360 < 205)
    track_xy = track.centerline.xy
    half = float(np.max(np.hypot(track_xy[:, 0] - bx, track_xy[:, 1] - by))) + 6.0  # the map frame is rotated: a square around the start
    mb = MapBuilder(np.array([-half, -half]), np.array([half, half]))
    core = RaceDriverCore(cfg, laps, static_start_s=1.5)
    dt, t = 0.002, 0.0
    next_ctrl, next_scan, next_slam, next_map = 0.0, 0.05, 0.0, 0.5
    scan_queue: list = []  # (release time, stamp, ranges, odom pose at stamp)
    slam_queue: list = []  # (release time, map->odom)
    graph: list = []
    last_graph_xy = None
    map_odom_raw = None
    log_rows, truth_rows = [], []
    verdict, reason = "timeout", ""
    min_clr = math.inf
    lap_marks: list[float] = []
    t_move = None
    stuck_since = None
    t_wall0 = time.perf_counter()
    v_true = 0.0
    a_body = 0.0
    while t < max_sim_s:
        # --- truth motion (rear-axle bicycle)
        act.step(dt)
        a_body += (dt / 0.1) * ((act.speed - v_true) / dt - a_body)  # longitudinal acceleration, 0.1 s lag
        v_true = act.speed
        delta = math.radians(act.steer_deg)
        rear = (bx + veh.rear_offset_m * math.cos(byaw), by + veh.rear_offset_m * math.sin(byaw))
        w = v_true * math.tan(delta) / veh.wheelbase_m
        mid = byaw + 0.5 * w * dt
        rear = (rear[0] + v_true * dt * math.cos(mid), rear[1] + v_true * dt * math.sin(mid))
        new_yaw = byaw + w * dt
        nbx, nby = rear[0] - veh.rear_offset_m * math.cos(new_yaw), rear[1] - veh.rear_offset_m * math.sin(new_yaw)
        # odometry increment with its errors
        inc = se2_compose(se2_inverse((bx, by, byaw)), (nbx, nby, new_yaw))
        dist = math.hypot(inc[0], inc[1])
        odom = se2_compose(odom, (inc[0] * (1 + scale_err), inc[1] * (1 + scale_err), inc[2] + drift * dist + rng.normal(0, 2e-5)))
        bx, by, byaw = nbx, nby, new_yaw
        t += dt
        truth_map = se2_compose(se2_inverse(t_world_map), (bx, by, byaw))
        # --- SLAM correction at 5 Hz (truth + noise), delivered 0.2 s later
        if t >= next_slam:
            next_slam += 0.2
            noisy = (truth_map[0] + rng.normal(0, 0.03), truth_map[1] + rng.normal(0, 0.03), truth_map[2] + rng.normal(0, math.radians(0.3)))
            slam_queue.append((t + 0.2, se2_compose(noisy, se2_inverse(odom))))
            if last_graph_xy is None or math.hypot(noisy[0] - last_graph_xy[0], noisy[1] - last_graph_xy[1]) >= 0.25:
                graph.append(noisy[:2])
                last_graph_xy = noisy[:2]
        while slam_queue and slam_queue[0][0] <= t:
            map_odom_raw = slam_queue.pop(0)[1]
        # --- LiDAR revolution at 13 Hz, delivered with the estimator latency
        if t >= next_scan:
            next_scan += 1.0 / SCAN_HZ
            lx, ly = bx + 0.18 * math.cos(byaw), by + 0.18 * math.sin(byaw)
            dirs = np.column_stack((np.cos(angles + byaw), np.sin(angles + byaw), np.zeros(BEAMS)))
            r = caster.cast(np.tile([lx, ly, LIDAR_Z], (BEAMS, 1)), dirs, 12.0)
            # Body pitch on the suspension (0.6 deg per m/s^2, nose down when braking): the laser plane hits the floor.
            slope = math.tan(math.radians(0.6 * a_body)) * -np.cos(angles)
            floor = np.where(slope > 1e-4, LIDAR_Z / np.maximum(slope, 1e-4), np.inf)
            r = np.minimum(r, floor)
            r = r + rng.normal(0, 1, BEAMS) * (0.001 + 0.003 * np.where(np.isfinite(r), r, 0))
            r[occluded | (rng.random(BEAMS) < 0.03) | (r < 0.15)] = np.inf
            self_hit = (np.abs(np.degrees(angles) - 97.5) < 1.0) & (rng.random(BEAMS) < 0.45)  # the car's own wheel (as in Gazebo)
            r[self_hit] = 0.155
            u = rng.random(BEAMS)  # spurious returns, harsh sensor profile (sensors.yaml): short 1 %, random 0.6 %
            short = np.isfinite(r) & (u < 0.01)
            r[short] = np.minimum(r[short], 0.15 + rng.exponential(1.0, int(short.sum())))
            rand = np.isfinite(r) & (u >= 0.01) & (u < 0.016)
            r[rand] = rng.uniform(0.15, 12.0, int(rand.sum()))
            scan_queue.append((t + 0.15, t, r, odom))
            slam_pose = se2_compose(truth_map, (0.18, 0.0, 0.0))
            mb.add(slam_pose, r, angles)
        while scan_queue and scan_queue[0][0] <= t:
            _, stamp, r, od = scan_queue.pop(0)
            core.on_scan(stamp, r, -math.pi, 2 * math.pi / BEAMS, od, odom, v_true)
        if t >= next_map:
            next_map += 0.5
            if core.phase in ("closing", "planning"):
                core.on_map(t, mb.grid(rng))
                core.on_graph(t, np.asarray(graph))
        # --- control at 50 Hz
        if t >= next_ctrl:
            next_ctrl += 0.02
            v_cmd, steer, row = core.control(t, odom, v_true, map_odom_raw, True, 0.0)
            act.set_command(v_cmd, math.degrees(core.steer_command(steer)))
            log_rows.append(row)
            s_true, lat = ref_locator.locate(bx + veh.rear_offset_m * math.cos(byaw), by + veh.rear_offset_m * math.sin(byaw))
            truth_rows.append((t, bx, by, byaw, v_true, s_true, core.phase))
            clr = truth_tt.clear(bx, by, byaw)
            min_clr = min(min_clr, clr) if t_move else min_clr
            if v_true > 0.1 and t_move is None:
                t_move = t
            while s_true >= (len(lap_marks) + 1) * lap_len:
                lap_marks.append(t)
            hit = checker.collides(rect_corners(bx, by, byaw, veh.length_m, veh.width_m))
            if hit:
                verdict, reason = "failed", f"collision with {hit}"
                break
            if t_move and v_true < 0.05 and core.phase not in (DONE, "final"):
                stuck_since = stuck_since or t
                if t - stuck_since > 8.0:
                    verdict, reason = "failed", f"stuck in {core.phase}"
                    break
            else:
                stuck_since = None
            if core.phase == DONE:
                verdict = "completed"
                break
    wall_s = time.perf_counter() - t_wall0
    res = {"track": track_name, "motion": motion, "seed": seed, "laps": laps, "status": verdict, "failure_reason": reason, "t_end_s": t,
           "wall_s": wall_s, "true_progress_m": float(ref_locator.s_total), "lap_len_m": lap_len, "min_true_clearance_m": min_clr,
           "odometry": {"scale_err": scale_err, "drift_deg_per_m": math.degrees(drift)}}
    t0 = t_move or 0.0
    marks = [t0] + lap_marks
    res["true_lap_times_s"] = [b - a for a, b in zip(marks, marks[1:])]
    ev = core.events
    res["crossings"] = [{k: c[k] for k in ("t", "lap", "lateral", "kind")} for c in ev.get("crossings", [])]
    tr = np.array([(a, b, c, d, e, f) for a, b, c, d, e, f, _ in truth_rows])
    for c in res["crossings"]:  # where the car was (truth) when it believed it crossed the line
        k = int(np.searchsorted(tr[:, 0], c["t"]))
        k = min(k, len(tr) - 1)
        c["true_progress_error_m"] = float(tr[k, 5] - c["lap"] * lap_len)
    if core.plan is not None:
        p = core.plan
        res["plan"] = {k: p.diag[k] for k in ("race_length_m", "max_kappa", "min_map_clearance_m", "t_lap_pred_s", "mean_width_m", "forced_stations", "tightened_stations", "compute_s")}
        res["plan"]["lap_path_source"] = ev["planning"]["lap_path_source"]
        res["plan_truth"] = plan_truth_metrics(p.path.xy, p.path.heading, p.ref.xy, p.ref.normal, p.cor.lo, p.cor.hi, t_world_map, truth_tt, p.path.curvature)
        # Where the line is tightest against the real walls and curbs, and what the corridor was there.
        wxy = se2_apply(t_world_map, p.path.xy)
        wyaw = p.path.heading + t_world_map[2]
        clr_line = np.array([truth_tt.clear(x, y, w) for (x, y), w in zip(wxy, wyaw)])
        i = int(np.argmin(clr_line))
        c = p.cor
        res["plan_worst"] = {"s": float(p.path.s[i]), "world": [round(float(v), 2) for v in wxy[i]], "true_clearance_m": float(clr_line[i]),
                             "lo": float(c.lo[i]), "hi": float(c.hi[i]), "alpha": float(p.alpha[i]), "forced": bool(c.forced[i]),
                             "wall_l": bool(c.kind_l[i]), "wall_r": bool(c.kind_r[i]), "d_wall_l": float(c.d_wall_l[i]), "d_wall_r": float(c.d_wall_r[i]),
                             "d_obs_l": float(c.d_obs_l[i]), "d_obs_r": float(c.d_obs_r[i]), "map_clearance_m": float(p.clearance[i]) if len(p.clearance) else None}
    res["guard_events"] = len(ev.get("guard", []))
    res["plan_error"] = ev.get("plan_error")
    phases = np.array([r[6] for r in truth_rows])
    res["lap1_true_min_clearance_m"] = float(min((truth_tt.clear(x, y, w) for (_, x, y, w, _, _), ph in zip(tr[::10], phases[::10]) if ph in ("explore", "closing", "planning")), default=math.nan))
    race_idx = phases == "race"
    res["race_true_min_clearance_m"] = float(min((truth_tt.clear(x, y, w) for (_, x, y, w, _, _) in tr[race_idx][::5]), default=math.nan))
    res["race_max_speed_mps"] = float(tr[race_idx, 4].max()) if race_idx.any() else math.nan
    if out_dir is not None:
        d = out_dir / f"{track_name}__s{seed}"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "driver.csv", "w", encoding="utf-8") as f:
            f.write(",".join(LOG_COLUMNS) + "\n")
            for row in log_rows:
                f.write(",".join(str(v) for v in row) + "\n")
        np.savetxt(d / "truth.csv", tr, delimiter=",", header="t,x,y,yaw,speed,s", comments="")
        (d / "result.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        figure(d / "harness.png", track, tr, phases, core, t_world_map, res, veh)
    return res


def figure(path: Path, track, tr: np.ndarray, phases: np.ndarray, core: RaceDriverCore, t_world_map, res: dict, veh) -> None:  # noqa: ANN001
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from apex_fusion_research.core.clearance import outline_points  # noqa: PLC0415
    from apex_fusion_research.core.map_metrics import se2_apply  # noqa: PLC0415

    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    ink, ink2, surface, wall, curb = "#0b0b0b", "#52514e", "#fcfcfb", "#8a877e", "#d2cfc6"
    fig, ax = plt.subplots(figsize=(11, 8), facecolor=surface)
    ax.set_facecolor(surface)
    for kinds, col, lab in ((("wall", "obstacle"), wall, "paredes y pilares (el LiDAR los ve)"), (("curb",), curb, "bordillos (bajo el plano del LiDAR)")):
        pts, _ = outline_points(track.geometry, kinds=kinds)
        ax.scatter(pts[:, 0], pts[:, 1], s=1.2, color=col, linewidths=0, label=lab)
    lap1 = np.isin(phases, ["explore", "closing", "planning"])
    ax.plot(tr[lap1, 1], tr[lap1, 2], color=blue, lw=1.0, label="vuelta 1 reactiva (verdad)")
    race = np.isin(phases, ["race", "final", "done"])
    ax.plot(np.where(race, tr[:, 1], np.nan), np.where(race, tr[:, 2], np.nan), color=aqua, lw=1.0, label="vueltas planificadas (verdad)")
    if core.plan is not None:
        w = se2_apply(t_world_map, core.plan.path.xy)
        ax.plot(np.r_[w[:, 0], w[:1, 0]], np.r_[w[:, 1], w[:1, 1]], color=orange, lw=1.3, ls=(0, (4, 2)), label="línea planificada (mapa del carro)")
    ax.plot([t_world_map[0]], [t_world_map[1]], marker="o", ms=7, color=ink, ls="none", label="salida")
    ax.set_aspect("equal")
    ax.tick_params(colors=ink2, labelsize=8)
    laps = ", ".join(f"{x:.1f}" for x in res["true_lap_times_s"])
    ax.set_title(f"Harness 2D: {res['track']} semilla {res['seed']} — {res['status']} {res['failure_reason']}", loc="left", fontsize=11, color=ink, pad=16)
    ax.text(0.0, 1.01, f"vueltas [s]: {laps}   holgura real mín.: {100 * res['min_true_clearance_m']:.0f} cm", transform=ax.transAxes, fontsize=8, color=ink2)
    leg = ax.legend(fontsize=8, frameon=False, loc="lower left")
    for h in getattr(leg, "legend_handles", None) or leg.legendHandles:
        if hasattr(h, "set_sizes"):  # scatter points: readable in the legend, the start marker keeps its size
            h.set_sizes([20])
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=surface)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", default="test_unseen")
    ap.add_argument("--motion", default="medium", help="only seeds the spawn / actuator draws of the format")
    ap.add_argument("--seeds", type=int, nargs="+", default=[6, 7, 8, 9])
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--race-line", default="min_curvature", choices=["min_curvature", "centre"])
    ap.add_argument("--max-sim-s", type=float, default=400.0)
    ap.add_argument("--out", type=Path, default=SIM / "data" / "race_harness")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides section.field=value, e.g. reactive.v_explore_mps=1.2")
    a = ap.parse_args()
    from pose_dataset.vehicle import load_vehicle_config  # noqa: PLC0415

    cfg = RaceDriverConfig(vehicle=vehicle_limits(load_vehicle_config()))
    cfg.planner.mode = a.race_line
    for item in a.set:
        key, value = item.split("=", 1)
        sec, field = key.split(".", 1)
        obj = getattr(cfg, sec)
        cur = getattr(obj, field)
        setattr(obj, field, type(cur)(value) if not isinstance(cur, bool) else value.lower() in ("1", "true", "yes"))
    rows = []
    for seed in a.seeds:
        r = run_one(a.track, a.motion, seed, a.laps, cfg, a.out, a.max_sim_s)
        rows.append(r)
        pt = r.get("plan_truth", {})
        print(f"seed {seed}: {r['status']:9s} {r['failure_reason']:28s} laps {['%.1f' % x for x in r['true_lap_times_s']]} "
              f"min clr {100 * r['min_true_clearance_m']:.0f} cm (lap1 {100 * r['lap1_true_min_clearance_m']:.0f}, race {100 * r['race_true_min_clearance_m']:.0f}) "
              f"plan: sound {100 * pt.get('corridor_sound_frac', math.nan):.0f} % line clr {100 * pt.get('line_min_clearance_m', math.nan):.0f} cm "
              f"guard {r['guard_events']}  [{r['wall_s']:.0f} s]", flush=True)
    (a.out / f"summary_{a.track}.json").write_text(json.dumps(rows, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
