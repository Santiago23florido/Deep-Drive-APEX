"""Race driver of the car on an unseen track: explore, map, plan, race (ROS-independent).

The ROS node (``nodes/race_driver_node.py``) and the 2D development harness
(``tools/analysis/race_harness_2d.py``) feed this class with the same inputs,
all of them produced by the car itself:

* the learned odometry (pose in ``odom``, latency-compensated by the caller),
* the de-skewed LiDAR revolutions with the odometry pose at their stamp,
* slam_toolbox's correction ``map -> odom``, its occupancy grid and its graph.

Phases::

    WAIT -> EXPLORE (lap 1, reactive, the map is being built)
         -> CLOSING (start line crossed: keep driving reactively so the SLAM
                     closes the loop)
         -> PLANNING (race line computed on the map, still reactive)
         -> RACE (pure pursuit on the race line, laps counted on the map pose)
         -> FINAL -> DONE
    REACTIVE_ONLY: planning failed, the laps are finished reactively.

No track, no reference path and no ground truth enter here.
"""

from __future__ import annotations

from dataclasses import asdict
import math
import time
from typing import Any, Callable

import numpy as np

from .lap_closure import LapCounter, cut_loop
from .local_cloud import ScanAccumulator
from .map_metrics import se2_apply, se2_compose, se2_inverse
from .pose_correction import CorrectionFilter
from .race_config import RaceDriverConfig
from .race_planner import Grid, RacePlan, plan_race
from .reactive import LocalGrid, ReactiveCommand, ReactiveDriver, pursue
from .sim_paths import ensure_tools_path

ensure_tools_path()
from pose_dataset.controller import PurePursuit  # noqa: E402

WAIT, EXPLORE, CLOSING, PLANNING, RACE, REACTIVE_ONLY, FINAL, DONE = "wait", "explore", "closing", "planning", "race", "reactive_only", "final", "done"

LOG_COLUMNS = ["t", "phase", "x_map", "y_map", "yaw_map", "x_odom", "y_odom", "yaw_odom", "speed", "v_cmd", "steer_deg", "pose_age_s",
               "corr_raw_dx", "corr_raw_dy", "corr_raw_dyaw", "corr_app_dx", "corr_app_dy", "corr_app_dyaw",
               "mode", "planner", "left_wall", "right_wall", "d_wall", "d_target", "clr_min", "n_feasible", "react_ms",
               "lap_own", "travel", "s_race", "lat_race", "guided"]


class ImmediateFuture:
    """Synchronous stand-in for concurrent.futures.Future (harness, tests)."""

    def __init__(self, fn: Callable, *args: Any) -> None:  # noqa: ANN401
        try:
            self._result, self._exc = fn(*args), None
        except Exception as exc:  # noqa: BLE001
            self._result, self._exc = None, exc

    def done(self) -> bool:
        return True

    def result(self):  # noqa: ANN201
        if self._exc is not None:
            raise self._exc
        return self._result


def run_planner(grid: Grid, loop_xy: np.ndarray, start: tuple[float, float, float], side_log: dict, cfg: RaceDriverConfig) -> RacePlan:
    """Top-level (picklable) planning job."""
    return plan_race(grid, loop_xy, start, side_log, cfg.vehicle, cfg.planner, cfg.race, cfg.reactive.w_min_m)


class RaceDriverCore:
    def __init__(self, cfg: RaceDriverConfig, laps: int, submit: Callable[..., Any] | None = None, static_start_s: float = 1.5,
                 base_to_laser: tuple[float, float, float] = (0.18, 0.0, 0.0)) -> None:
        self.cfg = cfg
        self.laps = int(laps)
        self.submit = submit or (lambda fn, *a: ImmediateFuture(fn, *a))
        self.static_start_s = static_start_s
        self.base_to_laser = base_to_laser
        r = cfg.reactive
        self.acc = ScanAccumulator(r.accum_window_s, r.accum_distance_m, r.range_min_m, r.range_max_m, r.voxel_m)
        self.reactive = ReactiveDriver(cfg.vehicle, r)
        self.corr = CorrectionFilter(cfg.correction)
        self.phase = WAIT
        self.ready_since: float | None = None
        self.counter: LapCounter | None = None
        self.start: tuple[float, float, float] | None = None
        self.travel = 0.0
        self._last_odom_xy: tuple[float, float] | None = None
        self.cmd: ReactiveCommand | None = None
        self.local_path_odom: np.ndarray | None = None  # rear-axle path of the last reactive plan, odom frame
        self.state = None
        self.map_odom_raw: tuple[float, float, float] | None = None
        self.map_odom: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.pose_odom: tuple[float, float, float] | None = None
        self.pose_map: tuple[float, float, float] | None = None
        self.grid: tuple[float, Grid] | None = None
        self.graph: tuple[float, np.ndarray] | None = None
        self.path_log: list[tuple[float, float]] = []
        self.side_log: dict[str, list] = {"xy": [], "left": [], "right": []}
        self.close_travel: float | None = None
        self.close_t: float | None = None
        self.future = None
        self.plan: RacePlan | None = None
        self.pursuit: PurePursuit | None = None
        self.guided_until = -1.0
        self.guided = False
        self.steer = 0.0
        self.v_out = 0.0
        self.last_t: float | None = None
        self.corr_log: list[tuple[float, float]] = []  # (t, |raw - applied|) during closing
        self.events: dict[str, Any] = {"crossings": [], "guard": [], "config": asdict(cfg), "laps": self.laps}
        self.react_ms = 0.0
        self.s_race = float("nan")
        self._ahead_idx = np.zeros(0, dtype=int)  # plan stations of the last _race_line_in
        self.lat_race = float("nan")

    # ----------------------------------------------------------------- inputs
    def on_map(self, t: float, grid: Grid) -> None:
        self.grid = (t, grid)

    def on_graph(self, t: float, vertices: np.ndarray) -> None:
        self.graph = (t, np.asarray(vertices, dtype=float))

    def on_scan(self, t: float, ranges: np.ndarray, angle_min: float, angle_inc: float, pose_odom_at_stamp: tuple[float, float, float],
                pose_odom_now: tuple[float, float, float], speed: float) -> None:
        """A de-skewed revolution (pose of base_link at its stamp in odom) and the current pose."""
        veh, m = self.cfg.vehicle, self.cfg.reactive.body_margin_m
        # The car's own returns (a wheel at ~0.15 m) are fixed to the car, but the de-skewing moves each beam with the
        # motion between its sample and the stamp: up to one revolution of travel ahead. Nothing real can be there.
        ahead = max(0.0, speed) * self.cfg.reactive.lidar_period_s
        body = (-0.5 * veh.length_m - m, 0.5 * veh.length_m + m + ahead, -0.5 * veh.width_m - m, 0.5 * veh.width_m + m)
        self.acc.add_scan(int(t * 1e9), ranges, angle_min, angle_inc, pose_odom_at_stamp, self.base_to_laser, body)
        if self.phase in (WAIT, FINAL, DONE):
            return
        t0 = time.perf_counter()
        cloud = self.acc.cloud_in(pose_odom_now)
        guide = None
        v_cap = None
        if self.phase == RACE and not self._guard(t, cloud, pose_odom_now, speed):
            self.react_ms = (time.perf_counter() - t0) * 1e3
            return
        if self.phase in (RACE, PLANNING) and self.plan is not None:
            guide = self._race_line_in(pose_odom_now)
        cmd, state = self.reactive.step(cloud, pose_odom_now, self.travel, speed, guide, v_cap)
        self.cmd, self.state = cmd, state
        self.local_path_odom = se2_apply(pose_odom_now, cmd.path) if len(cmd.path) else None
        if self.phase in (EXPLORE, CLOSING) and self.pose_map is not None:
            self.side_log["xy"].append(self.pose_map[:2])
            self.side_log["left"].append(bool(state.left_wall))
            self.side_log["right"].append(bool(state.right_wall))
        self.react_ms = (time.perf_counter() - t0) * 1e3

    # ------------------------------------------------------------ race helpers
    def _race_line_in(self, pose_odom_now: tuple[float, float, float]) -> np.ndarray:
        """Race line points ahead of the car in the base frame (guide of the reactive planner)."""
        base_in_map = se2_compose(self.map_odom, pose_odom_now)
        pts = self.plan.path.xy
        d = np.hypot(pts[:, 0] - base_in_map[0], pts[:, 1] - base_in_map[1])
        k0 = int(np.argmin(d))
        idx = (k0 + np.arange(0, 60)) % len(pts)
        self._ahead_idx = idx
        return se2_apply(se2_inverse(base_in_map), pts[idx])

    def _guard(self, t: float, cloud: np.ndarray, pose_odom_now: tuple[float, float, float], speed: float) -> bool:
        """Live check of the race line ahead against the LiDAR (independent of the SLAM).
        Returns True when the reactive planner must take over (guided)."""
        rc, veh = self.cfg.race, self.cfg.vehicle
        ahead = self._race_line_in(pose_odom_now)
        seg = np.hypot(np.diff(ahead[:, 0]), np.diff(ahead[:, 1]))
        acc = np.concatenate(([0.0], np.cumsum(seg)))
        reach = speed * speed / (2.0 * rc.a_dec_mps2) + 0.2 * speed + 0.5
        keep = acc <= reach
        if keep.sum() < 2:
            return t < self.guided_until
        pts = ahead[keep]
        head = np.arctan2(np.gradient(pts[:, 1]), np.gradient(pts[:, 0]))
        grid = LocalGrid(cloud)
        clr = np.full(len(pts), np.inf)
        for off in veh.circle_offsets:
            clr = np.minimum(clr, grid.query(pts + off * np.column_stack((np.cos(head), np.sin(head)))) - veh.circle_radius)
        expected = self.plan.clearance[self._ahead_idx[keep]] if len(self.plan.clearance) == len(self.plan.path.xy) else np.full(len(clr), np.inf)
        # A new or moved obstacle: well below what the map promised. Small disagreements are map / cloud noise.
        trip = (clr < rc.guard_clearance_m) & (clr < expected - rc.guard_drop_m)
        if np.any(trip):
            if t >= self.guided_until:
                self.events["guard"].append(self._guard_record(t, pts, clr, expected, trip, pose_odom_now, speed))
            self.guided_until = t + rc.guard_hold_s
        return t < self.guided_until

    def _guard_record(self, t: float, pts: np.ndarray, clr: np.ndarray, expected: np.ndarray, trip: np.ndarray,
                      pose_odom_now: tuple[float, float, float], speed: float) -> dict:
        """What tripped the guard (diagnostics): the line point, the stored points near it (base frame) and their scans."""
        i = int(np.argmin(np.where(trip, clr, np.inf)))
        near: list = []
        scans: list = []
        inv = se2_inverse(pose_odom_now)
        for stamp, _, p in self.acc.scans:
            q = se2_apply(inv, p)
            d = np.hypot(q[:, 0] - pts[i, 0], q[:, 1] - pts[i, 1])
            m = d < self.cfg.vehicle.circle_radius + 0.25
            if m.any():
                scans.append([round(stamp * 1e-9 - t, 3), int(m.sum())])
                near.extend(np.round(q[m], 3).tolist())
        return {"t": t, "clearance_m": float(clr.min()), "expected_m": float(expected[i]), "line_pt_base": np.round(pts[i], 3).tolist(),
                "speed": float(speed), "pose_map": [round(float(v), 3) for v in se2_compose(self.map_odom, pose_odom_now)],
                "points_base": near[:40], "scans_rel_t_count": scans}

    def _start_race(self, rear_map: tuple[float, float, float]) -> None:
        veh = self.cfg.vehicle
        self.pursuit = PurePursuit(self.plan.path, veh.wheelbase_m, veh.controller_cfg(), 1.0)
        self.pursuit.reset(rear_map[0], rear_map[1])
        self.phase = RACE

    def _handover_ok(self, rear_map: tuple[float, float, float]) -> bool:
        p = self.plan.path
        k = int(np.argmin(np.hypot(p.xy[:, 0] - rear_map[0], p.xy[:, 1] - rear_map[1])))
        lat = math.hypot(p.xy[k, 0] - rear_map[0], p.xy[k, 1] - rear_map[1])
        dh = abs(math.atan2(math.sin(rear_map[2] - p.heading[k]), math.cos(rear_map[2] - p.heading[k])))
        ok = lat <= self.cfg.race.handover_max_lat_m and math.degrees(dh) <= self.cfg.race.handover_max_head_deg
        if ok:
            self.events["handover"] = {"lateral_m": lat, "heading_deg": math.degrees(dh)}
        return ok

    def _lap_path(self) -> tuple[np.ndarray, str]:
        if self.graph is not None and len(self.graph[1]) >= 10:
            cut = cut_loop(self.graph[1], self.start, self.cfg.closure)
            if cut is not None:
                return cut[0], "graph"
        cut = cut_loop(np.asarray(self.path_log), self.start, self.cfg.closure)
        if cut is not None:
            return cut[0], "log"
        return np.asarray(self.path_log), "log_uncut"

    def _submit_plan(self, t: float) -> None:
        loop, source = self._lap_path()
        grid = self.grid[1]
        side = {k: np.asarray(v) for k, v in self.side_log.items()}
        self.events["planning"] = {"t_submit": t, "lap_path_source": source, "lap_path_points": len(loop), "map_stamp": self.grid[0],
                                   "max_correction_step_m": max((c for _, c in self.corr_log), default=0.0)}
        self.plan_inputs = (grid, loop, side)
        self.future = self.submit(run_planner, grid, loop, self.start, side, self.cfg)
        self.phase = PLANNING

    # ---------------------------------------------------------------- control
    def control(self, t: float, pose_odom: tuple[float, float, float], speed: float, map_odom_raw: tuple[float, float, float] | None,
                ready: bool, pose_age: float = 0.0) -> tuple[float, float, list | None]:
        """50 Hz step: (speed command [m/s], steering [rad], log row)."""
        cfg, veh = self.cfg, self.cfg.vehicle
        dt = 0.0 if self.last_t is None else max(0.0, t - self.last_t)
        self.last_t = t
        if self._last_odom_xy is not None:
            self.travel += math.hypot(pose_odom[0] - self._last_odom_xy[0], pose_odom[1] - self._last_odom_xy[1])
        self._last_odom_xy = pose_odom[:2]
        self.pose_odom = pose_odom
        if map_odom_raw is not None:
            self.map_odom_raw = map_odom_raw
            prev = self.map_odom
            self.map_odom = self.corr.update(t, map_odom_raw, pose_odom)
            if self.phase == CLOSING:
                a, b = se2_compose(map_odom_raw, pose_odom), se2_compose(prev, pose_odom)
                self.corr_log.append((t, math.hypot(a[0] - b[0], a[1] - b[1])))
        pose_map = se2_compose(self.map_odom, pose_odom)
        self.pose_map = pose_map
        rear_map = se2_compose(pose_map, (veh.rear_offset_m, 0.0, 0.0))
        v_cmd, steer = 0.0, self.steer

        if self.phase == WAIT:
            if ready and self.map_odom_raw is not None and not self.acc.empty:
                self.ready_since = t if self.ready_since is None else self.ready_since
                if t - self.ready_since >= self.static_start_s:
                    self.phase = EXPLORE
                    self.start = pose_map
                    self.counter = LapCounter(pose_map, cfg.closure, t, self.travel)
                    self.events["t_explore"] = t
                    self.events["start_map"] = list(pose_map)
            return self._out(t, 0.0, 0.0, speed, pose_age)

        self.path_log.append(pose_map[:2])
        crossing = self.counter.update(t, pose_map, self.travel) if self.counter is not None else None
        if crossing is not None:
            self.events["crossings"].append(asdict(crossing))
        if crossing is not None and self.counter.laps >= self.laps:
            self.phase = FINAL
        elif crossing is not None and self.phase == EXPLORE:
            self.phase, self.close_travel, self.close_t = CLOSING, self.travel, t
            self.events["closing"] = {"t": t, "travel": self.travel}

        if self.phase == CLOSING and self.travel - self.close_travel >= cfg.closure.overlap_m:
            t_ov = self.events.setdefault("t_overlap_done", t)
            # The first map published after the overlap (the loop is closed in it), or the last one after a wait.
            if self.grid is not None and (self.grid[0] > t_ov or t - t_ov >= cfg.closure.map_fresh_wait_s):
                self._submit_plan(t)

        if self.phase == PLANNING and self.future is not None and self.future.done():
            try:
                self.plan = self.future.result()
                self.events["plan"] = {**self.plan.diag, "t_done": t}
                self.future = None
            except Exception as exc:  # noqa: BLE001
                self.events["plan_error"] = repr(exc)
                self.future, self.phase = None, REACTIVE_ONLY
        if self.phase == PLANNING and self.plan is not None and self._handover_ok(rear_map):
            self.events["t_race"] = t
            self._start_race(rear_map)

        self.guided = False
        if self.phase == RACE and t >= self.guided_until:
            s_tot, lat = self.pursuit.locate(rear_map[0], rear_map[1])
            self.s_race, self.lat_race = s_tot, lat
            steer = self.pursuit.steering(rear_map[0], rear_map[1], rear_map[2], speed)
            s_mod = float(self.plan.path.s[self.pursuit.idx])
            v_cmd = self.plan.speed_at(s_mod + max(0.3, 0.3 * speed))
            if self.counter.laps == self.laps - 1:  # last lap: brake to stop just past the start line
                dist = (self.plan.path.length - s_mod) % self.plan.path.length
                if dist > 0.5 * self.plan.path.length:
                    dist = 0.0
                v_cmd = min(v_cmd, math.sqrt(2.0 * cfg.race.a_dec_mps2 * (dist + cfg.race.finish_margin_m)))
        elif self.phase in (EXPLORE, CLOSING, PLANNING, REACTIVE_ONLY, RACE):
            self.guided = self.phase in (RACE, PLANNING) and self.plan is not None
            if self.phase == RACE:
                self.pursuit.reset(rear_map[0], rear_map[1])  # resume the line after the guided section
            if self.cmd is not None and self.local_path_odom is not None:
                rear_odom = se2_compose(pose_odom, (veh.rear_offset_m, 0.0, 0.0))
                ld = max(cfg.reactive.local_path_lookahead_m, 0.35 + 0.32 * speed)
                steer = pursue(self.local_path_odom, rear_odom, ld, veh.wheelbase_m)
                v_cmd = self.cmd.v
        if self.phase in (FINAL, DONE):
            v_cmd = 0.0
            if abs(speed) < 0.02:
                if self.phase != DONE:
                    self.events["t_done"] = t
                self.phase = DONE
        # Actuator-friendly commands: steering rate and speed-increase limits.
        rate = math.radians(cfg.reactive.steer_rate_dps if self.phase != RACE else veh.servo_rate_dps) * max(dt, 1e-3)
        lim = math.atan(veh.kappa_max * veh.wheelbase_m) if self.phase != RACE else math.radians(veh.steer_max_deg)
        steer = max(-lim, min(lim, steer))
        self.steer = self.steer + max(-rate, min(rate, steer - self.steer))
        a_up = cfg.reactive.a_acc_mps2 if self.phase != RACE else cfg.race.a_acc_mps2
        self.v_out = min(v_cmd, self.v_out + a_up * max(dt, 1e-3)) if v_cmd > self.v_out else v_cmd
        return self._out(t, self.v_out, self.steer, speed, pose_age)

    def steer_command(self, steer: float) -> float:
        """Steering angle to command so the linkage (right turns at ``right_ratio``) delivers ``steer``."""
        veh = self.cfg.vehicle
        cmd = steer / veh.right_ratio if steer < 0.0 else steer
        lim = math.radians(veh.steer_max_deg)
        return max(-lim, min(lim, cmd))

    def _out(self, t: float, v: float, steer: float, speed: float, age: float) -> tuple[float, float, list]:
        pm = self.pose_map or (float("nan"),) * 3
        po = self.pose_odom or (float("nan"),) * 3
        raw = self.map_odom_raw or (float("nan"),) * 3
        st, d = self.state, (self.cmd.diag if self.cmd is not None else {})
        row = [f"{t:.4f}", self.phase, f"{pm[0]:.4f}", f"{pm[1]:.4f}", f"{pm[2]:.5f}", f"{po[0]:.4f}", f"{po[1]:.4f}", f"{po[2]:.5f}",
               f"{speed:.3f}", f"{v:.3f}", f"{math.degrees(steer):.3f}", f"{age:.4f}",
               f"{raw[0]:.4f}", f"{raw[1]:.4f}", f"{raw[2]:.5f}", f"{self.map_odom[0]:.4f}", f"{self.map_odom[1]:.4f}", f"{self.map_odom[2]:.5f}",
               st.mode if st is not None else "", d.get("planner", ""), int(bool(st and st.left_wall)), int(bool(st and st.right_wall)),
               f"{d.get('d_wall', float('nan')):.3f}", f"{d.get('d_target', float('nan')):.3f}", f"{d.get('clr_min', float('nan')):.3f}",
               d.get("n_feasible", ""), f"{self.react_ms:.1f}", self.counter.laps if self.counter else 0, f"{self.travel:.3f}",
               f"{self.s_race:.3f}", f"{self.lat_race:.4f}", int(self.guided)]
        return v, steer, row

    def status(self) -> dict:
        d = self.cmd.diag if self.cmd is not None else {}
        return {"phase": self.phase, "lap_own": self.counter.laps if self.counter else 0, "laps": self.laps, "travel_m": self.travel,
                "mode": self.state.mode if self.state else "", "d_wall": d.get("d_wall"), "d_target": d.get("d_target"),
                "clearance_m": d.get("clr_min"), "planner": d.get("planner"), "react_ms": self.react_ms,
                "plan_ready": self.plan is not None, "guided": self.guided}
