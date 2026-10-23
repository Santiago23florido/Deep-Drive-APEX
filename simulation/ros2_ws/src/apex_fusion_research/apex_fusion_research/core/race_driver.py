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

from dataclasses import asdict, replace
import math
import time
from typing import Any, Callable

import numpy as np

from .adaptive import AdaptiveShaper
from .lap_closure import LapCounter, cut_loop
from .local_cloud import ScanAccumulator
from .map_metrics import se2_apply, se2_compose, se2_inverse
from .plant_id import PlantIdentifier
from .pose_correction import CorrectionFilter
from .race_config import RaceDriverConfig
from .race_planner import Grid, RacePlan, plan_race
from .reactive import LocalGrid, ReactiveCommand, ReactiveDriver, pursue
from .recovery import choose_recovery_arc, reverse_path_clear
from .sim_paths import ensure_tools_path

ensure_tools_path()
from pose_dataset.controller import PurePursuit  # noqa: E402

WAIT, EXPLORE, CLOSING, PLANNING, RACE, REACTIVE_ONLY, FINAL, DONE = "wait", "explore", "closing", "planning", "race", "reactive_only", "final", "done"
MAP_GUIDED = "map_guided"  # SLAM route + live local arcs when direct forward tracking is infeasible

LOG_COLUMNS = ["t", "phase", "x_map", "y_map", "yaw_map", "x_odom", "y_odom", "yaw_odom", "speed", "v_cmd", "steer_deg", "pose_age_s",
               "corr_raw_dx", "corr_raw_dy", "corr_raw_dyaw", "corr_app_dx", "corr_app_dy", "corr_app_dyaw",
               "mode", "planner", "left_wall", "right_wall", "d_wall", "d_target", "clr_min", "n_feasible", "react_ms",
               "lap_own", "travel", "s_race", "lat_race", "guided", "lane_known", "v_lane",
               # What the car has worked out about itself (plant_id.py) and what it
               # therefore allows itself (adaptive.py). Appended, never inserted:
               # the analysis tools read these by name.
               "kap_des", "kap_cmd", "kap_meas", "g_left", "g_right", "k_trim", "k_slip", "arm", "g_v",
               "u_dead", "tau_steer", "conf", "conf_arm", "v_env", "recover"]


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
        # What the car knows about its own plant, and what it does with it. Both
        # start at the datasheet, so a car that has learned nothing drives
        # exactly as it did before this existed.
        self.ident = PlantIdentifier(cfg)
        self.shaper = AdaptiveShaper(cfg)
        self.model = self.ident.prior()
        self.env = self.shaper.envelope(self.model)
        self.cmd_out = (0.0, 0.0)
        self.kappa_des = 0.0
        self.recover = 0  # 0 none, 1 creeping forward, 2 backing out
        self.block_since: float | None = None
        self.recover_until = -1.0
        self.recover_travel = 0.0
        self.recover_distance_limit = 0.0
        self.recover_distance = 0.0
        self.recover_xy: tuple[float, float] | None = None
        self.recover_count = 0
        self.recover_stuck_count = 0
        self.recover_progress_origin: tuple[float, float] | None = None
        self.recover_forward_distance = 0.0
        self.recover_kappa = 0.0
        self.recover_path_clear = True
        self.recover_geometric = False
        self.rear_clear = 0.0  # no rear observation has been processed yet
        self.phase = WAIT
        self.ready_since: float | None = None
        self.counter: LapCounter | None = None
        self.start: tuple[float, float, float] | None = None
        self.travel = 0.0
        self.mission_started_t: float | None = None
        self.mission_xy: tuple[float, float] | None = None
        self.mission_distance = 0.0
        self.mission_stop_reason: str | None = None
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
        self.map_route_s: float | None = None
        self.map_route_odom: tuple[float, float] | None = None
        self.guide_started_t: float | None = None
        self.guide_weight = 0.0
        self.lat_race = float("nan")

    # ----------------------------------------------------------------- inputs
    def on_map(self, t: float, grid: Grid) -> None:
        self.grid = (t, grid)

    def on_graph(self, t: float, vertices: np.ndarray) -> None:
        self.graph = (t, np.asarray(vertices, dtype=float))

    def on_odometry(self, stamp: float, pose: tuple[float, float, float],
                    sigma: tuple[float, float, float] | None, healthy: bool = True) -> None:
        """One odometry estimate, at the rate the LiDAR turns.

        This is the only channel the real car has, and the only one the
        identification uses: a pose increment in the odometry frame with the
        sigma the network publishes for it. Nothing here touches the map, so a
        SLAM correction cannot disturb the identification -- the two share no
        signal, which is what keeps them from fighting.
        """
        self.ident.on_estimate(stamp, pose, sigma, healthy)
        self.model = self.ident.model()
        self.env = self.shaper.envelope(self.model)
        # The reference point of every candidate path is the rear axle, and
        # where that is in the odometry frame is something the car measures
        # rather than assumes.
        self.reactive.set_rear_offset(self.model.rear_offset_m())
        self.reactive.set_margins(self.env.wall_margin_add_m, self.env.soft_margin_add_m)
        if self.cfg.adaptive.plan_with_identified_model:
            self._identified_plan_limits()

    def _identified_plan_limits(self) -> None:
        """Plan the same physical curvature that the learned inverse delivers."""
        m, env = self.model, self.env
        v = abs(self.ident.v_meas) if math.isfinite(self.ident.v_meas) else m.v_ref_mps
        left = m.kappa_out(env.kappa_limit_left or env.kappa_limit, v)
        right = -m.kappa_out(-(env.kappa_limit_right or env.kappa_limit), v)
        self.reactive.set_curvature_limits(max(1e-3, left), max(1e-3, right))

    def on_scan(self, t: float, ranges: np.ndarray, angle_min: float, angle_inc: float, pose_odom_at_stamp: tuple[float, float, float],
                pose_odom_now: tuple[float, float, float], speed: float) -> None:
        """A de-skewed revolution (pose of base_link at its stamp in odom) and the current pose."""
        veh, m = self.cfg.vehicle, self.cfg.reactive.body_margin_m
        # The car's own returns (a wheel at ~0.15 m) are fixed to the car, but the de-skewing moves each beam with the
        # motion between its sample and the stamp: up to one revolution of travel ahead. Nothing real can be there.
        ahead = max(0.0, speed) * self.cfg.reactive.lidar_period_s
        body = (-0.5 * veh.length_m - m, 0.5 * veh.length_m + m + ahead, -0.5 * veh.width_m - m, 0.5 * veh.width_m + m)
        self.acc.add_scan(int(t * 1e9), ranges, angle_min, angle_inc, pose_odom_at_stamp, self.base_to_laser, body)
        if self.phase in (FINAL, DONE) or (self.phase == WAIT and not self.cfg.reactive.preview_while_waiting):
            return
        t0 = time.perf_counter()
        cloud = self.acc.cloud_in(pose_odom_now)
        guide = None
        # Lap 1 is bounded by its clearance rather than by its time, so while
        # the car is unsure of itself it gives up a little speed and takes more
        # margin, and tightens as it learns.
        v_cap = None if self.phase == RACE else self.env.lap1_v_scale * self.cfg.reactive.v_explore_mps
        if self.phase == RACE and not self._guard(t, cloud, pose_odom_now, speed):
            self.react_ms = (time.perf_counter() - t0) * 1e3
            return
        if self.phase in (RACE, PLANNING, MAP_GUIDED) and self.plan is not None:
            guide = self._race_line_in(pose_odom_now)
        duration = self.cfg.race.guide_handover_s
        self.guide_weight = (0. if guide is None else 1. if duration <= 0 or self.guide_started_t is None else
                             max(0., min(1., (t - self.guide_started_t) / duration)))
        cmd, state = self.reactive.step(cloud, pose_odom_now, self.travel, speed, guide, v_cap,
                                        guide_weight=self.guide_weight)
        self.cmd, self.state = cmd, state
        self.local_path_odom = se2_apply(pose_odom_now, cmd.path) if len(cmd.path) else None
        self.rear_clear = self._rear_clearance(cloud)
        if self.recover == 2:
            remaining = max(0., self.recover_distance_limit - self.recover_distance)
            self.recover_path_clear = reverse_path_clear(cloud, self.reactive.veh, self.reactive.cfg,
                                                        self.recover_kappa, remaining, self.cfg.adaptive.recover_clearance_m)
        if self.phase in (EXPLORE, CLOSING) and self.pose_map is not None:
            self.side_log["xy"].append(self.pose_map[:2])
            self.side_log["left"].append(bool(state.left_wall))
            self.side_log["right"].append(bool(state.right_wall))
        self.react_ms = (time.perf_counter() - t0) * 1e3

    def _rear_clearance(self, cloud: np.ndarray) -> float:
        """Free distance behind the car in the accumulated cloud [m].

        The chassis shadows rear beams, so these are accumulated obstacle
        returns, not proof that unseen space is empty. No cloud means no
        permission to reverse. With a cloud, infinity means no observed rear
        obstacle; recovery still has its independent short distance/time cap.
        """
        veh = self.cfg.vehicle
        if cloud is None or not len(cloud):
            return 0.0
        half_w = 0.5 * veh.width_m + self.cfg.reactive.wall_margin_m
        behind = cloud[(cloud[:, 0] < -0.5 * veh.length_m) & (np.abs(cloud[:, 1]) < half_w)]
        if not len(behind):
            return float("inf")
        return float(-behind[:, 0].max() - 0.5 * veh.length_m)

    def _recovery(self, t: float, speed: float) -> tuple[float, float] | None:
        """Get out of a place the planner has no path out of.

        When no candidate is feasible the planner steers at the longest free
        path and brakes, and the car then stands there for the rest of the run.
        Creeping forward is enough when there is any room ahead; when there is
        none, the car backs out along the way it came. Both are bounded in
        distance and in time, and there are only so many attempts per lap.
        """
        a = self.cfg.adaptive
        if not a.recover_enabled or self.phase not in (EXPLORE, CLOSING, PLANNING, RACE, REACTIVE_ONLY, MAP_GUIDED):
            self.recover, self.block_since = 0, None
            return None
        if self.recover:
            rear_blocked = self.recover == 2 and (not self.recover_path_clear or (not self.recover_geometric
                           and (math.isnan(self.rear_clear) or self.rear_clear <= a.recover_clearance_m)))
            recovered_m = max(self.travel - self.recover_travel, self.recover_distance)
            done = (recovered_m >= self.recover_distance_limit) or (t >= self.recover_until)
            path_ready = (self.cmd is not None and self.cmd.feasible and self.cmd.v > 1e-6
                          and self.cmd.diag.get("free", 0.0) >= a.recover_exit_free_m)
            if done or rear_blocked or path_ready:
                self.events.setdefault("recovery", []).append(
                    {"t_end": t, "kind": self.recover, "travel_m": recovered_m,
                     "rear_blocked": rear_blocked})
                self.recover, self.block_since = 0, None
                self.recover_xy = None
                self.recover_progress_origin = self.pose_odom[:2] if self.pose_odom is not None else None
                self.recover_forward_distance = 0.0
                self.v_out = 0.0
                return (0.0, 0.0) if rear_blocked or done else None
            if self.recover == 2:
                return -a.recover_speed_mps, math.atan(self.recover_kappa * self.cfg.vehicle.wheelbase_m)
            return a.recover_creep_mps, self.steer
        # Two different stops end a run the same way. The planner can find no
        # feasible candidate, or it can find one and still command zero because
        # the lane rule says the car is beyond what it can prove: it has lost
        # the wall it was following, and the curbs are invisible. Both leave the
        # car standing until the referee calls it stuck.
        blocked = self.cmd is not None and (not self.cmd.feasible or self.cmd.v <= 1e-6) and abs(speed) < 0.1
        if not blocked:
            self.block_since = None
            return None
        self.block_since = t if self.block_since is None else self.block_since
        if t - self.block_since < a.recover_block_s or self.recover_count >= a.recover_max_per_lap:
            return None
        if a.recover_max_without_progress and self.recover_stuck_count >= a.recover_max_without_progress:
            return None
        free = float(self.cmd.diag.get("free", 0.0)) if self.cmd is not None else 0.0
        lost_wall = self.state is not None and self.state.ref is None
        # Creeping forward is only allowed where the lane is proven. Having lost
        # the wall, the way out is backwards, over ground the car has already
        # driven and where it could still see: that keeps the lane rule intact
        # instead of trading it for a metre of blind progress.
        creep = free >= a.recover_creep_free_m and not lost_wall
        distance_limit = a.recover_distance_m
        self.recover_kappa = 0.0
        self.recover_geometric = False
        forward_free = None
        if not creep and not self.acc.empty and self.pose_odom is not None:
            arc = choose_recovery_arc(self.acc.cloud_in(self.pose_odom), self.reactive.veh, self.reactive.cfg,
                                      distance_limit, self.cmd.kappa if self.cmd is not None else 0., a.recover_clearance_m)
            if arc is None:
                return None
            self.recover_kappa, distance_limit, forward_free = arc.kappa, arc.distance_m, arc.forward_free_m
            self.recover_geometric = True
        elif not creep:
            if math.isnan(self.rear_clear) or self.rear_clear <= a.recover_clearance_m:
                return None
            distance_limit = min(distance_limit, self.rear_clear - a.recover_clearance_m)
        self.recover_path_clear = True
        if distance_limit < 0.05:
            return None
        self.recover = 1 if creep else 2
        self.recover_count += 1
        self.recover_stuck_count += 1
        self.recover_progress_origin = None
        self.recover_forward_distance = 0.0
        self.recover_travel = self.travel
        self.recover_distance = 0.0
        self.recover_xy = self.pose_odom[:2] if self.pose_odom is not None else None
        self.recover_distance_limit = distance_limit
        self.recover_until = t + a.recover_timeout_s
        self.events.setdefault("recovery", []).append(
            {"t": t, "kind": self.recover, "free_m": free, "rear_clear_m": self.rear_clear,
             "distance_limit_m": distance_limit, "phase": self.phase})
        self.events["recovery"][-1].update(kappa=self.recover_kappa, forward_free_probe_m=forward_free)
        return (-a.recover_speed_mps, math.atan(self.recover_kappa * self.cfg.vehicle.wheelbase_m)) if self.recover == 2 else (a.recover_creep_mps, self.steer)

    def _recovery_progress(self, t: float, pose: tuple[float, float, float], speed: float) -> None:
        """Replenish attempts only after measured forward escape, not rocking.

        Require both net displacement from the end of the last maneuver and
        forward travel under a usable forward command. Reverse travel, parked
        pose noise, SLAM corrections and elapsed time earn no allowance.
        """
        a = self.cfg.adaptive
        if not a.recover_max_without_progress or not self.recover_stuck_count or self.recover:
            return
        origin, previous = self.recover_progress_origin, self._last_odom_xy
        forward = (origin is not None and previous is not None and speed > .05 and self.cmd_out[0] > 0
                   and self.cmd is not None and self.cmd.feasible and self.cmd.v > 1e-6)
        if not forward:
            return
        dx, dy = pose[0] - previous[0], pose[1] - previous[1]
        self.recover_forward_distance += max(0., dx * math.cos(pose[2]) + dy * math.sin(pose[2]))
        net = math.hypot(pose[0] - origin[0], pose[1] - origin[1])
        if min(net, self.recover_forward_distance) >= a.recover_progress_reset_m:
            self.events.setdefault("recovery_progress", []).append({"t": t, "net_m": net,
                "forward_m": self.recover_forward_distance, "attempts_replenished": self.recover_stuck_count})
            self.recover_stuck_count = 0
            self.recover_progress_origin = None
            self.recover_forward_distance = 0.0

    # ------------------------------------------------------------ race helpers
    def _race_line_in(self, pose_odom_now: tuple[float, float, float]) -> np.ndarray:
        """Race line points ahead of the car in the base frame (guide of the reactive planner)."""
        base_in_map = se2_compose(self.map_odom, pose_odom_now)
        route = self.plan.route if self.phase == MAP_GUIDED else None
        pts = (route if route is not None else self.plan.path).xy
        d = np.hypot(pts[:, 0] - base_in_map[0], pts[:, 1] - base_in_map[1])
        k0 = int(np.argmin(d))
        if self.phase == MAP_GUIDED and route is not None:
            # Position alone can jump to a nearby branch of a route, notably
            # around a seam or a reversing maneuver. Follow its station
            # continuously using smooth odometry, with heading as evidence.
            heading = np.angle(np.exp(1j * (route.heading - base_in_map[2])))
            cost = d * d + .04 * heading * heading
            if self.map_route_s is not None and self.map_route_odom is not None:
                moved = math.hypot(pose_odom_now[0] - self.map_route_odom[0], pose_odom_now[1] - self.map_route_odom[1])
                delta = (route.s - self.map_route_s + .5 * route.length) % route.length - .5 * route.length
                cost = np.where(abs(delta) <= .15 + 2 * moved, cost + .1 * delta * delta, np.inf)
                k0 = int(np.argmin(cost))
                self.map_route_s += float(delta[k0])
            else:
                k0 = int(np.argmin(cost))
                self.map_route_s = float(route.s[k0])
            self.map_route_odom = pose_odom_now[:2]
        idx = (k0 + np.arange(0, 60)) % len(pts)
        if self.phase == MAP_GUIDED:
            idx = idx[:min(len(idx), len(pts))]
            distance = np.concatenate(([0.], np.cumsum(np.linalg.norm(np.diff(pts[idx], axis=0), axis=1))))
            idx = idx[distance <= 2.2]
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
        cfg = self.cfg
        if cfg.adaptive.plan_with_identified_model:
            cfg = replace(cfg, vehicle=self.reactive.veh)
        if self.env.blind_margin_add_m > 1e-6:
            # The invisible lane edge is a prior, not a measurement, and an
            # unsure plant tracks the line less well: widen the margin to it.
            cfg = replace(cfg, planner=replace(cfg.planner, blind_margin_m=cfg.planner.blind_margin_m + self.env.blind_margin_add_m))
        self.events["planning"]["blind_margin_m"] = cfg.planner.blind_margin_m
        self.future = self.submit(run_planner, grid, loop, self.start, side, cfg)
        self.phase = PLANNING

    # ---------------------------------------------------------------- control
    def _stop_mission(self, t: float, reason: str) -> None:
        self.mission_stop_reason = reason
        self.phase, self.v_out, self.recover = DONE, 0.0, 0
        self.events["t_done"] = t
        self.events["mission"] = {"stop_reason": reason, "distance_m": self.mission_distance,
                                  "elapsed_s": 0.0 if self.mission_started_t is None else t - self.mission_started_t}

    def control(self, t: float, pose_odom: tuple[float, float, float], speed: float, map_odom_raw: tuple[float, float, float] | None,
                ready: bool, pose_age: float = 0.0) -> tuple[float, float, list | None]:
        """50 Hz step: (speed command [m/s], steering [rad], log row)."""
        cfg, veh = self.cfg, self.cfg.vehicle
        dt = 0.0 if self.last_t is None else max(0.0, t - self.last_t)
        self.last_t = t
        # A zero-velocity observation must not turn position corrections or
        # parked ICP noise into distance travelled and eventual lap crossings.
        if self._last_odom_xy is not None and speed != 0.0:
            self.travel += math.hypot(pose_odom[0] - self._last_odom_xy[0], pose_odom[1] - self._last_odom_xy[1])
        self._recovery_progress(t, pose_odom, speed)
        self._last_odom_xy = pose_odom[:2]
        self.pose_odom = pose_odom
        if self.recover:
            if self.recover_xy is not None:
                self.recover_distance += math.hypot(pose_odom[0] - self.recover_xy[0], pose_odom[1] - self.recover_xy[1])
            self.recover_xy = pose_odom[:2]
        if map_odom_raw is not None:
            self.map_odom_raw = map_odom_raw
            prev = self.map_odom
            self.map_odom = self.corr.update(t, map_odom_raw, pose_odom)
            if self.phase == CLOSING:
                a, b = se2_compose(map_odom_raw, pose_odom), se2_compose(prev, pose_odom)
                self.corr_log.append((t, math.hypot(a[0] - b[0], a[1] - b[1])))
        pose_map = se2_compose(self.map_odom, pose_odom)
        self.pose_map = pose_map
        rear_map = se2_compose(pose_map, (self.model.rear_offset_m(), 0.0, 0.0))
        v_cmd, steer = 0.0, self.steer

        mission = cfg.mission
        if mission.enabled and self.mission_started_t is not None:
            # Count the path, including bends or reversing pose observations,
            # rather than displacement from the start. This independent trial
            # bound does not depend on the velocity gate of the lap counter.
            if self.mission_stop_reason is None:
                if self.mission_xy is not None:
                    self.mission_distance += math.hypot(pose_odom[0] - self.mission_xy[0], pose_odom[1] - self.mission_xy[1])
                self.mission_xy = pose_odom[:2]
                if t - self.mission_started_t >= mission.max_duration_s:
                    self._stop_mission(t, "duration")
                elif self.mission_distance >= mission.max_distance_m:
                    self._stop_mission(t, "distance")
            if self.mission_stop_reason is not None:
                self.ident.push_command(t, 0.0, 0.0)
                self.cmd_out = (0.0, 0.0)
                return self._out(t, 0.0, self.steer, speed, pose_age)

        if self.phase == WAIT:
            if ready and self.map_odom_raw is not None and not self.acc.empty:
                self.ready_since = t if self.ready_since is None else self.ready_since
                if t - self.ready_since >= self.static_start_s:
                    self.phase = EXPLORE
                    self.start = pose_map
                    self.counter = LapCounter(pose_map, cfg.closure, t, self.travel)
                    self.events["t_explore"] = t
                    self.events["start_map"] = list(pose_map)
                    if mission.enabled:
                        self.mission_started_t, self.mission_xy = t, pose_odom[:2]
            return self._out(t, 0.0, 0.0, speed, pose_age)

        self.path_log.append(pose_map[:2])
        crossing = self.counter.update(t, pose_map, self.travel) if self.counter is not None else None
        if crossing is not None:
            self.events["crossings"].append(asdict(crossing))
            self.recover_count = 0
            self.recover_stuck_count = 0
            self.recover_progress_origin = None
            self.recover_forward_distance = 0.0
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
                self.guide_started_t = t
                if not self.plan.diag.get("direct_tracking_allowed", True):
                    # Keep the same live-checked local controller and add the
                    # observed SLAM route as guidance. No infeasible forward
                    # candidate is handed to pure pursuit or the actuators.
                    self.phase = MAP_GUIDED
                    self.events["t_map_guided"] = t
                    self.events["handover"] = {"mode": MAP_GUIDED, "local_planner_retained": True}
            except Exception as exc:  # noqa: BLE001
                self.events["plan_error"] = repr(exc)
                self.future, self.phase = None, REACTIVE_ONLY
        guide_ready = (self.guide_started_t is None or t - self.guide_started_t >= cfg.race.guide_handover_s)
        if self.phase == PLANNING and self.plan is not None and guide_ready and self._handover_ok(rear_map):
            self.events["t_race"] = t
            self._start_race(rear_map)

        self.guided = False
        if self.phase == RACE and t >= self.guided_until:
            s_tot, lat = self.pursuit.locate(rear_map[0], rear_map[1])
            self.s_race, self.lat_race = s_tot, lat
            steer = self.pursuit.steering(rear_map[0], rear_map[1], rear_map[2], speed)
            s_mod = float(self.plan.path.s[self.pursuit.idx])
            v_cmd = self.plan.speed_at(s_mod + max(0.3, 0.3 * speed))
            # The car asks for no more speed and no more lateral grip than its
            # confidence in its own plant justifies. Scaling the planned speed
            # by sqrt(a_lat_eff / a_lat) is the speed map the planner would have
            # produced with the reduced lateral acceleration.
            v_cmd = min(v_cmd * math.sqrt(self.env.a_lat_mps2 / max(cfg.race.a_lat_mps2, 1e-6)), self.env.v_max_mps)
            if self.counter.laps == self.laps - 1:  # last lap: brake to stop just past the start line
                dist = (self.plan.path.length - s_mod) % self.plan.path.length
                if dist > 0.5 * self.plan.path.length:
                    dist = 0.0
                v_cmd = min(v_cmd, math.sqrt(2.0 * cfg.race.a_dec_mps2 * (dist + cfg.race.finish_margin_m)))
        elif self.phase in (EXPLORE, CLOSING, PLANNING, REACTIVE_ONLY, RACE, MAP_GUIDED):
            self.guided = self.phase in (RACE, PLANNING, MAP_GUIDED) and self.plan is not None
            if self.phase == RACE:
                self.pursuit.reset(rear_map[0], rear_map[1])  # resume the line after the guided section
            if self.cmd is not None and self.local_path_odom is not None:
                rear_odom = se2_compose(pose_odom, (self.model.rear_offset_m(), 0.0, 0.0))
                # Looking further ahead is how an unsure car buys damping; it
                # never answers uncertainty by pushing harder.
                ld = max(cfg.reactive.local_path_lookahead_m, 0.35 + 0.32 * speed)
                ld += self.env.lookahead_add_m + self.env.lookahead_add_s * max(speed, 0.0)
                steer = pursue(self.local_path_odom, rear_odom, ld, veh.wheelbase_m)
                v_cmd = self.cmd.v
        got_out = self._recovery(t, speed)
        if got_out is not None:
            v_cmd, steer = got_out
        if self.phase in (FINAL, DONE):
            v_cmd = 0.0
            if abs(speed) < 0.02:
                if self.phase != DONE:
                    self.events["t_done"] = t
                self.phase = DONE
        # Actuator-friendly commands: steering rate and speed-increase limits.
        rate = math.radians(cfg.reactive.steer_rate_dps if self.phase != RACE else veh.servo_rate_dps) * max(dt, 1e-3)
        geometry = self.reactive.veh if cfg.adaptive.plan_with_identified_model else veh
        lim_left = math.radians(geometry.steering_limit_deg(True, planning=self.phase != RACE))
        lim_right = math.radians(geometry.steering_limit_deg(False, planning=self.phase != RACE))
        steer = max(-lim_right, min(lim_left, steer))
        self.steer = self.steer + max(-rate, min(rate, steer - self.steer))
        a_up = cfg.reactive.a_acc_mps2 if self.phase != RACE else cfg.race.a_acc_mps2
        self.v_out = min(v_cmd, self.v_out + a_up * max(dt, 1e-3)) if v_cmd > self.v_out else v_cmd
        if v_cmd < 0.0:  # backing out: the ramp above only shapes forward speed
            self.v_out = v_cmd
        # Everything above decided what the car wants to do. The plant decides
        # what it has to be asked for: the steering angle is the planners'
        # encoding of a curvature (the wheelbase in it cancels on the way out),
        # and the identified map is inverted on that curvature.
        self.kappa_des = math.tan(self.steer) / veh.wheelbase_m
        v_pub, kappa_pub = self.shaper.shape(t, self.kappa_des, self.v_out, speed, self.model, self.env)
        kappa_pub = max(-veh.curvature_limit(False), min(veh.curvature_limit(True), kappa_pub))
        if mission.enabled:
            if mission.forward_only and self.v_out < 0.0:
                self._stop_mission(t, "reverse requested")
                v_pub, kappa_pub = 0.0, 0.0
            else:
                # Negative speed feedback is a request to slow down, not
                # permission to enter the ESC's reverse sequence in this trial.
                # Feedback that asks to slow a forward request must never
                # select reverse; only an explicit negative target may do so.
                floor = 0.0 if mission.forward_only or self.v_out >= 0.0 else -min(mission.max_command_mps, mission.max_reverse_command_mps)
                v_pub = max(floor, min(mission.max_command_mps, v_pub))
        # Identification observes the bounded command actually published,
        # including the trial cap, rather than an unclipped internal request.
        self.ident.push_command(t, kappa_pub, v_pub)
        self.cmd_out = (v_pub, kappa_pub)
        return self._out(t, v_pub, math.atan(kappa_pub * veh.wheelbase_m), speed, pose_age)

    def steer_command(self, steer: float) -> float:
        """Steering angle to publish, limited to what the servo can reach.

        The asymmetry of the linkage used to be divided out here. It is now the
        prior of the right-hand gain of the identified map, which is the place
        it belongs: the car measures it instead of assuming it, and applying
        both would correct it twice.
        """
        veh = self.cfg.vehicle
        return max(-math.radians(veh.steering_limit_deg(False)),
                   min(math.radians(veh.steering_limit_deg(True)), steer))

    def _out(self, t: float, v: float, steer: float, speed: float, age: float) -> tuple[float, float, list]:
        pm = self.pose_map or (float("nan"),) * 3
        po = self.pose_odom or (float("nan"),) * 3
        raw = self.map_odom_raw or (float("nan"),) * 3
        st, d = self.state, (self.cmd.diag if self.cmd is not None else {})
        m, env = self.model, self.env
        row = [f"{t:.4f}", self.phase, f"{pm[0]:.4f}", f"{pm[1]:.4f}", f"{pm[2]:.5f}", f"{po[0]:.4f}", f"{po[1]:.4f}", f"{po[2]:.5f}",
               f"{speed:.3f}", f"{v:.3f}", f"{math.degrees(steer):.3f}", f"{age:.4f}",
               f"{raw[0]:.4f}", f"{raw[1]:.4f}", f"{raw[2]:.5f}", f"{self.map_odom[0]:.4f}", f"{self.map_odom[1]:.4f}", f"{self.map_odom[2]:.5f}",
               st.mode if st is not None else "", d.get("planner", ""), int(bool(st and st.left_wall)), int(bool(st and st.right_wall)),
               f"{d.get('d_wall', float('nan')):.3f}", f"{d.get('d_target', float('nan')):.3f}", f"{d.get('clr_min', float('nan')):.3f}",
               d.get("n_feasible", ""), f"{self.react_ms:.1f}", self.counter.laps if self.counter else 0, f"{self.travel:.3f}",
               f"{self.s_race:.3f}", f"{self.lat_race:.4f}", int(self.guided), f"{d.get('lane_known', float('nan')):.2f}", f"{d.get('v_lane', float('nan')):.2f}",
               f"{self.kappa_des:.4f}", f"{self.cmd_out[1]:.4f}", f"{self.ident.fresh_kappa(t):.4f}",
               f"{m.gain_left:.4f}", f"{m.gain_right:.4f}", f"{m.trim:.4f}", f"{m.slip:.5f}", f"{m.arm_m:.4f}",
               f"{m.v_gain:.4f}", f"{m.dead_mps:.4f}", f"{m.tau_steer_s:.3f}",
               f"{m.confidence:.3f}", f"{self.ident.conf_lever:.3f}", f"{env.v_max_mps:.2f}", self.recover]
        return v, steer, row

    def adaptation(self) -> dict:
        """What the car worked out about itself, for the run report."""
        return {**self.ident.status(), "envelope": {"v_max_mps": self.env.v_max_mps, "a_lat_mps2": self.env.a_lat_mps2,
                                                   "lap1_v_scale": self.env.lap1_v_scale,
                                                   "blind_margin_add_m": self.env.blind_margin_add_m,
                                                   "kappa_limit": self.env.kappa_limit,
                                                   "kappa_limit_left": self.env.kappa_limit_left,
                                                   "kappa_limit_right": self.env.kappa_limit_right}}

    def status(self) -> dict:
        d = self.cmd.diag if self.cmd is not None else {}
        return {"phase": self.phase, "lap_own": self.counter.laps if self.counter else 0, "laps": self.laps, "travel_m": self.travel,
                "recover": self.recover, "recover_count": self.recover_count,
                "recover_stuck_count": self.recover_stuck_count,
                "recover_max_without_progress": self.cfg.adaptive.recover_max_without_progress,
                "recover_geometric": self.recover_geometric,
                "recover_max_per_lap": self.cfg.adaptive.recover_max_per_lap,
                "rear_clearance_m": self.rear_clear,
                "preview_ready": bool(self.cfg.reactive.preview_while_waiting and self.phase == WAIT
                                      and self.cmd is not None and self.cmd.feasible
                                      and self.local_path_odom is not None and len(self.local_path_odom) >= 2),
                "mode": self.state.mode if self.state else "", "d_wall": d.get("d_wall"), "d_target": d.get("d_target"),
                "clearance_m": d.get("clr_min"), "planner": d.get("planner"), "react_ms": self.react_ms,
                "plan_ready": self.plan is not None, "guided": self.guided,
                "direct_tracking_allowed": None if self.plan is None else self.plan.diag.get("direct_tracking_allowed", True),
                "planning_kappa_left": self.reactive.veh.curvature_limit(True, planning=True),
                "planning_kappa_right": self.reactive.veh.curvature_limit(False, planning=True),
                "map_guide_weight": self.guide_weight,
                "map_route_s_m": self.map_route_s,
                "mission": None if not self.cfg.mission.enabled else {
                    "distance_m": self.mission_distance, "stop_reason": self.mission_stop_reason,
                    "max_distance_m": self.cfg.mission.max_distance_m,
                    "max_duration_s": self.cfg.mission.max_duration_s,
                    "max_command_mps": self.cfg.mission.max_command_mps,
                    "max_reverse_command_mps": self.cfg.mission.max_reverse_command_mps},
                "sign_fault": self.ident.sign_fault,
                "confidence": self.model.confidence, "arm_m": self.model.arm_m, "g_left": self.model.gain_left}
