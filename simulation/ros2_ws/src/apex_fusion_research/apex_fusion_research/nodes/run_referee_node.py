"""Referee of a live validation run (ground truth, evaluation only).

Applies the success and safety rules of the pose-dataset generator to the
true pose of the car: progress along the reference path of the format,
lateral error, collision of the car footprint with the track geometry,
rollover, stuck and timeout. The run is ``completed`` once the true progress
reaches the planned distance and the car has stood still for
``static_end_s``, ``failed`` at the first violated rule.

``mode:=race`` (the race driver: reactive lap 1, then its own race line) does
not tie the car to the seeded reference path: the lateral error is reported,
not judged; the car fails when its footprint leaves the true lane by more
than ``max_lane_excess_m`` or hits anything (walls, curbs, pillars). The
timeout follows from a minimum average speed, and the true lap times and
the minimum clearance of each lap are recorded. The verdict is
published (latched JSON) and written to ``<output_dir>/run_result.json``; the
launcher stops the simulation when that file appears. Nothing here feeds the
car: the driver only sees the estimated pose.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import rclpy
from geometry_msgs.msg import Point32
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud
from std_msgs.msg import String

from ._common import LATCHED_QOS, json_msg
from ._pose_tools import run_setup


class RunRefereeNode(Node):
    def __init__(self) -> None:
        super().__init__("apex_run_referee")
        dp = self.declare_parameter
        dp("track", "val_mixed")
        dp("motion", "medium")
        dp("seed", 1)
        dp("laps", 2.0)
        dp("v_ref", 0.0)
        dp("truth_topic", "/apex/sim/ground_truth/base_odom")
        dp("status_topic", "/apex/sim/run_status")
        dp("output_dir", "")
        dp("static_end_s", 1.0)
        dp("check_period_s", 0.02)
        dp("timeout_factor", 2.5)
        dp("wait_first_motion_s", 120.0)
        dp("track_topic", "/apex/sim/ground_truth/perfect_map_points")
        dp("scan_height_m", 0.12)  # nominal LiDAR height: walls crossing it are what the scanner sees
        dp("mode", "format")  # format (dataset driver on the seeded path) | race (race driver)
        dp("min_avg_speed_mps", 0.4)
        dp("race_extra_s", 60.0)
        dp("max_lane_excess_m", 0.30)
        dp("race_stuck_timeout_s", 8.0)
        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        rs = run_setup(str(gp("track")), str(gp("motion")), int(gp("seed")), float(gp("laps")), float(gp("v_ref")))
        from pose_dataset.controller import PurePursuit  # noqa: PLC0415
        from pose_dataset.geometry import FootprintChecker, rect_corners  # noqa: PLC0415

        setup, vcfg = rs["setup"], rs["vcfg"]
        self.rect_corners = rect_corners
        self.checker = FootprintChecker(rs["track"].geometry)
        self.locator = PurePursuit(setup["path"], float(vcfg["model"]["wheelbase_m"]), vcfg["controller"], 1.0)
        self.total = float(gp("laps")) * setup["path"].length
        self.safety = vcfg["safety"]
        self.len = float(vcfg["model"]["chassis_length_m"])
        self.width = float(vcfg["model"]["footprint_width_m"])
        self.rear = rs["rear_offset"]
        self.static_end_ns = int(float(gp("static_end_s")) * 1e9)
        self.check_ns = int(float(gp("check_period_s")) * 1e9)
        self.mode = str(gp("mode"))
        self.lap_len = setup["path"].length
        if self.mode == "race":
            from ..core.race_config import vehicle_limits  # noqa: PLC0415
            from ..core.race_eval import TrueTrack  # noqa: PLC0415

            self.true_track = TrueTrack(rs["track"], vehicle_limits(vcfg))
            self.timeout_s = self.total / float(gp("min_avg_speed_mps")) + float(gp("race_extra_s"))
        else:
            self.timeout_s = float(gp("timeout_factor")) * setup["plan"].planned_duration() + 15.0
        self.max_excess = float(gp("max_lane_excess_m"))
        self.race_stuck_s = float(gp("race_stuck_timeout_s"))
        self.lap_times: list[float] = []
        self.lap_min_clr: list[float] = [math.inf]
        self.max_excess_seen = -math.inf
        self.wait_s = float(gp("wait_first_motion_s"))
        self.out_dir = Path(str(gp("output_dir"))) if str(gp("output_dir")) else None
        self.status_pub = self.create_publisher(String, str(gp("status_topic")), LATCHED_QOS)
        self.create_subscription(Odometry, str(gp("truth_topic")), self._on_truth, 100)
        # Real track walls at the scan height (evaluation of the SLAM map).
        self.track_pub = self.create_publisher(PointCloud, str(gp("track_topic")), 2)
        self.track_points = self._wall_points(rs["track"].geometry, float(gp("scan_height_m")))
        self.create_timer(1.0, self._publish_track)
        self.last_check = -(2**62)
        self.t_first: int | None = None
        self.t_move: int | None = None
        self.still_since: int | None = None
        self.stuck_since: int | None = None
        self.s = 0.0
        self.max_lat = 0.0
        self.max_rp = 0.0
        self.path_m = 0.0
        self.prev_xy: tuple[float, float] | None = None
        self.verdict: dict | None = None
        self.meta = {"track": gp("track"), "motion": gp("motion"), "seed": gp("seed"), "laps": gp("laps"), "planned_m": self.total,
                     "spawn": rs["spawn"], "nominal_start": rs["nominal_start"], "plan": setup["plan"].description}

    @staticmethod
    def _wall_points(geometry, z: float, step: float = 0.02) -> list[tuple[float, float]]:  # noqa: ANN001
        """Outline of every box / cylinder crossing the height ``z``, every ``step``."""
        pts = []
        for b in geometry.boxes:
            if b.kind == "ground" or not (b.center[2] - 0.5 * b.size[2] <= z <= b.center[2] + 0.5 * b.size[2]):
                continue
            hx, hy = 0.5 * b.size[0], 0.5 * b.size[1]
            c, s = math.cos(b.yaw), math.sin(b.yaw)
            corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
            for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1]):
                n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
                for i in range(n):
                    u = i / n
                    lx, ly = x0 + u * (x1 - x0), y0 + u * (y1 - y0)
                    pts.append((b.center[0] + c * lx - s * ly, b.center[1] + s * lx + c * ly))
        for cy in geometry.cylinders:
            if not (cy.center[2] - 0.5 * cy.length <= z <= cy.center[2] + 0.5 * cy.length):
                continue
            n = max(8, int(2 * math.pi * cy.radius / step))
            pts += [(cy.center[0] + cy.radius * math.cos(2 * math.pi * i / n), cy.center[1] + cy.radius * math.sin(2 * math.pi * i / n)) for i in range(n)]
        return pts

    def _publish_track(self) -> None:
        msg = PointCloud()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.points = [Point32(x=float(x), y=float(y), z=0.0) for x, y in self.track_points]
        self.track_pub.publish(msg)

    def _on_truth(self, m: Odometry) -> None:
        if self.verdict is not None:
            return
        t = Time.from_msg(m.header.stamp).nanoseconds
        if t - self.last_check < self.check_ns:
            return
        self.last_check = t
        self.t_first = self.t_first or t
        p, q = m.pose.pose.position, m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
        speed = float(m.twist.twist.linear.x)
        if self.prev_xy is not None:
            self.path_m += math.hypot(p.x - self.prev_xy[0], p.y - self.prev_xy[1])
        self.prev_xy = (p.x, p.y)
        self.s, lat = self.locator.locate(p.x + self.rear * math.cos(yaw), p.y + self.rear * math.sin(yaw))
        if speed > 0.1 and self.t_move is None:
            self.t_move = t
        moving = self.t_move is not None
        if moving:
            self.max_lat = max(self.max_lat, abs(lat))
            self.max_rp = max(self.max_rp, abs(roll), abs(pitch))
        reason = None
        race = self.mode == "race"
        if race and moving:
            clr = self.true_track.clear(p.x, p.y, yaw)
            self.lap_min_clr[-1] = min(self.lap_min_clr[-1], clr)
            excess = float(self.true_track.lane_excess(p.x, p.y, yaw)[0])
            self.max_excess_seen = max(self.max_excess_seen, excess)
            while self.s >= (len(self.lap_times) + 1) * self.lap_len:
                self.lap_times.append(t * 1e-9)
                self.lap_min_clr.append(math.inf)
        hit = self.checker.collides(self.rect_corners(p.x, p.y, yaw, self.len, self.width))
        if hit:
            reason = f"collision with {hit}"
        elif race and moving and excess > self.max_excess:
            reason = f"off_track (footprint {excess:.2f} m outside the lane)"
        elif not race and moving and abs(lat) > float(self.safety["max_lateral_error_m"]):
            reason = f"off_track (lateral error {lat:.2f} m)"
        elif max(abs(roll), abs(pitch)) > math.radians(float(self.safety["max_roll_pitch_deg"])):
            reason = "rollover"
        elif moving and (t - self.t_move) * 1e-9 > self.timeout_s:
            reason = "timeout"
        elif not moving and (t - self.t_first) * 1e-9 > self.wait_s:
            reason = "never started"
        if moving and speed < 0.05 and self.s < self.total - 1.0:
            self.stuck_since = self.stuck_since or t
        else:
            self.stuck_since = None
        # A long stop that the plan does not explain (stops last <= 2 s).
        stuck_s = self.race_stuck_s if race else float(self.safety["stuck_timeout_s"]) + 2.0
        if reason is None and self.stuck_since is not None and (t - self.stuck_since) * 1e-9 > stuck_s:
            reason = "stopped / stuck"
        if reason:
            self._finish(t, "failed", reason)
            return
        if self.s >= self.total - 1.0 and speed < 0.02:
            self.still_since = self.still_since or t
            if t - self.still_since >= self.static_end_ns:
                self._finish(t, "completed", "")
        else:
            self.still_since = None
        if int(t // 500_000_000) != int((t - self.check_ns) // 500_000_000):
            self.status_pub.publish(json_msg({"state": "running", "s_true_m": self.s, "lateral_m": lat, "total_m": self.total, "speed_mps": speed}))

    def _finish(self, t: int, status: str, reason: str) -> None:
        self.verdict = {
            "status": status, "failure_reason": reason, "lap_completed": status == "completed",
            "progress_m": self.s, "planned_m": self.total, "driven_m": self.path_m, "max_lateral_error_m": self.max_lat,
            "max_roll_pitch_deg": math.degrees(self.max_rp), "t_end_s": t * 1e-9,
            "drive_s": (t - self.t_move) * 1e-9 if self.t_move else 0.0, **self.meta, "mode": self.mode,
        }
        if self.mode == "race":
            t0 = self.t_move * 1e-9 if self.t_move else 0.0
            marks = [t0] + self.lap_times
            self.verdict.update({
                "lap_times_s": [b - a for a, b in zip(marks, marks[1:])], "lap_len_m": self.lap_len,
                "lap_min_clearance_m": [c for c in self.lap_min_clr if math.isfinite(c)],
                "min_clearance_m": min((c for c in self.lap_min_clr if math.isfinite(c)), default=math.nan),
                "max_lane_excess_m": self.max_excess_seen,
            })
        self.status_pub.publish(json_msg({"state": status, **self.verdict}))
        self.get_logger().info(f"run {status}{': ' + reason if reason else ''} (true progress {self.s:.1f} / {self.total:.1f} m, max lateral {self.max_lat:.2f} m)")
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / "run_result.json").write_text(json.dumps(self.verdict, indent=1, default=float), encoding="utf-8")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RunRefereeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
