"""Closed-loop driver of the validation formats on the car's own estimate.

The driver of the pose dataset (pure pursuit on the seeded reference path of
the track + the planned speed map of the motion profile, 50 Hz, same gains),
fed with the pose the car estimates instead of the exact one:

    pose_world = T_world_map * T_map_odom (SLAM) * pose_odom (odometry, propagated
                 with the IMU to its latest sample)

``T_world_map`` is where the car believes it starts: its actual start pose
(``start_pose:=true``, the reference path is defined from where the car is,
as when it plans in its own map; used once, never updated with the truth) or
the nominal start line (``start_pose:=nominal``: the seeded placement error of
the format, up to 2 deg, then rotates the whole believed path). Odometry
drift and SLAM corrections act on the control like on the real car. ``pose_source:=truth`` drives on the exact pose
instead (diagnostic: separates the estimator from the closed loop).

Output: ``geometry_msgs/Twist`` on ``/apex/cmd_vel_track`` (speed, yaw rate =
speed * tan(steering) / wheelbase), the interface of the real car.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from ._common import json_msg
from ._pose_tools import run_setup

PHASE_WAIT, PHASE_DRIVE, PHASE_HOLD, PHASE_FINAL, PHASE_DONE = "wait", "drive", "hold", "final", "done"


def _yaw(q) -> float:  # noqa: ANN001
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def _compose(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    c, s = math.cos(a[2]), math.sin(a[2])
    return a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2]


class TrackDriverNode(Node):
    def __init__(self) -> None:
        super().__init__("apex_track_driver")
        dp = self.declare_parameter
        dp("track", "val_mixed")
        dp("motion", "medium")
        dp("seed", 1)
        dp("laps", 2.0)
        dp("v_ref", 0.0)
        dp("pose_source", "estimate")  # estimate | truth
        dp("start_pose", "true")  # true: path from the actual start pose | nominal: from the start line
        dp("odom_topic", "/apex/odometry/learned_predicted")
        dp("estimator_status_topic", "/apex/learned_odometry/status")
        dp("truth_topic", "/apex/sim/ground_truth/base_odom")
        dp("map_frame", "map")
        dp("odom_frame", "odom_learned")
        dp("cmd_topic", "/apex/cmd_vel_track")
        dp("rate_hz", 50.0)
        dp("static_start_s", 1.5)  # IMU at rest after the estimator is ready (like the dataset)
        dp("max_pose_age_s", 0.3)
        dp("log_csv", "")
        dp("status_topic", "/apex/track_driver/status")
        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        rs = run_setup(str(gp("track")), str(gp("motion")), int(gp("seed")), float(gp("laps")), float(gp("v_ref")))
        from pose_dataset.controller import PurePursuit  # noqa: PLC0415

        setup, vcfg = rs["setup"], rs["vcfg"]
        self.path, self.plan = setup["path"], setup["plan"]
        self.stops = list(self.plan.stops)
        self.wheelbase = float(vcfg["model"]["wheelbase_m"])
        self.pursuit = PurePursuit(self.path, self.wheelbase, vcfg["controller"], setup["gain_scale"])
        self.total = float(gp("laps")) * self.path.length
        self.rear = rs["rear_offset"]
        self.t_world_map = rs["spawn"] if str(gp("start_pose")) == "true" else rs["nominal_start"]
        self.source = str(gp("pose_source"))
        self.map_frame, self.odom_frame = str(gp("map_frame")), str(gp("odom_frame"))
        self.static_ns = int(float(gp("static_start_s")) * 1e9)
        self.max_age_ns = int(float(gp("max_pose_age_s")) * 1e9)
        self.phase = PHASE_WAIT
        self.ready_since: int | None = None
        self.hold_since: int | None = None
        self.stop_idx = 0
        self.est: tuple[int, tuple[float, float, float], float, float] | None = None  # stamp, pose (odom), speed, yaw rate
        self.truth: tuple[int, tuple[float, float, float], float] | None = None
        self.estimator_ready = self.source == "truth"
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.cmd_pub = self.create_publisher(Twist, str(gp("cmd_topic")), 20)
        self.status_pub = self.create_publisher(String, str(gp("status_topic")), 10)
        self.create_subscription(Odometry, str(gp("odom_topic")), self._on_est, 50)
        self.create_subscription(Odometry, str(gp("truth_topic")), self._on_truth, 50)
        self.create_subscription(String, str(gp("estimator_status_topic")), self._on_est_status, 10)
        self.create_timer(1.0 / float(gp("rate_hz")), self._control)
        self.log = None
        if str(gp("log_csv")):
            Path(str(gp("log_csv"))).parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(str(gp("log_csv")), "w", newline="", encoding="utf-8")
            self.log = csv.writer(self._log_file)
            self.log.writerow(["t", "phase", "x", "y", "yaw", "speed", "s", "lateral", "v_cmd", "steer_deg", "pose_age_s", "map_dx", "map_dy", "map_dyaw"])
        self.get_logger().info(f"driving {gp('track')} / {gp('motion')} / seed {gp('seed')} ({gp('laps')} laps, {self.path.length:.1f} m per lap) on the {self.source} pose")

    def _on_est(self, m: Odometry) -> None:
        p = m.pose.pose
        self.est = (Time.from_msg(m.header.stamp).nanoseconds, (p.position.x, p.position.y, _yaw(p.orientation)), float(m.twist.twist.linear.x),
                    float(m.twist.twist.angular.z))

    def _on_truth(self, m: Odometry) -> None:
        p = m.pose.pose
        self.truth = (Time.from_msg(m.header.stamp).nanoseconds, (p.position.x, p.position.y, _yaw(p.orientation)), float(m.twist.twist.linear.x))

    def _on_est_status(self, m: String) -> None:
        try:
            self.estimator_ready = self.estimator_ready or bool(json.loads(m.data).get("ready", False))
        except ValueError:
            pass

    def _pose(self, now: int) -> tuple[tuple[float, float, float], float, float, tuple[float, float, float]] | None:
        """(world pose, speed, age [s], map->odom correction) of the selected source."""
        if self.source == "truth":
            if self.truth is None:
                return None
            return self.truth[1], self.truth[2], (now - self.truth[0]) * 1e-9, (0.0, 0.0, 0.0)
        if self.est is None:
            return None
        try:
            tf = self.tf.lookup_transform(self.map_frame, self.odom_frame, Time())
            q = tf.transform.rotation
            map_odom = (tf.transform.translation.x, tf.transform.translation.y, _yaw(q))
        except TransformException:
            map_odom = (0.0, 0.0, 0.0)  # SLAM not started yet: map = odom
        # Latency compensation: advance the estimate by its age with the
        # estimated speed and yaw rate (constant twist).
        age = max(0.0, (now - self.est[0]) * 1e-9)
        v, w = self.est[2], self.est[3]
        x, y, yaw = self.est[1]
        mid = yaw + 0.5 * w * age
        odom_now = (x + v * age * math.cos(mid), y + v * age * math.sin(mid), yaw + w * age)
        world = _compose(self.t_world_map, _compose(map_odom, odom_now))
        return world, v, age, map_odom

    def _publish(self, v: float, steer: float) -> None:
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(v * math.tan(steer) / self.wheelbase)
        self.cmd_pub.publish(msg)

    def _control(self) -> None:
        now = self.get_clock().now().nanoseconds
        pose = self._pose(now)
        if self.phase == PHASE_WAIT:
            self._publish(0.0, 0.0)
            if pose is not None and self.estimator_ready:
                self.ready_since = self.ready_since or now
                if now - self.ready_since >= self.static_ns:
                    self.phase = PHASE_DRIVE
                    self.get_logger().info("estimator ready and static start done: driving")
            return
        if pose is None or pose[2] * 1e9 > self.max_age_ns:
            self._publish(0.0, 0.0)  # no fresh pose: stop, like the APEX tracker
            self._status(now, "stale_pose", pose)
            return
        (x, y, yaw), speed, age, map_odom = pose
        xr, yr = x + self.rear * math.cos(yaw), y + self.rear * math.sin(yaw)
        s_tot, lat = self.pursuit.locate(xr, yr)
        steer = self.pursuit.steering(xr, yr, yaw, speed)
        v = self.plan.target(s_tot + max(0.3, 0.3 * speed))
        if self.phase == PHASE_DRIVE and self.stop_idx < len(self.stops) and s_tot >= self.stops[self.stop_idx][0] - 0.35:
            self.phase, self.hold_since = PHASE_HOLD, None
        if self.phase == PHASE_HOLD:
            v = 0.0
            if abs(speed) < 0.02 and self.hold_since is None:
                self.hold_since = now
            if self.hold_since is not None and (now - self.hold_since) * 1e-9 >= self.stops[self.stop_idx][1]:
                self.stop_idx += 1
                self.phase = PHASE_DRIVE
                v = self.plan.target(s_tot + 0.3)
        if self.phase == PHASE_DRIVE and s_tot >= self.total - 0.6:
            self.phase = PHASE_FINAL
        if self.phase in (PHASE_FINAL, PHASE_DONE):
            v = 0.0
            if abs(speed) < 0.02:
                self.phase = PHASE_DONE
        self._publish(v, steer)
        if self.log is not None:
            self.log.writerow([f"{now * 1e-9:.4f}", self.phase, f"{x:.4f}", f"{y:.4f}", f"{yaw:.5f}", f"{speed:.3f}", f"{s_tot:.3f}", f"{lat:.4f}",
                               f"{v:.3f}", f"{math.degrees(steer):.3f}", f"{age:.4f}", f"{map_odom[0]:.4f}", f"{map_odom[1]:.4f}", f"{map_odom[2]:.5f}"])
        self._status(now, "ok", pose, s_tot, lat)

    def _status(self, now: int, state: str, pose, s: float = float("nan"), lat: float = float("nan")) -> None:  # noqa: ANN001
        if now % 500_000_000 < 20_000_000:  # ~2 Hz
            self.status_pub.publish(json_msg({"phase": self.phase, "state": state, "s_est_m": s, "lateral_est_m": lat, "total_m": self.total,
                                              "stop_idx": self.stop_idx, "source": self.source}))

    def destroy_node(self) -> None:
        if self.log is not None:
            self._log_file.close()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrackDriverNode()
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
