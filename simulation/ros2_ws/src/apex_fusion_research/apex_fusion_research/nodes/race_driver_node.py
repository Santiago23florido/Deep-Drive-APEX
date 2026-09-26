"""Race driver of the car on a track it has never seen (explore, map, plan, race).

The car arrives with its trained odometry model and nothing else about the
track: no map, no reference path, no speed calibration. It

1. drives lap 1 reactively from the LiDAR (follows the wall it sees, keeps
   clear of everything it sees, never farther from the wall than the generic
   lane rule allows) while slam_toolbox builds the map on the learned odometry;
2. detects the closure of the lap on its own map pose and keeps driving a few
   metres so the SLAM closes the loop;
3. plans a race line on its map (in a separate process, still driving) and
   follows it with pure pursuit for the remaining laps, counted on its map pose.

Inputs (all produced by the car): the learned odometry (``/apex/odometry/
learned``, ``/apex/odometry/learned_predicted``, the de-skewed scan, its
status), slam_toolbox (``/map``, its graph, TF ``map -> odom_learned``). No
ground-truth topic, no track, motion or seed parameter.

Output: ``geometry_msgs/Twist`` on ``/apex/cmd_vel_track`` (the real car's
interface). Logs: ``driver.csv`` (map frame), ``race_events.json`` and the
plan artifacts (``<output_dir>/``: map snapshot, lap-1 path, corridor, race line).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
import multiprocessing
import os
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path as PathMsg
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from ..core.race_config import RaceDriverConfig, vehicle_limits
from ..core.race_driver import DONE, LOG_COLUMNS, RaceDriverCore
from ..core.race_planner import Grid, save_plan
from ..core.sim_paths import ensure_tools_path
from ._common import LATCHED_QOS, ConfigParameters, json_msg

MAP_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def _yaw(q) -> float:  # noqa: ANN001
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def _nice() -> None:
    os.nice(10)  # the planner must not delay the estimator or the SLAM


class RaceDriverNode(Node):
    def __init__(self) -> None:
        super().__init__("apex_race_driver")
        dp = self.declare_parameter
        dp("laps", 3)
        dp("vehicle_yaml", "")  # the car's own description (model / esc / servo / controller sections)
        dp("odom_topic", "/apex/odometry/learned")
        dp("predicted_topic", "/apex/odometry/learned_predicted")
        dp("scan_topic", "/apex/learned_odometry/scan_deskewed")
        dp("estimator_status_topic", "/apex/learned_odometry/status")
        dp("map_topic", "/map")
        dp("graph_topic", "/slam_toolbox/graph_visualization")
        dp("map_frame", "map")
        dp("odom_frame", "odom_learned")
        dp("base_frame", "base_link")
        dp("laser_frame", "laser")
        dp("cmd_topic", "/apex/cmd_vel_track")
        dp("status_topic", "/apex/race_driver/status")
        dp("rate_hz", 50.0)
        dp("static_start_s", 1.5)
        dp("max_pose_age_s", 0.3)
        dp("output_dir", "")
        dp("log_csv", "")
        dp("events_json", "")
        dp("planner_in_process", False)  # true: plan in the node's process (debug); false: a separate process
        self.params = ConfigParameters(self, RaceDriverConfig)
        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        cfg = self.params.config
        yaml_path = str(gp("vehicle_yaml"))
        if yaml_path:
            ensure_tools_path()
            from pose_dataset.vehicle import load_vehicle_config  # noqa: PLC0415

            cfg.vehicle = vehicle_limits(load_vehicle_config(Path(yaml_path)), cfg.vehicle.steer_plan_deg)
        self.cfg = cfg
        self.frames = {k: str(gp(f"{k}_frame")) for k in ("map", "odom", "base", "laser")}
        self.max_age = float(gp("max_pose_age_s"))
        self.out_dir = Path(str(gp("output_dir"))) if str(gp("output_dir")) else None
        self.events_path = Path(str(gp("events_json"))) if str(gp("events_json")) else None
        submit = None
        self.pool = None
        if not bool(gp("planner_in_process")):
            self.pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"), initializer=_nice)
            submit = self.pool.submit
        self.core = RaceDriverCore(cfg, int(gp("laps")), submit, float(gp("static_start_s")))
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.base_to_laser: tuple[float, float, float] | None = None
        self.poses: dict[int, tuple[float, float, float]] = {}  # odometry pose at each scan stamp
        self.pred: tuple[int, tuple[float, float, float], float, float] | None = None  # stamp, pose, speed, yaw rate
        self.ready = False
        self.plan_saved = False
        self.cmd_pub = self.create_publisher(Twist, str(gp("cmd_topic")), 20)
        self.status_pub = self.create_publisher(String, str(gp("status_topic")), 10)
        self.local_pub = self.create_publisher(PathMsg, "/apex/race_driver/local_path", 5)
        self.line_pub = self.create_publisher(PathMsg, "/apex/race_driver/race_line", LATCHED_QOS)
        self.cor_pub = [self.create_publisher(PathMsg, f"/apex/race_driver/corridor_{s}", LATCHED_QOS) for s in ("left", "right")]
        self.create_subscription(Odometry, str(gp("predicted_topic")), self._on_pred, 50)
        self.create_subscription(Odometry, str(gp("odom_topic")), self._on_odom, 50)
        self.create_subscription(LaserScan, str(gp("scan_topic")), self._on_scan, 10)
        self.create_subscription(String, str(gp("estimator_status_topic")), self._on_status, 10)
        self.create_subscription(OccupancyGrid, str(gp("map_topic")), self._on_map, MAP_QOS)
        self.create_subscription(MarkerArray, str(gp("graph_topic")), self._on_graph, 2)
        self.create_timer(1.0 / float(gp("rate_hz")), self._control)
        self.create_timer(0.5, self._publish_status)
        self.log = None
        if str(gp("log_csv")):
            Path(str(gp("log_csv"))).parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(str(gp("log_csv")), "w", newline="", encoding="utf-8")
            self.log = csv.writer(self._log_file)
            self.log.writerow(LOG_COLUMNS)
        self.core.events["subscribed_topics"] = sorted({s.topic_name for s in self.subscriptions})
        self.get_logger().info(f"race driver: {self.core.laps} laps on an unknown track (lap 1 reactive, w_min {cfg.reactive.w_min_m} m, "
                               f"race line {cfg.planner.mode}); subscribed to {self.core.events['subscribed_topics']}")

    # ------------------------------------------------------------------ inputs
    def _now(self) -> int:
        return self.get_clock().now().nanoseconds

    def _on_status(self, m: String) -> None:
        try:
            self.ready = self.ready or bool(json.loads(m.data).get("ready", False))
        except ValueError:
            pass

    def _on_pred(self, m: Odometry) -> None:
        p = m.pose.pose
        self.pred = (Time.from_msg(m.header.stamp).nanoseconds, (p.position.x, p.position.y, _yaw(p.orientation)),
                     float(m.twist.twist.linear.x), float(m.twist.twist.angular.z))

    def _on_odom(self, m: Odometry) -> None:
        p = m.pose.pose
        self.poses[Time.from_msg(m.header.stamp).nanoseconds] = (p.position.x, p.position.y, _yaw(p.orientation))
        if len(self.poses) > 200:
            for k in sorted(self.poses)[:-100]:
                del self.poses[k]

    def _pose_now(self, now: int) -> tuple[tuple[float, float, float], float, float] | None:
        """Predicted odometry pose advanced by its age (constant twist), speed, age [s]."""
        if self.pred is None:
            return None
        age = max(0.0, (now - self.pred[0]) * 1e-9)
        v, w = self.pred[2], self.pred[3]
        x, y, yaw = self.pred[1]
        mid = yaw + 0.5 * w * age
        return (x + v * age * math.cos(mid), y + v * age * math.sin(mid), yaw + w * age), v, age

    def _laser_offset(self) -> tuple[float, float, float] | None:
        if self.base_to_laser is None:
            try:
                tf = self.tf.lookup_transform(self.frames["base"], self.frames["laser"], Time())
                self.base_to_laser = (tf.transform.translation.x, tf.transform.translation.y, _yaw(tf.transform.rotation))
                self.core.base_to_laser = self.base_to_laser
            except TransformException:
                return None
        return self.base_to_laser

    def _on_scan(self, m: LaserScan) -> None:
        if self._laser_offset() is None:
            return
        stamp = Time.from_msg(m.header.stamp).nanoseconds
        pose = self.poses.get(stamp)
        if pose is None:
            try:
                tf = self.tf.lookup_transform(self.frames["odom"], self.frames["base"], Time(nanoseconds=stamp))
                pose = (tf.transform.translation.x, tf.transform.translation.y, _yaw(tf.transform.rotation))
            except TransformException:
                return
        now = self._pose_now(self._now())
        if now is None:
            return
        self.core.on_scan(stamp * 1e-9, np.asarray(m.ranges, dtype=float), float(m.angle_min), float(m.angle_increment), pose, now[0], now[1])
        if self.core.local_path_odom is not None:
            self._publish_path(self.local_pub, self.core.local_path_odom, self.frames["odom"])

    def _on_map(self, m: OccupancyGrid) -> None:
        if abs(_yaw(m.info.origin.orientation)) > 1e-6:
            self.get_logger().warn("rotated map origin: not supported", throttle_duration_sec=30.0)
            return
        grid = Grid.from_flat(m.data, m.info.width, m.info.height, m.info.resolution, m.info.origin.position.x, m.info.origin.position.y)
        self.core.on_map(self._now() * 1e-9, grid)

    def _on_graph(self, m: MarkerArray) -> None:
        verts = sorted((mk.id, mk.pose.position.x, mk.pose.position.y) for mk in m.markers if mk.type == Marker.SPHERE and mk.action == Marker.ADD)
        if verts:
            self.core.on_graph(self._now() * 1e-9, np.array([(x, y) for _, x, y in verts]))

    # ----------------------------------------------------------------- control
    def _map_odom(self) -> tuple[float, float, float] | None:
        try:
            tf = self.tf.lookup_transform(self.frames["map"], self.frames["odom"], Time())
            return tf.transform.translation.x, tf.transform.translation.y, _yaw(tf.transform.rotation)
        except TransformException:
            return None

    def _control(self) -> None:
        now = self._now()
        pose = self._pose_now(now)
        if pose is None or (pose[2] > self.max_age and self.core.phase not in ("wait",)):
            self._publish_cmd(0.0, 0.0)  # no fresh pose: stop, like the APEX tracker
            return
        v, steer, row = self.core.control(now * 1e-9, pose[0], pose[1], self._map_odom(), self.ready, pose[2])
        self._publish_cmd(v, self.core.steer_command(steer))
        if self.log is not None and row is not None:
            self.log.writerow(row)
        if self.core.plan is not None and not self.plan_saved:
            self._save_plan()
        if self.core.phase == DONE and not getattr(self, "_done_written", False):
            self._done_written = True
            self._write_events()
            self.get_logger().info(f"race finished: {self.core.counter.laps} laps")

    def _publish_cmd(self, v: float, steer: float) -> None:
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(v * math.tan(steer) / self.cfg.vehicle.wheelbase_m)
        self.cmd_pub.publish(msg)

    # ------------------------------------------------------------------ output
    def _publish_path(self, pub, xy: np.ndarray, frame: str, closed: bool = False) -> None:  # noqa: ANN001
        msg = PathMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame
        pts = np.vstack((xy, xy[:1])) if closed else xy
        for x, y in pts:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        pub.publish(msg)

    def _save_plan(self) -> None:
        self.plan_saved = True
        plan = self.core.plan
        self._publish_path(self.line_pub, plan.path.xy, self.frames["map"], closed=True)
        n = plan.ref.normal
        for pub, bound in zip(self.cor_pub, (plan.cor.hi, plan.cor.lo)):
            self._publish_path(pub, plan.ref.xy + bound[:, None] * n, self.frames["map"], closed=True)
        self.get_logger().info(f"race line ready: {plan.path.length:.1f} m, predicted lap {plan.t_lap_pred:.1f} s, "
                               f"map clearance >= {100 * plan.diag['min_map_clearance_m']:.0f} cm (from the {self.core.events['planning']['lap_path_source']})")
        if self.out_dir is not None:
            grid, loop, side = self.core.plan_inputs
            save_plan(plan, grid, loop, side, self.out_dir, {"planning": self.core.events.get("planning"), "start_map": self.core.events.get("start_map")})
        self._write_events()

    def _write_events(self) -> None:
        if self.events_path is None:
            return
        ev = {k: v for k, v in self.core.events.items() if k != "config"}
        ev["config"] = self.core.events.get("config")
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.write_text(json.dumps(ev, indent=1, default=float), encoding="utf-8")

    def _publish_status(self) -> None:
        self.status_pub.publish(json_msg(self.core.status()))

    def destroy_node(self) -> None:
        self._write_events()
        if self.log is not None:
            self._log_file.close()
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RaceDriverNode()
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
