"""Race driver of the real car on a track it has never seen (explore, map, plan, race).

This is the simulation node
(``simulation/ros2_ws/src/apex_fusion_research/apex_fusion_research/nodes/race_driver_node.py``)
with the differences the real car needs, and nothing else. The driving logic it
drives -- ``core/race_driver.py``, ``core/reactive.py``, ``core/race_planner.py``,
``core/plant_id.py``, ``core/adaptive.py`` -- is byte-identical to the
simulation's (``test/test_core_parity.py``).

The car arrives with its trained odometry model and nothing else about the
track: no map, no reference path, no speed calibration. It

1. drives lap 1 reactively from the LiDAR (follows the wall it sees, keeps
   clear of everything it sees, never farther from the wall than the generic
   lane rule allows) while slam_toolbox builds the map on the learned odometry;
2. detects the closure of the lap on its own map pose and keeps driving a few
   metres so the SLAM closes the loop;
3. plans a race line on its map (in a separate process, still driving) and
   follows it with pure pursuit for the remaining laps, counted on its map pose.

Differences from the simulation node, each one a real-car requirement:

* the car's description is read by ``core/vehicle_io.py``, not by the
  simulation's ``pose_dataset.vehicle`` (which also builds the Gazebo model);
* the estimator's health gate is numeric (``estimator_health.py``);
* ``/apex/tracking/arm`` must be true *and fresh* or the car is commanded to
  zero: on the real car the arm is the only thing between a bug and a moving
  vehicle, and a dead arm publisher must stop it;
* the de-skewed scan is subscribed best-effort, depth 1: when the reactive step
  is slower than a revolution the middleware keeps the newest scan instead of
  queueing stale ones, and a scan older than ``scan_stale_s`` is dropped;
* the control timer runs in its own callback group on a second executor thread
  and takes the core lock *without blocking*: while a reactive step holds the
  core, the control tick re-publishes the last command, and if the core stays
  busy longer than ``core_busy_stop_s`` it commands zero. The bridge's 0.35 s
  watchdog must never be the thing that discovers a slow planner;
* the planner worker is pinned away from the estimator and the SLAM.

Output: ``geometry_msgs/Twist`` on ``/apex/cmd_vel_track``. Logs: ``driver.csv``
(map frame), ``race_events.json`` and the plan artifacts.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from collections import deque
from contextlib import contextmanager
import csv
import json
import math
import multiprocessing
import os
from pathlib import Path
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path as PathMsg
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy._rclpy_pybind11 import RCLError
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from ..core.race_config import RaceDriverConfig, vehicle_limits
from ..core.race_driver import DONE, FINAL, LOG_COLUMNS, RaceDriverCore
from ..core.race_planner import Grid, save_plan
from ..core.vehicle_io import load_vehicle_config
from ._common import LATCHED_QOS, ConfigParameters, json_msg
from .estimator_health import estimator_ok

MAP_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
# Newest revolution only: a reactive step slower than 100 ms must not build a
# queue of stale scans to plan on. Best-effort so it never back-pressures the
# estimator, which publishes reliably.
SCAN_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
# Predicted poses are a latest-value channel, not a measurement history.
# Reading them must remain possible while a scan occupies the sensor group.
PREDICTED_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
# The arm is deliberately not TRANSIENT_LOCAL: a latched `true` must not
# survive the death of whoever published it.
ARM_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)

# Columns of inputs_csv: the arguments of RaceDriverCore.on_scan, the travel and the phase at that call, then the ranges.
INPUT_COLUMNS = ["stamp_ns", "now_ns", "phase", "angle_min", "angle_inc", "x_stamp", "y_stamp", "yaw_stamp", "x_now", "y_now", "yaw_now", "speed", "travel",
                 "laser_x", "laser_y", "laser_yaw"]


def _yaw(q) -> float:  # noqa: ANN001
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def _nice(cpus: list[int]) -> None:
    os.nice(10)  # the planner must not delay the estimator or the SLAM
    if cpus:  # ... and must not run on their cores at all
        try:
            os.sched_setaffinity(0, set(cpus))
        except OSError:
            pass


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
        dp("arm_topic", "/apex/tracking/arm")
        dp("require_arm", True)  # false only for a bench replay with no actuator in the loop
        dp("arm_timeout_s", 0.5)  # an arm older than this is not an arm
        dp("rate_hz", 50.0)
        dp("static_start_s", 1.5)
        dp("max_pose_age_s", 0.3)
        dp("max_scan_age_s", 0.6)  # the last LiDAR correction of the estimator (the predicted pose only needs the IMU)
        dp("scan_stale_s", 0.25)  # a de-skewed revolution older than this is superseded; drop it
        dp("core_busy_stop_s", 0.25)  # the core busy longer than this: command zero, do not hold
        dp("imu_rate_hz", 100.0)  # calibration rates of the health gate (WT61PC at 100 Hz, A2M8 Express at 10 Hz)
        dp("lidar_rate_hz", 10.0)
        dp("planner_cpus", [2, 3])  # cores the planner worker may use (empty: no pinning)
        dp("output_dir", "")
        dp("log_csv", "")
        dp("events_json", "")
        dp("inputs_csv", "")  # lap-1 scans as the reactive driver received them (offline replay: tools/analysis/replay_reactive.py)
        dp("planner_in_process", False)  # true: plan in the node's process (debug); false: a separate process
        self.params = ConfigParameters(self, RaceDriverConfig)
        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        cfg = self.params.config
        yaml_path = str(gp("vehicle_yaml"))
        if yaml_path:
            configured = cfg.vehicle
            cfg.vehicle = vehicle_limits(load_vehicle_config(Path(yaml_path)), configured.steer_plan_deg)
            for name in ("steer_max_left_deg", "steer_max_right_deg", "steer_plan_left_deg", "steer_plan_right_deg"):
                setattr(cfg.vehicle, name, getattr(configured, name))
        self.cfg = cfg
        self.frames = {k: str(gp(f"{k}_frame")) for k in ("map", "odom", "base", "laser")}
        self.max_age = float(gp("max_pose_age_s"))
        self.max_scan_age = float(gp("max_scan_age_s"))
        self.scan_stale = float(gp("scan_stale_s"))
        self.core_busy_stop = float(gp("core_busy_stop_s"))
        self.rates = (float(gp("imu_rate_hz")), float(gp("lidar_rate_hz")))
        self.require_arm = bool(gp("require_arm"))
        self.arm_timeout = float(gp("arm_timeout_s"))
        self.scan_age_stops = 0
        self.counters = {"disarmed": 0, "core_busy_hold": 0, "core_busy_stop": 0, "scan_stale": 0, "pose_age_stop": 0}
        self.out_dir = Path(str(gp("output_dir"))) if str(gp("output_dir")) else None
        self.events_path = Path(str(gp("events_json"))) if str(gp("events_json")) else None
        submit = None
        self.pool = None
        if not bool(gp("planner_in_process")):
            cpus = [int(c) for c in (gp("planner_cpus") or [])]
            self.pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                            initializer=_nice, initargs=(cpus,))
            # Start the worker now, while the car is standing still. A spawned
            # worker is a fresh interpreter that re-imports numpy and scipy, and
            # ProcessPoolExecutor creates it inside the first submit() -- which
            # happens in the control callback at the end of lap 1, with the car
            # driving. Measured on a workstation: a 3.6 s stall in the control
            # loop, ten times the bridge's command watchdog.
            try:
                self.pool.submit(os.getpid).result(timeout=60.0)
                self.get_logger().info("planner worker ready")
            except Exception as exc:  # noqa: BLE001 - a missing worker is not fatal, it is slower
                self.get_logger().warning(f"planner worker did not pre-start: {exc}")
            submit = self.pool.submit
        self.core = RaceDriverCore(cfg, int(gp("laps")), submit, float(gp("static_start_s")))
        # The core is touched by the sensor callbacks and by the control timer,
        # which run on two executor threads: the control tick never waits for it.
        self._core_lock = threading.Lock()
        # A run of scan/odom/map callbacks must leave a turn for a control
        # tick that already missed the lock. The sensor thread waits outside
        # the lock; control itself always remains nonblocking.
        self._control_served = threading.Event()
        self._control_served.set()
        self._handoff_timeout_s = min(self.core_busy_stop, 2.0 / float(gp("rate_hz")))
        self._fallback_commands = deque()
        # Who holds the core, and for how long. The control tick cannot run
        # while one of these is in progress, so this is the budget that decides
        # whether the command stream stays alive.
        self.hold_ms: dict[str, list[float]] = {k: [0.0, 0.0, 0.0] for k in ("scan", "odom", "map", "graph", "status", "control")}
        self._busy_since: int | None = None
        self._last_cmd = (0.0, 0.0)  # (v, steer) actually published, for the hold
        self._last_cmd_ns: int | None = None
        self._cmd_gap_max_ms = 0.0
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.base_to_laser: tuple[float, float, float] | None = None
        self.poses: dict[int, tuple[float, float, float]] = {}  # odometry pose at each scan stamp
        self.pred: tuple[int, tuple[float, float, float], float, float] | None = None  # stamp, pose, speed, yaw rate
        self.ready = False
        self.estimator_ok = False  # until the health report says the rates are right
        self.estimator_reason = "no health report"
        self.armed = not self.require_arm
        self._mission_deadline_s = None
        self._mission_time_stop = False
        self.arm_ns: int | None = None
        self.plan_saved = False
        # Heavy core inputs serialize in sensors. Latest predicted pose and
        # health updates share the short control group, never the scan queue.
        sensors = MutuallyExclusiveCallbackGroup()
        control = MutuallyExclusiveCallbackGroup()
        self.cmd_pub = self.create_publisher(Twist, str(gp("cmd_topic")), 20)
        self.status_pub = self.create_publisher(String, str(gp("status_topic")), 10)
        self.local_pub = self.create_publisher(PathMsg, "/apex/race_driver/local_path", 5)
        self.line_pub = self.create_publisher(PathMsg, "/apex/race_driver/race_line", LATCHED_QOS)
        self.cor_pub = [self.create_publisher(PathMsg, f"/apex/race_driver/corridor_{s}", LATCHED_QOS) for s in ("left", "right")]
        self.create_subscription(Odometry, str(gp("predicted_topic")), self._on_pred, PREDICTED_QOS, callback_group=control)
        self.create_subscription(Odometry, str(gp("odom_topic")), self._on_odom, 50, callback_group=sensors)
        self.create_subscription(LaserScan, str(gp("scan_topic")), self._on_scan, SCAN_QOS, callback_group=sensors)
        self.create_subscription(String, str(gp("estimator_status_topic")), self._on_status, 1, callback_group=control)
        self.create_subscription(OccupancyGrid, str(gp("map_topic")), self._on_map, MAP_QOS, callback_group=sensors)
        self.create_subscription(MarkerArray, str(gp("graph_topic")), self._on_graph, 2, callback_group=sensors)
        self.create_subscription(Bool, str(gp("arm_topic")), self._on_arm, ARM_QOS, callback_group=control)
        self.create_timer(1.0 / float(gp("rate_hz")), self._control, callback_group=control)
        self.create_timer(0.5, self._publish_status, callback_group=sensors)
        self.log = None
        if str(gp("log_csv")):
            Path(str(gp("log_csv"))).parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(str(gp("log_csv")), "w", newline="", encoding="utf-8")
            self.log = csv.writer(self._log_file)
            self.log.writerow(LOG_COLUMNS)
        self.inputs = None
        if str(gp("inputs_csv")):
            Path(str(gp("inputs_csv"))).parent.mkdir(parents=True, exist_ok=True)
            self._inputs_file = open(str(gp("inputs_csv")), "w", newline="", encoding="utf-8")
            self.inputs = csv.writer(self._inputs_file)
            self.inputs.writerow(INPUT_COLUMNS)
        self.core.events["subscribed_topics"] = sorted({s.topic_name for s in self.subscriptions})
        self.get_logger().info(f"race driver: {self.core.laps} laps on an unknown track (lap 1 reactive, w_min {cfg.reactive.w_min_m} m, "
                               f"race line {cfg.planner.mode}), steer <= {cfg.vehicle.steer_max_deg:.1f} deg, "
                               f"v_explore {cfg.reactive.v_explore_mps:.2f} / v_max {cfg.race.v_max_mps:.2f} m/s; "
                               f"arm {'required on ' + str(gp('arm_topic')) if self.require_arm else 'NOT required'}; "
                               f"subscribed to {self.core.events['subscribed_topics']}")

    @contextmanager
    def _core(self, what: str):
        """Hold the core lock, and record how long for."""
        self._control_served.wait(timeout=self._handoff_timeout_s)
        t0 = time.perf_counter()
        with self._core_lock:
            wait = (time.perf_counter() - t0) * 1e3
            try:
                self._flush_fallback_commands()
                yield
            finally:
                held = (time.perf_counter() - t0) * 1e3 - wait
                row = self.hold_ms[what]
                row[0] += 1.0
                row[1] += held
                row[2] = max(row[2], held)

    def _flush_fallback_commands(self) -> None:
        """Under the core lock, teach identification what was really sent.

        Holds and watchdog zeros happen outside core.control(). Omitting them
        makes pauses look like a motor or steering response to an old command.
        The control callback is the only producer; deque transfers are atomic.
        """
        while self._fallback_commands:
            t, curvature, speed = self._fallback_commands.popleft()
            self.core.ident.push_command(t, curvature, speed)

    def hold_report(self) -> dict[str, dict[str, float]]:
        return {k: {"n": int(n), "mean_ms": round(total / n, 1) if n else 0.0, "max_ms": round(worst, 1)}
                for k, (n, total, worst) in self.hold_ms.items() if n}

    # ------------------------------------------------------------------ inputs
    def _now(self) -> int:
        return self.get_clock().now().nanoseconds

    def _on_arm(self, m: Bool) -> None:
        self.armed = bool(m.data)
        self.arm_ns = self._now()

    def _on_status(self, m: String) -> None:
        try:
            payload = json.loads(m.data)
        except ValueError:
            return
        self.ready = self.ready or bool(payload.get("ready", False))
        # An estimator whose sensors are off *rate* is not a channel to
        # identify a plant from: the model integrates the gyro over each
        # revolution and expects the interval it was trained with. Jitter
        # around the right rate is a different matter -- it is reported too,
        # and what a noisier estimate deserves is a larger sigma, which it
        # already has. So the gate is on the measured rates, not on the words.
        ok, reason = estimator_ok(payload.get("health"), *self.rates)
        if ok != self.estimator_ok:
            self.get_logger().info(f"estimator usable for identification: {ok}" + (f" ({reason})" if reason else ""))
        self.estimator_ok, self.estimator_reason = ok, reason

    def _on_pred(self, m: Odometry) -> None:
        p = m.pose.pose
        self.pred = (Time.from_msg(m.header.stamp).nanoseconds, (p.position.x, p.position.y, _yaw(p.orientation)),
                     float(m.twist.twist.linear.x), float(m.twist.twist.angular.z))

    def _on_odom(self, m: Odometry) -> None:
        p = m.pose.pose
        stamp = Time.from_msg(m.header.stamp).nanoseconds
        pose = (p.position.x, p.position.y, _yaw(p.orientation))
        self.poses[stamp] = pose
        if len(self.poses) > 200:
            for k in sorted(self.poses)[:-100]:
                del self.poses[k]
        # This topic carries the only measurement of the car's own behaviour it
        # has: the pose increment of one revolution, with the sigma the network
        # publishes for it on the twist covariance (dx, dy, dyaw).
        cov = m.twist.covariance
        sigma = tuple(math.sqrt(max(float(cov[i]), 0.0)) for i in (0, 7, 35))
        with self._core("odom"):
            self.core.on_odometry(stamp * 1e-9, pose, sigma if all(sigma) else None, self.estimator_ok)

    def _pose_now(self, now: int) -> tuple[tuple[float, float, float], float, float] | None:
        """Predicted odometry pose advanced by its age (constant twist), speed, age [s]."""
        pred = self.pred  # one atomic snapshot across the sensor/control groups
        if pred is None:
            return None
        age = max(0.0, (now - pred[0]) * 1e-9)
        v, w = pred[2], pred[3]
        x, y, yaw = pred[1]
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
        t_now = self._now()
        if (t_now - stamp) * 1e-9 > self.scan_stale:
            # A newer revolution exists (or will, within one period): planning on
            # this one would steer on where the car was, not where it is.
            self.counters["scan_stale"] += 1
            return
        pose = self.poses.get(stamp)
        if pose is None:
            try:
                tf = self.tf.lookup_transform(self.frames["odom"], self.frames["base"], Time(nanoseconds=stamp))
                pose = (tf.transform.translation.x, tf.transform.translation.y, _yaw(tf.transform.rotation))
            except TransformException:
                return
        now = self._pose_now(t_now)
        if now is None:
            return
        if self.inputs is not None and self.core.phase not in (FINAL, DONE):
            vals = (*pose, *now[0], now[1], self.core.travel, *self.base_to_laser)
            self.inputs.writerow([stamp, t_now, self.core.phase, f"{m.angle_min:.9f}", f"{m.angle_increment:.9f}", *(f"{v:.6f}" for v in vals),
                                  *(f"{r:.4f}" for r in m.ranges)])
        with self._core("scan"):
            self.core.on_scan(stamp * 1e-9, np.asarray(m.ranges, dtype=float), float(m.angle_min), float(m.angle_increment), pose, now[0], now[1])
            local = self.core.local_path_odom
        if local is not None:
            self._publish_path(self.local_pub, local, self.frames["odom"])

    def _on_map(self, m: OccupancyGrid) -> None:
        if abs(_yaw(m.info.origin.orientation)) > 1e-6:
            self.get_logger().warn("rotated map origin: not supported", throttle_duration_sec=30.0)
            return
        grid = Grid.from_flat(m.data, m.info.width, m.info.height, m.info.resolution, m.info.origin.position.x, m.info.origin.position.y)
        with self._core("map"):
            self.core.on_map(self._now() * 1e-9, grid)

    def _on_graph(self, m: MarkerArray) -> None:
        verts = sorted((mk.id, mk.pose.position.x, mk.pose.position.y) for mk in m.markers if mk.type == Marker.SPHERE and mk.action == Marker.ADD)
        if verts:
            with self._core("graph"):
                self.core.on_graph(self._now() * 1e-9, np.array([(x, y) for _, x, y in verts]))

    # ----------------------------------------------------------------- control
    def _map_odom(self) -> tuple[float, float, float] | None:
        try:
            tf = self.tf.lookup_transform(self.frames["map"], self.frames["odom"], Time())
            return tf.transform.translation.x, tf.transform.translation.y, _yaw(tf.transform.rotation)
        except TransformException:
            return None

    def _scan_age(self, now: int) -> float:
        """Age [s] of the last estimate corrected by a LiDAR scan (inf before the first)."""
        return (now - max(self.poses)) * 1e-9 if self.poses else math.inf

    def _arm_reason(self, now: int) -> str:
        """Why the car may not be commanded, or an empty string."""
        if not self.require_arm:
            return ""
        if not self.armed:
            return "disarmed"
        if self.arm_ns is None or (now - self.arm_ns) * 1e-9 > self.arm_timeout:
            return "arm stale"
        return ""

    def _control(self) -> None:
        now = self._now()
        steady_now = time.monotonic()
        if self._arm_reason(now):
            self.counters["disarmed"] += 1
            self._publish_cmd(0.0, 0.0)
            self._control_served.set()
            return
        if self._mission_deadline_s is not None and steady_now >= self._mission_deadline_s:
            self._mission_time_stop = True
        if self._mission_time_stop:
            # The deadline must cut the command even while perception owns
            # the core lock. Latch the core's DONE event once it is available.
            self._publish_cmd(0.0, 0.0)
            if self._core_lock.acquire(blocking=False):
                try:
                    self._flush_fallback_commands()
                    if self.core.phase != DONE:
                        self.core._stop_mission(now * 1e-9, "duration")
                    if not getattr(self, "_done_written", False):
                        self._done_written = True
                        self._write_events()
                finally:
                    self._core_lock.release()
                    self._control_served.set()
            else:
                self._control_served.clear()
            return
        if not self._core_lock.acquire(blocking=False):
            self._control_served.clear()
            self._hold_command(now)  # a reactive step or the map is in the core
            return
        t_lock = time.perf_counter()
        try:
            self._busy_since = None
            self._flush_fallback_commands()
            pose = self._pose_now(now)
            if pose is None or (pose[2] > self.max_age and self.core.phase not in ("wait",)):
                self.counters["pose_age_stop"] += 1
                self._publish_cmd(0.0, 0.0)  # no fresh pose: stop, like the APEX tracker
                return
            if self.core.phase not in ("wait",) and self._scan_age(now) > self.max_scan_age:
                # The IMU keeps the predicted pose fresh, but without LiDAR corrections it
                # drifts: an estimator that stalled or lost the LiDAR must stop the car too.
                self.scan_age_stops += 1
                self._publish_cmd(0.0, 0.0)
                return
            v, steer, row = self.core.control(now * 1e-9, pose[0], pose[1], self._map_odom(), self.ready, pose[2])
            if self.cfg.mission.enabled and self.core.mission_started_t is not None and self._mission_deadline_s is None:
                self._mission_deadline_s = steady_now + self.cfg.mission.max_duration_s
            self._publish_cmd(v, self.core.steer_command(steer), record_fallback=False)
            if self.log is not None and row is not None:
                self.log.writerow(row)
            if self.core.plan is not None and not self.plan_saved:
                self._save_plan()
            if self.core.phase == DONE and not getattr(self, "_done_written", False):
                self._done_written = True
                self._write_events()
                self.get_logger().info(f"race finished: {self.core.counter.laps} laps")
        finally:
            row = self.hold_ms["control"]
            held = (time.perf_counter() - t_lock) * 1e3
            row[0] += 1.0
            row[1] += held
            row[2] = max(row[2], held)
            self._core_lock.release()
            self._control_served.set()

    def _hold_command(self, now: int) -> None:
        """The core is busy: hold the last command briefly, then stop the car.

        The command stream is what the actuation bridge watches (0.35 s
        watchdog). Holding for at most ``core_busy_stop_s`` keeps a slow
        reactive step from looking like a dead driver, and stopping after it
        keeps a *hung* one from looking alive.
        """
        if self._busy_since is None:
            self._busy_since = now
        if (now - self._busy_since) * 1e-9 > self.core_busy_stop:
            self.counters["core_busy_stop"] += 1
            self.get_logger().warn("driving core busy: commanding zero", throttle_duration_sec=5.0)
            self._publish_cmd(0.0, 0.0)
            return
        self.counters["core_busy_hold"] += 1
        self._publish_cmd(*self._last_cmd)

    def _publish_cmd(self, v: float, steer: float, *, record_fallback: bool = True) -> None:
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(v * math.tan(steer) / self.cfg.vehicle.wheelbase_m)
        publish_start = time.monotonic()
        self.cmd_pub.publish(msg)
        publish_ms = (time.monotonic() - publish_start) * 1e3
        if publish_ms >= 100.0:
            self.get_logger().warning(f"command publish blocked: {publish_ms:.3f} ms, v={v:.3f} m/s")
        self._last_cmd = (float(v), float(steer))
        now = self._now()
        if record_fallback:
            self._fallback_commands.append((now * 1e-9, math.tan(steer) / self.cfg.vehicle.wheelbase_m if v else 0.0, float(v)))
        if self._last_cmd_ns is not None:
            self._cmd_gap_max_ms = max(self._cmd_gap_max_ms, (now - self._last_cmd_ns) * 1e-6)
        self._last_cmd_ns = now

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
        direct = plan.diag.get("direct_tracking_allowed", True)
        published = plan.path if direct or plan.route is None else plan.route
        self._publish_path(self.line_pub, published.xy, self.frames["map"], closed=True)
        n = plan.ref.normal
        for pub, bound in zip(self.cor_pub, (plan.cor.hi, plan.cor.lo)):
            self._publish_path(pub, plan.ref.xy + bound[:, None] * n, self.frames["map"], closed=True)
        if direct:
            self.get_logger().info(f"race line ready: {plan.path.length:.1f} m, predicted lap {plan.t_lap_pred:.1f} s, "
                                   f"map clearance >= {100 * plan.diag['min_map_clearance_m']:.0f} cm (from the {self.core.events['planning']['lap_path_source']})")
        else:
            self.get_logger().info("SLAM route ready; live LiDAR local planning retained: forward candidate rejected for "
                                   + ", ".join(plan.diag.get("direct_tracking_rejection", [])))
        if self.out_dir is not None:
            grid, loop, side = self.core.plan_inputs
            save_plan(plan, grid, loop, side, self.out_dir, {"planning": self.core.events.get("planning"), "start_map": self.core.events.get("start_map")})
        self._write_events()

    def _write_events(self) -> None:
        if self.events_path is None:
            return
        ev = {k: v for k, v in self.core.events.items() if k != "config"}
        ev["scan_age_stops"] = getattr(self, "scan_age_stops", 0)  # control ticks stopped by max_scan_age_s
        ev["mission_time_stop"] = self._mission_time_stop
        ev["counters"] = dict(self.counters)
        ev["cmd_gap_max_ms"] = round(self._cmd_gap_max_ms, 2)
        ev["core_hold_ms"] = self.hold_report()
        ev["adaptation"] = self.core.adaptation()
        ev["config"] = self.core.events.get("config")
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.write_text(json.dumps(ev, indent=1, default=float), encoding="utf-8")

    def _publish_status(self) -> None:
        now = self._now()
        pose = self._pose_now(now)
        with self._core("status"):
            status = self.core.status()
        self.status_pub.publish(json_msg({
            **status,
            "scan_age_stops": self.scan_age_stops,
            "mission_time_stop": self._mission_time_stop,
            "counters": dict(self.counters),
            "cmd_gap_max_ms": round(self._cmd_gap_max_ms, 2),
            "core_hold_ms": self.hold_report(),
            "armed": bool(self.armed) and not self._arm_reason(now),
            "arm_age_s": None if self.arm_ns is None else round((now - self.arm_ns) * 1e-9, 2),
            "estimator_usable": self.estimator_ok,
            "predicted_pose_age_s": None if pose is None else pose[2],
            "estimator_reason": self.estimator_reason,
        }))

    def destroy_node(self) -> None:
        self._write_events()
        if self.log is not None:
            self._log_file.close()
        if self.inputs is not None:
            self._inputs_file.close()
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RaceDriverNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RCLError:
        # The container's cleanup trap stops every node with SIGTERM, and the
        # context is torn down under the executor while it is waiting. That is
        # a normal stop, not a failure, and the run's events still have to be
        # written below -- a run whose race_events.json is missing cannot be
        # graded afterwards.
        pass
    finally:
        try:
            executor.remove_node(node)
            node.destroy_node()
        except RCLError:
            node._write_events()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
