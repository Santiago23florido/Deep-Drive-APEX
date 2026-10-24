#!/usr/bin/env python3
"""Grade the car's race stack before it is allowed to move.

The companion of ``lidar_bringup_check.py``, one level up: that one grades the
LiDAR against what the odometry network was trained for, this one grades the
whole graph -- estimator, SLAM, driver, actuation bridge -- against what the
car needs in order to drive itself.

It is the gate of ladder steps D1 (live sensors, driver publishing to a dry
topic) and D2 (recorded sensors replayed into the same graph). Everything it
checks is measured from the outside, on the topics, so it grades what is
actually running and not what the configuration says.

The three measurements that decide a go / no-go, and why:

* **the command gap.** The bridge cuts the car to neutral when it has not seen
  a Twist for ``command_timeout_s`` (0.35 s). The driver publishes at 50 Hz,
  but a slow reactive step or a planner burst can stall it. The p99.9 gap must
  stay far below the timeout.
* **the estimator's latency and health.** The pose the controller uses is
  extrapolated from the last corrected estimate; ``max_scan_age_s`` (0.6 s)
  stops the car when corrections dry up.
* **CPU and clock.** A Pi that drops to 1.7 GHz or hits an under-voltage event
  loses a third of the budget the whole architecture was sized against.

Inside the container:

    python3 /work/ros2_ws/scripts/race_bringup_check.py --duration 180 --json /work/ros2_ws/apex_race/d1.json
    python3 /work/ros2_ws/scripts/race_bringup_check.py --duration 180 --cmd-topic /apex/race_driver/cmd_vel_dry --expect-phase explore
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

MAP_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
SENSOR_QOS = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE)

# What each stream must deliver. (topic, type, qos, nominal Hz, tolerance)
STREAMS = [
    ("/apex/imu/data_raw", Imu, SENSOR_QOS, 100.0, 0.10),
    ("/lidar/scan_raw", LaserScan, SENSOR_QOS, 10.0, 0.05),
    ("/apex/odometry/learned", Odometry, SENSOR_QOS, 10.0, 0.10),
    ("/apex/odometry/learned_predicted", Odometry, SENSOR_QOS, 100.0, 0.15),
    ("/apex/learned_odometry/scan_deskewed", LaserScan, SENSOR_QOS, 10.0, 0.10),
]

# The processes of the race stack, by a distinctive fragment of their cmdline.
PROCESSES = {
    "lidar": "rplidar_publisher_node",
    "imu": "nano_accel_serial_node",
    "estimator": "learned_odometry_node",
    "slam": "slam_toolbox",
    "driver": "race_driver_node",
    "bridge": "cmd_vel_to_apex_actuation_node",
}

CMD_GAP_MAX_MS = 100.0      # 3.5x below the bridge's 0.35 s watchdog
LATENCY_P95_MAX_MS = 250.0  # stamp -> estimate; max_scan_age_s is 600 ms
COMPUTE_P95_MAX_MS = 40.0   # per scan, at 10 Hz on one core
CORE_MAX = 0.85             # no single core above this
TOTAL_MAX = 3.0             # of 4 cores
TEMP_MAX_C = 75.0


class Row:
    def __init__(self, name: str, ok: bool | None, detail: str) -> None:
        self.name, self.ok, self.detail = name, ok, detail

    def line(self) -> str:
        tag = "INFO" if self.ok is None else ("PASS" if self.ok else "FAIL")
        return f"[{tag}] {self.name:38s} {self.detail}"


class RaceCheck(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("race_bringup_check")
        self.args = args
        self.stamps: dict[str, list[float]] = defaultdict(list)
        self.cmd_times: list[float] = []
        self.last_status: dict[str, dict] = {}
        self.map_seen = 0
        for topic, msg_type, qos, _, _ in STREAMS:
            self.create_subscription(msg_type, topic, self._counter(topic), qos)
        self.create_subscription(OccupancyGrid, "/map", self._on_map, MAP_QOS)
        for topic in ("/apex/learned_odometry/status", "/apex/race_driver/status", "/apex/vehicle/drive_bridge_status"):
            self.create_subscription(String, topic, self._on_json(topic), 10)
        self.create_subscription(Twist, args.cmd_topic, self._on_cmd, 20)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.odom_first: tuple[float, float] | None = None
        self.odom_last: tuple[float, float] | None = None
        self.predicted_speeds: list[float] = []

    def sample_odom(self) -> None:
        """Where the odometry thinks the car is, in its own frame."""
        try:
            tf = self.tf.lookup_transform("odom_learned", "base_link", Time())
        except TransformException:
            return
        xy = (tf.transform.translation.x, tf.transform.translation.y)
        if self.odom_first is None:
            self.odom_first = xy
        self.odom_last = xy

    def _counter(self, topic: str):
        def cb(_msg) -> None:  # noqa: ANN001
            self.stamps[topic].append(time.monotonic())
            if topic == "/apex/odometry/learned_predicted":
                self.predicted_speeds.append(math.hypot(_msg.twist.twist.linear.x, _msg.twist.twist.linear.y))
        return cb

    def _on_map(self, _msg: OccupancyGrid) -> None:
        self.map_seen += 1
        self.stamps["/map"].append(time.monotonic())

    def _on_json(self, topic: str):
        def cb(msg: String) -> None:
            try:
                self.last_status[topic] = json.loads(msg.data)
            except ValueError:
                return
            self.stamps[topic].append(time.monotonic())
        return cb

    def _on_cmd(self, _msg: Twist) -> None:
        self.cmd_times.append(time.monotonic())

    def reset_observations(self) -> None:
        """Start a measurement window after DDS discovery has settled.

        Preserve status reports: producer/bridge historical fault counters
        must not be erased by warming up this observer.
        """
        self.stamps.clear()
        self.cmd_times.clear()
        self.predicted_speeds.clear()
        self.map_seen = 0
        self.odom_first = self.odom_last = None

    # -------------------------------------------------------------- the grading
    def rows(self) -> list[Row]:
        out: list[Row] = []
        out += self._stream_rows()
        out += self._estimator_rows()
        out += self._slam_rows()
        out += self._driver_rows()
        out += self._command_rows()
        out += self._bridge_rows()
        out += self._drift_rows()
        out += self._platform_rows()
        return out

    def _drift_rows(self) -> list[Row]:
        """How far the odometry walked while the car did not.

        A stationary car still feeds the estimator scans and IMU samples, and a
        standstill longer than anything in the training set is where this model
        drifts: measured on this car, 3 cm over the first minute and 70 cm by
        the second. That decides how long the stack may idle before a run, so
        it is measured rather than remembered.
        """
        if self.odom_first is None or self.odom_last is None:
            return [Row("odometry drift", None, "no odom_learned -> base_link transform")]
        dx = self.odom_last[0] - self.odom_first[0]
        dy = self.odom_last[1] - self.odom_first[1]
        drift = math.hypot(dx, dy)
        rate = 60.0 * drift / max(self.args.duration, 1e-6)
        detail = f"{drift:.3f} m over the check ({rate:.2f} m/min), dx {dx:+.3f} dy {dy:+.3f}"
        if not self.args.stationary:
            return [Row("odometry travel", None, detail + " -- pass --stationary to grade it")]
        rows = [Row("odometry drift at rest", drift <= self.args.drift_max_m,
                    detail + f"; allowed {self.args.drift_max_m:.2f} m")]
        if self.predicted_speeds:
            speed = float(np.percentile(self.predicted_speeds, 99.9))
            rows.append(Row("predicted speed at rest", speed < 0.01,
                            f"p99.9 {speed:.4f} m/s < 0.01 m/s; max {max(self.predicted_speeds):.4f} m/s"))
        else:
            rows.append(Row("predicted speed at rest", False, "no predicted odometry"))
        return rows

    def _stream_rows(self) -> list[Row]:
        rows = []
        for topic, _, _, nominal, tol in STREAMS:
            ts = self.stamps.get(topic, [])
            if len(ts) < 3:
                rows.append(Row(f"stream {topic}", False, f"{len(ts)} messages"))
                continue
            d = np.diff(np.asarray(ts))
            rate = 1.0 / float(np.median(d))
            ok = abs(rate / nominal - 1.0) <= tol
            rows.append(Row(f"stream {topic}", ok,
                            f"{rate:.2f} Hz (nominal {nominal:.0f}, tol {100 * tol:.0f} %), "
                            f"{len(ts)} msgs, worst gap {1e3 * float(d.max()):.0f} ms"))
        return rows

    def _estimator_rows(self) -> list[Row]:
        st = self.last_status.get("/apex/learned_odometry/status")
        if not st:
            return [Row("estimator status", False, "no /apex/learned_odometry/status")]
        rows = [Row("estimator ready", bool(st.get("ready")), f"ready={st.get('ready')} scans={st.get('scans')}")]
        for key, limit, label in (("latency_ms_p95", LATENCY_P95_MAX_MS, "stamp -> estimate"),
                                  ("compute_ms_p95", COMPUTE_P95_MAX_MS, "per scan")):
            value = st.get(key)
            if value is None:
                rows.append(Row(f"estimator {key}", None, "not reported"))
                continue
            rows.append(Row(f"estimator {key}", float(value) <= limit,
                            f"{float(value):.0f} ms <= {limit:.0f} ms ({label})"))
        health = st.get("health") or {}
        rows.append(Row("estimator health rates", None,
                        f"IMU {health.get('imu_rate_hz', 0):.1f} Hz (jitter {health.get('imu_period_jitter_ms', 0):.1f} ms), "
                        f"LiDAR {health.get('lidar_rate_hz', 0):.2f} Hz (jitter {health.get('lidar_period_jitter_ms', 0):.1f} ms), "
                        f"valid bins {100 * health.get('valid_bins_fraction', 0):.0f} %"))
        if "estimator_usable" in (self.last_status.get("/apex/race_driver/status") or {}):
            drv = self.last_status["/apex/race_driver/status"]
            rows.append(Row("estimator usable for identification", bool(drv.get("estimator_usable")),
                            drv.get("estimator_reason") or "rates within tolerance"))
        pending = st.get("pending")
        if pending is not None:
            rows.append(Row("estimator backlog", float(pending) <= 1.0, f"{pending} scans queued"))
        return rows

    def _slam_rows(self) -> list[Row]:
        rows = [Row("slam /map published", self.map_seen > 0, f"{self.map_seen} maps")]
        chain = [("map", "odom_learned"), ("odom_learned", "base_link"), ("base_link", "laser")]
        for parent, child in chain:
            try:
                tf = self.tf.lookup_transform(parent, child, Time())
                t = tf.transform.translation
                rows.append(Row(f"tf {parent} -> {child}", True, f"({t.x:+.3f}, {t.y:+.3f}, {t.z:+.3f})"))
            except TransformException as exc:
                rows.append(Row(f"tf {parent} -> {child}", False, str(exc)[:70]))
        return rows

    def _driver_rows(self) -> list[Row]:
        st = self.last_status.get("/apex/race_driver/status")
        if not st:
            return [Row("driver status", False, "no /apex/race_driver/status")]
        rows = []
        phase = str(st.get("phase", "?"))
        if self.args.expect_phase:
            rows.append(Row("driver phase", phase == self.args.expect_phase,
                            f"{phase} (expected {self.args.expect_phase})"))
        else:
            rows.append(Row("driver phase", None, phase))
        if self.args.stationary:
            laps = int(st.get("lap_own", 0))
            rows.append(Row("driver laps at rest", laps == 0, f"{laps} laps, travel {st.get('travel_m', 0):.3f} m"))
        if "sign_fault" in st:
            rows.append(Row("driver steering sign", not bool(st["sign_fault"]),
                            f"sign_fault={st['sign_fault']}"))
        counters = st.get("counters") or {}
        rows.append(Row("driver scan_age stops", int(st.get("scan_age_stops", 0)) == 0,
                        f"{st.get('scan_age_stops', 0)} control ticks stopped by max_scan_age_s"))
        rows.append(Row("driver core never hung", int(counters.get("core_busy_stop", 0)) == 0,
                        f"core_busy_stop={counters.get('core_busy_stop', 0)} hold={counters.get('core_busy_hold', 0)} "
                        f"stale scans={counters.get('scan_stale', 0)}"))
        rows.append(Row("driver armed", None, f"armed={st.get('armed')} age={st.get('arm_age_s')} s"))
        if "react_ms_p95" in st:
            rows.append(Row("driver reactive step p95", float(st["react_ms_p95"]) < 120.0, f"{st['react_ms_p95']} ms"))
        return rows

    def _command_rows(self) -> list[Row]:
        ts = np.asarray(self.cmd_times)
        if len(ts) < 10:
            return [Row(f"command stream {self.args.cmd_topic}", False, f"{len(ts)} messages")]
        d = 1e3 * np.diff(ts)
        p999 = float(np.percentile(d, 99.9))
        rows = [Row(f"command rate {self.args.cmd_topic}", abs((1e3 / float(np.median(d))) / 50.0 - 1.0) < 0.2,
                    f"{1e3 / float(np.median(d)):.1f} Hz, {len(ts)} msgs"),
                Row("command gap p99.9", p999 < CMD_GAP_MAX_MS,
                    f"{p999:.0f} ms < {CMD_GAP_MAX_MS:.0f} ms (bridge watchdog 350 ms); max {float(d.max()):.0f} ms")]
        return rows

    def _bridge_rows(self) -> list[Row]:
        st = self.last_status.get("/apex/vehicle/drive_bridge_status")
        if not st:
            return [Row("drive bridge", None, "not running (dry run)")]
        rows = [Row("bridge profile", st.get("actuation_profile") == self.args.expect_profile,
                    f"{st.get('actuation_profile')} (expected {self.args.expect_profile})")]
        smap = st.get("speed_map")
        if smap:
            rows.append(Row("bridge speed map", None,
                            f"{smap['esc_zero_pct']:.0f} % + {smap['pct_per_mps']:.0f} %/(m/s), ceiling "
                            f"{smap['max_pct']:.0f} % -> {smap['full_scale_mps']:.2f} m/s full scale"))
        authority = st.get("steering_authority_deg")
        limit = st.get("max_steering_deg")
        if authority is not None and limit is not None:
            servo = st.get("steering_servo_state", {})
            left, right = (float(st.get(f"max_steering_{side}_deg", limit)) for side in ("left", "right"))
            al, ar = (float(servo.get(f"authority_{side}_deg", authority)) for side in ("left", "right"))
            rows.append(Row("steering limit within authority", left <= al + 1e-6 and right <= ar + 1e-6,
                            f"left/right clamps {left:.1f}/{right:.1f} deg <= servo {al:.2f}/{ar:.2f} deg"))
        rows.append(Row("bridge command gap", float(st.get("cmd_gap_ms_max", 0.0)) < CMD_GAP_MAX_MS,
                        f"worst {float(st.get('cmd_gap_ms_max', 0.0)):.0f} ms as seen by the bridge"))
        return rows

    def _platform_rows(self) -> list[Row]:
        rows: list[Row] = []
        throttled = _vcgencmd("get_throttled")
        if throttled:
            bits = int(throttled.split("=")[-1], 16)
            rows.append(Row("no live under-voltage", not (bits & 0x1),
                            f"get_throttled={throttled.split('=')[-1]} "
                            f"(live: under-volt {bool(bits & 0x1)}, throttled {bool(bits & 0x4)}; "
                            f"since boot: under-volt {bool(bits & 0x10000)})"))
        temp = _vcgencmd("measure_temp")
        if temp:
            value = float(temp.split("=")[-1].rstrip("'C"))
            rows.append(Row("temperature", value < TEMP_MAX_C, f"{value:.1f} C < {TEMP_MAX_C:.0f} C"))
        governors = sorted({p.read_text().strip() for p in Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_governor")}) \
            if Path("/sys/devices/system/cpu/cpufreq").is_dir() else []
        if governors:
            # Informational, not a gate: forcing `performance` on this car put
            # the 5 V rail into under-voltage with the LiDAR motor turning, so
            # the stack leaves the governor alone and the Pi 5 boosts by itself.
            rows.append(Row("cpu governor", None, ", ".join(governors)))
        return rows


def _vcgencmd(arg: str) -> str:
    for binary in ("/usr/bin/vcgencmd", "vcgencmd"):
        try:
            return subprocess.run([binary, arg], capture_output=True, text=True, timeout=2).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
    return ""


def _proc_cpu() -> dict[int, tuple[str, float]]:
    """{pid: (role, utime+stime in seconds)} for the race-stack processes."""
    out: dict[int, tuple[str, float]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().decode(errors="replace")
            stat = (entry / "stat").read_text().rsplit(") ", 1)[-1].split()
        except OSError:
            continue
        role = next((r for r, frag in PROCESSES.items() if frag in cmdline), None)
        if role is None:
            continue
        ticks = float(stat[11]) + float(stat[12])  # utime, stime after the comm field
        out[int(entry.name)] = (role, ticks / os.sysconf("SC_CLK_TCK"))
    return out


def cpu_rows(before: dict[int, tuple[str, float]], after: dict[int, tuple[str, float]], elapsed: float) -> list[Row]:
    per_role: dict[str, float] = defaultdict(float)
    for pid, (role, t1) in after.items():
        t0 = before.get(pid, (role, t1))[1]
        per_role[role] += max(0.0, t1 - t0) / max(elapsed, 1e-6)
    rows = []
    for role in PROCESSES:
        if role in per_role:
            rows.append(Row(f"cpu {role}", per_role[role] <= CORE_MAX,
                            f"{100 * per_role[role]:.0f} % of one core"))
    total = sum(per_role.values())
    rows.append(Row("cpu total", total <= TOTAL_MAX, f"{100 * total:.0f} % of {100 * TOTAL_MAX:.0f} % budget "
                                                     f"({os.cpu_count()} cores present)"))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--settle-s", type=float, default=2.0,
                    help="spin for DDS discovery before the graded window; historical fault counters remain graded")
    ap.add_argument("--cmd-topic", default="/apex/cmd_vel_track")
    ap.add_argument("--expect-phase", default="", help="fail unless the driver reaches this phase (e.g. explore, race)")
    ap.add_argument("--expect-profile", default="affine", help="actuation profile the bridge must report")
    ap.add_argument("--stationary", action="store_true",
                    help="the car is standing still: grade the odometry's phantom travel as a gate")
    ap.add_argument("--drift-max-m", type=float, default=0.10,
                    help="phantom travel allowed over the whole check when --stationary (default 0.10 m)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    if not math.isfinite(args.settle_s) or args.settle_s < 0.0:
        ap.error("--settle-s must be finite and nonnegative")

    rclpy.init()
    node = RaceCheck(args)
    before, t0 = _proc_cpu(), time.monotonic()
    try:
        settle_until = time.monotonic() + args.settle_s
        while rclpy.ok() and time.monotonic() < settle_until:
            rclpy.spin_once(node, timeout_sec=0.1)
        node.reset_observations()
        before, t0 = _proc_cpu(), time.monotonic()
        end = t0 + args.duration
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
            node.sample_odom()
    except KeyboardInterrupt:
        pass
    elapsed = time.monotonic() - t0
    rows = node.rows() + cpu_rows(before, _proc_cpu(), elapsed)
    node.destroy_node()
    rclpy.shutdown()

    print(f"\nrace bring-up check: {elapsed:.0f} s on {args.cmd_topic} (after {args.settle_s:g} s DDS settling)\n" + "-" * 78)
    for row in rows:
        print(row.line())
    failed = [r.name for r in rows if r.ok is False]
    print("-" * 78)
    print(f"{'FAIL' if failed else 'PASS'}: {len(failed)} of {sum(1 for r in rows if r.ok is not None)} graded checks failed"
          + (f" -> {', '.join(failed)}" if failed else ""))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "duration_s": round(elapsed, 1), "cmd_topic": args.cmd_topic, "passed": not failed,
            "settle_s": args.settle_s,
            "checks": [{"name": r.name, "ok": r.ok, "detail": r.detail} for r in rows],
        }, indent=1))
        print(f"written: {args.json}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
