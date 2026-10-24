#!/usr/bin/env python3
"""Replay recorded real2sim runs through slam_toolbox offline, to tune it.

A Gazebo run records what the car received: the raw LiDAR revolutions
(``lidar_scans.csv``), the raw IMU (``imu_raw.csv``), with the sensor errors
of the profile, and the truth (``truth_track.csv``). This script rebuilds
what slam_toolbox was given in the run and feeds it again, with the run's
SLAM configuration or a modified one, so a SLAM change is screened on every
recorded run in minutes instead of one Gazebo run per seed. The truth is
only used to score.

  prepare   (learning/.venv python) LiveOdometry, the node's estimator (same
            checkpoint and sensor profile), replayed on the recorded sensors.
            Written to ``<run>/slam_replay/input.npz``: the TF
            ``odom_learned -> base_link`` the node published in the run (its
            ``estimator.csv``; the offline replay differs by ~0.1 mm per step,
            IMU samples at the revolution boundary, which adds up to cm over a
            run) and the de-skewed revolutions, rebuilt offline.
  run       (ROS sourced, no conda) ``sync_slam_toolbox_node`` fed from that
            file, one process and one ROS domain per run, in parallel, on
            CycloneDDS (Fast DDS takes ~40 s in WSL to deliver the first
            lifecycle request). /clock follows the scans. Output
            ``<run>/slam_replay/<tag>.npz``: the SLAM ``/pose`` (base_link in
            the map frame at every accepted map update), observed incoming
            scans, map->odom transforms and map-content hashes. At rest the
            motion gate deliberately skips new map updates; /pose count is
            not a received-scan count.
            ``--set key=value`` overrides slam_toolbox parameters;
            ``--perturb yaw_rw_deg_rtm=, scale=, seed=`` adds errors to the
            odometry (a worse car, or with a tiny ``yaw_rw`` another
            realization of the same run: the SLAM is chaotic, one replay per
            run is one sample).
  compare   metrics of tags over the runs.

The replay is open loop: the car drives the recorded path whatever the SLAM
does. That is the whole of lap 1 (the reactive driver works in the odometry
frame) and the input of the race plan; the race laps would follow another
line with another map, so the Gazebo confirmation still decides.

The sync node processes every scan it receives, in order, so a replay is
repeatable (50 of 51 runs bit-identical; Ceres threads) as long as no scan is
dropped: slam_toolbox subscribes with the sensor-data QoS (best effort, depth
5), hence ``--pace-ms`` 20 with 8 jobs (4 ms with 12 jobs dropped scans). The
car runs the async node, which processes the same scans while it keeps up
(one every 0.2 s at most, a few ms each).

Metrics per run, against the truth (base_link; the map frame placed at the
true start, ``--anchor true``, or where the car believes it started):

  open-loop error  SLAM error at the last pose before lap 1 closes its loop
                   (the first correction >= 15 cm after 75 % of lap 1), and the
                   odometry alone at the same stamp (map -> odom held at its
                   first value). The SLAM should not be worse.
  closure step     the correction applied when the loop closes.
  race p95         SLAM error p95 from 8 s after the lap-1 crossing (laps 2-3):
                   the loop did close and the map holds.
  ATE              RMSE over the whole run.

  learning/.venv/bin/python tools/analysis/replay_slam.py prepare data/real2sim/fin_s* --jobs 12
  python3 tools/analysis/replay_slam.py run data/real2sim/fin_s* --tag base --jobs 8 [--set key=value ...]
  python3 tools/analysis/replay_slam.py compare data/real2sim/fin_s* --tags base,tuned [--list-failures]

Debugging: ``REPLAY_DEBUG=1`` prints the stages of ``_one``;
``REPLAY_MAX_SCANS=n`` stops after n scans.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

SIM = Path(__file__).resolve().parents[2]
PKG = SIM / "ros2_ws" / "src" / "apex_fusion_research"
SLAM_YAML = PKG / "config" / "slam" / "slam_toolbox_learned.yaml"
SENSORS_YAML = SIM / "tools" / "pose_dataset" / "config" / "sensors.yaml"
LOOP_STEP_M = 0.15  # a correction at least this large late in lap 1 is the loop closure
RACE_AFTER_S = 8.0  # race-phase window starts this long after the lap-1 crossing (overlap + map)
WORSE_M = 0.10  # a run fails lap 1 when the SLAM is this much worse than the odometry alone before the loop closes
RACE_BAD_M = 0.35  # a run fails the race phase when its SLAM error p95 exceeds this (blind margin 0.20 + belief 0.15)


def _csv(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)


def se2(a, b) -> tuple[float, float, float]:  # noqa: ANN001
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2])


def se2_inv(a) -> tuple[float, float, float]:  # noqa: ANN001
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2])


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# ------------------------------------------------------------------ prepare
def prepare_run(run: str) -> dict:
    """Replay the node's estimator on the recorded sensors of one run (``learning/.venv``)."""
    sys.path.insert(0, str(SIM / "learning" / "lidar_imu_pose"))
    sys.path.insert(0, str(SIM / "tools"))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch  # noqa: PLC0415

    torch.set_num_threads(1)
    from live_odometry import LidarCalibration, LiveOdometry  # noqa: PLC0415
    from pose_dataset.sensors import load_sensor_profiles  # noqa: PLC0415
    from sqlite_windows import imu_group_delay_s  # noqa: PLC0415

    rd = Path(run)
    rc = json.loads((rd / "run_config.json").read_text(encoding="utf-8"))
    prof = load_sensor_profiles(SENSORS_YAML)[rc["sensor_profile"]]
    cal = LidarCalibration.from_profile(prof["lidar"])
    live = LiveOdometry(rc["checkpoint"], cal, device="cpu", imu_wait_s=10.0, imu_rate_hz=float(prof["imu"]["rate_hz"]),
                        imu_group_delay_s=imu_group_delay_s(prof["imu"]), icp_backend="polar")
    sc = np.genfromtxt(rd / "lidar_scans.csv", delimiter=",", skip_header=1, filling_values=np.nan)
    stamps, scan_time = sc[:, 0].astype(np.int64), sc[:, 1]
    ranges = np.where(np.isfinite(sc[:, 2:]), sc[:, 2:], np.inf).astype(np.float32)
    imu = np.loadtxt(rd / "imu_raw.csv", delimiter=",", skiprows=1)
    imu_t = (imu[:, 0].astype(np.int64) * 1_000_000_000 + imu[:, 1].astype(np.int64))
    # Exact-end replay (latency_profile.replay_run): every IMU sample up to the end of revolution k, then scan k.
    out_t, out_pose, out_scan = [], [], []
    j, lag = 0, live.imu_delay_ns
    for k in range(len(stamps)):
        end = int(stamps[k]) + round(scan_time[k] * 1e9)
        while j < len(imu_t) and (j == 0 or imu_t[j - 1] - lag <= end):
            live.add_imu(int(imu_t[j]), imu[j, 5:8], imu[j, 2:5])
            j += 1
        live.add_scan(int(stamps[k]), ranges[k], float(scan_time[k]), arrival_ns=end)
        for e in live.process(now_ns=end):
            out_t.append(e.stamp_ns)
            out_pose.append(e.pose.copy())
            out_scan.append(e.deskewed_ranges.astype(np.float32))
    t, pose = np.asarray(out_t, dtype=np.int64), np.asarray(out_pose)
    # The TF slam_toolbox received is the live one (estimator.csv): the offline replay differs by
    # ~0.1 mm per step (IMU samples at the revolution boundary), which adds up to cm over a run.
    # The de-skewed revolutions are the offline ones (their velocity differs by far less than 1 mm).
    est = _csv(rd / "estimator.csv")
    common, ia, ib = np.intersect1d(t, est["stamp_ns"].astype(np.int64), return_indices=True)
    live_pose = np.column_stack((est["x"][ib], est["y"][ib], est["yaw"][ib]))
    dpos = np.hypot(pose[ia, 0] - live_pose[:, 0], pose[ia, 1] - live_pose[:, 1])
    lx, ly, lz = (float(v) for v in prof["lidar"]["extrinsic"]["xyz"])
    out = rd / "slam_replay"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / "input.npz", stamp_ns=common, pose=live_pose, pose_offline=pose[ia], deskewed=np.asarray(out_scan)[ia],
                        laser_xyz=np.array([lx, ly, lz]), angle_min=cal.angle_min, angle_increment=cal.angle_increment,
                        range_min=cal.range_min, range_max=cal.range_max, rate_hz=cal.rate_hz)
    dyaw = np.degrees(np.abs([wrap(a - b) for a, b in zip(pose[ia, 2], live_pose[:, 2])]))
    return {"run": rd.name, "scans": len(t), "matched": len(common), "dpos_p99_mm": float(np.percentile(dpos, 99) * 1e3) if len(dpos) else None,
            "dpos_max_mm": float(dpos.max() * 1e3) if len(dpos) else None, "dyaw_max_deg": float(dyaw.max()) if len(dyaw) else None}


def cmd_prepare(args: argparse.Namespace) -> None:
    runs = [r for r in args.runs if (Path(r) / "lidar_scans.csv").exists()]
    with ProcessPoolExecutor(args.jobs) as ex:
        for r in ex.map(prepare_run, runs):
            print(json.dumps(r), flush=True)


# ---------------------------------------------------------------------- run
def slam_params(overrides: dict, range_max: float) -> dict:
    """The run's slam_toolbox parameters (as apex_real2sim.launch.py passes them), then ``overrides``."""
    import yaml  # noqa: PLC0415

    base = yaml.safe_load(SLAM_YAML.read_text(encoding="utf-8"))["/**"]["ros__parameters"]
    return {**base, "max_laser_range": float(range_max), "use_lifecycle_manager": False, "use_sim_time": True,
            "scan_queue_size": 10000,  # replay only: the sync node must not drop a scan the replay publishes ahead of it
            **overrides}


def parse_value(v: str):  # noqa: ANN201
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def replay_one(run: str, tag: str, overrides: dict, domain: int, perturb: dict, pace_s: float = 0.02) -> dict:
    """slam_toolbox (sync) fed with one run's input; writes ``<run>/slam_replay/<tag>.npz`` (``ROS_DOMAIN_ID`` set by the caller)."""
    import rclpy  # noqa: PLC0415
    import yaml  # noqa: PLC0415
    from geometry_msgs.msg import (  # noqa: PLC0415
        PoseWithCovarianceStamped,
        TransformStamped,
    )
    from lifecycle_msgs.srv import ChangeState  # noqa: PLC0415
    from nav_msgs.msg import OccupancyGrid  # noqa: PLC0415
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: PLC0415
    from rosgraph_msgs.msg import Clock  # noqa: PLC0415
    from sensor_msgs.msg import LaserScan  # noqa: PLC0415
    from tf2_msgs.msg import TFMessage  # noqa: PLC0415

    rd = Path(run)
    z = np.load(rd / "slam_replay" / "input.npz")
    stamps, pose, scans = z["stamp_ns"], perturb_odometry(z["pose"], z["stamp_ns"], perturb), z["deskewed"]
    params = slam_params(overrides, float(z["range_max"]))
    tmp = Path(tempfile.mkdtemp(prefix="slam_replay_"))
    (tmp / "p.yaml").write_text(yaml.safe_dump({"slam_toolbox": {"ros__parameters": params}}), encoding="utf-8")
    log = open(tmp / "slam.log", "w")  # noqa: SIM115
    proc = subprocess.Popen(["ros2", "run", "slam_toolbox", "sync_slam_toolbox_node", "--ros-args", "--params-file", str(tmp / "p.yaml"),
                             "-r", "__node:=slam_toolbox"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    rclpy.init()
    node = rclpy.create_node("slam_replay")
    clock_pub = node.create_publisher(Clock, "/clock", 10)
    tf_pub = node.create_publisher(TFMessage, "/tf", 100)
    static_pub = node.create_publisher(TFMessage, "/tf_static", QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    scan_pub = node.create_publisher(LaserScan, str(params["scan_topic"]), 100)
    got: list[tuple[int, float, float, float]] = []
    observed_scans: list[int] = []
    map_odom: list[tuple[int, float, float, float]] = []
    maps: list[dict] = []

    def on_pose(m: PoseWithCovarianceStamped) -> None:
        q = m.pose.pose.orientation
        got.append((m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec, m.pose.pose.position.x, m.pose.pose.position.y,
                    2.0 * math.atan2(q.z, q.w)))

    node.create_subscription(PoseWithCovarianceStamped, "/pose", on_pose, 1000)

    def on_tf(m: TFMessage) -> None:
        for t in m.transforms:
            if t.header.frame_id == params["map_frame"] and t.child_frame_id == params["odom_frame"]:
                q = t.transform.rotation
                map_odom.append((t.header.stamp.sec * 1_000_000_000 + t.header.stamp.nanosec,
                                 t.transform.translation.x, t.transform.translation.y,
                                 math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))))

    def on_map(m: OccupancyGrid) -> None:
        maps.append(dict(stamp_ns=m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec,
                         width=m.info.width, height=m.info.height, resolution=m.info.resolution,
                         origin=[m.info.origin.position.x, m.info.origin.position.y],
                         data_sha256=hashlib.sha256(bytes((v + 1) & 0xFF for v in m.data)).hexdigest()))

    def on_scan(m: LaserScan) -> None:
        observed_scans.append(m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec)

    # /pose counts accepted map updates. At rest the configured motion gate
    # intentionally admits only the first scan; record transport observations
    # and map->odom separately instead of treating that as lost input.
    best_effort = QoSProfile(depth=1000, reliability=ReliabilityPolicy.BEST_EFFORT)
    node.create_subscription(TFMessage, "/tf", on_tf, best_effort)
    node.create_subscription(LaserScan, str(params["scan_topic"]), on_scan, best_effort)
    node.create_subscription(OccupancyGrid, "/map", on_map,
                             QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def stamp(ns: int):  # noqa: ANN202
        from builtin_interfaces.msg import Time  # noqa: PLC0415

        return Time(sec=int(ns // 1_000_000_000), nanosec=int(ns % 1_000_000_000))

    def tf(ns: int, parent: str, child: str, x: float, y: float, yaw: float, zz: float = 0.0) -> TransformStamped:
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = stamp(ns), parent, child
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = float(x), float(y), float(zz)
        t.transform.rotation.z, t.transform.rotation.w = math.sin(0.5 * yaw), math.cos(0.5 * yaw)
        return t

    dbg = (lambda *a: print(f"[{time.monotonic() - T0:7.2f}]", *a, file=sys.stderr, flush=True)) if os.environ.get("REPLAY_DEBUG") else (lambda *a: None)
    T0 = time.monotonic()

    def spin(seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=min(0.01, max(0.0, end - time.monotonic())))

    cpu_s = float("nan")
    try:
        t0 = int(stamps[0]) - 1_000_000_000
        clock_pub.publish(Clock(clock=stamp(t0)))
        lx, ly, lz = (float(v) for v in z["laser_xyz"])
        static_pub.publish(TFMessage(transforms=[tf(0, str(params["base_frame"]), "laser", lx, ly, 0.0, lz)]))
        for srv_name, transition in (("configure", 1), ("activate", 3)):
            cli = node.create_client(ChangeState, "/slam_toolbox/change_state")
            if not cli.wait_for_service(timeout_sec=30.0):
                raise RuntimeError(f"slam_toolbox did not come up ({srv_name})")
            req = ChangeState.Request()
            req.transition.id = transition
            dbg("service up", srv_name)
            fut = cli.call_async(req)
            while not fut.done():
                clock_pub.publish(Clock(clock=stamp(t0)))
                spin(0.05)
        dbg("active")
        # Every publisher matched before the first scan (a scan published before the match is lost).
        end = time.monotonic() + 30.0
        while time.monotonic() < end and not (scan_pub.get_subscription_count() and tf_pub.get_subscription_count()
                                              and clock_pub.get_subscription_count() and node.count_publishers("/pose")):
            spin(0.05)
        spin(0.5)
        n = min(len(stamps), int(os.environ.get("REPLAY_MAX_SCANS", len(stamps))))
        for k in range(n):
            ns = int(stamps[k])
            clock_pub.publish(Clock(clock=stamp(ns + 150_000_000)))  # the node publishes ~150 ms after the stamp (Gazebo latency p50)
            tf_pub.publish(TFMessage(transforms=[tf(ns, str(params["odom_frame"]), str(params["base_frame"]), *pose[k])]))
            s = LaserScan()
            s.header.stamp, s.header.frame_id = stamp(ns), "laser"
            s.angle_min, s.angle_increment = float(z["angle_min"]), float(z["angle_increment"])
            s.angle_max = float(z["angle_min"] + z["angle_increment"] * (scans.shape[1] - 1))
            s.time_increment, s.scan_time = 0.0, float(1.0 / z["rate_hz"])
            s.range_min, s.range_max = float(z["range_min"]), float(z["range_max"])
            s.ranges = scans[k].astype(float).tolist()
            scan_pub.publish(s)
            # slam_toolbox subscribes with the sensor-data QoS (best effort, depth 5): scans published
            # faster than its executor takes them are dropped, and the replay stops being deterministic.
            spin(pace_s)
            if k % 20 == 0:
                dbg("scan", k, "poses", len(got))
        # Drain: until no new pose for 3 s.
        last, quiet = len(got), time.monotonic()
        while time.monotonic() - quiet < 3.0:
            clock_pub.publish(Clock(clock=stamp(int(stamps[-1]) + 200_000_000)))
            spin(0.1)
            if len(got) != last:
                last, quiet = len(got), time.monotonic()
        cpu_s = slam_cpu_s(proc.pid)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        log.close()
    g = np.array(got, dtype=float).reshape(-1, 4)
    mt = np.array(map_odom, dtype=float).reshape(-1, 4)
    np.savez_compressed(rd / "slam_replay" / f"{tag}.npz", stamp_ns=g[:, 0].astype(np.int64), pose=g[:, 1:], odom_stamp_ns=stamps, odom=pose,
                        params=json.dumps(params), perturb=json.dumps(perturb), cpu_s=cpu_s,
                        observed_scan_stamp_ns=np.asarray(observed_scans, dtype=np.int64),
                        map_odom_stamp_ns=mt[:, 0].astype(np.int64), map_odom=mt[:, 1:], maps=json.dumps(maps),
                        duration_s=(int(stamps[-1]) - int(stamps[0])) * 1e-9)
    shutil.copy(tmp / "slam.log", rd / "slam_replay" / f"{tag}.log")
    shutil.rmtree(tmp, ignore_errors=True)
    return {"run": rd.name, "tag": tag, "poses": len(g), "observed_scans": len(observed_scans),
            "map_odom_transforms": len(mt), "maps": len(maps), "cpu_s": round(cpu_s, 2)}


def slam_cpu_s(pgid: int) -> float:
    """CPU time [s] (user + system, all threads) the slam_toolbox process has used so far (run start included).

    Its cost on the car: the same work, per processed scan and per second of run, on a slower core
    (``learning/lidar_imu_pose/pi4_budget.py``). The replay's pacing does not change it (CPU, not wall time)."""
    total = 0.0
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            txt = stat.read_text()
        except OSError:
            continue
        comm = txt[txt.index("(") + 1:txt.rindex(")")]
        fields = txt[txt.rindex(")") + 2:].split()
        if int(fields[2]) == pgid and comm.startswith("sync_slam"):  # fields[2]: process group; 11/12: utime, stime
            total += (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    return total


def perturb_odometry(pose: np.ndarray, stamps: np.ndarray, p: dict) -> np.ndarray:
    """A worse odometry than the simulated one (robustness): per-step heading and length errors re-integrated.

    ``yaw_rw_deg_rtm``  heading random walk [deg/sqrt(m)] (the odometry's heading drift, as a real gyro
                        bias and ICP would add), ``scale`` length error [-], ``seed``."""
    if not p:
        return pose
    rng = np.random.default_rng(int(p.get("seed", 0)))
    out = [tuple(pose[0])]
    for k in range(1, len(pose)):
        c, s = math.cos(pose[k - 1, 2]), math.sin(pose[k - 1, 2])
        dx, dy = pose[k, 0] - pose[k - 1, 0], pose[k, 1] - pose[k - 1, 1]
        d = (c * dx + s * dy, -s * dx + c * dy, wrap(pose[k, 2] - pose[k - 1, 2]))
        ds = math.hypot(d[0], d[1])
        dyaw = d[2] + math.radians(float(p.get("yaw_rw_deg_rtm", 0.0))) * math.sqrt(ds) * rng.standard_normal()
        scale = 1.0 + float(p.get("scale", 0.0))
        out.append(se2(out[-1], (scale * d[0], scale * d[1], dyaw)))
    return np.asarray(out)


def cmd_run(args: argparse.Namespace) -> None:
    overrides = {k: parse_value(v) for k, v in (s.split("=", 1) for s in args.set)}
    perturb = {k: parse_value(v) for k, v in (s.split("=", 1) for s in args.perturb)}
    runs = [r for r in args.runs if (Path(r) / "slam_replay" / "input.npz").exists()]
    env = {k: v for k, v in os.environ.items()}
    env["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    # Fast DDS takes ~40 s in WSL to deliver the first lifecycle request; Cyclone ~0.4 s (the SLAM does not depend on it).
    env["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"
    procs: list[tuple[subprocess.Popen, str]] = []
    pending = list(enumerate(runs))
    t0 = time.time()
    while pending or procs:
        while pending and len(procs) < args.jobs:
            i, r = pending.pop(0)
            e = {**env, "ROS_DOMAIN_ID": str(100 + (i % 100))}
            cmd = [sys.executable, __file__, "_one", r, "--tag", args.tag, "--config", json.dumps(overrides), "--perturb-json", json.dumps(perturb),
                   "--pace-ms", str(args.pace_ms)]
            procs.append((subprocess.Popen(cmd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True), r))
        time.sleep(0.2)
        for pr in [p for p in procs if p[0].poll() is not None]:
            procs.remove(pr)
            out = pr[0].stdout.read().strip().splitlines()
            print(out[-1] if out else f"{pr[1]}: no output", flush=True)
    print(f"{len(runs)} runs in {time.time() - t0:.0f} s")


def cmd_one(args: argparse.Namespace) -> None:
    print(json.dumps(replay_one(args.run, args.tag, json.loads(args.config), int(os.environ.get("ROS_DOMAIN_ID", "0")),
                                json.loads(args.perturb_json), args.pace_ms * 1e-3)), flush=True)


# ------------------------------------------------------------------ metrics
def t_world_map(rd: Path, anchor: str) -> tuple[float, float, float]:
    """Where the map frame (the car's start) lies in the world. ``nominal``: where the car believes it
    started (``slam/alignment.json``: the nominal start line, as the evaluation of the run); ``true``:
    where it really started (``spawn_true``), so the error is the SLAM's own drift, without the start
    placement error (up to ~1 deg, i.e. 17 cm at 10 m) that no SLAM setting can remove."""
    if anchor == "true":
        return tuple(json.loads((rd / "run_config.json").read_text(encoding="utf-8"))["spawn_true"])
    return tuple(json.loads((rd / "slam" / "alignment.json").read_text(encoding="utf-8"))["learned"]["T_world_map"])


def run_metrics(run: str, tag: str, anchor: str = "true") -> dict | None:
    rd = Path(run)
    f = rd / "slam_replay" / f"{tag}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    ev = json.loads((rd / "evaluation.json").read_text(encoding="utf-8"))
    cr = [c for c in (ev.get("race") or {}).get("crossings") or [] if c["lap"] == 1]
    tr = _csv(rd / "truth_track.csv")
    tt = tr["t_ns"].astype(np.int64)
    twm = t_world_map(rd, anchor)
    st, sp = z["stamp_ns"], z["pose"]
    if len(st) == 0:
        return {"run": rd.name, "poses": 0}
    ot, op = z["odom_stamp_ns"], z["odom"]

    def truth(ns: int) -> tuple[float, float]:
        j = min(int(np.searchsorted(tt, ns)), len(tt) - 1)
        return float(tr["x"][j]), float(tr["y"][j])

    def odom_at(ns: int) -> tuple[float, float, float]:
        k = min(int(np.searchsorted(ot, ns)), len(ot) - 1)
        return tuple(op[k])

    # map -> odom held at its first value = the odometry alone, in the map frame
    m0 = se2(tuple(sp[0]), se2_inv(odom_at(int(st[0]))))
    err, err_odom, corr = [], [], [0.0]
    prev_mo = m0
    for k in range(len(st)):
        o = odom_at(int(st[k]))
        mo = se2(tuple(sp[k]), se2_inv(o))  # map -> odom after scan k
        tx, ty = truth(int(st[k]))
        w = se2(twm, tuple(sp[k]))
        wo = se2(twm, se2(m0, o))
        err.append(math.hypot(w[0] - tx, w[1] - ty))
        err_odom.append(math.hypot(wo[0] - tx, wo[1] - ty))
        if k:
            a, b = se2(mo, o), se2(prev_mo, o)
            corr.append(math.hypot(a[0] - b[0], a[1] - b[1]))
        prev_mo = mo
    err, err_odom, corr, ts = np.array(err), np.array(err_odom), np.array(corr), st * 1e-9
    out = {"run": rd.name, "poses": len(st), "ate_m": float(np.sqrt(np.mean(err ** 2)))}
    if cr:
        t_c = cr[0]["t"]
        t_start = ts[0]
        late = (ts > t_start + 0.75 * (t_c - t_start)) & (corr >= LOOP_STEP_M)
        k_lc = int(np.argmax(late)) if late.any() else None
        k_pre = (k_lc - 1) if k_lc is not None else max(0, int(np.searchsorted(ts, t_c)) - 1)
        race = ts > t_c + RACE_AFTER_S
        out.update({"open_loop_m": float(err[k_pre]), "open_loop_odom_m": float(err_odom[k_pre]), "open_loop_max_m": float(err[:k_pre + 1].max()),
                    "closure_step_m": float(corr[k_lc]) if k_lc is not None else None, "closed": k_lc is not None,
                    "t_closure_minus_crossing_s": float(ts[k_lc] - t_c) if k_lc is not None else None,
                    "race_p95_m": float(np.percentile(err[race], 95)) if race.any() else None})
    return out


def cmd_compare(args: argparse.Namespace) -> None:
    tags = args.tags.split(",")
    rows = {t: [m for m in (run_metrics(r, t, args.anchor) for r in args.runs) if m] for t in tags}
    if args.per_run:
        for t in tags:
            for m in rows[t]:
                print(t, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}))
    print(f"{'config':22s} {'runs':>4s} {'open-loop med/p90/max':>22s} {'odom med/p90':>13s} {'worse>10cm':>10s} "
          f"{'closure p90':>11s} {'race p95 med/max':>17s} {'race>0.35':>9s} {'ATE med/max':>12s}")
    for t in tags:
        r = [m for m in rows[t] if "open_loop_m" in m]
        if not r:
            print(f"{t:22s} no data")
            continue
        ol = np.array([m["open_loop_m"] for m in r])
        od = np.array([m["open_loop_odom_m"] for m in r])
        cs = np.array([m["closure_step_m"] for m in r if m["closure_step_m"] is not None])
        rp = np.array([m["race_p95_m"] for m in r if m["race_p95_m"] is not None])
        ate = np.array([m["ate_m"] for m in r])
        worse = [m["run"] for m in r if m["open_loop_m"] > m["open_loop_odom_m"] + WORSE_M]
        bad_race = [m["run"] for m in r if (m["race_p95_m"] or 0.0) > RACE_BAD_M]
        print(f"{t:22s} {len(r):4d} {np.median(ol):6.2f} /{np.percentile(ol, 90):5.2f} /{ol.max():5.2f}  {np.median(od):6.2f} /{np.percentile(od, 90):5.2f} "
              f"{len(worse):10d} {np.percentile(cs, 90) if len(cs) else float('nan'):11.2f} {np.median(rp):8.2f} /{rp.max():5.2f} {len(bad_race):9d} "
              f"{np.median(ate):6.2f} /{ate.max():5.2f}")
        if args.list_failures:
            print(f"{'':22s} worse: {' '.join(worse)} | race: {' '.join(bad_race)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="rebuild the SLAM input of each run (learning/.venv python)")
    p.add_argument("runs", nargs="+")
    p.add_argument("--jobs", type=int, default=8)
    p.set_defaults(fn=cmd_prepare)
    p = sub.add_parser("run", help="replay the runs through slam_toolbox (ROS sourced, no conda)")
    p.add_argument("runs", nargs="+")
    p.add_argument("--tag", required=True, help="name of this configuration (output <run>/slam_replay/<tag>.npz)")
    p.add_argument("--set", action="append", default=[], help="slam_toolbox parameter override key=value")
    p.add_argument("--perturb", action="append", default=[], help="degrade the odometry: yaw_rw_deg_rtm=, scale=, seed=")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--pace-ms", type=float, default=20.0, help="wall time between published scans (drops if too fast, see replay_one)")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("_one")
    p.add_argument("run")
    p.add_argument("--tag", required=True)
    p.add_argument("--config", default="{}")
    p.add_argument("--perturb-json", default="{}")
    p.add_argument("--pace-ms", type=float, default=20.0)
    p.set_defaults(fn=cmd_one)
    p = sub.add_parser("compare", help="metrics of configurations over the runs")
    p.add_argument("runs", nargs="+")
    p.add_argument("--tags", required=True)
    p.add_argument("--per-run", action="store_true")
    p.add_argument("--list-failures", action="store_true")
    p.add_argument("--anchor", choices=("true", "nominal"), default="true", help="map placed at the true start (SLAM drift only) or the nominal one")
    p.set_defaults(fn=cmd_compare)
    a = ap.parse_args()
    a.fn(a)
