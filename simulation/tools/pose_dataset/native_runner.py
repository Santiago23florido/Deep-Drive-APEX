"""Gazebo-native sensors for one trajectory (pose dataset v2, real2sim).

Same lock-step, in-process simulation as ``gz_runner`` (driver, actuators,
phases and safety checks are the same, and ``gz_runner`` itself is left
untouched so the v1 trajectory cache stays valid), with three differences:

* the world is the installed Gazebo world of the track
  (``rc_sim_description/worlds/pose_dataset/<track>.world``), the one the live
  simulation loads, with the walls as real collisions and visuals;
* the car comes from the same xacro as the live simulation with its sprung
  body (``suspension:=true``) and Gazebo's own sensors: a gpu_lidar geometry
  capture and a 1 kHz IMU with Gazebo's native noise model (plus its
  noise-free twin for the bias labels), mounted at the true mounts drawn for
  this run;
* the measurements are Gazebo's, turned into what the real devices output by
  the realism layer shared with the live simulation
  (``apex_fusion_research.core.lidar_rolling`` and ``imu_chip``).

The result (per sensor-profile run: ``synth`` + QA pickles and the summary
JSON) has the layout of ``gz_runner`` so ``campaign`` and ``db`` store it
unchanged. Runs of one trajectory whose profiles share a ``hardware`` key use
the same physical sensors (e.g. the A2M8 in Sensitivity and in compatible
mode).
"""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import threading
import time
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from .controller import PurePursuit
from .geometry import FootprintChecker, rect_corners, write_xml
from .gz_runner import CTRL_COLUMNS, PHASE_DRIVE, PHASE_FINAL, PHASE_HOLD, PHASE_SETTLE, PHASE_STATIC, STATE_COLUMNS, build_run_setup, quat_to_rpy
from .qa import run_qa
from .sensors import Trajectory, draw_extrinsic, ground_truth_rows, load_sensor_profiles, quat_to_mat, rng_for, seed_int
from .tracks import load_track
from .vehicle import SIM_ROOT, Actuators, load_vehicle_config

from apex_fusion_research.core.imu_chip import ImuChip, chip_config_from_profile, gazebo_noise_from_profile  # noqa: E402
from apex_fusion_research.core.lidar_rolling import RollingLidar, rolling_config_from_profile  # noqa: E402

WORLDS_DIR = SIM_ROOT / "ros2_ws" / "src" / "rc_sim_description" / "worlds" / "pose_dataset"
XACRO = SIM_ROOT / "ros2_ws" / "src" / "rc_sim_description" / "urdf" / "rc_car.urdf.xacro"
EXTRA_COLUMNS = ["susp_roll", "susp_pitch", "sprung_x", "sprung_y", "sprung_z", "sprung_qw", "sprung_qx", "sprung_qy", "sprung_qz"]
TOPIC_SCAN, TOPIC_IMU, TOPIC_IMU_CLEAN = "/apex/sim/scan", "/apex/sim/imu", "/apex/sim/imu_clean"
NATIVE_IMU_HZ = 1000.0


def _clean_env() -> dict[str, str]:
    return {"HOME": os.environ.get("HOME", "/tmp"), "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}


def _axes(v: list[float]) -> str:
    return " ".join(f"{float(x):.9g}" for x in v)


# ------------------------------------------------------------------- models
def hardware_of(profile_name: str, profile: dict[str, Any]) -> str:
    return str(profile.get("hardware", profile_name))


def car_model(profile: dict[str, Any], lidar_ext: dict[str, Any], imu_ext: dict[str, Any], out_path: Path) -> tuple[ET.Element, dict[str, Any]]:
    """The live-simulation car (xacro) with suspension and this run's sensors.

    Joint controllers and publishers are removed (the actuators are applied
    in-process, exactly like ``gz_runner``); the sensors stay."""
    li, imu = profile["lidar"], profile["imu"]
    gz = li.get("gazebo", {})
    noise = gazebo_noise_from_profile(imu, NATIVE_IMU_HZ)
    args = {
        "suspension": "true", "camera": "false", "imu_clean_reference": "true", "imu_clean_topic": TOPIC_IMU_CLEAN,
        "imu_update_rate": f"{NATIVE_IMU_HZ:g}",
        "lidar_update_rate": f"{float(gz.get('capture_rate_hz', 26.0)):g}",
        "lidar_samples": str(int(gz.get("capture_samples", 1440))),
        "lidar_range_min": f"{float(gz.get('capture_range_min_m', 0.05)):g}",
        "lidar_range_max": f"{1.2 * float(li['range_max_m']):g}",
        "lidar_range_resolution": "0.0001",
        "lidar_xyz": _axes(lidar_ext["xyz_true"]), "lidar_rpy": _axes(lidar_ext["rpy_true_rad"]),
        "imu_xyz": _axes(imu_ext["xyz_true"]), "imu_rpy": _axes(imu_ext["rpy_true_rad"]),
        **{f"imu_{k}": _axes(v) for k, v in noise.items()},
    }
    arg_str = " ".join(f'{k}:="{v}"' for k, v in args.items())
    urdf = subprocess.run(["bash", "-c", f"source /opt/ros/jazzy/setup.bash && xacro {XACRO} {arg_str}"],
                          check=True, capture_output=True, text=True, env=_clean_env()).stdout
    tmp = out_path.with_suffix(".urdf")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(urdf, encoding="utf-8")
    sdf = subprocess.run(["gz", "sdf", "-p", str(tmp)], check=True, capture_output=True, text=True, env=_clean_env()).stdout
    tmp.unlink(missing_ok=True)
    model = ET.fromstring(sdf).find("model")
    for plugin in list(model.findall("plugin")):
        model.remove(plugin)
    model.set("name", "rc_car")
    info: dict[str, Any] = {"xacro_args": args}
    for link in model.findall("link"):
        for sensor in link.findall("sensor"):
            pose = [float(v) for v in (sensor.findtext("pose") or "0 0 0 0 0 0").split()]
            info[f"{sensor.get('name')}_link"] = link.get("name")
            info[f"{sensor.get('name')}_pose"] = pose
    ET.ElementTree(model).write(out_path, encoding="unicode")
    return model, info


def native_world(track_name: str, car: ET.Element, car_pose: tuple[float, ...], out_path: Path) -> Path:
    """The installed Gazebo world of the track, running as fast as possible."""
    text = (WORLDS_DIR / f"{track_name}.world").read_text(encoding="utf-8")
    root = ET.fromstring(re.sub(r"<!--.*?-->", "", text, flags=re.S))
    world = root.find("world")
    physics = world.find("physics")
    for tag, value in (("real_time_factor", "0"), ("real_time_update_rate", "0")):
        el = physics.find(tag)
        if el is not None:
            el.text = value
    for plugin in list(world.findall("plugin")):
        if any(s in plugin.get("name", "") for s in ("SceneBroadcaster", "UserCommands")):
            world.remove(plugin)
    car = copy.deepcopy(car)
    for old in car.findall("pose"):
        car.remove(old)
    pose = ET.Element("pose")
    pose.text = " ".join(f"{v:.6f}" for v in car_pose)
    car.insert(0, pose)
    world.append(car)
    write_xml(root, out_path)
    return out_path


def _pose_mat(p: list[float]) -> tuple[np.ndarray, np.ndarray]:
    from .sensors import rpy_to_mat

    return np.asarray(p[:3], float), rpy_to_mat(*p[3:6])


# --------------------------------------------------------------- simulation
def run_native(job: dict[str, Any]) -> dict[str, Any]:
    from gz.common5 import set_verbosity
    import gz.math7
    from gz.msgs10.imu_pb2 import IMU
    from gz.msgs10.laserscan_pb2 import LaserScan
    from gz.sim8 import Joint, Link, Model, TestFixture, World, world_entity
    from gz.transport13 import Node

    wall_start = time.time()
    if job.get("track_overrides"):
        raise NotImplementedError("track_overrides (e.g. ground friction) are not supported by the native backend")
    vcfg = load_vehicle_config()
    track = load_track(job["track"])
    profiles = load_sensor_profiles()
    runs = job.get("runs", [])
    hw = {hardware_of(r["sensor"], profiles[r["sensor"]]) for r in runs}
    if len(hw) != 1:
        raise ValueError(f"one physical sensor set per trajectory is supported, got {sorted(hw)}")
    hardware = hw.pop()
    base_profile = profiles[runs[0]["sensor"]]
    tkey = job["trajectory_key"]
    setup = build_run_setup(job, track, vcfg)
    path, plan = setup["path"], setup["plan"]
    dt = float(vcfg["run"]["physics_step_s"])
    step_ns = int(round(dt * 1e9))
    settle_s = float(vcfg["model"]["settle_s"])
    static_start = float(vcfg["run"]["static_start_s"])
    static_end = float(vcfg["run"]["static_end_s"])
    safety = vcfg["safety"]
    planned = plan.planned_duration()
    timeout_s = settle_s + static_start + safety["timeout_factor"] * planned + 15.0
    if "max_sim_s" in job:
        timeout_s = min(timeout_s, float(job["max_sim_s"]))
    n_max = int(timeout_s / dt) + 10
    record_start = int(round(settle_s / dt))

    heading0 = float(path.heading[0])
    rear_offset = -0.15
    lat = setup["init_lateral_error"]
    x0 = path.xy[0, 0] - rear_offset * math.cos(heading0) - lat * math.sin(heading0)
    y0 = path.xy[0, 1] - rear_offset * math.sin(heading0) + lat * math.cos(heading0)
    yaw0 = heading0 + setup["init_yaw_error"]
    car_pose = (x0, y0, float(vcfg["model"]["spawn_z_m"]), 0.0, 0.0, yaw0)

    # Physical sensor set of this trajectory: true mounts drawn once (shared by its runs).
    rng_hw = rng_for(tkey, hardware, "mounts")
    lidar_ext = draw_extrinsic(base_profile["lidar"], rng_hw)
    imu_ext = draw_extrinsic(base_profile["imu"], rng_hw)
    tmp_dir = Path(job.get("tmp_dir") or "/tmp")
    car, car_info = car_model(base_profile, lidar_ext, imu_ext, tmp_dir / f"{tkey}_car.sdf")
    world_path = native_world(job["track"], car, car_pose, tmp_dir / f"{tkey}_world.sdf")
    lidar_link, lidar_pose = car_info["lidar_link"], car_info["lidar_pose"]
    l_off, l_rot = _pose_mat(lidar_pose)

    # Realism layer per run (sensor profile) on the shared hardware.
    vib = vcfg["body_dynamics"]["vibration"]
    chip = ImuChip(chip_config_from_profile(base_profile["imu"], vib, seed_int(tkey, hardware, "imu_chip")))
    lidars = {}
    for r in runs:
        cfg = rolling_config_from_profile(profiles[r["sensor"]]["lidar"], seed_int(tkey, r["sensor"], "lidar"))
        cfg.start_grid_s = dt
        lidars[r["run_key"]] = RollingLidar(cfg)
    imu_out: list[Any] = []
    revs: dict[str, list[Any]] = {k: [] for k in lidars}

    # Sensor streams (gz-transport callbacks run on transport threads).
    lock = threading.Lock()
    q_imu: dict[int, list[float]] = {}
    q_clean: dict[int, list[float]] = {}
    q_scan: list[tuple[int, np.ndarray, float, float]] = []

    def stamp(m) -> int:  # noqa: ANN001
        return int(m.header.stamp.sec) * 1_000_000_000 + int(m.header.stamp.nsec)

    def on_imu(m, target) -> None:  # noqa: ANN001
        v = [m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z, m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z]
        with lock:
            target[stamp(m)] = v

    def on_scan(m) -> None:  # noqa: ANN001
        with lock:
            q_scan.append((stamp(m), np.asarray(m.ranges, dtype=float), float(m.angle_min), float(m.angle_step)))

    node = Node()
    node.subscribe(IMU, TOPIC_IMU, lambda m: on_imu(m, q_imu))
    node.subscribe(IMU, TOPIC_IMU_CLEAN, lambda m: on_imu(m, q_clean))
    node.subscribe(LaserScan, TOPIC_SCAN, on_scan)

    state = np.zeros((n_max, len(STATE_COLUMNS) + len(EXTRA_COLUMNS)), dtype=np.float64)
    ctrl = np.zeros((n_max, len(CTRL_COLUMNS)), dtype=np.float32)
    actuators = Actuators(vcfg, setup["actuator_perturbation"])
    pursuit = PurePursuit(path, actuators.wheelbase, vcfg["controller"], setup["gain_scale"])
    checker = FootprintChecker(track.geometry)
    footprint_len = float(vcfg["model"]["chassis_length_m"])
    footprint_w = float(vcfg["model"]["footprint_width_m"])
    ctrl_every = max(1, int(round(1.0 / (float(vcfg["esc"]["command_rate_hz"]) * dt))))
    check_every = max(1, int(round(float(safety["collision_check_period_s"]) / dt)))
    total_distance = float(job.get("laps", 1.0)) * path.length
    st: dict[str, Any] = {
        "init": False, "k": 0, "phase": PHASE_SETTLE, "done": False, "status": "running", "reason": "",
        "stop_idx": 0, "hold_since": None, "lap_completed": False, "final_since": None, "stuck_since": None,
        "s": 0.0, "lat": 0.0, "plan_v": 0.0, "events": [], "drive_start_ns": None, "lap_ns": None,
        "max_lat": 0.0, "max_rp": 0.0, "max_slip": 0.0,
    }
    stops = list(plan.stops)
    roll_pitch_limit = math.radians(float(safety["max_roll_pitch_deg"]))

    def event(k: int, kind: str, **details: Any) -> None:
        st["events"].append({"step": k, "t_ns": k * step_ns, "type": kind, "details": details})

    def pre_update(info, ecm):  # noqa: ANN001 - gz callback signature (mirrors gz_runner.run_physics)
        if st["done"]:
            return
        if not st["init"]:
            model = Model(World(world_entity(ecm)).model_by_name(ecm, "rc_car"))
            st["base"] = Link(model.link_by_name(ecm, "base_link"))
            st["base"].enable_velocity_checks(ecm, True)
            st["base"].enable_acceleration_checks(ecm, True)
            st["sprung"] = Link(model.link_by_name(ecm, lidar_link))
            names = ("rear_left_wheel_joint", "rear_right_wheel_joint", "front_left_wheel_steer_joint", "front_right_wheel_steer_joint",
                     "suspension_roll_joint", "suspension_pitch_joint")
            st["joints"] = [Joint(model.joint_by_name(ecm, n)) for n in names]
            for j in st["joints"]:
                j.enable_position_check(ecm, True)
                j.enable_velocity_check(ecm, True)
            st["init"] = True
        k = int(info.iterations) - 1
        if k >= n_max:
            st["done"], st["status"], st["reason"] = True, "failed", "buffer_full"
            return
        st["k"] = k
        base = st["base"]
        pose = base.world_pose(ecm)
        p, q = pose.pos(), pose.rot()
        row = state[k]
        row[0], row[1], row[2] = p.x(), p.y(), p.z()
        row[3], row[4], row[5], row[6] = q.w(), q.x(), q.y(), q.z()
        v = base.world_linear_velocity(ecm)
        if v is not None:
            row[7], row[8], row[9] = v.x(), v.y(), v.z()
        w = base.world_angular_velocity(ecm)
        if w is not None:
            row[10], row[11], row[12] = w.x(), w.y(), w.z()
        a = base.world_linear_acceleration(ecm)
        if a is not None:
            row[13], row[14], row[15] = a.x(), a.y(), a.z()
        al = base.world_angular_acceleration(ecm)
        if al is not None:
            row[16], row[17], row[18] = al.x(), al.y(), al.z()
        jl, jr, kl, kr, jroll, jpitch = st["joints"]
        row[19] = (jroll.position(ecm) or [0.0])[0]
        row[20] = (jpitch.position(ecm) or [0.0])[0]
        sp = st["sprung"].world_pose(ecm)
        row[21], row[22], row[23] = sp.pos().x(), sp.pos().y(), sp.pos().z()
        row[24], row[25], row[26], row[27] = sp.rot().w(), sp.rot().x(), sp.rot().y(), sp.rot().z()
        kpos_l = (kl.position(ecm) or [0.0])[0]
        kpos_r = (kr.position(ecm) or [0.0])[0]
        t = k * dt
        yaw = math.atan2(2 * (row[3] * row[6] + row[4] * row[5]), 1 - 2 * (row[5] ** 2 + row[6] ** 2))
        speed = row[7] * math.cos(yaw) + row[8] * math.sin(yaw)

        phase = st["phase"]
        if phase == PHASE_SETTLE and t >= settle_s:
            phase = st["phase"] = PHASE_STATIC
            event(k, "static_start")
        if phase == PHASE_STATIC and t >= settle_s + static_start:
            phase = st["phase"] = PHASE_DRIVE
            st["drive_start_ns"] = k * step_ns
            event(k, "drive_start", v_target=plan.target(0.3))
        if k % ctrl_every == 0 and phase in (PHASE_DRIVE, PHASE_HOLD, PHASE_FINAL):
            xr = row[0] + rear_offset * math.cos(yaw)
            yr = row[1] + rear_offset * math.sin(yaw)
            s_tot, lat_err = pursuit.locate(xr, yr)
            st["s"], st["lat"] = s_tot, lat_err
            st["max_lat"] = max(st["max_lat"], abs(lat_err))
            steer = pursuit.steering(xr, yr, yaw, speed)
            v_target = plan.target(s_tot + max(0.3, 0.3 * speed))
            if phase == PHASE_DRIVE and st["stop_idx"] < len(stops) and s_tot >= stops[st["stop_idx"]][0] - 0.35:
                phase = st["phase"] = PHASE_HOLD
                st["hold_since"] = None
                event(k, "stop_begin", s=round(s_tot, 3), planned_dwell_s=stops[st["stop_idx"]][1])
            if phase == PHASE_HOLD:
                v_target = 0.0
                if abs(speed) < 0.02 and st["hold_since"] is None:
                    st["hold_since"] = t
                if st["hold_since"] is not None and t - st["hold_since"] >= stops[st["stop_idx"]][1]:
                    event(k, "stop_end", s=round(s_tot, 3))
                    st["stop_idx"] += 1
                    phase = st["phase"] = PHASE_DRIVE
                    v_target = plan.target(s_tot + 0.3)
            if phase == PHASE_DRIVE and s_tot >= total_distance - 0.6:
                phase = st["phase"] = PHASE_FINAL
                event(k, "final_brake", s=round(s_tot, 3))
            if phase == PHASE_FINAL:
                v_target = 0.0
                if abs(speed) < 0.02 and st["final_since"] is None:
                    st["final_since"] = t
                    st["lap_completed"] = s_tot >= total_distance - 1.0
                    st["lap_ns"] = k * step_ns
                    event(k, "lap_complete" if st["lap_completed"] else "stopped_short", s=round(s_tot, 3))
                if st["final_since"] is not None and t - st["final_since"] >= static_end:
                    st["done"], st["status"] = True, "completed"
                    event(k, "run_end")
            st["plan_v"] = v_target
            actuators.set_command(v_target, math.degrees(steer))
        elif phase in (PHASE_SETTLE, PHASE_STATIC):
            actuators.set_command(0.0, 0.0)
        actuators.step(dt)
        om_l, om_r = actuators.wheel_omegas()
        jl.set_velocity(ecm, [om_l])
        jr.set_velocity(ecm, [om_r])
        tq_l, tq_r = actuators.knuckle_torques((kpos_l, kpos_r), dt)
        kl.set_force(ecm, [tq_l])
        kr.set_force(ecm, [tq_r])
        c = ctrl[k]
        c[0], c[1] = kpos_l, kpos_r
        c[2], c[3] = (jl.velocity(ecm) or [0.0])[0], (jr.velocity(ecm) or [0.0])[0]
        c[4], c[5], c[6], c[7] = actuators.speed_cmd, actuators.speed, actuators.steer_cmd_deg, actuators.steer_deg
        c[8], c[9], c[10], c[11] = st["s"], st["lat"], st["plan_v"], phase
        if k % check_every == 0 and phase >= PHASE_STATIC:
            roll, pitch, _ = quat_to_rpy(row[3], row[4], row[5], row[6])
            st["max_rp"] = max(st["max_rp"], abs(roll), abs(pitch))
            if speed > 0.5:
                vb_y = -row[7] * math.sin(yaw) + row[8] * math.cos(yaw)
                st["max_slip"] = max(st["max_slip"], abs(math.atan2(vb_y + rear_offset * row[12], speed)))
            reason = None
            hit = checker.collides(rect_corners(row[0], row[1], yaw, footprint_len, footprint_w))
            if hit:
                reason = f"collision with {hit}"
            elif abs(st["lat"]) > float(safety["max_lateral_error_m"]):
                reason = f"off_track (lateral error {st['lat']:.2f} m)"
            elif max(abs(roll), abs(pitch)) > roll_pitch_limit:
                reason = "rollover"
            elif phase == PHASE_DRIVE and actuators.speed > 0.3 and speed < 0.05:
                st["stuck_since"] = st["stuck_since"] if st["stuck_since"] is not None else t
                if t - st["stuck_since"] > float(safety["stuck_timeout_s"]):
                    reason = "stuck"
            else:
                st["stuck_since"] = None
            if reason is None and t > timeout_s - 0.01:
                reason = "timeout"
            if reason:
                st["done"], st["status"], st["reason"] = True, "failed", reason
                event(k, "failure", reason=reason, s=round(st["s"], 3))

    def laser_planar(k_idx: np.ndarray) -> np.ndarray:
        """Planar pose (x, y, yaw) of the LiDAR frame at physics steps ``k_idx``."""
        rows = state[k_idx]
        rot = quat_to_mat(rows[:, 24:28])
        pos = rows[:, 21:24] + np.einsum("nij,j->ni", rot, l_off)
        r_s = rot @ l_rot
        return np.column_stack((pos[:, 0], pos[:, 1], np.arctan2(r_s[:, 1, 0], r_s[:, 0, 0])))

    fed = {"k": record_start}

    def drain(final: bool = False) -> None:
        """Feed the realism layer with every sensor sample up to the last step."""
        # A sample stamped T belongs to the state of step T / dt, written by
        # the next pre-update: keep later samples for the next drain.
        horizon = (st["k"] if not final else n_max) * step_ns
        with lock:
            imu_keys = sorted(t for t in q_imu if t in q_clean and t <= horizon)
            imu_rows = [(t, q_imu.pop(t), q_clean.pop(t)) for t in imu_keys]
            scans = [s for s in q_scan if s[0] <= horizon]
            q_scan[:] = [s for s in q_scan if s[0] > horizon]
        rec_ns = record_start * step_ns
        imu_rows = [r for r in imu_rows if r[0] >= rec_ns]
        if imu_rows:
            t_ns = np.array([r[0] for r in imu_rows], dtype=np.int64)
            noisy = np.array([r[1] for r in imu_rows])
            clean = np.array([r[2] for r in imu_rows])
            kk = np.clip(t_ns // step_ns, 0, n_max - 1)
            speed = np.hypot(state[kk, 7], state[kk, 8])
            imu_out.extend(chip.push(t_ns, noisy[:, :3], noisy[:, 3:], speed, clean[:, :3], clean[:, 3:]))
        last = st["k"]
        if last > fed["k"]:
            ks = np.arange(fed["k"], last)
            poses = laser_planar(ks)
            for lid in lidars.values():
                for k_, (x, y, yw) in zip(ks, poses):
                    lid.add_pose(k_ * dt, float(x), float(y), float(yw))
            fed["k"] = last
        for t_ns, ranges, a_min, a_step in scans:
            k_ = int(t_ns // step_ns)
            if t_ns < rec_ns or k_ > st["k"]:
                continue
            x, y, yw = laser_planar(np.array([k_]))[0]
            for lid in lidars.values():
                lid.add_capture(t_ns * 1e-9, ranges, a_min, a_step, (float(x), float(y), float(yw)))
        for key, lid in lidars.items():
            revs[key].extend(lid.poll())

    set_verbosity(1)
    gz.math7.Rand.seed(int(seed_int(tkey, hardware, "gazebo_noise") % (2**31 - 1)))
    fixture = TestFixture(str(world_path))
    fixture.on_pre_update(pre_update)
    fixture.finalize()
    server = fixture.server()
    chunk = 50
    while not st["done"]:
        server.run(True, chunk, False)
        drain()
    time.sleep(0.3)  # let the transport threads deliver the last messages
    drain(final=True)
    n = st["k"] + 1
    sim_wall = time.time() - wall_start
    summary = {
        "trajectory_key": tkey, "status": st["status"], "failure_reason": st["reason"], "lap_completed": bool(st["lap_completed"]),
        "steps": n, "physics_step_ns": step_ns, "record_start_ns": record_start * step_ns, "drive_start_ns": st["drive_start_ns"],
        "lap_end_ns": st["lap_ns"], "simulated_s": n * dt, "wall_s": sim_wall, "real_time_factor": n * dt / max(sim_wall, 1e-9),
        "distance_m": float(st["s"]), "path_length_m": path.length, "laps": float(job.get("laps", 1.0)),
        "max_lateral_error_m": st["max_lat"], "max_roll_pitch_deg": math.degrees(st["max_rp"]), "max_rear_slip_deg": math.degrees(st["max_slip"]),
        "planned_duration_s": planned, "events": st["events"], "sensor_backend": "gazebo_native", "hardware": hardware,
        "car_model": car_info,
        "setup": {"direction": setup["direction"], "start_s": setup["start_s"], "lateral_offset_m": setup["lateral_offset"],
                  "lookahead_gain_scale": setup["gain_scale"], "actuator_perturbation": setup["actuator_perturbation"],
                  "initial_yaw_error_rad": setup["init_yaw_error"], "initial_lateral_error_m": setup["init_lateral_error"],
                  "weave": setup["weave"], "v_ref_mps": setup["v_ref"], "plan": plan.description, "car_initial_pose": list(car_pose)},
        "job": {k2: v2 for k2, v2 in job.items() if k2 not in ("tmp_dir",)},
        "gz_partition": os.environ.get("GZ_PARTITION", ""),
    }
    out = Path(job["out_npz"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, state=state[:n], ctrl=ctrl[:n], state_columns=np.array(STATE_COLUMNS + EXTRA_COLUMNS),
                        ctrl_columns=np.array(CTRL_COLUMNS), summary=np.array(json.dumps(summary, default=float)))
    for p in (world_path, tmp_dir / f"{tkey}_car.sdf"):
        p.unlink(missing_ok=True)
    summary["runs"] = []
    if summary["status"] == "completed":
        summary["runs"] = synthesize_runs(job, summary, profiles, imu_out, revs, chip, lidars, lidar_ext, imu_ext)
    return summary


def synthesize_runs(job: dict[str, Any], summary: dict[str, Any], profiles: dict[str, Any], imu_out: list[Any], revs: dict[str, list[Any]],
                    chip: ImuChip, lidars: dict[str, RollingLidar], lidar_ext: dict[str, Any], imu_ext: dict[str, Any]) -> list[dict[str, Any]]:
    """``gz_runner``-compatible synth dict + QA of every run."""
    traj = Trajectory(Path(job["out_npz"]))
    st = np.load(job["out_npz"])["state"]
    body = {"roll": st[:, 19], "pitch": st[:, 20]}
    step = traj.step_ns
    imu_true = np.array([o.t_true_ns for o in imu_out], dtype=np.int64)
    keep = np.round(imu_true / step) < traj.n  # the nearest physics step must exist (last sample of the run)
    imu_out = [o for o, k in zip(imu_out, keep) if k]
    imu_idx = np.round(np.array([o.t_true_ns for o in imu_out]) / step).astype(np.int64).clip(0, traj.n - 1)
    imu_values = np.array([np.concatenate((o.accel, o.gyro)) for o in imu_out]).reshape(-1, 6)
    imu_bias = np.array([np.concatenate((o.bias_accel, o.bias_gyro)) for o in imu_out]).reshape(-1, 6)
    fs_g = float(chip.cfg.gyro.full_scale)
    fs_a = float(chip.cfg.accel.full_scale)
    saturated = int(np.sum((np.abs(imu_values[:, 3:]) >= 0.9999 * fs_g).any(1) if fs_g > 0 else 0) + np.sum((np.abs(imu_values[:, :3]) >= 0.9999 * fs_a).any(1) if fs_a > 0 else 0))
    results = []
    for r in job.get("runs", []):
        t0 = time.time()
        profile = copy.deepcopy(profiles[r["sensor"]])
        lid = lidars[r["run_key"]]
        rv = [v for v in revs[r["run_key"]] if not v.lost and int(round(v.t_end_ns / step)) < traj.n]
        lost = [int(v.t_start_ns // step) for v in revs[r["run_key"]] if v.lost]
        beams = int(profile["lidar"]["beams"])
        ranges = np.array([v.ranges for v in rv], dtype=np.float32).reshape(-1, beams)
        valid = np.isfinite(ranges)
        cfg = lid.cfg
        lidar = {
            "start_idx": np.array([v.t_start_ns // step for v in rv], dtype=np.int64),
            "end_idx": np.array([int(round(v.t_end_ns / step)) for v in rv], dtype=np.int64),
            "t_report_ns": np.array([v.stamp_ns for v in rv], dtype=np.int64),
            "ranges": ranges, "valid": valid, "outcome": np.where(valid, 1, 2).astype(np.uint8),
            "period_ns": np.array([v.period_s * 1e9 for v in rv]), "lost_scans": lost,
            "angle_min": cfg.angle_min_rad, "angle_increment": 2.0 * math.pi / beams,
            "range_min": cfg.range_min, "range_max": cfg.range_max, "beams": beams,
            "time_increment_ns": 1e9 / cfg.rate_hz / beams, "extrinsic": lidar_ext,
            "realization": {**lid.noise.realization(), "scan_direction": cfg.direction, "scan_start_angle_deg": cfg.start_angle_deg,
                            "sample_rate_hz": cfg.sample_rate_hz, "binning": cfg.binning},
        }
        imu = {
            "true_idx": imu_idx, "t_report_ns": np.array([o.stamp_ns for o in imu_out], dtype=np.int64),
            "values": imu_values, "bias_true": imu_bias,
            "n_expected": int(len(imu_out) + chip.samples_dropped), "n_dropped": int(chip.samples_dropped), "bursts": [],
            "saturated": saturated, "extrinsic": imu_ext, "realization": chip.realization(), "period_ns": chip.period_s * 1e9,
        }
        gt_idx = np.unique(np.concatenate((imu["true_idx"], lidar["start_idx"], lidar["end_idx"])))
        synth = {"imu": imu, "lidar": lidar, "gt": ground_truth_rows(traj, body, gt_idx), "profile": profile,
                 "body_params": {"source": "gazebo suspension joints", "car_model": summary.get("car_model", {})}}
        qa = run_qa(synth, summary, float(profile["imu"]["rate_hz"]), float(profile["lidar"]["rate_hz"]))
        out = Path(job["out_npz"]).parent / f"{r['run_key']}.pkl"
        with open(out, "wb") as fh:
            pickle.dump({"synth": synth, "qa": qa}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        results.append({"run_key": r["run_key"], "file": str(out), "qa_passed": qa["passed"], "synth_wall_s": time.time() - t0})
    return results


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    job = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    out = Path(job["out_npz"])
    summary = run_native(job)
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: summary.get(k) for k in ("trajectory_key", "status", "failure_reason", "simulated_s", "wall_s", "real_time_factor")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
