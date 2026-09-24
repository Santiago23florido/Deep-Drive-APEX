"""Headless, in-process, lock-step Gazebo simulation of one trajectory.

gz-sim 8 runs inside this Python process through its official bindings
(``gz.sim8.TestFixture``); no GUI, no rendering, no ROS and no transport in
the control loop. A single pre-update callback, executed at every 1 ms
physics step:

1. reads the exact state of the car (pose, twist and accelerations of
   ``base_link`` in the world frame, knuckle and wheel joint states);
2. runs the driver at 50 Hz (pure pursuit + speed plan) and the actuator
   models at 1 kHz;
3. writes the wheel velocity and knuckle torque commands to the ECM.

Because the controller runs inside the simulation step, a run is fully
deterministic for a given job (seed), and the simulation runs as fast as the
CPU allows (``real_time_factor = 0``) with the same 1 ms physics step as the
ROS worlds. Several jobs run in parallel processes (see ``campaign.py``).
"""

from __future__ import annotations

import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np

from .controller import PurePursuit, driving_path, load_motion_profiles, plan_speed
from .geometry import KIND_GROUND, FootprintChecker, WorldGeometry, rect_corners, world_sdf, write_xml
from .tracks import Track, load_track
from .vehicle import Actuators, car_model_element, load_vehicle_config

STATE_COLUMNS = [
    "x", "y", "z", "qw", "qx", "qy", "qz",
    "vx", "vy", "vz", "wx", "wy", "wz",
    "ax", "ay", "az", "alx", "aly", "alz",
]
CTRL_COLUMNS = [
    "knuckle_left_rad", "knuckle_right_rad", "wheel_rl_radps", "wheel_rr_radps",
    "esc_cmd_mps", "esc_applied_mps", "servo_cmd_deg", "servo_applied_deg",
    "s_progress_m", "lateral_error_m", "plan_target_mps", "phase",
]
PHASE_SETTLE, PHASE_STATIC, PHASE_DRIVE, PHASE_HOLD, PHASE_FINAL = 0, 1, 2, 3, 4


def seed_streams(seed: int, *labels: str) -> dict[str, np.random.Generator]:
    """Independent, reproducible random streams derived from one integer seed."""
    import zlib

    words = [int(seed)] + [zlib.crc32(label.encode()) for label in labels]
    names = ["trajectory", "plan"]
    return {n: np.random.default_rng(s) for n, s in zip(names, np.random.SeedSequence(words).spawn(len(names)))}


def quat_to_rpy(qw: float, qx: float, qy: float, qz: float) -> tuple[float, float, float]:
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return roll, pitch, yaw


def build_run_setup(job: dict[str, Any], track: Track, vcfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve every random choice of a trajectory from its seed."""
    if "speed_override_mps" in job:
        profile = {"kind": "constant", "fraction": 1.0}
    else:
        profile = dict(load_motion_profiles()[job["motion"]])
    rngs = seed_streams(job["seed"], job["track"], job["motion"])
    rt = rngs["trajectory"]
    calibration = bool(job.get("calibration", False))
    direction = int(job.get("direction", 1))
    offset_max = float(track.spec.get("lateral_offset_max", 0.10))
    start_s = 0.0 if calibration else float(rt.uniform(0.0, track.reference.length))
    lateral_offset = 0.0 if calibration else float(rt.uniform(-offset_max, offset_max))
    jitter = float(vcfg["controller"]["seed_gain_jitter"])
    gain_scale = 1.0 if calibration else float(rt.uniform(1.0 - jitter, 1.0 + jitter))
    perturb = {} if calibration else {"esc_tau_scale": float(rt.uniform(0.9, 1.1)), "servo_tau_scale": float(rt.uniform(0.9, 1.1))}
    init_yaw_err = 0.0 if calibration else float(rt.uniform(-math.radians(2.0), math.radians(2.0)))
    init_lat_err = 0.0 if calibration else float(rt.uniform(-0.05, 0.05))
    weave = None
    if profile["kind"] == "weave":
        weave = {"amplitude": float(profile["weave_amplitude_m"]), "period": float(rt.uniform(*profile["weave_period_m"]))}
    path = driving_path(track, direction, start_s, lateral_offset, weave)
    v_ref = float(job["speed_override_mps"] if "speed_override_mps" in job else job["v_ref"])
    plan = plan_speed(profile, path, float(job.get("laps", 1.0)), v_ref, rngs["plan"])
    return {
        "profile": profile,
        "direction": direction,
        "start_s": start_s,
        "lateral_offset": lateral_offset,
        "gain_scale": gain_scale,
        "actuator_perturbation": perturb,
        "init_yaw_error": init_yaw_err,
        "init_lateral_error": init_lat_err,
        "weave": weave,
        "path": path,
        "plan": plan,
        "v_ref": v_ref,
    }


def physics_world(track: Track, car_pose: tuple[float, ...], out_path: Path, physical_obstacles: bool) -> Path:
    geom = WorldGeometry()
    for b in track.geometry.boxes:
        if b.kind == KIND_GROUND or physical_obstacles:
            geom.boxes.append(b)
    if physical_obstacles:
        geom.cylinders.extend(track.geometry.cylinders)
    root = world_sdf(geom, ground_mu=track.ground_mu, extra_models=[car_model_element(car_pose)], real_time_factor=0.0)
    world = root.find("world")
    # No GUI streaming is needed in-process.
    for plugin in list(world.findall("plugin")):
        if "SceneBroadcaster" in plugin.get("name", ""):
            world.remove(plugin)
    write_xml(root, out_path)
    return out_path


def run_physics(job: dict[str, Any]) -> dict[str, Any]:
    """Simulate one trajectory and write ``job['out_npz']``. Returns a summary."""
    from gz.common5 import set_verbosity
    import gz.math7  # noqa: F401  (registers Pose3d/Vector3d return types)
    from gz.sim8 import Joint, Link, Model, TestFixture, World, world_entity

    wall_start = time.time()
    vcfg = load_vehicle_config()
    track = load_track(job["track"])
    track.ground_mu = float(job.get("track_overrides", {}).get("ground_mu", track.ground_mu))
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

    # Initial pose: path start, rear axle on the path, small seeded errors.
    heading0 = float(path.heading[0])
    rear_offset = -0.15
    lat = setup["init_lateral_error"]
    x0 = path.xy[0, 0] - rear_offset * math.cos(heading0) - lat * math.sin(heading0)
    y0 = path.xy[0, 1] - rear_offset * math.sin(heading0) + lat * math.cos(heading0)
    yaw0 = heading0 + setup["init_yaw_error"]
    car_pose = (x0, y0, float(vcfg["model"]["spawn_z_m"]), 0.0, 0.0, yaw0)

    tmp_dir = Path(job.get("tmp_dir") or tempfile.mkdtemp(prefix="pose_ds_"))
    world_path = physics_world(track, car_pose, tmp_dir / f"{job['trajectory_key']}_world.sdf", bool(job.get("physical_obstacles", False)))

    state = np.zeros((n_max, len(STATE_COLUMNS)), dtype=np.float64)
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
        "stop_idx": 0, "hold_since": None, "lap_completed": False,
        "final_since": None, "stuck_since": None, "s": 0.0, "lat": 0.0, "plan_v": 0.0,
        "events": [], "drive_start_ns": None, "lap_ns": None, "max_lat": 0.0, "max_rp": 0.0, "max_slip": 0.0,
    }
    stops = list(plan.stops)
    roll_pitch_limit = math.radians(float(safety["max_roll_pitch_deg"]))

    def event(k: int, kind: str, **details: Any) -> None:
        st["events"].append({"step": k, "t_ns": k * step_ns, "type": kind, "details": details})

    def pre_update(info, ecm):  # noqa: ANN001 - gz callback signature
        if st["done"]:
            return
        if not st["init"]:
            world = World(world_entity(ecm))
            model = Model(world.model_by_name(ecm, "rc_car"))
            st["base"] = Link(model.link_by_name(ecm, "base_link"))
            st["base"].enable_velocity_checks(ecm, True)
            st["base"].enable_acceleration_checks(ecm, True)
            st["joints"] = [Joint(model.joint_by_name(ecm, n)) for n in ("rear_left_wheel_joint", "rear_right_wheel_joint", "front_left_wheel_steer_joint", "front_right_wheel_steer_joint")]
            for j in st["joints"]:
                j.enable_position_check(ecm, True)
                j.enable_velocity_check(ecm, True)
            st["init"] = True
        k = int(info.iterations) - 1  # the ECM holds the state at the end of step k-1 == time k*dt
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
        jl, jr, kl, kr = st["joints"]
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
            # Precomputed speed map, read slightly ahead to compensate the ESC lag.
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
                # Rear-axle slip angle: zero for a car rolling without lateral slip.
                vb_y = -row[7] * math.sin(yaw) + row[8] * math.cos(yaw)
                v_rear_y = vb_y + rear_offset * row[12]
                st["max_slip"] = max(st["max_slip"], abs(math.atan2(v_rear_y, speed)))
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

    set_verbosity(1)
    fixture = TestFixture(str(world_path))
    fixture.on_pre_update(pre_update)
    fixture.finalize()
    server = fixture.server()
    chunk = 1000
    while not st["done"]:
        server.run(True, chunk, False)
    n = st["k"] + 1
    sim_wall = time.time() - wall_start

    out = Path(job["out_npz"])
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "trajectory_key": job["trajectory_key"],
        "status": st["status"],
        "failure_reason": st["reason"],
        "lap_completed": bool(st["lap_completed"]),
        "steps": n,
        "physics_step_ns": step_ns,
        "record_start_ns": int(round(settle_s / dt)) * step_ns,
        "drive_start_ns": st["drive_start_ns"],
        "lap_end_ns": st["lap_ns"],
        "simulated_s": n * dt,
        "wall_s": sim_wall,
        "real_time_factor": n * dt / max(sim_wall, 1e-9),
        "distance_m": float(st["s"]),
        "path_length_m": path.length,
        "laps": float(job.get("laps", 1.0)),
        "max_lateral_error_m": st["max_lat"],
        "max_roll_pitch_deg": math.degrees(st["max_rp"]),
        "max_rear_slip_deg": math.degrees(st["max_slip"]),
        "planned_duration_s": planned,
        "events": st["events"],
        "setup": {
            "direction": setup["direction"],
            "start_s": setup["start_s"],
            "lateral_offset_m": setup["lateral_offset"],
            "lookahead_gain_scale": setup["gain_scale"],
            "actuator_perturbation": setup["actuator_perturbation"],
            "initial_yaw_error_rad": setup["init_yaw_error"],
            "initial_lateral_error_m": setup["init_lateral_error"],
            "weave": setup["weave"],
            "v_ref_mps": setup["v_ref"],
            "plan": plan.description,
            "car_initial_pose": list(car_pose),
        },
        "job": {k2: v2 for k2, v2 in job.items() if k2 not in ("tmp_dir",)},
        "gz_partition": os.environ.get("GZ_PARTITION", ""),
    }
    np.savez_compressed(
        out,
        state=state[:n],
        ctrl=ctrl[:n],
        state_columns=np.array(STATE_COLUMNS),
        ctrl_columns=np.array(CTRL_COLUMNS),
        summary=np.array(json.dumps(summary)),
    )
    try:
        world_path.unlink()
    except OSError:
        pass
    return summary


def synthesize_job(job: dict[str, Any], summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Sensor synthesis + QA for every pending run of a finished trajectory."""
    import pickle

    from .qa import run_qa
    from .raycast import RayCaster
    from .sensors import Trajectory, body_motion, load_sensor_profiles, synthesize_run

    traj = Trajectory(Path(job["out_npz"]))
    track = load_track(job["track"])
    vcfg = load_vehicle_config()
    body = body_motion(traj, vcfg["body_dynamics"], job["trajectory_key"])
    caster = RayCaster(track.geometry)
    profiles = load_sensor_profiles()
    results = []
    for run in job.get("runs", []):
        t0 = time.time()
        profile = profiles[run["sensor"]]
        synth = synthesize_run(traj, track, run["sensor"], profile, job["trajectory_key"], body, caster)
        synth["body_params"] = body["params"]
        qa = run_qa(synth, summary, float(profile["imu"]["rate_hz"]), float(profile["lidar"]["rate_hz"]))
        out = Path(job["out_npz"]).parent / f"{run['run_key']}.pkl"
        with open(out, "wb") as fh:
            pickle.dump({"synth": synth, "qa": qa}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        results.append({"run_key": run["run_key"], "file": str(out), "qa_passed": qa["passed"], "synth_wall_s": time.time() - t0})
    return results


def physics_hash(job: dict[str, Any]) -> str:
    """Hash of everything that determines a trajectory (cache validity)."""
    import hashlib

    from .tracks import load_track_specs
    from .vehicle import PACKAGE_DIR

    motion = load_motion_profiles().get(job["motion"], {})
    payload = {
        "job": {k: job.get(k) for k in ("track", "motion", "seed", "direction", "laps", "v_ref", "track_overrides", "speed_override_mps", "calibration", "max_sim_s", "physical_obstacles")},
        "track": load_track_specs()[job["track"]],
        "motion": motion,
        "vehicle": (PACKAGE_DIR / "config" / "vehicle.yaml").read_text(encoding="utf-8"),
        "car_model": (PACKAGE_DIR / "config" / "rc_car_model.sdf").read_text(encoding="utf-8"),
        # Only the code that determines the physics: editing the sensor
        # synthesis or QA reuses the cached trajectories.
        "code": [inspect.getsource(f) for f in (seed_streams, quat_to_rpy, build_run_setup, physics_world, run_physics)]
        + [(PACKAGE_DIR / f).read_text(encoding="utf-8") for f in ("controller.py", "vehicle.py", "tracks.py", "geometry.py")],
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    job = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    out = Path(job["out_npz"])
    summary_path = out.with_suffix(".summary.json")
    physics_path = out.with_suffix(".physics.json")
    digest = physics_hash(job)
    cached = None
    if out.exists() and physics_path.exists():
        cached = json.loads(physics_path.read_text(encoding="utf-8"))
        if cached.get("physics_hash") != digest:
            cached = None  # configuration or code changed: simulate again
    if cached is not None:
        summary = cached
        summary["physics_cached"] = True
    else:
        summary = run_physics(job)
        summary["physics_hash"] = digest
        physics_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    summary["runs"] = []
    if summary["status"] == "completed" and job.get("runs"):
        summary["runs"] = synthesize_job(job, summary)
    summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: summary.get(k) for k in ("trajectory_key", "status", "failure_reason", "simulated_s", "wall_s", "real_time_factor")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
