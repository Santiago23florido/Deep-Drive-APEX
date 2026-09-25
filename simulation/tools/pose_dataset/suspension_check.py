"""Measure the sprung-body response of the Gazebo car (real2sim suspension).

Drives the car of ``rc_car.urdf.xacro`` (``suspension:=true``) in-process
with the dataset actuators on a flat ground: a steady circle (roll per g of
lateral acceleration), release to straight driving (damped roll oscillation:
frequency and damping ratio) and a full brake (pitch per g of longitudinal
acceleration). Prints a JSON summary. Needs gz-sim 8 Python bindings, so run
it in a clean environment:

    env -i HOME=$HOME PATH=/usr/bin:/bin learning/.venv/bin/python -m pose_dataset.suspension_check
(from simulation/tools). Targets: vehicle.yaml body_dynamics.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .geometry import KIND_GROUND, Box, WorldGeometry, world_sdf, write_xml
from .vehicle import SIM_ROOT, Actuators, car_model_element, load_vehicle_config


def suspended_car_sdf(out: Path) -> Path:
    xacro = SIM_ROOT / "ros2_ws" / "src" / "rc_sim_description" / "urdf" / "rc_car.urdf.xacro"
    env = {"HOME": str(Path.home()), "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
    urdf = subprocess.run(["bash", "-c", f"source /opt/ros/jazzy/setup.bash && xacro {xacro} suspension:=true camera:=false"],
                          check=True, capture_output=True, text=True, env=env).stdout
    tmp = out.with_suffix(".urdf")
    tmp.write_text(urdf, encoding="utf-8")
    sdf = subprocess.run(["gz", "sdf", "-p", str(tmp)], check=True, capture_output=True, text=True, env=env).stdout
    root = ET.fromstring(sdf)
    model = root.find("model")
    for link in model.findall("link"):
        for s in list(link.findall("sensor")):
            link.remove(s)
    for p in list(model.findall("plugin")):
        model.remove(p)
    out.write_text('<?xml version="1.0" ?>\n' + ET.tostring(root, encoding="unicode"), encoding="utf-8")
    return out


def measure() -> dict:
    from gz.common5 import set_verbosity
    import gz.math7  # noqa: F401
    from gz.sim8 import Joint, Link, Model, TestFixture, World, world_entity

    tmp = Path(tempfile.mkdtemp(prefix="susp_"))
    car = suspended_car_sdf(tmp / "car.sdf")
    geom = WorldGeometry()
    geom.boxes.append(Box((0.0, 0.0, -0.05), (80.0, 80.0, 0.1), 0.0, KIND_GROUND, "ground"))
    world = tmp / "world.sdf"
    write_xml(world_sdf(geom, ground_mu=1.0, extra_models=[car_model_element((0, 0, 0.003, 0, 0, 0), car)], real_time_factor=0.0), world)
    act = Actuators(load_vehicle_config())
    dt = 0.001
    st: dict = {"init": False}
    log: list = []

    def command(t: float) -> tuple[float, float]:
        if t < 1.0:
            return 0.0, 0.0
        if t < 9.0:
            return 2.0, 10.0  # steady circle
        if t < 12.0:
            return 2.0, 0.0  # release: roll oscillation
        return 0.0, 0.0  # full brake

    def pre(info, ecm):  # noqa: ANN001
        if not st["init"]:
            mdl = Model(World(world_entity(ecm)).model_by_name(ecm, "rc_car"))
            st["base"] = Link(mdl.link_by_name(ecm, "base_link"))
            st["base"].enable_acceleration_checks(ecm, True)
            names = ("rear_left_wheel_joint", "rear_right_wheel_joint", "front_left_wheel_steer_joint", "front_right_wheel_steer_joint",
                     "suspension_roll_joint", "suspension_pitch_joint")
            st["j"] = [Joint(mdl.joint_by_name(ecm, n)) for n in names]
            for j in st["j"]:
                j.enable_position_check(ecm, True)
            st["init"] = True
        k = int(info.iterations) - 1
        t = k * dt
        q = st["base"].world_pose(ecm).rot()
        a = st["base"].world_linear_acceleration(ecm)
        yaw = math.atan2(2 * (q.w() * q.z() + q.x() * q.y()), 1 - 2 * (q.y() ** 2 + q.z() ** 2))
        ax = ay = 0.0
        if a is not None:
            ax = a.x() * math.cos(yaw) + a.y() * math.sin(yaw)
            ay = -a.x() * math.sin(yaw) + a.y() * math.cos(yaw)
        jl, jr, kl, kr, jroll, jpitch = st["j"]
        if k % 20 == 0:
            act.set_command(*command(t))
        act.step(dt)
        ol, orr = act.wheel_omegas()
        jl.set_velocity(ecm, [ol])
        jr.set_velocity(ecm, [orr])
        tl, tr = act.knuckle_torques(((kl.position(ecm) or [0.0])[0], (kr.position(ecm) or [0.0])[0]), dt)
        kl.set_force(ecm, [tl])
        kr.set_force(ecm, [tr])
        log.append((t, ax, ay, (jroll.position(ecm) or [0.0])[0], (jpitch.position(ecm) or [0.0])[0]))

    set_verbosity(1)
    fx = TestFixture(str(world))
    fx.on_pre_update(pre)
    fx.finalize()
    srv = fx.server()
    for _ in range(14):
        srv.run(True, 1000, False)
    t, ax, ay, roll, pitch = np.array(log).T
    k = np.ones(50) / 50
    ay_s, ax_s = np.convolve(ay, k, "same"), np.convolve(ax, k, "same")
    circle = (t > 5.0) & (t < 8.5)
    roll_gain = math.degrees(roll[circle].mean()) / (ay_s[circle].mean() / 9.80665)
    rest = pitch[(t > 0.5) & (t < 0.95)].mean()
    brake = (t > 12.3) & (t < 12.75)
    pitch_gain = math.degrees(np.median(pitch[brake]) - rest) / (np.median(ax_s[brake]) / 9.80665)
    seg = (t > 9.0) & (t < 10.5)
    r = roll[seg] - roll[(t > 10.5) & (t < 11.5)].mean()
    zc = t[seg][1:][np.diff(np.sign(r)) != 0]
    f_d = 1.0 / (2.0 * np.median(np.diff(zc))) if len(zc) >= 3 else float("nan")
    peaks = np.abs(r[np.r_[False, (np.abs(r[1:-1]) > np.abs(r[:-2])) & (np.abs(r[1:-1]) >= np.abs(r[2:])), False]])
    zeta = float("nan")
    if len(peaks) >= 2:
        dec = math.log(peaks[0] / peaks[1])  # half-cycle decrement
        zeta = dec / math.sqrt(math.pi ** 2 + dec ** 2)
    f_n = f_d / math.sqrt(1.0 - zeta ** 2) if np.isfinite(zeta) and zeta < 1 else float("nan")
    return {"roll_deg_per_g": roll_gain, "pitch_deg_per_g": abs(pitch_gain), "pitch_at_rest_deg": math.degrees(rest),
            "damped_freq_hz": f_d, "natural_freq_hz": f_n, "damping_ratio": zeta,
            "outward_roll": bool(roll[circle].mean() * ay_s[circle].mean() > 0)}


if __name__ == "__main__":
    json.dump(measure(), sys.stdout, indent=1)
    print()
