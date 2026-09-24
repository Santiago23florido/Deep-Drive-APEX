"""APEX car model for the in-process simulation and its actuator dynamics.

The Gazebo model is generated from the same xacro as the ROS simulation
(``rc_sim_description/urdf/rc_car.urdf.xacro``). Rendering sensors (camera,
GPU LiDAR) and the Gazebo IMU are removed because the dataset synthesizes the
measurements from the exact simulator state, and the joint-controller plugins
are removed because :class:`Actuators` applies the very same control laws
deterministically at every physics step:

* ESC: first-order lag + acceleration / braking rate limits, as in
  ``apex_gz_vehicle_bridge.py``, then an electronic differential that turns the
  speed into rear-wheel angular velocities (Gazebo ``JointController``
  semantics: the wheel joint follows the commanded velocity);
* servo: first-order lag + rate limit + asymmetric linkage ratio, Ackermann
  split and a PID torque on each knuckle identical to Gazebo's
  ``JointPositionController`` (gz::math::PID, same gains and limits).
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
from typing import Any

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
SIM_ROOT = PACKAGE_DIR.parents[1]
SHIPPED_MODEL = PACKAGE_DIR / "config" / "rc_car_model.sdf"


def load_vehicle_config(path: Path | None = None) -> dict[str, Any]:
    return yaml.safe_load((path or PACKAGE_DIR / "config" / "vehicle.yaml").read_text(encoding="utf-8"))


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _clean_env() -> dict[str, str]:
    return {"HOME": os.environ.get("HOME", "/tmp"), "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}


def generate_model_sdf(xacro_path: Path, out_path: Path) -> Path:
    """xacro -> URDF (ROS Jazzy xacro) -> SDF (``gz sdf -p``) -> strip sensors/plugins."""
    ros_setup = Path("/opt/ros/jazzy/setup.bash")
    if not ros_setup.exists():
        raise RuntimeError("xacro needs ROS 2 Jazzy (/opt/ros/jazzy)")
    urdf = subprocess.run(
        ["bash", "-c", f"source {ros_setup} && xacro {xacro_path} imu_gyro_noise_stddev_rps:=0.0 imu_accel_noise_stddev_mps2:=0.0"],
        check=True,
        capture_output=True,
        text=True,
        env=_clean_env(),
    ).stdout
    tmp_urdf = out_path.with_suffix(".urdf")
    tmp_urdf.parent.mkdir(parents=True, exist_ok=True)
    tmp_urdf.write_text(urdf, encoding="utf-8")
    gz = shutil.which("gz", path="/usr/bin:/bin") or "gz"
    sdf = subprocess.run([gz, "sdf", "-p", str(tmp_urdf)], check=True, capture_output=True, text=True, env=_clean_env()).stdout
    tmp_urdf.unlink(missing_ok=True)
    root = ET.fromstring(sdf)
    model = root.find("model")
    base = model.find("link[@name='base_link']")
    for sensor in list(base.findall("sensor")):
        base.remove(sensor)
    for plugin in list(model.findall("plugin")):
        model.remove(plugin)
    ET.indent(root, space="  ")
    header = f"<!-- Generated from {xacro_path.name} sha1={_sha1(xacro_path)}; sensors and plugins removed. -->\n"
    out_path.write_text('<?xml version="1.0" ?>\n' + header + ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
    return out_path


def model_sdf_path(cache_dir: Path | None = None) -> Path:
    """Return an up-to-date car SDF (regenerated if the xacro changed)."""
    cfg = load_vehicle_config()
    xacro = SIM_ROOT / cfg["model"]["xacro"]
    digest = _sha1(xacro)
    if SHIPPED_MODEL.exists() and f"sha1={digest}" in SHIPPED_MODEL.read_text(encoding="utf-8")[:400]:
        return SHIPPED_MODEL
    target = (cache_dir or SIM_ROOT / "data" / "multiscenario_pose" / "cache") / "vehicle" / f"rc_car_{digest[:12]}.sdf"
    if target.exists():
        return target
    try:
        return generate_model_sdf(xacro, target)
    except Exception as exc:  # pragma: no cover - depends on the host
        if SHIPPED_MODEL.exists():
            print(f"[vehicle] WARNING: cannot regenerate the car model ({exc}); using the shipped copy")
            return SHIPPED_MODEL
        raise


def car_model_element(pose_xyzrpy: tuple[float, ...], sdf_path: Path | None = None) -> ET.Element:
    root = ET.parse(str(sdf_path or model_sdf_path())).getroot()
    model = root.find("model")
    model.set("name", "rc_car")
    for old in model.findall("pose"):
        model.remove(old)
    pose = ET.Element("pose")
    pose.text = " ".join(f"{v:.6f}" for v in pose_xyzrpy)
    model.insert(0, pose)
    return model


# ------------------------------------------------------------------ actuators
def _first_order(current: float, target: float, tau: float, dt: float) -> float:
    return current + (1.0 - math.exp(-dt / max(tau, 1e-6))) * (target - current)


def _rate_limit(current: float, target: float, rate: float, dt: float) -> float:
    step = rate * dt
    delta = target - current
    if rate <= 0.0 or abs(delta) <= step:
        return target
    return current + math.copysign(step, delta)


class GzPid:
    """Replica of gz::math::PID::Update as used by JointPositionController."""

    def __init__(self, p: float, i: float, d: float, i_limit: float, cmd_limit: float) -> None:
        self.p, self.i, self.d = p, i, d
        self.i_limit = i_limit
        self.cmd_limit = cmd_limit
        self.i_err = 0.0
        self.p_err_last = 0.0

    def update(self, error: float, dt: float) -> float:
        if dt <= 0.0 or not math.isfinite(error):
            return 0.0
        self.i_err = min(self.i_limit, max(-self.i_limit, self.i_err + self.i * dt * error))
        d_err = (error - self.p_err_last) / dt
        self.p_err_last = error
        cmd = -self.p * error - self.i_err - self.d * d_err
        return min(self.cmd_limit, max(-self.cmd_limit, cmd))


class Actuators:
    """ESC + servo + Ackermann + electronic differential, stepped at 1 kHz."""

    def __init__(self, cfg: dict[str, Any], perturb: dict[str, float] | None = None) -> None:
        m, esc, servo = cfg["model"], cfg["esc"], cfg["servo"]
        perturb = perturb or {}
        self.wheelbase = float(m["wheelbase_m"])
        self.half_track = 0.5 * float(m["track_width_m"])
        self.wheel_radius = float(m["wheel_radius_m"])
        self.steer_limit_deg = float(m["steering_limit_deg"])
        self.esc_tau = float(esc["response_tau_s"]) * perturb.get("esc_tau_scale", 1.0)
        self.accel_limit = float(esc["accel_limit_mps2"])
        self.decel_limit = float(esc["decel_limit_mps2"])
        self.top_speed = float(esc["top_speed_mps"])
        self.min_effective = float(esc["min_effective_speed_mps"])
        self.servo_tau = float(servo["response_tau_s"]) * perturb.get("servo_tau_scale", 1.0)
        self.servo_rate = float(servo["rate_limit_deg_per_s"])
        self.left_ratio = float(servo["left_ratio"])
        self.right_ratio = float(servo["right_ratio"])
        self.pids = [
            GzPid(servo["pid_p"], servo["pid_i"], servo["pid_d"], servo["pid_i_limit"], servo["torque_limit_nm"]) for _ in range(2)
        ]
        self.speed = 0.0  # applied ESC speed [m/s]
        self.steer_deg = 0.0  # applied servo angle (centre wheel) [deg]
        self.speed_cmd = 0.0
        self.steer_cmd_deg = 0.0

    def set_command(self, speed_mps: float, steer_deg: float) -> None:
        speed = min(self.top_speed, max(0.0, float(speed_mps)))
        if 1e-6 < speed < self.min_effective:
            speed = self.min_effective
        self.speed_cmd = speed
        steer = max(-self.steer_limit_deg, min(self.steer_limit_deg, float(steer_deg)))
        steer *= self.left_ratio if steer >= 0.0 else self.right_ratio
        self.steer_cmd_deg = steer

    def step(self, dt: float) -> None:
        after = _first_order(self.speed, self.speed_cmd, self.esc_tau, dt)
        rate = self.accel_limit if after >= self.speed else self.decel_limit
        self.speed = _rate_limit(self.speed, after, rate, dt)
        if self.speed_cmd <= 1e-6 and abs(self.speed) < 1e-3:
            self.speed = 0.0
        after_s = _first_order(self.steer_deg, self.steer_cmd_deg, self.servo_tau, dt)
        self.steer_deg = _rate_limit(self.steer_deg, after_s, self.servo_rate, dt)
        self.steer_deg = max(-self.steer_limit_deg, min(self.steer_limit_deg, self.steer_deg))

    def knuckle_targets(self) -> tuple[float, float]:
        delta = math.radians(self.steer_deg)
        t = math.tan(delta)
        if abs(t) < 1e-6:
            return 0.0, 0.0
        radius = self.wheelbase / t
        return math.atan(self.wheelbase / (radius - self.half_track)), math.atan(self.wheelbase / (radius + self.half_track))

    def wheel_omegas(self) -> tuple[float, float]:
        delta = math.radians(self.steer_deg)
        yaw_rate = self.speed / self.wheelbase * math.tan(delta)
        return (self.speed - yaw_rate * self.half_track) / self.wheel_radius, (self.speed + yaw_rate * self.half_track) / self.wheel_radius

    def knuckle_torques(self, positions: tuple[float, float], dt: float) -> tuple[float, float]:
        targets = self.knuckle_targets()
        return tuple(pid.update(pos - tgt, dt) for pid, pos, tgt in zip(self.pids, positions, targets))  # type: ignore[return-value]
