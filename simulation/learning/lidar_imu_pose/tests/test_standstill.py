"""The learned odometry must not walk away while the car is parked.

Two levels. The first is the detector's state machine, which is pure
arithmetic. The second is the thing that actually matters: a parked stream
through the real estimator, with the real checkpoint, asserting that three
minutes of standing still cost millimetres instead of metres -- and that not one
driving revolution of the validated real2sim run is held.

Needs the v5 checkpoint and the replay fixture (skipped otherwise); CPU only.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
SIM = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SIM / "tools"))
sys.path.insert(0, str(SIM / "tools" / "analysis"))

from standstill import StandstillConfig, StandstillHold  # noqa: E402

# v5 on purpose: the inference-time hold has to work on the model that has the
# bug, and the tests below are what keep that true. The cured weights (v7,
# README section 16) are measured separately, with the hold off.
CKPT = SIM / "learning" / "outputs" / "real2sim_v5_10hz" / "hybrid_submap_v4" / "best_model.pt"
CKPT_V7 = SIM / "learning" / "outputs" / "real2sim_v7e_standstill" / "hybrid_submap_v4" / "best_model.pt"
FIXTURE = SIM.parent / "real_vehicle" / "data" / "replay" / "final_s13_sensors.npz"
STILL = dict(standstill=True, step_m=0.001, yaw_rad=0.0, eig_ratio=0.3)


# --------------------------------------------------------------- the detector
def test_entering_needs_the_icp_flag_confirmed():
    h = StandstillHold(StandstillConfig(confirm_scans=3))
    assert [h.update(**STILL) for _ in range(4)] == [False, False, True, True]
    assert h.entries == 1


def test_the_flag_alone_does_not_enter_without_the_geometry():
    h = StandstillHold()
    assert not any(h.update(standstill=True, step_m=0.05, yaw_rad=0.0, eig_ratio=0.3) for _ in range(20))


def test_staying_does_not_need_the_flag():
    """The vibration term of the ICP's flag cannot tell parked from driving, so
    it may not be what keeps a confirmed hold -- only geometry may."""
    h = StandstillHold(StandstillConfig(confirm_scans=2))
    assert [h.update(**STILL) for _ in range(2)] == [False, True]
    assert h.update(standstill=False, step_m=0.001, yaw_rad=0.0, eig_ratio=0.3), "a flag flicker released the hold"
    assert h.released == 0


@pytest.mark.parametrize("bad", ({"step_m": 0.05}, {"yaw_rad": 0.1}, {"eig_ratio": 0.001}))
def test_geometry_releases_immediately(bad: dict):
    h = StandstillHold(StandstillConfig(confirm_scans=2))
    for _ in range(3):
        h.update(**STILL)
    assert h.holding
    assert not h.update(**{**STILL, **bad})
    assert h.released == 1 and h.run == 0


def test_the_slowest_commanded_speed_releases_the_hold():
    """At the driver's speed floor one revolution is well above the gate."""
    cfg = StandstillConfig()
    v_min_mps, revolution_s = 0.35, 0.1  # race_driver.yaml, 10 Hz
    assert v_min_mps * revolution_s > 5 * cfg.max_step_m
    # And so is the slowest speed the ESC can deliver at all.
    assert 0.07 * revolution_s > cfg.max_step_m


def test_disabled_is_inert():
    h = StandstillHold(StandstillConfig(enable=False))
    assert not any(h.update(**STILL) for _ in range(10))
    assert h.status() == {"holding": 0, "entries": 0, "held": 0, "released": 0}


# ------------------------------------------------------------ the real thing
needs_data = pytest.mark.skipif(not (CKPT.exists() and FIXTURE.exists()),
                                reason="needs the v5 checkpoint and the replay fixture")


def _replay(mode: str, hold: bool, seconds: float = 180.0, imu: str = "real", checkpoint: Path = CKPT):
    import standstill_drift as sd  # noqa: PLC0415

    args = type("A", (), dict(seconds=seconds, imu=imu, static_s=4.4, fixture=str(FIXTURE), checkpoint=str(checkpoint),
                              profile="APEX_real", confirm_scans=StandstillConfig.confirm_scans,
                              max_step_m=StandstillConfig.max_step_m, seed=7))()
    prof = sd.load_sensor_profiles(sd.SENSORS_YAML)["APEX_real"]
    cal = sd.LidarCalibration.from_profile(prof["lidar"])
    scans, imus = sd.parked_stream(args, np.load(args.fixture, allow_pickle=True), cal) if mode == "parked" \
        else sd.recorded_stream(np.load(args.fixture, allow_pickle=True))
    live = sd.build_estimator(args, prof, hold)
    rows, j = [], 0
    for stamp, r, st in scans:
        end = stamp + int(round(st * 1e9))
        while j < len(imus) and imus[j][0] <= end + int(2e7):
            s = imus[j][1]
            live.add_imu(imus[j][0], np.asarray(s[3:], dtype=np.float64), np.asarray(s[:3], dtype=np.float64))
            j += 1
        live.add_scan(stamp, r, st)
        for est in live.process(end):
            row = sd.row(est, stamp * 1e-9)
            prediction = live.predict()
            row["pred_speed"] = float(np.linalg.norm(prediction[2]))
            rows.append(row)
    return live, {k: np.array([r[k] for r in rows]) for k in rows[0]}


@needs_data
def test_confirmed_standstill_also_holds_the_fast_prediction():
    """The controller must not integrate IMU tilt between held scan poses.

    On the parked car this sawtooth accumulated ten fictitious metres even
    though the scan-stamped pose stayed within millimetres of the origin.
    """
    live, _ = _replay("parked", hold=True, seconds=5.0, imu="real")
    assert live.hold.holding
    pose_at_scan = live.pose.copy()
    start = live._imu_t[-1]
    for i in range(1, 21):
        live.add_imu(start + live.imu_delay_ns + i * 10_000_000,
                     np.array([0.0, 0.0, 0.01]), np.array([-0.7, -0.2, 9.8]))
        stamp, pose, velocity, rate = live.predict()
        assert stamp == start + i * 10_000_000, "prediction stopped publishing fresh IMU stamps"
        np.testing.assert_array_equal(pose, pose_at_scan)
        np.testing.assert_array_equal(velocity, np.zeros(2))
        assert rate == 0.0


@needs_data
def test_three_minutes_parked_cost_millimetres():
    live, a = _replay("parked", hold=True)
    drift = math.hypot(a["x"][-1], a["y"][-1])
    assert drift < 0.02, f"{drift:.3f} m of drift in {a['t'][-1]:.0f} s parked"
    assert a["hold"].mean() > 0.95, "the hold barely engaged on a parked stream"
    assert live.hold.released <= 2, live.hold.status()
    # The velocity state is the one thing a parked car knows exactly, and the
    # accelerometer bias must not ramp into its clamp (it reached -1.0 on the car).
    assert abs(a["vx"][-1]) < 1e-9 and abs(a["vy"][-1]) < 1e-9
    assert abs(a["bias_ay"][-1]) < 0.5


@needs_data
def test_without_the_hold_it_walks_metres():
    """The bug this guards against, in the same harness: it must still be there."""
    _, a = _replay("parked", hold=False, seconds=60.0)
    assert math.hypot(a["x"][-1], a["y"][-1]) > 1.0, "the parked stream no longer reproduces the drift"


@needs_data
def test_no_driving_revolution_is_held():
    live, a = _replay("run", hold=True)
    held = a["hold"] > 0.5
    step = np.hypot(a["icp_dx"], a["icp_dy"])
    assert held.any(), "the stationary prefix of the run was not held"
    assert step[held].max() <= StandstillConfig.max_step_m + 1e-9
    # Every held revolution is one the classical ICP also called a standstill,
    # bar the flag flicker the stay condition exists to absorb.
    flagged = a["still"][held] > 0.5
    assert flagged.mean() > 0.95, f"{100 * (1 - flagged.mean()):.1f} % of held revolutions were not flagged still"
    assert live.hold.entries <= 3, live.hold.status()
    assert np.all(a["pred_speed"][held] == 0.0), "fast prediction moved during the held prefix"
    assert a["pred_speed"][~held & (step > 0.03)].max() > 0.1, "prediction failed to resume during recorded motion"


# ------------------------------------------------------- the cured weights
@pytest.mark.skipif(not (CKPT_V7.exists() and FIXTURE.exists()), reason="needs the v7 checkpoint and the replay fixture")
@pytest.mark.parametrize("imu", ("sim", "real"))
def test_v7_stands_still_without_any_help(imu: str):
    """v7 must not need the hold: three parked minutes, the model alone.

    The thresholds are the measurement with room to spare, not the measurement
    itself (0.141 m simulated, 0.132 m on the car's static signature, against
    v5's 21.3 and 23.8 m). ``--imu real`` is the harder case and the one the car
    presents: a gyro that reports exactly 0.000 rad/s, which only the
    ``APEX_real_az`` runs of the training set contain.
    """
    _, a = _replay("parked", hold=False, imu=imu, checkpoint=CKPT_V7)
    drift = math.hypot(a["x"][-1], a["y"][-1])
    assert drift < 0.30, f"{drift:.3f} m of drift in {a['t'][-1]:.0f} s parked, with no hold"
    assert not a["hold"].any(), "the hold was supposed to be off"
    # And the accelerometer bias must not ramp into its clamp, which is what v5
    # did (-1.00000) and what made the drift grow with every minute parked.
    assert abs(a["bias_ay"][-1]) < 0.8


@pytest.mark.skipif(not (CKPT_V7.exists() and FIXTURE.exists()), reason="needs the v7 checkpoint and the replay fixture")
def test_v7_still_drives_the_validated_run():
    """The driving half of the trade: the recorded run, end to end."""
    _, a = _replay("run", hold=False, checkpoint=CKPT_V7)
    step = np.hypot(a["dx"], a["dy"])
    assert step.mean() > 0.10, "the run did not drive"
    assert abs(float(np.hypot(a["x"][-1], a["y"][-1])) - 1.77) < 0.6, "the trajectory moved away from the measured one"
