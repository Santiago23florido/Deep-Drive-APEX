"""The estimator must recover a plant it was not told, from the odometry only."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from _plant_stub import SIGMA, Plant, drive
from apex_fusion_research.core.plant_id import ARM, GAIN_L, GAIN_R, TRIM, PlantIdentifier, curvature_measurement, lever_measurement
from apex_fusion_research.core.race_config import RaceDriverConfig


def weave(k: float = 0.5, period: float = 6.0, v: float = 1.5):
    return lambda t: (k * math.sin(2.0 * math.pi * t / period), v)


def _increment(kappa: float, dyaw: float, arm: float) -> tuple[float, float]:
    return math.sin(dyaw) / kappa + arm * (math.cos(dyaw) - 1.0), (1.0 - math.cos(dyaw)) / kappa + arm * math.sin(dyaw)


@pytest.mark.parametrize("arm", [0.05, 0.15, 0.25])
@pytest.mark.parametrize("kappa", [-0.95, -0.05, 0.05, 0.95])
@pytest.mark.parametrize("dyaw", [0.02, 0.3, 0.5])
def test_the_two_measurement_equations_are_exact(arm, kappa, dyaw):
    """The algebra, not the filter: a slip here would be silent everywhere."""
    dx, dy = _increment(kappa, dyaw, arm)
    y, regressor = lever_measurement(dx, dy, dyaw)
    assert y / regressor == pytest.approx(arm, abs=1e-12)
    assert curvature_measurement(dx, dyaw, arm) == pytest.approx(kappa, abs=1e-12)


def test_arm_is_recovered_through_the_odometry_noise():
    cfg = RaceDriverConfig()
    ident = PlantIdentifier(cfg)
    plant = Plant(arm=0.21)  # the LiDAR 6 cm further forward than the car believes
    drive(ident, plant, 60.0, weave())
    sigma = math.sqrt(float(ident.lever.p[ARM, ARM]))
    assert ident.lever.x[ARM] == pytest.approx(0.21, abs=0.005)
    assert sigma < 0.01
    assert ident.conf_lever > 0.7


def test_the_curvature_map_is_recovered_including_its_asymmetry():
    cfg = RaceDriverConfig()
    ident = PlantIdentifier(cfg)
    plant = Plant(gain_l=1.25, gain_r=0.82, trim=0.06, arm=0.15)
    drive(ident, plant, 90.0, weave())
    model = ident.model()
    # The gain of each side is only meaningful together with the speed
    # dependence it was identified at, and that combination is what the
    # inverse uses, so that is what is asserted.
    assert model.gain_at(0.4, 1.5) == pytest.approx(1.25, abs=0.03)
    assert model.gain_at(-0.4, 1.5) == pytest.approx(0.82, abs=0.03)
    assert model.trim == pytest.approx(0.06, abs=0.025)
    assert ident.conf_steer > 0.7


def test_the_speed_dependence_is_recovered_when_the_speed_varies():
    """A map identified at one speed must still hold at the race speed, which
    is the whole point: lap 1 is driven at 1.5 m/s and the race at 3.5."""
    cfg = RaceDriverConfig()
    ident = PlantIdentifier(cfg)
    plant = Plant(gain_l=1.1, gain_r=1.1, slip=-0.012)
    drive(ident, plant, 120.0, lambda t: (0.5 * math.sin(2.0 * math.pi * t / 6.0), 1.6 + 1.2 * math.sin(0.13 * t)))
    model = ident.model()
    assert model.gain_at(0.4, 1.5) == pytest.approx(1.1 - 0.012 * 1.5 ** 2, abs=0.03)
    assert model.gain_at(0.4, 3.5) == pytest.approx(1.1 - 0.012 * 3.5 ** 2, abs=0.04)


def test_a_speed_dependence_is_never_invented_from_one_speed():
    """At a single speed the slip column and the gain column are the same
    regressor. The honest answer is to leave the slip at its prior, and the
    error that remains when extrapolating is then bounded by its sigma."""
    cfg = RaceDriverConfig()
    ident = PlantIdentifier(cfg)
    drive(ident, Plant(gain_l=1.1, gain_r=1.1), 90.0, weave())
    model = ident.model()
    assert model.slip == pytest.approx(0.0, abs=1e-6)
    worst = cfg.adaptive.slip_sigma * (cfg.race.v_max_mps ** 2 - model.v_ref_mps ** 2)
    assert worst < 0.08, f"extrapolating to v_max could be wrong by {worst:.3f}"


def test_the_lag_bank_finds_the_lag_of_the_plant():
    cfg = RaceDriverConfig()
    ident = PlantIdentifier(cfg)
    drive(ident, Plant(tau_k=0.18), 90.0, weave(period=4.0))
    assert ident.tau_k[ident.i_steer] == pytest.approx(0.18, abs=0.03)


def test_an_unexercised_side_keeps_the_nominal_value():
    """A track that only turns one way must leave the other gain alone, at the
    datasheet value the unmodified driver uses, with its full uncertainty."""
    cfg = RaceDriverConfig()
    ident = PlantIdentifier(cfg)
    drive(ident, Plant(gain_l=1.25, gain_r=0.82), 40.0, lambda t: (0.45, 1.5))
    assert ident.steer.x[GAIN_R] == pytest.approx(cfg.vehicle.right_ratio, abs=1e-6)
    assert ident.model().conf_right == pytest.approx(0.0, abs=1e-6)
    assert ident.steer.x[GAIN_L] == pytest.approx(1.25, abs=0.06)


def test_every_estimate_stays_inside_its_box_and_never_loses_certainty():
    cfg = RaceDriverConfig()
    ident = PlantIdentifier(cfg)
    rng = np.random.default_rng(3)
    plant = Plant(gain_l=3.0, gain_r=0.1, trim=1.0, arm=0.9)  # absurd: the boxes must hold
    drive(ident, plant, 40.0, weave(k=0.9), rng=rng)
    for block in (ident.lever, ident.steer, ident.long):
        assert np.all(block.x >= block.box[:, 0] - 1e-9)
        assert np.all(block.x <= block.box[:, 1] + 1e-9)
        assert np.all(np.diag(block.p) <= block.p0 + 1e-12)


def test_the_gates_refuse_without_moving_the_estimate():
    cfg = RaceDriverConfig()
    ident = PlantIdentifier(cfg)
    before = ident.steer.x.copy()
    # Too slow to measure a curvature, and the interval of a dropped revolution.
    drive(ident, Plant(gain_l=1.3), 10.0, lambda t: (0.5, 0.2))
    assert ident.rejected["slow"] > 0
    ident.on_estimate(100.0, (0.0, 0.0, 0.0), SIGMA)
    ident.on_estimate(100.4, (0.4, 0.0, 0.2), SIGMA)
    assert ident.rejected["dt"] > 0
    ident.on_estimate(101.0, (1.0, 0.0, 0.3), (0.05, 0.05, 0.05))
    assert ident.rejected["sigma"] > 0 or ident.rejected["dt"] > 0
    assert np.allclose(ident.steer.x, before, atol=1e-9)


def test_a_reversed_steering_is_refused_not_fitted():
    cfg = RaceDriverConfig()
    ident = PlantIdentifier(cfg)
    drive(ident, Plant(sign=-1.0), 30.0, weave())
    assert ident.sign_fault
    assert ident.confidence == 0.0
    assert ident.model() == ident.prior()


def test_parked_pose_noise_cannot_vote_for_reversed_steering():
    ident = PlantIdentifier(RaceDriverConfig())
    for i in range(100):
        t = i * 0.1
        ident.push_command(t, 0.4, 0.7)
        ident.on_estimate(t, (i * 0.0002, 0.0, -i * 0.00001), SIGMA)
    assert ident.rejected["slow"] > 0
    assert not ident.sign_fault
    assert sum(ident.sign_votes) == 0
    assert ident.steer.n == 0


def test_the_watchdog_restores_the_datasheet_when_it_predicts_better():
    cfg = RaceDriverConfig()
    ident = PlantIdentifier(cfg)
    moved = Plant(gain_l=1.5, gain_r=1.5)
    drive(ident, moved, 40.0, weave())
    assert ident.steer.x[GAIN_L] > 1.2
    nominal = Plant(gain_l=1.0, gain_r=cfg.vehicle.right_ratio)  # the plant is now the prior
    nominal.yaw = moved.yaw
    ident.prev = None
    drive(ident, nominal, 60.0, weave())
    assert ident.trips >= 1


def test_the_speed_map_and_the_start_threshold_are_found():
    cfg = RaceDriverConfig()
    ident = PlantIdentifier(cfg)
    plant = Plant(v_gain=0.82, dead=0.18, arm=0.15)
    # Stand still under a small command (the threshold is above it), then drive.
    drive(ident, plant, 6.0, lambda t: (0.0, 0.10))
    assert ident.dead >= cfg.vehicle.min_effective_mps
    # A constant speed cannot separate the gain from the offset (the two
    # regressors are collinear), and a race varies its speed anyway.
    drive(ident, plant, 90.0, lambda t: (0.3 * math.sin(t), 1.6 + 1.0 * math.sin(0.11 * t)))
    assert ident.long.x[0] == pytest.approx(0.82, abs=0.06)
    assert ident.conf_long > 0.6


def test_one_update_costs_what_the_pi_can_afford():
    cfg = RaceDriverConfig()
    ident = PlantIdentifier(cfg)
    plant = Plant()
    drive(ident, plant, 5.0, weave())
    poses = [(0.15 * i, 0.001 * i, 0.02 * i) for i in range(200)]
    t0 = time.perf_counter()
    for i, pose in enumerate(poses):
        ident.push_command(100.0 + 0.1 * i, 0.3, 1.5)
        ident.on_estimate(100.0 + 0.1 * i, pose, SIGMA)
    per_update_us = 1e6 * (time.perf_counter() - t0) / len(poses)
    assert per_update_us < 400.0, f"{per_update_us:.0f} us per estimate"
