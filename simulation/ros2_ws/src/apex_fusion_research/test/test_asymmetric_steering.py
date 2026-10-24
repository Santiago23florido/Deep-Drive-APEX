"""Unequal servo travel must survive the whole curvature command chain."""
import math

import numpy as np
import pytest

from apex_fusion_research.core.adaptive import AdaptiveShaper
from apex_fusion_research.core.plant_id import PlantIdentifier
from apex_fusion_research.core.race_config import RaceDriverConfig
from apex_fusion_research.core.race_driver import RaceDriverCore
from apex_fusion_research.core.reactive import ReactivePlanner


def config():
    cfg = RaceDriverConfig()
    cfg.vehicle.steer_max_deg = cfg.vehicle.steer_plan_deg = 12.5
    cfg.vehicle.steer_max_left_deg = cfg.vehicle.steer_plan_left_deg = 22.5
    cfg.vehicle.steer_max_right_deg = cfg.vehicle.steer_plan_right_deg = 12.5
    cfg.adaptive.sat_cmd_frac = 1.0
    return cfg


def test_bicycle_candidates_include_zero_and_both_reachable_extrema():
    cfg = config()
    planner = ReactivePlanner(cfg.vehicle, cfg.reactive)
    for candidates, count in ((planner.k1, 21), (planner.k2, 11)):
        assert len(candidates) == count
        assert np.all(np.diff(candidates) > 0)
        assert 0.0 in candidates
        assert candidates[0] == pytest.approx(-math.tan(math.radians(12.5)) / .3)
        assert candidates[-1] == pytest.approx(math.tan(math.radians(22.5)) / .3)


def test_inverse_and_low_confidence_do_not_reintroduce_the_symmetric_clamp():
    cfg = config()
    shaper = AdaptiveShaper(cfg)
    prior = PlantIdentifier(cfg).prior()
    assert prior.confidence == 0
    for target, expected in ((10, 22.5), (-10, -12.5)):
        shaper.reset()
        _, kappa = shaper.shape(0, target, .2, .2, prior, shaper.envelope(prior))
        assert math.degrees(math.atan(kappa * .3)) == pytest.approx(expected)


def test_driver_publication_limits_both_sides_even_with_large_inverse_requests():
    core = RaceDriverCore(config(), 1)
    for request, expected in ((100, 22.5), (-100, -12.5), (18, 18)):
        assert math.degrees(core.steer_command(math.radians(request))) == pytest.approx(expected)


def test_saturation_observations_are_normalized_to_their_side():
    cfg = config()
    ident = PlantIdentifier(cfg)
    # Repeated physical curvature at each configured extreme must represent
    # the same saturation fraction, rather than left travel shrinking right.
    for side in (True, False):
        ident.sat_seen.clear()
        k = cfg.vehicle.curvature_limit(side, planning=True)
        for _ in range(cfg.adaptive.sat_window):
            ident._saturation(k if side else -k * cfg.vehicle.right_ratio, 0)
        assert ident.kappa_sat == pytest.approx(cfg.vehicle.kappa_max)
