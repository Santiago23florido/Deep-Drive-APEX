"""Unidentified gain/offset must not suppress forward commands after a reverse."""
import numpy as np
import pytest

from apex_race.core.adaptive import AdaptiveShaper
from apex_race.core.plant_id import PlantIdentifier
from apex_race.core.race_config import RaceDriverConfig


def reverse_observation(identifier):
    # A low-speed observation similar to the recorded degenerate fit. One
    # regressor cannot distinguish a gain change from a constant offset.
    identifier.long.update(np.array([-.224, 1.]), -.03, .05**2)


def test_single_reverse_observation_preserves_original_forward_response():
    cfg = RaceDriverConfig()
    identifier = PlantIdentifier(cfg)
    prior = identifier.model()
    reverse_observation(identifier)
    assert identifier.long.x[1] > .1
    assert identifier.conf_long == 0.
    model = identifier.model()
    before = AdaptiveShaper(cfg)
    after = AdaptiveShaper(cfg)
    for index in range(20):
        t = index * .02
        a = before.shape(t, .2, .15, .1, prior, before.envelope(prior))
        b = after.shape(t, .2, .15, .1, model, after.envelope(model))
        assert b == pytest.approx(a)
    assert identifier.status()['v_offset_estimated'] > .1
    assert identifier.status()['v_offset'] == 0.
    assert not identifier.status()['longitudinal_applied']


def test_repeated_one_level_measurements_do_not_make_the_offset_identifiable():
    identifier = PlantIdentifier(RaceDriverConfig())
    for _ in range(100):
        reverse_observation(identifier)
    assert identifier.long.n >= identifier.a.n_long
    assert not identifier.longitudinal_ready
    assert identifier.model().speed_in(.15) == pytest.approx(.15)


def identified_speed_map():
    identifier = PlantIdentifier(RaceDriverConfig())
    for _ in range(100):
        for command in (.3, .8, 1.8):
            identifier.long.update(np.array([command, 1.]), .85 * command + .04, .05**2)
    return identifier


def test_identifiable_speed_map_is_applied_independently_of_lateral_confidence():
    identifier = identified_speed_map()
    assert identifier.confidence == 0.
    assert identifier.longitudinal_ready
    model = identifier.model()
    assert model.v_gain == pytest.approx(.85, abs=.01)
    assert model.v_offset == pytest.approx(.04, abs=.01)
    assert model.speed_in(.15) == pytest.approx((.15 - .04) / .85, abs=.01)


def test_inconsistent_residuals_keep_the_prior_without_disabling_launch_learning():
    identifier = identified_speed_map()
    identifier.long.chi = identifier.a.chi_max + 1.
    identifier.dead, identifier.dead_rev = .12, .24
    assert not identifier.longitudinal_ready
    model = identifier.model()
    assert (model.v_gain, model.v_offset) == (1., 0.)
    assert (model.dead_mps, model.dead_rev_mps) == (.12, .24)
