import math

import numpy as np

from apex_fusion_research.core.lidar_noise import (
    OUTCOME_DROPOUT,
    OUTCOME_HIT,
    OUTCOME_RANDOM,
    OUTCOME_SHORT,
    LidarNoiseConfig,
    LidarNoiseModel,
)

N_BEAMS = 360
INC = 2.0 * math.pi / N_BEAMS


def circle_scan(radius=3.0):
    """Sensor at the centre of a circular room: every beam hits at normal incidence."""
    return np.full(N_BEAMS, radius)


def run(cfg, scan, repeats=200):
    model = LidarNoiseModel(cfg)
    results = [model.apply(scan, -math.pi, INC, 0.05, 12.0) for _ in range(repeats)]
    return model, results


def test_ideal_config_is_identity():
    scan = circle_scan()
    _, results = run(LidarNoiseConfig(), scan, repeats=3)
    for res in results:
        np.testing.assert_array_equal(res.ranges, scan)
        assert np.all(res.outcome == OUTCOME_HIT)


def test_gaussian_sigma_matches_heteroscedastic_law():
    cfg = LidarNoiseConfig(range_sigma_const_m=0.004, range_sigma_prop=0.01)
    radius = 3.0
    _, results = run(cfg, circle_scan(radius))
    residual = np.concatenate([r.ranges - radius for r in results])
    expected = math.hypot(0.004, 0.01 * radius)
    assert abs(residual.mean()) < 3 * expected / math.sqrt(residual.size) + 1e-4
    assert abs(residual.std() / expected - 1.0) < 0.02


def test_normal_incidence_is_detected_on_circle():
    _, results = run(LidarNoiseConfig(), circle_scan(), repeats=1)
    np.testing.assert_allclose(results[0].cos_incidence, 1.0, atol=1e-6)


def test_outcome_probabilities():
    cfg = LidarNoiseConfig(dropout_prob_base=0.05, short_prob=0.03, random_prob=0.02)
    _, results = run(cfg, circle_scan(), repeats=400)
    outcome = np.concatenate([r.outcome for r in results])
    n = outcome.size
    p_drop = np.mean(outcome == OUTCOME_DROPOUT)
    p_short = np.mean(outcome == OUTCOME_SHORT)
    p_rand = np.mean(outcome == OUTCOME_RANDOM)
    assert abs(p_drop - 0.05) < 4 * math.sqrt(0.05 * 0.95 / n)
    # SHORT/RANDOM are conditional on a return (no dropout).
    assert abs(p_short - 0.95 * 0.03) < 4 * math.sqrt(0.03 / n)
    assert abs(p_rand - 0.95 * 0.02) < 4 * math.sqrt(0.02 / n)
    for res in results:
        assert np.all(np.isinf(res.ranges[res.outcome == OUTCOME_DROPOUT]))
        short = res.ranges[res.outcome == OUTCOME_SHORT]
        assert np.all((short >= 0.05) & (short <= 3.0 + 1e-9))


def test_seed_reproducibility_and_systematic_errors():
    cfg = LidarNoiseConfig(range_sigma_const_m=0.01, range_bias_std_m=0.02, range_scale_std=0.01, seed=3)
    a = LidarNoiseModel(cfg).apply(circle_scan(), -math.pi, INC, 0.05, 12.0)
    b = LidarNoiseModel(cfg).apply(circle_scan(), -math.pi, INC, 0.05, 12.0)
    np.testing.assert_array_equal(a.ranges, b.ranges)
    c = LidarNoiseModel(LidarNoiseConfig(**{**cfg.__dict__, "seed": 4})).apply(
        circle_scan(), -math.pi, INC, 0.05, 12.0
    )
    assert not np.array_equal(a.ranges, c.ranges)


def test_grazing_incidence_on_wall_increases_sigma_and_dropouts():
    # Infinite wall y = 1 seen from the origin: incidence grows towards the sides.
    angles = -math.pi + INC * np.arange(N_BEAMS)
    with np.errstate(divide="ignore"):
        scan = np.where(np.sin(angles) > 0.05, 1.0 / np.sin(angles), np.inf)
    scan[scan > 12.0] = np.inf
    cfg = LidarNoiseConfig(range_sigma_const_m=0.01, incidence_sigma_gain=1.0, dropout_prob_grazing=1.0)
    res = LidarNoiseModel(cfg).apply(scan, -math.pi, INC, 0.05, 12.0)
    beam_up = int(round((math.pi / 2 + math.pi) / INC))  # beam pointing at the wall
    np.testing.assert_allclose(res.cos_incidence[beam_up], 1.0, atol=1e-3)
    geom_cos = np.abs(np.sin(angles))
    ok = np.isfinite(scan) & (res.cos_incidence < 1.0)
    np.testing.assert_allclose(res.cos_incidence[ok], geom_cos[ok], atol=0.02)
    grazing = ok & (geom_cos < math.cos(math.radians(76)))
    assert np.any(grazing)
    assert np.all(res.outcome[grazing] == OUTCOME_DROPOUT)
