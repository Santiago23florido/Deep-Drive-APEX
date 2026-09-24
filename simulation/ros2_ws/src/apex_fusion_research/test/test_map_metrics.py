import math

import numpy as np

from apex_fusion_research.core.map_metrics import (
    absolute_trajectory_error,
    icp_2d,
    map_similarity,
    observed_subset,
    se2_apply,
    se2_compose,
    se2_inverse,
)


def ring(n=400, r=2.0):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack((r * np.cos(a) * 1.5, r * np.sin(a)))


def test_se2_inverse_and_compose():
    a = (1.0, -2.0, 0.7)
    np.testing.assert_allclose(se2_compose(a, se2_inverse(a)), (0, 0, 0), atol=1e-12)
    pts = np.array([[1.0, 0.0]])
    np.testing.assert_allclose(se2_apply((0, 0, math.pi / 2), pts), [[0.0, 1.0]], atol=1e-12)


def test_map_similarity_identity_and_offset():
    ref = ring()
    m = map_similarity(ref, ref, 0.05)
    assert m["precision"] == 1.0 and m["coverage"] == 1.0 and m["chamfer_m"] == 0.0
    shifted = ref + np.array([0.2, 0.0])
    m2 = map_similarity(ref, shifted, 0.05)
    assert m2["precision"] < 1.0 and m2["chamfer_m"] > 0.0
    partial = ref[:100]
    m3 = map_similarity(ref, partial, 0.05)
    assert m3["precision"] == 1.0 and abs(m3["coverage"] - 0.25) < 0.02


def test_icp_recovers_known_transform():
    dst = ring()
    true_t = (0.3, -0.2, 0.15)
    src = se2_apply(se2_inverse(true_t), dst)
    t, rms = icp_2d(src, dst, max_correspondence_m=1.0)
    np.testing.assert_allclose(t, true_t, atol=1e-3)
    assert rms < 1e-3


def test_observed_subset():
    ref = np.array([[0.0, 0.0], [5.0, 5.0]])
    obs = np.array([[0.02, 0.0]])
    np.testing.assert_allclose(observed_subset(ref, obs, 0.1), [[0.0, 0.0]])


def test_absolute_trajectory_error():
    t = np.linspace(0, 10, 11)
    ref = np.column_stack((t, np.zeros_like(t)))
    est = ref + np.array([0.0, 0.3])
    ate = absolute_trajectory_error(t, ref, t[::2], est[::2])
    assert abs(ate["rmse_m"] - 0.3) < 1e-12 and ate["n"] == 6
