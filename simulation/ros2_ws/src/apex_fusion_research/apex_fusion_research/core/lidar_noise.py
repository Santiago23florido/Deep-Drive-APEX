"""Stochastic 2D LiDAR measurement model.

The model corrupts an *ideal* range scan (ray-cast by Gazebo with zero noise)
beam by beam. It extends the classical beam mixture model of Thrun, Burgard
and Fox (Probabilistic Robotics, 2005, Sec. 6.3) with a heteroscedastic,
incidence-dependent Gaussian term, systematic calibration errors and angular
encoder jitter, which are the dominant error sources of low-cost triangulation
and time-of-flight scanners such as the RPLIDAR family.

For beam ``i`` with nominal azimuth ``theta_i`` and ideal range ``r_i``:

1. Angular jitter. The beam is actually fired along ``theta_i + d_theta``,
   ``d_theta ~ N(0, sigma_theta^2)``. The true range along the jittered
   direction ``r*_i`` is interpolated from the ideal scan when the neighbour
   beams lie on the same surface; the published azimuth stays ``theta_i``.
2. Incidence angle ``alpha_i`` between the beam and the local surface normal
   is estimated from the ideal scan geometry (neighbour beams).
3. Outcome, drawn per beam:
   * DROPOUT (no return, published as +inf) with probability
     ``p_drop = p0 + p_r (r*/r_max)^2 + p_graze * [alpha > alpha_graze]``;
   * otherwise a mixture of
     - SHORT  (prob. ``p_short``): unexpected close return,
       ``z ~ TruncExp(lambda_short)`` on ``[r_min, r*]``;
     - RANDOM (prob. ``p_rand``): spurious return, ``z ~ U(r_min, r_max)``;
     - HIT    (remaining): ``z = (1 + s) r* + b + e``,
       ``e ~ N(0, sigma(r*, alpha)^2)`` with
       ``sigma = sqrt(sigma_0^2 + (k r*)^2) * (1 + g (1/cos(alpha) - 1))``.
   ``b ~ N(0, sigma_b^2)`` and ``s ~ N(0, sigma_s^2)`` are drawn once per
   realization (power-up), modelling range offset and scale calibration errors.
4. Quantization to the sensor range resolution and REP-117 range handling
   (``-inf`` below ``range_min``, ``+inf`` above ``range_max``).

All randomness comes from a seeded ``numpy.random.Generator`` so every run is
reproducible, which is required for Monte-Carlo studies.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

# Per-beam outcome codes (also published for diagnostics).
OUTCOME_NO_TRUTH = 0  # ideal scan had no valid return; value passed through
OUTCOME_HIT = 1
OUTCOME_DROPOUT = 2
OUTCOME_SHORT = 3
OUTCOME_RANDOM = 4
OUTCOME_NAMES = {
    OUTCOME_NO_TRUTH: "no_truth",
    OUTCOME_HIT: "hit",
    OUTCOME_DROPOUT: "dropout",
    OUTCOME_SHORT: "short",
    OUTCOME_RANDOM: "random",
}


@dataclass
class LidarNoiseConfig:
    """Parameters of :class:`LidarNoiseModel`. Defaults = ideal sensor."""

    enabled: bool = True
    seed: int = 7

    # HIT component: heteroscedastic Gaussian range noise.
    range_sigma_const_m: float = 0.0  # sigma_0 [m]
    range_sigma_prop: float = 0.0  # k [-], proportional term (fraction of range)
    incidence_sigma_gain: float = 0.0  # g [-], growth with 1/cos(alpha)
    max_incidence_angle_deg: float = 85.0  # clamp for 1/cos(alpha) and geometry test

    # Systematic calibration errors drawn once per realization.
    range_bias_std_m: float = 0.0  # sigma_b [m]
    range_scale_std: float = 0.0  # sigma_s [-]

    # Angular encoder jitter.
    angle_jitter_std_rad: float = 0.0  # sigma_theta [rad]

    # DROPOUT component (missed returns).
    dropout_prob_base: float = 0.0  # p0
    dropout_prob_range_coeff: float = 0.0  # p_r, multiplies (r / r_max)^2
    grazing_angle_deg: float = 75.0  # alpha_graze
    dropout_prob_grazing: float = 0.0  # p_graze

    # SHORT component (unexpected close obstacles, dust, cross-talk).
    short_prob: float = 0.0  # p_short
    short_rate_per_m: float = 1.0  # lambda_short [1/m]

    # RANDOM component (spurious readings uniformly distributed).
    random_prob: float = 0.0  # p_rand

    # Output formatting.
    range_resolution_m: float = 0.0  # quantization step, 0 disables it
    discontinuity_abs_m: float = 0.05  # same-surface test tolerance for neighbours


@dataclass
class LidarNoiseResult:
    ranges: np.ndarray  # corrupted ranges (float32-compatible)
    outcome: np.ndarray  # per-beam OUTCOME_* codes
    true_ranges: np.ndarray  # ideal range along the jittered direction
    cos_incidence: np.ndarray  # estimated cos(alpha), 1 where unknown
    sigma: np.ndarray  # HIT standard deviation used per beam [m]


class LidarNoiseModel:
    """Apply the stochastic LiDAR model to ideal scans."""

    def __init__(self, config: LidarNoiseConfig) -> None:
        self.config = config
        seeds = np.random.SeedSequence(int(config.seed)).spawn(2)
        static_rng = np.random.default_rng(seeds[0])
        self._rng = np.random.default_rng(seeds[1])
        # Systematic errors for this realization (sensor power-up).
        self.range_bias_m = float(static_rng.normal(0.0, config.range_bias_std_m))
        self.range_scale = float(static_rng.normal(0.0, config.range_scale_std))

    def realization(self) -> dict[str, float]:
        return {"range_bias_m": self.range_bias_m, "range_scale": self.range_scale}

    # ------------------------------------------------------------------ public
    def apply(
        self,
        ranges: np.ndarray,
        angle_min: float,
        angle_increment: float,
        range_min: float,
        range_max: float,
    ) -> LidarNoiseResult:
        cfg = self.config
        ideal = np.asarray(ranges, dtype=float)
        n = ideal.size
        angles = angle_min + angle_increment * np.arange(n)
        wraps = abs(abs(angle_increment) * n - 2.0 * math.pi) < 1.5 * abs(angle_increment)

        if not cfg.enabled or n == 0:
            valid = self._valid(ideal, range_min, range_max)
            outcome = np.where(valid, OUTCOME_HIT, OUTCOME_NO_TRUTH)
            return LidarNoiseResult(
                ideal.copy(), outcome, ideal.copy(), np.ones(n), np.zeros(n)
            )

        # Draw every random number up-front: the stream consumption per scan
        # is independent of outcomes, keeping realizations comparable when a
        # single parameter is changed (common random numbers).
        rng = self._rng
        u_jitter = rng.standard_normal(n)
        u_drop = rng.random(n)
        u_mix = rng.random(n)
        u_gauss = rng.standard_normal(n)
        u_short = rng.random(n)
        u_random = rng.random(n)

        truth = self._jittered_truth(
            ideal, u_jitter * cfg.angle_jitter_std_rad, angle_increment, wraps
        )
        valid = self._valid(truth, range_min, range_max)
        cos_inc = self._incidence_cosine(truth, angles, angle_increment, valid, wraps)

        cos_floor = math.cos(math.radians(cfg.max_incidence_angle_deg))
        inc_factor = 1.0 + cfg.incidence_sigma_gain * (1.0 / np.maximum(cos_inc, cos_floor) - 1.0)
        r_safe = np.where(valid, truth, 0.0)
        sigma = np.sqrt(cfg.range_sigma_const_m**2 + (cfg.range_sigma_prop * r_safe) ** 2)
        sigma = sigma * inc_factor

        grazing = cos_inc < math.cos(math.radians(cfg.grazing_angle_deg))
        p_drop = (
            cfg.dropout_prob_base
            + cfg.dropout_prob_range_coeff * (r_safe / max(range_max, 1e-9)) ** 2
            + cfg.dropout_prob_grazing * grazing
        )
        p_drop = np.clip(p_drop, 0.0, 1.0)

        dropout = valid & (u_drop < p_drop)
        returned = valid & ~dropout
        short = returned & (u_mix < cfg.short_prob)
        random_ = returned & ~short & (u_mix < cfg.short_prob + cfg.random_prob)
        hit = returned & ~short & ~random_

        out = ideal.copy()
        out[hit] = (1.0 + self.range_scale) * truth[hit] + self.range_bias_m + sigma[hit] * u_gauss[hit]
        if np.any(short):
            span = np.maximum(truth[short] - range_min, 0.0)
            lam = max(cfg.short_rate_per_m, 1e-9)
            # Inverse CDF of the exponential truncated to [0, span].
            out[short] = range_min - np.log1p(-u_short[short] * (-np.expm1(-lam * span))) / lam
        out[random_] = range_min + u_random[random_] * (range_max - range_min)
        out[dropout] = np.inf

        if cfg.range_resolution_m > 0.0:
            finite = np.isfinite(out) & valid
            out[finite] = np.round(out[finite] / cfg.range_resolution_m) * cfg.range_resolution_m

        measured = valid & ~dropout
        out[measured & (out < range_min)] = -np.inf
        out[measured & (out > range_max)] = np.inf

        outcome = np.full(n, OUTCOME_NO_TRUTH, dtype=np.int8)
        outcome[hit] = OUTCOME_HIT
        outcome[dropout] = OUTCOME_DROPOUT
        outcome[short] = OUTCOME_SHORT
        outcome[random_] = OUTCOME_RANDOM
        return LidarNoiseResult(out, outcome, truth, cos_inc, np.where(hit, sigma, 0.0))

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _valid(r: np.ndarray, range_min: float, range_max: float) -> np.ndarray:
        return np.isfinite(r) & (r >= range_min) & (r <= range_max)

    def _same_surface(self, r_a: np.ndarray, r_b: np.ndarray, beams_apart: float, inc: float) -> np.ndarray:
        """Neighbour test: the range change allowed between two beams on one
        planar surface seen at up to ``max_incidence_angle`` incidence."""
        tan_max = math.tan(math.radians(self.config.max_incidence_angle_deg))
        r_mean = 0.5 * (np.abs(r_a) + np.abs(r_b))
        tol = self.config.discontinuity_abs_m + r_mean * abs(beams_apart * inc) * tan_max
        with np.errstate(invalid="ignore"):
            return np.isfinite(r_a) & np.isfinite(r_b) & (np.abs(r_a - r_b) <= tol)

    def _jittered_truth(
        self, ideal: np.ndarray, d_theta: np.ndarray, inc: float, wraps: bool
    ) -> np.ndarray:
        n = ideal.size
        if n < 2 or not np.any(d_theta):
            return ideal.copy()
        pos = np.arange(n) + d_theta / inc
        lo = np.floor(pos).astype(int)
        frac = pos - lo
        hi = lo + 1
        if wraps:
            lo %= n
            hi %= n
        else:
            lo = np.clip(lo, 0, n - 1)
            hi = np.clip(hi, 0, n - 1)
        r_lo, r_hi = ideal[lo], ideal[hi]
        interp_ok = self._same_surface(r_lo, r_hi, 1.0, inc)
        with np.errstate(invalid="ignore"):
            interpolated = (1.0 - frac) * r_lo + frac * r_hi
        nearest = np.where(frac < 0.5, r_lo, r_hi)
        return np.where(interp_ok, interpolated, nearest)

    def _incidence_cosine(
        self, r: np.ndarray, angles: np.ndarray, inc: float, valid: np.ndarray, wraps: bool
    ) -> np.ndarray:
        n = r.size
        cos_inc = np.ones(n)
        if n < 3:
            return cos_inc
        idx = np.arange(n)
        prev_i, next_i = idx - 1, idx + 1
        if wraps:
            prev_i %= n
            next_i %= n
        else:
            prev_i = np.clip(prev_i, 0, n - 1)
            next_i = np.clip(next_i, 0, n - 1)
        r_prev, r_next = r[prev_i], r[next_i]
        ok = valid & (prev_i != idx) & (next_i != idx) & self._same_surface(r_prev, r_next, 2.0, inc)
        if not np.any(ok):
            return cos_inc
        # Non-finite neighbours are masked out by ``ok``; silence their NaNs.
        with np.errstate(invalid="ignore", divide="ignore"):
            px_prev = r_prev * np.cos(angles[prev_i])
            py_prev = r_prev * np.sin(angles[prev_i])
            px_next = r_next * np.cos(angles[next_i])
            py_next = r_next * np.sin(angles[next_i])
            tx, ty = px_next - px_prev, py_next - py_prev
            norm_t = np.hypot(tx, ty)
            ok &= norm_t > 1e-9
            # Beam direction d and unit tangent t: cos(alpha) = |d x t|.
            cross = np.abs(np.cos(angles) * ty - np.sin(angles) * tx) / norm_t
        cos_inc[ok] = np.clip(cross[ok], 0.0, 1.0)
        return cos_inc


def outcome_fractions(outcome: np.ndarray) -> dict[str, float]:
    """Fraction of each outcome among beams that had a valid ideal return."""
    measured = outcome != OUTCOME_NO_TRUTH
    total = int(np.count_nonzero(measured))
    if total == 0:
        return {name: 0.0 for code, name in OUTCOME_NAMES.items() if code != OUTCOME_NO_TRUTH}
    return {
        name: float(np.count_nonzero(outcome == code)) / total
        for code, name in OUTCOME_NAMES.items()
        if code != OUTCOME_NO_TRUTH
    }
