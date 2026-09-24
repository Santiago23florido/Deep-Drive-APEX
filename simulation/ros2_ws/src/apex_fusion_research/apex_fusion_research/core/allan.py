"""Allan variance analysis (IEEE Std 952-1997, Annex C).

Used to (a) validate that the simulated IMU reproduces the configured noise
coefficients and (b) identify the coefficients of a real IMU from a static
recording so that the simulation can be calibrated against hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import nnls

# AVAR(tau) of a flicker (bias instability) process on its flat region:
# sigma^2 = (2 ln 2 / pi) B^2  -> ADEV = 0.664 B.
_BIAS_INSTABILITY_FACTOR = 2.0 * math.log(2.0) / math.pi


def overlapping_allan_deviation(
    samples: np.ndarray, dt: float, num_taus: int = 60, max_fraction: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    """Overlapping Allan deviation of a rate signal sampled every ``dt``.

    Returns ``(tau, adev)`` for log-spaced cluster sizes up to
    ``max_fraction`` of the record length.
    """
    y = np.asarray(samples, dtype=float).ravel()
    n = y.size
    if n < 10:
        raise ValueError("need at least 10 samples")
    theta = np.concatenate(([0.0], np.cumsum(y) * dt))
    max_m = max(1, int(n * max_fraction))
    m_values = np.unique(np.logspace(0, math.log10(max_m), num_taus).astype(int))
    taus, adev = [], []
    for m in m_values:
        if n + 1 - 2 * m < 1:
            break
        tau = m * dt
        d = theta[2 * m :] - 2.0 * theta[m:-m] + theta[: -2 * m]
        avar = float(np.sum(d * d)) / (2.0 * tau * tau * d.size)
        taus.append(tau)
        adev.append(math.sqrt(avar))
    return np.asarray(taus), np.asarray(adev)


@dataclass
class AllanCoefficients:
    noise_density: float  # N (value of the -1/2 slope line at tau = 1 s)
    bias_instability: float  # B (flat region, ADEV = 0.664 B)
    random_walk: float  # K (value of the +1/2 slope line at tau = 3 s)


def fit_allan_coefficients(tau: np.ndarray, adev: np.ndarray) -> AllanCoefficients:
    """Fit ``AVAR = N^2/tau + 0.4413 B^2 + K^2 tau / 3`` with non-negative
    least squares in relative error (each point weighted by 1/AVAR)."""
    tau = np.asarray(tau, float)
    avar = np.asarray(adev, float) ** 2
    design = np.column_stack((1.0 / tau, np.full_like(tau, _BIAS_INSTABILITY_FACTOR), tau / 3.0))
    coef, _ = nnls(design / avar[:, None], np.ones_like(tau))
    return AllanCoefficients(
        noise_density=math.sqrt(coef[0]),
        bias_instability=math.sqrt(coef[1]),
        random_walk=math.sqrt(coef[2]),
    )
