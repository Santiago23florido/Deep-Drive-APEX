"""Planar geometry and map / trajectory comparison metrics (ROS-independent).

Poses are ``(x, y, yaw)``; ``T_a_b`` maps coordinates expressed in frame ``b``
into frame ``a``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree


# ------------------------------------------------------------------ SE(2)
def se2_compose(a, b) -> tuple[float, float, float]:
    """``a (+) b``: pose ``b`` expressed in frame ``a``, returned in ``a``'s parent."""
    ax, ay, at = a
    bx, by, bt = b
    c, s = math.cos(at), math.sin(at)
    return (ax + c * bx - s * by, ay + s * bx + c * by, _wrap(at + bt))


def se2_inverse(a) -> tuple[float, float, float]:
    ax, ay, at = a
    c, s = math.cos(at), math.sin(at)
    return (-(c * ax + s * ay), s * ax - c * ay, _wrap(-at))


def se2_apply(t, xy: np.ndarray) -> np.ndarray:
    """Transform an (N, 2) point array with pose ``t``."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    c, s = math.cos(t[2]), math.sin(t[2])
    return xy @ np.array([[c, s], [-s, c]]) + np.array([t[0], t[1]])


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ------------------------------------------------------------ map metrics
def map_similarity(reference: np.ndarray, estimate: np.ndarray, threshold_m: float = 0.10) -> dict:
    """Symmetric point-map comparison.

    * ``precision``: fraction of estimated points within ``threshold`` of the
      reference (how much of the map is correct);
    * ``coverage``: fraction of reference points within ``threshold`` of the
      estimate (how much of the reference was mapped, i.e. recall);
    * ``chamfer_m``: mean of both mean nearest-neighbour distances.
    """
    ref = np.asarray(reference, dtype=float).reshape(-1, 2)
    est = np.asarray(estimate, dtype=float).reshape(-1, 2)
    if ref.shape[0] == 0 or est.shape[0] == 0:
        return {"precision": float("nan"), "coverage": float("nan"), "chamfer_m": float("nan"),
                "mean_est_to_ref_m": float("nan"), "mean_ref_to_est_m": float("nan"),
                "n_reference": int(ref.shape[0]), "n_estimate": int(est.shape[0]), "threshold_m": threshold_m}
    d_est, _ = cKDTree(ref).query(est)
    d_ref, _ = cKDTree(est).query(ref)
    return {
        "precision": float(np.mean(d_est <= threshold_m)),
        "coverage": float(np.mean(d_ref <= threshold_m)),
        "chamfer_m": float(0.5 * (d_est.mean() + d_ref.mean())),
        "mean_est_to_ref_m": float(d_est.mean()),
        "mean_ref_to_est_m": float(d_ref.mean()),
        "n_reference": int(ref.shape[0]),
        "n_estimate": int(est.shape[0]),
        "threshold_m": threshold_m,
    }


def observed_subset(reference: np.ndarray, observations: np.ndarray, radius_m: float = 0.10) -> np.ndarray:
    """Reference points that lie within ``radius`` of an ideal observation,
    i.e. the part of the real track the sensor actually saw."""
    ref = np.asarray(reference, dtype=float).reshape(-1, 2)
    obs = np.asarray(observations, dtype=float).reshape(-1, 2)
    if ref.shape[0] == 0 or obs.shape[0] == 0:
        return ref[:0]
    d, _ = cKDTree(obs).query(ref)
    return ref[d <= radius_m]


def _estimate_normals(points: np.ndarray, tree: cKDTree, k: int = 6) -> np.ndarray:
    """Unit normals from the PCA of the ``k`` nearest neighbours."""
    k = min(k, points.shape[0])
    _, idx = tree.query(points, k=k)
    neigh = points[idx] - points[idx].mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", neigh, neigh)
    _, vecs = np.linalg.eigh(cov)
    return vecs[:, :, 0]  # eigenvector of the smallest eigenvalue


def icp_2d(
    source: np.ndarray,
    target: np.ndarray,
    initial=(0.0, 0.0, 0.0),
    iterations: int = 100,
    max_correspondence_m: float = 0.5,
    tolerance: float = 1e-9,
) -> tuple[tuple[float, float, float], float]:
    """Rigid point-to-line ICP; returns ``(T_target_source, rms)``.

    Point-to-line residuals (target normals from local PCA) avoid the biased
    fixed points of point-to-point ICP on sampled walls. Correspondences
    farther than ``max_correspondence_m`` are rejected, so partially
    overlapping maps can be registered. ``rms`` is the point-to-point RMS of
    the accepted correspondences at the solution.
    """
    src = np.asarray(source, dtype=float).reshape(-1, 2)
    dst = np.asarray(target, dtype=float).reshape(-1, 2)
    if src.shape[0] < 3 or dst.shape[0] < 3:
        return tuple(initial), float("nan")
    tree = cKDTree(dst)
    normals = _estimate_normals(dst, tree)
    t = tuple(float(v) for v in initial)
    for _ in range(iterations):
        moved = se2_apply(t, src)
        d, idx = tree.query(moved)
        keep = d <= max_correspondence_m
        if np.count_nonzero(keep) < 3:
            break
        p, q, n = moved[keep], dst[idx[keep]], normals[idx[keep]]
        # r = n . (R(dth) p + dt - q), linearised around dth = 0.
        a = np.column_stack((n[:, 0], n[:, 1], n[:, 0] * -p[:, 1] + n[:, 1] * p[:, 0]))
        b = -np.einsum("ij,ij->i", n, p - q)
        delta, *_ = np.linalg.lstsq(a.T @ a + 1e-9 * np.eye(3), a.T @ b, rcond=None)
        t = se2_compose((float(delta[0]), float(delta[1]), float(delta[2])), t)
        if math.hypot(delta[0], delta[1]) < tolerance and abs(delta[2]) < tolerance:
            break
    d, _ = tree.query(se2_apply(t, src))
    keep = d <= max_correspondence_m
    rms = float(np.sqrt(np.mean(d[keep] ** 2))) if np.any(keep) else float("nan")
    return t, rms


# ----------------------------------------------------- trajectory metrics
def absolute_trajectory_error(t_ref, xy_ref, t_est, xy_est) -> dict:
    """Position error of an estimated trajectory against a reference,
    interpolating the reference at the estimate timestamps."""
    t_ref = np.asarray(t_ref, dtype=float)
    xy_ref = np.asarray(xy_ref, dtype=float).reshape(-1, 2)
    t_est = np.asarray(t_est, dtype=float)
    xy_est = np.asarray(xy_est, dtype=float).reshape(-1, 2)
    inside = (t_est >= t_ref[0]) & (t_est <= t_ref[-1]) if t_ref.size else np.zeros(t_est.size, bool)
    if t_ref.size < 2 or not np.any(inside):
        return {"rmse_m": float("nan"), "mean_m": float("nan"), "max_m": float("nan"), "final_m": float("nan"), "n": 0}
    ref = np.column_stack(
        (np.interp(t_est[inside], t_ref, xy_ref[:, 0]), np.interp(t_est[inside], t_ref, xy_ref[:, 1]))
    )
    err = np.linalg.norm(xy_est[inside] - ref, axis=1)
    return {
        "rmse_m": float(np.sqrt(np.mean(err**2))),
        "mean_m": float(err.mean()),
        "max_m": float(err.max()),
        "final_m": float(err[-1]),
        "n": int(err.size),
    }
