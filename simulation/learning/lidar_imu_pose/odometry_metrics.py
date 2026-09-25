"""Odometry metrics that do not depend on the scan rate or on closed loops.

The metrics of ``gridsearch_sqlite`` (kept for continuity as ``legacy_*``)
had three problems: the per-interval accuracy used an absolute 2 cm
threshold, so a 20 Hz sensor (short intervals) looked better than a 10 Hz
one at the same quality; the 0.5 deg heading condition never bound (the raw
gyro is already at 0.07 deg); and the drift was measured at the end of
two-lap runs, where closed circuits bring the estimate back near the start.

Here every method is scored on the same intervals with:

per interval (k = scan k-1 -> scan k)
  speed_err_cmps       translation error / dt (rate independent)
  rel_err_pct          translation error / true step length, moving intervals
  yawrate_err_dps      heading error / dt
  acc_speed10_pct      intervals with speed error < 10 cm/s
  acc_rel5_pct         moving intervals with relative error < 5 %
  acc_yawrate05_pct    intervals with heading-rate error < 0.5 deg/s
  stationary_*         false motion while the car is stopped
short horizons (what a SLAM front end consumes)
  rpe_{H}s_cm / _pct / _deg   error of the relative pose over H seconds
distance segments (KITTI-style drift, no loop-closure effect)
  t_rel_{L}m_pct, r_rel_{L}m_degpm and their means over L
uncertainty (when the method gives a sigma per axis)
  cover1/2_* : fraction of errors inside 1 / 2 sigma (ideal 68.3 / 95.4 %)

References
  [1] A. Geiger, P. Lenz, R. Urtasun, "Are we ready for autonomous driving?
      The KITTI vision benchmark suite", CVPR 2012 (drift over segments of
      fixed length).
  [2] J. Sturm, N. Engelhard, F. Endres, W. Burgard, D. Cremers, "A benchmark
      for the evaluation of RGB-D SLAM systems", IROS 2012 (relative pose
      error over a time horizon, absolute trajectory error).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

HORIZONS_S = (0.5, 1.0, 2.0, 5.0, 10.0)
SEGMENTS_M = (2.0, 5.0, 10.0, 20.0, 40.0)
MOVING_STEP_M = 0.02
STATIONARY_STEP_M = 0.003
LEGACY_STRICT = (0.02, math.radians(0.5))
LEGACY_LOOSE = (0.05, math.radians(1.0))


@dataclass
class SplitTruth:
    target: np.ndarray  # [N, 3] exact increments
    dt: np.ndarray  # [N]
    t_s: np.ndarray  # [N] reported scan stamps (s)
    runs: list[dict[str, Any]]

    @property
    def n(self) -> int:
        return len(self.target)


def split_truth(data: dict[str, Any]) -> SplitTruth:
    return SplitTruth(
        target=data["target"].numpy().astype(np.float64),
        dt=data["dt"].numpy().astype(np.float64),
        t_s=data["t_ns"].numpy().astype(np.float64) * 1e-9,
        runs=data["runs"],
    )


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def compose(deltas: np.ndarray) -> np.ndarray:
    """Poses [n+1, 3] of consecutive increments, starting at the identity."""
    yaw = np.concatenate(([0.0], np.cumsum(deltas[:, 2])))
    c, s = np.cos(yaw[:-1]), np.sin(yaw[:-1])
    dx = c * deltas[:, 0] - s * deltas[:, 1]
    dy = s * deltas[:, 0] + c * deltas[:, 1]
    xy = np.vstack((np.zeros((1, 2)), np.column_stack((np.cumsum(dx), np.cumsum(dy)))))
    return np.column_stack((xy, yaw))


def relative_errors(est: np.ndarray, gt: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translation and heading error of the relative poses a -> b."""
    def rel(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = p[b, :2] - p[a, :2]
        c, s = np.cos(p[a, 2]), np.sin(p[a, 2])
        return np.column_stack((c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1])), p[b, 2] - p[a, 2]

    te, ye = rel(est)
    tg, yg = rel(gt)
    return np.linalg.norm(te - tg, axis=1), np.abs(_wrap(ye - yg))


def _blocks(valid: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, end) blocks of True."""
    if not valid.any():
        return []
    edges = np.flatnonzero(np.diff(np.concatenate(([0], valid.astype(np.int8), [0]))))
    return list(zip(edges[::2], edges[1::2]))


def evaluate(
    pred: np.ndarray,
    truth: SplitTruth,
    mask: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
    flags: dict[str, np.ndarray] | None = None,
    groups: tuple[str, ...] = ("sensor", "motion", "track"),
) -> dict[str, Any]:
    """Scores ``pred`` ([N, 3], NaN where the method has no estimate).

    ``mask`` selects the intervals to score (use the same mask for every
    method being compared); index 0 of every run is always excluded.
    ``flags`` are per-interval booleans (e.g. degenerate geometry) used for
    extra breakdowns of the per-interval metrics."""
    valid = np.isfinite(pred).all(axis=1)
    if mask is not None:
        valid &= mask
    for r in truth.runs:
        valid[r["offset"]] = False
    per_run = []
    for r in truth.runs:
        per_run.append(_run_metrics(pred, truth, valid, r))
    out = {"all": _summarise(pred, truth, valid, sigma, per_run)}
    for key in groups:
        out[key] = {}
        for g in sorted({r[key] for r in truth.runs}):
            sel = np.zeros_like(valid)
            runs_g = []
            for r, pr in zip(truth.runs, per_run):
                if r[key] == g:
                    sel[r["offset"] : r["offset"] + r["n"]] = True
                    runs_g.append(pr)
            out[key][g] = _summarise(pred, truth, valid & sel, sigma, runs_g)
    for name, flag in (flags or {}).items():
        out[f"flag_{name}"] = {
            "true": _summarise(pred, truth, valid & flag, sigma, None),
            "false": _summarise(pred, truth, valid & ~flag, sigma, None),
        }
    return out


def _run_metrics(pred: np.ndarray, truth: SplitTruth, valid: np.ndarray, run: dict[str, Any]) -> dict[str, Any]:
    o, n = run["offset"], run["n"]
    res: dict[str, Any] = {"rpe": {h: ([], [], []) for h in HORIZONS_S}, "seg": {L: ([], []) for L in SEGMENTS_M}, "final": []}
    for a0, b0 in _blocks(valid[o : o + n]):
        sl = slice(o + a0, o + b0)
        est = compose(pred[sl].astype(np.float64))
        gt = compose(truth.target[sl])
        t = np.concatenate(([truth.t_s[o + a0 - 1]], truth.t_s[sl]))
        dist = np.concatenate(([0.0], np.cumsum(np.linalg.norm(truth.target[sl, :2], axis=1))))
        starts = np.arange(0, len(est) - 1)
        for h in HORIZONS_S:
            b = np.searchsorted(t, t[starts] + h - 1e-6)
            ok = b < len(t)
            if ok.any():
                a, bb = starts[ok], b[ok]
                te, ye = relative_errors(est, gt, a, bb)
                path = dist[bb] - dist[a]
                res["rpe"][h][0].append(te)
                res["rpe"][h][1].append(ye)
                res["rpe"][h][2].append(np.where(path > 0.05, te / np.maximum(path, 1e-9), np.nan))
        for L in SEGMENTS_M:
            st = starts[::2]
            b = np.searchsorted(dist, dist[st] + L)
            ok = b < len(dist)
            if ok.any():
                te, ye = relative_errors(est, gt, st[ok], b[ok])
                res["seg"][L][0].append(te / L)
                res["seg"][L][1].append(ye / L)
        if dist[-1] > 1.0:
            res["final"].append((float(np.linalg.norm(est[-1, :2] - gt[-1, :2])), float(dist[-1])))
    return res


def _cat(parts: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts) if parts else np.zeros(0)


def _summarise(pred, truth, valid, sigma, per_run) -> dict[str, float]:
    idx = np.flatnonzero(valid)
    out: dict[str, float] = {"intervals": int(len(idx))}
    if len(idx) == 0:
        return out
    p, g, dt = pred[idx].astype(np.float64), truth.target[idx], truth.dt[idx]
    trans = np.linalg.norm(p[:, :2] - g[:, :2], axis=1)
    yaw = np.abs(_wrap(p[:, 2] - g[:, 2]))
    step = np.linalg.norm(g[:, :2], axis=1)
    moving = step >= MOVING_STEP_M
    stationary = (step < STATIONARY_STEP_M) & (np.abs(g[:, 2]) < 5e-4)
    speed = trans / dt
    yawrate = yaw / dt
    rel = trans[moving] / step[moving]
    out.update({
        "speed_err_cmps": 100 * float(speed.mean()),
        "speed_err_median_cmps": 100 * float(np.median(speed)),
        "rel_err_pct": 100 * float(rel.mean()) if len(rel) else math.nan,
        "rel_err_median_pct": 100 * float(np.median(rel)) if len(rel) else math.nan,
        "yawrate_err_dps": math.degrees(float(yawrate.mean())),
        "acc_speed10_pct": 100 * float((speed < 0.10).mean()),
        "acc_rel5_pct": 100 * float((rel < 0.05).mean()) if len(rel) else math.nan,
        "acc_yawrate05_pct": 100 * float((yawrate < math.radians(0.5)).mean()),
        "trans_mae_cm": 100 * float(trans.mean()),
        "yaw_mae_deg": math.degrees(float(yaw.mean())),
        "legacy_accuracy_pct": 100 * float(((trans < LEGACY_STRICT[0]) & (yaw < LEGACY_STRICT[1])).mean()),
        "legacy_accuracy_loose_pct": 100 * float(((trans < LEGACY_LOOSE[0]) & (yaw < LEGACY_LOOSE[1])).mean()),
        "moving_fraction_pct": 100 * float(moving.mean()),
        "stationary_intervals": int(stationary.sum()),
    })
    if stationary.any():
        out["stationary_false_motion_mm"] = 1000 * float(np.linalg.norm(p[stationary, :2], axis=1).mean())
        out["stationary_false_yaw_deg"] = math.degrees(float(np.abs(p[stationary, 2]).mean()))
    if sigma is not None:
        s = sigma[idx].astype(np.float64)
        err = np.column_stack((p[:, 0] - g[:, 0], p[:, 1] - g[:, 1], _wrap(p[:, 2] - g[:, 2])))
        z = np.abs(err) / np.maximum(s, 1e-12)
        for i, axis in enumerate(("x", "y", "yaw")):
            out[f"cover1_{axis}_pct"] = 100 * float((z[:, i] < 1).mean())
            out[f"cover2_{axis}_pct"] = 100 * float((z[:, i] < 2).mean())
            out[f"z_rms_{axis}"] = float(np.sqrt(np.mean(z[:, i] ** 2)))
    if per_run is None:
        return out
    for h in HORIZONS_S:
        te = _cat([x for r in per_run for x in r["rpe"][h][0]])
        ye = _cat([x for r in per_run for x in r["rpe"][h][1]])
        pe = _cat([x for r in per_run for x in r["rpe"][h][2]])
        if len(te):
            key = f"{h:g}s"
            out[f"rpe_{key}_cm"] = 100 * float(te.mean())
            out[f"rpe_{key}_pct"] = 100 * float(np.nanmean(pe))
            out[f"rpe_{key}_deg"] = math.degrees(float(ye.mean()))
    t_all, r_all = [], []
    for L in SEGMENTS_M:
        te = _cat([x for r in per_run for x in r["seg"][L][0]])
        ye = _cat([x for r in per_run for x in r["seg"][L][1]])
        if len(te):
            out[f"t_rel_{L:g}m_pct"] = 100 * float(te.mean())
            out[f"r_rel_{L:g}m_degpm"] = math.degrees(float(ye.mean()))
            t_all.append(out[f"t_rel_{L:g}m_pct"])
            r_all.append(out[f"r_rel_{L:g}m_degpm"])
    if t_all:
        out["t_rel_pct"] = float(np.mean(t_all))
        out["r_rel_degpm"] = float(np.mean(r_all))
    finals = [f for r in per_run for f in r["final"]]
    if finals:
        out["legacy_final_drift_median_pct"] = 100 * float(np.median([e / d for e, d in finals]))
    return out


HEADLINE = (
    ("speed_err_cmps", "speed error", "cm/s"),
    ("rel_err_pct", "relative step error", "%"),
    ("yawrate_err_dps", "heading-rate error", "deg/s"),
    ("rpe_1s_cm", "RPE over 1 s", "cm"),
    ("t_rel_pct", "segment drift (2-40 m)", "%"),
    ("r_rel_degpm", "segment heading drift", "deg/m"),
    ("acc_rel5_pct", "accuracy: rel. error < 5 %", "%"),
    ("acc_yawrate05_pct", "accuracy: heading < 0.5 deg/s", "%"),
)
