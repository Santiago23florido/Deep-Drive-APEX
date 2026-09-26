#!/usr/bin/env python3
"""Classical LiDAR-inertial scan-to-scan odometry (the reference to beat).

Causal, sensor-only, one estimate per scan interval, batched over all runs of
a split on the GPU. For interval k (scan k-1 -> scan k):

1. prediction: translation from the previous estimate (constant velocity)
   plus the mean IMU specific force, heading from the gyro integral minus
   the running gyro-bias estimate;
2. de-skew: the scanner is rolling (every beam is fired at its own time
   after the stamp), so every beam is moved to the pose of its sweep start:
   the heading at the beam's firing time from the gyro profile of that sweep
   (the gyro integrated sample by sample, minus the bias; ``deskew_profile``)
   and the translation along that heading with the current velocity
   estimate. The de-skew is repeated after the first solve with the refined
   velocity. Without the profile the rate is constant over the sweep, which
   bends the scan whenever the yaw rate changes (entering or leaving a curve);
3. point-to-line ICP (Censi-style) between the de-skewed scans in base_link
   (nominal mount), with normals from local PCA along the beam order,
   heteroscedastic point weights and a Cauchy kernel;
4. MAP solve: the ICP normal equations plus the prediction as a Gaussian
   prior. In a straight corridor the walls give no information along the
   corridor axis. Point-to-point ICP then collapses towards "no motion"
   (nearest neighbours pull the scans onto each other), while here that
   direction simply falls back to the prediction instead of to zero. The
   smallest eigenvalue of the ICP-only translation information is stored, so
   degenerate intervals can be reported separately;
5. updates: velocity from the solved increment, gyro bias from the ICP
   heading when the heading is well constrained.

Reference (``IcpConfig.submap_keyframes``): 0 = scan-to-scan (every scan is
registered against the previous one, so every interval adds its own
alignment error); K > 0 = scan-to-submap: the target is a local map made of
the last K keyframes (a scan becomes a keyframe after 0.5 m or 10 deg of
motion), kept in an odometry frame with their estimated poses and brought to
the frame of the previous scan for the solve. While the reference keyframes
stay the same, consecutive scans are all anchored to the same geometry, so
the error grows per keyframe instead of per scan, and features seen from
earlier positions keep constraining the motion. A keyframe is inserted one
interval late, when the velocity during its own sweep is known, with the
de-skew of that final estimate. The submap solution is kept only if it
agrees with the scan-to-scan one within ``submap_agree_m`` (or, with
``submap_agree_rel``, within that fraction of the step: their disagreement
grows with the motion per interval).

Variant ``submap_v3`` (``ICP_VARIANTS``, version 3) targets fast motion: the
heading of every beam from the gyro profile of its sweep, and the relative
agreement (30 % of the step). Use it with IMU stamps compensated for the chip
filter delay (``sqlite_windows.imu_group_delay_s``), which a lagging gyro
otherwise turns into a heading error proportional to the yaw acceleration.

Outputs per interval: increment, 1-sigma per axis (scaled by the residual
chi^2), and the along-weakest-direction ICP sigma (degeneracy indicator).

The combination, the batched GPU implementation, the map refinement with the
agreement check and every parameter are this project's own choices (tuned on
the validation split only); the building blocks are standard:

References
  [1] P. J. Besl, N. D. McKay, "A method for registration of 3-D shapes",
      IEEE TPAMI 14(2), 1992 (ICP).
  [2] Y. Chen, G. Medioni, "Object modelling by registration of multiple range
      images", Image and Vision Computing 10(3), 1992 (point-to-plane metric).
  [3] A. Censi, "An ICP variant using a point-to-line metric", IEEE ICRA 2008
      (point-to-line ICP for 2D scans; step 3).
  [4] J. Zhang, S. Singh, "LOAM: Lidar Odometry and Mapping in Real-time",
      RSS 2014 (motion compensation of a spinning scanner; step 2).
  [5] I. Vizzo, T. Guadagnino, B. Mersch, L. Wiesmann, J. Behley, C. Stachniss,
      "KISS-ICP: In Defense of Point-to-Point ICP - Simple, Accurate, and
      Robust Registration If Done the Right Way", IEEE RA-L 8(2), 2023
      (constant-velocity prediction and de-skew, local map).
  [6] J. Zhang, M. Kaess, S. Singh, "On degeneracy of optimization-based state
      estimation problems", IEEE ICRA 2016 (eigenvalue degeneracy; step 4).
  [7] Z. Zhang, "Parameter estimation techniques: a tutorial with application
      to conic fitting", Image and Vision Computing 15(1), 1997 (M-estimators,
      Cauchy kernel).
  [8] W. Hess, D. Kohler, H. Rapp, D. Andor, "Real-time loop closure in 2D
      LIDAR SLAM", IEEE ICRA 2016 (Cartographer: scan-to-submap matching).
  [9] P. D. Groves, "Principles of GNSS, Inertial, and Multisensor Integrated
      Navigation Systems", 2nd ed., Artech House, 2013 (inertial prediction,
      bias estimation with a Kalman filter).
  [10] I. Skog, P. Handel, J.-O. Nilsson, J. Rantakokko, "Zero-velocity
      detection - An algorithm evaluation", IEEE Trans. Biomedical Engineering
      57(11), 2010 (standstill detection, zero-velocity update).
Not yet cross-checked against an established implementation (e.g. CSM /
PLICP or KISS-ICP) on the same data.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
import time

import torch
from torch import Tensor

from sqlite_streams import StreamData, sweep_interp


@dataclass(frozen=True)
class IcpConfig:
    version: int = 2  # bumped when the algorithm changes (2: FP32 forced, map refinement from the scan solution; 3: ICP_VARIANTS["submap_v3"])
    iterations: int = 8  # Gauss-Newton iterations per de-skew pass
    deskew_passes: int = 2
    gates_m: tuple[float, ...] = (0.40, 0.40, 0.25, 0.25, 0.15, 0.15, 0.15, 0.15)
    model_sigma_m: float = 0.003  # added to the datasheet range noise (de-skew, normals, mount)
    deskew: bool = True  # rotation (gyro) part of the de-skew
    deskew_translation: bool = True  # translation (velocity) part
    deskew_profile: bool = False  # heading from the gyro profile of the sweep (False: constant rate over the sweep, versions <= 2)
    cauchy: float = 2.0  # robust kernel scale, in point sigmas ...
    # ... but never below this per-iteration floor: a wide kernel first, so
    # a prediction that is far off (start, stops, hard acceleration) can
    # still be corrected by the few points that observe it, instead of those
    # points being down-weighted as outliers of the prediction.
    robust_floor_m: tuple[float, ...] = (0.20, 0.10, 0.05, 0.03, 0.02, 0.0, 0.0, 0.0)
    planarity_max: float = 0.05  # PCA eigenvalue ratio of a usable line patch
    min_pairs: int = 25
    prior_trans_sigma_m: float = 0.02  # constant-velocity prediction error per interval
    gyro_noise_rad: float = 5.0e-4  # gyro integral error per interval besides the bias
    bias_sigma0_rps: float = 0.03  # initial gyro-bias uncertainty (turn-on bias)
    bias_walk_rps: float = 1.0e-4  # bias change per interval (process noise)
    gyro_scale_error: float = 0.02  # gyro scale factor uncertainty (inflates bias measurements in turns)
    standstill: bool = True
    standstill_accel_std: float = 0.15  # m/s^2: vibration-free specific force => stopped
    standstill_rate_rps: float = 0.02  # |bias-corrected gyro rate| below this
    standstill_speed_mps: float = 0.3  # and the predicted speed below this
    standstill_sigma_m: float = 0.002
    degenerate_ratio: float = 0.03  # ICP-only translation information: weakest / strongest eigenvalue
    submap_keyframes: int = 0  # 0: scan-to-scan; K: register against the last K keyframes
    keyframe_dist_m: float = 0.5
    keyframe_yaw_rad: float = 0.17
    map_passes: int = 1  # refinement against the local map, from the scan-to-scan solution
    map_iterations: int = 4
    submap_agree_m: float = 0.05  # submap estimate kept only within this of the scan-to-scan one ...
    submap_agree_rad: float = 0.0087  # ... and 0.5 deg; otherwise scan-to-scan and a fresh local map
    submap_agree_rel: float = 0.0  # translation agreement also accepted within this fraction of the step (0: fixed 5 cm)
    max_lanes: int = 128  # runs processed together on the GPU (memory)


ICP_VARIANTS = {
    "scan": IcpConfig(),
    "submap": IcpConfig(submap_keyframes=5),
    # Fast motion: de-skew from the gyro profile of each sweep, and a submap
    # solution kept within 5 cm or 30 % of the step of the scan-to-scan one
    # (at 3.5-5 m/s the fixed 5 cm rejected 5 % of the refinements).
    "submap_v3": IcpConfig(version=3, submap_keyframes=5, deskew_profile=True, submap_agree_rel=0.3),
}


@contextmanager
def fp32_matmul():
    """Full FP32 for the ICP: TF32 matmuls (enabled for training elsewhere)
    make cdist and the normal equations lose ~3 decimal digits."""
    tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32


def rot(a: Tensor) -> tuple[Tensor, Tensor]:
    return torch.cos(a), torch.sin(a)


def deskew(ranges: Tensor, valid: Tensor, rate: Tensor, vel: Tensor, time_inc: Tensor, lidar_xy: Tensor,
           angle_min: Tensor, angle_inc: Tensor, beam_time: Tensor | None = None,
           sweep: tuple[Tensor, Tensor] | None = None) -> Tensor:
    """Beam endpoints in base_link at the start of the sweep.

    ranges/valid: [R, N]; rate: [R] yaw rate during the sweep; vel: [R, 2]
    body velocity during the sweep, in the sweep-start frame.
    ``beam_time`` [R, N]: acquisition time of every beam after the stamp
    (clockwise scanners, sweeps that start at another angle); by default
    beam i is fired at ``i * time_inc``.
    ``sweep`` = (heading [R, K+1], duration [R]): heading of the car at K+1
    equally spaced instants of the sweep (gyro profile minus bias). It
    replaces the constant ``rate``, and the body velocity is carried along
    that heading (midpoint rule per segment)."""
    i = torch.arange(ranges.shape[-1], device=ranges.device, dtype=ranges.dtype)
    tau = i[None] * time_inc[:, None] if beam_time is None else beam_time
    theta = angle_min[:, None] + i[None] * angle_inc[:, None]
    qx = lidar_xy[:, :1] + ranges * torch.cos(theta)
    qy = lidar_xy[:, 1:] + ranges * torch.sin(theta)
    if sweep is None:
        psi = rate[:, None] * tau
        ch, sh = rot(0.5 * psi)
        px = (ch * vel[:, :1] - sh * vel[:, 1:]) * tau
        py = (sh * vel[:, :1] + ch * vel[:, 1:]) * tau
    else:
        heading, duration = sweep
        k = heading.shape[1] - 1
        cm, sm = rot(0.5 * (heading[:, 1:] + heading[:, :-1]))
        seg = duration[:, None] / k
        step = torch.stack(((cm * vel[:, :1] - sm * vel[:, 1:]) * seg, (sm * vel[:, :1] + cm * vel[:, 1:]) * seg), dim=-1)
        track = torch.cat((torch.zeros_like(step[:, :1]), torch.cumsum(step, dim=1)), dim=1)  # [R, K+1, 2]
        psi = sweep_interp(heading, duration, tau)
        p = sweep_interp(track, duration, tau)
        px, py = p[..., 0], p[..., 1]
    c, s = rot(psi)
    pts = torch.stack((c * qx - s * qy + px, s * qx + c * qy + py), dim=-1)
    return torch.where(valid[..., None], pts, torch.full_like(pts, 1.0e4))


def line_normals(pts: Tensor, valid: Tensor, ranges: Tensor, angle_inc: Tensor, planarity_max: float) -> tuple[Tensor, Tensor]:
    """Normals from PCA over 5 consecutive beams that lie on one continuous surface."""
    n = pts.shape[1]
    idx = (torch.arange(n, device=pts.device)[:, None] + torch.arange(-2, 3, device=pts.device)[None]) % n
    q = pts[:, idx]  # [R, N, 5, 2]
    v = valid[:, idx].all(-1)
    gap = torch.linalg.vector_norm(q[:, :, 1:] - q[:, :, :-1], dim=-1)
    v &= (gap < 0.05 + 3.0 * ranges[..., None] * angle_inc[:, None, None]).all(-1)
    d = q - q.mean(2, keepdim=True)
    a = d[..., 0].square().mean(-1)
    b = (d[..., 0] * d[..., 1]).mean(-1)
    c = d[..., 1].square().mean(-1)
    phi = 0.5 * torch.atan2(2 * b, a - c)  # principal (tangent) direction
    half = 0.5 * (a + c)
    root = torch.sqrt(((a - c) * 0.5).square() + b.square())
    ratio = (half - root) / (half + root).clamp_min(1e-12)
    v &= ratio < planarity_max
    normal = torch.stack((-torch.sin(phi), torch.cos(phi)), dim=-1)
    return normal, v


def _rotate(v: Tensor, angle: Tensor) -> Tensor:
    c, s = rot(angle)
    return torch.stack((c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]), dim=1)


def _compose(pose: Tensor, x: Tensor) -> Tensor:
    c, s = rot(pose[:, 2])
    return torch.stack((pose[:, 0] + c * x[:, 0] - s * x[:, 1], pose[:, 1] + s * x[:, 0] + c * x[:, 1], pose[:, 2] + x[:, 2]), dim=1)


@torch.no_grad()
def run_icp(sd: StreamData, cfg: IcpConfig = IcpConfig(), log=print) -> dict[str, Tensor]:
    """Runs are processed in groups of similar length (at most ``max_lanes``)."""
    n_total = sd.target.shape[0]
    logs = {name: torch.full((n_total,) + shape, float("nan"), device=sd.device) for name, shape in (
        ("pred", (3,)), ("sigma", (3,)), ("weak_sigma_m", ()), ("eig_ratio", ()), ("icp_yaw_sigma_rad", ()),
        ("gyro_bias", ()), ("pairs", ()), ("standstill", ()), ("map_used", ()), ("map_diff", (2,)), ("yaw_icp", ()), ("yaw_info", ()))}
    order = torch.argsort(sd.length, descending=True)
    t0 = time.time()
    with fp32_matmul():
        for g, lo in enumerate(range(0, len(order), cfg.max_lanes)):
            _run_group(sd, cfg, order[lo : lo + cfg.max_lanes], logs)
            log(f"[icp {sd.split}] group {g + 1}/{math.ceil(len(order) / cfg.max_lanes)} done ({time.time() - t0:.0f} s)")
    return {**{name: val.cpu() for name, val in logs.items()}, "config": asdict(cfg)}


def _register(cfg: IcpConfig, b: dict[str, Tensor], x_prior: Tensor, x0: Tensor, lam: Tensor, rate_prev: Tensor, rate_cur: Tensor,
              sig_pt: Tensor, geo: tuple, dt: Tensor, ridx: Tensor, local_map: tuple[Tensor, Tensor, Tensor] | None,
              sweeps: tuple = (None, None)) -> dict[str, Tensor]:
    """De-skew passes + robust Gauss-Newton (MAP with the prediction as prior),
    against the previous scan or against ``local_map`` (points, normals,
    validity in the frame of the previous scan). The map refinement starts
    from the scan-to-scan solution, so it skips the wide first stages of the
    gate / kernel schedules."""
    x = x0.clone()
    passes, iterations, first = (cfg.deskew_passes, cfg.iterations, 0) if local_map is None else (cfg.map_passes, cfg.map_iterations, 2)
    for _ in range(passes):
        vel = x[:, :2] / dt[:, None] if cfg.deskew and cfg.deskew_translation else torch.zeros_like(x[:, :2])
        tgt = deskew(b["prev_ranges"], b["prev_valid"], rate_prev, vel, *geo, sweep=sweeps[0])
        src = deskew(b["cur_ranges"], b["cur_valid"], rate_cur, _rotate(vel, -x[:, 2]), *geo, sweep=sweeps[1])
        normal, nvalid = line_normals(tgt, b["prev_valid"], b["prev_ranges"], geo[3], cfg.planarity_max)
        target, t_normal, t_valid = (tgt, normal, nvalid) if local_map is None else local_map
        for it in range(first, first + iterations):
            c, s = rot(x[:, 2])
            sx, sy = src[..., 0], src[..., 1]
            y = torch.stack((c[:, None] * sx - s[:, None] * sy + x[:, None, 0], s[:, None] * sx + c[:, None] * sy + x[:, None, 1]), dim=-1)
            dist, j = torch.cdist(y, target).min(dim=-1)
            nj = t_normal[ridx[:, None], j]
            use = b["cur_valid"] & (dist < cfg.gates_m[min(it, len(cfg.gates_m) - 1)]) & t_valid[ridx[:, None], j]
            e = ((y - target[ridx[:, None], j]) * nj).sum(-1)
            # d y / d yaw = R(yaw + pi/2) src
            dyx = -s[:, None] * sx - c[:, None] * sy
            dyy = c[:, None] * sx - s[:, None] * sy
            jac = torch.stack((nj[..., 0], nj[..., 1], nj[..., 0] * dyx + nj[..., 1] * dyy), dim=-1)
            kernel = torch.clamp(cfg.cauchy * sig_pt, min=cfg.robust_floor_m[min(it, len(cfg.robust_floor_m) - 1)])
            w = use.float() / sig_pt.square() / (1.0 + (e / kernel).square())
            h_icp = torch.einsum("rn,rni,rnj->rij", w, jac, jac)
            h = h_icp + torch.diag_embed(lam)
            g = torch.einsum("rn,rni,rn->ri", w, jac, e) + lam * (x - x_prior)
            enough = use.sum(-1) >= cfg.min_pairs
            step = torch.linalg.solve(h, -g.unsqueeze(-1)).squeeze(-1)
            x = torch.where(enough[:, None], x + step, x_prior)
    return {"x": x, "h": h, "h_icp": h_icp, "w": w, "e": e, "use": use, "enough": enough, "tgt_scan": (tgt, normal, nvalid)}


class IcpStream:
    """Causal state of the classical odometry for ``R`` lanes (runs).

    ``step(b, act)`` processes one scan interval of every lane (``b``: the
    fields of ``StreamData.gather`` for that interval, ``act``: lanes that
    have one) and returns the per-lane estimates. The batch evaluation
    (``run_icp``) and the live node use this same code."""

    def __init__(self, cfg: IcpConfig, geo: tuple, lidar_sigma: Tensor, beams: int, device: torch.device) -> None:
        self.cfg = cfg
        self.geo = geo  # (time_increment [R], lidar_xy [R, 2], angle_min [R], angle_increment [R], beam_time [R, N] | None)
        self.lidar_sigma = lidar_sigma
        runs = lidar_sigma.shape[0]
        dev = device
        self.v = torch.zeros(runs, 2, device=dev)
        self.bg = torch.zeros(runs, device=dev)  # gyro-bias estimate (scalar Kalman filter per run)
        self.p_bg = torch.full((runs,), cfg.bias_sigma0_rps**2, device=dev)
        self.ridx = torch.arange(runs, device=dev)
        self.pose = torch.zeros(runs, 3, device=dev)  # scan k-1 in the odometry frame
        kf = cfg.submap_keyframes
        if kf:
            self.kf_pts = torch.full((runs, kf, beams, 2), 1.0e4, device=dev)
            self.kf_nrm = torch.zeros(runs, kf, beams, 2, device=dev)
            self.kf_ok = torch.zeros(runs, kf, beams, dtype=torch.bool, device=dev)
            self.kf_count = torch.zeros(runs, dtype=torch.long, device=dev)
            self.kf_last = torch.zeros(runs, 3, device=dev)

    @torch.no_grad()
    def step(self, b: dict[str, Tensor], act: Tensor) -> dict[str, Tensor]:
        cfg, geo, ridx = self.cfg, self.geo, self.ridx
        runs = act.shape[0]
        kf = cfg.submap_keyframes
        v, bg, p_bg, pose = self.v, self.bg, self.p_bg, self.pose
        dt = b["dt"]
        p_bg = p_bg + cfg.bias_walk_rps**2
        # Prediction: constant velocity + IMU specific force, gyro minus bias.
        x_prior = torch.cat((v * dt[:, None] + 0.5 * b["accel"] * dt[:, None].square(), (b["gyro_integral"] - bg * dt)[:, None]), dim=1)
        sig_t = torch.full_like(dt, cfg.prior_trans_sigma_m)
        sig_y = torch.sqrt(cfg.gyro_noise_rad**2 + p_bg * dt.square())
        # Standstill (zero-velocity / zero-rate update): no vibration, no rotation, slow prediction.
        still = torch.zeros_like(act)
        if cfg.standstill:
            still = (b["accel_std"] < cfg.standstill_accel_std) & ((b["gyro_integral"] / dt - bg).abs() < cfg.standstill_rate_rps) \
                & (torch.linalg.vector_norm(v, dim=1) < cfg.standstill_speed_mps)
            x_prior = torch.where(still[:, None], torch.zeros_like(x_prior), x_prior)
            sig_t = torch.where(still, torch.full_like(sig_t, cfg.standstill_sigma_m), sig_t)
            sig_y = torch.where(still, torch.full_like(sig_y, cfg.gyro_noise_rad * 0.2), sig_y)
        lam = torch.stack((sig_t.pow(-2), sig_t.pow(-2), sig_y.pow(-2)), dim=1)
        rate_prev, rate_cur = b["w_prev"] - bg, b["w_cur"] - bg
        if not cfg.deskew:
            rate_prev, rate_cur = torch.zeros_like(rate_prev), torch.zeros_like(rate_cur)
        sweeps = (None, None)
        if cfg.deskew and cfg.deskew_profile:
            f = torch.linspace(0.0, 1.0, b["prof_prev"].shape[-1], device=dt.device)
            sweeps = ((b["prof_prev"] - (bg * dt)[:, None] * f, dt), (b["prof_cur"] - (bg * b["dt_next"])[:, None] * f, b["dt_next"]))
        sig_pt = cfg.model_sigma_m + self.lidar_sigma[:, :1] + self.lidar_sigma[:, 1:] * b["cur_ranges"]
        # 1) scan-to-scan registration (robust); 2) in submap mode, refined
        # against the local map from that solution, kept only if both agree.
        reg = _register(cfg, b, x_prior, x_prior, lam, rate_prev, rate_cur, sig_pt, geo, dt, ridx, None, sweeps)
        map_used = torch.zeros(runs, device=dt.device)
        map_diff = torch.full((runs, 2), float("nan"), device=dt.device)
        if kf:
            kf_pts, kf_nrm, kf_ok, kf_count = self.kf_pts, self.kf_nrm, self.kf_ok, self.kf_count
            c, s = rot(-pose[:, 2])
            d = kf_pts - pose[:, None, None, :2]
            map_pts = torch.stack((c[:, None, None] * d[..., 0] - s[:, None, None] * d[..., 1], s[:, None, None] * d[..., 0] + c[:, None, None] * d[..., 1]), dim=-1)
            map_nrm = torch.stack((c[:, None, None] * kf_nrm[..., 0] - s[:, None, None] * kf_nrm[..., 1], s[:, None, None] * kf_nrm[..., 0] + c[:, None, None] * kf_nrm[..., 1]), dim=-1)
            map_pts = torch.where(kf_ok[..., None], map_pts, torch.full_like(map_pts, 1.0e4)).reshape(runs, kf * map_pts.shape[2], 2)
            local_map = (map_pts, map_nrm.reshape(runs, -1, 2), kf_ok.reshape(runs, -1))
            reg_map = _register(cfg, b, x_prior, reg["x"], lam, rate_prev, rate_cur, sig_pt, geo, dt, ridx, local_map, sweeps)
            diff = reg_map["x"] - reg["x"]
            map_diff = torch.stack((torch.linalg.vector_norm(diff[:, :2], dim=1), diff[:, 2].abs()), dim=1)
            agree_m = torch.clamp(cfg.submap_agree_rel * torch.linalg.vector_norm(reg["x"][:, :2], dim=1), min=cfg.submap_agree_m)
            agree = (kf_count > 0) & reg_map["enough"] & (torch.linalg.vector_norm(diff[:, :2], dim=1) < agree_m) & (diff[:, 2].abs() < cfg.submap_agree_rad)
            # Disagreement: keep the scan-to-scan estimate and restart the local map.
            reset = act & (kf_count > 0) & ~agree
            self.kf_ok = torch.where(reset[:, None, None], torch.zeros_like(kf_ok), kf_ok)
            self.kf_count = torch.where(reset, torch.zeros_like(kf_count), kf_count)
            reg = {key: torch.where(agree.reshape((-1,) + (1,) * (val.dim() - 1)), reg_map[key], val) if key != "tgt_scan" else val for key, val in reg.items()}
            map_used = agree.float()
        x, h, h_icp, w, e, use, enough = (reg[key] for key in ("x", "h", "h_icp", "w", "e", "use", "enough"))
        tgt, normal, nvalid = reg["tgt_scan"]
        # Uncertainty: MAP covariance scaled by the residual chi^2 per dof.
        chi2 = ((w * e.square()).sum(-1) / use.sum(-1).clamp_min(1)).clamp_min(1.0)
        cov = torch.linalg.inv(h) * chi2[:, None, None]
        eigs = torch.linalg.eigvalsh(h_icp[:, :2, :2]).clamp_min(1e-6)
        # ICP-only heading information (translation marginalised) and the
        # ICP-only heading recovered from the MAP solution.
        yaw_info = (h_icp[:, 2, 2] - (h_icp[:, 2:, :2] @ torch.linalg.pinv(h_icp[:, :2, :2]) @ h_icp[:, :2, 2:]).reshape(-1)).clamp_min(1e-6) / chi2
        yaw_icp = x[:, 2] + (lam[:, 2] / yaw_info) * (x[:, 2] - x_prior[:, 2])
        # Gyro-bias Kalman update: from the ICP heading while moving, from the
        # raw gyro integral at standstill (true rotation = 0).
        rate = b["gyro_integral"] / dt
        meas = torch.where(still, rate, (b["gyro_integral"] - yaw_icp) / dt)
        r_meas = torch.where(still, torch.full_like(dt, cfg.gyro_noise_rad**2), 1.0 / yaw_info + cfg.gyro_noise_rad**2 + (cfg.gyro_scale_error * rate * dt).square()) / dt.square()
        innov = meas - bg
        ok = act & (enough | still) & (innov.square() < 9.0 * (p_bg + r_meas))
        gain = p_bg / (p_bg + r_meas)
        self.bg = torch.where(ok, bg + gain * innov, bg)
        self.p_bg = torch.where(ok, (1 - gain) * p_bg, p_bg)
        # Velocity at the end of the interval, in the frame of scan k.
        v_end = x[:, :2] / dt[:, None] + 0.5 * b["accel"] * dt[:, None]
        self.v = torch.where(act[:, None], _rotate(v_end, -x[:, 2]), v)
        if kf:
            # Scan k-1 becomes a keyframe (with the de-skew of its final
            # velocity estimate) when it is far enough from the last one.
            moved = torch.linalg.vector_norm(pose[:, :2] - self.kf_last[:, :2], dim=1)
            turned = torch.atan2(torch.sin(pose[:, 2] - self.kf_last[:, 2]), torch.cos(pose[:, 2] - self.kf_last[:, 2])).abs()
            insert = act & ((self.kf_count == 0) | (moved > cfg.keyframe_dist_m) | (turned > cfg.keyframe_yaw_rad))
            slot = self.kf_count % kf
            c, s = rot(pose[:, 2])
            world = torch.stack((c[:, None] * tgt[..., 0] - s[:, None] * tgt[..., 1] + pose[:, None, 0], s[:, None] * tgt[..., 0] + c[:, None] * tgt[..., 1] + pose[:, None, 1]), dim=-1)
            wnrm = torch.stack((c[:, None] * normal[..., 0] - s[:, None] * normal[..., 1], s[:, None] * normal[..., 0] + c[:, None] * normal[..., 1]), dim=-1)
            rows = ridx[insert]
            self.kf_pts[rows, slot[insert]] = world[insert]
            self.kf_nrm[rows, slot[insert]] = wnrm[insert]
            self.kf_ok[rows, slot[insert]] = nvalid[insert]
            self.kf_last = torch.where(insert[:, None], pose, self.kf_last)
            self.kf_count = self.kf_count + insert.long()
        self.pose = torch.where(act[:, None], _compose(pose, x), pose)
        return {"pred": x, "sigma": torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1)), "weak_sigma_m": eigs[:, 0].rsqrt(),
                "eig_ratio": eigs[:, 0] / eigs[:, 1], "icp_yaw_sigma_rad": yaw_info.rsqrt(), "gyro_bias": self.bg,
                "pairs": use.sum(-1).float(), "standstill": still.float(), "map_used": map_used,
                "map_diff": map_diff,  # submap vs scan-to-scan solution (m, rad), diagnostic
                "yaw_icp": yaw_icp, "yaw_info": yaw_info,  # LiDAR-only heading and its information (IMU time-offset calibration)
                # de-skewed previous scan (base_link at the start of its sweep), for consumers such as a SLAM
                "prev_points": tgt, "prev_points_valid": nvalid}


def _run_group(sd: StreamData, cfg: IcpConfig, group: Tensor, logs: dict[str, Tensor]) -> None:
    offset, length = sd.offset[group], sd.length[group]
    geo = (sd.time_increment[group], sd.lidar_xy[group], sd.angle_min[group], sd.angle_increment[group], sd.beam_time(group))
    stream = IcpStream(cfg, geo, sd.lidar_sigma[group], sd.lidar.shape[-1], sd.device)
    for k in range(1, int(length.max())):
        act = k < length
        ids = offset + torch.minimum(torch.full_like(length, k), length - 1)
        b = {key: val[:, 0] for key, val in sd.gather(ids[:, None]).items()}
        out = stream.step(b, act)
        sel = ids[act]
        if cfg.submap_keyframes:
            logs["map_used"][sel] = out["map_used"][act]
            logs["map_diff"][sel] = out["map_diff"][act]
        for name in ("pred", "sigma", "weak_sigma_m", "eig_ratio", "icp_yaw_sigma_rad", "gyro_bias", "pairs", "standstill", "yaw_icp", "yaw_info"):
            logs[name][sel] = out[name][act]
