#!/usr/bin/env python3
"""Train and evaluate the streaming LiDAR-inertial network (v2).

One configuration, no grid search. Stages (outputs in
``learning/outputs/streaming_v2/``):

``icp``    runs the classical reference (``icp_odometry``) on train, validation
           and test and caches its estimates (``--icp submap``: the
           scan-to-submap variant);
``train``  trains ``StreamingPoseNet`` (``--hybrid``: the variant that takes the
           classical odometry as input, output in ``hybrid/``, or in
           ``hybrid_submap/`` with ``--icp submap``; needs the ``icp`` stage)
           with truncated back-propagation through
           time: every lane walks through a whole run chunk by chunk and keeps
           its state (half of the lanes start at the first scan of a run, the
           rest at a random scan with a fresh state); the checkpoint is the
           evaluation with the lowest validation segment drift;
``report`` scores, on exactly the same intervals, the hybrid and the pure
           streaming networks (one causal pass per run), the best grid-search
           model (its native 6-interval windows) and the classical reference,
           on validation and test, with ``odometry_metrics`` and draws the
           figures.

References
  [1] R. J. Williams, J. Peng, "An efficient gradient-based algorithm for
      on-line training of recurrent network trajectories", Neural Computation
      2(4), 1990 (truncated back-propagation through time).
  [2] A. Kendall, Y. Gal, "What Uncertainties Do We Need in Bayesian Deep
      Learning for Computer Vision?", NeurIPS 2017 (heteroscedastic Gaussian
      negative log-likelihood).
  [3] I. Loshchilov, F. Hutter, "Decoupled Weight Decay Regularization",
      ICLR 2019 (AdamW).
  [4] L. N. Smith, N. Topin, "Super-Convergence: Very Fast Training of Neural
      Networks Using Large Learning Rates", 2019 (one-cycle schedule).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn

import odometry_metrics as om
from architectures import PoseNet
from gridsearch_sqlite import DEFAULT_DB, INK, INK_2, LOSS_SCALE, ROOT, SERIES, SURFACE, _style
from icp_odometry import ICP_VARIANTS, IcpConfig, run_icp
from sqlite_streams import StreamData, load_split
from sqlite_windows import WindowSampler
from streaming_model import StreamingPoseNet
from train_pose_fusion import compose_planar

OUT = ROOT / "outputs" / "streaming_v2"
CACHE = ROOT / "outputs" / "cache_sqlite"
OLD_MODEL = ROOT / "outputs" / "gridsearch_sqlite" / "best_final" / "best_model.pt"
NOREG_MODEL = OUT / "attempt2_no_regularization" / "best_model.pt"
UNITS = (0.01, 0.01, 0.001)  # loss units: cm, cm, mrad
METHODS = ("hybrid_submap", "hybrid", "grid_best", "icp_submap", "icp", "streaming_v2", "streaming_v2_noreg")
HEADLINE_METHODS = ("hybrid_submap", "hybrid", "grid_best", "icp_submap", "icp", "streaming_v2")
LABEL = {"hybrid_submap": "hybrid: ICP scan-to-submap + network", "hybrid": "hybrid: ICP scan-to-scan + network",
         "grid_best": "grid-search best (windows)", "icp_submap": "classical ICP scan-to-submap + IMU", "icp": "classical ICP scan-to-scan + IMU",
         "streaming_v2": "pure network v2 (regularised)", "streaming_v2_noreg": "pure network v2, no regularisation"}
# Categorical slots 1-4 in their validated order (adjacent pairs only), one
# hue per family; the variants of a family share its hue and are hatched
# (bars) / dashed (lines): scan-to-scan vs submap, unregularised vs regularised.
YELLOW = "#eda100"  # slot 4
COLOR = {"hybrid_submap": SERIES[0], "hybrid": SERIES[0], "grid_best": SERIES[1], "icp_submap": SERIES[2], "icp": SERIES[2],
         "streaming_v2": YELLOW, "streaming_v2_noreg": YELLOW}
HATCH = {"hybrid": "////", "icp": "////", "streaming_v2_noreg": "////"}
DASH = {m: (0, (4, 2)) for m in HATCH}
NET_DIRS = {"hybrid_submap": OUT / "hybrid_submap", "hybrid": OUT / "hybrid", "streaming_v2": OUT}
GRAY = "#9a9890"


def _bar(ax, x, h, width, method, **kw):
    if method in HATCH:
        return ax.bar(x, h, width=width, facecolor=SURFACE, edgecolor=COLOR[method], hatch=HATCH[method], lw=1.2, label=LABEL[method], **kw)
    return ax.bar(x, h, width=width, color=COLOR[method], label=LABEL[method], **kw)


def subset_runs(data: dict[str, Any], step: int = 1, sensors: tuple[str, ...] = ()) -> tuple[dict[str, Any], np.ndarray]:
    """Every ``step``-th run of a split (of the ``sensors`` profiles, all when
    empty), re-indexed, and the selected global indices."""
    keep = [i for i, r in enumerate(data["runs"]) if not sensors or r["sensor"] in sensors][::step]
    n_total = data["target"].shape[0]
    sel = np.concatenate([np.arange(data["runs"][i]["offset"], data["runs"][i]["offset"] + data["runs"][i]["n"]) for i in keep])
    out = {k: (v[sel] if torch.is_tensor(v) and v.shape[:1] == (n_total,) else v) for k, v in data.items()}
    for k in ("range_max", "angle_min", "angle_increment", "time_increment_s", "lidar_xy", "lidar_sigma", "beam_time_frac"):
        out[k] = data[k][keep]
    runs, off = [], 0
    for i in keep:
        runs.append({**data["runs"][i], "offset": off})
        off += data["runs"][i]["n"]
    out["runs"] = runs
    out["run_index"] = torch.cat([torch.full((r["n"],), j, dtype=torch.int64) for j, r in enumerate(runs)])
    return out, sel


@dataclass(frozen=True)
class TrainConfig:
    hidden: int = 128
    lanes: int = 64
    chunk: int = 32
    epochs: int = 40
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-3
    fresh_start_prob: float = 0.5
    mirror_prob: float = 0.5  # lanes that drive a left-right mirrored world (whole walk)
    beam_dropout_max: float = 0.08  # extra missing LiDAR returns, drawn per chunk in [0, max]
    eval_every: int = 2
    seed: int = 23
    hybrid: bool = False  # classical odometry as input
    icp_variant: str = "scan"  # which classical odometry: "scan" (scan-to-scan) or "submap"
    sensors: tuple[str, ...] = ()  # sensor profiles to train and select on (empty: every profile of the database)


# ---------------------------------------------------------------- training
class Lanes:
    """Parallel walks through training runs, ``chunk`` intervals at a time."""

    def __init__(self, sd: StreamData, cfg: TrainConfig, rng: np.random.Generator) -> None:
        self.sd, self.cfg, self.rng = sd, cfg, rng
        self.n = np.array([r["n"] for r in sd.runs])
        self.offset = np.array([r["offset"] for r in sd.runs])
        usable = self.n - 1 >= cfg.chunk
        self.weights = np.where(usable, self.n - 1, 0) / np.where(usable, self.n - 1, 0).sum()
        self.run = np.zeros(cfg.lanes, dtype=np.int64)
        self.pos = np.zeros(cfg.lanes, dtype=np.int64)
        self.mirror = np.zeros(cfg.lanes, dtype=bool)
        for i in range(cfg.lanes):
            self._new(i)

    def _new(self, i: int) -> None:
        r = int(self.rng.choice(len(self.n), p=self.weights))
        self.run[i] = r
        self.mirror[i] = self.rng.random() < self.cfg.mirror_prob
        last_start = self.n[r] - self.cfg.chunk
        self.pos[i] = 1 if self.rng.random() < self.cfg.fresh_start_prob else int(self.rng.integers(1, last_start + 1))

    def next(self) -> tuple[Tensor, Tensor, Tensor]:
        reset = np.zeros(self.cfg.lanes, dtype=bool)
        for i in range(self.cfg.lanes):
            if self.pos[i] + self.cfg.chunk > self.n[self.run[i]]:
                self._new(i)
                reset[i] = True
        idx = self.offset[self.run][:, None] + self.pos[:, None] + np.arange(self.cfg.chunk)[None]
        self.pos += self.cfg.chunk
        dev = self.sd.device
        return torch.from_numpy(idx).to(dev), torch.from_numpy(reset).to(dev), torch.from_numpy(self.mirror.copy()).to(dev)


def augment(batch: dict[str, Tensor], mirror: Tensor, beam_dropout_max: float) -> dict[str, Tensor]:
    """Mirror flag for the model, mirrored labels, extra missing LiDAR returns."""
    out = dict(batch)
    out["mirror"] = mirror
    sign = 1.0 - 2.0 * mirror.float()[:, None]
    out["target"] = batch["target"] * torch.stack((torch.ones_like(sign), sign, sign), dim=-1)
    out["v_body"] = batch["v_body"] * torch.stack((torch.ones_like(sign), sign), dim=-1)
    out["bias_truth"] = batch["bias_truth"] * torch.stack((sign, torch.ones_like(sign), sign), dim=-1)  # gyro z, accel x, accel y
    lanes = mirror.shape[0]
    rate = torch.rand(lanes, 1, 1, device=mirror.device) * beam_dropout_max
    for key in ("prev", "cur"):
        keep = torch.rand_like(batch[f"{key}_ranges"]) >= rate
        out[f"{key}_valid"] = batch[f"{key}_valid"] & keep
        out[f"{key}_ranges"] = batch[f"{key}_ranges"] * out[f"{key}_valid"]
    return out


def _wrap(a: Tensor) -> Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def streaming_loss(out: dict[str, Tensor], batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
    delta, target = out["delta"], batch["target"]
    units = delta.new_tensor(UNITS)
    err = torch.cat((delta[..., :2] - target[..., :2], _wrap(delta[..., 2:] - target[..., 2:])), dim=-1)
    r = err / units
    lv = out["logvar"]
    nll = (0.5 * (torch.exp(-lv) * r.square() + lv)).mean()
    l1 = nn.functional.smooth_l1_loss(r, torch.zeros_like(r))
    comp = delta.new_zeros(())
    b, t, _ = delta.shape
    for h in (8, t):
        if t % h == 0:
            cp = compose_planar(delta.reshape(b * t // h, h, 3))
            ct = compose_planar(target.reshape(b * t // h, h, 3))
            ce = torch.cat((cp[:, :2] - ct[:, :2], _wrap(cp[:, 2:] - ct[:, 2:])), dim=-1) / units
            comp = comp + nn.functional.smooth_l1_loss(ce, torch.zeros_like(ce))
    vel = nn.functional.smooth_l1_loss((out["velocity"] - batch["v_body"]) / 0.05, torch.zeros_like(out["velocity"]))
    bias_units = delta.new_tensor([1.0e-3, 0.02, 0.02])
    bias = nn.functional.smooth_l1_loss((out["bias"] - batch["bias_truth"]) / bias_units, torch.zeros_like(out["bias"]))
    total = nll + 0.1 * l1 + 0.05 * comp + 0.1 * vel + 0.2 * bias
    parts = {"loss": total, "nll": nll, "comp": comp, "vel": vel, "bias": bias, "trans_mae_cm": 100 * torch.linalg.vector_norm(err[..., :2], dim=-1).mean()}
    return total, {k: float(v.detach()) for k, v in parts.items()}


@torch.no_grad()
def stream_predict(model: StreamingPoseNet, sd: StreamData, chunk: int = 64) -> dict[str, Tensor]:
    """One causal pass over every run of the split, from its first scan."""
    model.eval()
    runs = len(sd.runs)
    n_total = sd.target.shape[0]
    pred = torch.full((n_total, 3), float("nan"), device=sd.device)
    sigma = torch.full((n_total, 3), float("nan"), device=sd.device)
    bias = torch.full((n_total, 3), float("nan"), device=sd.device)
    gain = torch.full((n_total, 5), float("nan"), device=sd.device)
    state = model.initial_state(runs, sd.device)
    units = pred.new_tensor(UNITS)
    for s in range(1, int(sd.length.max()), chunk):
        ks = torch.arange(s, s + chunk, device=sd.device)
        idx = sd.offset[:, None] + torch.minimum(ks[None], sd.length[:, None] - 1)
        out, state = model(sd.gather(idx), state)
        ok = ks[None] < sd.length[:, None]
        pred[idx[ok]] = out["delta"][ok]
        sigma[idx[ok]] = torch.exp(0.5 * out["logvar"][ok]) * units
        bias[idx[ok]] = out["bias"][ok]
        gain[idx[ok]] = out["gain"][ok]
    return {"pred": pred.cpu(), "sigma": sigma.cpu(), "bias": bias.cpu(), "gain": gain.cpu()}


def train(cfg: TrainConfig, db: Path, out: Path, log, icp_dir: Path = OUT) -> None:
    torch.manual_seed(cfg.seed)
    dev = torch.device("cuda")
    data, sel = {}, {}
    for s in ("train", "validation"):
        full = load_split(db, CACHE, s)
        data[s], sel[s] = subset_runs(full, 1, cfg.sensors) if cfg.sensors else (full, None)
        if cfg.hybrid and sel[s] is not None:
            load_or_run_icp(s, StreamData(full, dev), icp_dir, log, cfg.icp_variant, db)  # cached on the whole split
        del full
    sds = {s: StreamData(d, dev) for s, d in data.items()}
    if cfg.hybrid:
        for s, sd in sds.items():
            res = load_or_run_icp(s, sd if sel[s] is None else None, icp_dir, log, cfg.icp_variant, db)
            sd.attach_icp(res if sel[s] is None else {k: (v[sel[s]] if torch.is_tensor(v) else v) for k, v in res.items()})
    if cfg.sensors:
        log(f"[train] sensor profiles {list(cfg.sensors)}: {len(data['train']['runs'])} train / {len(data['validation']['runs'])} validation runs")
    truth_val = om.split_truth(data["validation"])
    model = StreamingPoseNet(cfg.hidden, hybrid=cfg.hybrid).to(dev)
    params = sum(p.numel() for p in model.parameters())
    steps_per_epoch = math.ceil(sds["train"].intervals / (cfg.lanes * cfg.chunk))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.learning_rate, total_steps=cfg.epochs * steps_per_epoch, pct_start=0.1)
    lanes = Lanes(sds["train"], cfg, np.random.default_rng(cfg.seed))
    state = model.initial_state(cfg.lanes, dev)
    log(f"[train] {params} parameters, {steps_per_epoch} steps/epoch of {cfg.lanes} lanes x {cfg.chunk} intervals, {cfg.epochs} epochs")
    history, best = [], (math.inf, 0)
    t_start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        sums: dict[str, float] = {}
        for _ in range(steps_per_epoch):
            idx, reset, mirror = lanes.next()
            state = model.reset({k: v.detach() for k, v in state.items()}, reset)
            batch = augment(sds["train"].gather(idx), mirror, cfg.beam_dropout_max)
            result, state = model(batch, state)
            loss, parts = streaming_loss(result, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v / steps_per_epoch
        rec = {"epoch": epoch, "seconds": time.time() - t0, **{f"train_{k}": v for k, v in sums.items()}}
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            val = stream_predict(model, sds["validation"])
            m = om.evaluate(val["pred"].numpy(), truth_val, sigma=val["sigma"].numpy(), groups=())["all"]
            rec.update({f"val_{k}": v for k, v in m.items()})
            if m["t_rel_pct"] < best[0]:
                best = (m["t_rel_pct"], epoch)
                torch.save({"model": model.state_dict(), "config": asdict(cfg), "epoch": epoch, "val": m, "db": _db_tag(db)}, out / "best_model.pt")
            log(f"[train] epoch {epoch}/{cfg.epochs} {rec['seconds']:.0f}s loss {sums['loss']:.3f} (nll {sums['nll']:.2f} comp {sums['comp']:.2f} vel {sums['vel']:.2f} bias {sums['bias']:.2f}) trans {sums['trans_mae_cm']:.2f} cm | val speed {m['speed_err_cmps']:.2f} cm/s "
                f"rel {m['rel_err_pct']:.2f} % yawrate {m['yawrate_err_dps']:.3f} deg/s rpe1s {m['rpe_1s_cm']:.2f} cm t_rel {m['t_rel_pct']:.2f} % r_rel {m['r_rel_degpm']:.3f} deg/m")
        else:
            log(f"[train] epoch {epoch}/{cfg.epochs} {rec['seconds']:.0f}s loss {sums['loss']:.3f} trans {sums['trans_mae_cm']:.2f} cm")
        history.append(rec)
        (out / "history.json").write_text(json.dumps({"config": asdict(cfg), "parameters": params, "best_epoch": best[1], "history": history}, indent=1), encoding="utf-8")
    log(f"[train] done in {time.time() - t_start:.0f} s, best epoch {best[1]} (validation segment drift {best[0]:.3f} %)")
    plot_history(history, best[1], out / "training_curves.png")


# ------------------------------------------------------- other estimators
@torch.no_grad()
def grid_best_predict(data: dict[str, Any], dev: torch.device) -> dict[str, Tensor]:
    """The best grid-search model in its native mode (independent 6-interval windows)."""
    ck = torch.load(OLD_MODEL, map_location=dev, weights_only=False)
    cfg = ck["config"]
    model = PoseNet(cfg["architecture"], cfg["hidden"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    sampler = WindowSampler(data, dev, cfg["steps"], cfg["context"])
    starts = sampler.starts(cfg["steps"])
    n_total = data["target"].shape[0]
    pred = torch.full((n_total, 3), float("nan"))
    sigma = torch.full((n_total, 3), float("nan"))
    scale = torch.tensor(LOSS_SCALE)
    for i in range(0, len(starts), 256):
        b = sampler.batch(starts[i : i + 256])
        out = model(b["lidar"], b["imu"], b["imu_lengths"], b["dt"])
        idx = b["index"].reshape(-1).cpu()
        pred[idx] = out["delta"].reshape(-1, 3).cpu()
        sigma[idx] = torch.exp(0.5 * out["logvar"].reshape(-1, 3).cpu()) / scale
    return {"pred": pred, "sigma": sigma}




def _same_icp(cached: dict[str, Any], cfg: IcpConfig) -> bool:
    """Result-relevant fields equal (fields added later count with their default)."""
    default = {**asdict(IcpConfig()), "version": 1}  # caches without a version predate version 2
    return all(cached.get(k, default[k]) == v for k, v in asdict(cfg).items() if k != "max_lanes")


def _db_tag(db: Path) -> dict[str, Any]:
    return {"name": db.name, "bytes": db.stat().st_size}


def load_or_run_icp(split: str, sd: StreamData | None, out: Path, log, variant: str = "scan", db: Path = DEFAULT_DB) -> dict[str, Any]:
    """Cached classical odometry of a split; a cache is reused only for the
    same ICP configuration and the same database (caches written before the
    database tag belong to the default, v1 database)."""
    cfg = ICP_VARIANTS[variant]
    path = out / (f"icp_{split}.pt" if variant == "scan" else f"icp_{variant}_{split}.pt")
    if path.exists():
        cached = torch.load(path, weights_only=False)
        same_db = cached.get("db", _db_tag(DEFAULT_DB) if DEFAULT_DB.exists() else None) == _db_tag(db)
        if _same_icp(cached["config"], cfg) and same_db:
            return cached
        if not same_db:
            log(f"[icp] {path.name} was computed on another database: recomputing")
    if sd is None:
        raise FileNotFoundError(f"{path}: run the icp stage first")
    t0 = time.time()
    res = run_icp(sd, cfg, log=log)
    res["db"] = _db_tag(db)
    log(f"[icp] {split}: {time.time() - t0:.0f} s")
    torch.save(res, path)
    return res


# ------------------------------------------------------------------ report
def _load_net(path: Path, dev: torch.device, regularize: bool) -> tuple[StreamingPoseNet, dict[str, Any]]:
    ck = torch.load(path, map_location=dev, weights_only=False)
    model = StreamingPoseNet(ck["config"]["hidden"], regularize=regularize, hybrid=ck["config"].get("hybrid", False)).to(dev)
    model.load_state_dict(ck["model"])
    return model, ck


def report(db: Path, out: Path, log, sensors: tuple[str, ...] = ()) -> None:
    dev = torch.device("cuda")
    nets, summary = {}, {"icp_configs": {k: asdict(v) for k, v in ICP_VARIANTS.items()},
                         "selection_metric": "validation t_rel_pct (segment drift, mean over 2-40 m)", "networks": {}, "splits": {}}
    variant_of = {"hybrid": "scan", "hybrid_submap": "submap"}
    net_dirs = {"hybrid_submap": out / "hybrid_submap", "hybrid": out / "hybrid", "streaming_v2": out}
    v1_study = out.resolve() == OUT.resolve()
    for name, path, regularize in (("hybrid_submap", net_dirs["hybrid_submap"] / "best_model.pt", True), ("hybrid", net_dirs["hybrid"] / "best_model.pt", True),
                                   ("streaming_v2", net_dirs["streaming_v2"] / "best_model.pt", True),
                                   ("streaming_v2_noreg", NOREG_MODEL if v1_study else out / "none", False)):
        if path.exists():
            nets[name], ck = _load_net(path, dev, regularize)
            summary["networks"][name] = {"checkpoint": str(path.relative_to(ROOT)), "selected_epoch": ck["epoch"], "config": ck["config"],
                                         "parameters": sum(p.numel() for p in nets[name].parameters())}
    icp_variants = [v for v in ("scan", "submap") if (out / (f"icp_test.pt" if v == "scan" else f"icp_{v}_test.pt")).exists()]
    # The grid-search model belongs to the v1 study (its own sensors); a
    # report on another database compares the networks and the classical odometry.
    methods = [m for m in METHODS if m in nets or (m == "grid_best" and v1_study) or (m == "icp" and "scan" in icp_variants)
               or (m == "icp_submap" and "submap" in icp_variants)]
    # Overfitting: the same networks on a quarter of the training runs.
    train_sub, sel = subset_runs(load_split(db, CACHE, "train"), 4, sensors)
    sd_train = StreamData(train_sub, dev)
    summary["train_subset"] = {}
    for m, net in nets.items():
        if m in variant_of:
            icp_train = load_or_run_icp("train", None, out, log, variant_of[m], db)
            sd_train.attach_icp({k: (v[sel] if torch.is_tensor(v) else v) for k, v in icp_train.items()})
        summary["train_subset"][m] = om.evaluate(stream_predict(net, sd_train, chunk=16)["pred"].numpy(), om.split_truth(train_sub), groups=())["all"]
    del sd_train
    runs_for_plot = {}
    for split in ("validation", "test"):
        data, sel_s = subset_runs(load_split(db, CACHE, split), 1, sensors)
        sd = StreamData(data, dev)
        truth = om.split_truth(data)
        icps = {v: {k: (x[sel_s] if torch.is_tensor(x) else x) for k, x in load_or_run_icp(split, None, out, log, v, db).items()} for v in icp_variants}
        est = {}
        for m, net in nets.items():
            if m in variant_of:
                sd.attach_icp(icps[variant_of[m]])
            est[m] = stream_predict(net, sd)
        if "grid_best" in methods:
            est["grid_best"] = grid_best_predict(data, dev)
        if "scan" in icps:
            est["icp"] = icps["scan"]
        if "submap" in icps:
            est["icp_submap"] = icps["submap"]
        preds = {m: est[m]["pred"].numpy().astype(np.float64) for m in methods}
        common = np.logical_and.reduce([np.isfinite(p).all(axis=1) for p in preds.values()])
        degenerate = next(iter(icps.values()))["eig_ratio"].numpy() < IcpConfig().degenerate_ratio
        res = {m: om.evaluate(preds[m], truth, mask=common, sigma=est[m]["sigma"].numpy(), flags={"corridor": degenerate}) for m in methods}
        gyro = np.column_stack((truth.target[:, :2], data["gyro_integral"].numpy()))  # heading-only reference (exact translation)
        res["raw_gyro_heading"] = {"all": {k: v for k, v in om.evaluate(gyro, truth, mask=common, groups=())["all"].items() if "yaw" in k or k.startswith("r_rel")}}
        summary["splits"][split] = {
            "intervals_scored": int(common.sum()), "corridor_fraction_pct": 100 * float(degenerate[common].mean()),
            "gyro_bias_error_dps": {**{m: _bias_err(est[m]["bias"].numpy()[:, 0], data, common) for m in nets},
                                    **{m: _bias_err(est[m]["gyro_bias"].numpy(), data, common) for m in ("icp", "icp_submap") if m in est},
                                    "no_estimate": _bias_err(np.zeros(len(common)), data, common)},
            **res,
        }
        runs_for_plot[split] = (truth, preds, common)
        log(f"[report] {split}: " + " | ".join(f"{m}: speed {res[m]['all']['speed_err_cmps']:.2f} cm/s, t_rel {res[m]['all']['t_rel_pct']:.2f} %, r_rel {res[m]['all']['r_rel_degpm']:.3f} deg/m" for m in methods))
    (out / "report.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    plot_report(summary, runs_for_plot, methods, out)
    best = min((m for m in methods), key=lambda m: summary["splits"]["validation"][m]["all"]["t_rel_pct"])
    summary["best_on_validation"] = best
    (out / "report.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    log(f"[report] best on validation (segment drift): {best}")


def _bias_err(est: np.ndarray, data: dict[str, Any], mask: np.ndarray) -> float:
    true = data["bias_truth"].numpy()[:, 0]
    return math.degrees(float(np.nanmean(np.abs(est[mask] - true[mask]))))


# ------------------------------------------------------------------- plots
def plot_history(history: list[dict[str, Any]], best_epoch: int, path: Path) -> None:
    ev = [h for h in history if "val_t_rel_pct" in h]
    panels = (
        ("train_loss", "Training loss", "loss", history),
        ("val_speed_err_cmps", "Validation speed error", "cm/s", ev),
        ("val_t_rel_pct", "Validation segment drift (2-40 m)", "% of distance", ev),
        ("val_yawrate_err_dps", "Validation heading-rate error", "deg/s", ev),
    )
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8), facecolor=SURFACE)
    for ax, (key, title, unit, rows) in zip(axes, panels):
        ax.plot([h["epoch"] for h in rows], [h[key] for h in rows], color=SERIES[0], lw=2, marker="o" if rows is ev else None, ms=4)
        ax.axvline(best_epoch, color=INK_2, ls=":", lw=1)
        _style(ax, title, "epoch", unit)
    axes[2].annotate("selected", (best_epoch, min(h["val_t_rel_pct"] for h in ev)), textcoords="offset points", xytext=(6, 8), fontsize=8, color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_report(summary: dict[str, Any], runs: dict[str, Any], methods: list[str], out: Path) -> None:
    splits = ("validation", "test")
    all_methods = methods
    methods = [m for m in HEADLINE_METHODS if m in all_methods]
    # 1. Headline metrics, one panel per metric (single axis each).
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5), facecolor=SURFACE)
    width = 0.8 / len(methods)
    for ax, (key, title, unit) in zip(axes.flat, om.HEADLINE):
        for i, m in enumerate(methods):
            vals = [summary["splits"][s][m]["all"][key] for s in splits]
            bars = _bar(ax, np.arange(2) + (i - (len(methods) - 1) / 2) * width, vals, width * 0.92, m)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, val, f" {val:.3g}", ha="center", va="bottom", fontsize=7, color=INK, rotation=90)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.18)  # room for the rotated value labels
        if key in ("yawrate_err_dps", "r_rel_degpm"):
            for x, s in enumerate(splits):
                g = summary["splits"][s]["raw_gyro_heading"]["all"][key]
                ax.hlines(g, x - 0.45, x + 0.45, color=GRAY, ls="--", lw=1.2)
                ax.text(x - 0.45, g, "raw gyro", fontsize=7, color=INK_2, va="bottom", ha="left",
                        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.5))
        ax.set_xticks(range(2))
        ax.set_xticklabels(splits, color=INK_2)
        better = "higher is better" if key.startswith("acc") else "lower is better"
        _style(ax, f"{title} ({better})", "", unit)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Same intervals for every method; test = unseen track + unseen sensor/speed combinations", fontsize=12, color=INK, x=0.01, y=0.995, ha="left")
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.965), ncol=len(methods), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "comparison_headline.png", dpi=95, facecolor=fig.get_facecolor())
    plt.close(fig)

    # 2. Overfitting: training runs vs unseen splits, for both network variants.
    nets = [m for m in all_methods if m in summary["train_subset"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), facecolor=SURFACE)
    for ax, (key, title, unit) in zip(axes, (("speed_err_cmps", "Speed error", "cm/s"), ("t_rel_pct", "Segment drift (2-40 m)", "%"))):
        groups = ["train (1/4 of the runs)", "validation", "test"]
        w = 0.8 / len(nets)
        for i, m in enumerate(nets):
            vals = [summary["train_subset"][m][key]] + [summary["splits"][s][m]["all"][key] for s in splits]
            bars = _bar(ax, np.arange(3) + (i - (len(nets) - 1) / 2) * w, vals, w * 0.92, m)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3g}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(range(3))
        ax.set_xticklabels(groups, color=INK_2)
        _style(ax, f"{title}: seen vs unseen runs (streaming, lower is better)", "", unit)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "overfitting_gap.png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)

    # 3. Error growth with horizon and with distance (test).
    test = summary["splits"]["test"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.4), facecolor=SURFACE)
    for m in methods:
        a = test[m]["all"]
        style = dict(color=COLOR[m], lw=2, marker="o", ms=4, label=LABEL[m], ls=DASH.get(m, "-"))
        axes[0].plot(om.HORIZONS_S, [a[f"rpe_{h:g}s_cm"] for h in om.HORIZONS_S], **style)
        axes[1].plot(om.HORIZONS_S, [a[f"rpe_{h:g}s_deg"] for h in om.HORIZONS_S], **style)
        axes[2].plot(om.SEGMENTS_M, [a[f"t_rel_{L:g}m_pct"] for L in om.SEGMENTS_M], **style)
    _style(axes[0], "Test: relative translation error vs horizon", "horizon [s]", "error [cm]")
    _style(axes[1], "Test: relative heading error vs horizon", "horizon [s]", "error [deg]")
    _style(axes[2], "Test: segment drift vs segment length", "segment length [m]", "error [% of length]")
    for ax in axes:
        ax.set_xscale("log")
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "error_vs_horizon.png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)

    # 4. Breakdowns (test): sensor profile, motion profile, corridor geometry.
    fig, axes = plt.subplots(2, 3, figsize=(20, 8.5), facecolor=SURFACE)
    for col, key in enumerate(("sensor", "motion", "flag_corridor")):
        groups = list(test[methods[0]][key].keys())
        for row, (metric, unit, title) in enumerate((("speed_err_cmps", "cm/s", "speed error"), ("yawrate_err_dps", "deg/s", "heading-rate error"))):
            ax = axes[row, col]
            for i, m in enumerate(methods):
                vals = [test[m][key][g].get(metric, np.nan) for g in groups]
                _bar(ax, np.arange(len(groups)) + (i - (len(methods) - 1) / 2) * width, vals, width * 0.92, m)
            names = [{"true": "corridor-like (ICP degenerate)", "false": "well constrained"}.get(g, g) for g in groups]
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8, color=INK_2)
            _style(ax, f"Test {title} by {key.replace('flag_', '')}", "", unit)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "breakdown_test.png", dpi=95, facecolor=fig.get_facecolor())
    plt.close(fig)

    # 5. Trajectories of test runs (dead reckoning over the scored block).
    truth, preds, common = runs["test"]
    picks, seen = [], set()
    for r in sorted(truth.runs, key=lambda r: (r["track"] != "test_unseen", r["sensor"], r["motion"])):
        if (r["track"], r["sensor"]) not in seen:
            picks.append(r)
            seen.add((r["track"], r["sensor"]))
        if len(picks) == 6:
            break
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor=SURFACE)
    for ax, r in zip(axes.flat, picks):
        sl = slice(r["offset"], r["offset"] + r["n"])
        ok = common[sl]
        gt = om.compose(truth.target[sl][ok])
        ax.plot(gt[:, 0], gt[:, 1], color=INK, lw=2.4, label="ground truth")
        for m in methods:
            if m in HATCH:  # one method per family keeps the map readable
                continue
            est = om.compose(preds[m][sl][ok])
            ax.plot(est[:, 0], est[:, 1], color=COLOR[m], lw=1.4, label=LABEL[m])
        ax.set_aspect("equal")
        _style(ax, f"{r['track']} · {r['motion']} · {r['sensor']}", "x [m]", "y [m]")
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Test runs integrated from their first scored scan (no loop closure, no map)", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "test_trajectories.png", dpi=90, facecolor=fig.get_facecolor())
    plt.close(fig)


# --------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("icp", "train", "report"))
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--hybrid", action="store_true", help="train the variant with the classical odometry as input (output in hybrid/)")
    ap.add_argument("--icp", choices=tuple(ICP_VARIANTS), default="scan", help="classical odometry variant (icp stage and --hybrid)")
    ap.add_argument("--sensors", nargs="*", default=[], help="train / report on these sensor profiles only (e.g. APEX_real)")
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(4)
    stage_out = args.out
    if args.stage == "train" and args.hybrid:
        stage_out = args.out / ("hybrid" if args.icp == "scan" else f"hybrid_{args.icp}")
    stage_out.mkdir(parents=True, exist_ok=True)
    log_path = stage_out / f"log_{args.stage}.txt"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    if args.stage == "icp":
        for split in ("validation", "test", "train"):  # train: input of the hybrid network
            load_or_run_icp(split, StreamData(load_split(args.db, CACHE, split), torch.device("cuda")), args.out, log, args.icp, args.db)
    elif args.stage == "train":
        train(TrainConfig(epochs=args.epochs, hybrid=args.hybrid, icp_variant=args.icp, sensors=tuple(args.sensors)), args.db, stage_out, log, icp_dir=args.out)
    else:
        report(args.db, args.out, log, tuple(args.sensors))


if __name__ == "__main__":
    main()
