#!/usr/bin/env python3
"""Grid search of the LiDAR–IMU pose architectures on the multi-scenario dataset.

Stages (all outputs in ``learning/outputs/gridsearch_sqlite/``):

``train``   trains grid configurations (architecture x hidden size x learning
            rate); the checkpoint of each one is the epoch with the lowest
            validation RPE; train/validation curves are saved per config.
``report``  compares every configuration on validation only, picks the best,
            and draws the comparison figures.
``final``   retrains the best configuration longer, then evaluates it once on
            validation and on the test split (unseen track + unseen
            sensor/speed combinations) with trajectory and breakdown plots.

Supervision is the exact simulator pose (relative increments between scans);
the network inputs are only the noisy LiDAR scans, the IMU samples of each
interval and their real timestamps. The loss is the supervised loss of
``train_pose_fusion.py`` with identical weights.

Metrics (per scan interval unless stated):
  trans_mae_cm / trans_rmse_cm   translation error of the increment
  yaw_mae_deg / yaw_rmse_deg     heading error of the increment
  accuracy_pct                   intervals with error < 2 cm AND < 0.5 deg
  accuracy_loose_pct             intervals with error < 5 cm AND < 1 deg
  rpe_trans_cm / rpe_yaw_deg     error of the pose composed over one window
                                 (6 intervals, ~0.3-0.6 s)
  drift_pct / ate_rmse_m         dead reckoning over each whole run (two laps)
  gyro_yaw_mae_deg               raw gyro integral, reference for the heading
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import itertools
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

from architectures import ARCHITECTURES, PoseNet
from sqlite_windows import WindowSampler, load_or_build, split_statistics
from train_pose_fusion import compose_planar

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT.parent / "data" / "multiscenario_pose" / "pose_dataset.sqlite3"
OUT = ROOT / "outputs" / "gridsearch_sqlite"
LOSS_SCALE = (12.0, 18.0, 8.0)
ACC_STRICT = (0.02, math.radians(0.5))
ACC_LOOSE = (0.05, math.radians(1.0))

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical slots (one per architecture)
INK, INK_2, GRID_C, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ARCH_COLOR = dict(zip(ARCHITECTURES, SERIES))
ARCH_LABEL = {
    "single_context_direct": "single GRU, direct",
    "single_context_velocity": "single GRU + velocity",
    "dual_pair_context_velocity": "pair+context GRU + velocity",
}


@dataclass(frozen=True)
class RunConfig:
    architecture: str
    hidden: int
    learning_rate: float
    epochs: int = 8
    batch_size: int = 64
    weight_decay: float = 1.0e-5
    steps: int = 6
    context: int = 5
    train_stride: int = 6  # = steps: every interval is a target once per epoch
    seed: int = 23

    @property
    def name(self) -> str:
        return f"{self.architecture}__h{self.hidden}__lr{self.learning_rate:g}"


GRID = {
    "architecture": list(ARCHITECTURES),
    "hidden": [64, 96, 128],
    "learning_rate": [1.0e-3, 3.0e-4],
}


def grid_configs(epochs: int) -> list[RunConfig]:
    return [RunConfig(a, h, lr, epochs=epochs) for a, h, lr in itertools.product(GRID["architecture"], GRID["hidden"], GRID["learning_rate"])]


# --------------------------------------------------------------------- loss
def supervised_loss(out: dict[str, Tensor], batch: dict[str, Tensor]) -> Tensor:
    """``train_pose_fusion.pose_loss`` with exact targets (quality = 1)."""
    delta, logvar, bias, target = out["delta"], out["logvar"], out["bias"], batch["target"]
    scale = delta.new_tensor(LOSS_SCALE)
    residual = (delta - target) * scale
    nll = (0.5 * (torch.exp(-logvar) * residual.square() + logvar)).mean()
    gyro_delta = batch["gyro_integral"] - bias[..., 5] * batch["dt"]
    imu_yaw = nn.functional.smooth_l1_loss(delta[..., 2], gyro_delta, beta=0.03)
    motion_smooth = (delta[:, 1:] - delta[:, :-1]).square().mean()
    bias_smooth = (bias[:, 1:] - bias[:, :-1]).square().mean()
    bias_prior = bias.square().mean()
    total = nll + 10.0 * imu_yaw + 0.05 * motion_smooth + 0.04 * bias_smooth + 0.005 * bias_prior
    if "velocity" in out:
        target_velocity = target[..., :2] / batch["dt"].unsqueeze(-1).clamp_min(1.0e-3)
        total = total + 0.20 * nn.functional.smooth_l1_loss(out["velocity"], target_velocity, beta=0.05)
    total = total + 0.10 * nn.functional.smooth_l1_loss(compose_planar(delta) * scale, compose_planar(target) * scale, beta=0.10)
    return total


def step_errors(delta: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    trans = torch.linalg.vector_norm(delta[..., :2] - target[..., :2], dim=-1)
    yaw = torch.abs(torch.atan2(torch.sin(delta[..., 2] - target[..., 2]), torch.cos(delta[..., 2] - target[..., 2])))
    return trans, yaw


# --------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model: PoseNet, sampler: WindowSampler, batch_size: int = 256, with_runs: bool = False) -> dict[str, Any]:
    model.eval()
    starts = sampler.starts(sampler.steps)
    trans_all, yaw_all, gyro_all, rpe_t, rpe_y, loss_sum, n_batches = [], [], [], [], [], 0.0, 0
    preds = torch.empty((len(starts), sampler.steps, 3), device=sampler.device)
    for i in range(0, len(starts), batch_size):
        batch = sampler.batch(starts[i : i + batch_size])
        out = model(batch["lidar"], batch["imu"], batch["imu_lengths"], batch["dt"])
        loss_sum += float(supervised_loss(out, batch))
        n_batches += 1
        delta, target = out["delta"], batch["target"]
        preds[i : i + len(delta)] = delta
        t_err, y_err = step_errors(delta, target)
        trans_all.append(t_err.flatten())
        yaw_all.append(y_err.flatten())
        g = batch["gyro_integral"] - target[..., 2]
        gyro_all.append(torch.abs(torch.atan2(torch.sin(g), torch.cos(g))).flatten())
        cp, ct = compose_planar(delta), compose_planar(target)
        rpe_t.append(torch.linalg.vector_norm(cp[:, :2] - ct[:, :2], dim=-1))
        dy = cp[:, 2] - ct[:, 2]
        rpe_y.append(torch.abs(torch.atan2(torch.sin(dy), torch.cos(dy))))
    trans, yaw, gyro = torch.cat(trans_all), torch.cat(yaw_all), torch.cat(gyro_all)
    rt, ry = torch.cat(rpe_t), torch.cat(rpe_y)
    metrics = {
        "loss": loss_sum / max(n_batches, 1),
        "trans_mae_cm": float(trans.mean()) * 100,
        "trans_rmse_cm": float(trans.square().mean().sqrt()) * 100,
        "yaw_mae_deg": math.degrees(float(yaw.mean())),
        "yaw_rmse_deg": math.degrees(float(yaw.square().mean().sqrt())),
        "accuracy_pct": 100.0 * float(((trans < ACC_STRICT[0]) & (yaw < ACC_STRICT[1])).float().mean()),
        "accuracy_loose_pct": 100.0 * float(((trans < ACC_LOOSE[0]) & (yaw < ACC_LOOSE[1])).float().mean()),
        "rpe_trans_cm": float(rt.mean()) * 100,
        "rpe_yaw_deg": math.degrees(float(ry.mean())),
        "gyro_yaw_mae_deg": math.degrees(float(gyro.mean())),
        "intervals": int(trans.numel()),
    }
    if with_runs:
        metrics["runs"] = run_trajectories(sampler, starts, preds)
        per_run = metrics["runs"]
        metrics["drift_pct_median"] = float(np.median([r["drift_pct"] for r in per_run]))
        metrics["ate_rmse_m_median"] = float(np.median([r["ate_rmse_m"] for r in per_run]))
        metrics["final_yaw_err_deg_median"] = float(np.median([r["final_yaw_err_deg"] for r in per_run]))
        metrics["per_interval"] = {"trans_m": trans.cpu().numpy(), "yaw_rad": yaw.cpu().numpy()}
    return metrics


def _integrate(pose0: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    poses = np.empty((len(deltas) + 1, 3))
    poses[0] = pose0
    x, y, yaw = pose0
    for i, (dx, dy, dyaw) in enumerate(deltas):
        c, s = math.cos(yaw), math.sin(yaw)
        x, y, yaw = x + c * dx - s * dy, y + s * dx + c * dy, yaw + dyaw
        poses[i + 1] = (x, y, yaw)
    return poses


def run_trajectories(sampler: WindowSampler, starts: Tensor, preds: Tensor) -> list[dict[str, Any]]:
    """Dead reckoning over every run with the non-overlapping windows."""
    starts_np = starts.cpu().numpy()
    preds_np = preds.cpu().numpy()
    targets = sampler.target.cpu().numpy()
    out = []
    for run in sampler.runs:
        sel = (starts_np >= run["offset"]) & (starts_np < run["offset"] + run["n"])
        if not np.any(sel):
            continue
        s0 = int(starts_np[sel][0])
        deltas = preds_np[sel].reshape(-1, 3)
        truth_d = np.concatenate([targets[s : s + sampler.steps] for s in starts_np[sel]])
        pose0 = np.zeros(3)
        est = _integrate(pose0, deltas)
        gt = _integrate(pose0, truth_d)
        err = np.linalg.norm(est[:, :2] - gt[:, :2], axis=1)
        dist = float(np.sum(np.linalg.norm(truth_d[:, :2], axis=1)))
        yaw_err = (est[-1, 2] - gt[-1, 2] + math.pi) % (2 * math.pi) - math.pi
        out.append({
            "run_key": run["run_key"], "track": run["track"], "motion": run["motion"], "sensor": run["sensor"],
            "start_index": s0 - run["offset"], "distance_m": dist, "ate_rmse_m": float(np.sqrt(np.mean(err**2))),
            "final_err_m": float(err[-1]), "drift_pct": 100.0 * float(err[-1]) / max(dist, 1e-6),
            "final_yaw_err_deg": abs(math.degrees(yaw_err)), "est": est, "gt": gt,
        })
    return out


# ----------------------------------------------------------------- training
def train_config(cfg: RunConfig, samplers: dict[str, WindowSampler], out_dir: Path, log=print) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = samplers["train"].device
    model = PoseNet(cfg.architecture, cfg.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    phase_rng = np.random.default_rng(cfg.seed)
    train_starts = samplers["train"].starts(cfg.train_stride)
    steps_per_epoch = math.ceil(len(train_starts) / cfg.batch_size) + 1
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.learning_rate, total_steps=cfg.epochs * steps_per_epoch, pct_start=0.15)
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    history = []
    best = (math.inf, -1)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_starts = samplers["train"].starts(cfg.train_stride, rng=phase_rng)
        perm = train_starts[torch.randperm(len(train_starts), generator=generator).to(device)]
        sums = {"loss": 0.0, "trans": 0.0, "yaw": 0.0, "acc": 0.0, "n": 0}
        t0 = time.time()
        for i in range(0, len(perm), cfg.batch_size):
            batch = samplers["train"].batch(perm[i : i + cfg.batch_size])
            out = model(batch["lidar"], batch["imu"], batch["imu_lengths"], batch["dt"])
            loss = supervised_loss(out, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                t_err, y_err = step_errors(out["delta"], batch["target"])
                sums["loss"] += float(loss) * len(t_err)
                sums["trans"] += float(t_err.mean()) * len(t_err)
                sums["yaw"] += float(y_err.mean()) * len(t_err)
                sums["acc"] += float(((t_err < ACC_STRICT[0]) & (y_err < ACC_STRICT[1])).float().mean()) * len(t_err)
                sums["n"] += len(t_err)
        val = evaluate(model, samplers["validation"])
        n = max(sums["n"], 1)
        rec = {
            "epoch": epoch, "seconds": time.time() - t0,
            "train_loss": sums["loss"] / n, "train_trans_mae_cm": 100 * sums["trans"] / n,
            "train_yaw_mae_deg": math.degrees(sums["yaw"] / n), "train_accuracy_pct": 100 * sums["acc"] / n,
            **{f"val_{k}": v for k, v in val.items()},
        }
        history.append(rec)
        log(f"[{cfg.name}] epoch {epoch}/{cfg.epochs} {rec['seconds']:.0f}s train loss {rec['train_loss']:.3f} acc {rec['train_accuracy_pct']:.1f}% | "
            f"val loss {val['loss']:.3f} acc {val['accuracy_pct']:.1f}% trans {val['trans_mae_cm']:.2f} cm yaw {val['yaw_mae_deg']:.3f} deg rpe {val['rpe_trans_cm']:.2f} cm")
        if val["rpe_trans_cm"] < best[0]:
            best = (val["rpe_trans_cm"], epoch)
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "epoch": epoch, "val": val}, out_dir / "best_model.pt")
    ck = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    val = evaluate(model, samplers["validation"], with_runs=True)
    result = {
        "name": cfg.name, "config": asdict(cfg), "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": ck["epoch"], "train_seconds": time.time() - t_start,
        "validation": {k: v for k, v in val.items() if k not in ("runs", "per_interval")},
        "history": history,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    plot_curves(history, cfg.name, out_dir / "training_curves.png", ck["epoch"])
    return result


# -------------------------------------------------------------------- plots
def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(color=GRID_C, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_C)


def plot_curves(history: list[dict[str, Any]], title: str, path: Path, best_epoch: int) -> None:
    ep = [h["epoch"] for h in history]
    panels = (
        ("loss", "Loss", "loss"),
        ("accuracy_pct", "Accuracy (error < 2 cm and < 0.5°)", "% of intervals"),
        ("trans_mae_cm", "Translation error per interval", "MAE [cm]"),
        ("yaw_mae_deg", "Heading error per interval", "MAE [deg]"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2), facecolor=SURFACE)
    for ax, (key, name, ylabel) in zip(axes, panels):
        ax.plot(ep, [h[f"train_{key}"] for h in history], color=SERIES[0], lw=2, marker="o", ms=4, label="train")
        ax.plot(ep, [h[f"val_{key}"] for h in history], color=SERIES[1], lw=2, marker="o", ms=4, label="validation")
        ax.axvline(best_epoch, color=INK_2, ls=":", lw=1)
        _style(ax, name, "epoch", ylabel)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"{title}  (dotted line: selected epoch {best_epoch})", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_grid(results: list[dict[str, Any]], best: str, out: Path) -> None:
    res = sorted(results, key=lambda r: r["validation"]["rpe_trans_cm"])
    # 1. Ranking by the selection metric.
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(res) + 1.4), facecolor=SURFACE)
    y = np.arange(len(res))[::-1]
    vals = [r["validation"]["rpe_trans_cm"] for r in res]
    ax.barh(y, vals, color=[ARCH_COLOR[r["config"]["architecture"]] for r in res], height=0.7)
    for yi, r, v in zip(y, res, vals):
        tag = "  ◀ best" if r["name"] == best else ""
        ax.text(v, yi, f" {v:.2f} cm · acc {r['validation']['accuracy_pct']:.1f}%{tag}", va="center", fontsize=8, color=INK, fontweight="bold" if tag else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels([f"h{r['config']['hidden']}  lr {r['config']['learning_rate']:g}" for r in res], fontsize=8, color=INK_2)
    ax.set_xlim(0, max(vals) * 1.45)
    _style(ax, "Validation RPE over a 6-interval window (lower is better) — selection metric", "translation error [cm]")
    handles = [plt.Rectangle((0, 0), 1, 1, color=ARCH_COLOR[a]) for a in ARCHITECTURES]
    ax.legend(handles, [ARCH_LABEL[a] for a in ARCHITECTURES], fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "grid_ranking_validation.png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)

    # 2. Heat maps hidden x learning rate per architecture (one metric, one hue).
    hs, lrs = GRID["hidden"], GRID["learning_rate"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), facecolor=SURFACE)
    lookup = {(r["config"]["architecture"], r["config"]["hidden"], r["config"]["learning_rate"]): r for r in results}
    vmin, vmax = min(vals), max(vals)
    for ax, arch in zip(axes, ARCHITECTURES):
        grid = np.full((len(lrs), len(hs)), np.nan)
        for i, lr in enumerate(lrs):
            for j, h in enumerate(hs):
                if (arch, h, lr) in lookup:
                    grid[i, j] = lookup[(arch, h, lr)]["validation"]["rpe_trans_cm"]
        im = ax.imshow(grid, cmap="Blues_r", vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(len(lrs)):
            for j in range(len(hs)):
                if np.isfinite(grid[i, j]):
                    name = lookup[(arch, hs[j], lrs[i])]["name"]
                    ax.text(j, i, f"{grid[i, j]:.2f}" + ("\nbest" if name == best else ""), ha="center", va="center", fontsize=9,
                            color="white" if grid[i, j] < (vmin + vmax) / 2 else INK, fontweight="bold" if name == best else "normal")
        ax.set_xticks(range(len(hs)))
        ax.set_xticklabels([f"hidden {h}" for h in hs], fontsize=8, color=INK_2)
        ax.set_yticks(range(len(lrs)))
        ax.set_yticklabels([f"lr {lr:g}" for lr in lrs], fontsize=8, color=INK_2)
        ax.set_title(ARCH_LABEL[arch], fontsize=10, color=INK, loc="left")
    fig.colorbar(im, ax=axes, shrink=0.85, label="validation RPE [cm]")
    fig.suptitle("Grid search: validation RPE per architecture (darker = better)", fontsize=12, color=INK, x=0.01, ha="left")
    fig.savefig(out / "grid_heatmaps_validation.png", dpi=100, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    # 3. Validation accuracy per epoch, small multiples per architecture.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), facecolor=SURFACE, sharey=True)
    for ax, arch in zip(axes, ARCHITECTURES):
        for r in results:
            if r["config"]["architecture"] != arch:
                continue
            h = r["history"]
            style = {64: ":", 96: "--", 128: "-"}[r["config"]["hidden"]]
            alpha = 1.0 if r["config"]["learning_rate"] == 1e-3 else 0.55
            ax.plot([e["epoch"] for e in h], [e["val_accuracy_pct"] for e in h], ls=style, color=ARCH_COLOR[arch], alpha=alpha, lw=2.2 if r["name"] == best else 1.4,
                    label=f"h{r['config']['hidden']} lr{r['config']['learning_rate']:g}" + (" (best)" if r["name"] == best else ""))
        _style(ax, ARCH_LABEL[arch], "epoch", "validation accuracy [%]")
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Validation accuracy (interval error < 2 cm and < 0.5°) during training", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "grid_accuracy_curves.png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_final(result: dict[str, Any], evals: dict[str, dict[str, Any]], out: Path) -> None:
    test = evals["test"]
    # Trajectories of six test runs spread over tracks and sensors.
    runs = sorted(test["runs"], key=lambda r: (r["track"], r["sensor"], r["motion"]))
    picks = []
    seen = set()
    for r in runs:
        key = (r["track"], r["sensor"])
        if key not in seen and (r["track"] == "test_unseen" or len(picks) < 3):
            picks.append(r)
            seen.add(key)
        if len(picks) == 6:
            break
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), facecolor=SURFACE)
    for ax, r in zip(axes.flat, picks):
        ax.plot(r["gt"][:, 0], r["gt"][:, 1], color=INK, lw=2.2, label="ground truth")
        ax.plot(r["est"][:, 0], r["est"][:, 1], color=SERIES[1], lw=1.6, label="network dead reckoning")
        ax.scatter([0], [0], color=SERIES[2], s=40, zorder=3, label="start")
        ax.set_aspect("equal")
        _style(ax, f"{r['track']} · {r['motion']} · {r['sensor']}\n{r['distance_m']:.0f} m, drift {r['drift_pct']:.2f} %, ATE {r['ate_rmse_m']:.2f} m", "x [m]", "y [m]")
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"Test runs integrated from the first scan only (best model: {result['name']})", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "test_trajectories.png", dpi=95, facecolor=fig.get_facecolor())
    plt.close(fig)

    # Breakdown of the test split.
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), facecolor=SURFACE)
    for ax, key in zip(axes, ("track", "sensor", "motion")):
        groups = sorted({r[key] for r in test["runs"]})
        vals = [np.median([r["drift_pct"] for r in test["runs"] if r[key] == g]) for g in groups]
        bars = ax.bar(range(len(groups)), vals, color=SERIES[0], width=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=25, ha="right", fontsize=8, color=INK_2)
        _style(ax, f"Test drift by {key}", "", "median drift [% of distance]")
    fig.tight_layout()
    fig.savefig(out / "test_breakdown.png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)

    # Error distributions: validation vs test, with the gyro reference for heading.
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), facecolor=SURFACE)
    for split, color in (("validation", SERIES[0]), ("test", SERIES[1])):
        pi = evals[split]["per_interval"]
        axes[0].hist(pi["trans_m"] * 100, bins=np.linspace(0, 8, 81), histtype="step", lw=2, color=color, density=True, label=split)
        axes[1].hist(np.degrees(pi["yaw_rad"]), bins=np.linspace(0, 1.5, 76), histtype="step", lw=2, color=color, density=True, label=split)
    axes[0].axvline(ACC_STRICT[0] * 100, color=INK_2, ls=":")
    axes[1].axvline(math.degrees(ACC_STRICT[1]), color=INK_2, ls=":")
    _style(axes[0], "Translation error per scan interval (dotted: accuracy threshold)", "error [cm]", "density")
    _style(axes[1], "Heading error per scan interval", "error [deg]", "density")
    for ax in axes:
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "error_distributions.png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


@torch.no_grad()
def reference_baselines(samplers: dict[str, WindowSampler]) -> dict[str, Any]:
    """Trivial predictors on the same intervals, to put the metrics in context:
    zero motion, the mean training increment, and the mean training increment
    with the raw gyro integral as heading."""
    mean_delta = samplers["train"].target.mean(dim=0)
    out: dict[str, Any] = {}
    for split in ("validation", "test"):
        sp = samplers[split]
        starts = sp.starts(sp.steps)
        idx = (starts[:, None] + torch.arange(sp.steps, device=sp.device)).flatten()
        target = sp.target[idx]
        preds = {
            "zero_motion": torch.zeros_like(target),
            "mean_train_increment": mean_delta.expand_as(target),
            "mean_increment_plus_gyro_yaw": torch.cat((mean_delta[:2].expand(len(target), 2), sp.gyro[idx, None]), dim=1),
        }
        out[split] = {}
        for name, p in preds.items():
            t_err, y_err = step_errors(p, target)
            out[split][name] = {
                "trans_mae_cm": 100 * float(t_err.mean()),
                "yaw_mae_deg": math.degrees(float(y_err.mean())),
                "accuracy_pct": 100.0 * float(((t_err < ACC_STRICT[0]) & (y_err < ACC_STRICT[1])).float().mean()),
            }
    return out


def plot_references(summary: dict[str, Any], path: Path) -> None:
    """Best model vs reference predictors on the same intervals, per split."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), facecolor=SURFACE)
    for col, split in enumerate(("validation", "test")):
        base = summary["reference_baselines"][split]
        rows = [("best network", summary[split]["trans_mae_cm"], SERIES[0])]
        if "icp_reference" in summary:
            rows.append(("ICP, ideal heading", summary["icp_reference"][split]["icp_trans_mae_cm"], SERIES[1]))
        rows += [("mean increment", base["mean_train_increment"]["trans_mae_cm"], "#9a9890"), ("zero motion", base["zero_motion"]["trans_mae_cm"], "#9a9890")]
        yaw_rows = [("best network", summary[split]["yaw_mae_deg"], SERIES[0]), ("raw gyro integral", summary[split]["gyro_yaw_mae_deg"], SERIES[2]),
                    ("zero rotation", base["zero_motion"]["yaw_mae_deg"], "#9a9890")]
        for row, (data, unit, title) in enumerate(((rows, "cm", "translation"), (yaw_rows, "deg", "heading"))):
            ax = axes[row, col]
            y = np.arange(len(data))[::-1]
            ax.barh(y, [d[1] for d in data], color=[d[2] for d in data], height=0.6)
            for yi, d in zip(y, data):
                ax.text(d[1], yi, f" {d[1]:.3f} {unit}" if unit == "deg" else f" {d[1]:.2f} {unit}", va="center", fontsize=9, color=INK)
            ax.set_yticks(y)
            ax.set_yticklabels([d[0] for d in data], fontsize=9, color=INK_2)
            ax.set_xlim(0, max(d[1] for d in data) * 1.3)
            _style(ax, f"{split}: {title} error per scan interval (MAE, lower is better)", f"MAE [{unit}]")
    fig.tight_layout()
    fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------- CLI
def load_samplers(db: Path, device: torch.device, steps: int, context: int, splits=("train", "validation", "test")) -> dict[str, WindowSampler]:
    cache = ROOT / "outputs" / "cache_sqlite"
    return {s: WindowSampler(load_or_build(db, cache, s), device, steps, context) for s in splits}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("train", "report", "final"))
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--epochs", type=int, default=8, help="epochs per grid configuration (final: --final-epochs)")
    ap.add_argument("--final-epochs", type=int, default=24)
    ap.add_argument("--shard", default="0/1", help="i/n: train every n-th configuration starting at i (parallel processes)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(3)
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / f"log_{args.stage}_{args.shard.replace('/', 'of')}.txt"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    if args.stage == "train":
        i, n = (int(v) for v in args.shard.split("/"))
        configs = grid_configs(args.epochs)[i::n]
        samplers = load_samplers(args.db, device, 6, 5, ("train", "validation"))
        for cfg in configs:
            if (args.out / "configs" / cfg.name / "result.json").exists():
                log(f"[skip] {cfg.name} already trained")
                continue
            train_config(cfg, samplers, args.out / "configs" / cfg.name, log)
        return

    results = [json.loads(p.read_text()) for p in sorted((args.out / "configs").glob("*/result.json"))]
    if not results:
        raise SystemExit("no trained configurations")
    best = min(results, key=lambda r: r["validation"]["rpe_trans_cm"])
    if args.stage == "report":
        plot_grid(results, best["name"], args.out)
        table = sorted(({"name": r["name"], **r["config"], "parameters": r["parameters"], "best_epoch": r["best_epoch"], **r["validation"]} for r in results), key=lambda r: r["rpe_trans_cm"])
        (args.out / "grid_results.json").write_text(json.dumps({"selection_metric": "validation rpe_trans_cm", "best": best["name"], "grid": GRID, "results": table}, indent=1), encoding="utf-8")
        keys = ["name", "architecture", "hidden", "learning_rate", "parameters", "best_epoch", "rpe_trans_cm", "rpe_yaw_deg", "accuracy_pct", "accuracy_loose_pct", "trans_mae_cm", "yaw_mae_deg", "drift_pct_median", "ate_rmse_m_median", "gyro_yaw_mae_deg"]
        lines = [",".join(keys)] + [",".join(str(row.get(k, "")) for k in keys) for row in table]
        (args.out / "grid_results.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        log(f"[report] best on validation: {best['name']} (RPE {best['validation']['rpe_trans_cm']:.2f} cm, accuracy {best['validation']['accuracy_pct']:.1f} %)")
        return

    # final: retrain the best configuration longer, evaluate validation and test once.
    cfg = replace(RunConfig(**best["config"]), epochs=args.final_epochs)
    samplers = load_samplers(args.db, device, cfg.steps, cfg.context)
    final_dir = args.out / "best_final"
    if (final_dir / "result.json").exists():
        (final_dir / "result.json").unlink()
    result = train_config(cfg, {"train": samplers["train"], "validation": samplers["validation"]}, final_dir, log)
    model = PoseNet(cfg.architecture, cfg.hidden).to(device)
    model.load_state_dict(torch.load(final_dir / "best_model.pt", map_location=device, weights_only=False)["model"])
    evals = {s: evaluate(model, samplers[s], with_runs=True) for s in ("validation", "test")}
    plot_final(result, evals, final_dir)
    baselines = reference_baselines(samplers)
    summary = {
        "best_configuration": result["name"], "config": result["config"], "parameters": result["parameters"],
        "selected_epoch": result["best_epoch"], "train_seconds": result["train_seconds"],
        "data": {s: split_statistics(load_or_build(args.db, ROOT / "outputs" / "cache_sqlite", s)) for s in ("train", "validation", "test")},
    }
    for s, ev in evals.items():
        summary[s] = {k: v for k, v in ev.items() if k not in ("runs", "per_interval")}
        by = {}
        for key in ("track", "sensor", "motion"):
            by[key] = {g: {"drift_pct_median": float(np.median([r["drift_pct"] for r in ev["runs"] if r[key] == g])),
                           "ate_rmse_m_median": float(np.median([r["ate_rmse_m"] for r in ev["runs"] if r[key] == g])),
                           "runs": sum(1 for r in ev["runs"] if r[key] == g)} for g in sorted({r[key] for r in ev["runs"]})}
        summary[s]["breakdown"] = by
    summary["reference_baselines"] = baselines
    icp_path = args.out / "icp_reference.json"
    if icp_path.exists():
        summary["icp_reference"] = json.loads(icp_path.read_text(encoding="utf-8"))
    plot_references(summary, final_dir / "comparison_with_references.png")
    (final_dir / "final_metrics.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    log(f"[final] {result['name']}: validation acc {evals['validation']['accuracy_pct']:.1f}% RPE {evals['validation']['rpe_trans_cm']:.2f} cm | "
        f"test acc {evals['test']['accuracy_pct']:.1f}% RPE {evals['test']['rpe_trans_cm']:.2f} cm drift {evals['test']['drift_pct_median']:.2f}%")


if __name__ == "__main__":
    main()
