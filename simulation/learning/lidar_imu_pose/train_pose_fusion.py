#!/usr/bin/env python3
"""Train a LiDAR/IMU incremental-pose network.

By default, targets come from sensor-only LiDAR/IMU odometry. With
``--supervised-ground-truth``, exact simulator poses provide relative-motion
targets for an explicit supervised experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class Config:
    beams: int = 360
    context_scans: int = 5
    sequence_steps: int = 6
    batch_size: int = 24
    epochs: int = 30
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-5
    hidden_size: int = 96
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    seed: int = 23
    num_workers: int = 0
    icp_stride: int = 4
    icp_iterations: int = 7
    icp_max_correspondence_m: float = 0.30
    nll_weight: float = 1.0
    imu_yaw_weight: float = 10.0
    motion_smooth_weight: float = 0.05
    bias_smooth_weight: float = 0.04
    bias_prior_weight: float = 0.005
    velocity_weight: float = 0.20
    composition_weight: float = 0.10


def stamp(row: dict[str, str]) -> float:
    return int(row["stamp_sec"]) + int(row["stamp_nanosec"]) * 1.0e-9


def wrap_angle(value: np.ndarray | float) -> np.ndarray | float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def read_cutoff(run_dir: Path) -> float:
    manifest = run_dir / "CAPTURE_MANIFEST.json"
    if not manifest.exists():
        return math.inf
    return float(json.loads(manifest.read_text(encoding="utf-8"))["lap_finish_sim_time_s"])


def load_scans(measurements: Path, cutoff_s: float, beams: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with (measurements / "scan_index.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if stamp(row) <= cutoff_s]
    if len(rows) < 10:
        raise RuntimeError("not enough LiDAR scans in the selected run")
    scan_ids = np.asarray([int(row["scan_index"]) for row in rows], dtype=np.int64)
    times = np.asarray([stamp(row) for row in rows], dtype=np.float64)
    range_max = np.asarray([float(row["range_max_m"]) for row in rows], dtype=np.float32)
    id_to_row = {int(scan_id): idx for idx, scan_id in enumerate(scan_ids)}
    ranges = np.full((len(rows), beams), np.nan, dtype=np.float32)
    with (measurements / "lidar_points.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            output_idx = id_to_row.get(int(row["scan_index"]))
            if output_idx is None:
                continue
            beam_idx = int(row["beam_index"])
            if 0 <= beam_idx < beams:
                ranges[output_idx, beam_idx] = float(row["range_m"])
    # Channel 0 is normalized range; channel 1 explicitly marks real returns.
    valid = np.isfinite(ranges)
    normalized = np.where(valid, ranges / range_max[:, None], 1.0).astype(np.float32)
    lidar = np.stack((normalized, valid.astype(np.float32)), axis=1)
    return times, ranges, lidar


def load_imu(measurements: Path, cutoff_s: float) -> tuple[np.ndarray, np.ndarray]:
    fields = ("ax_mps2", "ay_mps2", "az_mps2", "gx_rps", "gy_rps", "gz_rps")
    times: list[float] = []
    values: list[list[float]] = []
    with (measurements / "imu_raw.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            t = stamp(row)
            if t > cutoff_s:
                break
            times.append(t)
            values.append([float(row[name]) for name in fields])
    imu = np.asarray(values, dtype=np.float32)
    # Stable, physical normalization. Remove nominal gravity before scaling.
    imu[:, 2] -= 9.80665
    imu[:, :3] /= 9.80665
    return np.asarray(times, dtype=np.float64), imu


def window_imu_at_lidar_intervals(
    imu_times: np.ndarray,
    imu: np.ndarray,
    scan_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep the real IMU samples in each LiDAR interval, padding only for batching."""
    bounds: list[tuple[int, int]] = [(0, 0)]
    counts = np.ones(len(scan_times), dtype=np.int64)
    for idx in range(1, len(scan_times)):
        lo = int(np.searchsorted(imu_times, scan_times[idx - 1], side="right"))
        hi = int(np.searchsorted(imu_times, scan_times[idx], side="right"))
        if hi <= lo:
            nearest = int(np.clip(np.searchsorted(imu_times, scan_times[idx]), 0, len(imu_times) - 1))
            lo, hi = nearest, nearest + 1
        bounds.append((lo, hi))
        counts[idx] = hi - lo
    counts[0] = counts[1]
    bounds[0] = bounds[1]
    max_samples = int(counts.max())
    result = np.zeros((len(scan_times), max_samples, 6), dtype=np.float32)
    dt = np.zeros(len(scan_times), dtype=np.float32)
    integrated_gyro_z = np.zeros(len(scan_times), dtype=np.float32)
    for idx in range(1, len(scan_times)):
        t0, t1 = float(scan_times[idx - 1]), float(scan_times[idx])
        dt[idx] = max(t1 - t0, 1.0e-4)
        lo, hi = bounds[idx]
        result[idx, : hi - lo] = imu[lo:hi]
        # Integrate at exact interval boundaries without altering the network input.
        inner_times = imu_times[lo:hi]
        integration_times = np.concatenate(([t0], inner_times[(inner_times > t0) & (inner_times < t1)], [t1]))
        integration_values = np.interp(integration_times, imu_times, imu[:, 5])
        integrated_gyro_z[idx] = float(np.trapezoid(integration_values, integration_times))
    result[0] = result[1]
    dt[0] = dt[1]
    integrated_gyro_z[0] = integrated_gyro_z[1]
    return result, counts, dt, integrated_gyro_z


def ranges_to_points(ranges: Tensor, stride: int) -> Tensor:
    indices = torch.arange(0, ranges.numel(), stride, device=ranges.device)
    selected = ranges[indices]
    keep = torch.isfinite(selected)
    angles = torch.linspace(-math.pi, math.pi, ranges.numel(), device=ranges.device, dtype=ranges.dtype)[indices][keep]
    selected = selected[keep]
    return torch.stack((selected * torch.cos(angles), selected * torch.sin(angles)), dim=1)


def rigid_fit(source: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vh = torch.linalg.svd(covariance)
    rotation = vh.mT @ u.mT
    if torch.linalg.det(rotation) < 0.0:
        vh = vh.clone()
        vh[-1] *= -1.0
        rotation = vh.mT @ u.mT
    translation = target_center - rotation @ source_center
    return rotation, translation


def lidar_icp_delta(
    previous_ranges: np.ndarray,
    current_ranges: np.ndarray,
    initial_yaw_rad: float,
    cfg: Config,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Map current-scan points into the previous frame (vehicle motion).

    The initial rotation comes only from the raw gyro integral. This prevents
    nearest-neighbour ICP from collapsing to zero yaw in symmetric corridors.
    """
    target = ranges_to_points(torch.as_tensor(previous_ranges, device=device), cfg.icp_stride)
    source = ranges_to_points(torch.as_tensor(current_ranges, device=device), cfg.icp_stride)
    if len(source) < 20 or len(target) < 20:
        return np.zeros(3, dtype=np.float32), 0.05
    initial_yaw = source.new_tensor(initial_yaw_rad)
    c, s = torch.cos(initial_yaw), torch.sin(initial_yaw)
    rotation_total = torch.stack((torch.stack((c, -s)), torch.stack((s, c))))
    translation_total = torch.zeros(2, device=device, dtype=source.dtype)
    transformed = (rotation_total @ source.mT).mT
    last_rmse = cfg.icp_max_correspondence_m
    inlier_ratio = 0.0
    for _ in range(cfg.icp_iterations):
        # At most 90x90 for the default stride: no SciPy dependency required.
        distances = torch.cdist(transformed, target)
        distances, nearest = distances.min(dim=1)
        keep = distances < cfg.icp_max_correspondence_m
        if int(keep.sum()) < 12:
            break
        r_step, t_step = rigid_fit(transformed[keep], target[nearest[keep]])
        transformed = (r_step @ transformed.mT).mT + t_step
        rotation_total = r_step @ rotation_total
        translation_total = r_step @ translation_total + t_step
        last_rmse = float(torch.sqrt(torch.mean(distances[keep].square())).item())
        inlier_ratio = float(keep.float().mean().item())
    # Rotation is directly observable from the gyro over this short interval.
    # Nearest-neighbour ICP is allowed to refine translation, but its yaw is
    # intentionally not trusted in the long, nearly symmetric corridors.
    yaw = float(initial_yaw_rad)
    quality = float(np.clip(inlier_ratio * math.exp(-last_rmse / 0.12), 0.05, 1.0))
    motion = torch.cat((translation_total, translation_total.new_tensor([yaw]))).cpu().numpy()
    return motion.astype(np.float32), quality


def build_icp_targets(
    ranges: np.ndarray,
    gyro_integral: np.ndarray,
    cfg: Config,
    cache_path: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    fingerprint = hashlib.sha256(
        f"gyro_yaw_v3:{ranges.shape}:{cfg.icp_stride}:{cfg.icp_iterations}:{cfg.icp_max_correspondence_m}".encode()
    ).hexdigest()[:16]
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if payload["fingerprint"] == fingerprint:
            return payload["delta"].numpy(), payload["quality"].numpy()
    delta = np.zeros((len(ranges), 3), dtype=np.float32)
    quality = np.full(len(ranges), 0.05, dtype=np.float32)
    with torch.no_grad():
        for idx in range(1, len(ranges)):
            delta[idx], quality[idx] = lidar_icp_delta(
                ranges[idx - 1], ranges[idx], float(gyro_integral[idx]), cfg, device
            )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"fingerprint": fingerprint, "delta": torch.from_numpy(delta), "quality": torch.from_numpy(quality)}, cache_path)
    return delta, quality


class SensorWindowDataset(Dataset):
    """Sensor windows plus precomputed relative-motion supervision."""

    def __init__(
        self,
        lidar: np.ndarray,
        imu_windows: np.ndarray,
        imu_lengths: np.ndarray,
        dt: np.ndarray,
        gyro_integral: np.ndarray,
        icp_delta: np.ndarray,
        icp_quality: np.ndarray,
        starts: Iterable[int],
        sequence_steps: int,
        context_scans: int,
    ) -> None:
        self.lidar = lidar
        self.imu = imu_windows
        self.imu_lengths = imu_lengths
        self.dt = dt
        self.gyro_integral = gyro_integral
        self.icp_delta = icp_delta
        self.icp_quality = icp_quality
        self.starts = np.asarray(list(starts), dtype=np.int64)
        self.steps = sequence_steps
        self.context_scans = context_scans

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        start = int(self.starts[item])
        targets = np.arange(start, start + self.steps)
        scan_context = np.stack(
            [self.lidar[t - self.context_scans + 1 : t + 1] for t in targets]
        )
        return {
            "lidar": torch.from_numpy(scan_context),
            "imu": torch.from_numpy(self.imu[targets]),
            "imu_lengths": torch.from_numpy(self.imu_lengths[targets]),
            "dt": torch.from_numpy(self.dt[targets]),
            "gyro_integral": torch.from_numpy(self.gyro_integral[targets]),
            "icp_delta": torch.from_numpy(self.icp_delta[targets]),
            "icp_quality": torch.from_numpy(self.icp_quality[targets]),
            "target_indices": torch.from_numpy(targets),
        }


class LidarDualEncoder(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 32, 9, stride=2, padding=4), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv1d(32, 64, 7, stride=2, padding=3), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv1d(64, 96, 5, stride=2, padding=2), nn.GroupNorm(8, 96), nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.project = nn.Linear(96, hidden)
        self.pair_gru = nn.GRU(hidden, hidden, batch_first=True)
        self.context_gru = nn.GRU(hidden, hidden, batch_first=True)

    def forward(self, scans: Tensor) -> tuple[Tensor, Tensor]:
        # [B,T,C,2,beams]. Both branches share the spatial CNN, while their
        # GRUs specialize in pairwise motion and longer causal context.
        b, t, context, channels, beams = scans.shape
        encoded = self.cnn(scans.reshape(b * t * context, channels, beams)).squeeze(-1)
        encoded = self.project(encoded).reshape(b * t, context, -1)
        _, pair_state = self.pair_gru(encoded[:, -2:])
        _, context_state = self.context_gru(encoded)
        return pair_state[-1].reshape(b, t, -1), context_state[-1].reshape(b, t, -1)


class ImuEncoder(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(6, 32, 3, padding=1), nn.SiLU(),
            nn.Conv1d(32, 48, 3, padding=1), nn.SiLU(),
        )
        self.gru = nn.GRU(48, hidden, batch_first=True)

    def forward(self, imu: Tensor, lengths: Tensor) -> Tensor:
        b, t, samples, channels = imu.shape
        features = self.temporal(imu.reshape(b * t, samples, channels).transpose(1, 2)).transpose(1, 2)
        flat_lengths = lengths.reshape(-1)
        mask = torch.arange(samples, device=imu.device).unsqueeze(0) < flat_lengths.unsqueeze(1)
        features = features * mask.unsqueeze(-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            features, flat_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, state = self.gru(packed)
        return state[-1].reshape(b, t, -1)


class LidarImuPoseNet(nn.Module):
    def __init__(self, hidden: int = 96) -> None:
        super().__init__()
        self.lidar_encoder = LidarDualEncoder(hidden)
        self.imu_encoder = ImuEncoder(hidden)
        self.fuse = nn.Sequential(nn.Linear(3 * hidden, 2 * hidden), nn.SiLU(), nn.Dropout(0.10), nn.Linear(2 * hidden, hidden))
        self.fusion_gru = nn.GRU(hidden, hidden, num_layers=2, dropout=0.10, batch_first=True)
        self.pair_motion_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3))
        self.context_motion_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3))
        self.context_gate_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 3))
        self.logvar_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 3))
        self.bias_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 6))
        self.velocity_init_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 2))
        self.velocity_residual_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 2))
        self.register_buffer("motion_scale", torch.tensor([0.025, 0.025, 0.05]))
        self.register_buffer("bias_scale", torch.tensor([0.20, 0.20, 0.20, 0.02, 0.02, 0.02]))

    def forward(self, lidar: Tensor, imu: Tensor, imu_lengths: Tensor, dt: Tensor) -> dict[str, Tensor]:
        pair_features, context_features = self.lidar_encoder(lidar)
        imu_features = self.imu_encoder(imu, imu_lengths)
        fused = self.fuse(torch.cat((pair_features, context_features, imu_features), dim=-1))
        temporal, _ = self.fusion_gru(fused)
        bias = self.bias_head(temporal) * self.bias_scale
        pair_motion = torch.tanh(self.pair_motion_head(pair_features)) * self.motion_scale
        context_correction = torch.tanh(self.context_motion_head(temporal)) * self.motion_scale
        context_gate = torch.sigmoid(self.context_gate_head(temporal))
        residual_motion = pair_motion + context_gate * context_correction
        # Recurrent inertial mechanization for planar velocity and translation.
        # Learned terms only initialize/correct the physical integration.
        velocity = torch.tanh(self.velocity_init_head(temporal[:, 0])) * 0.40
        delta_velocity = torch.tanh(self.velocity_residual_head(temporal)) * 0.03
        acceleration = imu[..., :2].mean(dim=-2) * 9.80665 - bias[..., :2]
        translations: list[Tensor] = []
        velocities: list[Tensor] = []
        for step in range(temporal.shape[1]):
            step_dt = dt[:, step : step + 1]
            translations.append(
                velocity * step_dt
                + 0.5 * acceleration[:, step] * step_dt.square()
                + residual_motion[:, step, :2]
            )
            velocity = velocity + acceleration[:, step] * step_dt + delta_velocity[:, step]
            velocities.append(velocity)
        translation = torch.stack(translations, dim=1)
        # Physics-informed residual head: retain the measured short-term gyro
        # rotation and let the network correct its learned bias and residual.
        gyro_yaw = (imu[..., 5].mean(dim=-1) - bias[..., 5]) * dt
        yaw = residual_motion[..., 2:3] + gyro_yaw.unsqueeze(-1)
        delta = torch.cat((translation, yaw), dim=-1)
        return {
            "delta": delta,
            "logvar": self.logvar_head(temporal).clamp(-8.0, 3.0),
            "bias": bias,
            "velocity": torch.stack(velocities, dim=1),
            "context_gate": context_gate,
        }


def compose_planar(deltas: Tensor) -> Tensor:
    """Differentiably compose [dx,dy,dyaw] increments over the time axis."""
    x = torch.zeros_like(deltas[:, 0, 0])
    y = torch.zeros_like(x)
    yaw = torch.zeros_like(x)
    for step in range(deltas.shape[1]):
        dx, dy, dyaw = deltas[:, step].unbind(dim=-1)
        c, s = torch.cos(yaw), torch.sin(yaw)
        x = x + c * dx - s * dy
        y = y + s * dx + c * dy
        yaw = yaw + dyaw
    return torch.stack((x, y, yaw), dim=-1)


def pose_loss(output: dict[str, Tensor], batch: dict[str, Tensor], cfg: Config) -> tuple[Tensor, dict[str, float]]:
    delta, logvar, bias = output["delta"], output["logvar"], output["bias"]
    target = batch["icp_delta"]
    # Balance metres and radians, then let the covariance head explain residual scale.
    scale = delta.new_tensor([12.0, 18.0, 8.0])
    residual = (delta - target) * scale
    quality = batch["icp_quality"].unsqueeze(-1)
    nll = (quality * 0.5 * (torch.exp(-logvar) * residual.square() + logvar)).mean()

    predicted_gyro_delta = (batch["gyro_integral"] - bias[..., 5] * batch["dt"])
    imu_yaw = torch.nn.functional.smooth_l1_loss(delta[..., 2], predicted_gyro_delta, beta=0.03)
    motion_smooth = (delta[:, 1:] - delta[:, :-1]).square().mean() if delta.shape[1] > 1 else delta.new_zeros(())
    bias_smooth = (bias[:, 1:] - bias[:, :-1]).square().mean() if bias.shape[1] > 1 else bias.new_zeros(())
    bias_prior = bias.square().mean()
    target_velocity = target[..., :2] / batch["dt"].unsqueeze(-1).clamp_min(1.0e-3)
    velocity_loss = (
        quality * torch.nn.functional.smooth_l1_loss(
            output["velocity"], target_velocity, beta=0.05, reduction="none"
        )
    ).mean()
    composition_loss = torch.nn.functional.smooth_l1_loss(
        compose_planar(delta) * scale,
        compose_planar(target) * scale,
        beta=0.10,
    )
    total = (
        cfg.nll_weight * nll
        + cfg.imu_yaw_weight * imu_yaw
        + cfg.motion_smooth_weight * motion_smooth
        + cfg.bias_smooth_weight * bias_smooth
        + cfg.bias_prior_weight * bias_prior
        + cfg.velocity_weight * velocity_loss
        + cfg.composition_weight * composition_loss
    )
    metrics = {
        "loss": float(total.detach()),
        "nll": float(nll.detach()),
        "imu_yaw": float(imu_yaw.detach()),
        "velocity": float(velocity_loss.detach()),
        "composition": float(composition_loss.detach()),
        "translation_mae_m": float(torch.linalg.vector_norm(delta[..., :2] - target[..., :2], dim=-1).mean().detach()),
        "yaw_mae_deg": float(torch.rad2deg(torch.abs(delta[..., 2] - target[..., 2])).mean().detach()),
    }
    return total, metrics


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def epoch_pass(
    model: nn.Module,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    model.train(optimizer is not None)
    totals: dict[str, float] = {
        "loss": 0.0,
        "nll": 0.0,
        "imu_yaw": 0.0,
        "velocity": 0.0,
        "composition": 0.0,
        "translation_mae_m": 0.0,
        "yaw_mae_deg": 0.0,
    }
    count = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        with torch.set_grad_enabled(optimizer is not None):
            output = model(batch["lidar"], batch["imu"], batch["imu_lengths"], batch["dt"])
            loss, metrics = pose_loss(output, batch, cfg)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        batch_size = int(batch["lidar"].shape[0])
        count += batch_size
        for key in totals:
            totals[key] += metrics[key] * batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def predict_partition(model: nn.Module, dataset: SensorWindowDataset, cfg: Config, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Consecutive windows overlap. Use only the final prediction from each
    # window and retain each target index once.
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    records: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            output = model(batch["lidar"], batch["imu"], batch["imu_lengths"], batch["dt"])
            indices = batch["target_indices"][:, -1].cpu().numpy()
            delta = output["delta"][:, -1].cpu().numpy()
            variance = np.exp(output["logvar"][:, -1].cpu().numpy()) / np.square([12.0, 18.0, 8.0])
            for index, one_delta, one_variance in zip(indices, delta, variance):
                records[int(index)] = (one_delta, one_variance)
    ordered = np.asarray(sorted(records), dtype=np.int64)
    return ordered, np.stack([records[i][0] for i in ordered]), np.stack([records[i][1] for i in ordered])


def load_ground_truth(measurements: Path, query_times: np.ndarray) -> np.ndarray:
    """Interpolate exact simulator poses at the requested sensor timestamps."""
    rows = list(csv.DictReader((measurements / "ground_truth_odom.csv").open(newline="", encoding="utf-8")))
    times = np.asarray([stamp(row) for row in rows], dtype=np.float64)
    x = np.asarray([float(row["x_m"]) for row in rows])
    y = np.asarray([float(row["y_m"]) for row in rows])
    yaw = np.unwrap(np.asarray([float(row["yaw_rad"]) for row in rows]))
    return np.column_stack((np.interp(query_times, times, x), np.interp(query_times, times, y), np.interp(query_times, times, yaw)))


def ground_truth_deltas(poses: np.ndarray) -> np.ndarray:
    """Convert world poses to consecutive planar increments in the old body frame."""
    delta = np.zeros((len(poses), 3), dtype=np.float32)
    for idx in range(1, len(poses)):
        world_delta = poses[idx, :2] - poses[idx - 1, :2]
        c, s = math.cos(poses[idx - 1, 2]), math.sin(poses[idx - 1, 2])
        delta[idx, 0] = c * world_delta[0] + s * world_delta[1]
        delta[idx, 1] = -s * world_delta[0] + c * world_delta[1]
        delta[idx, 2] = float(wrap_angle(poses[idx, 2] - poses[idx - 1, 2]))
    delta[0] = delta[1]
    return delta


def integrate_deltas(initial_pose: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    poses = np.empty((len(deltas), 3), dtype=np.float64)
    pose = initial_pose.astype(np.float64).copy()
    for idx, (dx, dy, dyaw) in enumerate(deltas):
        c, s = math.cos(pose[2]), math.sin(pose[2])
        pose[0] += c * dx - s * dy
        pose[1] += s * dx + c * dy
        pose[2] = float(wrap_angle(pose[2] + dyaw))
        poses[idx] = pose
    return poses


def validation_report(
    model: nn.Module,
    dataset: SensorWindowDataset,
    scan_times: np.ndarray,
    measurements: Path,
    cfg: Config,
    device: torch.device,
    output_dir: Path,
    supervised: bool,
) -> dict[str, float]:
    indices, deltas, variances = predict_partition(model, dataset, cfg, device)
    times = scan_times[indices]
    truth = load_ground_truth(measurements, times)
    # Each predicted increment maps pose[index - 1] to pose[index]. Starting
    # at truth[0] would shift the complete prediction by one LiDAR interval.
    initial_truth = load_ground_truth(measurements, scan_times[[indices[0] - 1]])[0]
    prediction = integrate_deltas(initial_truth, deltas)
    position_error = np.linalg.norm(prediction[:, :2] - truth[:, :2], axis=1)
    yaw_error = np.abs(wrap_angle(prediction[:, 2] - truth[:, 2]))
    cumulative_sigma = np.sqrt(np.maximum(np.cumsum(variances[:, :2].sum(axis=1)), 1.0e-12))
    metrics = {
        "validation_samples": int(len(indices)),
        "position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
        "position_mae_m": float(np.mean(position_error)),
        "position_final_m": float(position_error[-1]),
        "yaw_rmse_deg": float(np.degrees(np.sqrt(np.mean(yaw_error**2)))),
        "yaw_mae_deg": float(np.degrees(np.mean(yaw_error))),
    }

    elapsed = times - times[0]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].plot(truth[:, 0], truth[:, 1], color="black", linewidth=2.2, label="Pose real (partición validación)")
    axes[0].plot(prediction[:, 0], prediction[:, 1], color="#e67e22", linewidth=1.8, label="CNN+GRU LiDAR–IMU")
    axes[0].scatter(truth[0, 0], truth[0, 1], color="green", s=45, zorder=3, label="Inicio")
    axes[0].set_title("Trayectoria de validación")
    axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("y [m]"); axes[0].axis("equal"); axes[0].grid(alpha=0.25); axes[0].legend()
    axes[1].plot(elapsed, position_error, color="#c0392b", label="Error de posición")
    axes[1].plot(elapsed, cumulative_sigma, color="#2980b9", linestyle="--", label="σ acumulada predicha")
    axes[1].set_title(f"RMSE posición: {metrics['position_rmse_m']:.3f} m")
    axes[1].set_xlabel("tiempo [s]"); axes[1].set_ylabel("error [m]"); axes[1].grid(alpha=0.25); axes[1].legend()
    axes[2].plot(elapsed, np.degrees(truth[:, 2] - truth[0, 2]), color="black", linewidth=2.0, label="Yaw real")
    axes[2].plot(elapsed, np.degrees(np.unwrap(prediction[:, 2]) - prediction[0, 2]), color="#8e44ad", label="Yaw estimado")
    axes[2].set_title(f"RMSE yaw: {metrics['yaw_rmse_deg']:.2f}°")
    axes[2].set_xlabel("tiempo [s]"); axes[2].set_ylabel("cambio de yaw [°]"); axes[2].grid(alpha=0.25); axes[2].legend()
    policy = "entrenamiento supervisado con pose real" if supervised else "pose real usada sólo para evaluar"
    fig.suptitle(f"Validación LiDAR–IMU: {policy}", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "validation_pose.png", dpi=170)
    plt.close(fig)
    return metrics


def plot_training_curves(history: list[dict[str, float]], output_dir: Path) -> None:
    """Plot optimization and regression-quality curves for train/validation."""
    epochs = [row["epoch"] for row in history]
    panels = (
        ("loss", "Loss total", "loss"),
        ("nll", "NLL de pose", "NLL"),
        ("translation_mae_m", "Error por incremento", "MAE traslación [m]"),
        ("yaw_mae_deg", "Error angular por incremento", "MAE yaw [°]"),
        ("velocity", "Pérdida de velocidad", "Smooth L1"),
        ("composition", "Pérdida de composición", "Smooth L1"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for axis, (key, title, ylabel) in zip(axes.flat, panels):
        axis.plot(epochs, [row[f"train_{key}"] for row in history], label="Entrenamiento", color="#2471a3")
        axis.plot(epochs, [row[f"val_{key}"] for row in history], label="Validación", color="#e67e22")
        best_epoch = int(np.argmin([row["val_loss"] for row in history]))
        axis.axvline(epochs[best_epoch], color="gray", linestyle=":", alpha=0.7)
        axis.set_title(title)
        axis.set_xlabel("época")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Curvas de aprendizaje LiDAR–IMU", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=170)
    plt.close(fig)


def split_starts(scan_count: int, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    first_target = cfg.context_scans - 1
    train_end = int(scan_count * cfg.train_fraction)
    val_end = int(scan_count * (cfg.train_fraction + cfg.val_fraction))

    def starts(lo: int, hi: int) -> np.ndarray:
        first = max(lo, first_target)
        last_exclusive = hi - cfg.sequence_steps + 1
        return np.arange(first, max(first, last_exclusive), dtype=np.int64)

    train = starts(first_target, train_end)
    val = starts(train_end, val_end)
    test = starts(val_end, scan_count)
    boundaries = {"scan_count": scan_count, "train_end": train_end, "validation_end": val_end}
    return train, val, test, boundaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="fusion_research run directory")
    parser.add_argument("--output-dir", type=Path, default=Path("learning/outputs/lidar_imu_pose"))
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--hidden-size", type=int, default=Config.hidden_size)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)
    parser.add_argument(
        "--supervised-ground-truth",
        action="store_true",
        help="train relative-pose targets from exact simulator ground truth instead of ICP",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(epochs=args.epochs, batch_size=args.batch_size, hidden_size=args.hidden_size, num_workers=args.num_workers)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    run_dir = args.run_dir.expanduser().resolve()
    measurements = run_dir / "measurements_noisy"
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cutoff = read_cutoff(run_dir)
    scan_times, raw_ranges, lidar = load_scans(measurements, cutoff, cfg.beams)
    imu_times, imu = load_imu(measurements, cutoff)
    imu_windows, imu_lengths, dt, gyro_integral = window_imu_at_lidar_intervals(
        imu_times, imu, scan_times
    )
    if args.supervised_ground_truth:
        exact_poses = load_ground_truth(measurements, scan_times)
        motion_targets = ground_truth_deltas(exact_poses)
        target_quality = np.ones(len(scan_times), dtype=np.float32)
        supervision = "simulator_ground_truth"
    else:
        motion_targets, target_quality = build_icp_targets(
            raw_ranges, gyro_integral, cfg, output_dir / "cache" / "icp_targets.pt", device
        )
        supervision = "sensor_only_icp_gyro"
    train_starts, val_starts, test_starts, boundaries = split_starts(len(scan_times), cfg)

    common = (lidar, imu_windows, imu_lengths, dt, gyro_integral, motion_targets, target_quality)
    train_set = SensorWindowDataset(*common, train_starts, cfg.sequence_steps, cfg.context_scans)
    val_set = SensorWindowDataset(*common, val_starts, cfg.sequence_steps, cfg.context_scans)
    test_set = SensorWindowDataset(*common, test_starts, cfg.sequence_steps, cfg.context_scans)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, generator=generator, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = LidarImuPoseNet(cfg.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1))
    history: list[dict[str, float]] = []
    best_val = math.inf
    started = time.time()
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = epoch_pass(model, train_loader, cfg, device, optimizer)
        val_metrics = epoch_pass(model, val_loader, cfg, device, None)
        scheduler.step()
        record = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)
        print(f"epoch {epoch:03d}/{cfg.epochs}: train={train_metrics['loss']:.5f} val={val_metrics['loss']:.5f}", flush=True)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {"model": model.state_dict(), "config": asdict(cfg), "epoch": epoch, "val_loss": best_val, "supervision": supervision},
                output_dir / "best_model.pt",
            )

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    test_metrics = epoch_pass(model, test_loader, cfg, device, None)
    validation_metrics = validation_report(
        model, val_set, scan_times, measurements, cfg, device, output_dir, args.supervised_ground_truth
    )
    plot_training_curves(history, output_dir)
    report = {
        "architecture": "shared LiDAR CNN1D with pair-GRU (2 scans) and context-GRU (5 scans); gated feature fusion; IMU temporal CNN+GRU; fusion GRU; recurrent velocity mechanization; motion/covariance/bias heads",
        "supervision": supervision,
        "ground_truth_policy": (
            "used to build relative-pose targets for train, validation, and test"
            if args.supervised_ground_truth
            else "not used by training/loss/checkpoint selection; used only by final validation report"
        ),
        "config": asdict(cfg),
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "sampling": {
            "lidar_hz_median": float(1.0 / np.median(np.diff(scan_times))),
            "imu_hz_median": float(1.0 / np.median(np.diff(imu_times))),
            "imu_samples_per_lidar_interval": {
                "min": int(imu_lengths[1:].min()),
                "median": float(np.median(imu_lengths[1:])),
                "max": int(imu_lengths[1:].max()),
            },
            "lidar_input_without_batch": [cfg.sequence_steps, cfg.context_scans, 2, cfg.beams],
            "imu_input_without_batch": [cfg.sequence_steps, int(imu_lengths.max()), 6],
            "imu_lengths_input_without_batch": [cfg.sequence_steps],
            "padding_policy": "zero padded to interval maximum; ignored by mask and packed GRU",
        },
        "split_boundaries": boundaries,
        "window_counts": {"train": len(train_set), "validation": len(val_set), "test": len(test_set)},
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_loss": float(checkpoint["val_loss"]),
        "test": test_metrics,
        "validation_ground_truth": validation_metrics,
        "training_seconds": time.time() - started,
        "outputs": {
            "checkpoint": "best_model.pt",
            "validation_plot": "validation_pose.png",
            "training_curves": "training_curves.png",
        },
    }
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
