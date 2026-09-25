"""The three LiDAR–IMU pose architectures developed in ``train_pose_fusion.py``.

The history of this folder produced three models (checkpoints in
``outputs/lidar_imu_pose``):

``single_context_direct``      (``best_model_before_velocity.pt``)
    shared LiDAR CNN1D + one GRU over the scan context, IMU CNN+GRU, fusion
    GRU, a motion head that regresses the increment directly.
``single_context_velocity``    (``best_model_before_dual_scan.pt``)
    same encoders plus the recurrent velocity mechanization
    ``v[t] = v[t-1] + a dt + dv_net``, ``dp = v dt + a dt^2 / 2 + dp_net``.
``dual_pair_context_velocity`` (``best_model.pt``, current ``LidarImuPoseNet``)
    pair GRU (last 2 scans) + context GRU (all scans) with a learned gate,
    plus the velocity mechanization.

Module names and shapes are identical to those checkpoints (verified by
``load_state_dict(strict=True)`` in ``gridsearch_sqlite.py``). Two changes are
needed for the multi-scenario dataset and are applied to all variants:

* the output caps (``motion_scale``, initial velocity, velocity residual) were
  tuned for the 0.24 m/s single-track capture; they are rescaled to the
  dataset (speeds up to 4.4 m/s, 0.43 m per scan interval);
* the IMU means over each interval ignore the zero padding (the original
  ``mean(dim=-2)`` averaged the padding too, so the result depended on the
  longest interval of the batch).

All variants keep the physics-informed yaw: gyro integral minus the
estimated bias plus a bounded learned residual.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from train_pose_fusion import ImuEncoder, LidarDualEncoder

ARCHITECTURES = ("single_context_direct", "single_context_velocity", "dual_pair_context_velocity")


@dataclass(frozen=True)
class OutputScales:
    residual_xy_m: float = 0.05  # learned translation residual per interval (velocity models)
    residual_yaw_rad: float = 0.05  # learned yaw residual on top of the gyro integral
    direct_xy_m: float = 0.50  # direct translation regression (no velocity model)
    velocity_init_mps: float = 5.0  # initial body velocity at the window start
    velocity_residual_mps: float = 0.30  # learned velocity correction per interval


class LidarContextEncoder(nn.Module):
    """Single-branch encoder of the first two architectures."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 32, 9, stride=2, padding=4), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv1d(32, 64, 7, stride=2, padding=3), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv1d(64, 96, 5, stride=2, padding=2), nn.GroupNorm(8, 96), nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.project = nn.Linear(96, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)

    def forward(self, scans: Tensor) -> Tensor:
        b, t, context, channels, beams = scans.shape
        encoded = self.cnn(scans.reshape(b * t * context, channels, beams)).squeeze(-1)
        encoded = self.project(encoded).reshape(b * t, context, -1)
        _, state = self.gru(encoded)
        return state[-1].reshape(b, t, -1)


def _head(inp: int, hidden: int, out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inp, hidden), nn.SiLU(), nn.Linear(hidden, out))


class PoseNet(nn.Module):
    def __init__(self, architecture: str, hidden: int = 96, scales: OutputScales = OutputScales()) -> None:
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"unknown architecture {architecture!r}")
        self.architecture = architecture
        self.dual = architecture == "dual_pair_context_velocity"
        self.velocity_model = architecture != "single_context_direct"
        self.lidar_encoder = LidarDualEncoder(hidden) if self.dual else LidarContextEncoder(hidden)
        self.imu_encoder = ImuEncoder(hidden)
        fuse_in = (3 if self.dual else 2) * hidden
        self.fuse = nn.Sequential(nn.Linear(fuse_in, 2 * hidden), nn.SiLU(), nn.Dropout(0.10), nn.Linear(2 * hidden, hidden))
        self.fusion_gru = nn.GRU(hidden, hidden, num_layers=2, dropout=0.10, batch_first=True)
        if self.dual:
            self.pair_motion_head = _head(hidden, hidden, 3)
            self.context_motion_head = _head(hidden, hidden, 3)
            self.context_gate_head = _head(hidden, hidden // 2, 3)
        else:
            self.motion_head = _head(hidden, hidden, 3)
        self.logvar_head = _head(hidden, hidden // 2, 3)
        self.bias_head = _head(hidden, hidden // 2, 6)
        if self.velocity_model:
            self.velocity_init_head = _head(hidden, hidden // 2, 2)
            self.velocity_residual_head = _head(hidden, hidden // 2, 2)
            motion = [scales.residual_xy_m, scales.residual_xy_m, scales.residual_yaw_rad]
        else:
            motion = [scales.direct_xy_m, scales.direct_xy_m, scales.residual_yaw_rad]
        self.scales = scales
        self.register_buffer("motion_scale", torch.tensor(motion))
        self.register_buffer("bias_scale", torch.tensor([0.20, 0.20, 0.20, 0.02, 0.02, 0.02]))

    def lidar_from_unique(self, scans: Tensor, steps: int) -> tuple[Tensor, Tensor | None]:
        """Same features as the encoders, computing the CNN once per unique scan.

        ``scans``: [B, steps + context - 1, 2, beams], the consecutive scans of
        the window. Target step t uses scans t .. t + context - 1. The CNN is
        applied per scan (GroupNorm is per sample), so the result is identical
        to encoding the [B, steps, context, 2, beams] tensor.
        """
        enc = self.lidar_encoder
        b, n, channels, beams = scans.shape
        context = n - steps + 1
        feats = enc.project(enc.cnn(scans.reshape(b * n, channels, beams)).squeeze(-1)).reshape(b, n, -1)
        windows = feats.unfold(1, context, 1).permute(0, 1, 3, 2).reshape(b * steps, context, -1)
        if self.dual:
            _, pair_state = enc.pair_gru(windows[:, -2:])
            _, context_state = enc.context_gru(windows)
            return pair_state[-1].reshape(b, steps, -1), context_state[-1].reshape(b, steps, -1)
        _, state = enc.gru(windows)
        return state[-1].reshape(b, steps, -1), None

    def forward(self, lidar: Tensor, imu: Tensor, imu_lengths: Tensor, dt: Tensor) -> dict[str, Tensor]:
        """``lidar``: [B, T, context, 2, beams] (original layout) or
        [B, T + context - 1, 2, beams] (unique consecutive scans)."""
        steps = imu.shape[1]
        if lidar.dim() == 4:
            first, second = self.lidar_from_unique(lidar, steps)
            pair_features, context_features = (first, second) if self.dual else (None, None)
            lidar_features = torch.cat((first, second), dim=-1) if self.dual else first
        elif self.dual:
            pair_features, context_features = self.lidar_encoder(lidar)
            lidar_features = torch.cat((pair_features, context_features), dim=-1)
        else:
            lidar_features = self.lidar_encoder(lidar)
        imu_features = self.imu_encoder(imu, imu_lengths)
        fused = self.fuse(torch.cat((lidar_features, imu_features), dim=-1))
        temporal, _ = self.fusion_gru(fused)
        bias = self.bias_head(temporal) * self.bias_scale
        out: dict[str, Tensor] = {"bias": bias, "logvar": self.logvar_head(temporal).clamp(-8.0, 3.0)}
        if self.dual:
            pair_motion = torch.tanh(self.pair_motion_head(pair_features)) * self.motion_scale
            gate = torch.sigmoid(self.context_gate_head(temporal))
            residual = pair_motion + gate * torch.tanh(self.context_motion_head(temporal)) * self.motion_scale
            out["context_gate"] = gate
        else:
            residual = torch.tanh(self.motion_head(temporal)) * self.motion_scale
        # Masked means over the real IMU samples of each interval.
        mask = (torch.arange(imu.shape[2], device=imu.device)[None, None, :] < imu_lengths[..., None]).to(imu.dtype)
        count = mask.sum(-1).clamp(min=1.0)
        acc_mean = (imu[..., :2] * mask[..., None]).sum(-2) / count[..., None]
        gyro_mean = (imu[..., 5] * mask).sum(-1) / count
        gyro_yaw = (gyro_mean - bias[..., 5]) * dt
        yaw = residual[..., 2:3] + gyro_yaw.unsqueeze(-1)
        if self.velocity_model:
            velocity = torch.tanh(self.velocity_init_head(temporal[:, 0])) * self.scales.velocity_init_mps
            delta_velocity = torch.tanh(self.velocity_residual_head(temporal)) * self.scales.velocity_residual_mps
            acceleration = acc_mean * 9.80665 - bias[..., :2]
            translations, velocities = [], []
            for step in range(temporal.shape[1]):
                step_dt = dt[:, step : step + 1]
                translations.append(velocity * step_dt + 0.5 * acceleration[:, step] * step_dt.square() + residual[:, step, :2])
                velocity = velocity + acceleration[:, step] * step_dt + delta_velocity[:, step]
                velocities.append(velocity)
            translation = torch.stack(translations, dim=1)
            out["velocity"] = torch.stack(velocities, dim=1)
        else:
            translation = residual[..., :2]
        out["delta"] = torch.cat((translation, yaw), dim=-1)
        return out
