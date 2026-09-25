"""Streaming LiDAR-inertial odometry network (v2).

Changes with respect to the three architectures of ``architectures.py``,
each one aimed at a limit measured on the grid-search model:

1. LiDAR scans are compared *before* any pooling. The previous encoder
   pooled every scan into one global vector (convolutions + global average
   pooling), which is almost invariant to a circular shift of the beams, i.e.
   to a rotation of the car: the heading was left to the gyro alone. Here the
   two scans of an interval are first put in the same frame with the raw gyro
   (the current scan rotated by the gyro increment, and both scans corrected
   for the rotation during their own rolling sweep), then stacked as channels
   together with their range difference, the beam direction (cos, sin) and
   the time of the beam inside the sweep. A circular-padded CNN keeps the
   angular layout (pooling to 9 sectors, not to 1), so the features are not
   rotation invariant and the residual rotation (= gyro error) is visible.
2. State persists across the whole run. The fusion GRU hidden state, the
   body velocity and the IMU bias estimates are carried from interval to
   interval (training: truncated back-propagation through time on lanes that
   walk through entire runs; evaluation: one pass over every run from its
   first scan). The previous model re-estimated the velocity from scratch
   every 6 intervals, which produced errors correlated inside each window.
3. Learned Kalman-style update instead of free regression. For every interval
   the IMU predicts the motion (dp = v dt + f dt^2 / 2, heading = gyro -
   bias); the LiDAR branch gives a measurement of the same motion; the GRU,
   which sees the innovation, outputs the gains. The velocity is propagated
   in the body frame (rotated by the heading increment; the old mechanization
   ignored that rotation). The gyro bias is an explicit state that can only
   move towards the bias the LiDAR measures (residual rotation / dt), with a
   learned gain bounded to 0.2 per interval: a free learned increment was
   tried first and the network used that integrator as spare memory (the
   bias drifted to the same wrong value on every sensor). The accelerometer
   biases are small bounded learned increments.

Regularisation (the first runs fitted the 5 training tracks: 1.2 cm per
interval on train, about 2 cm on the unseen validation track): the angular
sectors go through a 1x1 bottleneck (128 -> 32 channels) with dropout before
the dense projection, dropout at the input of the recurrent core, and an
optional left-right mirror of the whole world per lane (``batch["mirror"]``):
the scan channels are flipped in angle after the gyro compensation (the
rolling-sweep time channel is flipped with them, so it stays correct), and
the y / yaw components of the IMU and of the labels change sign.

Hybrid variant (``hybrid=True``): the classical scan-to-scan odometry of
``icp_odometry`` (point-to-line ICP + IMU prior + gyro-bias filter, run
causally on the same sensor data) is an extra input. Its increment becomes
the LiDAR measurement of the Kalman-style update, corrected by a bounded
learned term (+-5 cm, +-0.17 deg) computed from the scan-pair features and
the ICP quality indicators (sigmas, weakest/strongest information ratio =
corridor degeneracy, correspondences, standstill flag, its gyro bias). The
gains start close to "trust the ICP", so the network begins from the
classical solution and learns where to depart from it.

Inputs are only the noisy scans, the IMU samples and their timestamps; the
exact pose, velocity and bias are labels.

The architecture and the structured updates are this project's design; the
building blocks are standard:

References
  [1] G. Revach, N. Shlezinger, X. Ni, A. L. Escoriza, R. J. G. van Sloun,
      Y. C. Eldar, "KalmanNet: Neural Network Aided Kalman Filtering for
      Partially Known Dynamics", IEEE Trans. Signal Processing 70, 2022
      (learned gains inside a Kalman structure, innovation as input).
  [2] P. D. Groves, "Principles of GNSS, Inertial, and Multisensor Integrated
      Navigation Systems", 2nd ed., Artech House, 2013 (body-frame velocity
      mechanization, IMU biases).
  [3] J. Zhang, S. Singh, "LOAM: Lidar Odometry and Mapping in Real-time",
      RSS 2014 (rotation compensation of a rolling scan).
  [4] K. Cho et al., "Learning Phrase Representations using RNN
      Encoder-Decoder for Statistical Machine Translation", EMNLP 2014 (GRU).
  [5] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, R. Salakhutdinov,
      "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
      JMLR 15, 2014.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from train_pose_fusion import ImuEncoder

TWO_PI = 2.0 * math.pi


def rotate_scan(ranges: Tensor, valid: Tensor, shift: Tensor, sweep_rate: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Resample a 360-degree scan on its own angular grid after a rotation.

    Beam i (angle theta_i) is re-expressed at theta_i + shift + sweep_rate * i
    (``sweep_rate`` in radians per beam: rotation during the rolling sweep).
    Returns ranges, validity and the (fractional) source beam index / N of
    every output direction. Neighbouring beams are interpolated only when
    both are valid and lie on one surface; otherwise the nearest one is used.
    """
    n = ranges.shape[-1]
    inc = TWO_PI / n
    j = torch.arange(n, device=ranges.device, dtype=ranges.dtype)
    u = (j * inc - shift[..., None]) / (inc + sweep_rate[..., None])
    u = torch.remainder(u, n)
    i0 = torch.floor(u).long().clamp(max=n - 1)
    w = u - i0
    i1 = (i0 + 1) % n
    r0, r1 = ranges.gather(-1, i0), ranges.gather(-1, i1)
    v0, v1 = valid.gather(-1, i0), valid.gather(-1, i1)
    smooth = v0 & v1 & ((r0 - r1).abs() < 0.05 + 0.05 * torch.minimum(r0, r1))
    near_first = w < 0.5
    r = torch.where(smooth, r0 * (1 - w) + r1 * w, torch.where(near_first, r0, r1))
    v = torch.where(smooth, torch.ones_like(v0), torch.where(near_first, v0, v1))
    return r * v, v, u / n


def pair_channels(batch: dict[str, Tensor]) -> Tensor:
    """[B, T, 8, beams] input of the pair encoder, both scans in the frame of
    the previous scan, rotation compensated with the raw gyro."""
    beams = batch["prev_ranges"].shape[-1]
    inc = TWO_PI / beams
    dt_beam = batch["time_increment"]
    prev_r, prev_v, _ = rotate_scan(batch["prev_ranges"], batch["prev_valid"], torch.zeros_like(batch["dt"]), batch["w_prev"] * dt_beam)
    cur_r, cur_v, tau = rotate_scan(batch["cur_ranges"], batch["cur_valid"], batch["gyro_integral"], batch["w_cur"] * dt_beam)
    both = (prev_v & cur_v).float()
    theta = -math.pi + inc * torch.arange(beams, device=prev_r.device, dtype=prev_r.dtype)
    shape = prev_r.shape
    diff = (cur_r - prev_r) * both / 0.25
    return torch.stack((
        prev_r / 8.0, prev_v.float(), cur_r / 8.0, cur_v.float(),
        3.0 * torch.tanh(diff / 3.0), tau,
        torch.cos(theta).expand(shape), torch.sin(theta).expand(shape),
    ), dim=-2)


def _conv(cin: int, cout: int, k: int, stride: int) -> list[nn.Module]:
    return [nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2, padding_mode="circular"), nn.GroupNorm(8, cout), nn.SiLU()]


# Mirror (y -> -y) of the IMU channels (ax, ay, az, gx, gy, gz): the specific
# force is a vector, the angular rate a pseudovector.
IMU_MIRROR = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
SCAN_CHANNELS = 6  # channels flipped in angle by the mirror (not cos / sin of the beam)


def mirror_channels(ch: Tensor, mirror: Tensor) -> Tensor:
    """Flip the scan channels in angle (theta -> -theta) for mirrored lanes."""
    n = ch.shape[-1]
    perm = (n - torch.arange(n, device=ch.device)) % n
    flip = torch.zeros(ch.shape[-2], dtype=torch.bool, device=ch.device)
    flip[:SCAN_CHANNELS] = True
    return torch.where(mirror[:, None, None, None] & flip[None, None, :, None], ch[..., perm], ch)


class PairEncoder(nn.Module):
    def __init__(self, hidden: int, channels: int = 8, sectors: int = 9, bottleneck: int | None = 32, dropout: float = 0.2) -> None:
        super().__init__()
        convs = [*_conv(channels, 48, 7, 1), *_conv(48, 64, 5, 2), *_conv(64, 96, 5, 2), *_conv(96, 128, 3, 2), nn.AdaptiveAvgPool1d(sectors)]
        if bottleneck is None:  # layout of the unregularised run (attempt 2)
            self.cnn = nn.Sequential(*convs)
            self.project = nn.Sequential(nn.Flatten(), nn.Linear(128 * sectors, hidden), nn.SiLU())
        else:
            self.cnn = nn.Sequential(*convs, nn.Conv1d(128, bottleneck, 1), nn.SiLU())
            self.project = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(bottleneck * sectors, hidden), nn.SiLU())

    def forward(self, x: Tensor) -> Tensor:
        b, t, c, n = x.shape
        return self.project(self.cnn(x.reshape(b * t, c, n))).reshape(b, t, -1)


def _head(inp: int, hidden: int, out: int, init_scale: float = 1.0) -> nn.Sequential:
    head = nn.Sequential(nn.Linear(inp, hidden), nn.SiLU(), nn.Linear(hidden, out))
    with torch.no_grad():
        head[-1].weight.mul_(init_scale)
        head[-1].bias.zero_()
    return head


@dataclass(frozen=True)
class StreamScales:
    lidar_xy_m: float = 0.6  # LiDAR translation measurement range per interval
    lidar_yaw_rad: float = 0.02  # residual rotation after gyro compensation
    residual_xy_m: float = 0.01
    residual_yaw_rad: float = 0.002
    bias_gain_max: float = 0.2  # gyro-bias update gain per interval
    accel_bias_step: float = 0.005  # m/s^2 per interval
    bias_max: tuple[float, float, float] = (0.15, 0.6, 0.6)  # gyro z [rad/s], accel x, y [m/s^2]


PHYS = 19  # physical scalars fed to the recurrent core
ICP_INPUTS = 12  # hybrid: ICP increment (3) + quality indicators (9, see StreamData.attach_icp)
ICP_FEAT_MIRROR = (1.0,) * 8 + (-1.0,)  # the ICP gyro-bias feature changes sign in a mirrored world


class StreamingPoseNet(nn.Module):
    def __init__(self, hidden: int = 128, scales: StreamScales = StreamScales(), regularize: bool = True, hybrid: bool = False) -> None:
        super().__init__()
        self.hidden = hidden
        self.scales = scales
        self.hybrid = hybrid
        extra = ICP_INPUTS if hybrid else 0
        self.pair_encoder = PairEncoder(hidden) if regularize else PairEncoder(hidden, bottleneck=None)
        self.imu_encoder = ImuEncoder(hidden)
        self.measure = _head(hidden + extra, hidden, 3, init_scale=0.1 if hybrid else 1.0)
        self.inp = nn.Sequential(nn.Linear(2 * hidden + PHYS + extra, hidden), nn.SiLU(), nn.Dropout(0.1 if regularize else 0.0))
        self.cells = nn.ModuleList([nn.GRUCell(hidden, hidden), nn.GRUCell(hidden, hidden)])
        self.gain_head = _head(hidden, hidden // 2, 5)
        if hybrid:
            with torch.no_grad():  # K_p, K_yaw ~ 0.88 at start: begin from the ICP estimate
                self.gain_head[-1].bias.copy_(torch.tensor([2.0, 2.0, 0.0, 0.0, 2.0]))
        self.residual_head = _head(hidden, hidden // 2, 3, init_scale=0.1)
        self.bias_head = _head(hidden, hidden // 2, 3, init_scale=0.1)
        self.logvar_head = _head(hidden, hidden // 2, 3)
        self.register_buffer("bias_max", torch.tensor(scales.bias_max))

    def initial_state(self, lanes: int, device: torch.device) -> dict[str, Tensor]:
        return {
            "h": torch.zeros(len(self.cells), lanes, self.hidden, device=device),
            "v": torch.zeros(lanes, 2, device=device),
            "bias": torch.zeros(lanes, 3, device=device),
            "age": torch.zeros(lanes, device=device),
        }

    @staticmethod
    def reset(state: dict[str, Tensor], lanes: Tensor) -> dict[str, Tensor]:
        """Fresh state for the lanes in the boolean mask ``lanes``."""
        keep = (~lanes).float()
        return {"h": state["h"] * keep[None, :, None], "v": state["v"] * keep[:, None], "bias": state["bias"] * keep[:, None], "age": state["age"] * keep}

    def forward(self, batch: dict[str, Tensor], state: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        sc = self.scales
        channels = pair_channels(batch)
        imu_in, dt_all, gyro_all, acc_all = batch["imu"], batch["dt"], batch["gyro_integral"], batch["accel"]
        mirror = batch.get("mirror")
        if mirror is not None:
            channels = mirror_channels(channels, mirror)
            sign = 1.0 - 2.0 * mirror.float()
            imu_in = imu_in * torch.where(mirror[:, None, None, None], imu_in.new_tensor(IMU_MIRROR), imu_in.new_ones(6))
            gyro_all = gyro_all * sign[:, None]
            acc_all = acc_all * torch.stack((torch.ones_like(sign), sign), dim=1)[:, None, :]
        pair = self.pair_encoder(channels)
        imu = self.imu_encoder(imu_in, batch["imu_lengths"])
        if self.hybrid:
            icp, feat = batch["icp_pred"], batch["icp_feat"]
            if mirror is not None:
                icp = icp * torch.where(mirror[:, None, None], icp.new_tensor([1.0, -1.0, -1.0]), icp.new_ones(3))
                feat = feat * torch.where(mirror[:, None, None], feat.new_tensor(ICP_FEAT_MIRROR), feat.new_ones(len(ICP_FEAT_MIRROR)))
            icp_meas = torch.cat((icp[..., :2], icp[..., 2:] - gyro_all[..., None]), dim=-1)  # rotation vs raw gyro, like z
            icp_in = torch.cat((icp[..., :2] / 0.3, icp_meas[..., 2:] / 0.005, feat), dim=-1)
            z = icp_meas + torch.tanh(self.measure(torch.cat((pair, icp_in), dim=-1))) * pair.new_tensor([0.05, 0.05, 0.003])
        else:
            icp_in = None
            z = torch.tanh(self.measure(pair)) * pair.new_tensor([sc.lidar_xy_m, sc.lidar_xy_m, sc.lidar_yaw_rad])
        h = [state["h"][i] for i in range(len(self.cells))]
        v, bias, age = state["v"], state["bias"], state["age"]
        out = {name: [] for name in ("delta", "logvar", "velocity", "bias", "gain")}
        for t in range(pair.shape[1]):
            dt = dt_all[:, t : t + 1]
            f = acc_all[:, t] - bias[:, 1:]
            dp_pred = v * dt + 0.5 * f * dt.square()
            nu = z[:, t, :2] - dp_pred
            nu_yaw = z[:, t, 2:] + bias[:, :1] * dt  # LiDAR residual rotation vs. the one predicted by the bias
            phys = torch.cat((
                v / 3.0, dp_pred / 0.3, nu / 0.1, nu_yaw / 0.005, z[:, t, :2] / 0.3, z[:, t, 2:] / 0.005,
                bias / self.bias_max * 3.0, dt / 0.1, gyro_all[:, t : t + 1] / 0.05, f / 3.0,
                torch.clamp(age[:, None] / 50.0, max=1.0), (age[:, None] == 0).float(),
            ), dim=1)
            if icp_in is not None:
                phys = torch.cat((phys, icp_in[:, t]), dim=1)
            x = self.inp(torch.cat((pair[:, t], imu[:, t], phys), dim=1))
            for i, cell in enumerate(self.cells):
                h[i] = cell(x, h[i])
                x = h[i]
            gains = torch.sigmoid(self.gain_head(x))
            res = torch.tanh(self.residual_head(x)) * x.new_tensor([sc.residual_xy_m, sc.residual_xy_m, sc.residual_yaw_rad])
            k_p, k_v, k_yaw = gains[:, :2], gains[:, 2:4], gains[:, 4:5]
            dp = dp_pred + k_p * nu + res[:, :2]
            dyaw = gyro_all[:, t : t + 1] - bias[:, :1] * dt + k_yaw * nu_yaw + res[:, 2:]
            v_end = v + f * dt + k_v * nu / dt
            c, s = torch.cos(dyaw), torch.sin(dyaw)
            v = torch.cat((c * v_end[:, :1] + s * v_end[:, 1:], -s * v_end[:, :1] + c * v_end[:, 1:]), dim=1)
            # Gyro bias: move towards the LiDAR-measured bias (the residual
            # rotation after gyro compensation is -bias * dt).
            upd = self.bias_head(x)
            measured = torch.maximum(torch.minimum(-z[:, t, 2:] / dt, self.bias_max[:1]), -self.bias_max[:1])
            gyro_bias = bias[:, :1] + torch.sigmoid(upd[:, :1]) * sc.bias_gain_max * (measured - bias[:, :1])
            accel_bias = bias[:, 1:] + torch.tanh(upd[:, 1:]) * sc.accel_bias_step
            bias = torch.maximum(torch.minimum(torch.cat((gyro_bias, accel_bias), dim=1), self.bias_max), -self.bias_max)
            age = age + 1
            out["delta"].append(torch.cat((dp, dyaw), dim=1))
            out["logvar"].append(self.logvar_head(x).clamp(-8.0, 8.0))
            out["velocity"].append(v)
            out["bias"].append(bias)
            out["gain"].append(gains)
        result = {name: torch.stack(vals, dim=1) for name, vals in out.items()}
        result["lidar_measurement"] = z
        return result, {"h": torch.stack(h), "v": v, "bias": bias, "age": age}
