"""Tests of the streaming odometry stack (run: ``python -m pytest tests`` from
``learning/lidar_imu_pose``). Synthetic 2D worlds, CPU only."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import odometry_metrics as om  # noqa: E402
from icp_odometry import IcpConfig, run_icp  # noqa: E402
from sqlite_streams import StreamData  # noqa: E402
from streaming_model import StreamingPoseNet, rotate_scan  # noqa: E402
from train_streaming import stream_predict  # noqa: E402

BEAMS = 360


def cast(segments, origin, yaws, rmax):
    """Ranges of beams i (world direction yaws[i]) from ``origin`` against segments."""
    d = np.column_stack((np.cos(yaws), np.sin(yaws)))
    best = np.full(len(yaws), np.inf)
    for a, b in segments:
        a, b = np.asarray(a, float), np.asarray(b, float)
        e = b - a
        den = d[:, 0] * (-e[1]) - d[:, 1] * (-e[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            w = a - origin
            t = (w[0] * (-e[1]) - w[1] * (-e[0])) / den
            u = (d[:, 0] * w[1] - d[:, 1] * w[0]) / den
        hit = (np.abs(den) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1)
        best = np.where(hit & (t < best), t, best)
    return np.where(best <= rmax, best, np.inf)


def scan(segments, pose, rmax, rate=0.0, dt_beam=0.0):
    theta = -math.pi + 2 * math.pi / BEAMS * np.arange(BEAMS)
    tau = dt_beam * np.arange(BEAMS)
    return cast(segments, np.asarray(pose[:2]), pose[2] + rate * tau + theta, rmax)


def synthetic_split(segments, poses, rmax=4.0, dt=0.1, accel_noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = len(poses)
    lidar = np.zeros((n, 2, BEAMS), dtype=np.float16)
    for k, p in enumerate(poses):
        r = scan(segments, p, rmax)
        valid = np.isfinite(r)
        lidar[k, 0] = np.where(valid, r / rmax, 1.0)
        lidar[k, 1] = valid
    target = np.zeros((n, 3), dtype=np.float32)
    d = poses[1:, :2] - poses[:-1, :2]
    c, s = np.cos(poses[:-1, 2]), np.sin(poses[:-1, 2])
    target[1:, 0] = c * d[:, 0] + s * d[:, 1]
    target[1:, 1] = -s * d[:, 0] + c * d[:, 1]
    target[1:, 2] = poses[1:, 2] - poses[:-1, 2]
    target[0] = target[1]
    imu = np.zeros((n, 10, 6), dtype=np.float16)
    imu[..., :3] = rng.normal(0.0, accel_noise, (n, 10, 3))  # vibration of a moving car
    imu[..., 5] = (target[:, 2] / dt)[:, None]
    return {
        "split": "synthetic", "lidar": torch.from_numpy(lidar), "imu": torch.from_numpy(imu),
        "imu_len": torch.full((n,), 10, dtype=torch.int64), "dt": torch.full((n,), dt), "gyro_integral": torch.from_numpy(target[:, 2].copy()),
        "target": torch.from_numpy(target), "pose": torch.from_numpy(poses), "t_ns": torch.arange(n, dtype=torch.int64) * int(dt * 1e9),
        "runs": [{"run_id": 0, "run_key": "syn", "track": "syn", "motion": "m", "sensor": "S", "seed": 0, "offset": 0, "n": n}],
        "v_body": torch.zeros(n, 2), "bias_truth": torch.zeros(n, 3), "run_index": torch.zeros(n, dtype=torch.int64),
        "range_max": torch.tensor([rmax]), "angle_min": torch.tensor([-math.pi]), "angle_increment": torch.tensor([2 * math.pi / BEAMS]),
        "time_increment_s": torch.tensor([0.0]), "lidar_xy": torch.zeros(1, 2), "lidar_sigma": torch.tensor([[0.002, 0.0]]),
    }


ROOM = [((-3, -2), (3, -2)), ((3, -2), (3, 2)), ((3, 2), (-3, 2)), ((-3, 2), (-3, -2)), ((1, 2), (1, 1.4)), ((-2, -2), (-1.5, -1.2))]
# Straight corridor 2 m wide with one fin on the left wall at x = 1 (observable
# along the axis while it is within the 4 m range), then nothing but walls.
CORRIDOR = [((-100, 1), (100, 1)), ((-100, -1), (100, -1)), ((1, 1), (1, 0.7))]


def test_rotate_scan_undoes_rotation_and_sweep():
    pose = np.array([0.4, 0.3, 0.0])
    ref = scan(ROOM, pose, 12.0)
    yaw, rate, dt_beam = math.radians(7.3), 0.8, 1e-4  # rad/s during a rolling sweep of 36 ms
    rotated = scan(ROOM, pose + [0, 0, yaw], 12.0, rate=rate, dt_beam=dt_beam)
    r = torch.tensor(np.where(np.isfinite(rotated), rotated, 0.0))
    out, valid, _ = rotate_scan(r, torch.isfinite(torch.tensor(rotated)), torch.tensor(yaw, dtype=r.dtype), torch.tensor(rate * dt_beam, dtype=r.dtype))
    ok = valid.numpy() & np.isfinite(ref)
    err = np.abs(out.numpy()[ok] - ref[ok])
    assert ok.mean() > 0.95
    assert np.percentile(err, 90) < 0.01


def test_icp_corridor_falls_back_to_prediction_instead_of_zero():
    n = 45
    poses = np.column_stack((0.2 * np.arange(n), np.zeros(n), np.zeros(n)))  # 2 m/s along the corridor, starting from v = 0 in the filter
    sd = StreamData(synthetic_split(CORRIDOR, poses, accel_noise=0.02), torch.device("cpu"))
    res = run_icp(sd, IcpConfig(), log=lambda m: None)
    pred, ratio = res["pred"].numpy(), res["eig_ratio"].numpy()
    # Fin in range: the along-axis motion is observed, even from the first
    # interval where the prediction (v = 0) is 20 cm off.
    fin = slice(1, 20)
    assert np.abs(pred[fin, 0] - 0.2).max() < 0.01
    # Fin out of range: the walls say nothing about x (degenerate), and the
    # estimate follows the constant-velocity + IMU prediction instead of
    # collapsing to "no motion" like nearest-neighbour point-to-point ICP.
    pure = slice(26, n)
    assert (ratio[pure] < 1e-3).all()
    assert np.abs(pred[pure, 0] - 0.2).max() < 0.02
    assert np.abs(pred[pure, 1]).max() < 0.005 and np.abs(pred[pure, 2]).max() < 1e-3


def test_streaming_model_is_chunk_invariant():
    torch.manual_seed(0)
    n = 23
    t = np.arange(n) * 0.1
    poses = np.column_stack((0.5 * np.sin(t), 0.3 * t, 0.2 * t))
    sd = StreamData(synthetic_split(ROOM, poses, rmax=12.0), torch.device("cpu"))
    model = StreamingPoseNet(hidden=32)
    a = stream_predict(model, sd, chunk=5)["pred"]
    b = stream_predict(model, sd, chunk=64)["pred"]
    assert torch.isnan(a[0]).all() and torch.isfinite(a[1:]).all()
    assert torch.allclose(a[1:], b[1:], atol=1e-5)


def test_metrics_perfect_and_scaled_predictions():
    n = 400
    poses = np.column_stack((0.25 * np.arange(n), np.zeros(n), np.zeros(n)))
    data = synthetic_split(CORRIDOR, poses)
    truth = om.split_truth(data)
    perfect = om.evaluate(truth.target.copy(), truth)["all"]
    assert perfect["speed_err_cmps"] == pytest.approx(0.0, abs=1e-9)
    assert perfect["t_rel_pct"] == pytest.approx(0.0, abs=1e-9)
    scaled = truth.target.copy()
    scaled[:, :2] *= 1.1
    m = om.evaluate(scaled, truth)["all"]
    assert m["rel_err_pct"] == pytest.approx(10.0, rel=1e-4)
    assert m["t_rel_10m_pct"] == pytest.approx(10.0, rel=0.03)
    assert m["speed_err_cmps"] == pytest.approx(25.0, rel=1e-3)


def test_mirror_augmentation_matches_a_mirrored_world():
    """The model on (data, mirror=True) works in the mirrored world: it must
    give exactly what it gives on data simulated in the mirrored world
    (y -> -y); the training labels are mirrored accordingly.
    The rolling-sweep time channel is excluded (a mirrored world is scanned by
    a clockwise scanner, which the synthetic generator does not model)."""
    from streaming_model import IMU_MIRROR, mirror_channels, pair_channels
    torch.manual_seed(1)
    n = 12
    t = np.arange(n) * 0.1
    poses = np.column_stack((0.4 * t, 0.3 * np.sin(t), 0.25 * t))
    world_m = [((a[0], -a[1]), (b[0], -b[1])) for a, b in ROOM]
    poses_m = poses * [1, -1, -1]
    data = synthetic_split(ROOM, poses, rmax=12.0)
    data_m = synthetic_split(world_m, poses_m, rmax=12.0)
    data_m["imu"] = (data["imu"].float() * torch.tensor(IMU_MIRROR)).half()
    sd, sd_m = StreamData(data, torch.device("cpu")), StreamData(data_m, torch.device("cpu"))
    idx = torch.arange(1, n)[None]
    b, b_m = sd.gather(idx), sd_m.gather(idx)
    ch = mirror_channels(pair_channels(b), torch.tensor([True]))
    ch_m = pair_channels(b_m)
    keep = [0, 1, 2, 3, 4, 6, 7]
    assert torch.allclose(ch[:, :, keep], ch_m[:, :, keep], atol=1e-4)  # the range difference is scaled x4
    model = StreamingPoseNet(hidden=32).eval()
    with torch.no_grad():
        model.pair_encoder.cnn[0].weight[:, 5] = 0.0  # ignore the sweep-time channel
        out, _ = model({**b, "mirror": torch.tensor([True])}, model.initial_state(1, torch.device("cpu")))
        out_m, _ = model(b_m, model.initial_state(1, torch.device("cpu")))
    assert torch.allclose(out["delta"], out_m["delta"], atol=1e-4)
    assert (out["delta"][..., 2] * torch.from_numpy(data["target"][1:, 2].numpy()) < 0).all()  # really the mirrored yaw


def test_hybrid_model_streams_with_icp_inputs():
    """The hybrid variant takes the classical odometry as input; its streaming
    output must not depend on the chunking either."""
    torch.manual_seed(2)
    n = 23
    t = np.arange(n) * 0.1
    poses = np.column_stack((0.5 * np.sin(t), 0.3 * t, 0.2 * t))
    sd = StreamData(synthetic_split(ROOM, poses, rmax=12.0), torch.device("cpu"))
    sd.attach_icp(run_icp(sd, IcpConfig(), log=lambda m: None))
    model = StreamingPoseNet(hidden=32, hybrid=True)
    a = stream_predict(model, sd, chunk=5)["pred"]
    b = stream_predict(model, sd, chunk=64)["pred"]
    assert torch.isfinite(a[1:]).all()
    assert torch.allclose(a[1:], b[1:], atol=1e-5)


def test_submap_reference_reduces_drift_on_noisy_scans():
    """With 2 cm range noise, registering against a local map of keyframes
    accumulates much less error than scan-to-scan (errors grow per keyframe,
    not per scan)."""
    n, sigma = 60, 0.02
    errors = {"scan": [], "submap": []}
    for seed in range(3):
        poses = np.column_stack((-2.0 + 0.05 * np.arange(n), 0.2 * np.sin(np.arange(n) * 0.1), 0.02 * np.arange(n)))
        data = synthetic_split(ROOM, poses, rmax=12.0, accel_noise=0.02, seed=seed)
        rng = np.random.default_rng(seed + 100)
        lidar = data["lidar"].float().numpy()
        lidar[:, 0] = np.where(lidar[:, 1] > 0.5, lidar[:, 0] + rng.normal(0.0, sigma / 12.0, lidar[:, 0].shape), lidar[:, 0])
        data["lidar"], data["lidar_sigma"] = torch.from_numpy(lidar).half(), torch.tensor([[sigma, 0.0]])
        sd = StreamData(data, torch.device("cpu"))
        gt = om.compose(data["target"].numpy()[1:].astype(np.float64))
        for name, cfg in (("scan", IcpConfig()), ("submap", IcpConfig(submap_keyframes=5))):
            est = om.compose(run_icp(sd, cfg, log=lambda m: None)["pred"].numpy()[1:].astype(np.float64))
            errors[name].append(np.linalg.norm(est[-1, :2] - gt[-1, :2]))
    assert np.mean(errors["submap"]) < 0.5 * np.mean(errors["scan"])
