"""Realism layer on top of Gazebo's native sensors (imu_chip, lidar_rolling).

Synthetic inputs with analytic references: a 1 kHz IMU stream for the chip
model, and a 2D room seen by a moving sensor for the rolling LiDAR, whose
emulation from fast snapshots (like Gazebo's gpu_lidar) is compared with
exact per-sample ray casting.
"""

import math

import numpy as np

from apex_fusion_research.core.imu_chip import ImuChip, ImuChipConfig, gazebo_noise_from_profile
from apex_fusion_research.core.lidar_noise import LidarNoiseConfig
from apex_fusion_research.core.lidar_rolling import RollingLidar, RollingLidarConfig, bin_time_fraction

G = 9.80665


# --------------------------------------------------------------------- IMU
def _feed(chip: ImuChip, n: int, gyro_z: float = 0.0, noise_std: float = 0.0, seed: int = 0, chunk: int = 7):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 1_000_000
    g = np.zeros((n, 3))
    g[:, 2] = gyro_z
    a = np.tile([0.0, 0.0, G], (n, 1))
    g_noisy = g + rng.standard_normal((n, 3)) * noise_std
    out = []
    for s in range(0, n, chunk):
        out += chip.push(t[s:s + chunk], g_noisy[s:s + chunk], a[s:s + chunk], 0.0, g[s:s + chunk], a[s:s + chunk])
    return out


def test_chip_samples_at_its_own_rate_with_exact_instants():
    chip = ImuChip(ImuChipConfig(odr_hz=104.0, clock_ppm_std=0.0, seed=1))
    out = _feed(chip, 20000)
    t = np.array([o.t_true_ns for o in out]) * 1e-9
    dt = np.diff(t)
    # 104 Hz exactly, not the 9 / 10 ms of a sensor rounded to the 1 ms physics grid.
    assert abs(1.0 / dt.mean() - 104.0) < 0.01
    assert dt.std() < 1e-6
    stamps = np.array([o.stamp_ns for o in out])
    assert np.all(np.diff(stamps) > 0)


def test_chip_filter_shapes_white_noise_to_the_output_density():
    # One-sided output density n through a 2nd-order Butterworth low-pass of
    # cut-off fc: std = n * sqrt(ENBW), ENBW = pi / (2 sqrt 2) fc for this filter.
    n_out, fc = 3.0e-4, 33.0
    sigma_native = n_out * math.sqrt(500.0)  # Gazebo white noise at 1 kHz (gazebo_noise_from_profile)
    chip = ImuChip(ImuChipConfig(odr_hz=104.0, dlpf_gyro_hz=fc, dlpf_accel_hz=50.0, seed=2))
    out = _feed(chip, 60000, noise_std=sigma_native, seed=3)
    gz = np.array([o.gyro[2] for o in out])[50:]
    expect = n_out * math.sqrt(math.pi / (2.0 * math.sqrt(2.0)) * fc)
    assert abs(gz.std() / expect - 1.0) < 0.1


def test_chip_bias_label_tracks_the_native_bias():
    chip = ImuChip(ImuChipConfig(odr_hz=104.0, dlpf_gyro_hz=33.0, dlpf_accel_hz=50.0, seed=4))
    rng = np.random.default_rng(5)
    n = 8000
    t = np.arange(n) * 1_000_000
    clean = np.zeros((n, 3))
    bias = np.array([0.01, -0.1, -0.04])
    noisy = clean + bias + rng.standard_normal((n, 3)) * 5e-3
    a = np.tile([0.0, 0.0, G], (n, 1))
    out = chip.push(t, noisy, a, 0.0, clean, a)
    labels = np.array([o.bias_gyro for o in out])[300:]
    assert np.abs(labels - bias).max() < 1.5e-3


def test_gazebo_noise_of_the_real_profile():
    imu = {"gazebo_noise": {"gyro": {"noise_density": [3e-4, 6e-4, 2.8e-4], "turn_on_bias_std": [0.06, 0.08, 0.05]},
                            "accel": {"noise_density": 1e-2}}}
    gz = gazebo_noise_from_profile(imu, 1000.0)
    assert np.allclose(gz["gyro_white_std"], np.array([3e-4, 6e-4, 2.8e-4]) * math.sqrt(500.0))
    assert gz["gyro_bias_std"] == [0.06, 0.08, 0.05]
    assert np.allclose(gz["accel_white_std"], [1e-2 * math.sqrt(500.0)] * 3)


# ------------------------------------------------------------------- LiDAR
ROOM = [((-4, -3), (5, -3)), ((5, -3), (5, 3)), ((5, 3), (-4, 3)), ((-4, 3), (-4, -3)), ((1.5, 3), (1.5, 1.8)), ((-2, -3), (-1.2, -1.6))]


def _cast(origin: np.ndarray, angles: np.ndarray) -> np.ndarray:
    d = np.column_stack((np.cos(angles), np.sin(angles)))
    best = np.full(len(angles), np.inf)
    for a, b in ROOM:
        a, b = np.asarray(a, float), np.asarray(b, float)
        e = b - a
        den = d[:, 0] * (-e[1]) - d[:, 1] * (-e[0])
        w = a - origin
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (w[0] * (-e[1]) - w[1] * (-e[0])) / den
            u = (d[:, 0] * w[1] - d[:, 1] * w[0]) / den
        hit = (np.abs(den) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1)
        best = np.where(hit & (t < best), t, best)
    return best


def _pose(t: float) -> tuple[float, float, float]:
    """Car driving at 3 m/s while turning at 1.2 rad/s."""
    v, w = 3.0, 1.2
    yaw = w * t
    return (-2.0 + v / w * math.sin(yaw), -1.0 + v / w * (1.0 - math.cos(yaw)), yaw)


def _run(cfg: RollingLidarConfig, seconds: float = 1.2):
    lid = RollingLidar(cfg)
    cap = -math.pi + 2 * math.pi / 1440 * np.arange(1440)
    revs = []
    for k in range(int(seconds * 1000)):
        t = k * 1e-3
        x, y, yaw = _pose(t)
        lid.add_pose(t, x, y, yaw)
        if k % 38 == 0:  # ~26 Hz snapshots, like the gpu_lidar capture
            lid.add_capture(t, _cast(np.array([x, y]), yaw + cap), -math.pi, 2 * math.pi / 1440, (x, y, yaw))
        revs += lid.poll()
    return lid, revs


def test_rolling_emulation_matches_exact_per_sample_casting():
    cfg = RollingLidarConfig(rate_hz=13.0, sample_rate_hz=8000.0, direction="cw", start_angle_deg=89.0, binning="min_range",
                             range_min=0.15, range_max=12.0, noise=LidarNoiseConfig(enabled=False), seed=1)
    lid, revs = _run(cfg)
    assert len(revs) >= 12
    errs, distortion = [], []
    for rv in revs:
        n = rv.samples
        ts = rv.t_start_ns * 1e-9 + np.arange(n) / cfg.sample_rate_hz
        th = math.radians(89.0) - 2 * math.pi * np.arange(n) / (rv.period_s * cfg.sample_rate_hz)
        poses = np.array([_pose(t) for t in ts])
        exact = np.array([_cast(p[:2], np.array([p[2] + a]))[0] for p, a in zip(poses, th)])
        ref = lid._bin(th, np.where((exact >= 0.15) & (exact <= 12.0), exact, np.inf))
        frozen = lid._bin(th, _cast(poses[0, :2], poses[0, 2] + th))
        ok = np.isfinite(ref) & np.isfinite(rv.ideal)
        errs.append(np.abs(ref[ok] - rv.ideal[ok]))
        both = np.isfinite(frozen) & ok
        distortion.append(np.abs(frozen[both] - ref[both]))
    e, dist = np.concatenate(errs), np.concatenate(distortion)
    assert np.median(e) < 1e-3 and np.percentile(e, 95) < 0.01
    # The effect being modelled is much larger than the emulation error.
    assert np.median(dist) > 10 * np.median(e) + 0.005


def test_clockwise_order_occlusion_and_stamps():
    cfg = RollingLidarConfig(rate_hz=13.0, sample_rate_hz=2000.0, direction="cw", start_angle_deg=89.0, binning="min_range",
                             occluded_sectors_deg=((145.0, 205.0),), range_min=0.15, range_max=12.0,
                             noise=LidarNoiseConfig(enabled=False), start_grid_s=0.001, publish_latency_ms=3.0, seed=2)
    lid, revs = _run(cfg, 0.8)
    theta = -180.0 + np.arange(360)
    blind = (theta >= 146) | (theta <= -156)
    for rv in revs:
        assert not np.isfinite(rv.ranges[blind]).any()
        assert rv.t_start_ns % 1_000_000 == 0  # revolution starts on the physics grid
        assert rv.release_ns >= rv.t_end_ns + 3_000_000
        # 2 kHz compatible mode: ~154 samples for 360 bins.
        assert 0.3 < np.isfinite(rv.ranges).mean() < 0.5
    frac = bin_time_fraction(360, -math.pi, 2 * math.pi / 360, "cw", math.radians(89.0))
    assert frac[269] == 0.0 and abs(frac[268] - 1 / 360) < 1e-9 and abs(frac[270] - 359 / 360) < 1e-9  # 89 deg = bin 269
