import math

import numpy as np

from apex_fusion_research.core.allan import fit_allan_coefficients, overlapping_allan_deviation
from apex_fusion_research.core.imu_error import ImuErrorConfig, ImuErrorModel, TriadErrorConfig

DT = 1.0 / 120.0


def simulate(cfg, n, w=(0.0, 0.0, 0.0), f=(0.0, 0.0, 9.8)):
    model = ImuErrorModel(cfg)
    gyro = np.empty((n, 3))
    acc = np.empty((n, 3))
    for k in range(n):
        s = model.measure(np.array(w), np.array(f), DT)
        gyro[k], acc[k] = s.angular_velocity, s.specific_force
    return model, gyro, acc


def test_ideal_imu_is_identity():
    _, gyro, acc = simulate(ImuErrorConfig(), 50, w=(0.1, -0.2, 0.3))
    np.testing.assert_allclose(gyro, np.tile([0.1, -0.2, 0.3], (50, 1)))
    np.testing.assert_allclose(acc, np.tile([0.0, 0.0, 9.8], (50, 1)))


def test_white_noise_discrete_std():
    N = 1e-3
    cfg = ImuErrorConfig(gyro=TriadErrorConfig(noise_density=N))
    _, gyro, _ = simulate(cfg, 20000)
    np.testing.assert_allclose(gyro.std(axis=0), N / math.sqrt(DT), rtol=0.03)


def test_constant_errors_and_seed():
    cfg = ImuErrorConfig(
        seed=5,
        accel=TriadErrorConfig(turn_on_bias_std=0.1, scale_factor_std=0.01, misalignment_std_rad=0.01),
    )
    m1, _, acc1 = simulate(cfg, 5)
    m2, _, acc2 = simulate(cfg, 5)
    np.testing.assert_array_equal(acc1, acc2)
    expected = m1.accel.transfer_matrix @ np.array([0, 0, 9.8]) + m1.accel.turn_on_bias
    np.testing.assert_allclose(acc1[0], expected, atol=1e-12)


def test_changing_gyro_does_not_change_accel_realization():
    base = ImuErrorConfig(seed=9, accel=TriadErrorConfig(noise_density=1e-2, turn_on_bias_std=0.05))
    other = ImuErrorConfig(
        seed=9,
        gyro=TriadErrorConfig(noise_density=5e-3, turn_on_bias_std=0.01),
        accel=base.accel,
    )
    _, _, acc_a = simulate(base, 100)
    _, _, acc_b = simulate(other, 100)
    np.testing.assert_array_equal(acc_a, acc_b)


def test_quantization_and_saturation():
    cfg = ImuErrorConfig(gyro=TriadErrorConfig(full_scale=1.0, adc_bits=4))
    _, gyro, _ = simulate(cfg, 1, w=(0.33, 5.0, -5.0))
    lsb = 2.0 / 16
    np.testing.assert_allclose(gyro[0], [round(0.33 / lsb) * lsb, 1.0, -1.0])


def test_allan_recovers_white_noise_and_random_walk():
    N, K = 2e-3, 2e-4
    cfg = ImuErrorConfig(seed=11, gyro=TriadErrorConfig(noise_density=N, bias_random_walk=K))
    _, gyro, _ = simulate(cfg, int(3600 / DT) // 4)  # 15 min at 120 Hz
    tau, adev = overlapping_allan_deviation(gyro[:, 0], DT)
    coef = fit_allan_coefficients(tau, adev)
    assert abs(coef.noise_density / N - 1.0) < 0.1
    assert abs(coef.random_walk / K - 1.0) < 0.35


def test_gauss_markov_stationary_std():
    sigma = 0.01
    cfg = ImuErrorConfig(seed=2, gyro=TriadErrorConfig(bias_instability_std=sigma, bias_correlation_time_s=0.5))
    model = ImuErrorModel(cfg)
    values = []
    for _ in range(60000):
        model.measure(np.zeros(3), np.array([0, 0, 9.8]), DT)
        values.append(model.gyro.bias_gm.copy())
    np.testing.assert_allclose(np.std(values, axis=0), sigma, rtol=0.1)
