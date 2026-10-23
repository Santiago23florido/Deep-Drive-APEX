"""Sensor timing on the car: MCU / LiDAR clocks, IMU lines, RPLIDAR Express scan.

No ROS and no hardware: the LiDAR is a synthetic A2M8 that encodes its
samples into Express capsules exactly as the protocol defines them, and
delivers them with a random USB latency in random chunks.
"""

import numpy as np
import pytest

from apex_telemetry.imu.imu_line import parse_imu_line
from apex_telemetry.perception.rplidar_express import (
    CAPSULE_BYTES,
    CMD_EXPRESS_SCAN,
    CMD_SET_MOTOR_PWM,
    Capsule,
    DeviceInfo,
    ExpressScanPipeline,
    MotorRegulator,
    RPLidarExpressDriver,
    SupplyGuard,
    bin_revolution,
    command_packet,
    decode_capsule_pair,
    find_undervoltage_alarm,
    laser_angle_deg,
    motor_ramp,
)
from apex_telemetry.timing import CounterUnwrapper, DeviceClockMapper

BASE_NS = 1_790_000_000 * 10**9


# --- device clock ---------------------------------------------------------------
def _mapper_errors(rate_hz, duration_s, drift, lat_min_s, lat_mean_s, spike_prob, seed=0):
    rng = np.random.default_rng(seed)
    mapper = DeviceClockMapper()
    t_true = np.arange(0.0, duration_s, 1.0 / rate_hz)
    device = t_true * (1.0 + drift) + 42.0
    latency = lat_min_s + rng.exponential(lat_mean_s, t_true.size)
    latency[rng.random(t_true.size) < spike_prob] += 0.03
    errors = []
    for d, t, lat in zip(device, t_true, latency):
        out = mapper.add(d, BASE_NS + int(round((t + lat) * 1e9)))
        errors.append((out - BASE_NS) * 1e-9 - (t + lat_min_s))
    return t_true, np.asarray(errors), mapper


def test_clock_mapper_removes_reception_jitter_and_drift():
    t, err, mapper = _mapper_errors(104.0, 60.0, drift=80e-6, lat_min_s=3e-4, lat_mean_s=1e-3, spike_prob=0.01)
    settled = t > 3.0
    # Stamped on reception the error would be the latency: ~1 ms mean, 30 ms spikes.
    assert np.max(np.abs(err[settled])) < 2e-4
    assert mapper.stats().drift_ppm == pytest.approx(-80.0, abs=10.0)


def test_clock_mapper_follows_a_large_drift():
    t, err, _ = _mapper_errors(125.0, 60.0, drift=-3e-3, lat_min_s=3e-4, lat_mean_s=1e-3, spike_prob=0.0)
    assert np.max(np.abs(err[t > 3.0])) < 5e-4


def test_clock_mapper_never_stamps_after_reception():
    rng = np.random.default_rng(3)
    mapper = DeviceClockMapper()
    for i in range(3000):
        host = BASE_NS + int((i * 0.01 + 5e-4 + rng.exponential(1e-3)) * 1e9)
        assert mapper.add(i * 0.01, host) <= host


def test_clock_mapper_resets_when_the_device_restarts():
    mapper = DeviceClockMapper()
    for i in range(500):
        mapper.add(i * 0.01, BASE_NS + int(i * 0.01e9) + 300_000)
    out = mapper.add(0.0, BASE_NS + int(5.0e9))  # device time back to 0
    assert mapper.resets == 1
    assert out == BASE_NS + int(5.0e9)


def test_counter_unwrapper_handles_the_micros_wrap():
    unwrap = CounterUnwrapper(32)
    assert [unwrap(v) for v in (4294967000, 4294967290, 100, 5000)] == [4294967000, 4294967290, 2**32 + 100, 2**32 + 5000]


# --- IMU lines -------------------------------------------------------------------
def test_imu_line_v2_carries_the_mcu_time():
    s = parse_imu_line("0.1,-0.2,9.8,0.01,0.02,-0.03,123456789,42")
    assert (s.ax, s.az, s.gz, s.has_gyro) == (0.1, 9.8, -0.03, True)
    assert (s.t_us, s.seq) == (123456789, 42)


def test_imu_line_v1_and_status_lines():
    v1 = parse_imu_line("0.1,-0.2,9.8,0.01,0.02,-0.03")
    assert v1.has_gyro and v1.t_us is None and v1.seq is None
    assert parse_imu_line("INFO: Nano IMU stream v2 started (ax,ay,az,gx,gy,gz,t_us,seq)") is None
    assert parse_imu_line("ERROR: IMU initialization failed") is None
    assert parse_imu_line("0.1,0.2,9.8") is None
    assert parse_imu_line("0.1,0.2,9.8", allow_legacy_3axis=True).has_gyro is False
    assert parse_imu_line("0.1,-0.2,9.8,0.01,0.02,-0.03,-5,1") is None
    assert parse_imu_line("0.1,-0.2,9.8,0.01,0.02,-0.03,12.5,1") is None


# --- RPLIDAR protocol --------------------------------------------------------------
def encode_capsule(start_q6: int, samples, new_scan: bool = False) -> bytes:
    """Capsule bytes from 32 (distance_q2, offset_q3) samples (protocol layout)."""
    body = bytearray(((start_q6 & 0x7FFF) | (0x8000 if new_scan else 0)).to_bytes(2, "little"))
    for i in range(16):
        (d1, o1), (d2, o2) = samples[2 * i], samples[2 * i + 1]
        body += ((d1 & 0xFFFC) | ((o1 >> 4) & 0x3)).to_bytes(2, "little")
        body += ((d2 & 0xFFFC) | ((o2 >> 4) & 0x3)).to_bytes(2, "little")
        body.append((o1 & 0xF) | ((o2 & 0xF) << 4))
    checksum = 0
    for value in body:
        checksum ^= value
    return bytes((0xA0 | (checksum & 0xF), 0x50 | (checksum >> 4))) + bytes(body)


def test_command_packets():
    assert command_packet(0x25) == b"\xa5\x25"
    pwm = command_packet(CMD_SET_MOTOR_PWM, (660).to_bytes(2, "little"))
    assert pwm[:5] == b"\xa5\xf0\x02\x94\x02" and pwm[5] == 0xA5 ^ 0xF0 ^ 0x02 ^ 0x94 ^ 0x02
    assert command_packet(CMD_EXPRESS_SCAN, bytes(5)) == b"\xa5\x82\x05\x00\x00\x00\x00\x00" + bytes((0xA5 ^ 0x82 ^ 0x05,))


def test_capsule_round_trip_and_checksum():
    rng = np.random.default_rng(1)
    samples = [(int(rng.integers(0, 65536)), int(rng.integers(0, 64))) for _ in range(32)]
    raw = encode_capsule(1234, samples, new_scan=True)
    assert len(raw) == CAPSULE_BYTES
    cap = Capsule.from_bytes(raw, 7)
    assert cap.start_angle_q6 == 1234 and cap.new_scan and cap.host_ns == 7
    bad = bytearray(raw)
    bad[40] ^= 0x10
    assert Capsule.from_bytes(bytes(bad), 7) is None


def test_decode_matches_the_protocol_formula():
    """Angle of sample k = start + k * (next start - start) / 32 - offset / 8 deg (independent float formula)."""
    rng = np.random.default_rng(2)
    for _ in range(200):
        start = int(rng.integers(0, 360 * 64))
        step = int(rng.integers(1, 60 * 64))
        samples = [(int(rng.integers(0, 65536)), int(rng.integers(0, 64))) for _ in range(32)]
        prev = Capsule.from_bytes(encode_capsule(start, samples), 0)
        cur = Capsule.from_bytes(encode_capsule((start + step) % (360 * 64), samples), 0)
        angles, dists, _ = decode_capsule_pair(prev, cur)
        for k, (d, o) in enumerate(samples):
            expected = (start / 64.0 + k * step / 64.0 / 32.0 - o / 8.0) % 360.0
            diff = (angles[k] / 64.0 - expected + 180.0) % 360.0 - 180.0
            assert abs(diff) <= 1.0 / 64.0 + 1e-9
            assert dists[k] == d & 0xFFFC


def test_device_info_and_sensitivity_firmware():
    payload = bytes((0x28,)) + ((1 << 8) | 29).to_bytes(2, "little") + bytes((5,)) + bytes(range(16))
    info = DeviceInfo.parse(payload)
    assert info.firmware == "1.29" and info.supports_sensitivity and info.hardware == 5
    old = DeviceInfo.parse(bytes((0x28,)) + ((1 << 8) | 20).to_bytes(2, "little") + bytes(17))
    assert not old.supports_sensitivity


# --- synthetic A2M8 ------------------------------------------------------------------
class FakeA2M8:
    """Express stream of a LiDAR turning clockwise at ``rate_hz`` in a room.

    Laser frame (x forward, counter-clockwise): an object 0.5 m away on the
    left (+80..+100 deg), walls 3 m away elsewhere, no return behind the car.
    """

    def __init__(self, rate_hz=10.0, ts=250e-6, baud=115200, heading_offset=91.0, seed=0, t0=100.0):
        self.rate_hz, self.ts, self.baud, self.heading_offset = rate_hz, ts, baud, heading_offset
        self.rng = np.random.default_rng(seed)
        self.t0 = t0
        self.start_deg = 200.0  # device angle of sample 0

    def angle(self, n):
        return (self.start_deg + 360.0 * self.rate_hz * n * self.ts) % 360.0

    def distance_mm(self, device_deg):
        laser = (laser_angle_deg(device_deg, self.heading_offset) + 180.0) % 360.0 - 180.0
        if abs(laser) > 150.0:
            return 0.0
        return 500.0 if 80.0 <= laser <= 100.0 else 3000.0

    def sample_time(self, n):
        return self.t0 + n * self.ts

    def stream(self, n_capsules, corrupt=(), drop=()):
        """(bytes, host time at which each byte is available to a read, s)."""
        chunks, times = [], []
        bit_s = 10.0 / self.baud
        for c in range(n_capsules):
            if c in drop:
                continue
            samples = []
            for k in range(32):
                n = 32 * c + k
                samples.append((int(round(self.distance_mm(self.angle(n)) * 4)), 0))
            raw = bytearray(encode_capsule(int(round(self.angle(32 * c) * 64)) % (360 * 64), samples, new_scan=(c == 0)))
            if c in corrupt:
                raw[30] ^= 0x01
            # SDK timing model: the last byte leaves 1.5 Ts + transmission after the last sample.
            end = self.sample_time(32 * c + 31) + 1.5 * self.ts + CAPSULE_BYTES * bit_s
            chunks.append(bytes(raw))
            times.append(end - (CAPSULE_BYTES - 1 - np.arange(CAPSULE_BYTES)) * bit_s)
        data = b"".join(chunks)
        # USB: 0.2 ms minimum latency plus a random one; bytes stay in order.
        avail = np.concatenate(times) + 2e-4 + self.rng.exponential(3e-4, len(data))
        return data, np.maximum.accumulate(avail)


def feed_in_chunks(pipeline, stream, seed=0):
    """Read the port every 0.5-2 ms (sometimes 15 ms late); a read returns every byte available by then."""
    data, avail = stream
    rng = np.random.default_rng(seed)
    revs, pos, t = [], 0, avail[0]
    while pos < len(data):
        t += rng.uniform(5e-4, 2e-3) + (0.015 if rng.random() < 0.01 else 0.0)
        end = int(np.searchsorted(avail, t, side="right"))
        if end > pos:
            revs += pipeline.feed(data[pos:end], BASE_NS + int(round(t * 1e9)))
            pos = end
    return revs


def test_pipeline_publishes_complete_revolutions_stamped_at_their_first_sample():
    lidar = FakeA2M8()
    pipeline = ExpressScanPipeline(lidar.ts, lidar.baud)
    revs = feed_in_chunks(pipeline, lidar.stream(125 * 20))  # 20 s
    assert len(revs) >= 195
    assert pipeline.counters.dropped_incomplete == 0 and pipeline.counters.dropped_irregular == 0
    errors, periods = [], []
    for r in revs:
        rev = r.revolution
        assert rev.n_samples in (399, 400, 401)
        first_angle = lidar.angle(rev.first_index)
        assert first_angle > 359.0 or first_angle < 1.0  # starts at the LiDAR's 0 mark
        truth = lidar.sample_time(rev.first_index)
        errors.append((r.stamp_ns - BASE_NS) * 1e-9 - truth)
        periods.append(r.period_s)
    errors = np.asarray(errors[30:])  # after the clock envelope has settled
    # The residual is the minimum USB latency (0.2 ms here), calibrated on the car.
    assert np.all(np.abs(errors - 2e-4) < 3e-4)
    assert np.allclose(periods, 0.1, atol=3e-4)


def test_express_clock_recovers_after_sample_rate_change():
    """A changed sample-clock slope must not leave seconds of stale timing."""
    lidar = FakeA2M8(seed=19)
    # The reported sample period stays 250 us; the actual oscillator changes
    # by 0.5 % after 8 s, within the Express mapper's allowed drift. The
    # independent sample_time is the time truth, including across a lost packet.
    lidar.sample_time = lambda n: lidar.t0 + n * lidar.ts + 0.005 * max(0.0, n * lidar.ts - 8.0)
    pipeline = ExpressScanPipeline(lidar.ts, lidar.baud)
    revs = feed_in_chunks(pipeline, lidar.stream(125 * 18, corrupt={1000}), seed=19)
    settled = [r for r in revs if r.revolution.first_index * lidar.ts > 12.0]
    error = np.array([
        (r.stamp_ns - BASE_NS) * 1e-9 - lidar.sample_time(r.revolution.first_index)
        for r in settled
    ])
    assert len(settled) > 50
    assert pipeline.counters.dropped_incomplete == 1
    assert np.max(np.abs(error - 2e-4)) < 5e-4
    assert all(b.stamp_ns > a.stamp_ns for a, b in zip(revs, revs[1:]))


def test_scan_raw_layout_matches_the_simulation():
    lidar = FakeA2M8()
    pipeline = ExpressScanPipeline(lidar.ts, lidar.baud)
    rev = feed_in_chunks(pipeline, lidar.stream(125 * 3))[-1].revolution
    ranges = bin_revolution(rev.angles_deg, rev.dist_mm, heading_offset_deg=91.0)
    assert ranges.shape == (360,)
    # bin k is centred on -180 + k deg (laser frame, counter-clockwise, x forward)
    assert ranges[180 + 90] == pytest.approx(0.5)  # the object on the left
    assert ranges[180 + 0] == pytest.approx(3.0)  # ahead
    assert ranges[180 - 90] == pytest.approx(3.0)  # right
    assert np.isinf(ranges[0]) and ranges[0] > 0  # behind: no return
    assert np.isfinite(ranges).sum() >= 295  # 4 kHz at 10 Hz fills every 1-degree bin outside the blind sector


def test_bin_revolution_rep117():
    device = np.array([0.0, 1.0, 2.0, 2.2])
    dist = np.array([100.0, 0.0, 2000.0, 1500.0])  # 0.10 m (below range_min), no return, two returns in a bin
    ranges = bin_revolution(device, dist, heading_offset_deg=91.0)
    # device 0 deg -> laser +89 deg (the sync angle of the APEX_real profile)
    assert ranges[180 + 89] == -np.inf
    assert ranges[180 + 88] == np.inf
    assert ranges[180 + 87] == pytest.approx(1.5)  # nearest return kept


@pytest.mark.parametrize("legacy_clock", [False, True])
def test_lost_capsule_drops_only_its_revolution(legacy_clock):
    lidar = FakeA2M8(seed=4)
    mapper = DeviceClockMapper(max_drift=0.03) if legacy_clock else None
    pipeline = ExpressScanPipeline(lidar.ts, lidar.baud, mapper=mapper)
    revs = feed_in_chunks(pipeline, lidar.stream(125 * 20, corrupt={1000}), seed=4)
    assert pipeline.stream.checksum_errors == 1
    assert pipeline.counters.dropped_incomplete == 1
    assert len(revs) >= 194
    late = [r for r in revs if r.revolution.first_index > 32 * 1100]
    err = [(r.stamp_ns - BASE_NS) * 1e-9 - lidar.sample_time(r.revolution.first_index) for r in late]
    # A shorter envelope has fewer minimum-latency observations: permit
    # 0.5 ms absolute error, still below APEX_real's 1 ms timestamp jitter.
    # Retain the previous 0.3 ms accuracy bound for the injected 10 s mapper.
    accuracy_s = 3e-4 if legacy_clock else 5e-4
    assert np.all(np.abs(np.asarray(err) - 2e-4) < accuracy_s)


def test_missing_capsule_bytes_are_counted_and_the_clock_stays_right():
    lidar = FakeA2M8(seed=5)
    stream = lidar.stream(125 * 20, drop={1500})  # a whole capsule never arrives: no byte error, only the angle jumps
    pipeline = ExpressScanPipeline(lidar.ts, lidar.baud)
    revs = feed_in_chunks(pipeline, stream, seed=5)
    assert pipeline.decoder.lost_capsules == 1
    assert pipeline.counters.dropped_incomplete == 1
    late = [r for r in revs if r.revolution.first_index > 32 * 1600]
    err = [(r.stamp_ns - BASE_NS) * 1e-9 - lidar.sample_time(r.revolution.first_index) for r in late]
    assert late and np.all(np.abs(np.asarray(err) - 2e-4) < 3e-4)


def test_motor_regulator_reaches_the_target_rate():
    rng = np.random.default_rng(6)
    regulator = MotorRegulator(10.0, pwm=660)

    def plant(pwm):  # dead zone + gain: 13.03 Hz at PWM 660 as measured on the car
        return 13.03 * (pwm - 120) / (660 - 120)

    history = []
    for _ in range(400):
        rate = plant(regulator.pwm) * (1.0 + 0.01 * rng.standard_normal())
        regulator.update(1.0 / rate)
        history.append(plant(regulator.pwm))
    assert abs(history[-1] - 10.0) < 0.1
    # settled: no hunting in the last 200 revolutions
    assert np.ptp(history[-200:]) < 0.15


def test_motor_regulator_disabled_keeps_the_pwm():
    regulator = MotorRegulator(10.0, pwm=660, enabled=False)
    for _ in range(100):
        assert regulator.update(1.0 / 13.0) is None
    assert regulator.pwm == 660


# --- serial driver with a fake port ------------------------------------------------
class FakePort:
    """Answers the RPLIDAR requests the driver sends."""

    def __init__(self, *_, **__):
        self.written = []
        self.buffer = bytearray()
        self.dtr = True

    def _answer(self, kind, payload, mode=0):
        return b"\xa5\x5a" + (len(payload) | (mode << 30)).to_bytes(4, "little") + bytes((kind,)) + payload

    def write(self, data):
        self.written.append(bytes(data))
        cmd = data[1]
        if cmd == 0x50:
            self.buffer += self._answer(0x04, bytes((0x28,)) + ((1 << 8) | 29).to_bytes(2, "little") + bytes(17))
        elif cmd == 0x52:
            self.buffer += self._answer(0x06, bytes(3))
        elif cmd == 0x59:
            self.buffer += self._answer(0x15, (500).to_bytes(2, "little") + (250).to_bytes(2, "little"))
        elif cmd == 0xFF:
            self.buffer += self._answer(0xFF, (1).to_bytes(4, "little"))
        elif cmd == 0x82:
            self.buffer += b"\x00\x13" + b"\xa5\x5a" + (84 | (1 << 30)).to_bytes(4, "little") + b"\x82"

    def flush(self):
        pass

    def read(self, n):
        out, self.buffer = bytes(self.buffer[:n]), self.buffer[n:]
        return out

    @property
    def in_waiting(self):
        return len(self.buffer)

    def reset_input_buffer(self):
        self.buffer.clear()

    def close(self):
        pass


def test_driver_requests_with_a_fake_port():
    port = {}

    def factory(*args, **kwargs):
        port["p"] = FakePort()
        return port["p"]

    driver = RPLidarExpressDriver("/dev/null", serial_factory=factory, timeout_s=0.2)
    driver.open()
    assert driver.get_info().firmware == "1.29"
    assert driver.get_health() == (0, 0)
    assert driver.get_sample_rate() == (500, 250)
    assert driver.supports_motor_pwm()
    driver.start_motor(520)
    assert port["p"].dtr is False
    assert port["p"].written[-1] == command_packet(CMD_SET_MOTOR_PWM, (520).to_bytes(2, "little"))
    driver.start_express()
    assert port["p"].written[-1] == command_packet(CMD_EXPRESS_SCAN, bytes(5))
    driver.close()
    assert port["p"].dtr is True


def test_motor_pwm_query_without_answer_is_unknown_not_unsupported():
    class SilentAccBoard(FakePort):
        def write(self, data):
            if data[1] != 0xFF:
                super().write(data)

    driver = RPLidarExpressDriver("/dev/null", serial_factory=SilentAccBoard, timeout_s=0.1)
    driver.open()
    assert driver.supports_motor_pwm() is None
    driver.close()


def test_motor_ramp_climbs_linearly_to_the_target():
    ramp = motor_ramp(200, 520, ramp_s=3.0, step_s=0.1)
    assert len(ramp) == 30 and ramp[-1] == 520
    assert all(b > a for a, b in zip([200] + ramp, ramp))
    assert max(np.diff([200] + ramp)) <= 11
    assert motor_ramp(200, 520, ramp_s=0.0) == [520]
    assert motor_ramp(600, 520, ramp_s=3.0) == [520]  # never starts above the target


class FakeClockSupply:
    """An under-voltage flag file driven by a fake clock: up at the given times."""

    def __init__(self, tmp_path, low_at):
        self.t = 0.0
        self.low_at = low_at
        self.path = tmp_path / "in0_lcrit_alarm"
        self.update()

    def update(self):
        self.path.write_text("1\n" if any(a <= self.t < b for a, b in self.low_at) else "0\n")

    def sleep(self, s):
        self.t += s
        self.update()

    def guard(self, **kwargs):
        return SupplyGuard(str(self.path), sleep=self.sleep, clock=lambda: self.t, **kwargs)


def test_supply_guard_waits_for_a_clear_rail_before_the_motor_start(tmp_path):
    supply = FakeClockSupply(tmp_path, low_at=[(0.0, 2.0), (2.5, 3.0)])
    assert supply.guard(clear_s=1.0, wait_max_s=20.0).wait_clear()
    assert supply.t == pytest.approx(4.0, abs=0.06)  # clear from 3.0 s, for 1 s


def test_supply_guard_gives_up_on_a_supply_that_never_clears(tmp_path):
    supply = FakeClockSupply(tmp_path, low_at=[(0.0, 1e9)])
    assert not supply.guard(clear_s=1.0, wait_max_s=5.0).wait_clear()
    assert supply.t == pytest.approx(5.0, abs=0.06)


def test_supply_guard_holds_the_ramp_while_under_voltage(tmp_path):
    supply = FakeClockSupply(tmp_path, low_at=[(0.1, 0.6)])
    guard = supply.guard(hold_max_s=10.0)
    guard.ramp_wait(0.1)
    assert supply.t == pytest.approx(0.6, abs=0.06)
    guard.ramp_wait(0.1)
    assert supply.t == pytest.approx(0.7, abs=0.06)
    always = FakeClockSupply(tmp_path, low_at=[(0.0, 1e9)]).guard(hold_max_s=2.0)
    always.ramp_wait(0.1)  # bounded: a step never waits more than step + hold_max_s
    assert always._clock() == pytest.approx(2.1, abs=0.06)
    always.ramp_wait(0.1)
    assert always._clock() == pytest.approx(2.2, abs=0.06)  # budget is shared across steps


def test_supply_guard_total_ramp_hold_does_not_multiply_by_sixty_steps(tmp_path):
    supply = FakeClockSupply(tmp_path, low_at=[(0.0, 1e9)])
    guard = supply.guard(hold_max_s=10.0)
    for _ in range(60):
        guard.ramp_wait(0.1)
    assert supply.t == pytest.approx(16.0, abs=0.06)


@pytest.mark.parametrize("flags, expected", [("0x50000", False), ("0x50005", True)])
def test_supply_guard_uses_current_bit_from_fresh_host_status(tmp_path, flags, expected):
    from datetime import datetime, timezone
    supply = FakeClockSupply(tmp_path, low_at=[(0.0, 1e9)])  # latched hwmon alarm
    status = tmp_path / "power.csv"
    stamp = datetime.fromtimestamp(1000, timezone.utc).isoformat()
    status.write_text("utc_iso,lcrit_alarm,throttled,temp_c,freq_khz\n"
                      f"{stamp},1,{flags},52.0,1500000\n")
    guard = supply.guard(status_path=str(status), wall_clock=lambda: 1001.0)
    assert guard.undervoltage() is expected
    stale = supply.guard(status_path=str(status), wall_clock=lambda: 1004.0)
    assert stale.undervoltage() is True  # stale host data cannot clear the alarm


def test_supply_guard_incomplete_host_status_preserves_alarm(tmp_path):
    supply = FakeClockSupply(tmp_path, low_at=[(0.0, 1e9)])
    status = tmp_path / "power.csv"
    status.write_text("utc_iso,lcrit_alarm,throttled,temp_c,freq_khz\n2026-10-")
    assert supply.guard(status_path=str(status)).undervoltage() is True


def test_supply_guard_without_a_flag_is_a_plain_wait(tmp_path):
    slept = []
    guard = SupplyGuard(None, sleep=slept.append)
    assert guard.undervoltage() is None and guard.wait_clear()
    guard.ramp_wait(0.1)
    assert slept == [0.1]
    assert SupplyGuard(str(tmp_path / "missing")).undervoltage() is None


def test_find_undervoltage_alarm_picks_the_rpi_volt_hwmon(tmp_path):
    (tmp_path / "hwmon0").mkdir()
    (tmp_path / "hwmon0" / "name").write_text("cpu_thermal\n")
    (tmp_path / "hwmon2").mkdir()
    (tmp_path / "hwmon2" / "name").write_text("rpi_volt\n")
    (tmp_path / "hwmon2" / "in0_lcrit_alarm").write_text("0\n")
    assert find_undervoltage_alarm(str(tmp_path / "hwmon*")) == str(tmp_path / "hwmon2" / "in0_lcrit_alarm")
    assert find_undervoltage_alarm(str(tmp_path / "none*")) is None


def test_driver_soft_start_sets_the_low_pwm_before_enabling_the_motor():
    port = {}

    def factory(*args, **kwargs):
        port["p"] = FakePort()
        return port["p"]

    waits = []
    driver = RPLidarExpressDriver("/dev/null", serial_factory=factory, timeout_s=0.2)
    driver.open()
    driver.start_motor(520, start_pwm=200, ramp_s=1.0, step_s=0.1, wait=waits.append)
    pwms = [
        int.from_bytes(w[3:5], "little")
        for w in port["p"].written
        if w[:2] == command_packet(CMD_SET_MOTOR_PWM, bytes(2))[:2]
    ]
    assert pwms[0] == 200 and pwms[-1] == 520
    assert all(b >= a for a, b in zip(pwms, pwms[1:]))
    assert waits == [0.1] * 10
    assert port["p"].dtr is False
    driver.close()
