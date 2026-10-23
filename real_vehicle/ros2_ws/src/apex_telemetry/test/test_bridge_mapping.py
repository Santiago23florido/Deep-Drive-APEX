"""The command chain between the race driver and the wheels.

Three things are checked here, all without hardware:

* the affine speed map and the bicycle inverse, as arithmetic;
* the servo, whose duty window is asymmetric about its mechanical centre -- the
  reason a 48 deg request used to be delivered as 23 deg and reported as 23 deg;
* the bridge node itself, driven tick by tick with a fake PWM and a fake clock,
  because what matters is the *wiring*: that the affine profile really has no
  ramp, that the steering really is clamped and held, and that a direction
  change really does not sleep inside the control callback.

Follows the pattern of ``test_sensor_timing.py``: synthetic hardware, real code.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apex_telemetry.actuation import actuation as actuation_mod  # noqa: E402
from apex_telemetry.actuation.speed_map import SpeedMap, steer_deg_from_twist  # noqa: E402

# The shipped servo geometry (config/apex_params.yaml).
SERVO = {"dc_min": 5.0, "dc_max": 10.0, "center_trim_dc": 0.7, "direction_sign": -1.0,
         "limit_deg": 18.0, "min_authority_ratio": 0.0}
MOTOR = {"dc_min": 5.0, "dc_max": 10.0, "neutral_dc": 7.5, "reverse_brake_dc": 6.9,
         "reverse_brake_hold_s": 0.12, "reverse_neutral_hold_s": 0.12, "reverse_exit_hold_s": 0.15}
MAP_B = SpeedMap(esc_zero_pct=15.0, pct_per_mps=33.0, max_pct=45.0,
                 esc_zero_rev_pct=18.0, rev_pct_per_mps=33.0, max_rev_pct=33.0)


class FakeLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def _log(self, msg, *a, **kw):  # noqa: ANN001, ANN002, ANN003
        self.messages.append(str(msg))

    debug = info = warning = warn = error = _log


class FakePWM:
    """Records duty cycles instead of writing /sys/class/pwm."""

    instances: list["FakePWM"] = []

    def __init__(self, channel, frequency_hz, logger, *, backend="sysfs_pwm", node=None, publish_topic=""):  # noqa: ANN001
        self.channel, self.frequency_hz = int(channel), float(frequency_hz)
        self.duty: float | None = None
        self.history: list[float] = []
        self.enabled = False
        FakePWM.instances.append(self)

    def start(self, duty_cycle_pct: float) -> None:
        self.enabled = True
        self.set_duty_cycle(duty_cycle_pct)

    def set_duty_cycle(self, duty_cycle_pct: float) -> None:
        self.duty = float(duty_cycle_pct)
        self.history.append(self.duty)

    def disable(self) -> None:
        self.enabled = False

    def stop(self) -> None:
        self.disable()

    def get_state(self) -> dict:
        return {"backend": "fake", "duty_cycle_pct": self.duty, "enabled": self.enabled}


@pytest.fixture
def fake_pwm(monkeypatch):
    FakePWM.instances = []
    monkeypatch.setattr(actuation_mod, "HardwarePWM", FakePWM)
    return FakePWM


# ---------------------------------------------------------------- the speed map
def test_a_zero_command_is_a_zero_and_not_a_creep():
    assert MAP_B.pct(0.0) == 0.0
    assert MAP_B.pct(0.001) == 0.0
    assert MAP_B.pct(-0.001) == 0.0


def test_the_map_is_exactly_affine_and_monotone():
    previous = 0.0
    for i in range(1, 101):
        u = MAP_B.full_scale_mps() * i / 100.0
        pct = MAP_B.pct(u)
        assert pct == pytest.approx(15.0 + 33.0 * u, abs=1e-9)
        assert pct > previous
        previous = pct


def test_the_ceiling_is_a_hard_saturation():
    """The shaper caps its own output at 2.0 + 1.5*|v_des| m/s, far above the
    envelope, so the bridge must have a ceiling of its own."""
    assert MAP_B.pct(100.0) == 45.0
    assert MAP_B.pct(MAP_B.full_scale_mps()) == pytest.approx(45.0)
    assert MAP_B.pct(-100.0) == -33.0


def test_the_reverse_branch_has_its_own_gauge_and_ceiling():
    assert MAP_B.pct(-0.1) == pytest.approx(-(18.0 + 3.3))
    assert MAP_B.full_scale_reverse_mps() == pytest.approx((33.0 - 18.0) / 33.0)


def test_the_inverse_recovers_the_command():
    for u in (0.05, 0.2, 0.35, 0.7, 0.9, -0.1, -0.3):
        pct = MAP_B.pct(u)
        if abs(pct) in (45.0, 33.0):
            continue
        assert MAP_B.u_for_pct(pct) == pytest.approx(u, abs=1e-9)


def test_the_residual_deadband_is_a_creep_not_a_lurch():
    """The gauge offset sits below the believed ESC threshold on purpose."""
    assert MAP_B.residual_deadband_mps(17.0) == pytest.approx(2.0 / 33.0, abs=1e-9)
    assert MAP_B.residual_deadband_mps(14.0) == 0.0  # offset above the threshold: no deadband left


def test_a_map_without_room_above_its_offset_is_refused():
    with pytest.raises(ValueError):
        SpeedMap(esc_zero_pct=45.0, max_pct=45.0)
    with pytest.raises(ValueError):
        SpeedMap(pct_per_mps=0.0)


# ------------------------------------------------------------ the twist inverse
def test_the_curvature_round_trip_is_exact_and_the_wheelbase_cancels():
    """What the driver publishes is ``v * tan(steer) / L``; this must return
    exactly ``steer``, for any wheelbase, forwards and backwards. The adaptive
    layer's identified gain means the same thing on both sides only because of
    this identity."""
    for wheelbase in (0.25, 0.30, 0.46):
        for v in (0.05, 0.35, 0.7, 0.9, -0.3, -0.4):
            for deg in (-12.5, -7.0, -0.3, 0.0, 0.3, 7.0, 12.5):
                steer = math.radians(deg)
                yaw_rate = v * math.tan(steer) / wheelbase
                got, saturated = steer_deg_from_twist(v, yaw_rate, wheelbase, 12.5)
                assert got == pytest.approx(deg, abs=1e-9), (wheelbase, v, deg)
                assert not saturated


def test_a_request_beyond_the_limit_is_clamped_and_flagged():
    """The regression for the real-car defect: 46-48 deg requests against an
    18 deg mechanical limit, silently saturated by the duty bounds."""
    v, wheelbase = 0.5, 0.30
    for deg in (18.0, 46.0, 48.0, -48.0):
        yaw_rate = v * math.tan(math.radians(deg)) / wheelbase
        got, saturated = steer_deg_from_twist(v, yaw_rate, wheelbase, 12.5)
        assert abs(got) == pytest.approx(12.5)
        assert math.copysign(1.0, got) == math.copysign(1.0, deg)
        assert saturated


def test_below_the_decode_speed_there_is_no_steering_information():
    got, saturated = steer_deg_from_twist(0.0, 0.0, 0.30, 12.5, v_min_mps=0.02)
    assert got is None and not saturated
    got, _ = steer_deg_from_twist(0.019, 5.0, 0.30, 12.5, v_min_mps=0.02)
    assert got is None


# ------------------------------------------------------------------- the servo
def test_the_servo_authority_is_the_symmetric_angle(fake_pwm):
    servo = actuation_mod.SteeringServo(channel=1, frequency_hz=50.0, logger=FakeLogger(), **SERVO)
    # centre 0.5*(5+10)+0.7 = 8.2 %, scale 2.5/18 = 0.13889 %/deg
    # -> up 1.8 % = 12.96 deg, down 3.2 % = 23.04 deg; the symmetric one wins.
    assert servo.authority_deg == pytest.approx(1.8 / (2.5 / 18.0), abs=1e-9)
    assert servo.authority_deg == pytest.approx(12.96, abs=0.01)


@pytest.mark.parametrize('sign', [-1.0, 1.0])
def test_asymmetric_servo_uses_existing_window_without_changing_centre(fake_pwm, sign):
    servo = actuation_mod.SteeringServo(channel=1, frequency_hz=50.0, logger=FakeLogger(),
                                       **{**SERVO, 'direction_sign': sign}, asymmetric_limits=True)
    expected_left, expected_right = (23.04, 12.96) if sign < 0 else (12.96, 23.04)
    assert servo.get_state()['authority_left_deg'] == pytest.approx(expected_left)
    assert servo.get_state()['authority_right_deg'] == pytest.approx(expected_right)
    for request, expected in ((100.0, expected_left), (-100.0, -expected_right), (0.0, 0.0)):
        servo.set_angle_deg(request)
        state = servo.get_state()
        assert state['applied_deg'] == pytest.approx(expected)
        assert 5.0 <= state['pwm_dc'] <= 10.0
    assert servo.get_state()['pwm_dc'] == pytest.approx(8.2)


def test_directional_twist_round_trip_preserves_left_authority_and_right_limit():
    for v in (.2, -.2):
        for request in (-45.0, -12.5, 0.0, 12.5, 22.5, 45.0):
            angle, saturated = steer_deg_from_twist(
                v, v * math.tan(math.radians(request)) / .3, .3, 12.5,
                max_left_deg=22.5, max_right_deg=12.5)
            expected = max(-12.5, min(22.5, request))
            assert angle == pytest.approx(expected)
            assert saturated == (request != expected)


def test_the_servo_never_reports_an_angle_it_did_not_take(fake_pwm):
    servo = actuation_mod.SteeringServo(channel=1, frequency_hz=50.0, logger=FakeLogger(), **SERVO)
    for deg in (-48.0, -23.0, -18.0, -12.5, 0.0, 12.5, 18.0, 46.0, 48.0):
        servo.set_angle_deg(deg)
        state = servo.get_state()
        assert abs(state["applied_deg"]) <= servo.authority_deg + 1e-9, (deg, state)
        assert state["requested_deg"] == deg
        # and the duty stays inside the window
        assert SERVO["dc_min"] - 1e-9 <= state["pwm_dc"] <= SERVO["dc_max"] + 1e-9


def test_the_servo_is_symmetric_within_its_authority(fake_pwm):
    servo = actuation_mod.SteeringServo(channel=1, frequency_hz=50.0, logger=FakeLogger(), **SERVO)
    for deg in (0.0, 2.0, 6.0, 12.0, 12.5):
        servo.set_angle_deg(deg)
        plus = servo.get_state()["applied_deg"]
        servo.set_angle_deg(-deg)
        minus = servo.get_state()["applied_deg"]
        assert plus == pytest.approx(-minus, abs=1e-9), deg
        assert plus == pytest.approx(deg, abs=1e-9)


def test_centre_is_the_mechanical_centre(fake_pwm):
    servo = actuation_mod.SteeringServo(channel=1, frequency_hz=50.0, logger=FakeLogger(), **SERVO)
    servo.center()
    assert servo.get_state()["pwm_dc"] == pytest.approx(8.2, abs=1e-9)
    assert servo.get_state()["applied_deg"] == pytest.approx(0.0, abs=1e-9)


def test_the_driver_limit_is_inside_the_measured_authority(fake_pwm):
    """``vehicle_real.yaml``'s steering limit must be reachable both ways."""
    import yaml  # noqa: PLC0415

    vehicle = yaml.safe_load((Path(__file__).resolve().parents[2] / "apex_race" / "config" / "vehicle_real.yaml").read_text())
    servo = actuation_mod.SteeringServo(channel=1, frequency_hz=50.0, logger=FakeLogger(), **SERVO)
    assert float(vehicle["model"]["steering_limit_deg"]) <= servo.authority_deg


# -------------------------------------------------------- the reverse sequence
class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_the_direction_change_never_sleeps(fake_pwm, monkeypatch):
    """`_enter_reverse_mode` used to sleep 0.24 s inside the control callback,
    which freezes the steering on every recovery reversal."""
    monkeypatch.setattr(actuation_mod.time, "sleep", lambda *_: pytest.fail("the control callback slept"))
    clock = Clock()
    motor = actuation_mod.MaverickESCMotor(channel=0, frequency_hz=50.0, logger=FakeLogger(),
                                           async_reverse=True, clock=clock, **MOTOR)
    pwm = FakePWM.instances[-1]

    motor.set_speed_pct(25.0)
    assert motor.get_state()["reverse_phase"] == "fwd"
    assert pwm.duty > MOTOR["neutral_dc"]

    # Request reverse: brake first, and the ESC must not be given reverse yet.
    motor.set_speed_pct(-25.0)
    assert motor.get_state()["reverse_phase"] == "brake"
    assert pwm.duty == pytest.approx(MOTOR["reverse_brake_dc"])

    for _ in range(6):  # 6 control ticks at 50 Hz = 0.12 s
        clock.advance(0.02)
        motor.set_speed_pct(-25.0)
    assert motor.get_state()["reverse_phase"] == "neutral"
    assert pwm.duty == pytest.approx(MOTOR["neutral_dc"])

    for _ in range(7):
        clock.advance(0.02)
        motor.set_speed_pct(-25.0)
    state = motor.get_state()
    assert state["reverse_phase"] == "rev" and state["reverse_mode"]
    assert pwm.duty < MOTOR["neutral_dc"]
    assert clock.t == pytest.approx(0.26, abs=1e-9)  # >= brake 0.12 + neutral 0.12

    # ... and back to forward, through the exit hold.
    motor.set_speed_pct(20.0)
    assert motor.get_state()["reverse_phase"] == "exit"
    assert pwm.duty == pytest.approx(MOTOR["neutral_dc"])
    for _ in range(8):
        clock.advance(0.02)
        motor.set_speed_pct(20.0)
    assert motor.get_state()["reverse_phase"] == "fwd"
    assert pwm.duty > MOTOR["neutral_dc"]


def test_the_blocking_sequence_is_still_the_default(fake_pwm, monkeypatch):
    """The legacy stack keeps the behaviour it was validated with."""
    slept: list[float] = []
    monkeypatch.setattr(actuation_mod.time, "sleep", lambda s: slept.append(s))
    motor = actuation_mod.MaverickESCMotor(channel=0, frequency_hz=50.0, logger=FakeLogger(), **MOTOR)
    motor.set_speed_pct(-25.0)
    assert motor.get_state()["reverse_mode"] is True
    assert sum(slept) == pytest.approx(MOTOR["reverse_brake_hold_s"] + MOTOR["reverse_neutral_hold_s"])


def test_the_esc_deadband_maps_small_percents_to_neutral(fake_pwm):
    motor = actuation_mod.MaverickESCMotor(channel=0, frequency_hz=50.0, logger=FakeLogger(), **MOTOR)
    pwm = FakePWM.instances[-1]
    motor.set_speed_pct(0.5)
    assert pwm.duty == pytest.approx(MOTOR["neutral_dc"])
    motor.set_speed_pct(45.0)
    assert pwm.duty == pytest.approx(7.5 + 0.45 * 2.5)


# ------------------------------------------------------------------- the node
@pytest.fixture
def bridge(fake_pwm, request):
    """The real bridge node in the affine profile, with no ROS graph needed."""
    rclpy = pytest.importorskip("rclpy")
    from apex_telemetry.actuation import cmd_vel_to_apex_actuation_node as mod  # noqa: PLC0415

    overrides = dict(getattr(request, "param", {}) or {})
    params = {"actuation_profile": "affine", "esc_zero_pct": 15.0, "speed_pct_per_mps": 33.0,
              "max_speed_pct": 45.0, "esc_zero_rev_pct": 18.0, "reverse_pct_per_mps": 33.0,
              "max_reverse_speed_pct": 33.0, "max_steering_deg": 12.5, "steering_limit_deg": 18.0,
              "steering_dc_max": 10.0, "steering_center_trim_dc": 0.7, "steering_min_authority_ratio": 0.0,
              "wheelbase_m": 0.30, "decode_v_min_mps": 0.02, "control_rate_hz": 50.0,
              "steering_rate_limit_deg_per_s": 65.0, "max_linear_speed_mps": 0.95}
    params.update(overrides)
    args = ["--ros-args"]
    for key, value in params.items():
        args += ["-p", f"{key}:={value if not isinstance(value, bool) else str(value).lower()}"]
    rclpy.init(args=args)
    node = mod.CmdVelToApexActuationNode()
    try:
        yield node
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _command(node, v: float, steer_deg: float) -> None:
    from geometry_msgs.msg import Twist  # noqa: PLC0415

    msg = Twist()
    msg.linear.x = float(v)
    msg.angular.z = float(v * math.tan(math.radians(steer_deg)) / 0.30)
    node._cmd_cb(msg)


def _status(node) -> dict:
    """The last published status, as the pre-flight reads it."""
    payloads: list[str] = []
    node._status_pub.publish = lambda msg: payloads.append(msg.data)  # noqa: ARG005
    node._publish_state(timed_out=False)
    return json.loads(payloads[-1])


def test_the_node_reports_the_profile_it_is_running(bridge):
    status = _status(bridge)
    assert status["actuation_profile"] == "affine"
    assert status["speed_map"]["esc_zero_pct"] == 15.0
    assert status["speed_map"]["pct_per_mps"] == 33.0
    assert status["speed_map"]["max_pct"] == 45.0
    assert status["max_steering_deg"] == 12.5
    assert status["steering_authority_deg"] == pytest.approx(12.96, abs=0.01)
    assert status["speed_map"]["full_scale_mps"] == pytest.approx(0.909, abs=0.001)


def test_the_affine_profile_has_no_ramp_and_no_launch_boost(bridge):
    """One tick must deliver the commanded percent, not a fraction of it."""
    _command(bridge, 0.70, 0.0)
    bridge._control_step()
    assert bridge._last_applied_speed_pct == pytest.approx(15.0 + 33.0 * 0.70, abs=1e-9)
    status = _status(bridge)
    assert not status["speed_ramp_active"]
    assert not status["launch_boost_active"]
    # ... and a release goes straight to zero, with no ramp-down.
    _command(bridge, 0.0, 0.0)
    bridge._control_step()
    assert bridge._last_applied_speed_pct == 0.0


def test_the_node_clamps_the_computed_steering(bridge):
    """A 48 deg request (the logged defect) must leave as 12.5 deg."""
    _command(bridge, 0.50, 48.0)
    for _ in range(40):  # let the 65 deg/s rate limit catch up
        bridge._control_step()
    assert bridge._last_desired_steering_deg == pytest.approx(12.5, abs=1e-9)
    assert bridge._last_applied_steering_deg == pytest.approx(12.5, abs=0.01)


@pytest.mark.parametrize('bridge', [
    {'max_steering_left_deg': 22.5, 'max_steering_right_deg': 12.5},
    {'max_steering_left_deg': 22.5, 'max_steering_right_deg': 12.96},
], indirect=True)
def test_node_delivers_directional_limits_and_holds_them_at_zero_speed(bridge):
    for request, expected in ((40, 22.5), (-40, -bridge._max_steering_right_deg)):
        _command(bridge, .2, request)
        for _ in range(60):
            bridge._control_step()
        status = _status(bridge)
        assert status['applied_steering_deg'] == pytest.approx(expected)
        assert status['steering_servo_state']['asymmetric_limits']
        assert 5 <= status['steering_servo_state']['pwm_dc'] <= 10
        _command(bridge, 0, 0)
        bridge._control_step()
        assert _status(bridge)['applied_steering_deg'] == pytest.approx(expected)
        assert bridge._last_applied_speed_pct == 0


def test_the_servo_holds_its_angle_at_a_stop(bridge):
    _command(bridge, 0.50, 10.0)
    for _ in range(40):
        bridge._control_step()
    held = bridge._last_applied_steering_deg
    assert held == pytest.approx(10.0, abs=0.05)
    _command(bridge, 0.0, 0.0)  # a stop carries no steering information
    for _ in range(10):
        bridge._control_step()
    assert bridge._last_applied_steering_deg == pytest.approx(held, abs=1e-6)
    assert _status(bridge)["steering_held"]


def test_the_watchdog_still_cuts_to_neutral(bridge):
    _command(bridge, 0.70, 0.0)
    bridge._control_step()
    assert bridge._last_applied_speed_pct > 0.0
    bridge._last_cmd_monotonic -= 10.0  # no command for 10 s
    bridge._control_step()
    assert bridge._last_applied_speed_pct == 0.0
    assert _status(bridge)["timed_out"] is False  # _status forces the flag; the step set it
    assert bridge._motor.get_state()["pwm_dc"] == pytest.approx(7.5)


def test_the_command_gap_is_measured(bridge):
    _command(bridge, 0.5, 0.0)
    bridge._last_cmd_monotonic -= 0.2
    _command(bridge, 0.5, 0.0)
    assert bridge._cmd_gap_ms_max >= 200.0
    assert bridge._cmd_seq == 2


def test_command_reception_continues_during_blocked_actuator_status(bridge):
    """Reproduce the recorded bridge-dispatch stall without PWM hardware."""
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.executors import MultiThreadedExecutor

    source = rclpy.create_node('bridge_dispatch_test_source')
    pub = source.create_publisher(Twist, bridge._cmd_vel_topic, 20)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(bridge)
    blocked, release, block_next, stop = (threading.Event() for _ in range(4))
    original = bridge._status_pub.publish

    def delayed_status(message):
        if block_next.is_set():
            block_next.clear()
            blocked.set()
            release.wait(timeout=2.)
        original(message)

    def send_commands():
        message = Twist()
        message.linear.x = .2
        while not stop.is_set():
            pub.publish(message)
            stop.wait(.01)

    def await_condition(predicate, seconds=3.):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(.005)
        return False

    bridge._status_pub.publish = delayed_status
    receiver = threading.Thread(target=executor.spin)
    sender = threading.Thread(target=send_commands)
    receiver.start()
    sender.start()
    try:
        assert await_condition(lambda: bridge._cmd_seq >= 5)
        block_next.set()
        assert blocked.wait(timeout=1.)
        seq = bridge._cmd_seq
        assert await_condition(lambda: bridge._cmd_seq >= seq + 8, seconds=.5)
        assert not release.is_set()  # actuator/status callback still blocked
        release.set()
        stop.set()
        sender.join(timeout=1.)
        # A real command loss must still trigger the unchanged watchdog.
        assert await_condition(lambda: bridge._last_applied_speed_pct == 0., seconds=2.)
        assert bridge._motor.get_state()['pwm_dc'] == pytest.approx(7.5)
    finally:
        stop.set()
        release.set()
        sender.join(timeout=1.)
        executor.shutdown(timeout_sec=3.)
        receiver.join(timeout=3.)
        source.destroy_node()


@pytest.mark.parametrize("bridge", [{"actuation_profile": "legacy"}], indirect=True)
def test_the_legacy_profile_keeps_its_step_and_its_ramp(bridge):
    """Regression: the default profile is untouched."""
    status = _status(bridge)
    assert status["actuation_profile"] == "legacy"
    assert status["speed_map"] is None
    _command(bridge, 0.95, 0.0)
    bridge._control_step()
    # the legacy map is a ratio of max_linear_speed_mps between the two percents
    assert bridge._last_desired_speed_pct == pytest.approx(45.0, abs=1e-6)
    assert bridge._motor.get_state()["async_reverse"] is False
