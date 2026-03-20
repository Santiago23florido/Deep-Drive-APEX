import math
import unittest
from unittest import mock

from apex_telemetry import actuation


class _DummyLogger:
    def debug(self, *_args, **_kwargs):
        pass

    def info(self, *_args, **_kwargs):
        pass


class _DummyPWM:
    def __init__(self, channel, frequency_hz, logger):
        self.channel = channel
        self.frequency_hz = frequency_hz
        self.logger = logger
        self.calls = []

    def start(self, duty_cycle_pct):
        self.calls.append(("start", float(duty_cycle_pct)))

    def set_duty_cycle(self, duty_cycle_pct):
        self.calls.append(("set", float(duty_cycle_pct)))

    def stop(self):
        self.calls.append(("stop", None))


class SteeringServoTests(unittest.TestCase):
    def test_legacy_center_offset_preserves_configured_pwm_range(self):
        with mock.patch.object(actuation, "HardwarePWM", _DummyPWM):
            servo = actuation.SteeringServo(
                channel=1,
                frequency_hz=50.0,
                limit_deg=18.0,
                dc_min=5.0,
                dc_max=8.6,
                center_trim_dc=0.0,
                logger=_DummyLogger(),
            )

            self.assertAlmostEqual(servo._pwm.calls[0][1], 6.47, places=2)

            servo.set_angle_deg(0.0)
            self.assertAlmostEqual(servo._pwm.calls[-1][1], 6.47, places=2)

            servo.set_angle_deg(18.0)
            self.assertAlmostEqual(servo._pwm.calls[-1][1], 8.6, places=3)

            servo.set_angle_deg(-18.0)
            self.assertAlmostEqual(servo._pwm.calls[-1][1], 5.0, places=3)

            servo.set_angle_deg(180.0)
            self.assertAlmostEqual(servo._pwm.calls[-1][1], 8.6, places=3)

            state = servo.get_state()
            self.assertAlmostEqual(state["dc_center"], 6.47, places=2)
            self.assertAlmostEqual(state["legacy_center_offset_dc"], -0.33, places=2)


class MaverickMotorTests(unittest.TestCase):
    def test_reverse_sequence_matches_expected_brake_neutral_drive_pattern(self):
        sleep_calls = []

        with mock.patch.object(actuation, "HardwarePWM", _DummyPWM), mock.patch.object(
            actuation.time,
            "sleep",
            side_effect=lambda duration: sleep_calls.append(float(duration)),
        ):
            motor = actuation.MaverickESCMotor(
                channel=0,
                frequency_hz=50.0,
                dc_min=5.0,
                dc_max=10.0,
                neutral_dc=7.5,
                reverse_brake_dc=7.0,
                reverse_brake_hold_s=0.03,
                reverse_neutral_hold_s=0.03,
                reverse_exit_hold_s=0.10,
                logger=_DummyLogger(),
            )

            motor.set_speed_pct(-30.0)
            self.assertEqual(
                motor._pwm.calls,
                [
                    ("start", 7.5),
                    ("set", 7.0),
                    ("set", 7.5),
                    ("set", 6.75),
                ],
            )
            self.assertEqual([round(value, 2) for value in sleep_calls], [0.03, 0.03])

            motor.set_speed_pct(20.0)
            self.assertEqual(
                motor._pwm.calls[-2:],
                [
                    ("set", 7.5),
                    ("set", 8.0),
                ],
            )
            self.assertEqual([round(value, 2) for value in sleep_calls], [0.03, 0.03, 0.10])
            self.assertTrue(math.isclose(motor.get_state()["pwm_dc"], 8.0, rel_tol=0.0, abs_tol=1e-6))


if __name__ == "__main__":
    unittest.main()
