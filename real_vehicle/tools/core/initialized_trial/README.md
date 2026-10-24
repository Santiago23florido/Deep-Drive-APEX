# Opt-in calibrated forward trial

`run_oval.py` retains the finite two-lap mission, fresh-start checks, arm and
power guards, recording and stop behavior of the canonical acquisition helper.
The name is historical; it contains no rules specific to an oval track.

The calibrated forward bridge uses `15.42 + 5.24 * input`, with the existing
28.2 PWM ceiling. Reverse retains `18 + 33 * abs(input)` and the 31.2 ceiling.
The profile requests approximately 0.45–0.55 m/s forward and logs the actual
reference separately from the actuator input. Network weights are unchanged.
The approved helper directory is not overwritten.

Use the [PC entry point](../../pc/race_pi.sh), which installs this helper beside
the remaining canonical helpers inside the trial's RAM directory:

```bash
./real_vehicle/tools/pc/race_pi.sh --profile initialized run
```

This command moves the vehicle after its disarmed preflight passes. It syncs an
explicit source/configuration payload and verifies hashes first. Do not execute
this file in isolation from its companion acquisition helpers.

The [calibration report](../../../docs/calibration/speed_initialization_20261024/README.md)
contains the fit, replay comparison, simulation results and known failures.
Physical validation of this profile is pending; the approved profile remains
the default.
