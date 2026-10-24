# Longitudinal-only follow-up from run 211216

This experimental `tracking` profile calibrates the forward bridge using the
latest completed physical run. It preserves the approved steering estimation,
geometry, lane margins, reverse conversion and geometric recovery selection.
The former initialized profile remains available separately.

Six moving-segment medians give a median onboard-speed / previous-input ratio
of 3.811, rounded to **3.8**. The moving intercept is fixed at zero because these
closed-loop observations cannot independently establish both coefficients.
[calibration.json](calibration.json) retains all points, including slow segments,
selection and source hashes. The scatter is large; this is an initialization
for bounded feedback, not precise motor metrology.

Forward conversion is `PWM = 15 + (33 / 3.8) * input`. Actual forward references
are **0.36–0.42 m/s**, chosen near the observed moving pace. Proportional feedback
and its authority use the corresponding command units, with the previously
delay-tested gain 0.17. Launch/deadband command units are converted as well.
No measured velocity is scaled or time-shifted. Kalman starts at unit gain,
zero offset and zero confidence; its acceptance thresholds remain active.
Forward-only identification prevents reverse samples from changing the forward
inverse. Positive-direction overspeed can supply learning data.

Forward PWM remains at most 28.2, reverse at most 31.2. Two laps, 40 m estimated
travel and 300 s remain the finite mission limits. Recording is to RAM; copies
are made after stop. The runner verifies active profile parameters before arm.

```bash
./real_vehicle/tools/pc/race_pi.sh --profile tracking --dry-run run
./real_vehicle/tools/pc/race_pi.sh --profile tracking run
```

This is post-deployment hardware calibration with unchanged odometry weights.
It must not be labeled untouched zero-shot controller transfer. Physical
performance of this new profile is pending a fresh captured run. The approved
005311 acquisition and successful 211216 evidence are retained unchanged.
