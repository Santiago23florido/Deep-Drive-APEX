# Small forward trim and turn-anticipation follow-up

This experiment follows captured run `20261024_214635`, whose selected lap-2
speed MAE was 0.106 m/s at a 0.360 m/s reference and which made two reverse
recoveries on that lap. The new profile retains the reference and measured
velocity values. It adds 0.7 points to the forward bridge intercept (15 to 15.7)
and lowers the command cap from 1.52 to 1.43; maximum requested PWM is 28.118,
below the existing 28.2 ceiling. Reverse conversion and limits are unchanged.

The second change reduces `reactive.w_smooth` from 0.10 to 0.05, allowing the
forward candidate search to initiate a turn sooner. Geometry, collision/lane
constraints, servo endpoints and geometric reverse recovery remain. The
recorded-entry replay selected a stronger feasible right turn near 30–31 s;
at the already blocked 46–47 s poses every tested variant still stopped. This
is a static recorded-input comparison, not proof of avoiding reverses in a
new closed-loop run. All variants are retained in
[curve_entry_replay.json](curve_entry_replay.json).

The former global candidate exceeded curvature and clearance constraints,
so lap 2 used MAP_GUIDED local planning rather than direct optimized-path
tracking. SLAM availability alone does not make that forward path feasible.
The fallback may still require a reverse when no forward continuation meets
the vehicle footprint and steering constraints.

The same Kalman confidence threshold remains; no learned coefficients are
forced into use. Neural weights and shared controller core code are unchanged.
The next finite two-lap trial records raw sensors, actual references, commands,
model confidence and timing to RAM for comparison after stop. Hardware
performance of this refinement is pending. The earlier capture and approved
baseline remain available separately.
