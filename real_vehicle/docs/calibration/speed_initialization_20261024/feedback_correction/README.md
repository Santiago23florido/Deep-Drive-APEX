# Feedback conversion and failed follow-up trials

The changes on `fix/forward-feedback-unit-conversion` are **experimental**.
They have not demonstrated successful physical navigation. Approved acquisition
`20261024_005311` and its sealed evidence remain unchanged.

## Changes tested

- Commit `c933409` converts forward P gain and its authority floor by 6.3.
  Full original P gain oscillated in the 300 ms delay/high-gain motor case;
  the selected forward-only pre-conversion gain is 0.17. For a 0.20 m/s error,
  correction is 1.1224 PWM points instead of 0.262. Maximum corrective authority
  near 0.5 m/s is 1.9807 PWM points. Reverse P/launch and PWM ceilings remain.
- Commit `b6cb873` keeps nominal steering gains/trim/saturation until independent
  steering confidence reaches 0.5. Raw steering estimates remain logged.
  Both switches default to neutral behavior in the approved profile.
- Final regression suites: 300 real/telemetry/PC tests and 204 simulator research
  tests passed. Delay/load motor tests include 160/300 ms delays and gains
  0.55/0.85/1.0/1.15 with additional sag. These are development tests, not proof
  of physical track performance. No neural weights were changed.

## Physical outcomes

| Run | Profile/source | Outcome |
| --- | --- | --- |
| 20261024_193020 | Initialized, c933409 | Did not pass first curve; operator reported reverse in an unhelpful direction; stopped |
| 20261024_193920 | Initialized, b6cb873 | Operator confirmed corrected reverse direction, but still stuck in first curve; forward commands included +22.5 degrees left; stopped |
| 20261024_200836 | Approved parameters, neutral new switches | No meaningful forward motion; arm withdrawn after 4.40 s following a live Pi power fault |
| [20261024_211216](../physical_trial_211216/README.md) | Approved parameters, renewed battery and raised inner boundary | Two operator-confirmed laps without contact, 43.842 / 33.922 s and 6 / 4 reverse starts; legacy speed-reference error remains |

The 200836 record contains `throttled=0x50005`, `lcrit_alarm=1`, temperature
64.8 C at the fault, and a maximum driver gap of 40.53 ms. This is an actual
live undervoltage/throttling event rather than historical `0x50000` alone.
The [official Raspberry Pi flag definitions](https://www.raspberrypi.com/documentation/computers/os.html#get_throttled)
identify bits 0 and 2. The bridge/stack stopped and motor duty was verified at
1,500,000 ns over a 20,000,000 ns period (7.5% neutral). This attempt cannot
compare curve performance. Motor terminal voltage/current were not measured;
the data do not isolate battery, regulator or wiring, or explain all prior stalls.

## Corner diagnosis and limits

The 193920 point cloud shows the front wall and an opening to the right.
The actual forward planner repeatedly chose left steering. It followed one
long wall as the left reference, with target distance 0.36 m and an inferred
maximum distance of 0.64 m. The short inner segment was not a long tracked wall.

Offline static-cloud endpoint checks found right-hand candidates with up to
1.3 m collision-free prefix after a hypothetical 0.5 m backout, but the wall/lane
bound rejected them. Their proven-lane length was only 0.2 m. Collision-free
rays alone do not prove that a path avoids an unseen low boundary. Simply
removing that constraint is not a validated solution.

The recovery planner also assumed up to 0.5 m of reverse travel while execution
usually stopped after 3 seconds with only about 0.14--0.18 m recorded. It scored
forward circular probes, not the complete forward planner with its lane limits.
Extending reverse distance alone, resetting tracker/smoothing, changing wall
costs to use only the free prefix, and enabling the existing wall planner were
screened offline; none consistently corrected the recorded forward choice.
They were not deployed as fixes. The hypothetical endpoint checks are open-loop
kinematic diagnostics using a correlated onboard cloud, not physical success
or external ground truth.

The initialized-profile failures remain unresolved under their original
visibility/parameter conditions. The later approved-profile comparison completed
two physical laps after the battery and inner boundary changed; it cannot
isolate either change or validate the experimental profile. The experimental
switches remain opt-in and are not a replacement for the approved run.

Complete local datasets, snapshots, raw sensors and stop records are retained
under `archive/generated/paper/real_vehicle_<run>/` for all three run IDs.
Downloaded payload SHA256 verification passed: 77, 77 and 78 files respectively;
transient RAM wrapper files are listed separately in each copy report.
The analysis scripts and further traces remain under
`.codex/validation/20261024_feedback_units/`.
