# Approved two-lap trial launcher

These unchanged helper bytes were also used by the latest retained run
[`20261024_005311`](../../../docs/paper/20261024_005311/README.md), with laps of
46.837/14.594 s and 8/1 reverse starts. The directory name records their first
approval, not the latest controller version. From the PC, use
[`race_pi.sh preflight` or `race_pi.sh run`](../../pc/race_pi.sh), as described
in the [real-vehicle README](../../../README.md#from-the-pc-ssh-preflight-and-recognition-to-slam-run).
The PC wrapper delegates to these exact copies and the RAM trial wrapper.

These four Python files are byte-for-byte copies of the helpers used in the
operator-approved ROS run **20261024_000432**, diagnostic **20261024_000431**.
They are versioned here because the original diagnostic directory was local
working material. The complete source/configuration/model snapshot and English
figures are in [the captured run](../../../docs/paper/20261024_000432/README.md).

The helper paths on the Raspberry are part of this recorded setup. Install
these exact copies under the original diagnostic directory from the `APEX`
root, with the vehicle stack stopped:

```bash
mkdir -p data/diagnostics/20261022_open_segment
cp tools/core/validated_trial_20261024_000432/*.py data/diagnostics/20261022_open_segment/
```

For an explicitly authorized trial with the car positioned and track clear:

```bash
sudo python3 tools/core/run_ram_capture.py --root "$PWD" --execute
```

This retains the recorded two-lap/40 m estimated/300 s bounds, 0.40 m/s command
caps, asymmetric nominal steering limits, reverse recovery, arm/power/sensor
guards and temporary 2.0 GHz CPU ceiling. It starts a fresh map/controller,
records trial output in RAM, then verifies and persists it after stopping.
The final postflight report must confirm persistence before power is removed.
See [RAM storage details](../../../docs/ram_logging.md).

The commands assume the recorded Raspberry user `ensta`, workspace
`/home/ensta/AiAtonomousRc/APEX`, prepared ROS Docker image and odometry runtime.
No actuation is triggered by copying these files. The portable archive includes
the exact model weights and runtime source; dependency installation remains the
existing [race-stack setup](../../../docs/race_stack.md).
