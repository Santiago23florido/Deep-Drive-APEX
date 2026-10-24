# RAM capture for finite real-vehicle trials

The trial controller writes its CSV from its control callback. A slow storage
write can therefore delay a later command. In trial 20261023_233948 the command
source gap reached 309.55 ms and the control callback held its core lock for
301.5 ms. Buffer replay of the exact CSV triggers a write at the last row;
the duration of that actual write was not instrumented, so the SD-card cause
remains a hypothesis rather than an isolated measurement.

`tools/core/run_ram_capture.py` keeps the existing controller, model and limits,
and temporarily binds these two host output directories onto `/dev/shm`:

- `ros2_ws/apex_race`
- `data/diagnostics/20261022_open_segment`

The historical directories are temporarily hidden, not removed. Existing Docker
bind mounts include the RAM submounts. The runner checks `tmpfs` on both host
and container paths before arming. Raw IMU/LiDAR, receipt timing, driver and
estimator CSVs, actuator/status JSONL, power samples and per-process resource
samples therefore go to RAM. Ordinary OS and Docker/ROS console logging are
not globally disabled or claimed to be free of disk writes.

## Storage validation without actuation

With the vehicle stack stopped and motor neutral:

```sh
sudo python3 tools/core/run_ram_capture.py --root "$PWD" --storage-self-test
```

This uses an isolated container with networking disabled and no actuator-device
mounts. It writes known files through both host and container paths, verifies
tmpfs, and exercises verified persistence and restoration of the original paths.

## Finite trial

Once the car is physically ready for the agreed trial:

```sh
sudo python3 tools/core/run_ram_capture.py --root "$PWD" --execute
```

This runs the same `run_oval.py` profile via the existing temporary 2.0 GHz CPU
wrapper: two laps, at most 40 m estimated or 300 s, with the existing arm, power,
command-gap, sensor-age and actuator protections. The controller source and
configuration are not edited by the RAM wrapper.

## Persistence and recovery

After the child runner has disarmed and stopped all writers, the wrapper checks
motor neutral, copies the complete RAM tree into a new SD staging directory,
verifies SHA256 and sizes, and fsyncs the files and directories. It then unmounts
the two overlays, publishes only new trial directories without replacing older
evidence, updates the last-run pointer, and fsyncs publication metadata. It
removes RAM/staging only after success. The printed
`data/diagnostics/ram_capture_<token>.json` records the verification and paths.

Do not power-cycle the Pi until this confirmation: RAM output is volatile before
persistence. If backup or publication fails, the wrapper keeps the RAM source
and any staging copy. A new invocation refuses to cover another mount. Inspect
the printed recovery paths and stop the vehicle stack before recovery; do not
delete the remaining RAM or staging data to clear an error.

Tests cover preservation of old trials, destination collisions, copy corruption,
unexpected top-level output and symlinks. Actual storage-only tests on the Pi
also verified that nested Docker mounts see RAM and that the historical paths
return after persistence. A successful physical repeat is evidence about this
implementation; it does not establish that every earlier delay had one cause.
