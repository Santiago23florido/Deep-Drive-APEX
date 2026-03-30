# Final Test Commands

This file contains the commands to run the final curve-tracking tests with the current implementation.

The commands are split by machine and by terminal.

## Preconditions

- The PC workspace is `/home/santiago/AiAtonomousRc`.
- The Raspberry Pi workspace is `/home/ensta/AiAtonomousRc/APEX`.
- The Raspberry host is reachable as `ensta@raspberrypi`.
- The car must remain completely still before the run starts, otherwise the planner can remain in `waiting_static`.

## Terminal 1 on the PC

Upload the current ROS workspace to the Raspberry Pi:

```bash
cd /home/santiago/AiAtonomousRc
rsync -av APEX/ros2_ws/ ensta@raspberrypi:/home/ensta/AiAtonomousRc/APEX/ros2_ws/
```

## Terminal 2 on the Raspberry Pi

Start a new capture:

```bash
ssh ensta@raspberrypi
cd /home/ensta/AiAtonomousRc/APEX
export PATH="$HOME/local/bin:$PATH"
./tools/core/apex_core_down.sh
APEX_SKIP_BUILD=1 ./tools/capture/apex_curve_track_capture.sh --run-id curve_track_hard_stop_01 --timeout-s 22.0
```

## Terminal 3 on the Raspberry Pi

Monitor planner state in real time:

```bash
ssh ensta@raspberrypi
docker exec apex_pipeline /bin/bash -lc "source /opt/ros/jazzy/setup.bash && ros2 topic echo /apex/planning/curve_entry_status"
```

Expected progression:

- `waiting_fusion`
- `waiting_static`
- `collecting_snapshot`
- `ready`

If it stays in `vehicle_not_static`, the car moved too much during startup.

## Terminal 4 on the Raspberry Pi

Monitor tracker state in real time:

```bash
ssh ensta@raspberrypi
docker exec apex_pipeline /bin/bash -lc "source /opt/ros/jazzy/setup.bash && ros2 topic echo /apex/tracking/status"
```

Near the end of the run, the expected terminal cause is one of:

- `goal_proximity_stop`
- `goal_line_crossed`
- `goal_reached`

## Terminal 5 on the Raspberry Pi

Monitor actuation bridge state:

```bash
ssh ensta@raspberrypi
docker exec apex_pipeline /bin/bash -lc "source /opt/ros/jazzy/setup.bash && ros2 topic echo /apex/vehicle/drive_bridge_status"
```

This is useful to verify that:

- speed command drops to zero,
- steering centers on stop,
- and the bridge enters brake-to-neutral behavior.

## Terminal 2 on the PC

Fetch the latest captured run:

```bash
cd /home/santiago/AiAtonomousRc
./APEX/tools/capture/fetch_curve_track_capture.sh \
  ensta@raspberrypi \
  latest \
  /home/ensta/AiAtonomousRc/APEX/ros2_ws/apex_curve_track \
  "$(pwd)/APEX/apex_curve_track"
```

## Terminal 2 on the PC

Plot and inspect the run:

```bash
cd /home/santiago/AiAtonomousRc
RUN_ID=$(ls -1dt ./APEX/apex_curve_track/curve_track_hard_stop_01_* | head -n 1 | xargs -r basename)
python3 ./APEX/tools/analysis/plot_curve_tracking_run.py --run-dir "./APEX/apex_curve_track/$RUN_ID"
xdg-open "./APEX/apex_curve_track/$RUN_ID/analysis_curve_tracking/curve_tracking_overview.png"
cat "./APEX/apex_curve_track/$RUN_ID/analysis_curve_tracking/tracking_summary.json"
```

## If Plotting Fails

If `plot_curve_tracking_run.py` reports missing path points, the planner did not generate a path. The most common cause is:

- `planner_status.state = waiting_static`

In that case:

1. inspect the planner status topic live,
2. keep the vehicle still for longer before startup,
3. rerun the capture.

## Current Safety-Relevant Parameters

The final tests rely on these current behaviors:

- fixed speed command at `20%`,
- proximity stop in the tracker,
- active brake-to-neutral in the bridge,
- steering centering on zero-speed commands,
- and no software steering clamp in the tracker or bridge.
