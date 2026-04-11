# Final Tests README

This file is the short operational version of the final test procedure.

If you need the detailed explanation, use:

- [Final Test Commands](./final-test-commands.md)

## PC Terminal 1

```bash
cd /home/santiago/AiAtonomousRc
rsync -av APEX/ros2_ws/ ensta@raspberrypi:/home/ensta/AiAtonomousRc/APEX/ros2_ws/
```

## Raspberry Pi Terminal 1

```bash
ssh ensta@raspberrypi
cd /home/ensta/AiAtonomousRc/APEX
export PATH="$HOME/local/bin:$PATH"
./tools/core/apex_core_down.sh
APEX_SKIP_BUILD=1 ./tools/capture/apex_curve_track_capture.sh --run-id curve_track_hard_stop_01 --timeout-s 22.0
```

## Raspberry Pi Terminal 2

```bash
ssh ensta@raspberrypi
docker exec apex_pipeline /bin/bash -lc "source /opt/ros/jazzy/setup.bash && ros2 topic echo /apex/planning/curve_entry_status"
```

## Raspberry Pi Terminal 3

```bash
ssh ensta@raspberrypi
docker exec apex_pipeline /bin/bash -lc "source /opt/ros/jazzy/setup.bash && ros2 topic echo /apex/tracking/status"
```

## Raspberry Pi Terminal 4

```bash
ssh ensta@raspberrypi
docker exec apex_pipeline /bin/bash -lc "source /opt/ros/jazzy/setup.bash && ros2 topic echo /apex/vehicle/drive_bridge_status"
```

## PC Terminal 2

```bash
cd /home/santiago/AiAtonomousRc
./APEX/tools/capture/fetch_curve_track_capture.sh \
  ensta@raspberrypi \
  latest \
  /home/ensta/AiAtonomousRc/APEX/ros2_ws/apex_curve_track \
  "$(pwd)/APEX/apex_curve_track"
```

## PC Terminal 2

```bash
cd /home/santiago/AiAtonomousRc
RUN_ID=$(ls -1dt ./APEX/apex_curve_track/curve_track_hard_stop_01_* | head -n 1 | xargs -r basename)
python3 ./APEX/tools/analysis/plot_curve_tracking_run.py --run-dir "./APEX/apex_curve_track/$RUN_ID"
xdg-open "./APEX/apex_curve_track/$RUN_ID/analysis_curve_tracking/curve_tracking_overview.png"
cat "./APEX/apex_curve_track/$RUN_ID/analysis_curve_tracking/tracking_summary.json"
```

## Important Note

Keep the vehicle completely still before the run starts. If not, the planner can remain in `waiting_static` and no path will be generated.
