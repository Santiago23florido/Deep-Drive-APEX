# APEX Sim

Arranque principal:

```bash
./APEX/tools/sim/apex_sim_up.sh --scenario baseline --rviz
```

Para levantar además el mapa fijo de `slam_toolbox`:

```bash
./APEX/tools/sim/apex_sim_up.sh --scenario baseline --rviz --slam
```

Launch equivalente:

```bash
ros2 launch rc_sim_description apex_sim.launch.py scenario:=baseline rviz:=true
```

Launch equivalente con SLAM:

```bash
ros2 launch rc_sim_description apex_sim.launch.py scenario:=baseline rviz:=true use_slam:=true
```

Sin `--slam`, la RViz por defecto es la vista real de recognition:

```text
APEX/rviz/apex_recognition_live.rviz
```

Con `--slam`, la RViz por defecto pasa a:

```text
APEX/rviz/apex_recognition_slam_live.rviz
```

Esa config muestra los mismos topics principales del coche:

- `/apex/estimation/full_map_points`
- `/apex/estimation/live_map_points`
- `/apex/odometry/imu_lidar_fused`
- `/apex/estimation/path`
- `/apex/planning/recognition_tour_route`
- `/apex/planning/recognition_tour_local_path`
- `/lidar/scan_localization`
- `/map`
- `/map_updates`

Si quieres abrir otra config:

```bash
ros2 launch rc_sim_description apex_sim.launch.py \
  scenario:=baseline \
  rviz:=true \
  rviz_config:=/home/santiago/AiAtonomousRc/APEX/rviz/apex_slam_auto.rviz
```

Armar `recognition_tour` cuando la simulación ya está levantada:

```bash
./APEX/tools/sim/apex_arm_recognition_tour.sh
```

Captura completa de una corrida simulada:

```bash
./APEX/tools/sim/apex_recognition_tour_sim_capture.sh --scenario tight_right_saturation --timeout-s 60
```

Comparar una corrida real contra una simulada:

```bash
python3 APEX/tools/analysis/compare_recognition_tour_runs.py \
  --real-run APEX/apex_recognition_tour/<real_run> \
  --sim-run APEX/apex_recognition_tour/<sim_run>
```

Mapping manual con Xbox + mapa offline final:

```bash
./APEX/tools/sim/apex_manual_mapping_up.sh --scenario precision_fusion --rviz
```

Antes, genera una vez el puente Windows:

```bash
./APEX/tools/windows/build_apex_xbox_bridge.sh
```

Luego abre en Windows:

```text
APEX/tools/windows/dist/apex_xbox_bridge.exe
```

Por defecto:

- stick izquierdo `Y`: velocidad
- stick izquierdo `X`: dirección

Cuando termines la vuelta manual y quieras congelar la captura, correr `sensor_fusionn.py` y ver la nube final en RViz:

```bash
./APEX/tools/sim/apex_manual_mapping_finish.sh
```

Ese flujo deja los artefactos en:

```text
APEX/apex_forward_raw/manual_xbox_<timestamp>/
```

y publica el mapa final en:

- `/apex/sim/offline_map_points`
- `/apex/sim/offline_map_path`
- `/apex/sim/offline_map_status`

Escenarios disponibles:

- `baseline`
- `tight_right_saturation`
- `outer_long_inner_short`
- `startup_pose_jump`
- `narrowing_false_corridor`

Topics sim-only clave:

- `/apex/sim/pwm/steering_dc`
- `/apex/sim/pwm/motor_dc`
- `/apex/sim/ground_truth/odom`
- `/apex/sim/ground_truth/path`
- `/apex/sim/ground_truth/perfect_map_points`
- `/apex/sim/ground_truth/status`

## Replicado fielmente

| Área | Estado |
|---|---|
| Stack ROS2 real | Se lanza `apex_pipeline.launch.py` con el mismo `apex_params.yaml` como base |
| Fusión online | Sigue usando `imu_lidar_planar_fusion_node`; no entra ground truth en la estimación |
| Planner/tracker/bridge | Se ejecutan los nodos reales actuales y sus topics reales |
| Frames y offsets | `base_link`, `laser` y `rear_axle` con `lidar_offset_x_m=0.18` y `rear_axle_offset_x_m=-0.15` |
| Actuation logic | El cálculo de PWM, trims, clamps, rampas y `applied_steering_deg` sigue en `cmd_vel_to_apex_actuation_node` |
| Artefactos de capture | Se conserva el layout de `recognition_tour` y se añaden logs/CSV de GT |

## Aproximado

| Área | Aproximación actual |
|---|---|
| Planta ESC/servo | `apex_gz_vehicle_bridge.py` modela dinámica, rate limits, deadband y asimetría, pero sigue siendo un modelo físico calibrable |
| IMU no ideal | Bias, drift, ruido y mala calibración inicial están modelados en `nano_accel_serial_node` sim backend |
| LiDAR no ideal | Latencia, ruido, huecos e `inf` están modelados en `rplidar_publisher_node` sim backend sobre el scan de Gazebo |
| Pista escenarios | Las variantes de world son extensiones mínimas sobre `basic_track.world`, no una reconstrucción CAD exacta del circuito real |
| Clearances GT | Se calculan contra la geometría SDF de paredes del world activo |
