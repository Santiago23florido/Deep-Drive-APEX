# Trajectory Supervisor

`trajectory_supervisor_node` añade una capa local opcional entre la ruta global de mapa fijo y el tracker existente.

## Tópicos

Entradas por defecto:

- `/apex/planning/fixed_map_path` (`nav_msgs/Path`): trayectoria global fija.
- `/apex/planning/fixed_map_status` (`std_msgs/String` JSON): estado del planificador global.
- `/apex/odometry/fixed_map_localized` (`nav_msgs/Odometry`): pose localizada.
- `/lidar/scan_localization` (`sensor_msgs/LaserScan`): LiDAR en la convención APEX `forward-left`.

Salidas por defecto:

- `/apex/planning/trajectory_supervisor/local_path` (`nav_msgs/Path`): trayectoria local supervisada.
- `/apex/planning/trajectory_supervisor/status` (`std_msgs/String` JSON): estado `follow`, `avoid`, `rejoin` o `stop_recovery`.

El tracker no cambia por defecto. Para usar esta capa, arranca el pipeline con el tracker apuntando al `local_path` y status del supervisor.

## Ejecución

Con el wrapper de mapa fijo:

```bash
APEX_ENABLE_TRAJECTORY_SUPERVISOR=1 ./APEX/tools/core/apex_fixed_map_follow_up.sh latest
```

Equivalente manual dentro del pipeline:

```bash
export APEX_ENABLE_TRAJECTORY_SUPERVISOR=1
export APEX_RECOGNITION_TRACKER_PATH_TOPIC=/apex/planning/trajectory_supervisor/local_path
export APEX_RECOGNITION_TRACKER_PLANNING_STATUS_TOPIC=/apex/planning/trajectory_supervisor/status
./APEX/tools/capture/apex_raw_capture_up.sh
```

## Parámetros principales

- `lookahead_distance`: distancia de evaluación de la ruta global próxima.
- `local_window_length`: longitud del entorno local considerado con LiDAR.
- `local_window_width`: ancho lateral máximo para filtrar obstáculos locales.
- `obstacle_inflation_radius`: radio de seguridad alrededor de puntos LiDAR.
- `collision_distance_threshold`: separación mínima aceptada entre candidata y obstáculo.
- `path_corridor_width_m`: ancho del corredor usado para decidir si la ruta global está bloqueada.
- `candidate_offset_values`: offsets laterales evaluados respecto al path global.
- `num_candidate_paths`: número usado para generar offsets si `candidate_offset_values` está vacío.
- `rejoin_distance_threshold`: distancia máxima al path global para volver a `follow`.
- `max_avoid_curvature`: límite de curvatura; `0.0` deriva el valor desde `wheelbase_m` y `max_steering_deg`.
- `emergency_stop_distance`: distancia frontal corta para disparar `stop_recovery` si no hay candidata segura.
- `scan_projection_mode`: `apex_forward_left` para los scans actuales, o `laser_scan_angles` para LaserScan ROS estándar.

## Supuestos

- La ruta global ya existe y se publica como `nav_msgs/Path`.
- La pose y el path global están en el mismo frame.
- La salida preferida es `nav_msgs/Path`, no comandos directos, para reutilizar el tracker Pure Pursuit actual.
- Si el scan falta y `publish_pass_through_when_scan_missing` es `true`, el supervisor publica el tramo global sin evasión para no romper el comportamiento previo.
