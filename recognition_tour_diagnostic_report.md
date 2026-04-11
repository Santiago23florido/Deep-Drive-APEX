# Diagnostico de `recognition_tour_08_20260403T112810Z`

## 1. Contexto de la corrida

- Run analizado: `APEX/apex_recognition_tour/recognition_tour_08_20260403T112810Z`
- Artefactos principales usados para el diagnostico:
  - `capture_meta.json`
  - `analysis_recognition_tour/recognition_tour_summary.json`
  - `analysis_recognition_tour/recognition_tour_diagnostics.json`
  - `analysis_recognition_tour/recognition_tour_trajectory.csv`
  - `analysis_recognition_tour/recognition_tour_local_path_history.jsonl`
  - `lidar_points.csv`

### Parametros activos de la captura

Segun `capture_meta.json`, esta corrida uso:

- `mode = recognition_tour`
- `timeout_s = 60.0`
- `enable_imu_lidar_fusion = 1`
- `enable_recognition_tour_planner = 1`
- `enable_recognition_tour_tracker = 1`
- `enable_cmdvel_actuation_bridge = 1`
- `bridge_min_effective_speed_pct = 14.0`
- `bridge_max_speed_pct = 20.0`

### Resultado final observado

Segun `recognition_tour_summary.json`:

- `end_cause = aborted_path_loss`
- `time_total_s = 10.386224025000047`
- `trajectory_row_count = 332`
- `local_path_message_count = 68`
- `route_point_count = 45`
- `lidar_scan_count = 127`
- `lidar_point_count = 12974`

Estado final agregado:

- `fusion_status.state = tracking`
- `fusion_status.latest_pose.confidence = high`
- `planner_status.state = holding_last_path`
- `tracker_status.state = aborted_path_loss`

## 2. Como se estima actualmente la trayectoria

### 2.1 Fusion IMU + LiDAR

La pose usada por `recognition_tour` sale de `imu_lidar_planar_fusion_node`.

- Archivo: `APEX/ros2_ws/src/apex_telemetry/apex_telemetry/estimation/imu_lidar_planar_fusion_node.py`
- Publica odometria corregida y odometria predicha entre scans.
- Parametros relevantes:
  - `publish_predicted_odom_between_scans = True`
  - `predicted_odom_rate_hz = 30.0`
  - `max_prediction_horizon_s = 0.22`

En este run la fusion termina en un estado sano:

- `state = tracking`
- `alignment_ready = true`
- `best_effort_init = false`
- `high_confidence_pct = 100.0`
- `low_confidence_scan_count = 0`
- `fusion_prediction_age_p95_s = 0.015901529788970546`

Conclusion:

- en esta corrida la fusion no aparece como la causa principal del fallo

### 2.2 Planner local de `recognition_tour`

El planner relevante es `recognition_tour_planner_node`.

- Archivo: `APEX/ros2_ws/src/apex_telemetry/apex_telemetry/perception/recognition_tour_planner_node.py`
- Pipeline principal:
  - timer de planning a `12 Hz` en lineas cercanas a `629`
  - timer de publicacion de `status` a `5 Hz` en lineas cercanas a `630`
  - extraccion de centerline desde scans rodantes en lineas cercanas a `1379-1398`
  - fallback con `curve_window` en lineas cercanas a `1399-1405`
  - seleccion o rescate de local path en lineas cercanas a `1408-1415`
  - armado del `status` en lineas cercanas a `1452-1506`

Observaciones importantes del codigo:

- `_path_forward_span_m(path_xy)` usa solo la diferencia en `x` local entre el ultimo y el primer punto.
- Cuando el planner no obtiene un path nuevo pero aun conserva uno anterior, entra en `holding_last_path`.
- En ese estado, `path_forward_span_m` puede terminar en `0.0` porque la metrica se calcula sobre el candidato del ciclo actual, no necesariamente sobre el ultimo path retenido.

Eso significa que `path_forward_span_m = 0.0` es una senal util, pero no prueba por si sola que el path retenido fuera geometrica o cinematicamente nulo.

### 2.3 Tracker de `recognition_tour`

El tracker relevante es `recognition_tour_tracker_node`.

- Archivo: `APEX/ros2_ws/src/apex_telemetry/apex_telemetry/control/recognition_tour_tracker_node.py`
- Consume:
  - `Path` de `recognition_tour_local_path`
  - `planning_status`
  - odometria fusionada

Logica critica:

- `_path_age_s()` en lineas cercanas a `374-391`
- aborto por path stale en lineas cercanas a `468-474`
- aborto por falta de target hacia delante en lineas cercanas a `621-674`

Detalle clave:

- `_path_age_s()` toma el maximo entre `planning_status.local_path_age_s` y la edad del ultimo `Path` recibido localmente

Consecuencia:

- si el planner genera un `Path` nuevo, pero el `status` llega tarde o sigue reportando una edad vieja, el tracker puede seguir viendo el path como stale y abortar aunque ya exista una trayectoria nueva valida

### 2.4 Bridge de actuacion

El nodo que convierte `cmd_vel` en PWM real es `cmd_vel_to_apex_actuation_node`.

- Archivo: `APEX/ros2_ws/src/apex_telemetry/apex_telemetry/actuation/cmd_vel_to_apex_actuation_node.py`
- Servo real:
  - `APEX/ros2_ws/src/apex_telemetry/apex_telemetry/actuation/actuation.py`

Problema observado:

- el tracker calcula steering deseado por curvatura
- ese steering no se satura de forma explicita dentro del tracker
- el clamp real ocurre tarde, en la capa del servo
- por eso aparecen ordenes muy por encima del limite fisico del vehiculo

Eso degrada el seguimiento incluso cuando el planner si entrega un path util.

## 3. Evidencia observada en logs

### 3.1 Metricas agregadas

Segun `recognition_tour_diagnostics.json`:

- `primary_cause = planner`
- `path_age_p95_s = 0.6328295040000285`
- `planner_odom_age_p95_s = 0.6507511615753173`
- `tracker_odom_age_p95_s = 0.06470762491226197`
- `path_forward_span_p10_m = 0.0`
- `short_path_fraction = 0.1927710843373494`
- `fallback_fraction = 0.0`
- `steering_error_p95_deg = 23.603773199855386`
- `path_deviation_p95_m = 0.11877625574867313`
- `max_desired_steering_deg = 46.141602138499024`
- `max_applied_steering_deg = 43.1964710300669`

Lectura rapida:

- la fusion y la odometria del tracker no muestran una degradacion comparable al planner
- el envejecimiento del path y del `planning_status` si es grande
- el controlador esta intentando giros fisicamente demasiado agresivos

### 3.2 Conteo de estados durante la corrida

Desde `recognition_tour_trajectory.csv`:

Planner:

- `tracking`: 268 muestras
- `holding_last_path`: 44 muestras
- `waiting_local_path`: 20 muestras

Tracker:

- `tracking`: 278 muestras
- `holding_last_path`: 18 muestras
- `aborted_path_loss`: 36 muestras

La corrida no se cae por un solo salto aislado. Entra varias veces en estados intermedios antes del terminal.

### 3.3 Transiciones temporales criticas

Transiciones reconstruidas desde `recognition_tour_trajectory.csv`:

- `t = 7.957876245 s`
  - `planner = holding_last_path`
  - `tracker = tracking`
  - `path_age_s = 0.23757116899997754`
  - `planner_path_forward_span_m = 0.0`
- `t = 8.352105448 s`
  - `planner = tracking`
  - `tracker = tracking`
  - `path_age_s = 0.0`
  - `planner_path_forward_span_m = 1.0333087258499063`
- `t = 8.823316568 s`
  - `planner = holding_last_path`
  - `tracker = holding_last_path`
  - `path_age_s = 0.2918615400000135`
  - `planner_path_forward_span_m = 0.0`
- `t = 9.161433620 s`
  - `planner = waiting_local_path`
  - `tracker = holding_last_path`
  - `path_age_s = 0.6328295040000285`
  - `planner_path_forward_span_m = 0.0`
- `t = 9.546262277 s`
  - `planner = waiting_local_path`
  - `tracker = aborted_path_loss`
  - `path_age_s = 0.6328295040000285`
  - `planner_path_forward_span_m = 0.0`
- `t = 9.693984617 s`
  - `planner = tracking`
  - `tracker = aborted_path_loss`
  - `path_age_s = 0.0`
  - `planner_path_forward_span_m = 1.031793426144778`

Esta ultima transicion es la evidencia mas importante del run:

- el planner ya habia vuelto a `tracking`
- `path_age_s` ya habia vuelto a `0.0`
- el `forward_span` ya habia vuelto a una cifra sana, `1.031793426144778 m`
- pero el tracker ya habia quedado terminal

Esto apunta a una desincronizacion o a una ventana de decision demasiado agresiva entre planner y tracker.

### 3.4 Evidencia de steering fisicamente irreal

Ejemplos extremos extraidos de `recognition_tour_trajectory.csv`:

- `t = 2.410524272 s`
  - `desired_steering_deg = 47.98807215802111`
  - `applied_steering_deg = 29.347231961671447`
- `t = 7.625859292 s`
  - `desired_steering_deg = 47.91787041474917`
  - `applied_steering_deg = 37.54450086934277`
- `t = 2.121007971 s`
  - `desired_steering_deg = 47.78873692439812`
  - `applied_steering_deg = 9.883966928331876`

Con el hardware configurado a `18 deg`, estos numeros muestran dos problemas:

- el sistema esta pidiendo un steering que excede el limite mecanico
- ademas el propio log de `applied_steering_deg` supera ese limite, lo que indica que ese campo no refleja de forma fiable el clamp fisico final o que su semantica no esta alineada con la del servo real

### 3.5 Reconstruccion local del planner sobre el log

Se hizo una reconstruccion offline de la geometria disponible a varios instantes del run usando `lidar_points.csv`.

Resultados resumidos:

- alrededor de `t = 8.82 s`
  - `rolling_points = 575`
  - `valid_bins = 1`
  - `centerline_valid = false`
  - `fallback_curve_window_valid = false`
- alrededor de `t = 9.16 s`
  - `rolling_points = 510`
  - `valid_bins = 7`
  - `centerline_valid = true`
  - `fallback_curve_window_valid = false`
- alrededor de `t = 9.55 s`
  - `rolling_points = 498`
  - `valid_bins = 6`
  - `centerline_valid = true`
  - `fallback_curve_window_valid = false`

Interpretacion:

- si hubo una perdida real de geometria explotable en torno a `8.82 s`
- pero el centerline vuelve a ser valido antes del aborto terminal
- aun asi el tracker cae en terminal antes de recuperarse del todo

Eso refuerza que hay dos capas del problema:

- una fragilidad real del planner en ciertos tramos
- una gestion deficiente de recuperacion entre planner y tracker

## 4. Causas probables, ordenadas por impacto

### 4.1 Causa principal probable: desincronizacion planner-tracker

Hipotesis principal:

- el tracker puede abortar por edad de path o `status` desfasados aunque ya exista un `Path` nuevo

Evidencia:

- el planner corre a `12 Hz`, pero el `status` se publica por timer a `5 Hz`
- `_path_age_s()` en el tracker usa el peor caso entre el age del planner y el `Path` recibido
- el run muestra que a `t = 9.693984617 s` el planner ya vuelve a `tracking` con `path_age = 0.0`, pero el tracker ya estaba en `aborted_path_loss`
- `planner_odom_age_p95_s = 0.6507511615753173` y `path_age_p95_s = 0.6328295040000285` son consistentes con una cadena de planning o status que envejece demasiado para las ventanas del tracker

Impacto:

- muy alto
- explica por que la recuperacion llega tarde aunque la geometria vuelva a ser usable

### 4.2 Causa contribuyente importante: steering ordenado por encima del limite fisico

Hipotesis secundaria:

- el sistema pide giros superiores al limite fisico del servo, degradando el seguimiento real

Evidencia:

- `max_desired_steering_deg = 46.141602138499024`
- el limite mecanico configurado es `18 deg`
- `steering_error_p95_deg = 23.603773199855386`
- varios instantes del log muestran ordenes cercanas a `48 deg`

Impacto:

- alto
- no explica por si solo el aborto terminal, pero empeora el seguimiento del path, aumenta desviacion y reduce la robustez del planner y del tracker en curvas

### 4.3 Causa contribuyente: fragilidad del planner local ante perdida momentanea de centerline

Hipotesis:

- el planner pierde temporalmente centerline valida y su fallback actual no cubre suficiente terreno en este escenario

Evidencia:

- `fallback_fraction = 0.0`
- en la reconstruccion local alrededor de `8.82 s` el centerline queda invalido con solo `1` bin valido
- el fallback de `curve_window` tambien falla en los puntos muestreados problematicos

Impacto:

- medio a alto
- produce ventanas donde no hay path fresco, lo que activa la logica de stale path

## 5. Riesgos secundarios y senales que no parecen ser la causa principal

### 5.1 La fusion no aparece como causa dominante en esta corrida

Senales a favor de descartar la fusion como causa principal:

- `fusion_status.state = tracking`
- `alignment_ready = true`
- `best_effort_init = false`
- `high_confidence_pct = 100.0`
- `low_confidence_scan_count = 0`
- `fusion_prediction_age_p95_s = 0.015901529788970546`

Puede haber mejoras futuras en la fusion, pero este run no apunta ahi como primer problema.

### 5.2 `path_forward_span_m = 0.0` necesita interpretarse con cuidado

Ese valor si coincide con momentos problematicos, pero en `holding_last_path` no siempre significa que el ultimo path retenido tenga longitud o continuidad cero. Tambien puede significar que el planner no publico un candidato nuevo en ese ciclo.

Por eso no conviene usar este campo aislado como criterio unico de diagnostico.

### 5.3 El bridge reporta steering aplicado inconsistente con el limite fisico

Hay una incoherencia entre:

- el limite mecanico de `18 deg`
- los valores registrados como `applied_steering_deg`, que superan `40 deg`

Esto puede ser:

- una semantica equivocada del campo logueado
- un clamp que ocurre en otro dominio distinto al reportado
- o una falta de saturacion previa en la cadena de mando

No es la causa terminal principal del run, pero si una fuente de confusion de diagnostico y una degradacion real del control.

## 6. Cambios recomendados, en orden de implementacion

### 6.1 Prioridad 1: cerrar la desincronizacion planner-tracker

- Hacer que el tracker base la frescura del path en el `header.stamp` del `Path` actual, no en el maximo entre `status_age` y edad local.
- Publicar `status` inmediatamente cuando el planner publique un path nuevo.
- Incluir en `planning_status` un identificador temporal real del path:
  - `path_stamp_s`
  - o `path_seq`
- Mantener el timer de `5 Hz` solo como heartbeat, no como unica fuente de verdad temporal.

Resultado esperado:

- evitar `aborted_path_loss` falsos cuando ya existe path nuevo pero el `status` llega desfasado

### 6.2 Prioridad 2: alinear el control con el limite fisico de steering

- Saturar steering en el tracker antes de convertirlo a `cmd_vel`.
- Recalcular `angular_z` usando el steering ya clamped.
- Saturar tambien en el bridge antes del rate limit.
- Corregir la semantica de `applied_steering_deg` para que el log refleje el steering fisicamente realizable.

Resultado esperado:

- bajar `steering_error_p95_deg`
- reducir desviacion lateral
- evitar ordenes imposibles para el vehiculo

### 6.3 Prioridad 3: hacer mas robusto el planner local

- Cuando falle `_extract_centerline(...)`, intentar una trayectoria de rescate explicita desde el ultimo path aceptado.
- Diferenciar con claridad en `status`:
  - `tracking`
  - `rescue_previous_path`
  - `holding_last_path`
  - `waiting_local_path`
- Loguear:
  - `centerline_valid_bin_count`
  - `candidate_path_rejection_reason`
  - `fallback_curve_window_valid`
  - `rolling_scan_count`
  - `rolling_point_count`

Resultado esperado:

- mejor recuperacion en tramos donde la geometria se degrada temporalmente

### 6.4 Prioridad 4: mejorar el diagnostico futuro

- Guardar en el `status` las metricas del path realmente retenido, no solo del candidato del ciclo actual.
- Evitar que `path_forward_span_m = 0.0` represente al mismo tiempo:
  - ausencia de candidato nuevo
  - o patologia geometrica real

Resultado esperado:

- logs mas interpretables
- menos falsos positivos al inspeccionar runs

## 7. Conclusion operativa

La evidencia de este run apunta a que el problema dominante no es la estimacion base de pose, sino la coordinacion entre:

- planner local
- tracker
- y la semantica temporal del path o del `status`

La secuencia mas consistente con los logs es:

1. el planner pierde continuidad util de path en un tramo curvo
2. entra en `holding_last_path` y luego `waiting_local_path`
3. el tracker considera el path stale durante demasiado tiempo
4. aborta por `aborted_path_loss`
5. el planner recupera `tracking` poco despues, pero ya demasiado tarde

En paralelo, el controlador esta pidiendo steering muy por encima de la capacidad fisica del robot, lo que empeora el seguimiento y aumenta la probabilidad de entrar en la ventana de perdida del planner.

## 8. Resumen corto para reutilizar en otro prompt

Diagnostico principal del run `recognition_tour_08_20260403T112810Z`:

- La fusion IMU+LiDAR no parece ser la causa principal.
- El fallo termina por `aborted_path_loss`.
- La causa mas probable es una desincronizacion entre `recognition_tour_planner_node` y `recognition_tour_tracker_node`: el tracker envejece el path usando `planning_status.local_path_age_s` y puede abortar aunque ya exista un `Path` nuevo.
- Evidencia clave: a `t = 9.693984617 s` el planner vuelve a `tracking` con `path_age = 0.0` y `forward_span = 1.031793426144778 m`, pero el tracker ya esta terminal.
- Causa contribuyente fuerte: el sistema pide steering de hasta `46-48 deg` con un limite fisico de `18 deg`.
- Causa adicional: el planner pierde centerline valida en algunos tramos y el fallback actual no cubre bien esos huecos.
- Orden recomendado de correccion:
  1. sincronizar planner y tracker por timestamp real del path
  2. hacer clamping explicito de steering antes del bridge
  3. robustecer el rescate del planner y sus metricas de diagnostico
