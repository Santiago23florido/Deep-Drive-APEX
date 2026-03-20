# Raspberry Legacy vs ROS APEX Audit

## Resumen
- `APEX` mantiene LiDAR, SLAM, topics ROS2, deteccion de vuelta y guardado de mapa.
- El port realizado se limita a la logica de reconocimiento necesaria para que la vuelta use un comportamiento mas cercano al legacy.
- `pole-software` no se incorpora a la rama final ni queda como dependencia runtime.

## Diferencias Principales
- Navegacion:
  - `APEX` original elegia heading con apertura libre mas centrado lateral.
  - Legacy prioriza apertura libre con convolucion frontal y correccion `avoid_corner`.
  - El port cambia `recon_navigation.py` a esa seleccion legacy-compatible y mantiene la interfaz `ReconCommand`.
- Steering:
  - `APEX` original convertia heading a steering con una ley lineal `heading * steering_gain`.
  - Legacy usa una ley no lineal por tramos (`STEER_FACTOR`).
  - El port adopta esa ley no lineal y conserva el parametro `steering_gain` como escala relativa sobre la curva legacy.
- PWM del servo:
  - Legacy tiene un desfase implicito del centro PWM: promedio(`dc_min`, `dc_max`) `- 0.33`.
  - El port aplica ese offset en codigo, pero preserva el rango configurable `[dc_min, dc_max]` de `APEX`.
- Motor/reverse:
  - `APEX` ya tenia una secuencia brake-neutral-reverse compatible con el patron del legacy.
  - En esta iteracion no se cambia la interfaz del motor; se valida con tests la secuencia esperada.

## Lo Que Se Porta
- Seleccion de apertura libre con suavizado por convolucion.
- Correccion tipo `avoid_corner`.
- Conversion no lineal de heading a steering.
- Desfase legacy del centro PWM del servo.

## Lo Que Se Deja Fuera
- Driver LiDAR del legacy.
- Config loader JSON del legacy.
- Visualizacion y state machine del legacy.
- Dependencia directa con `pole-software`.
