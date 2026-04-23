# APEX

Raiz operativa del carro real: Raspberry, Docker, firmware, hardware, herramientas PC, workspace ROS 2 y datos reales.

El flujo minimo actual cubre:
- Nano IMU raw por serial
- reflasheo del Nano si no hay stream
- LiDAR estático
- detección de curvas en estático
- odometría raw basada en IMU
- pulso directo de motor/servo para captura corta
- exportación CSV de `imu_raw` y nube 2D LiDAR

No levanta ni usa:
- recon
- sensor fusion
- SLAM
- odometría LiDAR relativa
- navegación automática

## Estructura

```text
APEX/
├── tools/
│   ├── analysis/
│   │   ├── analyze_lidar_curve_snapshot.py
│   │   ├── build_lidar_snapshot.py
│   │   └── run_static_curve_analysis.py
│   ├── capture/
│   │   ├── apex_raw_capture_up.sh
│   │   ├── apex_rect_sensorfus_capture.sh
│   │   └── fetch_rect_sensorfus_capture.sh
│   ├── core/
│   │   ├── apex_core_down.sh
│   │   └── apex_core_up.sh
│   └── firmware/
│       ├── ensure_nano33_stream.sh
│       └── upload_nano33_iot.sh
└── ros2_ws/src/apex_telemetry/apex_telemetry/
    ├── actuation/
    ├── imu/
    ├── odometry/
    └── perception/
```

Los wrappers antiguos en `tools/*.sh` y `tools/*.py` se conservan para compatibilidad.

## Arranque mínimo en Raspberry

```bash
cd ~/AiAtonomousRc/APEX
export PATH="$HOME/local/bin:$PATH"
./tools/core/apex_core_down.sh
APEX_SKIP_BUILD=1 ./tools/capture/apex_raw_capture_up.sh
```

Eso arranca solo:
- `nano_accel_serial_node`
- `kinematics_estimator_node`
- `kinematics_odometry_node`
- `rplidar_publisher_node`
- `static_transform_publisher`

## Test del acelerómetro DFRobot I2C

Este test solo verifica comunicación I2C y lecturas crudas. No reemplaza todavía
el nodo `nano_accel_serial_node`.

Desde el PC, sincroniza el código a la Raspberry:

```bash
cd /home/santiago/AiAtonomousRc
./APEX/tools/core/sync_apex_code_to_pi.sh ensta@raspberrypi:/home/ensta/AiAtonomousRc/APEX/
```

En la Raspberry, con el contenedor `apex_pipeline` corriendo:

```bash
cd ~/AiAtonomousRc/APEX
./tools/hardware/check_dfrobot_accelerometer_i2c.sh --full-scan
```

Si sabes la dirección I2C del módulo, limita la prueba:

```bash
./tools/hardware/check_dfrobot_accelerometer_i2c.sh --address 0x19
```

Si no aparece `/dev/i2c-1`, habilita I2C en la Raspberry y reinicia el
contenedor. Si el módulo DFRobot es analógico, no aparecerá por I2C: hace falta
un ADC externo entre el sensor y la Raspberry.

Si el sensor está conectado a `TX/RX` de la Raspberry, usa el test UART en vez
del test I2C:

```bash
cd ~/AiAtonomousRc/APEX
./tools/hardware/check_dfrobot_accelerometer_uart.sh --port /dev/serial0 --baud 115200 --baud 9600
```

En UART, el `TX` del sensor debe ir al `RX` de la Raspberry y el `RX` del sensor
al `TX` de la Raspberry. Debe haber `GND` común y señal TTL de 3.3 V.

Para el DFRobot WT61PC/SEN0386, usa el parser específico de sus tramas binarias:

```bash
cd ~/AiAtonomousRc/APEX
./tools/hardware/check_dfrobot_wt61pc_uart.sh --port /dev/serial0 --baud 9600
```

## Captura corta raw

```bash
cd ~/AiAtonomousRc/APEX
./tools/capture/apex_rect_sensorfus_capture.sh \
  --run-id recta_raw_base_01 \
  --capture-duration-s 8.0 \
  --drive-delay-s 1.0 \
  --drive-duration-s 4.0 \
  --speed-pct 20 \
  --steering-deg 0
```

Archivos generados en Raspberry:
- `ros2_ws/apex_rect_sensorfus/<run>/imu_raw.csv`
- `ros2_ws/apex_rect_sensorfus/<run>/lidar_points.csv`
- `ros2_ws/apex_rect_sensorfus/<run>/pwm_trace.csv`

## Curva estática aislada

Captura solo LiDAR, sin movimiento ni actuation:

```bash
cd ~/AiAtonomousRc/APEX
./tools/capture/apex_static_curve_capture.sh \
  --run-id curva_estatica_01 \
  --capture-duration-s 4.0
```

Archivos generados en Raspberry:
- `ros2_ws/apex_static_curve/<run>/lidar_points.csv`
- `ros2_ws/apex_static_curve/<run>/lidar_snapshot.csv`
- `ros2_ws/apex_static_curve/<run>/capture.log`

## Traer al PC

```bash
cd /home/santiago/AiAtonomousRc
./APEX/tools/capture/fetch_rect_sensorfus_capture.sh \
  ensta@raspberrypi \
  latest \
  /home/ensta/AiAtonomousRc/APEX/ros2_ws/apex_rect_sensorfus \
  $(pwd)/APEX/data/apex_rect_sensorfus
```

## Curvas en estático

Construir el snapshot desde `lidar_points.csv`:

```bash
cd /home/santiago/AiAtonomousRc/APEX
python3 ./tools/analysis/build_lidar_snapshot.py \
  --run-dir ./data/apex_rect_sensorfus/curva_estatica_01_YYYYMMDDTHHMMSSZ
```

Construir el snapshot y generar directamente la imagen/json:

```bash
cd /home/santiago/AiAtonomousRc/APEX
python3 ./tools/analysis/run_static_curve_analysis.py \
  --run-dir ./data/apex_static_curve/curva_estatica_01_YYYYMMDDTHHMMSSZ
```

Archivos generados:
- `lidar_snapshot.csv`
- `lidar_snapshot_curve_analysis.json`
- `lidar_snapshot_curve_analysis.png`

Núcleo del análisis:
- `tools/analysis/analyze_lidar_curve_snapshot.py`
- `ros2_ws/src/apex_telemetry/apex_telemetry/perception/curve_window_detection.py`
