# LiDAR ROS2 Split: Raspberry (Publisher) + PC (Subscriber)

Esta carpeta quedo separada para despliegue distribuido por WiFi:

```
Lidar/
  common/
    lidar_scan_buffer.py      # Logica compartida de filtrado y mapeo 360
  raspberry/
    lidar_publisher_node.py   # Nodo ROS2 que lee RPLidar y publica /lidar/scan
    run_publisher.sh
    requirements.txt
  pc/
    lidar_subscriber_node.py  # Nodo ROS2 que se suscribe y muestra mediciones
    run_subscriber.sh
    requirements.txt
  legacy/
    ...                       # Codigo previo preservado
```

## 1) Lo que va en la Raspberry Pi 4

Requisitos:
- ROS2 instalado (misma distro que en el PC)
- LiDAR conectado (ej. `/dev/ttyUSB0`)
- Python deps: `numpy`, `rplidar-roboticia`

Comandos:

```bash
cd /home/santiago/AiAtonomousRc
python3 -m pip install -r Lidar/raspberry/requirements.txt

# Recomendado en ambos equipos (Raspberry y PC)
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42

# Ejecutar publisher
Lidar/raspberry/run_publisher.sh --port /dev/ttyUSB0 --baudrate 256000 --topic /lidar/scan
```

## 2) Lo que va en tu PC

Requisitos:
- ROS2 instalado (misma distro que Raspberry)
- Misma red WiFi

Comandos:

```bash
cd /home/santiago/AiAtonomousRc
python3 -m pip install -r Lidar/pc/requirements.txt

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42

# Subscriber con resumen en consola
Lidar/pc/run_subscriber.sh --topic /lidar/scan

# Opcional: imprimir las 360 mediciones completas
Lidar/pc/run_subscriber.sh --topic /lidar/scan --full
```

## 3) Validacion de conexion por WiFi

En el PC:

```bash
ros2 topic list | grep lidar
ros2 topic echo /lidar/scan --once
```

Si no aparece el topico:
- Verifica que ambos equipos tengan el mismo `ROS_DOMAIN_ID`.
- Verifica misma `RMW_IMPLEMENTATION` en ambos.
- Confirma que Raspberry y PC esten en la misma subred WiFi.
- Revisa firewall (DDS usa descubrimiento multicast).

## 4) Parametros importantes del publisher (Raspberry)

- `--heading-offset-deg`: corrige orientacion del LiDAR.
- `--fov-filter-deg`: recorta campo de vision (0-360).
- `--point-timeout-ms`: invalida angulos con datos viejos.
- `--range-min` / `--range-max`: limites del mensaje `LaserScan`.
