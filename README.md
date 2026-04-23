# AiAtonomousRc

Repositorio dividido en tres raices:

- `APEX/`: stack del carro real, Docker/Raspberry, firmware, hardware, herramientas PC y datos reales.
- `APEXsim/`: simulacion ROS 2/Gazebo, simulador legacy, RViz, herramientas y datos de simulacion.
- `APEX_external/`: material legado, artefactos generados y archivos no operativos.

Los workspaces ROS se construyen por separado para evitar paquetes duplicados.

## Carro Real

```bash
cd ~/AiAtonomousRc/APEX/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select apex_telemetry voiture_system
source install/setup.bash
```

Arranque principal en Raspberry/contendor:

```bash
cd ~/AiAtonomousRc/APEX
./tools/core/apex_core_up.sh
```

## Simulacion

```bash
cd ~/AiAtonomousRc/APEXsim/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select rc_sim_description apex_telemetry voiture_system
source install/setup.bash
```

Arranque recomendado:

```bash
cd ~/AiAtonomousRc
./APEXsim/tools/sim/apex_sim_up.sh --scenario baseline --rviz
```

Smoke test headless:

```bash
cd ~/AiAtonomousRc/APEXsim/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
timeout 45s ros2 launch rc_sim_description apex_sim.launch.py scenario:=baseline rviz:=false gazebo_gui:=false
```

Escenarios disponibles: `baseline`, `precision_fusion`, `tight_right_saturation`, `outer_long_inner_short`, `startup_pose_jump`, `narrowing_false_corridor`.
