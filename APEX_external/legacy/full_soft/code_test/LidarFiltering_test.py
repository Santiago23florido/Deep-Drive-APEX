import numpy as np
import matplotlib.pyplot as plt
from code.algorithm.LidarFiltering import LidarFiltering

FIELD_OF_VIEW = 180  # LiDAR real de 0-180°

# ----------------------------
# Funções auxiliares
# ----------------------------
def create_empty_lidar(default_distance=5.0):
    return np.ones(FIELD_OF_VIEW) * default_distance

def add_obstacle(lidar, start_angle, end_angle, distance):
    s = max(0, min(FIELD_OF_VIEW - 1, start_angle))
    e = max(0, min(FIELD_OF_VIEW - 1, end_angle))
    if s <= e:
        lidar[s:e + 1] = distance
    else:
        lidar[s:FIELD_OF_VIEW] = distance
        lidar[0:e + 1] = distance
    return lidar

def lidar_180_to_360(lidar_180, fill_value=5.0):
    """Expande LiDAR 0–180° para 360° para compatibilidade"""
    lidar_360 = np.ones(360) * fill_value
    lidar_360[0:FIELD_OF_VIEW] = lidar_180
    return lidar_360

def plot_lidar(lidar_180, target_angle, steer):
    angles_rad = np.deg2rad(np.arange(FIELD_OF_VIEW))
    plt.clf()  # Limpa o gráfico anterior
    ax = plt.subplot(111, polar=True)
    ax.plot(angles_rad, lidar_180, color='blue', label='LiDAR')
    ax.plot([np.deg2rad(target_angle), np.deg2rad(target_angle)],
            [0, np.max(lidar_180)],
            color='red', linewidth=2, label='Target')
    ax.set_thetamin(0)
    ax.set_thetamax(FIELD_OF_VIEW)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(f"Target: {target_angle:.1f}°, Steer: {steer:.1f}°")
    ax.legend(loc='upper right')
    plt.pause(0.01)  # Atualiza o gráfico

# ----------------------------
# Loop interativo multi-obstáculo
# ----------------------------
def run_simulation():
    print("=== Simulador Gráfico LiDAR Multi-Obstáculo ===")
    print("Digite obstáculos no formato: ang_inicial ang_final distancia")
    print("Ex: 80 100 0.65 | Digite 'q' para sair")

    lf = LidarFiltering()
    lidar_180 = create_empty_lidar()

    # Inicializa gráfico interativo
    plt.ion()
    plt.figure(figsize=(7,5))

    while True:
        user_input = input("> ")

        if user_input.lower() == 'q':
            break

        try:
            start, end, dist = user_input.split()
            start = int(start)
            end = int(end)
            dist = float(dist)

            # Adiciona novo obstáculo sem apagar os anteriores
            lidar_180 = add_obstacle(lidar_180, start, end, dist)

            # Expande para 360° para compatibilidade com LidarFiltering
            lidar_360 = lidar_180_to_360(lidar_180)

            # Calcula steer e target
            steer, target_angle = lf.compute_steer_from_lidar(lidar_360)

            print(f"Steer: {steer:.2f}° | Target angle: {target_angle:.2f}°")

            # Atualiza gráfico
            plot_lidar(lidar_180, target_angle, steer)

        except Exception as e:
            print("Erro:", e)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_simulation()
