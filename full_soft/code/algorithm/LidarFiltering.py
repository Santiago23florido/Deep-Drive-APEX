import numpy as np
from scipy.signal import convolve
from code.log_manager import get_component_logger
from code.config_loader import get_config


class LidarFiltering:
    """
    This class provides methods to process raw LIDAR data, including:
    - calculating the hitbox in polar coordinates,
    - shrinking the free space based on the hitbox,
    - applying a convolution filter to smooth the LIDAR data,
    - computing target angles and steering commands.
    """
    def __init__(self, config_file="config.json"):
        self.config = get_config(config_file)
        self.logger = get_component_logger("LidarFiltering")
        

        # Parameters from config
        self.lecture_max_distance      = self.config.lidar.lecture_max_distance
        self.hitbox_width              = self.config.navigation.hitbox_w
        self.hitbox_h1                 = self.config.navigation.hitbox_h1
        self.hitbox_h2                 = self.config.navigation.hitbox_h2

        self.field_of_view_deg         = self.config.lidar.field_of_view_deg
        self.convolution_size          = self.config.lidar.convolution_size
        self.avoid_corner_max_angle    = self.config.navigation.avoid_corner_max_angle
        self.avoid_corner_min_distance = self.config.navigation.avoid_corner_min_distance
        self.avoid_corner_scale_factor = self.config.navigation.avoid_corner_scale_factor
        self.steering_limit            = self.config.steering.limit
        self.steer_factor              = np.array([
            [0.00, 0.000],
            [10.0, 0.167],
            [20.0, 0.360],
            [30.0, 0.680],
            [40.0, 0.900],
            [50.0, 1.000]
        ])
        self.steer_factor[:, 1] *= self.steering_limit

        # Precompute hitbox in polar coordinates
        self.hitbox = self.calculate_hitbox_polar(
            self.hitbox_width,
            self.hitbox_h1,
            self.hitbox_h2
        )

    # --------------------------
    # Core math utilities
    # --------------------------
    @staticmethod
    def convert_rad_to_xy(distance: np.ndarray, angle_rad: np.ndarray):
        """Converts polar coordinates to Cartesian coordinates."""
        y = distance * np.cos(angle_rad)
        x = distance * np.sin(angle_rad)
        return x, y

    def calculate_hitbox_polar(self, w, h1, h2):
        """Calculates polar coordinates of the rectangular hitbox."""
        theta = np.linspace(0, 2*np.pi, 360, endpoint=False)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        eps = 1e-9
        cos_t = np.where(np.abs(cos_t) < eps, eps, cos_t)
        sin_t = np.where(np.abs(sin_t) < eps, eps, sin_t)

        d_x = w / np.abs(cos_t)
        h = np.where(sin_t >= 0, h1, h2)
        d_y = h / np.abs(sin_t)
        d = np.minimum(d_x, d_y)
        return d

    # --------------------------
    # LIDAR processing
    # --------------------------
    def shrink_space(self, raw_lidar):
        """Shrink the free space by the vehicle hitbox."""
        mask = raw_lidar > 0
        shrinked = np.copy(raw_lidar)
        shrinked[mask] -= self.hitbox[mask]
        shrinked = np.maximum(shrinked, 0.0)
        return shrinked

    def convolution_filter(self, distances):
        """Smooth and emphasize front using convolution."""
        shift = self.field_of_view_deg // 2

        # Simple smoothing kernel
        kernel = np.ones(self.convolution_size) / self.convolution_size

        # Roll distances to center front
        distances = np.roll(distances, shift)

        # Convolve
        filtered = convolve(distances, kernel, mode="same")

        # Return only FOV portion
        return filtered[:self.field_of_view_deg], np.arange(self.field_of_view_deg)

    # --------------------------
    # Steering computation
    # --------------------------
    def compute_angle(self, filtered_distances, filtered_angles, raw_lidar):
        """Compute target angle with corner avoidance."""
        target_angle = filtered_angles[np.argmax(filtered_distances)]
        delta = 0
        l_angle = 0
        r_angle = 0

        for index in range(self.avoid_corner_max_angle, 0, -1):
            l_dist = raw_lidar[(int(target_angle) + index) % 360]
            r_dist = raw_lidar[(int(target_angle) - index) % 360]

            if l_angle == 0 and l_dist < self.avoid_corner_min_distance:
                l_angle = index
            if r_angle == 0 and r_dist < self.avoid_corner_min_distance:
                r_angle = index

        if l_angle == r_angle:
            delta = 0
        elif l_angle > r_angle:
            delta = -self.avoid_corner_scale_factor * (self.avoid_corner_max_angle - r_angle)
        elif l_angle < r_angle:
            delta = +self.avoid_corner_scale_factor * (self.avoid_corner_max_angle - l_angle)

        target_angle += delta
        target_angle = (target_angle + 180) % 360 - 180
        return target_angle, delta

    def lerp(self, value, factor):
        """Linear interpolation helper for steering factor table."""
        indices = np.nonzero(value < factor[:, 0])[0]
        if len(indices) == 0:
            return factor[-1, 1]
        index = indices[0]
        if index == 0:
            return factor[0, 1]
        delta = factor[index] - factor[index - 1]
        scale = (value - factor[index - 1, 0]) / delta[0]
        return float(factor[index - 1, 1] + scale * delta[1])

    def compute_steer(self, alpha):
        """Compute steering command from target angle."""
        return np.sign(alpha) * self.lerp(np.abs(alpha), self.steer_factor)

    # --------------------------
    # High-level interface
    # --------------------------
    def compute_steer_from_lidar(self, raw_lidar):
        """Main interface: compute steering from raw LiDAR."""
        shrinked = self.shrink_space(raw_lidar)
        filtered_distances, filtered_angles = self.convolution_filter(shrinked)
        target_angle, _ = self.compute_angle(filtered_distances, filtered_angles, raw_lidar)
        steer = self.compute_steer(target_angle)
        return steer, target_angle

    def compute_angle_weighted(self, raw_lidar):
        """Simpler forward-weighted approach."""
        angles = np.deg2rad(np.arange(360))
        forward_weight = np.cos(angles)
        score = raw_lidar * forward_weight
        best_angle = np.argmax(score)
        return (best_angle + 180) % 360 - 180
