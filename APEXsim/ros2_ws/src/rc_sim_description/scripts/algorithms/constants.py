import numpy as np

# LiDAR configuration defaults (used by sim and real readers)
LIDAR_BAUDRATE = 115200
LIDAR_HEADING_OFFSET_DEG = 0
LIDAR_FOV_FILTER = 360
LIDAR_POINT_TIMEOUT_MS = 200

# Vehicle hitbox (meters)
HITBOX_W = 0.20
HITBOX_H1 = 0.35
HITBOX_H2 = 0.15

# Corner avoidance tuning
AVOID_CORNER_MAX_ANGLE = 35
AVOID_CORNER_MIN_DISTANCE = 0.35
AVOID_CORNER_SCALE_FACTOR = 0.6

# LiDAR processing
FIELD_OF_VIEW_DEG = 180
CONVOLUTION_SIZE = 81

# Steering mapping (degrees -> degrees)
STEER_FACTOR = np.array(
    [
        [0.0, 0.0],
        [10.0, 5.0],
        [20.0, 10.0],
        [30.0, 15.0],
        [45.0, 20.0],
        [60.0, 25.0],
        [90.0, 30.0],
    ],
    dtype=float,
)

# PWM constants for real hardware (duty cycle %)
ESC_DC_MIN = 5.0
ESC_DC_MAX = 10.0
STEER_CENTER = 7.5
STEER_VARIATION_RATE = (10.0 - 5.0) / 60.0
