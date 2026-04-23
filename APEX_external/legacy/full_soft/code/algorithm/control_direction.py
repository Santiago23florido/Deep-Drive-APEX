from algorithm.constants import *
from scipy.signal import convolve

def get_nonzero_points_in_hitbox(distances):
    if distances is None:
        raise ValueError("Error: LiDAR input is None.")

    x, y = convert_rad_to_xy(distances, np.arange(0, 360))

    mask_hitbox =  (y > 0) & (np.abs(y) <= HITBOX_H1) & (np.abs(x) <= HITBOX_W) & (y * x != 0.0)

    return x[mask_hitbox], y[mask_hitbox]

@staticmethod
def convert_rad_to_xy(distance, angle_rad):
    y = distance * np.cos(angle_rad)
    x = distance * np.sin(angle_rad)
    return x, y


def calculate_hitbox_polar(w, h1, h2):
    """
    Calculate the polar coordinate representation of a rectangular hitbox.
    This function computes the distance from the origin to the boundary of a 
    rectangle for each angle in a 360-degree range, effectively creating a 
    polar representation of the rectangular hitbox.
    Parameters
    ----------
    w : float
        Half-width of the rectangle (extends from -w to +w on the x-axis).
    h1 : float
        Height above the origin (extends from 0 to +h1 on the y-axis).
    h2 : float
        Height below the origin (extends from 0 to -h2 on the y-axis).
    Returns
    -------
    np.ndarray
        Array of shape (360,) containing distances from origin to hitbox boundary
        for each degree (0-359). Indexed such that index i corresponds to angle
        (i * 2π / 360) radians, with angles offset by -π/2 radians.
    Notes
    -----
    - Angles are sampled at 1-degree intervals (360 total samples).
    - For each angle, the function finds the intersection of a ray from the 
      origin with the rectangle's edges.
    - The function handles edge cases where cos(θ) or sin(θ) are near zero.
    - If a ray doesn't intersect the rectangle, the distance is set to 0.
    """
    rad_raw_angles = np.linspace(0, 2*np.pi, num=360, endpoint=False)
    
    polar_coords = []
    polar_angles = []
    
    for theta in rad_raw_angles:
        c = np.cos(theta)
        s = np.sin(theta)
        
        candidates = []
        
        if abs(c) > 1e-14:
            x_side = w if c > 0 else -w
            t_x = x_side / c
            if t_x >= 0:
                y_at_x = s * t_x
                if -h2 <= y_at_x <= h1:
                    candidates.append(t_x)
        
        if abs(s) > 1e-14:
            y_side = h1 if s > 0 else -h2
            t_y = y_side / s
            if t_y >= 0:
                x_at_y = c * t_y
                
                if -w <= x_at_y <= w:
                    candidates.append(t_y)
        
        if len(candidates) == 0:
            d = 0.0
        else:
            t = min(candidates)
            d = t
        
        polar_coords.append(d)
        polar_angles.append(theta - np.pi/2)
    
    angle_indices = np.round(
        np.array(polar_angles) / (2 * np.pi / 360)
    ).astype(int) % 360
    
    new_d_linha = np.zeros_like(polar_coords)
    new_d_linha[angle_indices] = polar_coords
    
    return new_d_linha

hitbox = calculate_hitbox_polar(HITBOX_W, HITBOX_H1, HITBOX_H2)

def shrink_space(raw_lidar):
    free_space_shrink_mask = raw_lidar > 0
    shrink_space_lidar = np.copy(raw_lidar)
    shrink_space_lidar[free_space_shrink_mask] -= hitbox[free_space_shrink_mask]
    
    return shrink_space_lidar

def compute_steer_from_lidar(raw_lidar):    
    filtreed_distances, filtreed_angles = convolution_filter(raw_lidar)
    target, _ = compute_angle(filtreed_distances, filtreed_angles, raw_lidar)
    steer = compute_steer(target)

    return steer, target

def compute_angle(filtred_distances, filtred_angles, raw_lidar):
    target_angle = filtred_angles[np.argmax(filtred_distances)]
    delta = 0

    l_angle = 0
    r_angle = 0
    
    for index in range(AVOID_CORNER_MAX_ANGLE, 0, -1):
        l_dist = raw_lidar[(target_angle + index) % 360]
        r_dist = raw_lidar[(target_angle - index) % 360]

        if l_angle == 0 and l_dist < AVOID_CORNER_MIN_DISTANCE:
            l_angle = index

        if r_angle == 0 and r_dist < AVOID_CORNER_MIN_DISTANCE:
            r_angle = index
    
    if l_angle == r_angle:
        delta = 0
    elif l_angle > r_angle:
        delta = -AVOID_CORNER_SCALE_FACTOR * (AVOID_CORNER_MAX_ANGLE - r_angle)
    elif l_angle < r_angle:
        delta = +AVOID_CORNER_SCALE_FACTOR * (AVOID_CORNER_MAX_ANGLE - l_angle)

    #print("delta = ", delta)
    
    target_angle += delta
    
    target_angle = (target_angle + 180) % 360 - 180
    
    return target_angle, delta

def compute_steer(alpha):
    return np.sign(alpha) * lerp(np.abs(alpha), STEER_FACTOR)

def convolution_filter2(distances):
    shift = FIELD_OF_VIEW_DEG // 2
    distances = np.roll(distances, shift)
    angles = np.arange(0,360)
    angles = np.roll(angles, shift)
 
    kernel = np.ones(CONVOLUTION_SIZE) / CONVOLUTION_SIZE
    distances = convolve(distances, kernel, mode="same")
    return distances[:FIELD_OF_VIEW_DEG], angles[:FIELD_OF_VIEW_DEG]

def convolution_filter(distances):

    # Modified convolution filter in the last test
    return convolution_filter2(distances)

def lerp(value: float, factor: np.ndarray) -> np.ndarray:
    indices = np.nonzero(value < factor[:, 0])[0]

    if len(indices) == 0:
        return factor[-1, 1]

    index = indices[0]

    delta = factor[index] - factor[index - 1]
    scale = (value - factor[index - 1, 0]) / delta[0]

    return factor[index - 1, 1] + scale * delta[1]

