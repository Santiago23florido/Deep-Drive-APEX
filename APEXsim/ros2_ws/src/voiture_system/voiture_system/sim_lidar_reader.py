#!/usr/bin/env python3
import math
import time
import numpy as np
import multiprocessing as mp

try:
    from algorithm.interfaces import LiDarInterface
    import algorithm.voiture_logger as cl
except Exception:
    try:
        from interfaces import LiDarInterface
        import voiture_logger as cl
    except Exception:
        class LiDarInterface:  # Fallback if interface is not importable
            pass
        class _DummyLogger:
            def info(self, *_args, **_kwargs):
                pass
        class _DummyCentralLogger:
            def __init__(self, *args, **kwargs):
                pass
            def get_logger(self):
                return _DummyLogger()
        cl = type('cl', (), {'CentralLogger': _DummyCentralLogger})

import rclpy
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


def _run_ros_process(
    topic,
    node_name,
    use_sim_time,
    heading_offset_deg,
    fov_filter,
    point_timeout_ms,
    noise_sigma,
    dropout_prob,
    bias_m,
    quantization_m,
    seed,
    last_lidar_read,
    last_lidar_update,
    stop_event,
    log_scans,
):
    rclpy.init(args=None)
    node = rclpy.create_node(node_name)
    if use_sim_time:
        node.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])

    rng = np.random.default_rng(seed)

    pre_filtered_distances = np.zeros(360, dtype=float)
    last_update_times = np.zeros(360, dtype=float)

    def apply_noise_and_quantize(value_m):
        if bias_m != 0.0:
            value_m += bias_m
        if noise_sigma > 0.0:
            value_m += rng.normal(0.0, noise_sigma)
        if quantization_m > 0.0:
            value_m = round(value_m / quantization_m) * quantization_m
        return value_m

    def scan_cb(msg: LaserScan):
        nonlocal pre_filtered_distances, last_update_times

        current_ms = time.time() * 1000.0
        angle = msg.angle_min

        for r in msg.ranges:
            idx = int(round(math.degrees(angle))) % 360
            angle += msg.angle_increment

            if not math.isfinite(r):
                continue
            if r < msg.range_min or r > msg.range_max:
                continue
            if dropout_prob > 0.0 and rng.random() < dropout_prob:
                continue

            r = apply_noise_and_quantize(r)
            if not math.isfinite(r):
                continue

            if r < msg.range_min:
                r = msg.range_min
            if r > msg.range_max:
                r = msg.range_max

            pre_filtered_distances[idx] = r
            last_update_times[idx] = current_ms

        shifted_distances = np.roll(pre_filtered_distances, int(heading_offset_deg))

        # Fill zeros by propagating previous value (same as real reader)
        for i in range(1, 360):
            if shifted_distances[i] == 0.0:
                shifted_distances[i] = shifted_distances[i - 1]

        # FOV filter centered at 0 deg
        if fov_filter is not None and fov_filter < 360:
            half_fov = fov_filter / 2.0
            angle_array = np.arange(360)
            diffs = (angle_array - 0) % 360
            keep_mask = (diffs <= half_fov) | (diffs >= 360 - half_fov)
            shifted_distances[~keep_mask] = 0.0

        # Timeout per angle
        if point_timeout_ms is not None and point_timeout_ms > 0:
            time_diffs = current_ms - last_update_times
            expired_mask = (time_diffs > point_timeout_ms) & (last_update_times > 0)
            shifted_distances[expired_mask] = 0.0
            last_update_times[expired_mask] = -1.0

        if log_scans:
            node.get_logger().info(str(shifted_distances.tolist()))

        with last_lidar_read.get_lock():
            for i in range(360):
                last_lidar_read[i] = shifted_distances[i]

        with last_lidar_update.get_lock():
            last_lidar_update.value = time.time()

    node.create_subscription(LaserScan, topic, scan_cb, qos_profile_sensor_data)

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    try:
        while rclpy.ok() and not stop_event.is_set():
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


class SimLidarReader(LiDarInterface):
    last_lidar_read = mp.Array('d', 360)
    last_lidar_update = mp.Value('d', 0.0)
    stop_event = mp.Event()

    def __init__(
        self,
        topic='/scan',
        heading_offset_deg=0,
        fov_filter=360,
        point_timeout_ms=200,
        noise_sigma=0.0,
        dropout_prob=0.0,
        bias_m=0.0,
        quantization_m=0.0,
        use_sim_time=True,
        seed=None,
        sensor_name='SimLidar',
        log_scans=False,
        node_name='sim_lidar_reader',
    ):
        self.topic = topic
        self.heading_offset_deg = heading_offset_deg
        self.fov_filter = fov_filter
        self.point_timeout_ms = point_timeout_ms

        self.noise_sigma = noise_sigma
        self.dropout_prob = dropout_prob
        self.bias_m = bias_m
        self.quantization_m = quantization_m
        self.use_sim_time = use_sim_time
        self.seed = seed
        self.log_scans = log_scans
        self.node_name = node_name

        self.last_lidar_read = mp.Array('d', 360)
        self.last_lidar_update = mp.Value('d', 0.0)
        self.stop_event = mp.Event()

        self.sensor_logger_instance = cl.CentralLogger(sensor_name=sensor_name)
        self.sensor_logger = self.sensor_logger_instance.get_logger()

        self._lidar_process = None
        self._start_lidar_process()

    def _start_lidar_process(self):
        if self._lidar_process is not None and self._lidar_process.is_alive():
            self.sensor_logger.info('[SimLidarReader] Process already running.')
            return

        self._lidar_process = mp.Process(
            target=_run_ros_process,
            args=(
                self.topic,
                self.node_name,
                self.use_sim_time,
                self.heading_offset_deg,
                self.fov_filter,
                self.point_timeout_ms,
                self.noise_sigma,
                self.dropout_prob,
                self.bias_m,
                self.quantization_m,
                self.seed,
                self.last_lidar_read,
                self.last_lidar_update,
                self.stop_event,
                self.log_scans,
            ),
            daemon=True,
        )
        self._lidar_process.start()
        self.sensor_logger.info('[SimLidarReader] ROS2 process started.')

    def stop(self):
        self.sensor_logger.info('[SimLidarReader] Stopping...')
        self.stop_event.set()

        if self._lidar_process and self._lidar_process.is_alive():
            self._lidar_process.join(timeout=2.0)

        self.sensor_logger.info('[SimLidarReader] Stopped.')

    def get_lidar_data(self) -> np.ndarray:
        data_copy = np.zeros(360, dtype=float)
        with self.last_lidar_read.get_lock():
            for i in range(360):
                data_copy[i] = self.last_lidar_read[i]
        return data_copy
