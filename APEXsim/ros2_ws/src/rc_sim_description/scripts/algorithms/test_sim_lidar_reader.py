#!/usr/bin/env python3
import time
import numpy as np
from sim_lidar_reader import SimLidarReader

if __name__ == '__main__':
    lidar = SimLidarReader(
        topic='/lidar_processed',
        heading_offset_deg=0,
        fov_filter=360,
        point_timeout_ms=200,
        noise_sigma=0.0,
        dropout_prob=0.0,
        bias_m=0.0,
        quantization_m=0.0,
    )

    try:
        while True:
            data = lidar.get_lidar_data()
            nonzero_count = np.count_nonzero(data)
            print(f'nonzero_count={nonzero_count}')
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        lidar.stop()
