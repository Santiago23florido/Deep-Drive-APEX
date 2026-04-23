from rplidar import RPLidar

lidar = RPLidar('/dev/ttyUSB0', baudrate=115200)
lidar.connect()
lidar.start_motor()

try:
    for meas in lidar.iter_measures():
        print(meas)  # quality, angle, distance
        # if i > 10:
            # break
finally:
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()
