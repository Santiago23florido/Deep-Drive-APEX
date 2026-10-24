"""Actual ROS callbacks must accept a current pose while scan planning is busy."""
import os
import threading
import time

import pytest

rclpy = pytest.importorskip('rclpy')
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rclpy.executors import MultiThreadedExecutor
from apex_race.nodes.race_driver_node import RaceDriverNode


def test_prediction_is_received_before_a_slow_scan_releases_the_core():
    entered, release = threading.Event(), threading.Event()

    class SlowScanNode(RaceDriverNode):
        def _on_scan(self, message):
            with self._core('scan'):
                entered.set()
                release.wait(timeout=3)

    # Private command/arm/input topics: this test cannot arm an actual car or
    # inject a second command publisher into its actuator stream.
    prefix = '/apex_pose_schedule_test_' + str(os.getpid())
    params = {'planner_in_process': 'true', 'require_arm': 'true'}
    for name in ('cmd_topic', 'arm_topic', 'predicted_topic', 'odom_topic', 'scan_topic',
                 'estimator_status_topic', 'status_topic', 'map_topic', 'graph_topic'):
        params[name] = prefix + '/' + name
    args = ['--ros-args', '-r', '__ns:=' + prefix]
    for key, value in params.items():
        args += ['-p', key + ':=' + value]
    rclpy.init(args=args)
    node = SlowScanNode()
    feeder = rclpy.create_node('pose_feeder', namespace=prefix)
    pose_pub = feeder.create_publisher(Odometry, params['predicted_topic'], 1)
    scan_pub = feeder.create_publisher(LaserScan, params['scan_topic'], 1)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    executor.add_node(feeder)
    worker = threading.Thread(target=executor.spin, daemon=True)
    worker.start()
    try:
        deadline = time.monotonic() + 4
        while (pose_pub.get_subscription_count() == 0 or scan_pub.get_subscription_count() == 0) and time.monotonic() < deadline:
            time.sleep(.02)
        assert pose_pub.get_subscription_count() and scan_pub.get_subscription_count()
        scan_pub.publish(LaserScan())
        assert entered.wait(timeout=1)
        assert node._core_lock.locked()
        for i in range(8):
            message = Odometry()
            message.header.stamp = feeder.get_clock().now().to_msg()
            message.pose.pose.position.x = float(i)
            message.pose.pose.orientation.w = 1.0
            pose_pub.publish(message)
            time.sleep(.02)
        deadline = time.monotonic() + .5
        while (node.pred is None or node.pred[1][0] != 7) and time.monotonic() < deadline:
            time.sleep(.01)
        assert not release.is_set() and node._core_lock.locked()
        assert node.pred is not None and node.pred[1][0] == 7
        predicted_sub = next(s for s in node.subscriptions if s.topic_name == params['predicted_topic'])
        scan_sub = next(s for s in node.subscriptions if s.topic_name == params['scan_topic'])
        assert predicted_sub.callback_group is not scan_sub.callback_group
        assert predicted_sub.qos_profile.depth == 1
    finally:
        release.set()
        executor.shutdown(timeout_sec=3)
        worker.join(timeout=3)
        node.destroy_node()
        feeder.destroy_node()
        rclpy.try_shutdown()
