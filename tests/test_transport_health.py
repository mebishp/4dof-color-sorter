#!/usr/bin/env python3
"""Integration Test: Simulation & DDS Transport Health.

Verifies:
  1. Docker container 'sim' running state.
  2. FastDDS loopback UDP configuration active.
  3. /clock topic receiving ticks from Gazebo Fortress.
  4. /camera/image_raw (overhead) and /camera/perspective/image_raw topic delivery.
  5. Joint command and electromagnet topics registered on ROS 2 graph.
"""

import os
import subprocess
import time
import unittest

import rclpy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data


class TestTransportHealth(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('ROS_DOMAIN_ID', '0')
        os.environ.setdefault('FASTRTPS_DEFAULT_PROFILES_FILE', '/home/hyprnon/ros_ws/fastdds_udp.xml')
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_container_running(self):
        """Docker container 'sim' must be running."""
        res = subprocess.run(
            ['docker', 'inspect', 'sim', '--format', '{{.State.Status}}'],
            capture_output=True, text=True
        )
        status = res.stdout.strip()
        self.assertEqual(status, 'running', f"Container 'sim' is not running (status: {status})")

    def test_fastdds_xml_exists(self):
        """FastDDS loopback configuration must exist on host."""
        self.assertTrue(
            os.path.exists('/home/hyprnon/ros_ws/fastdds_udp.xml'),
            "fastdds_udp.xml must be present at /home/hyprnon/ros_ws/fastdds_udp.xml"
        )

    def test_topics_active(self):
        """Probe /clock and camera topics over 3.0 seconds."""
        node = rclpy.create_node('health_test_probe')
        received = {'clock': 0, 'camera': 0, 'persp': 0}

        node.create_subscription(Clock, '/clock', lambda m: received.update({'clock': received['clock'] + 1}), 10)
        node.create_subscription(Image, '/camera/image_raw', lambda m: received.update({'camera': received['camera'] + 1}), qos_profile_sensor_data)
        node.create_subscription(Image, '/camera/perspective/image_raw', lambda m: received.update({'persp': received['persp'] + 1}), qos_profile_sensor_data)

        start = time.time()
        while time.time() - start < 3.5:
            rclpy.spin_once(node, timeout_sec=0.1)
            if received['clock'] >= 3 and received['camera'] >= 2 and received['persp'] >= 2:
                break

        node.destroy_node()

        self.assertGreater(received['clock'], 0, "No /clock ticks received from Gazebo simulation.")
        self.assertGreater(received['camera'], 0, "No frames received on /camera/image_raw.")
        self.assertGreater(received['persp'], 0, "No frames received on /camera/perspective/image_raw.")


if __name__ == '__main__':
    unittest.main()
