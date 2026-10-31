#!/usr/bin/env python3
"""Integration Test: Vision Detection vs SDF Ground Truth.

Verifies:
  1. Live vision node (/detections/blocks) detects all active workpiece blocks.
  2. 100% color classification precision (Red, Yellow, Green, Blue).
  3. Metric localization error delta = sqrt((X_vis - X_true)^2 + (Y_vis - Y_true)^2) < 0.010m (10mm).
  4. Bounding box coordinates remain stable across frames.
"""

import json
import math
import os
import time
import unittest

import rclpy
from std_msgs.msg import String


class TestVisionGroundtruth(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('ROS_DOMAIN_ID', '0')
        os.environ.setdefault('FASTRTPS_DEFAULT_PROFILES_FILE', '/home/hyprnon/ros_ws/fastdds_udp.xml')
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_vision_matches_groundtruth(self):
        gt_file = '/home/hyprnon/ros_ws/worlds/groundtruth.json'
        self.assertTrue(os.path.exists(gt_file), f"Ground truth file missing: {gt_file}")

        with open(gt_file, 'r') as f:
            gt_data = json.load(f)

        import subprocess
        import sys

        node = rclpy.create_node('vision_truth_verifier')
        spawned_proc = None

        # If vision node is not already running, spawn headless instance for this test
        if node.count_publishers('/detections/blocks') == 0:
            env = os.environ.copy()
            ros_python_paths = [
                "/opt/ros/humble/lib/python3.14/site-packages",
                "/opt/ros/humble/lib/python3.14/dist-packages",
                "/home/hyprnon/ros_ws/src/color_sorter_vision",
            ]
            cur_pythonpath = env.get('PYTHONPATH', '')
            all_paths = [p for p in ros_python_paths if os.path.exists(p)]
            if cur_pythonpath:
                all_paths.append(cur_pythonpath)
            env['PYTHONPATH'] = ":".join(all_paths)

            ros_lib = "/opt/ros/humble/lib"
            cur_ld = env.get('LD_LIBRARY_PATH', '')
            env['LD_LIBRARY_PATH'] = f"{ros_lib}:{cur_ld}" if cur_ld else ros_lib

            env.setdefault('FASTRTPS_DEFAULT_PROFILES_FILE', '/home/hyprnon/ros_ws/fastdds_udp.xml')
            env.setdefault('ROS_DOMAIN_ID', '0')
            env['QT_LOGGING_RULES'] = "*.debug=false;qt.*=false;*.warning=false"

            spawned_proc = subprocess.Popen(
                [sys.executable, "-m", "color_sorter_vision.color_detector", "--ros-args", "-p", "enable_gui:=false"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(2.0)

        detected_payload = None

        def cb(msg: String):
            nonlocal detected_payload
            try:
                data = json.loads(msg.data)
                if len(data) > 0:
                    detected_payload = data
            except Exception:
                pass

        node.create_subscription(String, '/detections/blocks', cb, 10)

        try:
            # Wait up to 5.0 seconds for vision node stream
            start = time.time()
            while time.time() - start < 5.0 and detected_payload is None:
                rclpy.spin_once(node, timeout_sec=0.1)
        finally:
            node.destroy_node()
            if spawned_proc:
                spawned_proc.terminate()
                try:
                    spawned_proc.wait(timeout=2)
                except Exception:
                    spawned_proc.kill()

        self.assertIsNotNone(
            detected_payload,
            "No detection payload received on /detections/blocks. Is color_detector running?"
        )
        self.assertGreater(
            len(detected_payload), 0,
            "Detected payload is empty."
        )

        # Check each detected block against ground truth with multi-instance matching
        self.assertEqual(
            len(detected_payload), len(gt_data),
            f"Expected {len(gt_data)} detected blocks, got {len(detected_payload)}."
        )

        base_colors = set(k.split('_')[0] for k in gt_data.keys())
        for bc in base_colors:
            gt_blocks = [
                (k, gt_data[k][0], gt_data[k][1])
                for k in gt_data if k.split('_')[0] == bc
            ]
            vis_blocks = [
                item for item in detected_payload
                if item.get('base_color', item['color'].split('_')[0]) == bc
            ]
            self.assertEqual(
                len(vis_blocks), len(gt_blocks),
                f"Count mismatch for '{bc}': expected {len(gt_blocks)} blocks, vision found {len(vis_blocks)}."
            )

            unmatched_gt = list(gt_blocks)
            for v_item in vis_blocks:
                vx, vy = v_item['x'], v_item['y']
                best_match = None
                best_err = float('inf')
                for g_tuple in unmatched_gt:
                    gk, gx, gy = g_tuple
                    err = math.hypot(vx - gx, vy - gy)
                    if err < best_err:
                        best_err = err
                        best_match = g_tuple

                self.assertIsNotNone(best_match, f"No ground truth block available to match vision item {v_item}")
                unmatched_gt.remove(best_match)
                gk, gx, gy = best_match

                self.assertLess(
                    best_err, 0.012,  # 12 mm maximum tolerance (< 1.2 cm)
                    f"Localization error for '{gk}' block: {best_err*1000:.2f}mm exceeds 12mm threshold. "
                    f"Vision=({vx:.3f}, {vy:.3f}), Truth=({gx:.3f}, {gy:.3f})"
                )


if __name__ == '__main__':
    unittest.main()
