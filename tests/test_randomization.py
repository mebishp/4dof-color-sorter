#!/usr/bin/env python3
"""Unit Test: Procedural Workpiece Randomizer.

Verifies, for layouts of 1..12 blocks:
  1. Every pose is valid staging (clear of bins and pedestal, reachable at contact + transit).
  2. No two blocks overlap, accounting for yaw (oriented-square test with clearance).
  3. Block keys cycle colors and never exceed bin capacity.
  4. Same seed -> same layout.
"""

import sys
import unittest

sys.path.insert(0, '/home/hyprnon/ros_ws')
from randomize_world import blocks_overlap, sample_collision_free_poses  # noqa: E402
from color_sorter_vision.workcell import (  # noqa: E402
    BIN_CAPACITY, MAX_BLOCKS, base_color, block_keys, staging_pose_valid,
)


class TestRandomization(unittest.TestCase):

    def _assert_layout_valid(self, poses, label):
        for i, (x, y, _) in enumerate(poses):
            self.assertTrue(staging_pose_valid(x, y), f"{label}: block {i} at ({x}, {y}) is not valid staging")
        for i in range(len(poses)):
            for j in range(i + 1, len(poses)):
                self.assertFalse(blocks_overlap(poses[i], poses[j], clearance=0.0),
                                 f"{label}: blocks {i} and {j} overlap: {poses[i]} {poses[j]}")

    def test_monte_carlo_four_blocks(self):
        for trial in range(50):
            poses = sample_collision_free_poses(4)
            self.assertEqual(len(poses), 4)
            self._assert_layout_valid(poses, f"trial {trial}")

    def test_every_supported_count(self):
        for n in range(1, MAX_BLOCKS + 1):
            for seed in range(5):
                poses = sample_collision_free_poses(n, seed=seed)
                self.assertEqual(len(poses), n)
                self._assert_layout_valid(poses, f"n={n} seed={seed}")

    def test_overlap_detects_yawed_corners(self):
        # 0.15 m apart: axis-aligned clears, but two 45-degree cubes (diagonal 0.17 m) collide.
        self.assertFalse(blocks_overlap((0, 0, 0), (0.15, 0, 0), clearance=0.0))
        self.assertTrue(blocks_overlap((0, 0, 0.785), (0.15, 0, 0.785), clearance=0.0))

    def test_block_keys_respect_bin_capacity(self):
        keys = block_keys(MAX_BLOCKS)
        self.assertEqual(keys[:4], ['red_1', 'yellow_1', 'green_1', 'blue_1'])
        for color in ('red', 'yellow', 'green', 'blue'):
            self.assertLessEqual(sum(base_color(k) == color for k in keys), BIN_CAPACITY)
        with self.assertRaises(ValueError):
            block_keys(MAX_BLOCKS + 1)

    def test_seed_reproducibility(self):
        self.assertEqual(sample_collision_free_poses(8, seed=42), sample_collision_free_poses(8, seed=42))
        self.assertNotEqual(sample_collision_free_poses(4, seed=42), sample_collision_free_poses(4, seed=999))


if __name__ == '__main__':
    unittest.main()
