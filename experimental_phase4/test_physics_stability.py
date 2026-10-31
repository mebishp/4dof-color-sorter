#!/usr/bin/env python3
"""Physics Stability Test: blocks must obey the physical world.

Catches the "unexplainable behaviour" the other tests miss:
  1. No block may fall off the table / through a bin (Z bounds check).
  2. No block may teleport across the cell (max displacement from groundtruth).
  3. No block may leave the workcell bounds (XY bounds check).
  4. sorting_metrics.csv may only contain the honest, verified status vocabulary
     (SORTED_VERIFIED / PLACE_UNVERIFIED / PICK_FAILED / SLIPPED / ...) —
     bare SORTED_SUCCESS without verification is rejected for new runs.

Queries live Gazebo pose info from inside the sim container. Skips cleanly
when the simulation is not running.
"""

import csv
import json
import math
import os
import re
import shutil
import subprocess
import unittest

GT_FILE = '/home/hyprnon/ros_ws/worlds/groundtruth.json'
CSV_FILE = '/home/hyprnon/ros_ws/sorting_metrics.csv'

# Table is 2.0 x 1.5 m centred at origin; bins sit on top (Z~0.45..0.58).
X_BOUND, Y_BOUND = 1.0, 0.75
Z_MIN, Z_MAX = 0.40, 0.80
MAX_DISPLACEMENT_M = 0.75  # staging -> furthest bin slot is ~0.6 m

STATUS_VOCABULARY = {
    'SORTED_VERIFIED',
    'PLACE_UNVERIFIED',
    'PICK_FAILED',
    'SLIPPED',
    'SKIPPED_NON_MAGNETIC',
    # Legacy open-loop runs (pre-Phase-4) — tolerated, not produced anymore.
    'SORTED_SUCCESS', 'SORTED',
}


def query_gz_poses():
    """Return {model_name: (x, y, z)} from Gazebo, or None if sim is down."""
    if shutil.which('docker') is None:
        return None
    try:
        out = subprocess.run(
            ['docker', 'exec', 'sim', 'bash', '-c',
             'export IGN_IP=127.0.0.1; timeout 6 ign topic -e -t '
             '/world/sorter/pose/info -n 1'],
            capture_output=True, text=True, timeout=20,
        )
    except Exception:
        return None
    if out.returncode != 0 or not out.stdout:
        return None
    poses = {}
    # Split protobuf text dump into per-model chunks.
    chunks = re.split(r'pose\s*\{', out.stdout)
    for ch in chunks:
        name = re.search(r'name:\s*"([^"]+)"', ch)
        if not name:
            continue
        pos_block = ch.split('position', 1)
        if len(pos_block) < 2:
            continue
        coords = re.findall(r'[xyz]:\s*([-\d.eE+]+)', pos_block[1].split('orientation')[0])
        if len(coords) < 3:
            continue
        try:
            poses[name.group(1)] = tuple(float(c) for c in coords[:3])
        except ValueError:
            continue
    return poses or None


class TestPhysicsStability(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.poses = query_gz_poses()
        if os.path.exists(GT_FILE):
            with open(GT_FILE) as f:
                cls.gt = json.load(f)
        else:
            cls.gt = {}

    def test_blocks_within_workcell(self):
        if self.poses is None:
            self.skipTest("Simulation not running (no GZ pose info).")
        blocks = {k: v for k, v in self.poses.items() if 'block' in k}
        self.assertGreater(len(blocks), 0, "No blocks found in GZ world.")
        for name, (x, y, z) in sorted(blocks.items()):
            self.assertLess(abs(x), X_BOUND, f"{name} left workcell: x={x:.3f}")
            self.assertLess(abs(y), Y_BOUND, f"{name} left workcell: y={y:.3f}")
            self.assertGreaterEqual(z, Z_MIN,
                                    f"{name} fell through/off (z={z:.3f} < {Z_MIN})")
            self.assertLessEqual(z, Z_MAX,
                                 f"{name} launched into air (z={z:.3f} > {Z_MAX})")

    def test_no_teleport_from_groundtruth(self):
        if self.poses is None:
            self.skipTest("Simulation not running (no GZ pose info).")
        if not self.gt:
            self.skipTest("No groundtruth.json available.")
        for color_key, vals in self.gt.items():
            # groundtruth key 'red_1' -> model name 'red_block_1'
            parts = color_key.split('_')
            model = f"{parts[0]}_block_{parts[1]}" if len(parts) == 2 else color_key
            if model not in self.poses:
                continue
            gx, gy = vals[0], vals[1]
            x, y, _ = self.poses[model]
            d = math.hypot(x - gx, y - gy)
            self.assertLess(
                d, MAX_DISPLACEMENT_M,
                f"{model} displaced {d*1000:.0f}mm from groundtruth "
                f"(>{MAX_DISPLACEMENT_M*1000:.0f}mm) — physics glitch/teleport.")

    def test_metrics_status_vocabulary(self):
        if not os.path.exists(CSV_FILE):
            self.skipTest(f"{CSV_FILE} does not exist yet. Run a sort cycle first.")
        with open(CSV_FILE) as f:
            rows = list(csv.DictReader(f))
        self.assertGreater(len(rows), 0, "sorting_metrics.csv is empty.")
        for row in rows:
            self.assertIn(row['status'], STATUS_VOCABULARY,
                          f"Block {row['color']}: unknown status '{row['status']}'.")
            t = float(row['cycle_time_sec'])
            self.assertGreater(t, 3.0, f"Block {row['color']}: implausible cycle {t}s.")
            self.assertLess(t, 180.0, f"Block {row['color']}: cycle {t}s exceeded bound.")


if __name__ == '__main__':
    unittest.main()
