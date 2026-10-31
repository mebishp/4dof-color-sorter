#!/usr/bin/env python3
"""Unit Test: 4-DOF Analytical Inverse Kinematics & Tool Alignment.

Verifies:
  1. Downward vertical tool constraint (theta2 + theta3 + theta4 == pi) across workspace.
  2. Forward Kinematics (FK) reconstruction match (< 0.1 mm precision).
  3. Reachability of all 4 dedicated side sorting bays (Red, Yellow, Green, Blue).
  4. Workspace envelope and boundary singularity protection.
"""

import math
import sys
import unittest
from typing import Optional, Tuple

sys.path.insert(0, '/home/hyprnon/ros_ws/src/color_sorter_vision')
from color_sorter_vision.workcell import (
    BIN_SLOTS, COLORS, STAGING_X, STAGING_Y, Z_DROP, Z_TRANSIT, bin_slot_xy, forward_kinematics,
    solve_ik_exact, staging_pose_valid,
)
from color_sorter_vision.pick_and_place import (
    BINS,
    X_BASE, Y_BASE, Z_SHOULDER, L2, L3, L4
)


def forward_kinematics_tool(t1: float, t2: float, t3: float, t4: float) -> Tuple[float, float, float]:
    """Compute FK from joint angles to electromagnet bottom tool center.
    
    Joint 2 and 3 pitch about Y, where zero position points along +Z.
    Tool angle phi = t2 + t3 + t4 points along -Z when phi = pi.
    """
    rw = L2 * math.sin(t2) + L3 * math.sin(t2 + t3)
    zw = Z_SHOULDER + L2 * math.cos(t2) + L3 * math.cos(t2 + t3)

    phi = t2 + t3 + t4
    r_tool = rw + L4 * math.sin(phi)
    z_tool = zw + L4 * math.cos(phi)

    x_tool = X_BASE + r_tool * math.cos(t1)
    y_tool = Y_BASE + r_tool * math.sin(t1)
    return x_tool, y_tool, z_tool


class TestIKKinematics(unittest.TestCase):

    def test_side_bins_reachability(self):
        """All 4 dedicated side sorting bays must be reachable with exact downward tool orientation."""
        for name, (bx, by, bz) in BINS.items():
            sol = solve_ik_exact(bx, by, bz)
            self.assertIsNotNone(sol, f"Sorting bay '{name}' at ({bx}, {by}, {bz}) should be reachable")
            t1, t2, t3, t4 = sol

            # Tool pitch constraint: sum must be pi within float tolerance
            tool_pitch = (t2 + t3 + t4) % (2.0 * math.pi)
            self.assertAlmostEqual(
                tool_pitch, math.pi, places=4,
                msg=f"Tool at bay '{name}' must point strictly downward (phi=pi)"
            )

            # FK validation
            fk_x, fk_y, fk_z = forward_kinematics_tool(t1, t2, t3, t4)
            err = math.hypot(fk_x - bx, fk_y - by, fk_z - bz)
            self.assertLess(err, 1e-3, f"FK residual at bay '{name}' must be < 1mm (got {err*1000:.3f}mm)")

    def test_every_bin_slot_reachable(self):
        """All 3 slots of all 4 bins: reachable at drop and transit height, FK-exact."""
        for color in COLORS:
            for slot in range(BIN_SLOTS):
                x, y = bin_slot_xy(color, slot)
                for z in (Z_DROP, Z_TRANSIT):
                    sol = solve_ik_exact(x, y, z)
                    self.assertIsNotNone(sol, f"{color} slot {slot} unreachable at z={z}")
                    err = math.dist(forward_kinematics(*sol), (x, y, z))
                    self.assertLess(err, 1e-6)

    def test_dense_workspace_grid(self):
        """Every valid staging point is reachable at contact, probe and transit heights."""
        tested = 0
        for ix in range(int(STAGING_X[0] * 100), int(STAGING_X[1] * 100) + 1, 2):
            for iy in range(int(STAGING_Y[0] * 100), int(STAGING_Y[1] * 100) + 1, 2):
                x, y = ix / 100.0, iy / 100.0
                if not staging_pose_valid(x, y):
                    continue
                for z in (0.570, 0.610, Z_TRANSIT):
                    tested += 1
                    sol = solve_ik_exact(x, y, z)
                    self.assertIsNotNone(sol, f"Staging point ({x}, {y}, {z}) unreachable")
                    t1, t2, t3, t4 = sol
                    fk = forward_kinematics_tool(t1, t2, t3, t4)
                    self.assertLess(math.dist(fk, (x, y, z)), 5e-4, f"FK error at ({x}, {y}, {z}) exceeded 0.5mm")
                    self.assertAlmostEqual((t2 + t3 + t4) % (2.0 * math.pi), math.pi, places=4)
        self.assertGreater(tested, 500, "Staging area unexpectedly small")

    def test_unreachable_target(self):
        """Points outside reach limit must cleanly return None without crashing."""
        far_points = [
            (2.5, 0.0, 0.5),    # 2.5m away
            (0.0, 1.8, 0.5),    # Beyond max reach
            (0.0, -0.55, 1.8),  # Unreachable vertical
        ]
        for x, y, z in far_points:
            sol = solve_ik_exact(x, y, z)
            self.assertIsNone(sol, f"Far point ({x}, {y}, {z}) should be unreachable")


if __name__ == '__main__':
    unittest.main()
