"""Single source of truth for workcell geometry, bins, colors and kinematics.

Imported by the controller, the vision node, randomize_world.py and the tests.
Anything that must agree with worlds/sorter.sdf.template lives here; when you
change a number in this file, check the template (and sorter_robot.urdf for
bins/robot) still matches.
"""

import json
import math
import os
from typing import Dict, List, Optional, Tuple

ROS_WS = '/home/hyprnon/ros_ws'
GROUNDTRUTH_PATH = f'{ROS_WS}/worlds/groundtruth.json'
METRICS_PATH = f'{ROS_WS}/sorting_metrics.csv'

# --- Robot kinematics (matches sorter_robot model in the SDF template) ------
X_BASE = 0.0
Y_BASE = -0.55
Z_BASE = 0.45
Z_SHOULDER = 0.65       # Shoulder pitch axis height (Z_BASE + 0.20 m pedestal)
L2 = 0.45               # Upper arm length
L3 = 0.42               # Forearm length
L4 = 0.12               # Wrist axis to bottom face of electromagnet
JOINT_NAMES = ('joint_1', 'joint_2', 'joint_3', 'joint_4')

# --- Table / blocks ---------------------------------------------------------
TABLE_Z = 0.45
BLOCK_SIZE = 0.12
BLOCK_HALF = BLOCK_SIZE / 2.0
BLOCK_REST_Z = TABLE_Z + BLOCK_HALF          # 0.51 block centre on table
BLOCK_TOP_Z = TABLE_Z + BLOCK_SIZE           # 0.57

# --- Motion heights (tool = bottom face of electromagnet) -------------------
Z_TRANSIT = 0.72        # carried block bottom at 0.60, clears table blocks (top 0.57)
Z_PROBE = BLOCK_TOP_Z + 0.04

# --- Sorting bins -----------------------------------------------------------
# Each bin is a static 0.46 x 0.26 m tray. The arm has no wrist yaw, so a block
# keeps the yaw it had when grasped (a 45-degree cube is 0.17 m wide). Two blocks
# always fit side by side on the floor at any yaw; the third is stacked on the
# outermost one. Floor positions are chosen at drop time from sensed poses.
BIN_SIZE_X = 0.46
BIN_SIZE_Y = 0.26
BIN_WALL = 0.015
BIN_INNER_HALF_X = BIN_SIZE_X / 2.0 - BIN_WALL   # 0.215
BIN_INNER_HALF_Y = BIN_SIZE_Y / 2.0 - BIN_WALL   # 0.115
BIN_Z = 0.452                                # tray model origin
BIN_FLOOR_TOP_Z = BIN_Z + 0.008              # 0.460
BIN_FLOOR_SLOTS = 2
BIN_CAPACITY = 3                             # 2 on the floor + 1 stacked
BIN_SLOT_PITCH = 0.17                        # nominal floor-slot spacing (worst-case yaw)
BIN_SLOTS = BIN_FLOOR_SLOTS
BIN_CENTERS: Dict[str, Tuple[float, float]] = {
    'red': (-0.58, -0.14),
    'yellow': (-0.58, -0.42),
    'green': (0.58, -0.14),
    'blue': (0.58, -0.42),
}
# Tool height when releasing a block in a bin: 5 mm above its resting top face.
Z_DROP = BIN_FLOOR_TOP_Z + BLOCK_SIZE + 0.005
Z_DROP_STACKED = Z_DROP + BLOCK_SIZE
# Carried blocks reach a bin via x = +/-CORRIDOR_X so they never sweep over
# another bin's stack (stack top 0.70 m vs carried block bottom 0.60 m).
CORRIDOR_X = 0.25

COLORS: Tuple[str, ...] = ('red', 'yellow', 'green', 'blue')
MAX_BLOCKS = BIN_CAPACITY * len(COLORS)

# --- Staging area where blocks spawn ----------------------------------------
STAGING_X = (-0.40, 0.40)
STAGING_Y = (-0.14, 0.28)
STAGING_BIN_MARGIN = 0.04     # block footprint clearance to any bin tray
STAGING_BASE_CLEARANCE = 0.40  # keep clear of the robot pedestal / folded arm

# --- Overhead camera (matches overhead_camera model) ------------------------
CAMERA_Z = 2.5
CAMERA_HFOV = 0.9

ITEM_METADATA: Dict[str, Dict] = {
    'red': {
        'material': 'Ferromagnetic Carbon Steel', 'is_magnetic': True, 'inductance_uh': 14.8,
        'rgba': (0.95, 0.12, 0.12, 0.95), 'sdf_diffuse': '1.0 0.08 0.08 1.0', 'sdf_ambient': '0.9 0.05 0.05 1.0',
    },
    'yellow': {
        'material': 'Ferromagnetic Brass-plated Steel', 'is_magnetic': True, 'inductance_uh': 12.5,
        'rgba': (0.95, 0.85, 0.05, 0.95), 'sdf_diffuse': '1.0 0.85 0.05 1.0', 'sdf_ambient': '0.9 0.75 0.05 1.0',
    },
    'green': {
        'material': 'Ferromagnetic Spring Steel', 'is_magnetic': True, 'inductance_uh': 14.2,
        'rgba': (0.12, 0.85, 0.18, 0.95), 'sdf_diffuse': '0.10 0.85 0.15 1.0', 'sdf_ambient': '0.08 0.75 0.12 1.0',
    },
    'blue': {
        'material': 'Ferromagnetic Nickel-plated Iron', 'is_magnetic': True, 'inductance_uh': 13.5,
        'rgba': (0.15, 0.35, 0.95, 0.95), 'sdf_diffuse': '0.12 0.35 0.95 1.0', 'sdf_ambient': '0.10 0.30 0.85 1.0',
    },
}


# --- Block identity ----------------------------------------------------------
def block_keys(num_blocks: int) -> List[str]:
    """Block keys for an N-block world, colors cycled: red_1, yellow_1, ..., red_2, ..."""
    if not 1 <= num_blocks <= MAX_BLOCKS:
        raise ValueError(f'num_blocks must be in [1, {MAX_BLOCKS}] (bins hold {BIN_CAPACITY} per color)')
    return [f'{COLORS[i % len(COLORS)]}_{i // len(COLORS) + 1}' for i in range(num_blocks)]


def base_color(item_key: str) -> str:
    return item_key.split('_')[0]


def model_name(item_key: str) -> str:
    """Gazebo model name for a block key: red_2 -> red_block_2."""
    color, idx = item_key.split('_')
    return f'{color}_block_{idx}'


def load_groundtruth(path: str = GROUNDTRUTH_PATH) -> Dict[str, Tuple[float, float, float, float]]:
    """Spawn poses written by randomize_world.py: {item_key: (x, y, z, yaw)}."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {k: tuple(v) for k, v in data.items()}


# --- Bins ---------------------------------------------------------------------
def bin_slot_xy(color: str, slot: int) -> Tuple[float, float]:
    """Nominal floor slot (0 = outer, 1 = inner) sized for worst-case 45-degree blocks.

    The controller computes actual drop points from sensed poses; these bound
    the reach the arm needs.
    """
    cx, cy = BIN_CENTERS[color]
    outward = 1.0 if cx > 0 else -1.0
    return cx + outward * (BIN_INNER_HALF_X - BLOCK_HALF * math.sqrt(2) - 0.008 - slot * BIN_SLOT_PITCH), cy


def bin_rect(color: str, margin: float = 0.0) -> Tuple[float, float, float, float]:
    """(xmin, xmax, ymin, ymax) of a bin tray footprint, grown by margin."""
    cx, cy = BIN_CENTERS[color]
    hx, hy = BIN_SIZE_X / 2.0 + margin, BIN_SIZE_Y / 2.0 + margin
    return cx - hx, cx + hx, cy - hy, cy + hy


def bin_containing(x: float, y: float, margin: float = 0.0) -> Optional[str]:
    """Color of the bin whose tray contains (x, y), or None."""
    for color in COLORS:
        x0, x1, y0, y1 = bin_rect(color, margin)
        if x0 <= x <= x1 and y0 <= y <= y1:
            return color
    return None


# --- Block footprints ----------------------------------------------------------
def footprint(x: float, y: float, yaw: float, half: float = BLOCK_HALF) -> List[Tuple[float, float]]:
    """Corners of a yawed square block footprint."""
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + c * dx - s * dy, y + s * dx + c * dy)
            for dx, dy in ((half, half), (-half, half), (-half, -half), (half, -half))]


def blocks_overlap(a: Tuple[float, float, float], b: Tuple[float, float, float], clearance: float = 0.025) -> bool:
    """Separating-axis test for two (x, y, yaw) blocks, each grown by clearance/2."""
    half = BLOCK_HALF + clearance / 2.0
    ca, cb = footprint(*a, half), footprint(*b, half)
    for yaw in (a[2], b[2]):
        for ax, ay in ((math.cos(yaw), math.sin(yaw)), (-math.sin(yaw), math.cos(yaw))):
            pa = [px * ax + py * ay for px, py in ca]
            pb = [px * ax + py * ay for px, py in cb]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return False
    return True


def footprint_inside_bin(color: str, x: float, y: float, yaw: float, margin: float = 0.006) -> bool:
    cx, cy = BIN_CENTERS[color]
    return all(abs(px - cx) <= BIN_INNER_HALF_X - margin and abs(py - cy) <= BIN_INNER_HALF_Y - margin
               for px, py in footprint(x, y, yaw))


def yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


# --- Kinematics ---------------------------------------------------------------
def solve_ik_exact(x: float, y: float, z: float) -> Optional[Tuple[float, float, float, float]]:
    """Analytical 4-DOF IK, tool pointing straight down (theta2+theta3+theta4 = pi).

    (x, y, z) is the desired position of the electromagnet's bottom face.
    Elbow-up solution. Returns None if unreachable.
    """
    dx = x - X_BASE
    dy = y - Y_BASE
    theta1 = math.atan2(dy, dx)
    rw = math.hypot(dx, dy)
    dzw = z + L4 - Z_SHOULDER  # wrist centre sits L4 straight above the tool face

    c3 = (rw**2 + dzw**2 - L2**2 - L3**2) / (2.0 * L2 * L3)
    if abs(c3) > 1.0:
        return None
    theta3 = math.atan2(math.sqrt(max(0.0, 1.0 - c3**2)), c3)

    a = L2 + L3 * math.cos(theta3)
    b = L3 * math.sin(theta3)
    theta2 = math.atan2(a * rw - b * dzw, b * rw + a * dzw)

    theta4 = math.pi - (theta2 + theta3)
    theta4 = (theta4 + math.pi) % (2.0 * math.pi) - math.pi
    return theta1, theta2, theta3, theta4


def forward_kinematics(t1: float, t2: float, t3: float, t4: float) -> Tuple[float, float, float]:
    """Tool (electromagnet bottom face) position from joint angles."""
    rw = L2 * math.sin(t2) + L3 * math.sin(t2 + t3)
    zw = Z_SHOULDER + L2 * math.cos(t2) + L3 * math.cos(t2 + t3)
    phi = t2 + t3 + t4
    r = rw + L4 * math.sin(phi)
    z = zw + L4 * math.cos(phi)
    return X_BASE + r * math.cos(t1), Y_BASE + r * math.sin(t1), z


def reachable(x: float, y: float, z: float, margin: float = 0.03) -> bool:
    """IK-reachable with a margin away from full arm extension (|cos theta3| <= 1 - margin)."""
    rw = math.hypot(x - X_BASE, y - Y_BASE)
    dzw = z + L4 - Z_SHOULDER
    c3 = (rw**2 + dzw**2 - L2**2 - L3**2) / (2.0 * L2 * L3)
    return abs(c3) <= 1.0 - margin


def staging_pose_valid(x: float, y: float) -> bool:
    """A block may spawn at (x, y): inside staging area, clear of bins/pedestal, pickable."""
    if not (STAGING_X[0] <= x <= STAGING_X[1] and STAGING_Y[0] <= y <= STAGING_Y[1]):
        return False
    # Block footprint (worst case yawed: half-diagonal) must clear every tray.
    if bin_containing(x, y, margin=BLOCK_HALF * math.sqrt(2) + STAGING_BIN_MARGIN) is not None:
        return False
    if math.hypot(x - X_BASE, y - Y_BASE) < STAGING_BASE_CLEARANCE:
        return False
    return reachable(x, y, BLOCK_TOP_Z) and reachable(x, y, Z_TRANSIT)
