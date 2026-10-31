#!/usr/bin/env python3
"""Procedural workpiece randomizer for the robotic sorter simulation.

Fills worlds/sorter.sdf.template with N blocks (colors cycled red, yellow,
green, blue) and one DetachableJoint electromagnet plugin per block, then writes:
  - worlds/sorter.sdf        (generated; never hand-edit)
  - worlds/groundtruth.json  ({item_key: [x, y, z, yaw]}; also the block manifest
                              that entrypoint_sim.sh bridges topics from)

Placement: inside the staging area, clear of every bin tray and the robot
pedestal, IK-reachable at contact and transit height, and non-overlapping
(exact oriented-square test with clearance). Dense layouts that random
sampling can't satisfy fall back to a jittered hex lattice.
"""

import argparse
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'color_sorter_vision'))
from color_sorter_vision.workcell import (  # noqa: E402
    BLOCK_REST_Z, ITEM_METADATA, MAX_BLOCKS, STAGING_X, STAGING_Y,
    base_color, block_keys, blocks_overlap, model_name, reachable, staging_pose_valid,
)

LATTICE_PITCH = 0.16
Pose = Tuple[float, float, float]  # (x, y, yaw)


def check_reachability_ik(x: float, y: float, z: float = 0.570) -> bool:
    """Kept for callers/tests: reachable at the given tool height."""
    return reachable(x, y, z)


def _random_layout(count: int, rng: random.Random, max_yaw: float, attempts: int) -> Optional[List[Pose]]:
    poses: List[Pose] = []
    for _ in range(attempts):
        if len(poses) == count:
            break
        x = round(rng.uniform(*STAGING_X), 3)
        y = round(rng.uniform(*STAGING_Y), 3)
        yaw = round(rng.uniform(-max_yaw, max_yaw), 3)
        if staging_pose_valid(x, y) and not any(blocks_overlap((x, y, yaw), p) for p in poses):
            poses.append((x, y, yaw))
    return poses if len(poses) == count else None


def _lattice_layout(count: int, rng: random.Random) -> Optional[List[Pose]]:
    """Jittered hex lattice: guaranteed spacing for dense layouts."""
    best: List[Tuple[float, float]] = []
    for ox in (i * 0.01 for i in range(16)):
        for oy in (i * 0.01 for i in range(16)):
            cells, row, y = [], 0, STAGING_Y[0] + oy
            while y <= STAGING_Y[1]:
                x = STAGING_X[0] + ox + (LATTICE_PITCH / 2.0 if row % 2 else 0.0)
                while x <= STAGING_X[1]:
                    if staging_pose_valid(x, y):
                        cells.append((x, y))
                    x += LATTICE_PITCH
                y += LATTICE_PITCH * math.sqrt(3) / 2.0
                row += 1
            if len(cells) > len(best):
                best = cells
    if len(best) < count:
        return None
    poses: List[Pose] = []
    for x, y in rng.sample(best, count):
        for _ in range(20):
            p = (round(x + rng.uniform(-0.006, 0.006), 3),
                 round(y + rng.uniform(-0.006, 0.006), 3),
                 round(rng.uniform(-0.2, 0.2), 3))
            if staging_pose_valid(p[0], p[1]) and not any(blocks_overlap(p, q) for q in poses):
                poses.append(p)
                break
        else:
            poses.append((x, y, 0.0))
    return poses


def sample_collision_free_poses(count: int = 4, seed: Optional[int] = None,
                                max_attempts: int = 20000) -> List[Pose]:
    """Sample non-overlapping, reachable (x, y, yaw) block poses on the table."""
    if not 1 <= count <= MAX_BLOCKS:
        raise ValueError(f'count must be in [1, {MAX_BLOCKS}]')
    rng = random.Random(seed)
    for max_yaw in (0.75, 0.4):
        poses = _random_layout(count, rng, max_yaw, max_attempts)
        if poses:
            return poses
    poses = _lattice_layout(count, rng)
    if poses is None:
        raise RuntimeError(f'Staging area cannot fit {count} blocks.')
    return poses


def generate_block_sdf(name: str, x: float, y: float, z: float, yaw: float, ambient: str, diffuse: str) -> str:
    return f"""    <model name="{name}">
      <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.3f}</pose>
      <link name="link">
        <inertial>
          <mass>0.5</mass>
          <inertia><ixx>0.0012</ixx><iyy>0.0012</iyy><izz>0.0012</izz></inertia>
        </inertial>
        <collision name="collision">
          <geometry><box><size>0.12 0.12 0.12</size></box></geometry>
          <surface>
            <friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction>
            <contact><ode><kp>50000.0</kp><kd>500.0</kd><max_vel>0.0</max_vel><min_depth>0.002</min_depth></ode></contact>
          </surface>
        </collision>
        <visual name="visual">
          <geometry><box><size>0.12 0.12 0.12</size></box></geometry>
          <material><ambient>{ambient}</ambient><diffuse>{diffuse}</diffuse></material>
        </visual>
      </link>
    </model>
"""


def generate_detachable_plugin_sdf(name: str, item_key: str) -> str:
    return f"""      <plugin filename="ignition-gazebo-detachable-joint-system" name="ignition::gazebo::systems::DetachableJoint">
        <parent_link>magnetic_tool_link</parent_link>
        <child_model>{name}</child_model>
        <child_link>link</child_link>
        <detach_topic>/electromagnet/detach_{item_key}</detach_topic>
        <attach_topic>/electromagnet/attach_{item_key}</attach_topic>
        <output_topic>/electromagnet/state_{item_key}</output_topic>
        <initial_attach>false</initial_attach>
      </plugin>"""


def build_world_sdf(poses: List[Pose], keys: List[str], template_path: str,
                    output_path: str) -> Dict[str, Tuple[float, float, float, float]]:
    with open(template_path) as f:
        content = f.read()

    blocks_xml, plugins_xml, ground_truth = [], [], {}
    for (x, y, yaw), key in zip(poses, keys):
        meta = ITEM_METADATA[base_color(key)]
        name = model_name(key)
        blocks_xml.append(generate_block_sdf(name, x, y, BLOCK_REST_Z, yaw, meta['sdf_ambient'], meta['sdf_diffuse']))
        plugins_xml.append(generate_detachable_plugin_sdf(name, key))
        ground_truth[key] = (x, y, BLOCK_REST_Z, yaw)

    for marker in ('<!-- @INSERT_WORKPIECE_BLOCKS@ -->', '<!-- @INSERT_ELECTROMAGNET_PLUGINS@ -->'):
        if marker not in content:
            raise ValueError(f'Template is missing placeholder {marker}')
    content = content.replace('<!-- @INSERT_WORKPIECE_BLOCKS@ -->', '\n'.join(blocks_xml))
    content = content.replace('<!-- @INSERT_ELECTROMAGNET_PLUGINS@ -->', '\n\n'.join(plugins_xml))

    with open(output_path, 'w') as f:
        f.write(content)
    return ground_truth


def main() -> None:
    ws = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Randomize sorter simulation workpieces')
    parser.add_argument('--num-blocks', type=int, default=4,
                        help=f'Blocks to place, 1..{MAX_BLOCKS}; colors cycle red/yellow/green/blue (default: 4)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for deterministic layouts')
    parser.add_argument('--template', default=f'{ws}/worlds/sorter.sdf.template')
    parser.add_argument('--output', default=f'{ws}/worlds/sorter.sdf')
    parser.add_argument('--groundtruth-json', default=f'{ws}/worlds/groundtruth.json')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    keys = block_keys(args.num_blocks)
    poses = sample_collision_free_poses(len(keys), seed=args.seed)
    ground_truth = build_world_sdf(poses, keys, args.template, args.output)
    with open(args.groundtruth_json, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    if not args.quiet:
        print(f'✓ Randomized {len(keys)} blocks into {args.output}')
        for k, (x, y, z, yaw) in ground_truth.items():
            print(f'  - {k.upper():<10}: X={x:+.3f}m, Y={y:+.3f}m, Z={z:.3f}m, Yaw={math.degrees(yaw):+.1f}°')


if __name__ == '__main__':
    main()
