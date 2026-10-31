#!/usr/bin/env python3
"""Closed-loop 4-DOF pick-and-place controller with electromagnet end-effector.

Feedback used (all bridged from Gazebo by entrypoint_sim.sh):
  /joint_states        measured joint positions/velocities (JointStatePublisher)
  /sim/poses           block world poses (SceneBroadcaster dynamic_pose/info),
                       standing in for the cell's proximity / bin-presence sensors
  /detections/blocks   overhead-camera detections from color_detector

Loop structure:
  - Motion: Cartesian straight-line setpoints streamed at 50 Hz. A setpoint
    governor pauses the trajectory clock while measured joints lag the command,
    so motion adapts to the sim's real-time factor; each move ends by waiting
    for joint convergence, then an outer Cartesian loop corrects residual tool
    error measured through forward kinematics.
  - Perception: each pick targets the freshest vision detection matching the
    block; the proximity sensor refines XY/height before the magnet engages.
  - Verification: a grasp counts only if the block rises with the tool; a sort
    counts only if the block comes to rest inside its color's bin. Failed grasps,
    drops and misplacements are retried from the block's measured position.
"""

import argparse
import csv
import json
import math
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float64, String
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray

from color_sorter_vision.workcell import (  # noqa: F401  (BINS, L2.. re-exported for tests)
    BIN_CAPACITY, BIN_CENTERS, BIN_INNER_HALF_X, BLOCK_HALF, BLOCK_REST_Z, BLOCK_SIZE, COLORS,
    CORRIDOR_X, ITEM_METADATA, JOINT_NAMES, L2, L3, L4, METRICS_PATH, X_BASE, Y_BASE, Z_DROP,
    Z_PROBE, Z_SHOULDER, Z_TRANSIT, base_color, bin_containing, blocks_overlap, footprint_inside_bin,
    forward_kinematics, load_groundtruth, model_name, reachable, solve_ik_exact, yaw_from_quaternion,
)

# Bins as (x, y, drop tool height) for callers that want a single drop point per color.
BINS: Dict[str, Tuple[float, float, float]] = {c: (*BIN_CENTERS[c], Z_DROP) for c in COLORS}

HOME = (0.0, -0.30, 0.78)
CONTROL_HZ = 50.0
SPEED_TRANSIT = 0.40      # m/s
SPEED_VERTICAL = 0.15     # m/s
GOVERNOR_LAG = 0.12       # rad: pause the trajectory clock above this tracking error
CONVERGED_RAD = 0.004     # rad: joint convergence tolerance at end of a move
CARTESIAN_TOL = 0.003     # m: outer-loop tool position tolerance
GRASP_XY_TOL = 0.025      # m: tool-to-block-centre misalignment allowed when energizing
GRASP_Z_TOL = 0.015       # m: tool-to-block-top gap allowed when energizing
LIFT_CHECK = 0.05         # m: test-lift height used to verify a grasp
VISION_MATCH_RADIUS = 0.10
VISION_MAX_AGE = 1.0      # s


class SorterController(Node):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__('pick_and_place_controller')
        self.max_attempts = max_attempts

        self.spawn_poses = load_groundtruth()
        if not self.spawn_poses:
            raise RuntimeError('worlds/groundtruth.json missing: run ./run.sh randomize first')
        self.items: List[str] = list(self.spawn_poses)

        self.pub_joints = [
            self.create_publisher(Float64, f'/model/sorter_robot/joint/{j}/cmd_pos', 10) for j in JOINT_NAMES
        ]
        self.pub_attach = {k: self.create_publisher(Empty, f'/electromagnet/attach_{k}', 10) for k in self.items}
        self.pub_detach = {k: self.create_publisher(Empty, f'/electromagnet/detach_{k}', 10) for k in self.items}
        self.pub_markers = self.create_publisher(MarkerArray, '/detections', 10)
        self.pub_status = self.create_publisher(
            String, '/sorting/status', QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        self._lock = threading.RLock()
        self.q_meas: Optional[Tuple[float, ...]] = None
        self.qd_meas: Tuple[float, ...] = (0.0,) * 4
        self.block_poses: Dict[str, Tuple[float, float, float]] = {}
        self.block_yaws: Dict[str, float] = {}
        self.block_pose_stamp = 0.0
        self.detections: List[Dict] = []
        self.detections_stamp = 0.0

        self.create_subscription(JointState, '/joint_states', self._on_joint_state, qos_profile_sensor_data)
        self.create_subscription(TFMessage, '/sim/poses', self._on_poses, 10)
        self.create_subscription(String, '/detections/blocks', self._on_detections, 10)
        self.create_timer(0.2, self._publish_markers)

        self.q_cmd: Optional[Tuple[float, ...]] = None
        self.tool_cmd = HOME
        self.held: Optional[str] = None
        self.metrics: List[Dict] = []

    # ------------------------------------------------------------------ feedback
    def _on_joint_state(self, msg: JointState) -> None:
        idx = {n: i for i, n in enumerate(msg.name)}
        if not all(j in idx for j in JOINT_NAMES):
            return
        q = tuple(msg.position[idx[j]] for j in JOINT_NAMES)
        qd = tuple(msg.velocity[idx[j]] for j in JOINT_NAMES) if len(msg.velocity) == len(msg.name) else (0.0,) * 4
        with self._lock:
            self.q_meas, self.qd_meas = q, qd

    def _on_poses(self, msg: TFMessage) -> None:
        poses, yaws = {}, {}
        for tf in msg.transforms:
            name = tf.child_frame_id
            if '_block_' in name:
                t, r = tf.transform.translation, tf.transform.rotation
                poses[name] = (t.x, t.y, t.z)
                yaws[name] = yaw_from_quaternion(r.x, r.y, r.z, r.w)
        if poses:
            with self._lock:
                self.block_poses.update(poses)
                self.block_yaws.update(yaws)
                self.block_pose_stamp = time.time()

    def _on_detections(self, msg: String) -> None:
        try:
            dets = json.loads(msg.data)
        except ValueError:
            return
        with self._lock:
            self.detections, self.detections_stamp = dets, time.time()

    def joints(self) -> Tuple[float, ...]:
        with self._lock:
            return self.q_meas

    def tool_measured(self) -> Tuple[float, float, float]:
        return forward_kinematics(*self.joints())

    def block_pose(self, key: str) -> Optional[Tuple[float, float, float]]:
        with self._lock:
            return self.block_poses.get(model_name(key))

    def wait_for_feedback(self, timeout: float = 15.0) -> None:
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                ready = self.q_meas is not None and self.block_poses
            if ready:
                return
            time.sleep(0.1)
        raise RuntimeError('No /joint_states or /sim/poses feedback: is the sim + bridge running?')

    # ------------------------------------------------------------------ motion
    def _command(self, q: Sequence[float]) -> None:
        for pub, v in zip(self.pub_joints, q):
            pub.publish(Float64(data=float(v)))
        self.q_cmd = tuple(q)

    def _tracking_error(self) -> float:
        q = self.joints()
        return max(abs(a - b) for a, b in zip(self.q_cmd, q))

    def _stream(self, duration: float, setpoint_at) -> None:
        """Stream setpoint_at(s) for s in [0,1] (smoothstep-timed), feedback-paced by the governor."""
        dt = 1.0 / CONTROL_HZ
        progress, stalled = 0.0, 0.0
        while progress < duration:
            if self.q_cmd is not None and self._tracking_error() > GOVERNOR_LAG:
                stalled += dt
                if stalled > 10.0:
                    q = self.joints()
                    raise RuntimeError(
                        'Joint tracking stalled for 10 s (arm blocked or sim paused): '
                        f'cmd={[round(v, 3) for v in self.q_cmd]} meas={[round(v, 3) for v in q]} '
                        f'tool={tuple(round(v, 3) for v in forward_kinematics(*q))} held={self.held}')
            else:
                progress = min(duration, progress + dt)
            s = progress / duration
            self._command(setpoint_at(s * s * (3.0 - 2.0 * s)))
            time.sleep(dt)

    def _wait_converged(self, timeout: float = 4.0) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                qd = self.qd_meas
            if self._tracking_error() < CONVERGED_RAD and max(abs(v) for v in qd) < 0.05:
                return True
            time.sleep(0.02)
        return False

    def move_joints(self, q_target: Sequence[float], duration: float) -> None:
        q0 = self.joints()
        self._stream(duration, lambda s: [a + (b - a) * s for a, b in zip(q0, q_target)])
        self._wait_converged()

    def move_linear(self, target: Tuple[float, float, float], speed: float = SPEED_TRANSIT,
                    precise: bool = False) -> float:
        """Straight-line tool move. Returns the final measured tool position error (m)."""
        x0, y0, z0 = self.tool_cmd
        x1, y1, z1 = target
        if solve_ik_exact(*target) is None:
            raise RuntimeError(f'Target ({x1:.3f}, {y1:.3f}, {z1:.3f}) is unreachable')

        def setpoint(s: float):
            return solve_ik_exact(x0 + (x1 - x0) * s, y0 + (y1 - y0) * s, z0 + (z1 - z0) * s)

        dist = math.dist(self.tool_cmd, target)
        self._stream(max(0.3, dist / speed), setpoint)
        self.tool_cmd = target
        self._wait_converged()

        # Outer loop: correct residual Cartesian error measured through FK.
        err = math.dist(self.tool_measured(), target)
        if precise:
            aim = list(target)
            for _ in range(3):
                if err < CARTESIAN_TOL:
                    break
                meas = self.tool_measured()
                aim = [a + (t - m) for a, t, m in zip(aim, target, meas)]
                sol = solve_ik_exact(*aim)
                if sol is None:
                    break
                self._command(sol)
                self._wait_converged(2.0)
                err = math.dist(self.tool_measured(), target)
        return err

    # ------------------------------------------------------------------ magnet
    def magnet(self, key: str, on: bool) -> None:
        (self.pub_attach if on else self.pub_detach)[key].publish(Empty())
        self.held = key if on else None
        time.sleep(0.25)

    def probe_magnetism(self, key: str) -> bool:
        meta = ITEM_METADATA[base_color(key)]
        self.get_logger().info(
            f"🔍 Inductive probe '{key}': {meta['inductance_uh']:.1f} uH, "
            f"{meta['material']} -> ferromagnetic={meta['is_magnetic']}")
        self.status(f"PROBE:{key}:{meta['is_magnetic']}:{meta['material']}")
        return meta['is_magnetic']

    # ------------------------------------------------------------------ perception
    def localize(self, key: str) -> Tuple[Optional[Tuple[float, float, float]], str]:
        """Best estimate of a block's position: fresh vision match first, pose sensor otherwise."""
        sensed = self.block_pose(key)
        prior = sensed or self.spawn_poses[key][:3]
        with self._lock:
            dets, age = self.detections, time.time() - self.detections_stamp
        if age < VISION_MAX_AGE:
            color = base_color(key)
            cands = [d for d in dets if d.get('base_color') == color]
            if cands:
                best = min(cands, key=lambda d: math.hypot(d['x'] - prior[0], d['y'] - prior[1]))
                if math.hypot(best['x'] - prior[0], best['y'] - prior[1]) < VISION_MATCH_RADIUS:
                    z = sensed[2] if sensed else BLOCK_REST_Z
                    return (best['x'], best['y'], z), 'vision'
        return sensed, 'pose-sensor'

    def bin_occupants(self, color: str, exclude: str) -> List[Tuple[float, float, float, float]]:
        """(x, y, z, yaw) of every block currently resting in a bin, except `exclude`."""
        with self._lock:
            items = [(p, self.block_yaws.get(n, 0.0)) for n, p in self.block_poses.items()
                     if n != model_name(exclude)]
        return [(*p, yaw) for p, yaw in items if bin_containing(p[0], p[1]) == color]

    def plan_drop(self, color: str, key: str, rel_yaw: float) -> Optional[Tuple[float, float, float]]:
        """Tool drop point for a held block, from sensed bin contents.

        The block's yaw in the bin will be rel_yaw + theta_1 at the drop point.
        Floor: outermost x along the bin axis whose rotated footprint stays inside
        the tray and clears every sensed occupant. Bin floor full: stack on the
        outermost floor block.
        """
        cx, cy = BIN_CENTERS[color]
        outward = 1.0 if cx > 0 else -1.0
        occupants = self.bin_occupants(color, exclude=key)
        floor = [o for o in occupants if o[2] < BLOCK_REST_Z + 0.05]
        step = 0.005
        for i in range(int(2 * BIN_INNER_HALF_X / step) + 1):
            x = cx + outward * (BIN_INNER_HALF_X - i * step)
            if not reachable(x, cy, Z_TRANSIT):
                continue
            q = solve_ik_exact(x, cy, Z_DROP)
            yaw = rel_yaw + q[0]
            if footprint_inside_bin(color, x, cy, yaw) and \
                    not any(blocks_overlap((x, cy, yaw), (o[0], o[1], o[3]), clearance=0.012) for o in floor):
                return x, cy, Z_DROP
        stacked_on = {id(f) for f in floor for o in occupants
                      if o[2] > BLOCK_REST_Z + 0.05 and math.hypot(o[0] - f[0], o[1] - f[1]) < BLOCK_HALF}
        bases = [f for f in floor if id(f) not in stacked_on]
        if not bases:
            return None
        base = max(bases, key=lambda f: outward * f[0])
        return base[0], base[1], base[2] + BLOCK_HALF + BLOCK_SIZE + 0.005

    def wait_block_settled(self, key: str, timeout: float = 2.0) -> Optional[Tuple[float, float, float]]:
        last, start = self.block_pose(key), time.time()
        while time.time() - start < timeout:
            time.sleep(0.15)
            cur = self.block_pose(key)
            if cur and last and math.dist(cur, last) < 0.001:
                return cur
            last = cur
        return last

    # ------------------------------------------------------------------ sorting
    def status(self, text: str) -> None:
        self.pub_status.publish(String(data=text))

    def attempt_pick_and_place(self, key: str) -> Tuple[str, str, str]:
        """One pick/place attempt. Returns (outcome, pick_pos, drop_pos).

        outcome: SORTED_SUCCESS | GRASP_FAILED | DROPPED | MISPLACED | MISSORTED | NO_SLOT | LOST
        """
        color = base_color(key)
        est, source = self.localize(key)
        if est is None:
            return 'LOST', '', ''
        bx, by, bz = est
        pick = f'({bx:.3f},{by:.3f},{bz:.3f})'
        if len(self.bin_occupants(color, exclude=key)) >= BIN_CAPACITY:
            self.get_logger().error(f'   {color} bin is full')
            return 'NO_SLOT', pick, ''
        self.get_logger().info(f'📦 {key}: target ({bx:+.3f}, {by:+.3f}) from {source}')

        self.move_linear((bx, by, Z_TRANSIT))
        self.move_linear((bx, by, Z_PROBE), SPEED_VERTICAL)
        if not self.probe_magnetism(key):
            self.move_linear((bx, by, Z_TRANSIT), SPEED_VERTICAL)
            return 'SKIPPED_NON_MAGNETIC', pick, ''

        # Fine alignment from the proximity sensor before contact.
        sensed = self.block_pose(key)
        if sensed and math.hypot(sensed[0] - bx, sensed[1] - by) > 0.005:
            self.get_logger().info(
                f'   proximity correction {1000 * math.hypot(sensed[0] - bx, sensed[1] - by):.1f} mm')
            bx, by = sensed[0], sensed[1]
            self.move_linear((bx, by, Z_PROBE), SPEED_VERTICAL)
        top = (sensed[2] if sensed else BLOCK_REST_Z) + BLOCK_HALF
        self.move_linear((bx, by, top + 0.002), SPEED_VERTICAL, precise=True)

        tool = self.tool_measured()
        sensed = self.block_pose(key)
        if sensed is None or math.hypot(tool[0] - sensed[0], tool[1] - sensed[1]) > GRASP_XY_TOL \
                or abs(tool[2] - (sensed[2] + BLOCK_HALF)) > GRASP_Z_TOL:
            self.move_linear((bx, by, Z_TRANSIT), SPEED_VERTICAL)
            return 'GRASP_FAILED', pick, ''

        self.magnet(key, True)
        z_before = sensed[2]
        self.move_linear((bx, by, top + LIFT_CHECK), SPEED_VERTICAL)
        lifted = self.block_pose(key)
        if lifted is None or lifted[2] - z_before < LIFT_CHECK * 0.6:
            rose = 1000 * (lifted[2] - z_before) if lifted else float('nan')
            self.get_logger().warn(f'   grasp check failed: block rose {rose:.0f} mm')
            self.magnet(key, False)
            self.move_linear((bx, by, Z_TRANSIT), SPEED_VERTICAL)
            return 'GRASP_FAILED', pick, ''
        self.move_linear((bx, by, Z_TRANSIT), SPEED_VERTICAL)

        # Yaw of the block relative to the tool is fixed while held (no wrist yaw joint).
        q = self.joints()
        with self._lock:
            rel_yaw = self.block_yaws.get(model_name(key), 0.0) - q[0]
        plan = self.plan_drop(color, key, rel_yaw)
        if plan is None:
            self.get_logger().error(f'   no room in {color} bin')
            plan = (*BIN_CENTERS[color], Z_DROP + BLOCK_SIZE)
        dx, dy, dz = plan
        self.get_logger().info(
            f'   drop at ({dx:+.3f}, {dy:+.3f}) {"stacked" if dz > Z_DROP + 0.05 else "floor"}')
        self.move_linear((math.copysign(CORRIDOR_X, dx), dy, Z_TRANSIT))
        self.move_linear((dx, dy, Z_TRANSIT))

        # Still holding it? (block hangs BLOCK_HALF below the tool face)
        tool, held = self.tool_measured(), self.block_pose(key)
        if held is None or math.dist(held, (tool[0], tool[1], tool[2] - BLOCK_HALF)) > 0.04:
            self.get_logger().warn('   block lost in transit')
            self.magnet(key, False)
            return 'DROPPED', pick, ''

        self.move_linear((dx, dy, dz), SPEED_VERTICAL, precise=True)
        self.magnet(key, False)
        self.move_linear((dx, dy, Z_TRANSIT), SPEED_VERTICAL)
        drop = f'({dx:.3f},{dy:.3f},{dz:.3f})'

        rest = self.wait_block_settled(key)
        landed_in = bin_containing(rest[0], rest[1]) if rest else None
        # Resting inside its own bin, on the floor or one block up (not hanging off the magnet).
        if landed_in == color and rest[2] < BLOCK_REST_Z + BLOCK_SIZE + 0.03:
            return 'SORTED_SUCCESS', pick, drop
        return ('MISSORTED' if landed_in else 'MISPLACED'), pick, drop

    def run(self) -> None:
        self.wait_for_feedback()
        for k in self.items:  # release anything a previous run left attached
            self.pub_detach[k].publish(Empty())
        self.status('ACTIVE')
        self.get_logger().info(f'🚀 Closed-loop sort of {len(self.items)} blocks: {", ".join(self.items)}')

        home_q = solve_ik_exact(*HOME)
        self.move_joints(home_q, duration=2.0)
        self.tool_cmd = HOME

        # Give vision a moment to publish a first catalog (optional input).
        start = time.time()
        while time.time() - start < 3.0 and time.time() - self.detections_stamp > VISION_MAX_AGE:
            time.sleep(0.1)
        if time.time() - self.detections_stamp > VISION_MAX_AGE:
            self.get_logger().warn('No vision detections: localizing from the pose sensor only')

        t_start = time.time()
        pending = list(self.items)
        while pending:
            # Greedy: nearest remaining block to the tool.
            tool = self.tool_cmd
            key = min(pending, key=lambda k: math.hypot(
                *(a - b for a, b in zip((self.block_pose(k) or self.spawn_poses[k])[:2], tool[:2]))))
            pending.remove(key)
            item_start, outcome, attempts = time.time(), 'LOST', 0
            pick = drop = ''
            self.status(f'PICK:{key}')
            while attempts < self.max_attempts:
                attempts += 1
                outcome, pick, drop = self.attempt_pick_and_place(key)
                if outcome in ('SORTED_SUCCESS', 'SKIPPED_NON_MAGNETIC', 'NO_SLOT', 'MISSORTED', 'LOST'):
                    break
                self.get_logger().warn(f'   attempt {attempts}/{self.max_attempts} for {key}: {outcome}, retrying')
            final = self.block_pose(key)
            cycle = time.time() - item_start
            icon = '✅' if outcome == 'SORTED_SUCCESS' else '❌'
            self.get_logger().info(f'{icon} {key}: {outcome} after {attempts} attempt(s), {cycle:.1f}s')
            self.metrics.append({
                'color': key,
                'status': outcome,
                'material': ITEM_METADATA[base_color(key)]['material'],
                'pick_pos': pick,
                'drop_pos': drop,
                'cycle_time_sec': round(cycle, 2),
                'attempts': attempts,
                'final_pos': f'({final[0]:.3f},{final[1]:.3f},{final[2]:.3f})' if final else '',
                'verified': outcome == 'SORTED_SUCCESS',
            })

        self.move_linear(HOME)
        total = time.time() - t_start
        sorted_ok = sum(1 for m in self.metrics if m['status'] == 'SORTED_SUCCESS')
        self.status(f'DONE:{sorted_ok}/{len(self.items)}')
        self.save_metrics(total, sorted_ok)

    def save_metrics(self, total: float, sorted_ok: int) -> None:
        fields = ['color', 'status', 'material', 'pick_pos', 'drop_pos', 'cycle_time_sec',
                  'attempts', 'final_pos', 'verified']
        with open(METRICS_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.metrics)

        retries = sum(m['attempts'] - 1 for m in self.metrics)
        print('\n' + '=' * 64)
        print('📊 SORTING REPORT (placement verified by bin pose sensor)')
        print('=' * 64)
        print(f'Blocks sorted & verified: {sorted_ok}/{len(self.items)}   retries: {retries}')
        print(f'Total time: {total:.1f}s   throughput: {sorted_ok / (total / 60.0):.1f} items/min')
        print(f'Metrics CSV: {METRICS_PATH}')
        print('-' * 64)
        for m in self.metrics:
            print(f"{m['color'].upper():<10} {m['status']:<16} attempts={m['attempts']}  {m['cycle_time_sec']:6.1f}s")
        print('=' * 64 + '\n')

    # ------------------------------------------------------------------ RViz
    def _publish_markers(self) -> None:
        """Cubes for blocks that vision doesn't show: held or already in a bin."""
        ma = MarkerArray()
        with self._lock:
            poses = dict(self.block_poses)
        for i, key in enumerate(self.items):
            p = poses.get(model_name(key))
            m = Marker()
            m.header.frame_id = 'world'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns, m.id, m.type = 'sorter_blocks', i, Marker.CUBE
            show = p is not None and (key == self.held or bin_containing(p[0], p[1]) is not None)
            if not show:
                m.action = Marker.DELETE
            else:
                m.action = Marker.ADD
                m.pose.position.x, m.pose.position.y, m.pose.position.z = p
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = m.scale.z = BLOCK_SIZE
                m.color.r, m.color.g, m.color.b, m.color.a = ITEM_METADATA[base_color(key)]['rgba']
            ma.markers.append(m)
        self.pub_markers.publish(ma)


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description='Closed-loop sorter controller')
    parser.add_argument('--max-attempts', type=int, default=3, help='Pick attempts per block (default 3)')
    cli, ros_args = parser.parse_known_args(args)

    rclpy.init(args=['pick_and_place'] + ros_args)
    node = SorterController(max_attempts=cli.max_attempts)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Cancelled by user.')
    except RuntimeError as e:
        node.get_logger().error(str(e))
        node.status(f'ERROR:{e}')
    finally:
        if node.held:
            node.magnet(node.held, False)
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
