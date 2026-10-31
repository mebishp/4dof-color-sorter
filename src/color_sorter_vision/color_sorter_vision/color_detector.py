"""HSV color detector: subscribes /camera/image_raw, publishes /detections.

No cv_bridge dependency -- sensor_msgs/Image <-> numpy is done manually,
so this runs natively on Arch (no cv_bridge package) and in the container.

Features:
  - Pixel-to-world projective mapping based on camera intrinsics & height.
  - Broadcasts static TF transform between 'world' and camera frame.
  - Publishes 3D bounding boxes and floating text markers in 'world' frame.
  - Publishes annotated debug image with pixel and metric coordinates.

Topics:
  sub: /camera/image_raw (sensor_msgs/Image, rgb8/bgr8/mono8)
  pub: /detections (visualization_msgs/MarkerArray, 3D cubes + text in 'world')
  pub: /detections/debug (sensor_msgs/Image, annotated BGR)
"""

import array
import json
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import TransformStamped
import tf2_ros

from color_sorter_vision.workcell import (
    BIN_CENTERS, BIN_SIZE_X, BIN_SIZE_Y, BLOCK_REST_Z, CAMERA_HFOV, CAMERA_Z, bin_containing,
)


def ros_image_to_numpy(msg: Image) -> np.ndarray:
    """Convert sensor_msgs/Image to numpy BGR array (uint8, HxWx3)."""
    h, w = msg.height, msg.width
    enc = (msg.encoding or '').lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    step = msg.step if msg.step else len(msg.data) // max(h, 1)

    if enc in ('rgb8',):
        arr = buf.reshape((h, step))[:, :w * 3].reshape((h, w, 3))
        return arr[:, :, ::-1].copy()  # RGB -> BGR for OpenCV
    if enc in ('bgr8',):
        arr = buf.reshape((h, step))[:, :w * 3].reshape((h, w, 3))
        return arr.copy()
    if enc in ('mono8', 'mono16', '8uc1'):
        arr = buf.reshape((h, step))[:, :w].reshape((h, w))
        import cv2
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if enc in ('rgba8',):
        arr = buf.reshape((h, step))[:, :w * 4].reshape((h, w, 4))
        return arr[:, :, [2, 1, 0]].copy()
    if enc in ('bgra8',):
        arr = buf.reshape((h, step))[:, :w * 4].reshape((h, w, 4))
        return arr[:, :, :3].copy()
    raise ValueError(f'unsupported encoding: {msg.encoding!r}')


def numpy_to_ros_image(bgr: np.ndarray, stamp, frame_id: str = 'overhead_camera/link/camera') -> Image:
    """Convert BGR uint8 array to sensor_msgs/Image (bgr8)."""
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = bgr.shape[:2]
    msg.encoding = 'bgr8'
    msg.is_bigendian = False
    msg.step = msg.width * 3
    # array('B') takes the generated setter's fast path; bytes is validated per element (~0.5 s/frame).
    msg.data = array.array('B', bgr.tobytes())
    return msg


# Calibrated HSV ranges (H: 0-179, S: 0-255, V: 0-255 in OpenCV).
COLOR_RANGES = {
    'red': [((0, 70, 70), (10, 255, 255)), ((170, 70, 70), (180, 255, 255))],
    'yellow': [((22, 120, 120), (35, 255, 255))],
    'green': [((35, 70, 70), (85, 255, 255))],
    'blue': [((98, 120, 70), (130, 255, 255))],
}
COLOR_BGR = {
    'red': (30, 30, 235),
    'yellow': (0, 215, 255),
    'green': (35, 220, 45),
    'blue': (240, 90, 30),
}
COLOR_RGBA = {
    'red': (0.95, 0.12, 0.12, 0.85),
    'yellow': (0.95, 0.85, 0.05, 0.85),
    'green': (0.12, 0.85, 0.18, 0.85),
    'blue': (0.15, 0.35, 0.95, 0.85),
}
MIN_AREA = 250.0


class ColorDetector(Node):
    def __init__(self) -> None:
        super().__init__('color_detector')

        # Parameters for camera & world geometry
        self.declare_parameter('camera_z', CAMERA_Z)
        self.declare_parameter('target_z', BLOCK_REST_Z)  # Block center height
        self.declare_parameter('hfov', CAMERA_HFOV)       # Horizontal FOV in radians
        self.declare_parameter('min_area', MIN_AREA)
        self.declare_parameter('enable_gui', True) # Native cv2.imshow demo window

        self.camera_z = self.get_parameter('camera_z').value
        self.target_z = self.get_parameter('target_z').value
        self.hfov = self.get_parameter('hfov').value
        self.min_area = self.get_parameter('min_area').value
        self.enable_gui = bool(self.get_parameter('enable_gui').value)
        self.gui_initialized = False

        # Static TF publisher: world -> cameras
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self._publish_static_tf()

        self.sub = self.create_subscription(
            Image, '/camera/image_raw', self.on_image, qos_profile_sensor_data)
        self.pub_markers = self.create_publisher(MarkerArray, '/detections', 10)
        self.pub_debug = self.create_publisher(
            Image, '/detections/debug', qos_profile_sensor_data)
        self.pub_blocks = self.create_publisher(
            String, '/detections/blocks', 10)
        self.get_logger().info('color_detector ready (multi-color calibration, demo bounding boxes & GUI active)')

    def _publish_static_tf(self) -> None:
        """Broadcast static transform from world to cameras."""
        transforms = []

        # 1. Overhead Camera (pose: 0 0 2.5 0 pi/2 pi/2)
        t1 = TransformStamped()
        t1.header.stamp = self.get_clock().now().to_msg()
        t1.header.frame_id = 'world'
        t1.child_frame_id = 'overhead_camera/link/camera'
        t1.transform.translation.x = 0.0
        t1.transform.translation.y = 0.0
        t1.transform.translation.z = float(self.camera_z)
        t1.transform.rotation.x = -0.5
        t1.transform.rotation.y = 0.5
        t1.transform.rotation.z = 0.5
        t1.transform.rotation.w = 0.5
        transforms.append(t1)

        # 2. Isometric 3D Perspective Camera (1.1, -1.4, 1.35)
        t2 = TransformStamped()
        t2.header.stamp = self.get_clock().now().to_msg()
        t2.header.frame_id = 'world'
        t2.child_frame_id = 'perspective_camera/link/perspective_camera'
        t2.transform.translation.x = 1.1
        t2.transform.translation.y = -1.4
        t2.transform.translation.z = 1.35
        t2.transform.rotation.x = -0.2226
        t2.transform.rotation.y = 0.0957
        t2.transform.rotation.z = 0.8913
        t2.transform.rotation.w = 0.3831
        transforms.append(t2)

        # 3. Front-Facing Eye-Level Camera (0.0, 1.25, 0.90)
        t3 = TransformStamped()
        t3.header.stamp = self.get_clock().now().to_msg()
        t3.header.frame_id = 'world'
        t3.child_frame_id = 'front_camera/link/front_camera'
        t3.transform.translation.x = 0.0
        t3.transform.translation.y = 1.25
        t3.transform.translation.z = 0.90
        t3.transform.rotation.x = 0.1231
        t3.transform.rotation.y = 0.1231
        t3.transform.rotation.z = -0.6963
        t3.transform.rotation.w = 0.6963
        transforms.append(t3)

        self.tf_broadcaster.sendTransform(transforms)

    def pixel_to_world(self, u: float, v: float, w: int, h: int) -> tuple[float, float, float]:
        """Convert pixel coordinate (u, v) on camera to world (X_w, Y_w, Z_w).
        With camera oriented at (0, 0, 2.5, 0, pi/2, pi/2):
          - Image X (left->right) maps to World +X (Left flank -> Right flank)
          - Image Y (top->down) maps to World -Y (Table front -> Robot base)
        """
        f = (w / 2.0) / math.tan(self.hfov / 2.0)
        d = self.camera_z - self.target_z

        x_cam = (u - w / 2.0) / f * d
        y_cam = (v - h / 2.0) / f * d

        world_x = x_cam
        world_y = -y_cam
        world_z = self.target_z
        return world_x, world_y, world_z

    def world_to_pixel(self, wx: float, wy: float, w: int, h: int) -> tuple[int, int]:
        """Project world coordinate (wx, wy) into camera image pixel (u, v)."""
        f = (w / 2.0) / math.tan(self.hfov / 2.0)
        d = self.camera_z - self.target_z
        x_cam = wx
        y_cam = -wy
        u = int(round(w / 2.0 + (x_cam / d) * f))
        v = int(round(h / 2.0 + (y_cam / d) * f))
        return u, v

    def on_image(self, msg: Image) -> None:
        import cv2
        try:
            bgr = ros_image_to_numpy(msg)
        except ValueError as e:
            self.get_logger().warn(str(e))
            return
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, w = bgr.shape[:2]

        markers = MarkerArray()
        detected_blocks = []
        color_id_map = {'red': 10, 'yellow': 15, 'green': 20, 'blue': 30}
        color_instance_count = {'red': 0, 'yellow': 0, 'green': 0, 'blue': 0}

        for color, ranges in COLOR_RANGES.items():
            mask = None
            for lo, hi in ranges:
                part = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
                mask = part if mask is None else cv2.bitwise_or(mask, part)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Sort for stable instance numbering frame to frame (top-to-bottom, left-to-right).
            for c in sorted(contours, key=lambda c: cv2.boundingRect(c)[1::-1]):
                area = cv2.contourArea(c)
                if area < self.min_area or area > 3500.0:
                    continue
                m = cv2.moments(c)
                if m['m00'] == 0:
                    continue
                cx = float(m['m10'] / m['m00'])
                cy = float(m['m01'] / m['m00'])

                wx, wy, wz = self.pixel_to_world(cx, cy, w, h)

                # Bin trays share block colors, and the pedestal sits at y=-0.55: not blocks.
                if bin_containing(wx, wy) is not None or wy < -0.48:
                    continue

                color_instance_count[color] += 1
                inst_idx = color_instance_count[color]
                block_color_id = f"{color}_{inst_idx}"

                icx, icy = int(cx), int(cy)
                is_mag = True
                mat_tag = "FERRO"

                # 1. Rotated Bounding Box
                rect = cv2.minAreaRect(c)
                box = np.int32(cv2.boxPoints(rect))
                cv2.drawContours(bgr, [box], 0, COLOR_BGR[color], 2)

                # Corner accent dots
                for pt in box:
                    cv2.circle(bgr, tuple(pt), 3, (255, 255, 255), -1)

                # Centroid crosshair
                cv2.drawMarker(bgr, (icx, icy), COLOR_BGR[color], cv2.MARKER_CROSS, 14, 2)

                # Label tag badge above box
                label_text = f"{color.upper()}_{inst_idx}: ({wx:+.2f},{wy:+.2f})m [{mat_tag}]"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                bx_tag = max(5, min(w - tw - 8, icx - tw // 2))
                by_tag = max(th + 36, icy - 18)
                cv2.rectangle(bgr, (bx_tag - 3, by_tag - th - 3), (bx_tag + tw + 3, by_tag + 3), (25, 25, 30), -1)
                cv2.rectangle(bgr, (bx_tag - 3, by_tag - th - 3), (bx_tag + tw + 3, by_tag + 3), COLOR_BGR[color], 1)
                cv2.putText(bgr, label_text, (bx_tag, by_tag), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (245, 245, 245), 1, cv2.LINE_AA)

                detected_blocks.append({
                    'id': f"{color}_block_{inst_idx}",
                    'color': block_color_id,
                    'base_color': color,
                    'instance': inst_idx,
                    'x': round(wx, 3),
                    'y': round(wy, 3),
                    'z': round(wz, 3),
                    'is_magnetic': is_mag
                })

                block_id = color_id_map.get(color, 100) * 10 + inst_idx

                # 3D Marker: CUBE in world frame matching 12cm block
                mk = Marker()
                mk.header.stamp = msg.header.stamp
                mk.header.frame_id = 'world'
                mk.ns = 'detections'
                mk.id = block_id
                mk.type = Marker.CUBE
                mk.action = Marker.ADD
                mk.pose.position.x = wx
                mk.pose.position.y = wy
                mk.pose.position.z = wz
                mk.pose.orientation.w = 1.0
                mk.scale.x = 0.12
                mk.scale.y = 0.12
                mk.scale.z = 0.12
                r, g, b, a = COLOR_RGBA[color]
                mk.color.r, mk.color.g, mk.color.b, mk.color.a = r, g, b, a
                mk.lifetime.sec = 1
                markers.markers.append(mk)

                # 3D Text label
                txt = Marker()
                txt.header.stamp = msg.header.stamp
                txt.header.frame_id = 'world'
                txt.ns = 'labels'
                txt.id = block_id + 100
                txt.type = Marker.TEXT_VIEW_FACING
                txt.action = Marker.ADD
                txt.pose.position.x = wx
                txt.pose.position.y = wy
                txt.pose.position.z = wz + 0.11
                txt.pose.orientation.w = 1.0
                txt.scale.z = 0.055
                txt.color.r, txt.color.g, txt.color.b, txt.color.a = 1.0, 1.0, 1.0, 1.0
                txt.text = f"{color.upper()}_BLOCK_{inst_idx}"
                txt.lifetime.sec = 1
                markers.markers.append(txt)

        # 2. Bin tray overlays
        bay_overlays = [(f"{c.upper()} BIN", cx, cy, COLOR_BGR[c]) for c, (cx, cy) in BIN_CENTERS.items()]
        f_scale = (w / 2.0) / math.tan(self.hfov / 2.0)
        d_cam = self.camera_z - self.target_z
        dx_pix = int(round((BIN_SIZE_X / d_cam) * f_scale))
        dy_pix = int(round((BIN_SIZE_Y / d_cam) * f_scale))
        half_x = dx_pix // 2
        half_y = dy_pix // 2

        for bname, bx, by, bcolor in bay_overlays:
            bu, bv = self.world_to_pixel(bx, by, w, h)
            cv2.rectangle(bgr, (bu - half_x, bv - half_y), (bu + half_x, bv + half_y), bcolor, 2)
            cv2.putText(bgr, bname, (bu - half_x + 3, bv - half_y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, bcolor, 1, cv2.LINE_AA)

        # 3. Top HUD Banner
        cv2.rectangle(bgr, (0, 0), (w, 32), (20, 22, 28), -1)
        cv2.line(bgr, (0, 32), (w, 32), (60, 65, 80), 1)
        status_txt = f"TOP-DOWN CV DEMO | {len(detected_blocks)} OBJECTS DETECTED | ALL FERROMAGNETIC"
        cv2.putText(bgr, status_txt, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 230, 255), 1, cv2.LINE_AA)

        # 3D Floating Labels for Side Sorting Bays
        bin_configs = [(102 + i, cx, cy, 0.45, f"{c.upper()}_BIN")
                       for i, (c, (cx, cy)) in enumerate(BIN_CENTERS.items())]
        for bid, bx, by, bz, bname in bin_configs:
            btxt = Marker()
            btxt.header.stamp = msg.header.stamp
            btxt.header.frame_id = 'world'
            btxt.ns = 'environment'
            btxt.id = bid + 10
            btxt.type = Marker.TEXT_VIEW_FACING
            btxt.action = Marker.ADD
            btxt.pose.position.x = bx
            btxt.pose.position.y = by
            btxt.pose.position.z = bz + 0.08
            btxt.pose.orientation.w = 1.0
            btxt.scale.z = 0.045
            btxt.color.r, btxt.color.g, btxt.color.b, btxt.color.a = 1.0, 1.0, 1.0, 0.95
            btxt.text = bname
            btxt.lifetime.sec = 1
            markers.markers.append(btxt)

        self.pub_markers.publish(markers)

        # Publish structured catalog of detected blocks (for closed-loop pick and place)
        self.pub_blocks.publish(String(data=json.dumps(detected_blocks)))
        self.pub_debug.publish(numpy_to_ros_image(bgr, msg.header.stamp, msg.header.frame_id))

        # Native OpenCV GUI popup window
        if self.enable_gui:
            try:
                cv2.imshow("Top-Down Vision Detector Demo", bgr)
                cv2.waitKey(1)
            except Exception:
                pass


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ColorDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Error in spin: {e}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
