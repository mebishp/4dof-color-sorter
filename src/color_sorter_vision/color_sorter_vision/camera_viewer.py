#!/usr/bin/env python3
"""Multi-Camera Live Grid Viewer for Robotic Sorter Simulation.

Displays all active camera angles in a high-resolution 2x2 dashboard window:
  - Top-Left:     Overhead Camera (/camera/image_raw)
  - Top-Right:    Top-Down CV Detection Overlay (/detections/debug)
  - Bottom-Left:  Front-Facing Eye-Level Camera (/camera/front/image_raw)
  - Bottom-Right: 3D Isometric Perspective Camera (/camera/perspective/image_raw)

Keyboard shortcuts:
  - 'q' or ESC: Exit viewer
  - 's': Save current multi-camera snapshot
"""

import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from color_sorter_vision.color_detector import ros_image_to_numpy


class MultiCameraViewer(Node):
    def __init__(self) -> None:
        super().__init__('multi_camera_viewer')

        self.frames = {
            'overhead': None,
            'debug': None,
            'front': None,
            'perspective': None,
        }
        self.fps_timers = {k: time.time() for k in self.frames}
        self.fps_counts = {k: 0 for k in self.frames}
        self.fps_values = {k: 0.0 for k in self.frames}
        # Rendered cells are cached and only rebuilt when their camera delivers a frame.
        self.cells = {k: None for k in self.frames}
        self.dirty = {k: True for k in self.frames}
        self.grid = None

        # Subscriptions
        self.sub_overhead = self.create_subscription(
            Image, '/camera/image_raw',
            lambda m: self._on_frame('overhead', m), qos_profile_sensor_data
        )
        self.sub_debug = self.create_subscription(
            Image, '/detections/debug',
            lambda m: self._on_frame('debug', m), qos_profile_sensor_data
        )
        self.sub_front = self.create_subscription(
            Image, '/camera/front/image_raw',
            lambda m: self._on_frame('front', m), qos_profile_sensor_data
        )
        self.sub_persp = self.create_subscription(
            Image, '/camera/perspective/image_raw',
            lambda m: self._on_frame('perspective', m), qos_profile_sensor_data
        )

        self.window_name = "Robotic Sorter - Multi-Camera Live Matrix"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 960)

        # 30 Hz UI refresh timer
        self.timer = self.create_timer(0.033, self.render_grid)
        self.get_logger().info("Multi-camera live matrix viewer initialized. Press 'q' or ESC to exit.")

    def _on_frame(self, key: str, msg: Image) -> None:
        try:
            self.frames[key] = ros_image_to_numpy(msg)
            self.dirty[key] = True
            now = time.time()
            self.fps_counts[key] += 1
            if now - self.fps_timers[key] >= 1.0:
                self.fps_values[key] = self.fps_counts[key] / (now - self.fps_timers[key])
                self.fps_counts[key] = 0
                self.fps_timers[key] = now
        except Exception:
            pass

    def _format_cell(self, key: str, title: str, subtitle: str, target_size=(640, 480)) -> np.ndarray:
        w_t, h_t = target_size
        img = self.frames[key]

        if img is None:
            # Standby blank canvas
            canvas = np.full((h_t, w_t, 3), 28, dtype=np.uint8)
            cv2.putText(canvas, f"{title}: Connecting...", (w_t // 2 - 140, h_t // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 130, 150), 1, cv2.LINE_AA)
        else:
            canvas = cv2.resize(img, (w_t, h_t), interpolation=cv2.INTER_AREA)

        # Header banner
        cv2.rectangle(canvas, (0, 0), (w_t, 26), (15, 17, 22), -1)
        cv2.line(canvas, (0, 26), (w_t, 26), (45, 50, 65), 1)

        fps = self.fps_values[key]
        header_text = f"{title} | {subtitle} ({fps:.1f} FPS)"
        cv2.putText(canvas, header_text, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1, cv2.LINE_AA)

        # Outer border
        cv2.rectangle(canvas, (0, 0), (w_t - 1, h_t - 1), (50, 55, 70), 1)
        return canvas

    CELLS = {
        'overhead': ('OVERHEAD VIEW', 'Topic: /camera/image_raw'),
        'debug': ('OPENCV DETECTOR OVERLAY', 'Topic: /detections/debug'),
        'front': ('FRONT EYE-LEVEL VIEW', 'Topic: /camera/front/image_raw'),
        'perspective': ('3D ISOMETRIC VIEW', 'Topic: /camera/perspective/image_raw'),
    }

    def render_grid(self) -> None:
        cell_size = (640, 480)
        changed = False
        for k, (title, subtitle) in self.CELLS.items():
            if self.dirty[k] or self.cells[k] is None:
                self.dirty[k] = False
                self.cells[k] = self._format_cell(k, title, subtitle, cell_size)
                changed = True

        if changed or self.grid is None:
            c = self.cells
            self.grid = np.vstack([np.hstack([c['overhead'], c['debug']]),
                                   np.hstack([c['front'], c['perspective']])])
            cv2.imshow(self.window_name, self.grid)
        grid = self.grid
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC or q
            self.get_logger().info("Exiting multi-camera viewer...")
            rclpy.shutdown()
        elif key == ord('s'):
            out_path = f"/home/hyprnon/multicam_{int(time.time())}.png"
            cv2.imwrite(out_path, grid)
            self.get_logger().info(f"Snapshot saved to: {out_path}")


def main(args=None) -> None:
    rclpy.init(args=args)
    viewer = MultiCameraViewer()
    try:
        rclpy.spin(viewer)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
