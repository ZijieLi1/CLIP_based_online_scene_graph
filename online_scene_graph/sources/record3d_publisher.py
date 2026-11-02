"""Record3D USB → ROS2 topics (color, depth, camera_info, pose).

One process: USB + rclpy. For Docker Humble / any env where record3d and
rclpy share an interpreter. On a robot this file is unused.

    python3 online_scene_graph/sources/record3d_publisher.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from online_scene_graph.sources.ros_msgs import (
    T_to_pose_msg,
    bgr_to_image_msg,
    depth_to_image_msg,
    k_to_camera_info,
    make_header,
    sensor_qos_profile,
)
from online_scene_graph.sources.record3d_device import DemoApp


def main():
    parser = argparse.ArgumentParser(description="Publish Record3D USB RGB-D + pose on ROS2.")
    parser.add_argument("--dev-idx", type=int, default=0)
    parser.add_argument("--frame-id", default="camera_optical_frame")
    parser.add_argument("--rgb-topic", default="/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/depth/image_raw")
    parser.add_argument("--info-topic", default="/camera/color/camera_info")
    parser.add_argument("--pose-topic", default="/camera/pose")
    args = parser.parse_args()

    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import CameraInfo, Image
        import torch
    except ImportError as exc:
        print(f"missing import: {exc}", file=sys.stderr)
        raise SystemExit(1)

    rclpy.init()
    node = rclpy.create_node("record3d_ros_publisher")
    qos = sensor_qos_profile()
    pub_rgb = node.create_publisher(Image, args.rgb_topic, qos)
    pub_depth = node.create_publisher(Image, args.depth_topic, qos)
    pub_info = node.create_publisher(CameraInfo, args.info_topic, qos)
    pub_pose = node.create_publisher(PoseStamped, args.pose_topic, qos)

    app = DemoApp()
    print("Connecting Record3D USB...")
    app.connect_to_device(dev_idx=args.dev_idx)
    print(
        f"Publishing  rgb={args.rgb_topic}  depth={args.depth_topic}  "
        f"info={args.info_topic}  pose={args.pose_topic}"
    )
    n = 0
    try:
        while rclpy.ok():
            rgb, depth, K4, T_wc = app.get_frame_data()
            if rgb is None:
                rclpy.spin_once(node, timeout_sec=0.0)
                continue
            K = K4.cpu().numpy()[:3, :3] if torch.is_tensor(K4) else np.asarray(K4)[:3, :3]
            T_wc = np.asarray(T_wc)
            stamp = node.get_clock().now().to_msg()
            header = make_header(stamp, args.frame_id)
            h, w = rgb.shape[:2]
            pub_rgb.publish(bgr_to_image_msg(rgb, header))
            pub_depth.publish(depth_to_image_msg(depth, header))
            pub_info.publish(k_to_camera_info(K, w, h, header))
            pub_pose.publish(T_to_pose_msg(T_wc, header))
            n += 1
            if n == 1 or n % 60 == 0:
                node.get_logger().info(
                    f"published {n}  rgb={rgb.shape}  depth={depth.shape}  fx={K[0, 0]:.1f}"
                )
            rclpy.spin_once(node, timeout_sec=0.0)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    print(f"stopped after {n} frames")


if __name__ == "__main__":
    main()
