"""Replay a Replica scene as ROS2 topics, exactly like a robot camera + VSLAM would.

The mapper sees the same RGB / depth / camera_info / pose topics it sees live,
so this exercises the real ROS ingest path with ground-truth poses.

    python3 online_scene_graph/sources/replica_publisher.py \
        --scene-dir /data/Replica/office0 --rate 2 --stride 5

Keep --rate low enough for the mapper: the ingest slot drops the oldest frame
when the mapper falls behind, which would make a replay non-reproducible.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import cv2
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


def load_scene(scene_dir: Path):
    colors = sorted(glob.glob(str(scene_dir / "results" / "frame*.jpg")))
    depths = sorted(glob.glob(str(scene_dir / "results" / "depth*.png")))
    poses = np.loadtxt(scene_dir / "traj.txt").reshape(-1, 4, 4).astype(np.float32)
    n = min(len(colors), len(depths), len(poses))
    if n == 0:
        raise SystemExit(f"no frames under {scene_dir}/results")
    cam = json.load(open(scene_dir.parent / "cam_params.json"))["camera"]
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]], dtype=np.float32)
    return colors[:n], depths[:n], poses[:n], K, float(cam["scale"])


def main():
    p = argparse.ArgumentParser(description="Publish a Replica scene on ROS2.")
    p.add_argument("--scene-dir", required=True, help="e.g. /data/Replica/office0")
    p.add_argument("--rate", type=float, default=2.0, help="frames per second published")
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--loop", action="store_true", help="repeat the sequence")
    p.add_argument("--frame-id", default="camera_optical_frame")
    p.add_argument("--rgb-topic", default="/camera/color/image_raw")
    p.add_argument("--depth-topic", default="/camera/depth/image_raw")
    p.add_argument("--info-topic", default="/camera/color/camera_info")
    p.add_argument("--pose-topic", default="/camera/pose")
    args = p.parse_args()

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import CameraInfo, Image

    colors, depths, poses, K, depth_scale = load_scene(Path(args.scene_dir))
    idx = list(range(len(colors)))[args.start : (None if args.end < 0 else args.end) : args.stride]

    rclpy.init()
    node = rclpy.create_node("replica_ros_publisher")
    qos = sensor_qos_profile()
    pub_rgb = node.create_publisher(Image, args.rgb_topic, qos)
    pub_depth = node.create_publisher(Image, args.depth_topic, qos)
    pub_info = node.create_publisher(CameraInfo, args.info_topic, qos)
    pub_pose = node.create_publisher(PoseStamped, args.pose_topic, qos)

    print(f"Replaying {len(idx)} frames from {args.scene_dir} at {args.rate} Hz")
    period = 1.0 / max(args.rate, 1e-3)
    sent = 0
    try:
        while rclpy.ok():
            for i in idx:
                t0 = time.time()
                bgr = cv2.imread(colors[i], cv2.IMREAD_COLOR)
                depth = cv2.imread(depths[i], cv2.IMREAD_UNCHANGED).astype(np.float32) / depth_scale
                header = make_header(node.get_clock().now().to_msg(), args.frame_id)
                h, w = bgr.shape[:2]
                pub_info.publish(k_to_camera_info(K, w, h, header))
                pub_pose.publish(T_to_pose_msg(poses[i], header))
                pub_depth.publish(depth_to_image_msg(depth, header))
                pub_rgb.publish(bgr_to_image_msg(bgr, header))
                sent += 1
                rclpy.spin_once(node, timeout_sec=0.0)
                time.sleep(max(0.0, period - (time.time() - t0)))
                if not rclpy.ok():
                    break
            if not args.loop:
                break
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    print(f"published {sent} frames")


if __name__ == "__main__":
    main()
