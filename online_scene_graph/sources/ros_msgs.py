"""Build ROS messages from mapper-native arrays (shared by the camera publishers)."""

from __future__ import annotations

import numpy as np

from online_scene_graph.slam.frame_ipc import rotation_matrix_to_quaternion


def sensor_qos_profile():
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


def make_header(stamp, frame_id: str):
    from std_msgs.msg import Header

    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    return header


def bgr_to_image_msg(rgb_bgr: np.ndarray, header):
    from sensor_msgs.msg import Image

    msg = Image()
    msg.header = header
    msg.height, msg.width = int(rgb_bgr.shape[0]), int(rgb_bgr.shape[1])
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(rgb_bgr, dtype=np.uint8).tobytes()
    return msg


def depth_to_image_msg(depth_m: np.ndarray, header):
    from sensor_msgs.msg import Image

    depth = np.ascontiguousarray(depth_m, dtype=np.float32)
    msg = Image()
    msg.header = header
    msg.height, msg.width = int(depth.shape[0]), int(depth.shape[1])
    msg.encoding = "32FC1"
    msg.is_bigendian = 0
    msg.step = msg.width * 4
    msg.data = depth.tobytes()
    return msg


def k_to_camera_info(K: np.ndarray, width: int, height: int, header):
    from sensor_msgs.msg import CameraInfo

    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    msg = CameraInfo()
    msg.header = header
    msg.width = int(width)
    msg.height = int(height)
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def T_to_pose_msg(T_wc: np.ndarray, header):
    from geometry_msgs.msg import PoseStamped

    q = rotation_matrix_to_quaternion(T_wc[:3, :3])
    msg = PoseStamped()
    msg.header = header
    msg.pose.position.x = float(T_wc[0, 3])
    msg.pose.position.y = float(T_wc[1, 3])
    msg.pose.position.z = float(T_wc[2, 3])
    msg.pose.orientation.x = float(q[0])
    msg.pose.orientation.y = float(q[1])
    msg.pose.orientation.z = float(q[2])
    msg.pose.orientation.w = float(q[3])
    return msg
