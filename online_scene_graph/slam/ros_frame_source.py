"""In-process ROS2 RGB-D + pose ingest. Callbacks only fill LatestFrameSlot.

Raw `sensor_msgs/Image` or compressed `sensor_msgs/CompressedImage` (JPEG color,
PNG `compressedDepth`). Topic suffix selects the type: `/compressed`,
`/compressedDepth`, otherwise raw.
"""

from __future__ import annotations

import struct
from typing import Optional

import numpy as np

from online_scene_graph.slam.frame_ipc import LatestFrameSlot, quaternion_to_rotation_matrix
from online_scene_graph.slam.frame_source import Frame
from online_scene_graph.slam.ros_runtime import add_node, ensure_rclpy, remove_node

# image_transport compressedDepth: ConfigHeader is enum + 2 floats (12 bytes).
_COMPRESSED_DEPTH_HEADER = 12


def sensor_qos_profile(reliable: bool = False, depth: int = 1):
    """BEST_EFFORT by default (phone/Replica publishers). The HSR publishes RELIABLE: use reliable=True there."""
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def topic_image_kind(topic: str) -> str:
    name = topic.rstrip("/").rsplit("/", 1)[-1]
    if name == "compressedDepth":
        return "compressed_depth"
    if name == "compressed":
        return "compressed"
    return "raw"


def image_msg_to_bgr(msg) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    if msg.encoding in ("bgr8", "8UC3"):
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3).copy()
    if msg.encoding in ("rgb8",):
        rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
        return rgb[:, :, ::-1].copy()
    raise ValueError(f"unsupported color encoding {msg.encoding}")


def compressed_color_to_bgr(msg) -> np.ndarray:
    import cv2

    buf = np.frombuffer(msg.data, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"failed to decode compressed color format={msg.format!r}")
    return bgr


def image_msg_to_depth_m(msg) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    if msg.encoding in ("32FC1", "32FC"):
        return np.frombuffer(msg.data, dtype=np.float32).reshape(h, w).copy()
    if msg.encoding in ("16UC1", "mono16"):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w).astype(np.float32) / 1000.0
    raise ValueError(f"unsupported depth encoding {msg.encoding}")


def compressed_depth_to_depth_m(msg) -> np.ndarray:
    """Decode image_transport `compressedDepth` (PNG) to meters."""
    import cv2

    fmt = msg.format or ""
    parts = [p.strip() for p in fmt.split(";")]
    encoding = parts[0].upper() if parts else ""
    rest = " ".join(parts[1:]).lower()
    if "rvl" in rest:
        raise ValueError("compressedDepth rvl is not supported; republish as png")
    raw = bytes(msg.data)
    if len(raw) <= _COMPRESSED_DEPTH_HEADER:
        raise ValueError("compressedDepth payload too short")
    png = np.frombuffer(raw[_COMPRESSED_DEPTH_HEADER:], dtype=np.uint8)
    img = cv2.imdecode(png, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"failed to decode compressedDepth format={fmt!r}")
    if encoding in ("16UC1", "MONO16") or (not encoding.startswith("32") and img.dtype == np.uint16):
        return img.astype(np.float32) / 1000.0
    if encoding in ("32FC1", "32FC"):
        _kind, quant_a, quant_b = struct.unpack("<iff", raw[:_COMPRESSED_DEPTH_HEADER])
        depth = quant_a / (img.astype(np.float32) - quant_b)
        depth[img == 0] = 0.0
        return depth.astype(np.float32)
    if img.dtype == np.float32:
        return img
    raise ValueError(f"unsupported compressedDepth format={fmt!r} dtype={img.dtype}")


def camera_info_to_K(msg) -> np.ndarray:
    return np.asarray(msg.k, dtype=np.float32).reshape(3, 3)


def pose_msg_to_T(msg) -> np.ndarray:
    p = msg.pose.position
    q = msg.pose.orientation
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = quaternion_to_rotation_matrix([q.x, q.y, q.z, q.w]).astype(np.float32)
    T[:3, 3] = [p.x, p.y, p.z]
    return T


def header_stamp_sec(header) -> float:
    return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9


class RosRgbDPoseSource:
    def __init__(
        self,
        rgb_topic: str,
        depth_topic: str,
        camera_info_topic: str,
        pose_topic: str,
        node_name: str = "osg_rgbd_sub",
        sync_slop: float = 0.1,
        reliable: bool = False,
        qos_depth: int = 1,
    ):
        self.reliable = bool(reliable)
        self.qos_depth = int(qos_depth)
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.camera_info_topic = camera_info_topic
        self.pose_topic = pose_topic
        self.node_name = node_name
        self.sync_slop = float(sync_slop)
        self.slot = LatestFrameSlot()
        self._node = None
        self._started = False
        self._subs = []
        self._sync = None
        self._counts = {"rgb": 0, "depth": 0, "info": 0, "pose": 0, "synced": 0}
        self._rgb_kind = topic_image_kind(rgb_topic)
        self._depth_kind = topic_image_kind(depth_topic)

    def start(self) -> None:
        if self._started:
            return
        from message_filters import ApproximateTimeSynchronizer, Subscriber, TimeSynchronizer
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import CameraInfo, CompressedImage, Image

        rclpy = ensure_rclpy()
        qos = sensor_qos_profile(self.reliable, self.qos_depth)
        self._node = rclpy.create_node(self.node_name)
        rgb_type = CompressedImage if self._rgb_kind == "compressed" else Image
        depth_type = CompressedImage if self._depth_kind == "compressed_depth" else Image
        if self._depth_kind == "compressed":
            raise ValueError(
                f"depth topic {self.depth_topic} is /compressed (8-bit). "
                "Use /compressedDepth for 16-bit depth."
            )
        rgb_sub = Subscriber(self._node, rgb_type, self.rgb_topic, qos_profile=qos)
        depth_sub = Subscriber(self._node, depth_type, self.depth_topic, qos_profile=qos)
        info_sub = Subscriber(self._node, CameraInfo, self.camera_info_topic, qos_profile=qos)
        # Pose comes from a separate node (report_camera_pose) that publishes BEST_EFFORT; a RELIABLE
        # reader cannot match it, and a best-effort reader matches either.
        pose_qos = sensor_qos_profile(False, self.qos_depth)
        pose_sub = Subscriber(self._node, PoseStamped, self.pose_topic, qos_profile=pose_qos)
        if self.sync_slop > 0.0:
            sync = ApproximateTimeSynchronizer(
                [rgb_sub, depth_sub, info_sub, pose_sub],
                queue_size=10,
                slop=self.sync_slop,
            )
            sync_kind = f"approx slop={self.sync_slop:.3f}s"
        else:
            sync = TimeSynchronizer([rgb_sub, depth_sub, info_sub, pose_sub], queue_size=2)
            sync_kind = "exact"
        sync.registerCallback(self._on_sync)
        for name, sub in (
            ("rgb", rgb_sub), ("depth", depth_sub), ("info", info_sub), ("pose", pose_sub)
        ):
            sub.registerCallback(lambda _m, name=name: self._count(name))
        self._subs = [rgb_sub, depth_sub, info_sub, pose_sub]
        self._sync = sync
        add_node(self._node)
        self._started = True
        print(
            "ROS FrameSource (in-process):\n"
            f"  rgb   {self.rgb_topic}  ({self._rgb_kind})\n"
            f"  depth {self.depth_topic}  ({self._depth_kind})\n"
            f"  info  {self.camera_info_topic}\n"
            f"  pose  {self.pose_topic}\n"
            f"  sync  {sync_kind}\n"
            f"  qos   {'RELIABLE' if self.reliable else 'BEST_EFFORT'} depth={self.qos_depth}"
        )

    def _count(self, name: str) -> None:
        self._counts[name] += 1

    def status(self) -> str:
        """Messages received per topic since start, for diagnosing a stalled stream."""
        return " ".join(f"{k}={v}" for k, v in self._counts.items())

    def _decode_rgb(self, msg) -> np.ndarray:
        if self._rgb_kind == "compressed":
            return compressed_color_to_bgr(msg)
        return image_msg_to_bgr(msg)

    def _decode_depth(self, msg) -> np.ndarray:
        if self._depth_kind == "compressed_depth":
            return compressed_depth_to_depth_m(msg)
        return image_msg_to_depth_m(msg)

    def _on_sync(self, rgb_msg, depth_msg, info_msg, pose_msg) -> None:
        self._counts["synced"] += 1
        try:
            rgb = self._decode_rgb(rgb_msg)
            depth = self._decode_depth(depth_msg)
            if depth.shape[:2] != rgb.shape[:2]:
                import cv2

                depth = cv2.resize(
                    depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST
                )
            self.slot.put(
                Frame(
                    rgb=rgb,
                    depth=depth,
                    K=camera_info_to_K(info_msg),
                    pose=pose_msg_to_T(pose_msg),
                    t=header_stamp_sec(rgb_msg.header),
                )
            )
        except Exception as exc:
            print(f"ROS FrameSource skip frame: {exc}")

    def next(self, timeout: float = 1.0) -> Optional[Frame]:
        return self.slot.get(timeout=timeout)

    def close(self) -> None:
        if self._node is not None:
            remove_node(self._node)
            self._node.destroy_node()
            self._node = None
        self._subs = []
        self._sync = None
        self._started = False
