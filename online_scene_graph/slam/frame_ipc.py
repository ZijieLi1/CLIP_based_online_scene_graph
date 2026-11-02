"""Drop-oldest frame slot and pose helpers. No sockets, no ROS subprocesses."""

from __future__ import annotations

from threading import Event, Lock
from typing import Any, Optional

import numpy as np


class LatestFrameSlot:
    """Depth-1 queue: put() overwrites; get() takes the newest item or times out."""

    def __init__(self):
        self._lock = Lock()
        self._frame: Optional[Any] = None
        self._dropped = 0
        self._event = Event()

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    def put(self, frame: Any) -> None:
        with self._lock:
            if self._frame is not None:
                self._dropped += 1
                if hasattr(frame, "dropped_before"):
                    frame.dropped_before = self._dropped
                elif isinstance(frame, dict):
                    frame["dropped_before"] = self._dropped
            self._frame = frame
            self._event.set()

    def get(self, timeout: float = 1.0) -> Optional[Any]:
        if not self._event.wait(timeout=timeout):
            return None
        with self._lock:
            frame = self._frame
            self._frame = None
            self._event.clear()
            return frame


def rotation_matrix_to_quaternion(R) -> np.ndarray:
    R = np.asarray(R)
    q = np.empty((4,), dtype=np.float32)
    q[3] = np.sqrt(np.maximum(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    q[0] = np.sqrt(np.maximum(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    q[1] = np.sqrt(np.maximum(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    q[2] = np.sqrt(np.maximum(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    q[0] *= np.sign(q[0] * (R[2, 1] - R[1, 2]))
    q[1] *= np.sign(q[1] * (R[0, 2] - R[2, 0]))
    q[2] *= np.sign(q[2] * (R[1, 0] - R[0, 1]))
    return q


def quaternion_to_rotation_matrix(q) -> np.ndarray:
    w, x, y, z = q[3], q[0], q[1], q[2]
    return np.array(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
        ]
    )
