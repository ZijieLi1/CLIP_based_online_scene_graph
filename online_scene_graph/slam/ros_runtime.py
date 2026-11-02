"""Shared rclpy context. One MultiThreadedExecutor for all mapper ROS nodes.

Do not call rclpy.shutdown() from a node close() — sibling nodes may still be spinning.
"""

from __future__ import annotations

from threading import Lock, Thread
from typing import Optional

_lock = Lock()
_executor = None
_thread: Optional[Thread] = None


def ensure_rclpy():
    import rclpy

    if not rclpy.ok():
        rclpy.init()
    return rclpy


def add_node(node) -> None:
    """Add a node to the process-wide executor (starts spin on first add)."""
    global _executor, _thread
    from rclpy.executors import MultiThreadedExecutor

    with _lock:
        ensure_rclpy()
        if _executor is None:
            _executor = MultiThreadedExecutor()
            _thread = Thread(target=_executor.spin, daemon=True)
            _thread.start()
        _executor.add_node(node)


def remove_node(node) -> None:
    with _lock:
        if _executor is not None and node is not None:
            try:
                _executor.remove_node(node)
            except Exception:
                pass
