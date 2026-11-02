"""Force Open3D/GLFW onto X11 before any Open3D import.

Wayland GLFW cannot set the window position; Open3D still does, then GLEW
fails and the process segfaults. If DISPLAY is set (XWayland), drop Wayland
for this process only.
"""

from __future__ import annotations

import os


def force_x11_for_open3d() -> bool:
    display = os.environ.get("DISPLAY")
    if not display:
        return False

    os.environ.setdefault("GLFW_PLATFORM", "x11")
    os.environ.setdefault("GDK_BACKEND", "x11")
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    if os.environ.get("XDG_SESSION_TYPE") == "wayland":
        os.environ["XDG_SESSION_TYPE"] = "x11"
    os.environ.pop("WAYLAND_DISPLAY", None)
    return True


force_x11_for_open3d()
