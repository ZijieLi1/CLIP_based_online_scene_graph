"""In-process ROS2 service /online_scene_graph/find. Mapping thread is the CLIP owner."""

from __future__ import annotations

import array
import json

from online_scene_graph.slam.grounding import GroundingService
from online_scene_graph.slam.ros_runtime import add_node, ensure_rclpy, remove_node


class FindRosService:
    def __init__(
        self,
        grounding: GroundingService,
        service_name: str = "/online_scene_graph/find",
        node_name: str = "osg_find",
        snapshot_service_name: str = "/online_scene_graph/snapshot",
    ):
        self.snapshot_service_name = snapshot_service_name
        self.grounding = grounding
        self.service_name = service_name
        self.node_name = node_name
        self._node = None
        self._started = False

    def start(self) -> bool:
        if self._started:
            return True
        try:
            from online_scene_graph_interfaces.srv import Find
        except ImportError as exc:
            print(
                f"Find ROS service skipped ({exc}). "
                "source the Humble overlay (entrypoint or "
                "`source /ws/ros/install/setup.bash` / `/opt/osg_ws/install/setup.bash`)."
            )
            return False

        rclpy = ensure_rclpy()
        self._node = rclpy.create_node(self.node_name)

        def on_find(request, response):
            reply = self.grounding.handle(
                {
                    "cmd": "find",
                    "text": request.text,
                    "k": request.k,
                    "min_sim": request.min_sim,
                    "min_obs": request.min_obs,
                }
            )
            response.ok = bool(reply.get("ok", False))
            response.message = str(reply.get("message", ""))
            response.generation = int(reply.get("generation", 0))
            response.json = json.dumps(reply.get("hits") or [])
            return response

        self._node.create_service(Find, self.service_name, on_find)

        try:
            from online_scene_graph_interfaces.srv import Snapshot
        except ImportError:
            print("Snapshot service skipped (rebuild online_scene_graph_interfaces); viz_query live mode unavailable.")
        else:

            def on_snapshot(request, response):
                try:
                    reply = self.grounding.snapshot(
                        since_generation=int(request.since_generation),
                        max_points=int(request.max_points_per_object),
                    )
                    response.ok = True
                    response.message = ""
                    response.generation = int(reply["generation"])
                    response.unchanged = bool(reply["unchanged"])
                    response.data = array.array("B", reply["data"])
                except Exception as exc:
                    response.ok = False
                    response.message = str(exc)
                    response.generation = 0
                    response.unchanged = False
                    response.data = array.array("B")
                return response

            self._node.create_service(Snapshot, self.snapshot_service_name, on_snapshot)
            print(f"Snapshot ROS service {self.snapshot_service_name}")
        add_node(self._node)
        self._started = True
        print(f"Find ROS service {self.service_name} (in-process)")
        return True

    def close(self) -> None:
        if self._node is not None:
            remove_node(self._node)
            self._node.destroy_node()
            self._node = None
        self._started = False
