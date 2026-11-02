"""Call /online_scene_graph/find on a running mapper.

    python3 online_scene_graph/scripts/find_query.py mug
"""

from __future__ import annotations

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="CLIP find() via ROS service.")
    parser.add_argument("text", help="query phrase")
    parser.add_argument("--service", default="/online_scene_graph/find")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min-sim", type=float, default=0.25)
    parser.add_argument("--min-obs", type=int, default=3)
    args = parser.parse_args()

    try:
        import rclpy
        from online_scene_graph_interfaces.srv import Find
    except ImportError as exc:
        print(
            f"{exc}\nSource Humble + online_scene_graph_interfaces, e.g.\n"
            "  source /opt/ros/humble/setup.bash\n"
            "  source /ws/ros/install/setup.bash",
            file=sys.stderr,
        )
        raise SystemExit(1)

    rclpy.init()
    node = rclpy.create_node("osg_find_cli")
    client = node.create_client(Find, args.service)
    if not client.wait_for_service(timeout_sec=5.0):
        print(f"service {args.service} not available (is the mapper running?)", file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit(1)
    req = Find.Request()
    req.text = args.text
    req.k = int(args.k)
    req.min_sim = float(args.min_sim)
    req.min_obs = int(args.min_obs)
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=30.0)
    result = future.result()
    node.destroy_node()
    rclpy.shutdown()
    if result is None:
        print("find() timed out", file=sys.stderr)
        raise SystemExit(1)
    if not result.ok:
        print(result.message or "find failed", file=sys.stderr)
        raise SystemExit(1)
    hits = json.loads(result.json or "[]")
    print(f"generation={result.generation}  hits={len(hits)}  query={args.text!r}")
    if not hits:
        print("  (none)")
        return
    for i, hit in enumerate(hits, start=1):
        c = hit.get("center") or [float("nan")] * 3
        print(
            f"  {i}. sim={hit['sim']:.3f}  n={hit['n_obs']:3d}  "
            f"{hit['class_name']:20s}  "
            f"xyz=({c[0]:6.2f}, {c[1]:6.2f}, {c[2]:6.2f})  {str(hit.get('id', ''))[:8]}"
        )


if __name__ == "__main__":
    main()
