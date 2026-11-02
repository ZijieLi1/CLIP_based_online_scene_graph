# online-scene-graph

Incremental open-vocabulary object map (YOLO → CLIP → 3D fuse/merge) that serves
`find(text)` while it is still mapping, as the upstream of a PDDL planner bridge.
Trimmed fork of [ConceptGraphs](https://github.com/concept-graphs/concept-graphs) (see `LICENSE`).
Design and phases: [`docs/online-graph-pddl-architecture.md`](docs/online-graph-pddl-architecture.md).

## One input path: ROS

The mapper only subscribes to four topics (`/camera/color/image_raw`, `/camera/depth/image_raw`,
`/camera/color/camera_info`, `/camera/pose`). Each camera is a publisher in `online_scene_graph/sources/`
(`replica_publisher.py`, `record3d_publisher.py`); a robot publishes them itself. Same mapper path everywhere,
so a Replica run tests the real ingest. Outputs go to `outputs/online/exps/online/`.

```bash
xhost +local:docker
docker compose build && docker compose up -d
docker compose exec osg bash

# shell 1: replay Replica as a camera + VSLAM (low rate so the drop-oldest slot never drops)
python3 online_scene_graph/sources/replica_publisher.py --scene-dir /data/Replica/office0 --rate 2 --stride 5
# shell 2: mapper
python3 online_scene_graph/slam/online_mapping.py
# shell 3: query the live map
python3 online_scene_graph/scripts/find_query.py mug
```

Service: `/online_scene_graph/find`. Set `REPLICA_DIR` to change the host Replica path.

## Wi-Fi stalls: "Waiting for frame" with depth stuck

Symptom: `received: rgb=... depth=N` where depth stops growing while rgb/info/pose keep climbing, and the mapper never
recovers. Measured cause: the HSR publishes its camera topics **RELIABLE**, and a BEST_EFFORT reader never asks for
retransmission, so after Wi-Fi loss the large `compressedDepth` stream dies for good (the robot itself keeps publishing a
clean 30 Hz; `tests/endtoend_gap.py` run on the robot and the workstation at once shows this). A RELIABLE reader
recovers. `ros_qos_reliable: true` (default for the HSR config) fixes it; the pose topic stays best-effort because
`report_camera_pose` publishes best-effort. Use `ros_qos_reliable: false` for the phone/Replica publishers.

Separately, the link only carries ~5 MB/s of UDP reliably (`tests/nettest.py`), and the full camera load is ~6.5 MB/s,
so throttle on the robot (`rgbd_throttle.launch.py`, `topic_tools throttle`; `rate:=` is a *minimum interval*, 15 gives
~8.5 Hz in practice, 20 gives ~15 Hz) and read the `/throttled/...` topics. Wired Ethernet would remove both issues.
