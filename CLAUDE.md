# online-scene-graph

ROS 2 Humble online open-vocabulary object mapper: detector (YOLOE etc.) -> OpenCLIP ViT-H-14 (+ optional DINOv2) -> 3D fuse/merge, serving `find(text)`. Architecture and model usage: `docs/dataflow.md`. ROS topic/QoS contract: `docs/ros-ingest.md`.

## Testing and running: always in the container

The host does not have the dependencies (ROS, torch, ultralytics, open_clip, open3d). Do not run or import the code on the host.

- Container: `online-scene-graph-osg-1` (compose service `osg`, repo bind-mounted at `/ws`, Replica at `/data/Replica`).
- If it is already up (`docker ps`), use `docker exec online-scene-graph-osg-1 bash -c '...'`. Only start it (`docker compose up -d`) if it is not running.
- Edits made on the host appear in `/ws` immediately. Files the container writes (outputs, core dumps) are root-owned, so remove them with `docker exec ... rm`, not from the host.
- `BASH_ENV=/ws/docker/env.sh` sources ROS and sets `PYTHONPATH`, so non-interactive `bash -c` already has ROS.

### ROS domain: use 150 for tests

The robot usually runs on domain 96 or 100, and the container defaults to 96. A test on the default domain can see or disturb the robot's topics. For every test, set a separate domain on **both** the publisher and the mapper:

```bash
docker exec online-scene-graph-osg-1 bash -c 'cd /ws && export ROS_DOMAIN_ID=150 && \
  (python3 online_scene_graph/sources/replica_publisher.py --scene-dir /data/Replica/office0 --rate 2 --stride 10 --end 600 > /tmp/pub.log 2>&1 &) ; sleep 2; \
  timeout -s INT 90 python3 online_scene_graph/slam/online_mapping.py use_rerun=false ros_qos_reliable=false \
    ros_rgb_topic=/camera/color/image_raw ros_depth_topic=/camera/depth/image_raw \
    ros_camera_info_topic=/camera/color/camera_info ros_pose_topic=/camera/pose run_name=<test_name> \
    < /dev/null > /tmp/map.log 2>&1; pkill -f replica_publisher; pkill -f <test_name>'
```

- Use a unique `run_name` and kill by that name afterwards. Never kill by a broad pattern: the user may have their own mapper running (check `ps` first and leave it alone).
- `timeout` needs `-s INT`: SIGTERM is ignored by the mapper and it outlives the timeout.
- Do not publish test data or call `ros2 service` on domains 96/100.

## Housekeeping

- **Keep docs current in the same change.** When behaviour, config keys, models, topics or data flow change, update the matching doc before finishing: `docs/dataflow.md` (models, flowchart, config table), `docs/ros-ingest.md` (topics/QoS), `README.md` (usage). A change that adds a Hydra key is not done until the key is documented.
- Config lives in `online_scene_graph/hydra_configs/`. `detector.yaml` is the single source of truth for the detector, masker, crop and DINO settings. `online_mapping.yaml` ends with `_self_`, so a key set there overrides `detector.yaml`; don't duplicate detector keys there.
- Test outputs go to `outputs/<run_name>/`; delete your own test outputs when done (via `docker exec`, they are root-owned).
- Don't leave scratch scripts in the repo. Put them in the scratchpad, or delete them when the user is finished.
- Don't delete untracked files you didn't create (e.g. `tests/bag_run.sh`, `ros/.../Snapshot.srv`, `core.*` dumps) without asking.
- Commit only when asked.

## Key notes

- Class labels only feed the background/large-box filters, the majority-vote `class_name`, and `find()` output. Association and merging use geometry plus CLIP/DINO features.
- `merge_obj2_into_obj1` raises on any unknown key, so a new per-object field must be added to its handled lists, to `to_serializable`/`load_serializable`, and to the keep-lists in `prepare_objects_save_vis`.
- CLIP image encoding is the speed bottleneck (about 20 ms per crop on an RTX 3090); more detections per frame means a slower loop.
