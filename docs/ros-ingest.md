# Live ingest: ROS topics only

The mapper has one input: four ROS topics. It knows nothing about phones, robots or datasets.
Every camera is a **publisher** under `online_scene_graph/sources/` that republishes onto those topics.

| Source | Publisher | Notes |
| --- | --- | --- |
| Replica replay | `sources/replica_publisher.py` | ground-truth poses, `--rate` keeps it reproducible |
| Phone (Record3D USB) | `sources/record3d_publisher.py` | needs host `usbmuxd`; applies the ARKit Y/Z flip before publishing |
| Robot | none (robot camera + VSLAM already publish) | remap the four `ros_*_topic` keys |

Default topics (`hydra_configs/online_mapping.yaml`):

```yaml
ros_rgb_topic: /camera/color/image_raw
ros_depth_topic: /camera/depth/image_raw
ros_camera_info_topic: /camera/color/camera_info
ros_pose_topic: /camera/pose
```

The mapper, `slam/ros_frame_source.py` (`RosRgbDPoseSource` -> `Frame`) and `slam/frame_ipc.py` (`LatestFrameSlot`) are identical for every source. YOLO, CLIP and fuse never see ROS.

```bash
# shell 1: any one publisher
python3 online_scene_graph/sources/replica_publisher.py --scene-dir /data/Replica/office0 --rate 2 --stride 5
python3 online_scene_graph/sources/record3d_publisher.py        # phone
# shell 2
python3 online_scene_graph/slam/online_mapping.py
```

Robot: point the four keys at the robot topics. A topic ending in `/compressed` or `/compressedDepth` is `sensor_msgs/CompressedImage`; anything else is raw `Image`. HSR:

```yaml
ros_rgb_topic: /head_rgbd_sensor/rgb/image_rect_color/compressed
ros_depth_topic: /head_rgbd_sensor/depth_registered/image_rect_raw/compressedDepth
ros_camera_info_topic: /head_rgbd_sensor/rgb/camera_info
ros_pose_topic: /hsr/cam_pose
```

Do not use `/compressed` for depth (8-bit JPEG).

## Frame / ROS contract

Every publisher must provide:

| Field | Meaning |
| --- | --- |
| color | rectified BGR8 (or RGB8), or JPEG `CompressedImage` on `…/compressed` |
| depth | registered to color, meters (`32FC1`) or millimetres (`16UC1`), or PNG `compressedDepth` |
| `CameraInfo.k` | pinhole of that color image (distortion ignored — send undistorted/rectified images) |
| pose | `T_wc`: OpenCV optical axes (x right, y down, z forward), world ← camera. Not `camera_link` (x forward, z up). Phone node applies the ARKit Y/Z flip **before** publish. |
| QoS | BEST_EFFORT, keep-last 1 |

Phone stamps all four messages with the same `now()` so ExactTime would work. A robot will not: RGB/depth come from the camera clock, pose from TF. `ros_sync_slop > 0` (default 0.1) uses ApproximateTime. Set `ros_sync_slop: 0` for ExactTime.

## Drop-oldest

If the mapper is slower than the camera, drop stale frames — never block the producer.

- ROS: BEST_EFFORT keep-last 1.
- `LatestFrameSlot` overwrite; mapper `next()` is a pull (`get`), not a callback.

## Realtime `find()` (service, not action)

CLIP lookup is a short request/response (encode text + cosine vs N objects). Use a **ROS service**, not an action.

`GroundingService.find()` lives in the mapper. It copies `clip_ft` + metadata under `map_lock`, then scores the copy so a merge cannot leak deleted ids. The mapping thread holds that lock only while it mutates the map. The service callback runs on the shared `rclpy` executor.

```text
ros2 service call /online_scene_graph/find
        FindRosService                       in-process rclpy
GroundingService.find()                      mapping process
```

Interfaces are built into the Docker image (`/opt/osg_ws`). Rebuild on the bind mount if you change `.srv`:

```bash
cd /ws/ros
rm -rf build install log
source /opt/ros/humble/setup.bash
colcon build --packages-select online_scene_graph_interfaces
source /ws/ros/install/setup.bash
```

While the mapper is running (`find_enabled: true`):

```bash
python3 online_scene_graph/scripts/find_query.py mug

ros2 service call /online_scene_graph/find online_scene_graph_interfaces/srv/Find \
  "{text: 'mug', k: 5, min_sim: 0.25, min_obs: 3}"
```

`k`, `min_sim`, or `min_obs` ≤ 0 means “use the mapper default”. Snapshot / PDDL is a later API.
