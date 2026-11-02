# Online concept graph as PDDL upstream

ConceptGraphs already fuses RGB-D into an incremental object map. The new work is serving CLIP queries and planner snapshots while that map is still growing — not rewriting the mapper, and not keeping RGB frames in RAM.

What landed, by date: [`progress.md`](progress.md).

| Status | What |
| --- | --- |
| Exists | Incremental fuse / match / merge |
| Missing | Query while mapping |
| None | PDDL / planner bridge |
| Real bottleneck | CLIP × detection count (YOLO26-seg + high conf is 15+ fps in a house) |

## Verdict

Incremental mapping from RGB-D with external VSLAM poses is the right idea — that loop already lives in `rerun_realtime_mapping.py`. The bad parts are (1) treating CLIP search as the PDDL feed, (2) keeping images in memory, and (3) letting an LM read a racing live map. Split grounding queries from a frozen planner snapshot.

### Keep

Per-frame detect → CLIP → project to 3D → spatial+visual match → merge into `MapObjectList`. Poses from another node are fine; this repo never ran VSLAM internally. CLIP cosine search via `MapObjectList.compute_similarities()` is the right primitive for “where is the mug”. Compact object state (uuid, class, clip_ft, downsampled pcd, bbox) should stay in process memory.

### Change

| Idea as stated | Problem | Do this instead |
| --- | --- | --- |
| Keep RGB / PCL on disk off, hold them in RAM | A session of frames in RAM is worse than disk. After CLIP encode + 3D project, the image is dead weight. Per-detection masks/crops already bloat each node. | Drop RGB/depth after the frame is fused. Keep only voxel-downsampled object pcds + bbox + clip_ft. Never store full RGB history. |
| Query CLIP while the map is incrementing, then feed that to an LM for PDDL | CLIP search is grounding, not a planning world. Merges delete nodes, rewrite clip_ft, and remap indices mid-query. PDDL from a racing graph is inconsistent. | Two APIs: live CLIP `find()` for skills, and `snapshot()` that freezes filtered objects + geometric predicates for the LM. |
| Save the whole map only at the end is the main gap | The mapper is already incremental. The gap is serving the in-memory map, not whether a pickle is written. | Don’t checkpoint by default. Expose a locked copy / generation counter. Optional JSON export of the snapshot, never full pcds, to the LM. |
| VLM `make_edges` for spatial relations | GPT-4V per frame is the slowest path and is already off (`make_edges: false`). PDDL needs cheap, deterministic predicates. | Compute on / in / adjacent from 3D AABBs. Keep VLM captions as an optional offline enricher, not on the hot path. |
| Open-vocab CLIP classes straight into PDDL types | “coffee mug” vs “cup” vs “mug” breaks a domain. CLIP scores also fire on incomplete 1-observation blobs. | Map CLIP/YOLO labels through a closed symbol table. Only emit objects with min observations + a resolved type. |

### What already exists (so we don’t rebuild it)

`online_scene_graph/slam/mapping.py` already associates detections to objects every frame. `merge_obj2_into_obj1` weighted-averages `clip_ft`. `gui_realtime_mapping.py` already runs mapping on a background thread. Scene-graph GPT / LLaVA lives in `build_scenegraph_cfslam.py` and is offline only. Live USB can go through ROS2 (`record3d_ros_publisher` + `RosRgbDPoseSource`). Live CLIP `find()` is a ROS service (`/online_scene_graph/find`); there is still no planner `snapshot()` / PDDL.

## What to turn off

Disk I/O is worth killing, but it will not make the loop realtime by itself. Detection (YOLOE/SAM) and per-mask CLIP dominate. Turn I/O off first so we can measure the real compute budget.

### Safe to disable now (Hydra)

| Knob | Current / default | For online mapping |
| --- | --- | --- |
| `save_pcd` | true (base); writes `pcd_*.pkl.gz` | false — in-memory only |
| `save_json` | true — obj/edge JSON | false until a planner snapshot is requested |
| `save_detections` | false in your rerun yaml | keep false; never write per-frame pkl + vis JPGs |
| `save_objects_all_frames` | false | keep false — this is a full map dump every frame |
| `periodically_save_pcd` | false | keep false |
| `save_video` / `vis_render` / `debug_render` | false in rerun yaml | keep false |
| `use_wandb` | false | keep false |
| `use_rerun` / `save_rerun` | true in your rerun yaml | false for robot; optional debug viewer only |
| `make_edges` | false | keep false — GPT-4V per frame |
| `force_detection` | true | true only while iterating; cache detections for replay benches |

### Also drop from the in-memory node (code change)

| Field | Why it exists | Online policy |
| --- | --- | --- |
| `mask`, `xyxy`, `color_path` lists | Replay / vis / LLaVA crops | Drop after CLIP; keep at most last crop if LM needs a picture |
| per-detection captions | VLM path | Skip unless captions are on |
| full RGB frame buffer | Dataset / stream writers | Discard at end of frame |
| obj pcd > ~2–5k points | Fusion quality | Keep voxel downsample (0.02–0.025 m); lower `obj_pcd_max_points` if RAM climbs |
| DBSCAN every N frames | Noise cleanup | `denoise_interval: -1` during run; once on snapshot |
| global merge every 5 frames | Dedup | Keep, but `snapshot()` must wait for merge to finish |

### Do not turn off (needed for queries / PDDL)

CLIP image encode per detection, spatial+visual matching, object uuids, voxel-downsampled pcds (for merge, not for the LM), and a min-observation filter. The rerun config has `obj_min_detections: 1` — too low for planning; use 3+ before a node is queryable or exported.

### Real cost order (after I/O is gone)

| Stage | Where | Notes |
| --- | --- | --- |
| CLIP ViT-H-14 per crop | `compute_clip_features_batched` | Scales with box count; this was the live bottleneck, not fuse |
| Detector | `detector_backends.run_detector` | `yolo26_seg` is the fast house path (~15 fps with high conf). YOLOE/`yoloe_pf` is slower because it proposes more boxes, not because the YOLO fwd is huge. `yolo11_sam` still pays SAM. |
| FAISS overlap matrix | `compute_spatial_similarities` | Grows with object count |
| DBSCAN denoise | `denoise_objects` | Heavy; not every frame |
| Rerun log of pcds | `orr_log_*` | Surprising CPU/GPU cost if left on |
| gzip pickle of map | `save_pointcloud` | Kill; was never the query path |

## Data flow

One writer owns the live map. Queries and the LM never read it directly — they read a generation-stamped copy. RGB-D is dropped after fuse. VSLAM is out of process.

```text
VSLAM pose → RGB-D frame → detect (YOLO26-seg / YOLOE) → CLIP feats → 3D pcd/bbox → Match/merge → Live map
                                                                              ├→ find(text)     [grounding]
                                                                              └→ snapshot() → LM → PDDL
```

### Per-frame (writer thread)

Ingest `(rgb, depth, K, T_wc, t)` → detect → CLIP crops → project valid masks to 3D → match against live objects → merge or spawn → bump `generation` → drop rgb/depth/masks. No pickle, no JPEG, no Rerun unless debug.

### Concurrent readers

`find(text)` encodes the query with the CLIP text tower (cached tokenizer), cosine vs stacked `clip_ft` of objects that meet `min_obs`. `snapshot()` copies uuid, resolved type, confidence, AABB/center, and geometric predicates. Point clouds and CLIP vectors do not leave the process toward the LM.

> Stream path `online_mapping.py` currently writes an RGB JPEG every frame even when `save_detections` is false. That writer has to die before this is actually in-memory.

## APIs

Two consumers, two APIs. In-process first (Python). ROS2 / ZMQ later as a thin wrap — do not design the mapper around middleware.

### GroundingService (live, best-effort)

Skills / pick-and-place “find X”. May return provisional objects. IDs can disappear on the next merge.

```python
find(text, k=5, min_sim=0.25, min_obs=3) -> list[Hit]
get(uuid) -> ObjectView | None
```

`Hit`: uuid, class, sim, center, aabb, n_obs, generation

### PlannerSnapshot (frozen)

LM input. Built only on request (or every N seconds). Waits until the current frame+merge finishes, then copies.

```python
snapshot(min_obs=3) -> SceneGraph
```

`SceneGraph`: generation, t, objects[], predicates[]

`Object`: id, type, aliases, conf, aabb, center — no pcd, no clip_ft, no image.

### Frame ingest

Replace dataset-index coupling with `FrameSource.next()`, which returns a `Frame` or `None`. Mapping loop stays the same. If the mapper is slower than the camera, drop oldest frames — never block VSLAM.

Live USB/ROS wiring and what to swap on a robot: [`ros-ingest.md`](ros-ingest.md). `rclpy` runs in the mapper process (Humble Docker). The robot bus is ROS topics; YOLO/CLIP/fuse only see `Frame`.

### Snapshot JSON the LM actually sees

Keep it small enough to paste into a prompt:

```json
{
  "generation": 42,
  "timestamp": 0.0,
  "objects": [
    {
      "id": "uuid",
      "type": "cup",
      "conf": 0.9,
      "aabb": [0.0, 0.0, 0.0, 0.1, 0.1, 0.1]
    }
  ],
  "predicates": [
    { "rel": "on", "a": "uuid-cup", "b": "uuid-table", "conf": 0.8 }
  ]
}
```

`rel` is one of `"on"` | `"in"` | `"adjacent"`.

The LM (outside this repo) compiles that into a PDDL problem against a fixed domain. ConceptGraphs should not emit PDDL strings until the snapshot schema is stable — otherwise you debug planner syntax and mapping bugs at the same time.

### Concurrency

One `threading.Lock` around mutate (match/merge/filter) and snapshot copy. CLIP `find` can run on a numpy copy of `clip_ft` + uuid list taken under the same lock (microseconds). Do not share Open3D geometries across threads without that copy. Generation is a monotonic int; clients discard hits whose generation is older than they asked for.

### What not to build yet

- FAISS on CLIP (N objects is tens–hundreds; linear cosine is fine)
- Per-frame GPT edges
- Saving `latest_pcd_save` as the “API”
- Sending crops of every object to the LM
- A second process that pickle-reloads the map every query

## Phases

Each phase should be shippable on Replica/Record3D without the robot. Later phases wrap the same in-process objects.

- [ ] **Phase 0 — Headless in-memory mapper.** Force `save_pcd` / json / detections / rerun / wandb off. Stop stream JPEG writes. Strip `mask` / `xyxy` / `color_path` from nodes after CLIP. Confirm RAM stays flat over a long Replica run.
- [x] **Phase 1 — FrameSource interface.** `Frame` + depth-1 `LatestFrameSlot`. USB (`UsbRecord3DFrameSource`) and ROS (`RosRgbDPoseSource` on color/depth/K/pose). Mapping loop calls `next()`; producer never blocks. Offline dataset FrameSource is still the old indexed loader, not this interface.
- [x] **Phase 2 — `GroundingService.find()`.** Lock, copy `clip_ft` + metadata, cosine search, filter `min_obs`. In-process ROS **service** `/online_scene_graph/find` (not an action). `snapshot()` / PDDL still later. See [`ros-ingest.md`](ros-ingest.md).
- [ ] **Phase 3 — PlannerSnapshot.** After merge: filter `min_obs`, resolve type via a closed label map, compute geometric on/in/adjacent from AABBs, emit JSON. No CLIP vectors, no pcds. Golden tests on Replica room2 vs hand-labeled predicates.
- [ ] **Phase 4 — LM adapter (separate module).** Prompt: domain predicates + snapshot JSON → PDDL problem. Do not bake planner-specific syntax into `slam/`. Version the snapshot schema. Evaluate with a frozen map first, then live.
- [ ] **Phase 5 — Optional.** Move detect+CLIP to a worker process if frame time is still over budget. Only after Phases 0–2 are measured. ROS2/ZMQ wrap of `find` / `snapshot` if other nodes need it.

### Suggested first PR

A `headless_online_mapping` Hydra config + `QueryableMap` wrapper around `MapObjectList` with `find` and `snapshot`, driven by the existing rerun mapping loop with viz/io flags false. No new detector. No PDDL yet.

## Risks

| Risk | Why | Mitigation |
| --- | --- | --- |
| Duplicate objects from pose lag / jumps | A delayed `T_wc` projects the same instance into a new cluster. VSLAM relocalization makes this worse. | Timestamp-align RGB-D to pose; drop frames with pose age above threshold; raise spatial sim weight after a jump. |
| ID churn on merge | `merge_overlap_objects` removes the loser; consumers holding an index/uuid get a ghost. | Export UUID of the survivor only. Snapshot lists `merged_into`. `find()` never returns list indices. |
| CLIP on immature nodes | 1-view crops are viewpoint-specific; “chair” matches a table corner. | `min_obs` gate; optionally require class vote agreement; don’t snapshot those nodes. |
| Mapper slower than camera | Open-vocab + CLIP on many crops will not do 30 Hz. YOLO26-seg + high conf already does 15+ fps on USB 960×720 in a house. Queue still needed if detect goes back to YOLOE. | Cap dets before CLIP; stride / drop-oldest; keep mapping pose-synced rather than every camera frame. |
| Geometric predicates wrong for PDDL | AABB on/in is crude (a mug “in” a table, a wall “adjacent” to everything). | Start with on-support (z-overlap + xy containment) and leave “in” for containers only. Golden-set the predicates. |
| Open-vocab vs domain types | Planner has `cup`; detector says `coffee_mug`. | Explicit synonym table. Unmapped labels stay in the snapshot as aliases but are not PDDL objects. |
| Lock stalls mapping | Deep-copying Open3D pcds under the lock for every query. | `find()` copies only `clip_ft` + metadata. `snapshot()` copies AABBs not pcds. Hold lock for copy, not for CLIP text encode. |
| Feeding the LM too often | Every frame a new problem file; planner never finishes. | Snapshot on demand or 1–2 Hz. LM/planner owns staleness (`generation`). |

## What to test

| Test | Pass condition |
| --- | --- |
| I/O off | Headless Replica run writes no pkl/jpg/rrd under the experiment dir (hydra yaml only). |
| Memory | RSS plateaus after warmup; does not track frame count. Heap dump shows no rgb history. |
| `find` during merge | Background thread calls `find('chair')` every frame while `merge_interval=1`. Zero crashes, zero hits with unknown uuid. |
| ID stability | Object uuid of a clearly tracked instance is constant across N frames; merge loser uuid is absent from later snapshots. |
| `min_obs` filter | Snapshot with `min_obs=3` contains no 1-detection blobs that CLIP would still match. |
| CLIP regression | On a frozen Replica map, text queries match the same top-1 as `visualize_cfslam_results.py`. |
| Pose delay injection | Shift `T_wc` by 200 ms in replay; measure extra object count. Bound the increase. |
| Frame drop | Push 30 Hz into a slow mapper; queue depth bounded; VSLAM callback never blocks. |
| Predicate golden | Hand-check on/in/adjacent on room2 vs snapshot JSON. Fail on false `in` for non-containers. |
| LM frozen-map | Snapshot from a finished map → LM → PDDL parses and has typed objects for the domain. Only then try live snapshots. |

## Bottom line

Keep the incremental mapper, kill I/O and RGB retention, add a locked CLIP `find()` plus a compact `snapshot()` for the LM. Do not stream live CLIP hits into PDDL, and do not generate PDDL inside the SLAM loop until the snapshot schema is tested on a frozen map.
