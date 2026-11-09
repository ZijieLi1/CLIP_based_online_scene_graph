# Dataflow and model usage

End-to-end view of the online mapper (`online_scene_graph/slam/online_mapping.py`). The default hot path runs four neural models per frame: YOLOE (`detector: yoloe`, text prompts from `yoloe_classes.txt`) to find objects, SAM-HQ `vit_b` in fp16 (`masker: sam_hq`) to refine each object's mask from its box, OpenCLIP ViT-H-14 for the semantic feature and DINOv2 ViT-B/14 (`use_dino: true`) for the instance feature. Everything after that is geometry (mask erosion, depth back-projection, voxel downsample, DBSCAN, axis-aligned boxes, overlap-based association).

> Note: the detector and masker are independent choices. `detector:` picks who finds the objects (`yoloe` default, `yoloe_pf`, `yolo26_seg`, `yolo11_sam`, or the slow `sam3`), and `masker:` optionally replaces that detector's masks with box-prompted ones (`sam_hq` default, `mobile_sam`, `fastsam`, or `none` to keep the detector's own). Measured speed for the main combinations (about 9 fps for YOLOE alone, 7.2 fps with SAM-HQ `vit_tiny`, 4.2 fps with SAM-HQ `vit_b` in fp32, 1.3 fps for SAM 3 alone; the fp16 `vit_b` default has not been timed live yet) is in [`performance.md`](performance.md).

## 1. System dataflow

```mermaid
flowchart TD
    subgraph SRC["Camera publishers (sources/) — or the robot itself"]
        REP["replica_publisher.py<br/>(dataset replay, GT pose)"]
        R3D["record3d_publisher.py<br/>(phone, ARKit pose)"]
        HSR["HSR robot<br/>(camera + VSLAM)"]
    end

    REP & R3D & HSR -->|"4 ROS topics: rgb, depth,<br/>camera_info, pose (T_wc)"| ROSIN

    subgraph INGEST["Ingest — slam/ros_frame_source.py, frame_ipc.py"]
        ROSIN["message_filters sync<br/>(ApproxTime, slop 0.1s)"]
        SLOT["LatestFrameSlot<br/>(drop-oldest, keep-last 1)"]
        ROSIN --> SLOT
    end

    SLOT -->|"Frame: rgb, depth, K, pose"| DET

    subgraph PERFRAME["Per-frame loop (mapping thread) — online_mapping.py"]
        subgraph DETG["2D detection + instance masks — utils/detector_backends.py"]
            direction LR
            DET["<b>Detector</b> (detector:)<br/>MODEL: YOLOE text prompts (default) | YOLOE-PF | YOLO26-seg | YOLO11 + MobileSAM | SAM 3"] --> MASKER["<b>Masker</b> (masker:)<br/>MODEL: SAM-HQ ViT-B fp16 (default) | MobileSAM | FastSAM | none<br/>box prompts; masker_blobs: one prompt per mask blob"]
        end
        CLIPIMG["<b>Per-detection embeddings</b><br/>utils/model_utils.py<br/>CLIP: OpenCLIP ViT-H-14 (laion2b) image tower -> clip_ft<br/>DINO: DINOv2 ViT-B/14 CLS (use_dino) -> dino_ft<br/>same crop for both (padded bbox, or masked if clip_mask_crop)"]
        FILT["Filter detections<br/>(mask area, conf, bbox ratio, bg classes)<br/>mask_subtract_contained"]
        subgraph BP["Back-project masked depth to 3D — slam/utils.py"]
            direction LR
            ERODE["Erode mask<br/>mask_erode_px"] --> UNPROJ["Unproject masked depth<br/>(K + pose) -> one pcd per mask blob"]
            UNPROJ --> VOX["Voxel downsample 2.5 cm"]
            VOX --> DBS["DBSCAN: keep largest cluster<br/>per 2D mask blob (dbscan_per_blob)"]
            DBS --> BBOX["Axis-aligned bounding box<br/>(world axes; get_bounding_box)"]
        end
        DL["DetectionList<br/>(pcd, bbox, clip_ft, dino_ft, class, conf)"]
        MATCH["Data association — slam/mapping.py<br/>spatial sim (overlap/IoU) + visual sim<br/>(1-w)*cos(clip_ft) + w*cos(dino_ft) -> sim_sum > 1.2"]
        MERGE["Merge matched / append new<br/>merge_obj2_into_obj1:<br/>fuse pcd, running-mean clip_ft + dino_ft,<br/>majority-vote class_name"]
        POST["Periodic post-process<br/>denoise (DBSCAN) / filter / merge overlaps<br/>(uses spatial + clip_ft + text-sim thresholds)"]
        MASKER --> CLIPIMG --> FILT --> ERODE
        BBOX --> DL
        DL --> MATCH --> MERGE --> POST
    end

    MAP[("MapObjectList<br/>per object: uuid, class_name, clip_ft, dino_ft,<br/>pcd, bbox (axis-aligned), num_detections")]
    POST --> MAP
    MATCH <-.->|"reads clip_ft + pcd"| MAP

    subgraph SERVE["Query serving — slam/grounding.py, find_ros.py"]
        LOCK{{"map_lock / clip_lock<br/>generation counter"}}
        CLIPTXT["<b>Text query embedding</b><br/>MODEL: same OpenCLIP ViT-H-14<br/>text tower (shared weights with mapper)"]
        FIND["find(): copy clip_ft + metadata under map_lock,<br/>cosine(query, objects), min_sim / min_obs, top-k"]
        SNAP["snapshot(): points + colors per object<br/>(npz bytes, generation-gated)"]
        LOCK --- FIND
        LOCK --- SNAP
        CLIPTXT --> FIND
    end
    MAP -->|"set_objects() bumps generation"| LOCK

    CLI["scripts/find_query.py<br/>ros2 service call"] -->|"srv Find(text,k,min_sim,min_obs)"| CLIPTXT
    FIND -->|"json hits: id, class, sim, center, aabb"| CLI
    VIZ["scripts/viz_query.py<br/>(Open3D GUI)"] -->|"srv Snapshot"| SNAP
    VIZ -->|"srv Find"| CLIPTXT

    MAP --> OUT["On exit (Ctrl+C -> y)<br/>outputs/online/exps/online/<br/>pcd_*.pkl.gz + objects/edges json"]
    MAP -.-> RR["Rerun viewer<br/>images every rerun_image_interval frames;<br/>map (pcd x3 + boxes) as static data every rerun_objects_interval frames"]
    PLAN["PDDL planner bridge<br/>(not built yet — see online-graph-pddl-architecture.md)"] -.-> SNAP
```

## 2. Where each model runs

| Stage | Model / algorithm | Code | Input -> output | Config |
| --- | --- | --- | --- | --- |
| Detector (default) | **YOLOE-26l-seg**, open vocab, text prompts from `yoloe_classes.txt` (26 entries: the 8 grocery categories, furniture, floor, ...; set by `classes_file` in `classes.yaml`); also produces its own masks | `detector_backends.py` | BGR frame -> boxes, conf, class ids, masks | `detector: yoloe`, `yoloe_weights`, `detection_conf` |
| Detector (alt) | **YOLOE-26m-seg-pf**, prompt-free, built-in vocabulary of 4,585 tags (includes non-object tags such as "action film"); native masks. The bg-class label filter rarely matches these names | `detector_backends.py` | frame -> boxes, conf, class ids, masks | `detector: yoloe_pf`, `yoloe_pf_weights` |
| Detector (alt) | **SAM 3** (`SAM3SemanticPredictor`), text-prompted with `classes_file` (`yoloe_classes.txt`); boxes and masks together; slow (image backbone and text prompts re-run each frame) | `detector_backends.Sam3TextDetector` | frame -> boxes, conf, class ids, masks | `detector: sam3`, `sam3_weights` (gated HF `facebook/sam3`; not in the repo) |
| Detector (alt) | **YOLO26s-seg**, native instance seg, closed vocab (COCO) | `detector_backends.py` | frame -> boxes, conf, class ids, masks | `detector: yolo26_seg` |
| Detector (alt) | **YOLO11l** boxes + **MobileSAM** masks (box-prompted, two-stage) | `detector_backends.py` | frame -> boxes -> SAM masks | `detector: yolo11_sam` |
| Mask refinement (default on) | **SAM-HQ** (`segment-anything-hq`, default `vit_b` run in fp16 via autocast, weights from Hugging Face `lkeab/hq-sam`): one box prompt -> one mask with the HQ output token, finer edges than the detector's own masks. Alternatives: **MobileSAM** (box prompt), **FastSAM** (box-matched mask; a box it cannot match keeps the detector's mask). Replaces the detector's masks | `detector_backends.HQSamMasker`, `masks_for_boxes` | boxes -> masks | `masker: none \| mobile_sam \| fastsam \| sam_hq`, `sam_hq_model: vit_tiny \| vit_b \| vit_l \| vit_h`, `sam_hq_half`, `sam_hq_weights` |
| Mask blob handling | A detection whose own mask is in several connected blobs (an occluder cuts it, so its box also covers the occluder) is prompted per blob and the masks unioned (`split_merge`), or prompted only with its biggest blob (`largest`) | `detector_backends.split_blobs` | detector masks -> one prompt per blob | `masker_blobs: none \| split_merge \| largest` (needs a masker) |
| Per-object semantic feature | **OpenCLIP ViT-H-14** (`laion2b_s32b_b79k`), image encoder, L2-normalized | `model_utils.compute_clip_features_batched` | crops (batched) -> `clip_ft` (1024-d) | `clip_mask_crop`, `clip_crop_padding` |
| Per-object instance feature | **DINOv2 ViT-B/14** (`vit_base_patch14_dinov2.lvd142m`, timm), CLS token, L2-normalized | `model_utils.compute_dino_features` | the same crops, squashed to 224x224 -> `dino_ft` (768-d) | `use_dino`, `dino_model`, `dino_img_size`, `dino_weight` |
| Query embedding | Same **ViT-H-14** text encoder (CLIP only; DINO has no text tower) | `grounding.GroundingService.encode_text` | text -> unit vector | shares `clip_model` and `clip_lock` with the mapper |
| Back-projection and cleanup | Erode each mask by `mask_erode_px` (5 px), unproject masked depth with K + pose (one cloud per mask blob), voxel-downsample 2.5 cm, DBSCAN keeping the largest cluster of each blob (eps 0.1 m, min 10 points), axis-aligned bounding box in world axes | `slam/utils.py` (`erode_masks`, `detections_to_obj_pcd_and_bbox`, `init_process_pcd_blobs`, `get_bounding_box`) | masks + depth + pose -> pcd + box | `mask_erode_px`, `dbscan_per_blob`, `dbscan_remove_noise`, `dbscan_eps`, `dbscan_min_points`, `downsample_voxel_size` |
| Data association | `(1-w)*cos(clip_ft) + w*cos(dino_ft)` + 3D overlap/IoU | `slam/mapping.py` | M detections x N objects -> match indices | `sim_threshold`, `dino_weight`, `spatial_sim_type`, `match_method` |
| Object fusion | Running-mean `clip_ft` and `dino_ft` (weights = `num_detections`, renormalised), point-cloud union. The merged object keeps the `id` (and `inst_color`) of the side created first (smaller `curr_obj_num`), whichever side is the target, so an object keeps one id for the run | `slam/utils.merge_obj2_into_obj1` | matched pair -> one object | `downsample_voxel_size`, `dbscan_*` |

### Why both CLIP and DINO

CLIP gives the semantic feature used for text queries and DINO gives an instance-level appearance feature used for matching. [Liu et al., *Data or Language Supervision: What Makes CLIP Better than DINO?* (arXiv 2510.11835)](https://arxiv.org/abs/2510.11835), a VLM-encoder study, trains CLIP and DINO under matched conditions and finds that CLIP "captures high-level semantics (e.g., object categories, text)" while DINO is "more sensitive to low-level cues like color schemes, and is more invariant to orientation change" (its summary: CLIP is more semantically meaningful, DINO emphasises visual similarity). The two also disagree on many image pairs, so each carries information the other lacks. That motivates CLIP for `find()` and a DINO term for telling visually distinct instances of one category apart. The paper tests neither association nor mapping and reports no hybrid experiments, so the 0.5 `dino_weight` and the gain on this system are untested; compare object count against ground-truth instances with `use_dino` on and off.

With `clip_mask_crop: false` (the current default) neither model sees a mask: both get the padded bbox crop. Set it to `true` for a masked, square-padded crop shared by both.

The detector's class label never affects association or merging; it is used by the background / large-box filters, the majority-vote `class_name`, and what `find()` returns. The per-detection `text_feats` path is commented out in `compute_clip_features_batched`, so `merge_text_sim_thresh` has no text feature to compare against.

### Not on the hot path

- **GPT-4V / LLaVA** (`utils/vlm.py`, `make_edges`) is off by default (`make_edges: false`). It was for spatial relations and is legacy from ConceptGraphs. The code is still imported by `general_utils.py`, so `openai` must be installed.
- **VSLAM** is not in this repo. Poses come in on the `pose` topic.

## 3. Threads and locks

```mermaid
flowchart LR
    subgraph ROSEXEC["rclpy executor thread"]
        SUBS["Subscribers -> LatestFrameSlot.put"]
        SRV["Find / Snapshot service callbacks"]
    end
    subgraph MAPT["Mapping thread (main)"]
        LOOP["next() -> detector + masker -> CLIP + DINO -> 3D -> match/merge"]
    end
    SUBS -->|"overwrite latest"| SLOT[(LatestFrameSlot)]
    SLOT -->|"pull via next()"| LOOP
    LOOP -->|"holds map_lock while mutating MapObjectList"| MAP[(MapObjectList)]
    LOOP -->|"holds clip_lock for CLIP/DINO image encode"| GPU["GPU: CLIP ViT-H-14<br/>(also DINOv2, SAM-HQ, detector in the mapping thread)"]
    SRV -->|"clip_lock for text encode"| GPU
    SRV -->|"map_lock only to copy clip_ft + metadata"| MAP
```

Query threads copy features out under `map_lock` and score the copy, so a concurrent merge cannot leak deleted ids. The `generation` counter lets clients detect a stale result.

## 4. Typical cost split

Measured on the live robot topics (RTX 3090, fp32), see [`performance.md`](performance.md) for the full per-step tables and caveats:

| Config | ms per frame | fps | Largest steps |
| --- | --- | --- | --- |
| YOLOE alone (`masker: none`) | 111 | 9.0 | CLIP 54 ms, back-projection 16, detector 16, DINOv2 12 |
| YOLOE + SAM-HQ `vit_tiny` | 140 | 7.2 | detector + masker 51 (SAM-HQ ~34), CLIP 49, back-projection 16, DINOv2 11 |
| YOLOE + SAM-HQ `vit_b` fp32 | 237 | 4.2 | SAM-HQ 124, CLIP 55, back-projection 17, DINOv2 12 |
| SAM 3 alone | 792 | 1.26 | detector 318, CLIP 307 (about 19 objects per frame), back-projection 78, DINOv2 52 |

CLIP image encoding scales with the number of detections per frame (one ViT-H crop each, about 16 to 20 ms in fp32), and so do DINOv2, back-projection and DBSCAN, so a detector that returns more objects makes the loop slower. The mapping logic itself (association, merging, filtering, periodic cleanup) is about 8 ms per frame. See [`online-graph-pddl-architecture.md`](online-graph-pddl-architecture.md) for the bottleneck analysis and [`ros-ingest.md`](ros-ingest.md) for the topic and QoS contract.
