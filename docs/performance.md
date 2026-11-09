# Performance: frame rate and per-step cost

Measured on 2026-11-09 with the live HSR camera topics (ROS domain 96), `use_rerun=false`. The numbers describe the code at commit `6916723` and the config below; they will drift as the code changes, so re-measure before relying on them. The data flow itself is in [`dataflow.md`](dataflow.md).

## Setup

| | |
| --- | --- |
| GPU / torch | RTX 3090 (24 GB), torch 2.0.1 + CUDA 11.8, fp32 everywhere |
| Input | RGB (compressed) and depth (compressedDepth) at ~30 Hz, camera_info ~30 Hz, pose 20 Hz |
| Models | CLIP ViT-H-14 (laion2b) and DINOv2 ViT-B/14 (224 px) on padded bbox crops (`clip_mask_crop: false`, `use_dino: true`) |
| Other config | `detection_conf` 0.78, `mask_erode_px` 5, `dbscan_remove_noise` on with `dbscan_per_blob`, `denoise_interval` 15, class list `yoloe_classes.txt` (26 classes) for `yoloe` and `sam3` |
| Method | The real mapper (`online_mapping.py`) run unmodified, with a timer patched around each step at runtime and `torch.cuda.synchronize()` before and after every timed call so GPU time is counted in the step that launched it. 90 s per run after 5 warm-up frames. CPU, memory and GPU sampled at 1 Hz with `psutil` and `nvidia-smi`. |

The mapper was compute-bound in every run: it never waited for a frame (the camera is faster than the mapper), so the frame rate below is the processing rate. Frames the mapper cannot keep up with are dropped (`LatestFrameSlot`, drop-oldest).

## Results

| Config | Frames in 90 s | ms per frame | fps |
| --- | --- | --- | --- |
| `detector=yoloe masker=none` | 796 | 111 | **9.0** |
| `detector=yoloe masker=sam_hq masker_blobs=split_merge` with `sam_hq_model: vit_b` (previous default, fp32) | 373 | 237 | **4.2** |
| same with `sam_hq_model: vit_tiny` | 634 | 140 | **7.2** |
| `sam_hq_model: vit_b` with `sam_hq_half: true` (**current default**; torch 2.9.1, run after the image was rebuilt) | 446 | 198 | **5.0** |
| `detector=sam3 masker=none` | 76 | 792 | **1.26** |

The `vit_tiny` row (same setup, 634 frames, run later): **140 ms per frame, 7.2 fps**. The SAM-HQ masker drops from about 124 ms to about 34 ms, CLIP stays near 49 ms, mean GPU utilisation 62%, whole-GPU memory about 6.7 GB (8.9 GB with `vit_b`), process CPU 346% mean. On 16 Replica frames (45 objects) `vit_tiny` masks overlapped `vit_b` masks with a mean IoU of 0.957 (8 objects below 0.9); on the couch and table cases it was ~15 to 20% smaller and in one inspected case left out the couch's back cushion, while `vit_b` showed leakage speckles. No ground truth, so neither is shown to be better.

### Where the time goes (ms per frame)

| Step | YOLOE alone | YOLOE + SAM-HQ | SAM 3 alone |
| --- | --- | --- | --- |
| Detector + masker (`run_detector`) | 16 | 142 | 318 |
| &nbsp;&nbsp;of which SAM-HQ masker | – | 124 | – |
| CLIP image features (crops + ViT-H) | 54 | 55 | 307 |
| Back-project masks to point clouds | 16 | 17 | 78 |
| DINOv2 features | 12 | 12 | 52 |
| Voxel + DBSCAN per object | 5 | 3 | 17 |
| Spatial similarity (overlap) | 2 | 2 | 9 |
| `erode_masks` | 1 | 1 | 8 |
| `merge_obj_matches` (fuse clouds) | 1 | 1 | 3 |
| Periodic denoise / filter / merge | <1 | <1 | ~4 |
| `filter_gobs`, `mask_subtract_contained`, visual similarity, `make_detection_list`, `resize_gobs` | <1 | <1 | ~4 |
| Unwrapped glue (ROS to numpy, tracker, locks) | ~3 | ~3 | ~0 (rounding) |
| Objects per frame (DBSCAN calls) | ~3 | ~5 | ~19 |

Percentages for the `vit_b` setup: SAM-HQ 53%, CLIP 23%, back-projection 7%, DINOv2 5%, everything else about 12%.

### Resources (1 Hz samples)

| | YOLOE alone | YOLOE + SAM-HQ | SAM 3 alone |
| --- | --- | --- | --- |
| Process CPU (100 = one core), mean / max | 407 / 442 | 255 / 281 | 169 / 652 |
| Process RSS, mean / max | 2.5 / 2.5 GB | 2.6 / 2.6 GB | 4.5 / 7.5 GB |
| GPU utilisation, mean / max | 63% / 71% | 75% / 96% | 56% / 100% |
| GPU memory, whole GPU, mean / max | 6.3 / 6.4 GB | 8.9 / 9.0 GB | 9.7 / 11.3 GB |

## What the numbers say

- **SAM-HQ roughly doubles the frame time** (111 to 237 ms). Its cost (~124 ms for about 3 boxes) is dominated by the image encoder, not by the number of boxes.
- **CLIP is the largest cost without SAM-HQ** (48% of the YOLOE-only frame) and scales with the number of crops, about 16 to 20 ms each.
- **SAM 3 is slow for two reasons:** its own forward pass (~318 ms) and the number of objects it returns. At `detection_conf` 0.78 it gives about 19 objects per frame against about 3 for YOLOE, so every per-object step (CLIP, DINOv2, back-projection, DBSCAN) costs 4 to 6 times more. CLIP alone goes from 54 to 307 ms.
- **The mapping logic is cheap:** association, merging, filtering and the periodic cleanup together are about 8 ms per frame.
- **Frames processed:** about 1 in 7 incoming frames at the YOLOE-only rate, 1 in 14 with SAM-HQ, 1 in 24 with SAM 3.

## Caveats

- **GPU memory is for the whole GPU,** not per process (`nvidia-smi` per-process numbers were not available inside the container). It includes anything else using the card. Compare the differences between runs: SAM-HQ adds about 2.6 GB, SAM 3 about 3 to 5 GB.
- **CPU percentages are hard to read.** The fastest configuration used the most CPU. The timers add CUDA synchronisation, which makes the CPU spin while waiting for the GPU, and image/depth decoding also uses CPU. These two were not separated.
- **The SAM 3 run covers only 76 frames,** so its per-step numbers carry more noise than the others (the "unwrapped" row came out slightly negative for that reason, ~-10 ms, and is shown as about 0).
- **One scene, one 90 s window** per config. The object count per frame depends on what the robot was looking at.
- **Rerun was off.** Its logging cost is not included; see the throttling settings `rerun_image_interval` and `rerun_objects_interval` in `online_mapping.yaml`.
- The profiling wrapper itself is not in the repo.

## Ideas to speed it up (none tested)

- **SAM-HQ:** `vit_b` already runs in fp16 (the default). In an offline Replica check on 45 objects fp16 matched fp32 masks with a mean IoU of 0.996. Live, the SAM-HQ step dropped from about 124 ms (fp32) to about 73 ms (fp16), and the frame from 237 to 198 ms, but that run used the rebuilt image (torch 2.9.1, ultralytics 8.4.208) and different scene content, so it is not a controlled comparison. Other options: `vit_tiny`, or running the masker only for new or uncertain detections.
- **CLIP (the main lever for any setup):** run it in half precision (it is fp32 now), skip it for large background objects such as wall and floor, cap the number of crops per frame, or re-encode an object only every few frames.
- **SAM 3:** most of the extra cost is object count. Raise `detection_conf` or remove classes like wall and floor from `yoloe_classes.txt`.
