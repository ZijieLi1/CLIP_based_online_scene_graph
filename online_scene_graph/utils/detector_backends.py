"""Hydra-selectable open-vocab / closed-vocab detection backends."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ultralytics import SAM, YOLO, YOLOE, FastSAM


DETECTOR_CHOICES = ("yolo11_sam", "yolo26_seg", "yoloe", "yoloe_pf")

# Optional box-prompted mask refiner that replaces the detector's own masks (cfg.masker).
#   null / none : use the detector's masks (yolo11_sam has none, so it falls back to mobile_sam)
#   mobile_sam  : MobileSAM, one box prompt -> one mask
#   fastsam     : FastSAM, picks the mask that matches each box (fast; may skip a box)
MASKER_CHOICES = (None, "none", "mobile_sam", "fastsam")


def ultralytics_class_names(model) -> list[str]:
    names = model.names
    if isinstance(names, dict):
        return [names[i] for i in range(len(names))]
    return list(names)


def masks_to_bool_numpy(masks_tensor: torch.Tensor, image_hw: tuple[int, int]) -> np.ndarray:
    image_h, image_w = image_hw
    masks_tensor = masks_tensor.float()
    if masks_tensor.shape[-2:] != (image_h, image_w):
        masks_tensor = torch.nn.functional.interpolate(
            masks_tensor.unsqueeze(1),
            size=(image_h, image_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    return (masks_tensor > 0.5).cpu().numpy()


def empty_detections(image_hw: tuple[int, int]):
    image_h, image_w = image_hw
    return (
        np.empty((0, 4), dtype=np.float32),
        np.array([], dtype=np.float32),
        np.array([], dtype=int),
        np.empty((0, image_h, image_w), dtype=bool),
    )


def init_detector(cfg, prompt_class_names: list[str]):
    backend = str(cfg.detector)
    if backend not in DETECTOR_CHOICES:
        raise ValueError(
            f"Unknown detector '{backend}'. Choose one of: {', '.join(DETECTOR_CHOICES)}"
        )

    masker = cfg.get("masker", None)
    masker = None if masker in (None, "none", "None", "") else str(masker)
    if masker not in MASKER_CHOICES:
        raise ValueError(f"Unknown masker '{masker}'. Choose one of: none, mobile_sam, fastsam")

    # yolo11 has boxes only, so it always needs a masker (default mobile_sam).
    if backend == "yolo11_sam" and masker is None:
        masker = "mobile_sam"
    sam_predictor = None
    if masker == "mobile_sam":
        sam_predictor = SAM(cfg.sam_weights)
    elif masker == "fastsam":
        sam_predictor = FastSAM(cfg.fastsam_weights)

    if backend == "yolo11_sam":
        detection_model = YOLO(cfg.yolo11_weights)
        class_names = ultralytics_class_names(detection_model)
    elif backend == "yolo26_seg":
        detection_model = YOLO(cfg.yolo26_seg_weights)
        class_names = ultralytics_class_names(detection_model)
    elif backend == "yoloe":
        detection_model = YOLOE(cfg.yoloe_weights)
        detection_model.eval()
        detection_model.set_classes(list(prompt_class_names))
        class_names = list(prompt_class_names)
    else:  # yoloe_pf
        detection_model = YOLOE(cfg.yoloe_pf_weights)
        detection_model.eval()
        class_names = ultralytics_class_names(detection_model)

    print(
        f"Using detector backend '{backend}' with {len(class_names)} classes; "
        f"masks from {masker if masker else 'the detector'}."
    )
    return backend, detection_model, sam_predictor, class_names


def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,4) x (M,4) xyxy -> (N,M) IoU."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).prod(-1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def masks_for_boxes(masker, image_bgr, xyxy_tensor, image_hw):
    """Box-prompted masks from `masker` (SAM or FastSAM).

    Returns (masks (N,H,W) bool, ok (N,) bool). ok[i] is False if the masker produced no mask for
    box i (FastSAM can skip boxes and does not say which), so the caller can fall back or drop it.
    """
    n = xyxy_tensor.shape[0]
    if isinstance(masker, FastSAM):
        out = masker.predict(
            image_bgr, bboxes=xyxy_tensor.cpu().tolist(), retina_masks=True,
            conf=0.25, iou=0.9, verbose=False,
        )[0]
        masks = np.zeros((n,) + tuple(image_hw), dtype=bool)
        ok = np.zeros(n, dtype=bool)
        if out.masks is None or out.boxes is None or out.boxes.xyxy.shape[0] == 0:
            return masks, ok
        got = masks_to_bool_numpy(out.masks.data, image_hw)
        iou = _box_iou(xyxy_tensor.cpu().numpy(), out.boxes.xyxy.cpu().numpy())
        used = set()
        for i in np.argsort(-iou.max(1)):  # most confident boxes pick first
            for j in np.argsort(-iou[i]):
                if iou[i, j] < 0.5:
                    break
                if int(j) not in used:
                    used.add(int(j)); masks[i] = got[j]; ok[i] = True
                    break
        return masks, ok
    torch.cuda.empty_cache()
    out = masker.predict(image_bgr, bboxes=xyxy_tensor, verbose=False)
    if out[0].masks is None:
        return np.zeros((n,) + tuple(image_hw), dtype=bool), np.zeros(n, dtype=bool)
    return masks_to_bool_numpy(out[0].masks.data, image_hw), np.ones(n, dtype=bool)


def run_detector(backend, detection_model, masker, image_bgr, image_hw, conf: float):
    """image_bgr: HxWx3 uint8 BGR array (OpenCV order, as ultralytics expects for numpy input).

    `masker` is the optional box-prompted mask model from init_detector (None = detector's masks).
    """
    results = detection_model.predict(image_bgr, conf=conf, verbose=False)
    det_result = results[0]
    if det_result.boxes is None or det_result.boxes.xyxy.numel() == 0:
        return empty_detections(image_hw)

    confidences = det_result.boxes.conf.cpu().numpy()
    class_ids = det_result.boxes.cls.cpu().numpy().astype(int)
    xyxy_tensor = det_result.boxes.xyxy
    xyxy_np = xyxy_tensor.cpu().numpy()

    own = None
    if det_result.masks is not None:
        own = masks_to_bool_numpy(det_result.masks.data, image_hw)

    if masker is None:
        if own is None:
            return empty_detections(image_hw)
        return xyxy_np, confidences, class_ids, own

    masks_np, ok = masks_for_boxes(masker, image_bgr, xyxy_tensor, image_hw)
    if own is not None:  # masker skipped a box: keep the detector's own mask for it
        masks_np[~ok] = own[~ok]
        ok = np.ones_like(ok)
    if not ok.all():  # box-only detector and no mask for some boxes: drop them
        xyxy_np, confidences, class_ids, masks_np = xyxy_np[ok], confidences[ok], class_ids[ok], masks_np[ok]
    if len(xyxy_np) == 0:
        return empty_detections(image_hw)
    return xyxy_np, confidences, class_ids, masks_np
