"""Hydra-selectable open-vocab / closed-vocab detection backends."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ultralytics import SAM, YOLO, YOLOE


DETECTOR_CHOICES = ("yolo11_sam", "yolo26_seg", "yoloe", "yoloe_pf")


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

    sam_predictor = None
    if backend == "yolo11_sam":
        detection_model = YOLO(cfg.yolo11_weights)
        sam_predictor = SAM(cfg.sam_weights)
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

    print(f"Using detector backend '{backend}' with {len(class_names)} classes.")
    return backend, detection_model, sam_predictor, class_names


def run_detector(backend, detection_model, sam_predictor, color_path: Path, image_hw, conf: float):
    results = detection_model.predict(str(color_path), conf=conf, verbose=False)
    det_result = results[0]
    if det_result.boxes is None or det_result.boxes.xyxy.numel() == 0:
        return empty_detections(image_hw)

    confidences = det_result.boxes.conf.cpu().numpy()
    class_ids = det_result.boxes.cls.cpu().numpy().astype(int)
    xyxy_tensor = det_result.boxes.xyxy
    xyxy_np = xyxy_tensor.cpu().numpy()

    if backend == "yolo11_sam":
        torch.cuda.empty_cache()
        sam_out = sam_predictor.predict(str(color_path), bboxes=xyxy_tensor, verbose=False)
        if sam_out[0].masks is None:
            return empty_detections(image_hw)
        masks_np = masks_to_bool_numpy(sam_out[0].masks.data, image_hw)
    else:
        if det_result.masks is None:
            return empty_detections(image_hw)
        masks_np = masks_to_bool_numpy(det_result.masks.data, image_hw)

    return xyxy_np, confidences, class_ids, masks_np
