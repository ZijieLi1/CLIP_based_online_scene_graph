"""Hydra-selectable open-vocab / closed-vocab detection backends."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import SAM, YOLO, YOLOE, FastSAM


DETECTOR_CHOICES = ("yolo11_sam", "yolo26_seg", "yoloe", "yoloe_pf", "sam3")

# Optional box-prompted mask refiner that replaces the detector's own masks (cfg.masker).
#   null / none : use the detector's masks (yolo11_sam has none, so it falls back to mobile_sam)
#   mobile_sam  : MobileSAM, one box prompt -> one mask
#   fastsam     : FastSAM, picks the mask that matches each box (fast; may skip a box)
#   sam_hq      : SAM-HQ (segment-anything-hq), one box prompt -> one mask, sharper on thin structure
MASKER_CHOICES = (None, "none", "mobile_sam", "fastsam", "sam_hq")


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


def _torch_compat_for_sam3():
    """Ultralytics' SAM 3 imports torch.nn.attention / torch.compiler (torch >= 2.5). On older torch
    (the container has 2.0.1) stub the two names it uses: a no-op sdpa_kernel (torch then picks the
    attention backend itself) and is_dynamo_compiling() -> False. Upgrading torch makes this a no-op."""
    import contextlib
    import enum
    import sys
    import types

    if not hasattr(torch.nn, "attention"):
        mod = types.ModuleType("torch.nn.attention")

        class SDPBackend(enum.Enum):
            ERROR = -1
            MATH = 0
            FLASH_ATTENTION = 1
            EFFICIENT_ATTENTION = 2
            CUDNN_ATTENTION = 3

        @contextlib.contextmanager
        def sdpa_kernel(backends, set_priority=False):
            yield

        mod.SDPBackend, mod.sdpa_kernel = SDPBackend, sdpa_kernel
        sys.modules["torch.nn.attention"] = mod
        torch.nn.attention = mod
    if not hasattr(torch, "compiler"):
        comp = types.ModuleType("torch.compiler")
        comp.is_dynamo_compiling = lambda: False
        sys.modules["torch.compiler"] = comp
        torch.compiler = comp


class HQSamMasker:
    """Box-prompted SAM-HQ (PyPI `segment-anything-hq`; Ultralytics has no SAM-HQ).

    Same job as MobileSAM as a masker: one box -> one mask, but with the HQ output token, which
    gives finer boundaries on thin or open structures (chair backs, handles). Weights come from
    the Hugging Face repo lkeab/hq-sam (sam_hq_<model_type>.pth) unless `weights` is a file path.
    """

    HF_REPO = "lkeab/hq-sam"

    def __init__(self, model_type: str = "vit_b", weights: str | None = None, device: str = "cuda", half: bool = True):
        try:
            from segment_anything_hq import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "masker=sam_hq needs the `segment-anything-hq` package "
                "(pip install --no-deps segment-anything-hq)."
            ) from exc
        if model_type not in sam_model_registry:
            raise ValueError(f"Unknown sam_hq_model '{model_type}'. Choose one of: vit_b, vit_l, vit_h, vit_tiny")
        if weights and Path(weights).exists():
            checkpoint = str(weights)
        else:
            from huggingface_hub import hf_hub_download

            checkpoint = hf_hub_download(self.HF_REPO, f"sam_hq_{model_type}.pth")
        self.device = torch.device(device)
        # fp16 via autocast (weights stay fp32): ~1.5x faster and ~40% less memory on vit_b, masks
        # within IoU ~0.996 of fp32 in a Replica check.
        self.half = bool(half) and self.device.type == "cuda"
        sam = sam_model_registry[model_type](checkpoint=checkpoint).to(self.device).eval()
        self.predictor = SamPredictor(sam)

    def _boxes_masks(self, boxes_xyxy, image_hw):
        boxes = self.predictor.transform.apply_boxes_torch(boxes_xyxy.to(self.device).float(), tuple(image_hw))
        out, _, _ = self.predictor.predict_torch(
            None, None, boxes=boxes, multimask_output=False, hq_token_only=False
        )
        return out[:, 0].cpu().numpy().astype(bool)

    @torch.no_grad()
    def masks(self, image_bgr, xyxy_tensor, image_hw):
        """(N,H,W) bool masks for N xyxy boxes (pixels)."""
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.half):
            self.predictor.set_image(np.ascontiguousarray(image_bgr[..., ::-1]))  # expects RGB
            return self._boxes_masks(xyxy_tensor, image_hw)


class Sam3TextDetector:
    """SAM 3 text-prompted detector + segmenter behind the same `.predict()` call as the YOLO models.

    Every class name is a text prompt; Results.boxes.cls indexes into `names`. Slow: the image
    backbone and the text prompts are re-run on every frame.
    """

    def __init__(self, weights: str, names: list[str], half: bool = True):
        _torch_compat_for_sam3()
        from ultralytics.models.sam import SAM3SemanticPredictor

        if not Path(weights).exists():
            raise FileNotFoundError(
                f"SAM 3 weights not found at '{weights}'. They are gated (Hugging Face: facebook/sam3): "
                "request access, download sam3.pt and set `sam3_weights` in detector.yaml."
            )
        self.names = list(names)
        self.predictor = SAM3SemanticPredictor(
            overrides=dict(task="segment", mode="predict", model=str(weights), half=bool(half),
                           save=False, verbose=False, conf=0.25)
        )

    def predict(self, image_bgr, conf: float = 0.25, verbose: bool = False):
        self.predictor.args.conf = conf
        self.predictor.set_image(image_bgr)
        return self.predictor(text=self.names)


def init_detector(cfg, prompt_class_names: list[str]):
    backend = str(cfg.detector)
    if backend not in DETECTOR_CHOICES:
        raise ValueError(
            f"Unknown detector '{backend}'. Choose one of: {', '.join(DETECTOR_CHOICES)}"
        )

    masker = cfg.get("masker", None)
    masker = None if masker in (None, "none", "None", "") else str(masker)
    if masker not in MASKER_CHOICES:
        raise ValueError(f"Unknown masker '{masker}'. Choose one of: none, mobile_sam, fastsam, sam_hq")

    # yolo11 has boxes only, so it always needs a masker (default mobile_sam).
    if backend == "yolo11_sam" and masker is None:
        masker = "mobile_sam"
    sam_predictor = None
    if masker == "mobile_sam":
        sam_predictor = SAM(cfg.sam_weights)
    elif masker == "fastsam":
        sam_predictor = FastSAM(cfg.fastsam_weights)
    elif masker == "sam_hq":
        sam_predictor = HQSamMasker(
            str(cfg.get("sam_hq_model", "vit_b")), cfg.get("sam_hq_weights", None), str(cfg.get("device", "cuda")),
            bool(cfg.get("sam_hq_half", True)),
        )

    if backend == "yolo11_sam":
        detection_model = YOLO(cfg.yolo11_weights)
        class_names = ultralytics_class_names(detection_model)
    elif backend == "yolo26_seg":
        detection_model = YOLO(cfg.yolo26_seg_weights)
        class_names = ultralytics_class_names(detection_model)
    elif backend == "sam3":
        class_names = list(prompt_class_names)
        detection_model = Sam3TextDetector(str(cfg.sam3_weights), class_names)
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
    if isinstance(masker, HQSamMasker):
        return masker.masks(image_bgr, xyxy_tensor, image_hw), np.ones(n, dtype=bool)
    torch.cuda.empty_cache()
    out = masker.predict(image_bgr, bboxes=xyxy_tensor, verbose=False)
    if out[0].masks is None:
        return np.zeros((n,) + tuple(image_hw), dtype=bool), np.zeros(n, dtype=bool)
    return masks_to_bool_numpy(out[0].masks.data, image_hw), np.ones(n, dtype=bool)


BLOB_MODES = ("none", "split_merge", "largest")


def split_blobs(own_masks: np.ndarray, xyxy: np.ndarray, mode: str, min_frac: float = 0.05, min_px: int = 100):
    """Connected blobs of each detection's own mask -> (boxes (M,4) xyxy, owner (M,) detection index).

    A detection mask in several pieces usually means something in front of the object cuts it, so
    the detection's box also covers the occluder. Each blob with at least max(min_px, min_frac of the
    mask area) pixels gets its own tight box (smaller specks are ignored); mode "largest" keeps only
    the biggest blob. A detection with no pixels keeps its original box.
    """
    boxes, owner = [], []
    for i, m in enumerate(own_masks):
        n, _, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        comps = sorted(((int(stats[k, cv2.CC_STAT_AREA]), k) for k in range(1, n)), reverse=True)
        if not comps:
            boxes.append(xyxy[i]); owner.append(i)
            continue
        total = sum(a for a, _ in comps)
        keep = [c for c in comps if c[0] >= max(min_px, min_frac * total)] or comps[:1]
        if mode == "largest":
            keep = keep[:1]
        for _, k in keep:
            x, y, w, h = (int(v) for v in stats[k, :4])
            boxes.append(np.array([x, y, x + w, y + h], dtype=np.float32)); owner.append(i)
    return np.asarray(boxes, dtype=np.float32), np.asarray(owner, dtype=int)


def run_detector(backend, detection_model, masker, image_bgr, image_hw, conf: float, blob_mode: str = "none"):
    """image_bgr: HxWx3 uint8 BGR array (OpenCV order, as ultralytics expects for numpy input).

    `masker` is the optional box-prompted mask model from init_detector (None = detector's masks).
    `blob_mode` (only with a masker and a detector that has its own masks): "split_merge" prompts the
    masker once per connected blob of each detection and unions the results; "largest" prompts only
    with the biggest blob (the detection's box shrinks to it); "none" uses the detector's box.
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

    if blob_mode in ("split_merge", "largest") and own is not None:
        blob_boxes, owner = split_blobs(own, xyxy_np, blob_mode)
        blob_tensor = torch.from_numpy(blob_boxes).to(xyxy_tensor.device, xyxy_tensor.dtype)
        blob_masks, blob_ok = masks_for_boxes(masker, image_bgr, blob_tensor, image_hw)
        masks_np = np.zeros((len(xyxy_np),) + tuple(image_hw), dtype=bool)
        ok = np.zeros(len(xyxy_np), dtype=bool)
        for b, i in enumerate(owner):
            if blob_ok[b]:
                masks_np[i] |= blob_masks[b]
                ok[i] = True
        if blob_mode == "largest":  # the detection's box follows the blob it was prompted with
            xyxy_np = xyxy_np.copy()
            for b, i in enumerate(owner):
                xyxy_np[i] = blob_boxes[b]
    else:
        masks_np, ok = masks_for_boxes(masker, image_bgr, xyxy_tensor, image_hw)
    if own is not None:  # masker skipped a box: keep the detector's own mask for it
        masks_np[~ok] = own[~ok]
        ok = np.ones_like(ok)
    if not ok.all():  # box-only detector and no mask for some boxes: drop them
        xyxy_np, confidences, class_ids, masks_np = xyxy_np[ok], confidences[ok], class_ids[ok], masks_np[ok]
    if len(xyxy_np) == 0:
        return empty_detections(image_hw)
    return xyxy_np, confidences, class_ids, masks_np
