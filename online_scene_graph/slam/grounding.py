"""Live CLIP grounding: find() copies features under a lock, then scores.

Mapping thread holds map_lock while it mutates MapObjectList. Query threads
copy clip_ft + metadata only — never Open3D geometries. CLIP text encode uses
clip_lock so it does not overlap the mapper's image encode.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any, Optional
from uuid import UUID

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Hit:
    id: str
    class_name: str
    sim: float
    center: list[float]
    aabb: list[float]
    n_obs: int
    generation: int

    def to_dict(self) -> dict:
        return asdict(self)


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _obj_id(obj: dict) -> str:
    raw = obj.get("id")
    if isinstance(raw, UUID):
        return str(raw)
    return str(raw)


def _aabb_and_center(obj: dict) -> tuple[list[float], list[float]]:
    if obj.get("center") is not None and obj.get("aabb") is not None:
        center = [float(x) for x in np.asarray(obj["center"]).reshape(-1)[:3]]
        aabb = [float(x) for x in np.asarray(obj["aabb"]).reshape(-1)[:6]]
        return center, aabb
    bbox = obj.get("bbox")
    if bbox is None:
        return [float("nan")] * 3, [float("nan")] * 6
    center = [float(x) for x in np.asarray(bbox.get_center()).reshape(-1)[:3]]
    aabb_box = bbox
    if hasattr(bbox, "get_axis_aligned_bounding_box"):
        aabb_box = bbox.get_axis_aligned_bounding_box()
    lo = np.asarray(aabb_box.get_min_bound(), dtype=np.float32).reshape(-1)[:3]
    hi = np.asarray(aabb_box.get_max_bound(), dtype=np.float32).reshape(-1)[:3]
    aabb = [float(x) for x in np.concatenate([lo, hi])]
    return center, aabb


class GroundingService:
    def __init__(
        self,
        clip_model=None,
        clip_tokenizer=None,
        device: str = "cuda",
        default_k: int = 5,
        default_min_sim: float = 0.25,
        default_min_obs: int = 3,
    ):
        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer
        self.device = device
        self.default_k = int(default_k)
        self.default_min_sim = float(default_min_sim)
        self.default_min_obs = int(default_min_obs)
        self.map_lock = Lock()
        self.clip_lock = Lock()
        self.objects: list = []
        self.generation = 0

    def set_objects(self, objects) -> None:
        """Caller must hold map_lock. Bumps generation so clients can discard stale hits."""
        self.objects = objects
        self.generation += 1

    def encode_text(self, text: str) -> np.ndarray:
        if self.clip_model is None or self.clip_tokenizer is None:
            raise RuntimeError("CLIP text tower is not loaded")
        with self.clip_lock:
            with torch.no_grad():
                tokens = self.clip_tokenizer([text]).to(self.device)
                q = self.clip_model.encode_text(tokens)
                q = F.normalize(q, dim=-1).squeeze(0)
        return q.detach().cpu().numpy().astype(np.float32)

    def _copy_index(self, min_obs: int) -> Optional[dict[str, Any]]:
        rows = []
        feats = []
        generation = self.generation
        for obj in self.objects:
            n_obs = int(obj.get("num_detections", 0))
            if n_obs < min_obs:
                continue
            ft = obj.get("clip_ft")
            if ft is None:
                continue
            vec = _as_numpy(ft).astype(np.float32).reshape(-1)
            norm = float(np.linalg.norm(vec))
            if norm < 1e-8:
                continue
            vec = vec / norm
            center, aabb = _aabb_and_center(obj)
            rows.append(
                {
                    "id": _obj_id(obj),
                    "class_name": str(obj.get("class_name", "")),
                    "n_obs": n_obs,
                    "center": center,
                    "aabb": aabb,
                }
            )
            feats.append(vec)
        if not rows:
            return None
        return {
            "generation": generation,
            "rows": rows,
            "feats": np.stack(feats, axis=0),
        }

    def find(
        self,
        text: str,
        k: Optional[int] = None,
        min_sim: Optional[float] = None,
        min_obs: Optional[int] = None,
        query_ft: Optional[np.ndarray] = None,
    ) -> list[Hit]:
        k = self.default_k if k is None or int(k) <= 0 else int(k)
        min_sim = self.default_min_sim if min_sim is None else float(min_sim)
        min_obs = self.default_min_obs if min_obs is None or int(min_obs) <= 0 else int(min_obs)
        if query_ft is None:
            query_ft = self.encode_text(text)
        query = np.asarray(query_ft, dtype=np.float32).reshape(-1)
        query = query / max(float(np.linalg.norm(query)), 1e-8)

        with self.map_lock:
            snapshot = self._copy_index(min_obs)
        if snapshot is None:
            return []

        sims = snapshot["feats"] @ query
        order = np.argsort(-sims)
        hits: list[Hit] = []
        for i in order:
            sim = float(sims[i])
            if sim < min_sim:
                continue
            row = snapshot["rows"][i]
            hits.append(
                Hit(
                    id=row["id"],
                    class_name=row["class_name"],
                    sim=sim,
                    center=row["center"],
                    aabb=row["aabb"],
                    n_obs=row["n_obs"],
                    generation=int(snapshot["generation"]),
                )
            )
            if len(hits) >= k:
                break
        return hits

    def get(self, obj_id: str) -> Optional[Hit]:
        with self.map_lock:
            generation = self.generation
            for obj in self.objects:
                if _obj_id(obj) != obj_id:
                    continue
                ft = obj.get("clip_ft")
                center, aabb = _aabb_and_center(obj)
                return Hit(
                    id=obj_id,
                    class_name=str(obj.get("class_name", "")),
                    sim=float("nan"),
                    center=center,
                    aabb=aabb,
                    n_obs=int(obj.get("num_detections", 0)),
                    generation=generation,
                )
        return None

    def handle(self, request: dict) -> dict:
        cmd = str(request.get("cmd", "find"))
        try:
            if cmd == "get":
                hit = self.get(str(request.get("id", "")))
                return {
                    "ok": hit is not None,
                    "generation": self.generation,
                    "hit": None if hit is None else hit.to_dict(),
                }
            hits = self.find(
                text=str(request.get("text", "")),
                k=request.get("k"),
                min_sim=request.get("min_sim"),
                min_obs=request.get("min_obs"),
            )
            return {
                "ok": True,
                "generation": hits[0].generation if hits else self.generation,
                "hits": [h.to_dict() for h in hits],
            }
        except Exception as exc:
            return {"ok": False, "message": str(exc), "hits": []}
