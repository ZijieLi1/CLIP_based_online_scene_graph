"""Frame ingest for live mapping: one latest slot, never block the producer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import torch


@dataclass
class Frame:
    """One RGB-D observation in mapper-native types (BGR uint8, depth meters, T_wc)."""

    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    pose: np.ndarray
    t: float
    dropped_before: int = 0

    def intrinsics_4x4(self) -> torch.Tensor:
        mat = torch.eye(4, dtype=torch.float32)
        mat[:3, :3] = torch.from_numpy(np.asarray(self.K, dtype=np.float32))
        return mat


class FrameSource(Protocol):
    def start(self) -> None: ...

    def next(self, timeout: float = 1.0) -> Optional[Frame]: ...

    def close(self) -> None: ...
