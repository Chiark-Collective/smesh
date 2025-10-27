from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from typing import Dict

@dataclass
class PointBatch:
    """A batch of point data with arbitrary attributes."""
    xyz: np.ndarray                       # (N, 3)
    attrs: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.xyz = self.xyz.astype(np.float32, copy=False)
        # Ensure attributes are 1D or 2D with matching length
        n = len(self.xyz)
        for k, v in list(self.attrs.items()):
            v = v
            if v.ndim == 1 and len(v) != n:
                if not k.startswith("_"):
                    raise ValueError(f"Attribute '{k}' length {len(v)} != {n}")
            if v.ndim == 2 and v.shape[0] != n:
                if not k.startswith("_"):
                    raise ValueError(f"Attribute '{k}' first dim {v.shape[0]} != {n}")
            self.attrs[k] = v
