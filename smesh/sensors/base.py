from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Protocol
import numpy as np
from ..core.intersector import RayBundle

@dataclass
class SensorBatch:
    bundle: RayBundle
    aux: Dict[str, np.ndarray] = field(default_factory=dict)  # alias to bundle.meta

class Sensor(Protocol):
    def batches(self, rng: Optional[np.random.Generator] = None) -> Iterable[SensorBatch]: ...
