from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Iterable, Optional, Sequence, Set
import numpy as np

from .scene import MeshScene
from .intersector import RayBundle, RayHits, Intersector, AutoIntersector
from .pointcloud import PointBatch
from .attributes import (
    AttributeComputer, RangeComputer, ReturnNumberComputer, IncidenceAngleComputer,
    ScanAngleComputer, IntensityLambertianComputer, ColorNormalProbe, GpsTimeComputer,
    BeamFootprintComputer
)
from .utils import get_logger

_log = get_logger()

@dataclass
class SamplerConfig:
    intersector: str = "auto"
    batch_size_rays: int = 100_000
    attributes: List[str] = field(default_factory=lambda: [
        "range", "incidence", "scan_angle", "intensity", "returns", "gps_time", "color_normal"
    ])
    beam_divergence_mrad: float = 0.3

_ATTR_FACTORY: Dict[str, Any] = {
    "range": RangeComputer,
    "returns": ReturnNumberComputer,
    "incidence": IncidenceAngleComputer,
    "scan_angle": ScanAngleComputer,
    "intensity": IntensityLambertianComputer,
    "color_normal": ColorNormalProbe,
    "gps_time": GpsTimeComputer,
    "beam_footprint": BeamFootprintComputer,
}

_PRODUCER_BY_ATTR: Dict[str, str] = {}
for _name, _cls in _ATTR_FACTORY.items():
    produces = getattr(_cls, "produces", set())
    for attr in produces:
        _PRODUCER_BY_ATTR[attr] = _name

class Sampler:
    """High-level orchestrator.

    Expects an upstream component (e.g., Sensor) to generate RayBundle objects
    and optional per-ray metadata (gps_time, scan_angle, etc.). This class
    performs intersection, attribute computation, and streams to a writer.
    """
    def __init__(self, scene: MeshScene, intersector: Optional[Intersector] = None, cfg: Optional[SamplerConfig] = None) -> None:
        self.scene = scene
        self.cfg = cfg or SamplerConfig()
        if intersector is not None:
            self.intersector = intersector
        else:
            if self.cfg.intersector == "vtk":
                from .intersector import VTKIntersector
                self.intersector = VTKIntersector(obb_tree=True)
            elif self.cfg.intersector == "embree":
                from .intersector import EmbreeIntersector
                self.intersector = EmbreeIntersector()
            else:
                self.intersector = AutoIntersector()

    def _build_attribute_chain(self) -> List[AttributeComputer]:
        resolved: List[str] = []
        added: Set[str] = set()

        def add_with_dependencies(name: str) -> None:
            if name in added:
                return
            if name not in _ATTR_FACTORY:
                _log.warning("Unknown attribute '%s' – skipping.", name)
                return
            cls = _ATTR_FACTORY[name]
            requires = getattr(cls, "requires", set())
            for attr in sorted(requires):
                producer = _PRODUCER_BY_ATTR.get(attr)
                if producer is not None:
                    add_with_dependencies(producer)
            added.add(name)
            resolved.append(name)

        for name in self.cfg.attributes:
            add_with_dependencies(name)

        chain: List[AttributeComputer] = []
        for name in resolved:
            if name == "range":
                chain.append(RangeComputer())
            elif name == "beam_footprint":
                chain.append(BeamFootprintComputer(self.cfg.beam_divergence_mrad))
            else:
                chain.append(_ATTR_FACTORY[name]())
        return chain

    def _ray_bundle_chunks(self, bundle: RayBundle) -> Iterable[RayBundle]:
        limit = int(self.cfg.batch_size_rays or 0)
        n_rays = len(bundle.origins)
        if limit <= 0 or n_rays <= limit:
            yield bundle
            return

        meta = getattr(bundle, "meta", {}) or {}
        per_ray_meta: Dict[str, np.ndarray] = {}
        static_meta: Dict[str, Any] = {}
        for key, value in meta.items():
            arr = value if isinstance(value, np.ndarray) else np.asarray(value)
            if arr.ndim >= 1 and arr.shape[0] == n_rays:
                per_ray_meta[key] = arr
            else:
                static_meta[key] = value

        for start in range(0, n_rays, limit):
            stop = min(start + limit, n_rays)
            chunk_meta: Dict[str, Any] = dict(static_meta)
            for key, arr in per_ray_meta.items():
                chunk_meta[key] = arr[start:stop]
            yield RayBundle(
                origins=bundle.origins[start:stop],
                directions=bundle.directions[start:stop],
                max_range=bundle.max_range,
                multi_hit=bundle.multi_hit,
                meta=chunk_meta,
            )

    def run_to_writer(self, writer, ray_batches: Iterable[RayBundle]) -> Dict[str, Any]:
        """Stream: for each RayBundle → intersect → PointBatch → attributes → write.

        Returns run statistics.
        """
        attrs_chain = self._build_attribute_chain()

        total_rays = 0
        total_points = 0

        for bundle in ray_batches:
            total_rays += len(bundle.origins)
            for chunk in self._ray_bundle_chunks(bundle):
                hits = self.intersector.intersect(self.scene, chunk)
                if hits.points.shape[0] == 0:
                    continue

                # Map per-ray metadata to per-hit arrays
                per_hit_attrs: Dict[str, np.ndarray] = {
                    "_ray_index": hits.ray_index,
                    "_hit_count_per_ray": hits.hit_count_per_ray,
                    "_ray_hit_distance": hits.distances,
                    "_ray_dir": chunk.directions[hits.ray_index],
                }
                meta = getattr(chunk, "meta", {})
                n_rays = len(chunk.origins)
                for key, value in meta.items():
                    arr = np.asarray(value)
                    if arr.ndim >= 1 and arr.shape[0] == n_rays:
                        per_hit_attrs[f"_{key}_per_ray"] = arr
                        mapped = arr[hits.ray_index]
                        if key == "scan_angle_deg":
                            per_hit_attrs["_scan_angle_deg_per_ray"] = arr
                        per_hit_attrs[key] = mapped.astype(arr.dtype, copy=False)
                    elif arr.ndim >= 1 and arr.shape[0] == len(hits.points):
                        per_hit_attrs[key] = arr
                    else:
                        per_hit_attrs[key] = arr

                batch = PointBatch(xyz=hits.points, attrs=per_hit_attrs)

                # Run attribute computers
                for comp in attrs_chain:
                    comp.compute(batch, self.scene)

                # Drop private attrs before writing
                for k in [k for k in list(batch.attrs.keys()) if k.startswith("_")]:
                    del batch.attrs[k]

                writer.write_batch(batch)
                total_points += len(batch.xyz)

        writer.close()
        stats = {"rays": total_rays, "points": total_points}
        _log.info("Sampler finished: %d rays → %d points", total_rays, total_points)
        return stats
