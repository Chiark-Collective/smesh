from __future__ import annotations
from dataclasses import dataclass, field
from typing import Set, Dict, Any, Iterable, Optional
import numpy as np
from .pointcloud import PointBatch
from .scene import MeshScene
from .utils import ensure_unit_vectors

class AttributeComputer:
    name: str = "base"
    requires: Set[str] = set()        # prerequisite attribute names
    produces: Set[str] = set()        # names it will produce

    def compute(self, batch: PointBatch, scene: MeshScene) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


class RangeComputer(AttributeComputer):
    name = "range"
    requires = {"_ray_hit_distance"}
    produces = {"range_m"}
    def __init__(self, ray_origins: Optional[np.ndarray] = None) -> None:
        self._ray_origins = ray_origins
    def compute(self, batch: PointBatch, scene: MeshScene) -> None:
        # Prefer 'range_m' from distances if provided via attrs; else recompute from origins
        if "_ray_hit_distance" in batch.attrs:
            batch.attrs["range_m"] = batch.attrs["_ray_hit_distance"].astype(np.float32, copy=False)
            return
        if self._ray_origins is None:
            raise ValueError("RangeComputer requires per-point '_ray_hit_distance' or ray_origins.")
        o = self._ray_origins[batch.attrs["_ray_index"].astype(np.int64)]
        r = np.linalg.norm(batch.xyz - o, axis=1).astype(np.float32, copy=False)
        batch.attrs["range_m"] = r


class ReturnNumberComputer(AttributeComputer):
    name = "returns"
    requires = {"_ray_index", "_hit_count_per_ray"}
    produces = {"return_number", "number_of_returns"}
    def compute(self, batch: PointBatch, scene: MeshScene) -> None:
        ray_index = batch.attrs.get("_ray_index")
        hits_per_ray = batch.attrs.get("_hit_count_per_ray")
        if ray_index is None or hits_per_ray is None:
            raise ValueError("ReturnNumberComputer requires '_ray_index' and '_hit_count_per_ray'.")
        # For each ray, assign ranks 1..K in order of appearance in the batch
        # The batch points for a given ray are expected to be in increasing distance order.
        rn = np.ones((len(batch.xyz),), dtype=np.uint8)
        nr = np.ones((len(batch.xyz),), dtype=np.uint8)
        # We'll compute rn by counting occurrences per ray
        last_idx = -1
        count_for_current_ray = 0
        for i, ri in enumerate(ray_index.astype(np.int64)):
            if ri != last_idx:
                last_idx = ri
                count_for_current_ray = 1
            else:
                count_for_current_ray += 1
            rn[i] = count_for_current_ray
        # Map number_of_returns from per-ray counts
        nr = hits_per_ray[ray_index.astype(np.int64)].astype(np.uint8, copy=False)
        batch.attrs["return_number"] = rn
        batch.attrs["number_of_returns"] = nr


class IncidenceAngleComputer(AttributeComputer):
    name = "incidence"
    requires = {"_ray_dir"}
    produces = {"incidence_deg"}
    def compute(self, batch: PointBatch, scene: MeshScene) -> None:
        dirs = batch.attrs.get("_ray_dir")
        if dirs is None:
            raise ValueError("IncidenceAngleComputer requires '_ray_dir' per point.")
        # Try to obtain normals from attrs or via probing
        nrm = batch.attrs.get("normal")
        if nrm is None:
            probe = scene.probe_attributes(batch.xyz)
            if "normal" in probe:
                nrm = probe["normal"]
                batch.attrs["normal"] = nrm
        if nrm is None:
            # Can't compute without normals
            return
        dirs = ensure_unit_vectors(dirs.astype(np.float64, copy=False))
        nrm = ensure_unit_vectors(nrm.astype(np.float64, copy=False))
        cos_theta = np.abs(np.einsum("ij,ij->i", -dirs, nrm))
        cos_theta = np.clip(cos_theta, 0.0, 1.0)
        theta = np.degrees(np.arccos(cos_theta)).astype(np.float32, copy=False)
        batch.attrs["incidence_deg"] = theta


class ScanAngleComputer(AttributeComputer):
    name = "scan_angle"
    requires = {"_scan_angle_deg_per_ray", "_ray_index"}
    produces = {"scan_angle_deg"}
    def __init__(self, mode: str = "deg") -> None:
        self.mode = mode
    def compute(self, batch: PointBatch, scene: MeshScene) -> None:
        # Expect per-ray 'scan_angle_deg' in metadata, which we've already mapped
        if "scan_angle_deg" in batch.attrs:
            return  # nothing to do
        if "_scan_angle_deg_per_ray" in batch.attrs and "_ray_index" in batch.attrs:
            batch.attrs["scan_angle_deg"] = batch.attrs["_scan_angle_deg_per_ray"][
                batch.attrs["_ray_index"].astype(np.int64)
            ].astype(np.float32, copy=False)


class IntensityLambertianComputer(AttributeComputer):
    name = "intensity"
    requires = {"incidence_deg", "range_m"}
    produces = {"intensity01"}
    def __init__(self, exponent: float = 1.0, inv_r2: bool = True) -> None:
        self.exponent = float(exponent)
        self.inv_r2 = bool(inv_r2)
    def compute(self, batch: PointBatch, scene: MeshScene) -> None:
        inc = batch.attrs.get("incidence_deg")
        if inc is None:
            return
        r = batch.attrs.get("range_m")
        if r is None:
            return
        cos_t = np.cos(np.radians(inc))
        cos_t = np.clip(cos_t, 0.0, 1.0)
        I = np.power(cos_t, self.exponent)
        if self.inv_r2:
            with np.errstate(divide='ignore'):
                I = I / np.maximum(r ** 2, 1e-6)
        # Normalize to [0,1] by dividing by max (avoid NaNs)
        maxv = float(np.max(I)) if I.size else 1.0
        if maxv > 0:
            I = I / maxv
        batch.attrs["intensity01"] = I.astype(np.float32, copy=False)


class ColorNormalProbe(AttributeComputer):
    name = "color_normal"
    produces = {"rgb", "normal"}
    def compute(self, batch: PointBatch, scene: MeshScene) -> None:
        probe = scene.probe_attributes(batch.xyz)
        for k in ("rgb", "normal"):
            if k in probe:
                batch.attrs[k] = probe[k]


class GpsTimeComputer(AttributeComputer):
    name = "gps_time"
    requires = {"_gps_time_per_ray", "_ray_index"}
    produces = {"gps_time"}
    def __init__(self, start_time_s: float = 0.0) -> None:
        self.start = float(start_time_s)
    def compute(self, batch: PointBatch, scene: MeshScene) -> None:
        if "_gps_time_per_ray" in batch.attrs and "_ray_index" in batch.attrs:
            batch.attrs["gps_time"] = self.start + batch.attrs["_gps_time_per_ray"][
                batch.attrs["_ray_index"].astype(np.int64)
            ].astype(np.float64, copy=False)


class BeamFootprintComputer(AttributeComputer):
    name = "beam_footprint"
    requires = {"range_m"}
    produces = {"beam_footprint_m"}
    def __init__(self, divergence_mrad: float = 0.3) -> None:
        self.div_mrad = float(divergence_mrad)
    def compute(self, batch: PointBatch, scene: MeshScene) -> None:
        r = batch.attrs.get("range_m")
        if r is None:
            return
        div_rad = self.div_mrad * 1e-3
        a = 0.5 * div_rad * r  # approximate radius ~ 0.5*div*range
        fp = np.column_stack([a, a]).astype(np.float32, copy=False)
        batch.attrs["beam_footprint_m"] = fp
