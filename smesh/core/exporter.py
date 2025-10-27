from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pathlib
import warnings

import laspy  # type: ignore
from .pointcloud import PointBatch
from .utils import get_logger

_log = get_logger()

@dataclass
class LasWriter:
    """Streaming LAS/LAZ writer using laspy (v2+).

    Header (scales/offsets/point format) is created lazily on first batch,
    so we can infer which ExtraBytes dimensions are needed from actual attrs.
    """
    path: str
    point_format: int = 8
    compress: bool = False
    scale: tuple[float, float, float] = (1e-3, 1e-3, 1e-3)
    offset: Optional[tuple[float, float, float]] = None

    def __post_init__(self) -> None:
        self._fh: Optional[laspy.LasWriter] = None  # type: ignore
        self._header: Optional[laspy.LasHeader] = None  # type: ignore
        self._defined_extras: Dict[str, Any] = {}

    # -- public API --
    def write_batch(self, batch: PointBatch) -> None:
        if self._fh is None:
            self._init_header_from_batch(batch)
        assert self._fh is not None and self._header is not None
        pts = self._point_record_from_batch(batch, self._header, self._defined_extras)
        self._fh.write_points(pts)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # -- internals --
    def _init_header_from_batch(self, batch: PointBatch) -> None:
        # Determine which standard fields we will populate
        pf = laspy.PointFormat(self.point_format)
        hdr = laspy.LasHeader(point_format=pf, version="1.4")
        hdr.scales = self.scale
        if self.offset is None:
            # Infer offset from first batch
            mn = np.min(batch.xyz, axis=0)
            hdr.offsets = (float(mn[0]), float(mn[1]), float(mn[2]))
        else:
            hdr.offsets = self.offset

        # Add ExtraBytes for non-standard attrs we recognize
        extras: Dict[str, laspy.ExtraBytesParams] = {}
        def add_extra(name: str, dtype: str) -> None:
            if name in extras:
                return
            extras[name] = laspy.ExtraBytesParams(name=name, type=dtype)

        # Normal
        if "normal" in batch.attrs:
            add_extra("NormalX", "float32")
            add_extra("NormalY", "float32")
            add_extra("NormalZ", "float32")
        # Incidence angle
        if "incidence_deg" in batch.attrs:
            add_extra("IncidenceAngle", "float32")
        # Range
        if "range_m" in batch.attrs:
            add_extra("Range", "float32")
        # Beam footprint
        if "beam_footprint_m" in batch.attrs:
            add_extra("BeamFootprintMajor", "float32")
            add_extra("BeamFootprintMinor", "float32")
        # IDs
        for name, dtype in [
            ("scanline_id", "uint32"),
            ("channel_id", "uint16"),
            ("revolution_id", "uint32"),
            ("flightline_id", "uint32"),
            ("cam_id", "uint16"),
            ("exposure_id", "uint16"),
        ]:
            if name in batch.attrs:
                add_extra(name, dtype)
        # Pixels
        for name in ("pixel_u", "pixel_v"):
            if name in batch.attrs:
                add_extra(name, "float32")
        # Ground-truth / misc
        if "gt" in batch.attrs:
            # Note: dtype unknown; assume uint8 unless provided as float32
            if batch.attrs["gt"].dtype == np.float32:
                add_extra("gt", "float32")
            else:
                add_extra("gt", "uint8")

        # Register extras
        for p in extras.values():
            hdr.add_extra_dim(p)
        self._defined_extras = {p.name: p for p in extras.values()}

        mode = "w"
        path = pathlib.Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = laspy.open(path, mode=mode, header=hdr, do_compress=self.compress)
        self._header = hdr
        _log.info("Opened %s (PF=%d, compress=%s)", path.name, self.point_format, self.compress)

    def _point_record_from_batch(
        self, batch: PointBatch, header: "laspy.LasHeader", defined_extras: Dict[str, Any]
    ) -> "laspy.ScaleAwarePointRecord":
        n = len(batch.xyz)
        pts = laspy.ScaleAwarePointRecord.zeros(n, header=header)

        # Coordinates (scaled floats are accepted by laspy)
        pts.x = batch.xyz[:, 0]
        pts.y = batch.xyz[:, 1]
        pts.z = batch.xyz[:, 2]

        # Intensity
        if "intensity01" in batch.attrs:
            v = np.clip(batch.attrs["intensity01"].astype(np.float64), 0.0, 1.0)
            pts.intensity = (v * 65535.0 + 0.5).astype(np.uint16)
        elif "intensity" in pts.point_format.dimension_names:
            # Default to 0 if field exists but we didn't compute intensity
            pts.intensity = np.zeros((n,), dtype=np.uint16)

        # Scan angle
        if "scan_angle_deg" in batch.attrs and "scan_angle" in pts.point_format.dimension_names:
            s = np.clip(np.round(batch.attrs["scan_angle_deg"]).astype(np.int16), -90, 90)
            pts.scan_angle = s

        # Returns
        if "return_number" in batch.attrs and "return_number" in pts.point_format.dimension_names:
            pts.return_number = batch.attrs["return_number"].astype(np.uint8, copy=False)
        if "number_of_returns" in batch.attrs and "number_of_returns" in pts.point_format.dimension_names:
            pts.number_of_returns = batch.attrs["number_of_returns"].astype(np.uint8, copy=False)

        # GPS time
        if "gps_time" in pts.point_format.dimension_names and "gps_time" in batch.attrs:
            pts.gps_time = batch.attrs["gps_time"].astype(np.float64, copy=False)

        # Color
        if "rgb" in batch.attrs and all(nm in pts.point_format.dimension_names for nm in ("red","green","blue")):
            rgb = batch.attrs["rgb"].astype(np.uint16, copy=False)
            if rgb.max() <= 255:
                rgb = (rgb.astype(np.uint16) * 257)  # 0..255 -> 0..65535
            pts.red = rgb[:, 0]
            pts.green = rgb[:, 1]
            pts.blue = rgb[:, 2]

        # Extra bytes mapping
        def set_extra(name: str, values: np.ndarray) -> None:
            if name not in defined_extras:
                warnings.warn(f"Extra dimension '{name}' was not declared in header; skipping.")
                return
            pts[name] = values

        if "normal" in batch.attrs:
            nrm = batch.attrs["normal"].astype(np.float32, copy=False)
            set_extra("NormalX", nrm[:, 0])
            set_extra("NormalY", nrm[:, 1])
            set_extra("NormalZ", nrm[:, 2])

        if "incidence_deg" in batch.attrs:
            set_extra("IncidenceAngle", batch.attrs["incidence_deg"].astype(np.float32, copy=False))

        if "range_m" in batch.attrs:
            set_extra("Range", batch.attrs["range_m"].astype(np.float32, copy=False))

        if "beam_footprint_m" in batch.attrs:
            fp = batch.attrs["beam_footprint_m"].astype(np.float32, copy=False)
            if fp.ndim == 1:
                major = fp
                minor = fp
            else:
                major = fp[:, 0]
                minor = fp[:, -1]
            set_extra("BeamFootprintMajor", major)
            set_extra("BeamFootprintMinor", minor)

        for k in ("scanline_id","channel_id","revolution_id","flightline_id","cam_id","exposure_id"):
            if k in batch.attrs:
                set_extra(k, batch.attrs[k])

        for k in ("pixel_u","pixel_v","gt"):
            if k in batch.attrs:
                set_extra(k, batch.attrs[k])

        return pts


# Minimal PLY and NPZ writers for testing / debugging
class PlyWriter:
    def __init__(self, path: str) -> None:
        self.path = path
        self._xyz_chunks: List[np.ndarray] = []

    def write_batch(self, batch: PointBatch) -> None:
        self._xyz_chunks.append(batch.xyz.astype(np.float32, copy=False))

    def close(self) -> None:
        if not self._xyz_chunks:
            return
        # Very simple ASCII PLY for xyz only (buffered, written once on close)
        path = pathlib.Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        xyz = np.vstack(self._xyz_chunks)
        with open(path, "w", encoding="utf-8") as f:
            n = len(xyz)
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("end_header\n")
            for x, y, z in xyz:
                f.write(f"{float(x)} {float(y)} {float(z)}\n")
        self._xyz_chunks.clear()


class NpzWriter:
    def __init__(self, path: str) -> None:
        self.path = path
        self._batches: List[PointBatch] = []

    def write_batch(self, batch: PointBatch) -> None:
        self._batches.append(batch)

    def close(self) -> None:
        if not self._batches:
            return
        path = pathlib.Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Concatenate
        xyz = np.vstack([b.xyz for b in self._batches])
        all_keys = sorted({k for b in self._batches for k in b.attrs.keys()})
        # Map each attr key to (tail_shape, dtype) from first batch that provides it
        key_meta: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = {}
        for b in self._batches:
            for k, v in b.attrs.items():
                if k not in key_meta:
                    tail = v.shape[1:] if v.ndim >= 2 else ()
                    key_meta[k] = (tail, v.dtype)

        out: Dict[str, np.ndarray] = {"xyz": xyz}
        for k in all_keys:
            tail_shape, dt = key_meta[k]
            vals: List[np.ndarray] = []
            for b in self._batches:
                if k in b.attrs:
                    vals.append(b.attrs[k].astype(dt, copy=False))
                else:
                    filler_shape = (len(b.xyz),) + tail_shape
                    vals.append(np.zeros(filler_shape, dtype=dt))
            out[k] = np.concatenate(vals, axis=0)
        np.savez_compressed(path, **out)
        self._batches.clear()
