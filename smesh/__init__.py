"""Smesh – Mesh→Point-Cloud Sensor Simulator (core skeleton).

This package contains the critical core components outlined in the design doc:
- MeshScene (core.scene)
- Intersector & Ray data structures (core.intersector) [VTK & Embree backends]
- PointBatch & helpers (core.pointcloud)
- AttributeComputer plug-in API + a few basics (core.attributes)
- Streaming LAS/LAZ writer (core.exporter)
- High-level Sampler orchestrator (core.sampler)

The code here is intentionally minimal but functional and type-annotated,
with optional dependencies guarded so you can start wiring sensors/trajectories.
"""

from .core.scene import MeshScene
from .core.intersector import (RayBundle, RayHits, Intersector,
                               VTKIntersector, EmbreeIntersector, AutoIntersector)
from .core.pointcloud import PointBatch
from .core.exporter import LasWriter, PlyWriter, NpzWriter
from .core.attributes import (
    AttributeComputer,
    RangeComputer, ReturnNumberComputer, IncidenceAngleComputer,
    ScanAngleComputer, IntensityLambertianComputer, ColorNormalProbe,
    GpsTimeComputer, BeamFootprintComputer
)
from .core.sampler import Sampler, SamplerConfig
from .sensors.lidar import LidarSensor
from .sensors.totalstation import TotalStationSensor
from .sensors.camera import CameraSensor
from .sensors.patterns import CameraPattern
