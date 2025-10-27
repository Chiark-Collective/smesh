
# **Smesh** – Design Sketch for an Open‑Source Mesh→Point‑Cloud Sensor Simulator

> A principled, configurable Python package for simulating point clouds from triangle meshes with realistic capture behaviors (LiDAR, photogrammetry, total station, etc.).

---

## 0) Goals & Non‑Goals

**Goals**

* Clean, composable APIs: **Trajectory** (platform motion) × **SensorModel** (ray/pixel generation) × **Intersector** (mesh hits) × **Noise** × **Attributes** × **Exporter**.
* First‑class support for multiple capture strategies:

  * Aerial LiDAR (oscillating mirror)
  * Mobile mapping LiDAR (spinning)
  * Total station (raster az/el)
  * Aerial photogrammetry (nadir/oblique)
  * Smartphone stereo (handheld multi‑view)
* Derive as many **standard attributes** as possible (scan angle, intensity, return numbers, GPS time, line/flight identifiers, incidence angle, beam footprint, RGB, normals, etc.) with a **switchboard** to include/exclude.
* Streamed operation to scale beyond memory: process rays in **batches**, write LAS/LAZ incrementally.
* CLI that reads **YAML config** or flags: `smesh sample --mesh scene.ply --config config.yaml --out out.las`.
* Solid tests: **unit** (each component) + **integration** (scenario end‑to‑end) + **reproducibility** (seeds).

**Non‑Goals (MVP)**

* Physically accurate photometry/material BRDFs.
* Full MVS photogrammetry; we approximate via raycast/Z‑buffer depth.
* GPU ray tracing (we keep the interface so we can add it later).

---

## 1) Package Layout

```
smesh/
  smesh/
    __init__.py
    config/
      schema.py              # pydantic/dataclass schemas + JSON schema export
      defaults.yaml
    core/
      scene.py               # MeshScene, mesh IO, acceleration builders
      intersector.py         # Intersector interface; vtk, embree backends
      raybundle.py           # RayBundle, RayHits data structures
      pointcloud.py          # PointBatch, PointCloud, merge/concat
      attributes.py          # AttributeComputer plug-ins
      utils.py               # math/transforms, RNG, logging
      exporter.py            # LAS/LAZ/PLY/NPZ writers (streaming)
    motion/
      pose.py                # Pose, time utils
      trajectory.py          # base + Static, Lawnmower, Polyline, Spline
    sensors/
      base.py                # Sensor base class
      lidar.py               # LidarSensor (oscillating/spinning/raster)
      camera.py              # CameraSensor (pinhole), PhotogrammetrySampler
      totalstation.py        # TS convenience
      patterns.py            # ScanPattern base + oscillating/spinning/raster
      noise.py               # NoiseModel base + LidarNoise + PhotoNoise + IMU drift
    cli/
      __init__.py
      main.py                # click/typer CLI entrypoint
    plugins/
      __init__.py            # registry for third-party extensions
  tests/
    unit/
    integration/
  examples/
    configs/
  docs/
    design.md                # this document
    api.md
    cli.md
  pyproject.toml
  README.md
  LICENSE (Apache-2.0)
```

---

## 2) Core Concepts & APIs

### 2.1 Scene & Intersections

```python
# core/scene.py
class MeshScene:
    def __init__(self, mesh_path: str | Path | None = None,
                 mesh_data: Optional[vtk.vtkPolyData] = None,
                 require_normals: bool = True,
                 build_colors: bool = True) -> None: ...
    def bounds(self) -> tuple[float, float, float, float, float, float]: ...
    def probe_attributes(self, xyz: np.ndarray) -> dict[str, np.ndarray]: ...
```

```python
# core/intersector.py
@dataclass
class RayBundle:
    origins: np.ndarray        # (M,3)
    directions: np.ndarray     # (M,3) unit
    max_range: float
    multi_hit: bool
    # optional per-ray metadata (time, channel, etc.) forwarded for attribute computation
    meta: dict[str, np.ndarray] = field(default_factory=dict)

@dataclass
class RayHits:
    # ragged via indexing
    points: np.ndarray         # (K,3) concatenated hits
    distances: np.ndarray      # (K,)
    ray_index: np.ndarray      # (K,) maps hit->source ray idx
    cell_ids: np.ndarray       # (K,) mesh triangle ids
    hit_count_per_ray: np.ndarray  # (M,)

class Intersector:
    def intersect(self, scene: MeshScene, bundle: RayBundle) -> RayHits: ...
```

**Backends**

* `VTKIntersector(obb_tree=True)`
* `EmbreeIntersector()` (via `trimesh.ray.ray_pyembree`, optional dep)
* `AutoIntersector()` picks the fastest available.

### 2.2 Motion & Time

```python
# motion/pose.py
@dataclass
class Pose:
    t: np.ndarray  # (3,)
    R: np.ndarray  # (3,3)
    # helpers: from_xyz_rpy, multiply, inverse, etc.

# motion/trajectory.py
class Trajectory(Protocol):
    def sample(self, t: float) -> Pose: ...
    def timeline(self) -> Iterable[tuple[float, Pose]]: ...  # discrete keyframes

class StaticTrajectory(Trajectory): ...
class LawnmowerTrajectory(Trajectory): ...
class PolylineTrajectory(Trajectory): ...
class SplineTrajectory(Trajectory): ...
```

### 2.3 Sensors (abstract) & Patterns

```python
# sensors/base.py
@dataclass
class SensorBatch:
    bundle: RayBundle
    # Anything needed later for attributes (scanline ids, az/el, pixel coords, etc.)
    aux: dict[str, np.ndarray]

class Sensor(ABC):
    """Produces RayBundles from a timed platform pose stream."""
    def __init__(self, boresight_R: np.ndarray | None = None,
                 lever_arm_t: np.ndarray | None = None) -> None: ...
    @abstractmethod
    def batches(self, trajectory: Trajectory, rng: np.random.Generator) -> Iterable[SensorBatch]: ...
```

**Patterns** (`sensors/patterns.py`)

* `ScanPattern` → unit directions in sensor frame at a given *relative* time.
* Implementations:

  * `OscillatingMirrorPattern(fov_deg, line_rate_hz, pulses_per_line)`
  * `SpinningPattern(vertical_angles_deg: list[float], rpm: float, prf_hz: int)`
  * `RasterAzElPattern(az_range, el_range, step)` (total station)

### 2.4 Noise

```python
# sensors/noise.py
class NoiseModel(ABC):
    def jitter_directions(self, dirs: np.ndarray, rng, sigma_deg: float) -> np.ndarray: ...
    def jitter_ranges(self, ranges: np.ndarray, rng, sigma_m: float | np.ndarray) -> np.ndarray: ...
    def dropout_mask(self, n: int, rng, keep_prob: float) -> np.ndarray: ...
    def time_jitter(self, t: np.ndarray, rng, sigma_s: float) -> np.ndarray: ...

class LidarNoise(NoiseModel): ...
class PhotogrammetryNoise(NoiseModel): ...
```

### 2.5 Attributes (computed post‑hit)

```python
# core/attributes.py
class AttributeComputer(ABC):
    """Adds/updates fields on PointBatch based on hits + auxiliary ray metadata."""
    name: str
    requires: set[str] = set()  # prerequisite attribute names (e.g., "normal")
    produces: set[str]

    @abstractmethod
    def compute(self, batch: "PointBatch", scene: MeshScene) -> None: ...

# Built-ins (examples)
ScanAngleComputer(mode="deg|rank")
IncidenceAngleComputer()
IntensityLambertianComputer(albedo_channel: str | None = None, exponent: float = 1.0, inv_r2=True)
ReturnNumberComputer()
RangeComputer()
BeamFootprintComputer(divergence_mrad: float)
GpsTimeComputer(start_time_s: float, per_ray_times: np.ndarray | None)
FlightlineIdComputer()
ColorNormalProbe()  # calls scene.probe_attributes for rgb/normal
ClassificationComputer(constant: int | None, from_cell_data: str | None)
```

Configure which to run via a list in config: `attributes: [range, incidence, scan_angle, intensity, ...]`.

### 2.6 Point Data & Export

```python
# core/pointcloud.py
@dataclass
class PointBatch:
    xyz: np.ndarray           # (N,3)
    attrs: dict[str, np.ndarray]
    # may include per-point ray meta already mapped to attrs

class PointCloud:
    def __iter__(self): ...    # iterate batches (for streaming)
    def materialize(self) -> "PointCloud": ...  # gather to memory

# core/exporter.py
class LasWriter:
    def __init__(self, path: str, point_format=8, compress=False, scale=(1e-3,1e-3,1e-3)) -> None: ...
    def write_batch(self, batch: PointBatch) -> None: ...
    def close(self) -> None: ...

class PlyWriter: ...
class NpzWriter: ...
```

**LAS field mapping (defaults)**

* `x,y,z` ← `batch.xyz`
* `intensity` ← `intensity01 * 65535`
* `scan_angle` ← `scan_angle_deg` (int16)
* `return_number`, `number_of_returns`, `gps_time`
* `red/green/blue` ← `rgb` (0..255 → 0..65535)
* Extra bytes: `NormalX/Y/Z`, `IncidenceAngle`, `BeamFootprint`, `ChannelId`, `LineId`, `FlightlineId`, `gt`, etc.

---

## 3) High‑Level Orchestrator (Sampler)

```python
# core/sampler.py
@dataclass
class SamplerConfig:
    intersector: str = "auto"
    batch_size_rays: int = 100_000
    attributes: list[str] = field(default_factory=lambda: ["range","incidence","scan_angle","intensity","returns","gps_time","color_normal"])

class Sampler:
    def __init__(self, scene: MeshScene, sensor: Sensor, cfg: SamplerConfig) -> None: ...
    def run_to_writer(self, writer) -> dict[str, Any]:
        """
        Stream batches:
          for Sensor.batches(...):
            - build RayBundle
            - Intersector.intersect(...)
            - convert to PointBatch
            - run AttributeComputers
            - writer.write_batch(batch)
        Returns run metadata (counts, timings).
        """
```

---

## 4) Concrete Capture Strategies

Below: what we ship in v0.1 (ready to use).

### 4.1 Aerial Photogrammetry

**What we simulate**

* Platform: lawnmower flight at altitude.
* Sensor: pinhole camera. For MVP, we raycast from selected pixels (subsample) per exposure; later we can replace with Z‑buffer depth rendering for density.
* Attributes: `gps_time`, `cam_id`, `exposure_id`, `pixel_u/v`, `normal`, `rgb`, optional `range/incidence`.

**Config (YAML)**

```yaml
type: photogrammetry
trajectory:
  kind: lawnmower
  altitude_m: 120
  speed_mps: 15
  line_spacing_m: 60
  heading_deg: 0
camera:
  res_px: [4000, 3000]
  focal_px: [2200, 2200]
  principal_px: [2000, 1500]
  pixel_stride: 12         # subsampling for speed
  shutter: global          # (later: rolling)
noise:
  px_sigma: 0.5
attributes: [gps_time, color_normal, range, incidence]
output:
  path: out_aerial_photogrammetry.las
  compress: false
```

**Notes**

* `exposure cadence` is derived from speed & line spacing; we map each exposure to a `gps_time`.
* Later: support desired GSD/overlap to auto‑compute line spacing & exposure rate.

### 4.2 Aerial LiDAR (Oscillating Mirror)

**What we simulate**

* Platform: lawnmower flight.
* Pattern: line scanner with `fov_deg`, `line_rate_hz`, `pulses_per_line`.
* Per‑ray time offsets: `t_line + i/pprf` (derived from `line_rate * pulses_per_line`).
* Attributes: `intensity`, `scan_angle_deg` (mirror deflection), `scanline_id`, `edge_of_flight_line` (based on lateral extent), `return_number/number_of_returns`, `gps_time`.

**Config**

```yaml
type: lidar
trajectory:
  kind: lawnmower
  altitude_m: 120
  speed_mps: 20
  line_spacing_m: 150
pattern:
  kind: oscillating
  fov_deg: 60
  line_rate_hz: 80
  pulses_per_line: 1200
noise:
  sigma_range_m: 0.03
  sigma_angle_deg: 0.03
  keep_prob: 0.998
sensor:
  max_range_m: 600
  beam_divergence_mrad: 0.3
  multi_return: true
attributes:
  - gps_time
  - scan_angle
  - returns
  - intensity
  - incidence
  - color_normal
  - range
  - flightline_id
output:
  path: out_aerial_lidar.laz
  compress: true            # requires lazrs
```

**Scan angle**
We define `scan_angle_deg` as the instantaneous mirror deflection (±FOV/2), signed with respect to the nadir axis in the sensor frame, mapped to LAS `scan_angle` (int16, degrees rounded). The `attributes.ScanAngleComputer` uses pattern’s deflection directly.

### 4.3 Mobile Mapping LiDAR (Spinning)

**What we simulate**

* Platform: vehicle drive‑by (polyline trajectory).
* Pattern: spinning LiDAR with `vertical_angles_deg`, `rpm`, `prf_hz`.
* Attributes: `channel_id`, `revolution_id`, `azimuth_deg`, `scan_angle_deg` (vertical angle), `return_number/number_of_returns`, `intensity`, `gps_time`.

**Config**

```yaml
type: lidar
trajectory:
  kind: polyline
  waypoints: [[-20,0,2], [20,0,2]]
  speed_mps: 5
pattern:
  kind: spinning
  vertical_angles_deg: [-15,1,-13,-3,-11,5,-9,7,-7,9,-5,11,-3,13,-1,15]
  rpm: 10
  prf_hz: 300000
noise:
  sigma_range_m: 0.015
  sigma_angle_deg: 0.05
  keep_prob: 0.995
sensor:
  max_range_m: 150
  multi_return: true
attributes: [gps_time, channel_id, azimuth_deg, scan_angle, intensity, returns, incidence, range, color_normal]
output:
  path: out_mobile_lidar.las
```

**Scan angle**
Here `scan_angle_deg` = **vertical channel angle** (signed). `azimuth_deg` is computed per pulse at time `t`.

### 4.4 Total Station (Raster Az/El)

**What we simulate**

* Platform: static pose.
* Pattern: azimuth‑elevation raster with small steps (high accuracy).
* Attributes: `return_number/number_of_returns` (usually single), `scan_angle_deg=el` or `rank`, `gps_time` (constant), `range`, `incidence`, `intensity`.

**Config**

```yaml
type: lidar
trajectory:
  kind: static
  xyzrpy: [-5, 0, 1.7, 0, 0, 0]
pattern:
  kind: raster
  az_deg: [-60, 60]
  el_deg: [-20, 45]
  step_deg: [0.1, 0.1]
noise:
  sigma_range_m: 0.003
  sigma_angle_deg: 0.01
  keep_prob: 0.9995
sensor:
  max_range_m: 250
  multi_return: false
attributes: [range, incidence, intensity, scan_angle, gps_time, color_normal]
output:
  path: out_total_station.las
```

### 4.5 Smartphone Stereo (Handheld Multi‑View)

**What we simulate**

* Platform: a few hand‑held exposures around the scene.
* Sensor: camera with high pixel noise, subsampled pixels.
* Attributes: `exposure_id`, `pixel_u/v`, `gps_time`, `rgb`, `normal`, optional `range/incidence`.

**Config**

```yaml
type: photogrammetry
exposures:
  - [0.0,  [-2, -2, 1.6, 0, 0,  45]]
  - [0.5,  [ 2, -2, 1.6, 0, 0, 135]]
  - [1.0,  [ 2,  2, 1.6, 0, 0,-135]]
  - [1.5,  [-2,  2, 1.6, 0, 0, -45]]
camera:
  res_px: [3024, 4032]
  focal_px: [1700, 1700]
  pixel_stride: 12
noise:
  px_sigma: 1.0
attributes: [gps_time, color_normal, range, incidence]
output:
  path: out_phone_stereo.las
```

---

## 5) CLI

### 5.1 Command

```
$ smesh sample --mesh scene.ply --config configs/aerial_lidar.yaml --out out.laz \
    [--engine auto|vtk|embree] [--chunk 200000] [--seed 123] [--no-color]
```

**Flags**

* `--mesh`: input mesh path (`.ply`, `.obj`, `.vtp`).
* `--config`: YAML file; can be combined with flags that override keys.
* `--out`: output `.las` or `.laz`.
* `--engine`: intersection backend.
* `--chunk`: rays per batch (memory/compute knob).
* `--seed`: RNG seed for reproducibility.
* `--attr include,list`: override attributes enabled.
* `--no-color`: skip probing RGB.
* `--progress/--no-progress`: progress bar.
* `--stats`: print run summary (points, drops, timing).
* `validate-config`: subcommand to check schema & defaults.

### 5.2 Config Schema (abridged)

```yaml
type: lidar | photogrammetry
mesh_units: meters

trajectory:
  kind: lawnmower | static | polyline | spline
  # per-kind fields...

sensor:            # lidar or camera specific
  max_range_m: 600
  multi_return: true
  beam_divergence_mrad: 0.3
  boresight_rpy_deg: [0,0,0]
  lever_arm_m: [0,0,0]

pattern:           # for lidar kinds
  kind: oscillating | spinning | raster
  # per-kind fields...

camera:            # for photogrammetry kinds
  res_px: [W,H]
  focal_px: [fx,fy]
  principal_px: [cx,cy]
  pixel_stride: 12

noise:
  # lidar or photo specific

attributes:
  - gps_time
  - range
  - incidence
  - intensity
  - scan_angle
  - returns
  - color_normal
  - channel_id
  - flightline_id
  # ...

output:
  path: out.las
  compress: false
  scale: [0.001, 0.001, 0.001]
engine: auto
chunk_rays: 100000
seed: 123
```

Validation via pydantic or JSON schema; CLI prints helpful errors and expanded defaults.

---

## 6) Resource Strategy (MVP)

* **Streaming batches**: generate rays in chunks (per scanline / per revolution / per exposure), intersect, compute attributes, and **write immediately**. Avoid materializing the entire cloud.
* **Ragged hits** (multi‑returns): use concatenated arrays + `hit_count_per_ray` + `ray_index` to keep memory tight.
* **Optional color/normal probing**: skip if not needed; expensive VTK probing only when requested.
* **Selectable intersector**: `embree` if installed → 10–100× faster; else fall back to VTK OBBTree.
* **Seeded RNG** per run for reproducibility.
* **Chunk knobs**: `chunk_rays`, `pixels_stride` to downsample camera rays.

---

## 7) Testing Plan

### 7.1 Unit Tests (by module)

**core/intersector**

* *Plane hit*: rays to z=0 plane at known angles → expected distances (analytic).
* *No‑hit*: rays upward → zero hits.
* *Multi‑hit*: box mesh with opposite faces → 2 hits, correct ordering by distance.
* *Backend parity*: VTK vs Embree yield same first‑hit within tolerance.

**sensors/patterns**

* Oscillating: directions span ±FOV/2; count = `pulses_per_line`; unit norm.
* Spinning: vertical angles match input; azimuth progresses with `rpm`; unit norm.
* Raster: az/el grid sizes and serpentine order.

**sensors/noise**

* `jitter_directions` keeps norms≈1; deterministic with seed; angular std≈spec.
* `jitter_ranges` Gaussian stats within tolerance; nonnegative distances (clamped).
* `dropout_mask` expected keep rate within binomial CI.

**attributes**

* Range: equals ‖p−o‖ for simple plane.
* Incidence: 0° at normal incidence on plane; increases with tilt.
* IntensityLambertian: proportional to cosθ/r²; check monotonic changes.
* ReturnNumber: for box scene, first<last; counts match `multi_return`.
* ScanAngle:

  * Oscillating: min/max equal ±FOV/2 within rounding.
  * Spinning: equals vertical channel angles.
* ColorNormalProbe: when mesh has no scalars, returns absent; when present, values within [0,255] and normals unit length.

**exporter**

* Round‑trip: write LAS, re‑open with laspy, verify counts, ranges, dtype, scales, offsets; verify presence of ExtraBytes where configured.

**motion/trajectory**

* Lawnmower footprint: lines alternate direction, spacing within tolerance.
* Static: single pose only.

### 7.2 Integration Tests (end‑to‑end)

* **Tiny scenes** (procedurally generated in tests):

  * Unit plane (10×10 m) with texture.
  * Box (2×2×2 m).
  * Step/ramp for varied incidence.
* **Scenarios**: run CLI for each of the 5 strategies with small configs.

  * Assert point count within expected interval.
  * Assert monotonic `gps_time`.
  * For aerial LiDAR: verify edge‑of‑flight‑line tags near extremes if enabled.
  * For spinning LiDAR: verify distribution of `channel_id`.
* **Reproducibility**: same seed → identical LAS byte‑wise (or equal arrays).
* **Performance sanity**: ensure each scenario under a time budget on CI.

### 7.3 Regression Tests

* Golden outputs for a fixed mesh + config (store small NPZ not LAS) with tolerances (means/std of attributes, density histograms).

---

## 8) Implementation Notes (per component)

### 8.1 Intersector details

* **VTK**: `vtkOBBTree.IntersectWithLine(p0, p1, vtkPoints, vtkIdList)`; we loop rays but keep vtk objects reused to reduce allocs.
* **Embree (optional)**: via `trimesh.ray.ray_pyembree`, batch intersection; supports multi‑hit via repeated march or mesh winding; first‑hit is default.

### 8.2 Time modeling

* For **oscillating**: line time increments by `1/line_rate_hz`; within a line, per‑pulse time increments by `1/(line_rate_hz*pulses_per_line)`.
* For **spinning**: azimuth(t)=ωt; PRF sets pulse spacing; per‑channel firing sequence cycles; compute `revolution_id=floor(azimuth/360°)`.

### 8.3 Attribute recipes

* **Incidence**: `θ = arccos( |dot(-dir, n)| )` using probed/per‑triangle normal.
* **Intensity**: `I ∝ (cosθ)^ρ / r²` (clip and normalize to [0,1]). Allow `albedo` multipliers if a scalar/texture channel is present.
* **Scan angle**:

  * Oscillating: mirror deflection angle (signed), clipped to ±90° for LAS.
  * Spinning: vertical channel angle.
  * Raster: elevation angle.
* **Edge of flight line** (aerial LiDAR): tag rays within top/bottom 5% of scanline index or near lateral strip boundaries (configurable).
* **Beam footprint**: ellipse axes `a≈b≈ 0.5 * divergence_rad * range`. Store as ExtraBytes or two fields.
* **Return numbers**: after sorting hits by distance per ray, set `return_number` and `number_of_returns`.
* **Normals/RGB**: `scene.probe_attributes()` uses `vtkProbeFilter`. If absent, skip.

---

## 9) Example Python Usage (Library)

```python
from smesh.core.scene import MeshScene
from smesh.core.sampler import Sampler, SamplerConfig
from smesh.core.exporter import LasWriter
from smesh.motion.trajectory import LawnmowerTrajectory
from smesh.sensors.patterns import OscillatingMirrorPattern
from smesh.sensors.lidar import LidarSensor
from smesh.sensors.noise import LidarNoise

scene = MeshScene(mesh_path="scene.ply")
traj = LawnmowerTrajectory(scene, altitude_m=120, speed_mps=20, line_spacing_m=150)
pattern = OscillatingMirrorPattern(fov_deg=60, line_rate_hz=80, pulses_per_line=1200)
noise = LidarNoise(sigma_range_m=0.03, sigma_angle_deg=0.03, keep_prob=0.998)
sensor = LidarSensor(trajectory=traj, pattern=pattern, noise=noise,
                     max_range_m=600, multi_return=True)

sampler = Sampler(scene, sensor, SamplerConfig(attributes=["range","incidence","scan_angle","intensity","returns","gps_time","color_normal"]))
writer = LasWriter("out.laz", compress=True)
sampler.run_to_writer(writer)
writer.close()
```

---

## 10) Roadmap (post‑MVP)

* **GPU/OptiX/RTX** intersector backend.
* **Z‑buffer photogrammetry** (EGL offscreen VTK/ModernGL) for dense depth.
* **Motion distortion** (rolling shutter / spinning over time) with pose interpolation (SLERP).
* **IMU/GNSS errors**: boresight/lever‑arm misalignment, drift.
* **Material‑aware intensity** and wavelength selection.
* **Filters**: voxel grid, Poisson disk subsampling, outlier models.
* **Preview tools**: quick 2D maps (density/coverage) and small web viewer export (e.g., Potree).

---

## 11) Appendix A — Attribute Canonical Names & LAS Mapping

| Attribute name          | Type / Units  | LAS field (if any)    | Notes                                    |
| ----------------------- | ------------- | --------------------- | ---------------------------------------- |
| `intensity01`           | float [0..1]  | `intensity` (uint16)  | Scaled ×65535                            |
| `scan_angle_deg`        | float deg     | `scan_angle` (int16)  | Rounded; clipped ±90                     |
| `return_number`         | uint8         | `return_number`       | 1..N                                     |
| `number_of_returns`     | uint8         | `number_of_returns`   | N                                        |
| `gps_time`              | float64 s     | `gps_time`            | Relative epoch (doc in header/user_data) |
| `rgb`                   | uint8[3]      | `red/green/blue`      | 0..255 → 0..65535                        |
| `normal`                | float32[3]    | ExtraBytes(3×float32) | `NormalX/Y/Z`                            |
| `incidence_deg`         | float32       | ExtraBytes(float32)   | Angle w.r.t. surface normal              |
| `range_m`               | float32       | ExtraBytes(float32)   |                                          |
| `beam_footprint_m`      | float32[2]    | ExtraBytes(2×float32) | Major/minor                              |
| `scanline_id`           | uint32        | ExtraBytes(uint32)    |                                          |
| `channel_id`            | uint16        | ExtraBytes(uint16)    | spinning LiDAR                           |
| `revolution_id`         | uint32        | ExtraBytes(uint32)    | spinning LiDAR                           |
| `flightline_id`         | uint32        | ExtraBytes(uint32)    | aerial                                   |
| `pixel_u`, `pixel_v`    | float32       | ExtraBytes            | photogrammetry                           |
| `cam_id`, `exposure_id` | uint16        | ExtraBytes            |                                          |
| `gt`                    | float32/uint8 | ExtraBytes            | your ground‑truth label                  |

---

## 12) Appendix B — Minimal CLI Example

`configs/aerial_lidar.yaml`

```yaml
type: lidar
trajectory: { kind: lawnmower, altitude_m: 120, speed_mps: 20, line_spacing_m: 150 }
pattern:    { kind: oscillating, fov_deg: 60, line_rate_hz: 80, pulses_per_line: 1200 }
sensor:     { max_range_m: 600, multi_return: true, beam_divergence_mrad: 0.3 }
noise:      { sigma_range_m: 0.03, sigma_angle_deg: 0.03, keep_prob: 0.998 }
attributes: [gps_time, scan_angle, returns, intensity, incidence, range, color_normal]
output:     { path: out_aerial_lidar.laz, compress: true }
engine: auto
chunk_rays: 100000
seed: 42
```

Run:

```
smesh sample --mesh scene.ply --config configs/aerial_lidar.yaml
```

---

## 13) Contributing Notes

* **Dependencies**: `vtk`, `laspy`, `numpy`, `PyYAML`, optional `trimesh` + `pyembree`, optional `lazrs` for LAZ. Keep optional deps soft to ease install.
* **Coding style**: type‑annotated, black/isort, mypy‑clean.
* **Docs**: `docs/` built with mkdocs; examples in `examples/`.
* **Plugin registry**: any package can register new `Intersector`, `AttributeComputer`, `Sensor` via entry points: `smesh.plugins`.

---

This plan keeps the public API small and composable, gives you high‑leverage configuration knobs, and sets us up for performance improvements (batching/GPU) without breaking the interface. If you want, I can next draft the concrete `Sampler`, `Intersector` (VTK), and `LasWriter` modules exactly as outlined so you can start coding against the skeleton immediately.
