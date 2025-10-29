from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Optional, Union, List

import yaml
from pydantic import BaseModel, Field, model_validator


class MeshConfig(BaseModel):
    path: Path


class StaticTrajectoryConfig(BaseModel):
    kind: Literal["static"]
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    start_time_s: float = 0.0


class LawnmowerTrajectoryConfig(BaseModel):
    kind: Literal["lawnmower"]
    altitude_m: float
    altitude_mode: Literal["above_top", "above_ground", "absolute"] = "above_top"
    speed_mps: float
    line_spacing_m: float
    heading_deg: float = 0.0
    start_time_s: float = 0.0


TrajectoryConfig = Annotated[
    Union[StaticTrajectoryConfig, LawnmowerTrajectoryConfig],
    Field(discriminator="kind"),
]


class OscillatingPatternConfig(BaseModel):
    kind: Literal["oscillating"]
    fov_deg: float
    line_rate_hz: float
    pulses_per_line: int


class SpinningPatternConfig(BaseModel):
    kind: Literal["spinning"]
    vertical_angles_deg: List[float]
    rpm: float
    prf_hz: int


class RasterPatternConfig(BaseModel):
    kind: Literal["raster"]
    az_range_deg: tuple[float, float]
    el_range_deg: tuple[float, float]
    step_deg: float


class CameraPatternConfig(BaseModel):
    kind: Literal["camera"]
    resolution_px: tuple[int, int]
    focal_px: tuple[float, float]
    principal_px: Optional[tuple[float, float]] = None
    pixel_stride: int = 1


PatternConfig = Annotated[
    Union[OscillatingPatternConfig, SpinningPatternConfig, RasterPatternConfig, CameraPatternConfig],
    Field(discriminator="kind"),
]


class LidarNoiseConfig(BaseModel):
    kind: Literal["lidar"] = "lidar"
    sigma_range_m: float = 0.0
    sigma_angle_deg: float = 0.0
    keep_prob: float = 1.0
    sigma_time_s: float = 0.0


class PhotogrammetryNoiseConfig(BaseModel):
    kind: Literal["photogrammetry"] = "photogrammetry"
    pixel_sigma: float = 0.0
    sigma_time_s: float = 0.0


class LidarSensorConfig(BaseModel):
    kind: Literal["lidar"] = "lidar"
    max_range_m: float = 1000.0
    multi_return: bool = True
    beam_divergence_mrad: float = 0.3
    boresight_rpy_deg: Optional[tuple[float, float, float]] = None
    lever_arm_m: Optional[tuple[float, float, float]] = None
    num_lines: Optional[int] = None


class TotalStationSensorConfig(BaseModel):
    kind: Literal["totalstation"] = "totalstation"
    max_range_m: float = 1000.0
    boresight_rpy_deg: Optional[tuple[float, float, float]] = None
    lever_arm_m: Optional[tuple[float, float, float]] = None


class CameraSensorConfig(BaseModel):
    kind: Literal["camera"] = "camera"
    frame_rate_hz: Optional[float] = None
    exposure_interval_s: Optional[float] = None
    num_frames: Optional[int] = None
    boresight_rpy_deg: Optional[tuple[float, float, float]] = None
    lever_arm_m: Optional[tuple[float, float, float]] = None


class SamplerConfigModel(BaseModel):
    intersector: str = "auto"
    batch_size_rays: int = 100_000
    beam_divergence_mrad: Optional[float] = None


NoiseConfig = Annotated[
    Union[LidarNoiseConfig, PhotogrammetryNoiseConfig],
    Field(discriminator="kind"),
]


SensorConfig = Annotated[
    Union[LidarSensorConfig, TotalStationSensorConfig, CameraSensorConfig],
    Field(discriminator="kind"),
]


ScenarioType = Literal["lidar", "totalstation", "photogrammetry"]


class OutputConfig(BaseModel):
    path: Path
    format: Literal["las", "laz", "npz", "ply"] = "las"
    compress: Optional[bool] = None
    point_format: int = 8

    @model_validator(mode="after")
    def _validate_format(self) -> "OutputConfig":
        if self.format == "laz" and self.compress is False:
            raise ValueError("format 'laz' implies compress=True")
        return self


class ScenarioConfig(BaseModel):
    type: ScenarioType = "lidar"
    mesh: MeshConfig
    trajectory: TrajectoryConfig
    pattern: Optional[PatternConfig] = None
    sensor: SensorConfig
    noise: Optional[NoiseConfig] = None
    sampler: SamplerConfigModel = SamplerConfigModel()
    attributes: List[str] = Field(default_factory=list)
    output: OutputConfig
    engine: str = "auto"
    seed: Optional[int] = None

    @model_validator(mode="after")
    def _ensure_defaults(self) -> "ScenarioConfig":
        if self.type == "lidar":
            if not isinstance(self.sensor, LidarSensorConfig):
                raise ValueError("Lidar scenario requires lidar sensor config")
            if self.pattern is None or not isinstance(self.pattern, (OscillatingPatternConfig, SpinningPatternConfig, RasterPatternConfig)):
                raise ValueError("Lidar scenario requires oscillating, spinning, or raster pattern")
            if not self.attributes:
                self.attributes = [
                    "range",
                    "incidence",
                    "scan_angle",
                    "intensity",
                    "returns",
                    "gps_time",
                ]
        elif self.type == "totalstation":
            if not isinstance(self.sensor, TotalStationSensorConfig):
                raise ValueError("Total station scenario requires totalstation sensor config")
            if self.pattern is None or not isinstance(self.pattern, RasterPatternConfig):
                raise ValueError("Total station scenario requires raster pattern")
            if not self.attributes:
                self.attributes = ["range", "incidence", "scan_angle"]
        elif self.type == "photogrammetry":
            if not isinstance(self.sensor, CameraSensorConfig):
                raise ValueError("Photogrammetry scenario requires camera sensor config")
            if self.pattern is None or not isinstance(self.pattern, CameraPatternConfig):
                raise ValueError("Photogrammetry scenario requires camera pattern")
            if not self.attributes:
                self.attributes = ["range", "incidence", "color_normal"]
        else:
            raise ValueError(f"Unsupported scenario type '{self.type}'")
        return self


def load_config(path: str | Path) -> ScenarioConfig:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping.")
    cfg = ScenarioConfig.model_validate(data)
    cfg.output.path = (path.parent / cfg.output.path).resolve()
    if cfg.mesh.path and not cfg.mesh.path.is_absolute():
        cfg.mesh.path = (path.parent / cfg.mesh.path).resolve()
    return cfg
