"""
Configuration management for terrain-change-detection.

Provides a typed pydantic model and YAML loader with sensible defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal, List, Any, Dict

from pydantic import BaseModel, Field, ValidationError
import yaml


# -----------------------
# Typed config structures
# -----------------------


class PathsConfig(BaseModel):
    base_dir: str = Field(default="data/raw")


class PreprocessingConfig(BaseModel):
    ground_only: bool = Field(default=True)
    classification_filter: List[int] = Field(default_factory=lambda: [2])


class DiscoveryConfig(BaseModel):
    data_dir_name: str = Field(default="data")
    metadata_dir_name: str = Field(default="metadata")


class CoarseRegistrationConfig(BaseModel):
    enabled: bool = Field(default=True)
    method: Literal["centroid", "pca", "phase", "open3d_fpfh", "none"] = Field(default="pca")
    voxel_size: float = Field(default=2.0, description="Voxel size for downsampling (if applicable)")
    phase_grid_cell: float = Field(default=2.0, description="Grid cell size for phase correlation (meters)")


class AlignmentICPConfig(BaseModel):
    max_iterations: int = Field(default=100)
    tolerance: float = Field(default=1e-6)
    max_correspondence_distance: float = Field(default=1.0)
    subsample_size: int = Field(default=50000)
    coarse: CoarseRegistrationConfig = Field(default_factory=CoarseRegistrationConfig)


class DetectionDoDConfig(BaseModel):
    cell_size: float = Field(default=1.0)
    aggregator: Literal["mean", "median", "p95", "p5"] = Field(default="mean")


class DetectionC2CConfig(BaseModel):
    max_points: int = Field(default=10000)
    max_distance: Optional[float] = Field(default=None)


class DetectionM3C2AutotuneConfig(BaseModel):
    target_neighbors: int = Field(default=16)
    max_depth_factor: float = Field(default=0.6)
    min_radius: float = Field(default=1.0)
    max_radius: float = Field(default=20.0)


class DetectionM3C2EPConfig(BaseModel):
    # If None, the workflow will pick OS-specific defaults
    workers: Optional[int] = Field(default=None)


class DetectionM3C2Config(BaseModel):
    core_points: int = Field(default=10000)
    autotune: DetectionM3C2AutotuneConfig = Field(default_factory=DetectionM3C2AutotuneConfig)
    ep: DetectionM3C2EPConfig = Field(default_factory=DetectionM3C2EPConfig)


class DetectionConfig(BaseModel):
    dod: DetectionDoDConfig = Field(default_factory=DetectionDoDConfig)
    c2c: DetectionC2CConfig = Field(default_factory=DetectionC2CConfig)
    m3c2: DetectionM3C2Config = Field(default_factory=DetectionM3C2Config)


class VisualizationConfig(BaseModel):
    backend: Literal["plotly", "pyvista", "pyvistaqt"] = Field(default="plotly")
    sample_size: int = Field(default=50000)


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    file: Optional[str] = Field(default=None)


class PerformanceConfig(BaseModel):
    numpy_threads: Literal["auto"] | int = Field(default="auto")


class AppConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    alignment: AlignmentICPConfig = Field(default_factory=AlignmentICPConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)


# -----------------------
# Loader
# -----------------------


def _project_root() -> Path:
    """
    Resolve the repository root directory.

    File is at: repo_root/src/terrain_change_detection/utils/config.py
    parents sequence:
      0 -> .../src/terrain_change_detection/utils
      1 -> .../src/terrain_change_detection
      2 -> .../src
      3 -> repo_root   <-- correct root
    """
    return Path(__file__).resolve().parents[3]


def load_config(path: Optional[str | Path] = None, *, allow_missing: bool = True) -> AppConfig:
    """
    Load configuration from YAML into a typed AppConfig.

    Search order when path is None:
    1) repo_root/config/default.yaml
    2) if missing and allow_missing=True: return default AppConfig()

    Args:
        path: Explicit YAML file path.
        allow_missing: If True, returns defaults when file missing; otherwise raises.

    Returns:
        AppConfig instance
    """
    cfg_path: Path
    if path is None:
        cfg_path = _project_root() / "config" / "default.yaml"
    else:
        cfg_path = Path(path)

    if not cfg_path.exists():
        if allow_missing:
            return AppConfig()
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    try:
        return AppConfig.model_validate(raw)
    except ValidationError as e:
        # Re-raise with context to help users fix the YAML
        raise ValueError(f"Invalid configuration in {cfg_path}: {e}")
