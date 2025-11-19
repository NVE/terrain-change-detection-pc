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
    source_type: Literal["hoydedata", "drone"] = Field(
        default="hoydedata",
        description="Data source type: 'hoydedata' requires data/ subdirectory, 'drone' does not"
    )
    data_dir_name: str = Field(default="data")
    metadata_dir_name: str = Field(default="metadata")


class CoarseRegistrationConfig(BaseModel):
    enabled: bool = Field(default=True)
    method: Literal["centroid", "pca", "phase", "open3d_fpfh", "none"] = Field(default="pca")
    voxel_size: float = Field(default=2.0, description="Voxel size for downsampling (if applicable)")
    phase_grid_cell: float = Field(default=2.0, description="Grid cell size for phase correlation (meters)")


class AlignmentMultiscaleConfig(BaseModel):
    enabled: bool = Field(
        default=False,
        description="Enable multi-scale ICP refinement (coarse + fine passes)",
    )
    coarse_subsample_size: int = Field(
        default=20000,
        description="Number of points per cloud for coarse ICP pass",
    )
    coarse_max_iterations: int = Field(
        default=30,
        description="Maximum ICP iterations for coarse pass",
    )
    coarse_max_correspondence_distance: Optional[float] = Field(
        default=None,
        description="Max correspondence distance for coarse pass (None = use alignment.max_correspondence_distance)",
    )


class AlignmentICPConfig(BaseModel):
    max_iterations: int = Field(default=100)
    tolerance: float = Field(default=1e-6)
    max_correspondence_distance: float = Field(default=1.0)
    subsample_size: int = Field(default=50000)
    convergence_translation_epsilon: float = Field(
        default=1e-4,
        description="Minimum translation step (meters) to continue ICP iterations",
    )
    convergence_rotation_epsilon_deg: float = Field(
        default=0.1,
        description="Minimum rotation step (degrees) to continue ICP iterations",
    )
    coarse: CoarseRegistrationConfig = Field(default_factory=CoarseRegistrationConfig)
    multiscale: AlignmentMultiscaleConfig = Field(default_factory=AlignmentMultiscaleConfig)


class DetectionDoDConfig(BaseModel):
    enabled: bool = Field(default=True)
    cell_size: float = Field(default=1.0)
    aggregator: Literal["mean", "median", "p95", "p5"] = Field(default="mean")


class DetectionC2CConfig(BaseModel):
    enabled: bool = Field(default=True)
    # Algorithm mode: 'euclidean' uses nearest-neighbor 3D distances;
    # 'vertical_plane' fits a local plane in the target and measures vertical offset.
    mode: Literal["euclidean", "vertical_plane"] = Field(default="euclidean")
    max_points: int = Field(default=10000)
    # For streaming C2C, a finite max_distance is required
    max_distance: Optional[float] = Field(default=None)
    # Local modeling parameters (used when mode='vertical_plane')
    radius: Optional[float] = Field(default=None, description="Search radius (m) for local plane fit")
    k_neighbors: int = Field(default=20, description="If radius is None, use k-NN for local plane fit")
    min_neighbors: int = Field(default=6, description="Minimum neighbors required to fit a plane")


class DetectionM3C2AutotuneConfig(BaseModel):
    # Source for density estimation: 'header' uses LAS headers/union
    # extent; 'sample' uses array points provided to the workflow.
    source: Literal["header", "sample"] = Field(default="header")
    target_neighbors: int = Field(default=16)
    max_depth_factor: float = Field(default=0.6)
    min_radius: float = Field(default=1.0)
    max_radius: float = Field(default=20.0)


class DetectionM3C2EPConfig(BaseModel):
    # If None, the workflow will pick OS-specific defaults
    workers: Optional[int] = Field(default=None)


class DetectionM3C2FixedConfig(BaseModel):
    # When use_autotune is False, use these fixed parameters
    # If normal_scale is None, defaults to radius
    # If depth_factor is None, defaults to autotune.max_depth_factor
    radius: Optional[float] = Field(default=None)
    normal_scale: Optional[float] = Field(default=None)
    depth_factor: Optional[float] = Field(default=None)


class DetectionM3C2Config(BaseModel):
    enabled: bool = Field(default=True)
    core_points: int = Field(default=10000)
    # Choose between autotuned parameters or fixed ones from config
    use_autotune: bool = Field(default=True)
    autotune: DetectionM3C2AutotuneConfig = Field(default_factory=DetectionM3C2AutotuneConfig)
    fixed: DetectionM3C2FixedConfig = Field(default_factory=DetectionM3C2FixedConfig)
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
    class OutOfCoreConfig(BaseModel):
        enabled: bool = Field(default=False, description="Enable out-of-core/streaming processing")
        tile_size_m: float = Field(default=500.0, description="Tile size in meters for tiled processing")
        halo_m: float = Field(default=20.0, description="Halo/buffer width around tiles in meters")
        chunk_points: int = Field(default=1_000_000, description="Number of points per chunk for streaming")
        streaming_mode: bool = Field(default=True, description="Use streaming for preprocessing when enabled")
        save_transformed_files: bool = Field(default=False, description="Save transformed LAZ files during alignment")
        output_dir: Optional[str] = Field(default=None, description="Directory for transformed files (auto if None)")
        memmap_dir: Optional[str] = Field(default=None, description="Directory for memory-mapped arrays in mosaicking (auto if None)")

    class ParallelConfig(BaseModel):
        enabled: bool = Field(default=True, description="Enable CPU parallelization for tile processing")
        n_workers: Optional[int] = Field(default=None, description="Number of worker processes (None = auto-detect: cpu_count - 1)")
        memory_limit_gb: Optional[float] = Field(default=None, description="Soft memory limit in GB to guide concurrency")
        threads_per_worker: Optional[int] = Field(default=1, description="BLAS/NumPy threads per worker process (mitigate oversubscription)")

    class GPUConfig(BaseModel):
        enabled: bool = Field(default=True, description="Enable GPU acceleration if available (graceful CPU fallback)")
        gpu_memory_limit_gb: Optional[float] = Field(default=None, description="Max GPU memory to use in GB (None = auto-detect 80% of available)")
        fallback_to_cpu: bool = Field(default=True, description="Automatically fall back to CPU if GPU fails or unavailable")
        use_for_c2c: bool = Field(default=True, description="Use GPU for C2C nearest neighbor searches")
        use_for_preprocessing: bool = Field(default=True, description="Use GPU for data preprocessing (transformations, filtering)")
        use_for_alignment: bool = Field(default=False, description="Use GPU for ICP alignment when available")
        batch_size: Optional[int] = Field(default=None, description="GPU batch size for operations (None = auto-calculate based on memory)")

    paths: PathsConfig = Field(default_factory=PathsConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    alignment: AlignmentICPConfig = Field(default_factory=AlignmentICPConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    outofcore: OutOfCoreConfig = Field(default_factory=OutOfCoreConfig)
    parallel: ParallelConfig = Field(default_factory=ParallelConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)


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
