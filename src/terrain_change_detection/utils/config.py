"""
Configuration management for terrain-change-detection.

Provides a typed pydantic model and YAML loader with sensible defaults.
"""

from __future__ import annotations

from copy import deepcopy
from difflib import get_close_matches
from pathlib import Path
from typing import Optional, Literal, List, Any, Dict, Sequence, get_args, get_origin

from pydantic import BaseModel, Field, TypeAdapter, ValidationError
import yaml


# -----------------------
# Typed config structures
# -----------------------


class PathsConfig(BaseModel):
    base_dir: str = Field(default="data/raw")
    output_dir: Optional[str] = Field(
        default=None,
        description="Base directory for all output files (auto-generate if None)",
    )
    output_crs: str = Field(
        default="EPSG:25833",
        description="CRS for outputs (auto-detect from LAZ if possible, fallback to this)",
    )


class PreprocessingConfig(BaseModel):
    ground_only: bool = Field(default=True)
    classification_filter: List[int] = Field(default_factory=lambda: [2])


class CoordinateConfig(BaseModel):
    """Configuration for local coordinate transformation."""

    use_local_coordinates: bool = Field(
        default=True,
        description="Transform to local coordinates for numerical stability during processing",
    )
    origin_method: Literal["min_bounds", "centroid", "first_point"] = Field(
        default="min_bounds",
        description="Method for determining local origin: 'min_bounds' guarantees positive coords",
    )
    include_z_offset: bool = Field(
        default=False,
        description="Also offset Z coordinates (usually not needed for terrain data)",
    )


class DiscoveryConfig(BaseModel):
    source_type: Literal["hoydedata", "drone"] = Field(
        default="hoydedata",
        description="Data source type: 'hoydedata' requires data/ subdirectory, 'drone' does not",
    )
    data_dir_name: str = Field(default="data")
    metadata_dir_name: str = Field(default="metadata")


class ClippingConfig(BaseModel):
    """Configuration for area clipping to focus on regions of interest."""

    enabled: bool = Field(
        default=False,
        description="Enable clipping point clouds to a region of interest before processing",
    )
    boundary_file: Optional[str] = Field(
        default=None,
        description="Path to GeoJSON or Shapefile defining the clipping boundary",
    )
    feature_name: Optional[str] = Field(
        default=None,
        description="Name of specific feature to use from the boundary file (if multiple features exist)",
    )
    save_clipped_files: bool = Field(
        default=False, description="Save clipped LAZ files to disk for reuse"
    )
    output_dir: Optional[str] = Field(
        default=None, description="Directory to save clipped files (auto if None)"
    )


class CoarseRegistrationConfig(BaseModel):
    enabled: bool = Field(default=False)
    method: Literal["centroid", "pca", "phase", "open3d_fpfh", "none"] = Field(
        default="centroid"
    )
    voxel_size: float = Field(
        default=2.0, description="Voxel size for downsampling (if applicable)"
    )
    phase_grid_cell: float = Field(
        default=2.0, description="Grid cell size for phase correlation (meters)"
    )


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
    enabled: bool = Field(
        default=True, description="Enable ICP fine registration for spatial alignment"
    )
    max_iterations: int = Field(default=100)
    tolerance: float = Field(default=1e-6)
    max_correspondence_distance: float = Field(default=1.0)
    subsample_size: int = Field(default=50000)
    subsample_mode: Literal["count", "percent"] = Field(
        default="count",
        description="How subsample size is specified: 'count' uses subsample_size, 'percent' uses subsample_percent",
    )
    subsample_percent: float = Field(
        default=10.0,
        description="Percentage of points to subsample when subsample_mode is 'percent'",
    )
    max_subsample_size: int = Field(
        default=500_000,
        description="Safety cap: maximum number of subsampled points regardless of mode",
    )
    convergence_translation_epsilon: float = Field(
        default=1e-4,
        description="Minimum translation step (meters) to continue ICP iterations",
    )
    convergence_rotation_epsilon_deg: float = Field(
        default=0.1,
        description="Minimum rotation step (degrees) to continue ICP iterations",
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducible subsampling. Use different values for independent runs.",
    )
    overlap_filter: bool = Field(
        default=True,
        description="Filter points to bounding-box overlap region before subsampling for ICP",
    )
    overlap_margin_m: float = Field(
        default=5.0,
        description="Margin in meters to expand overlap bounding box (accounts for alignment shifts)",
    )
    reference: Literal["t1", "t2"] = Field(
        default="t1",
        description="Which time period to use as the ICP reference. 't1' = earlier epoch, 't2' = later epoch.",
    )
    icp_backend: Literal["custom", "open3d"] = Field(
        default="custom",
        description="ICP implementation: 'custom' (built-in SVD-based) or 'open3d' (Open3D library)",
    )
    coarse: CoarseRegistrationConfig = Field(default_factory=CoarseRegistrationConfig)
    multiscale: AlignmentMultiscaleConfig = Field(
        default_factory=AlignmentMultiscaleConfig
    )
    export_aligned_pc: bool = Field(
        default=False, description="Export aligned point cloud as LAZ file"
    )


class DetectionDoDConfig(BaseModel):
    enabled: bool = Field(default=False)
    cell_size: float = Field(default=1.0)
    aggregator: Literal["mean", "median", "p95", "p5"] = Field(default="mean")
    export_raster: bool = Field(
        default=False, description="Export DoD as GeoTIFF raster"
    )


class DetectionC2CConfig(BaseModel):
    enabled: bool = Field(default=False)
    # Algorithm mode: 'euclidean' uses nearest-neighbor 3D distances;
    # 'vertical_plane' fits a local plane in the target and measures vertical offset.
    mode: Literal["euclidean", "vertical_plane"] = Field(default="euclidean")
    max_points: int = Field(default=9_000_000)
    # For streaming C2C, a finite max_distance is required
    max_distance: Optional[float] = Field(default=10.0)
    # Local modeling parameters (used when mode='vertical_plane')
    radius: Optional[float] = Field(
        default=None, description="Search radius (m) for local plane fit"
    )
    k_neighbors: int = Field(
        default=20, description="If radius is None, use k-NN for local plane fit"
    )
    min_neighbors: int = Field(
        default=6, description="Minimum neighbors required to fit a plane"
    )
    export_pc: bool = Field(
        default=False, description="Export C2C distances as LAZ point cloud"
    )
    export_raster: bool = Field(
        default=False, description="Export C2C as interpolated GeoTIFF raster"
    )


class DetectionM3C2AutotuneConfig(BaseModel):
    # Source for density estimation: 'header' uses LAS headers/union
    # extent; 'sample' uses array points provided to the workflow.
    source: Literal["header", "sample"] = Field(default="header")
    target_neighbors: int = Field(default=16)
    max_depth_factor: float = Field(default=1.0)
    min_radius: float = Field(default=1.0)
    max_radius: float = Field(default=20.0)


class DetectionM3C2FixedConfig(BaseModel):
    # When use_autotune is False, use these fixed parameters
    # If normal_scale is None, defaults to radius
    # If depth_factor is None, defaults to autotune.max_depth_factor
    radius: Optional[float] = Field(default=1.0)
    normal_scale: Optional[float] = Field(default=1.0)
    depth_factor: Optional[float] = Field(default=2.0)


class DetectionM3C2Config(BaseModel):
    enabled: bool = Field(default=True)
    core_points_percent: Optional[float] = Field(
        default=100.0,
        description="Percentage of reference ground points to use as M3C2 core points (e.g., 10.0 = 10%)",
    )
    core_points: Optional[int] = Field(
        default=None,
        description="(Deprecated) Absolute number of core points. If set, overrides core_points_percent.",
    )
    # Choose between autotuned parameters or fixed ones from config
    use_autotune: bool = Field(default=False)
    autotune: DetectionM3C2AutotuneConfig = Field(
        default_factory=DetectionM3C2AutotuneConfig
    )
    fixed: DetectionM3C2FixedConfig = Field(default_factory=DetectionM3C2FixedConfig)

    export_pc: bool = Field(
        default=True,
        description="Export M3C2 core points with distances as LAZ point cloud",
    )
    export_raster: bool = Field(
        default=True, description="Export M3C2 distances as interpolated GeoTIFF raster"
    )


class DetectionConfig(BaseModel):
    dod: DetectionDoDConfig = Field(default_factory=DetectionDoDConfig)
    c2c: DetectionC2CConfig = Field(default_factory=DetectionC2CConfig)
    m3c2: DetectionM3C2Config = Field(default_factory=DetectionM3C2Config)


class VisualizationConfig(BaseModel):
    backend: Literal["plotly", "pyvista", "pyvistaqt"] = Field(default="plotly")
    sample_size: int = Field(default=100000)


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )
    file: Optional[str] = Field(default=None)


class PerformanceConfig(BaseModel):
    numpy_threads: Literal["auto"] | int = Field(default="auto")


class AppConfig(BaseModel):
    class OutOfCoreConfig(BaseModel):
        enabled: bool = Field(
            default=False, description="Enable out-of-core/streaming processing"
        )
        tile_size_m: float = Field(
            default=500.0, description="Tile size in meters for tiled processing"
        )
        halo_m: float = Field(
            default=20.0, description="Halo/buffer width around tiles in meters"
        )
        chunk_points: int = Field(
            default=1_000_000, description="Number of points per chunk for streaming"
        )
        streaming_mode: bool = Field(
            default=True, description="Use streaming for preprocessing when enabled"
        )
        save_transformed_files: bool = Field(
            default=False, description="Save transformed LAZ files during alignment"
        )
        output_dir: Optional[str] = Field(
            default=None, description="Directory for transformed files (auto if None)"
        )
        memmap_dir: Optional[str] = Field(
            default=None,
            description="Directory for memory-mapped arrays in mosaicking (auto if None)",
        )

    class ParallelConfig(BaseModel):
        enabled: bool = Field(
            default=False, description="Enable CPU parallelization for tile processing"
        )
        n_workers: Optional[int] = Field(
            default=None,
            description="Number of worker processes (None = auto-detect: cpu_count - 1)",
        )
        memory_limit_gb: Optional[float] = Field(
            default=None, description="Soft memory limit in GB to guide concurrency"
        )
        threads_per_worker: Optional[int] = Field(
            default=1,
            description="BLAS/NumPy threads per worker process (mitigate oversubscription)",
        )

    class GPUConfig(BaseModel):
        enabled: bool = Field(
            default=False,
            description="Enable GPU acceleration if available (graceful CPU fallback)",
        )
        gpu_memory_limit_gb: Optional[float] = Field(
            default=None,
            description="Max GPU memory to use in GB (None = auto-detect 80% of available)",
        )
        fallback_to_cpu: bool = Field(
            default=True,
            description="Automatically fall back to CPU if GPU fails or unavailable",
        )
        use_for_c2c: bool = Field(
            default=True, description="Use GPU for C2C nearest neighbor searches"
        )
        use_for_dod: bool = Field(
            default=True, description="Use GPU for DoD grid accumulation operations"
        )
        use_for_preprocessing: bool = Field(
            default=True,
            description="Use GPU for data preprocessing (transformations, filtering)",
        )
        use_for_alignment: bool = Field(
            default=False, description="Use GPU for ICP alignment when available"
        )
        batch_size: Optional[int] = Field(
            default=None,
            description="GPU batch size for operations (None = auto-calculate based on memory)",
        )

    paths: PathsConfig = Field(default_factory=PathsConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    coordinates: CoordinateConfig = Field(default_factory=CoordinateConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    clipping: ClippingConfig = Field(default_factory=ClippingConfig)
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


def _default_config_path() -> Path:
    """Return the repository's canonical default config path."""
    return _project_root() / "config" / "default.yaml"


def _resolve_config_path(path: str | Path) -> Path:
    """Resolve config paths relative to the cwd first, then the repo root."""
    cfg_path = Path(path)
    if cfg_path.is_absolute() or cfg_path.exists():
        return cfg_path

    repo_relative = _project_root() / cfg_path
    if repo_relative.exists():
        return repo_relative

    return cfg_path


def _load_yaml_dict(path: Path, *, allow_missing: bool = True) -> Dict[str, Any]:
    """Load a YAML file and require a mapping at the document root."""
    if not path.exists():
        if allow_missing:
            return {}
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Invalid configuration in {path}: expected a YAML mapping at the top level."
        )
    return raw


def deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge nested dictionaries without mutating the inputs."""
    merged = deepcopy(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _annotation_model_class(annotation: Any) -> type[BaseModel] | None:
    """Return the nested BaseModel class for an annotation, if any."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation

    origin = get_origin(annotation)
    if origin is None:
        return None

    for arg in get_args(annotation):
        nested = _annotation_model_class(arg)
        if nested is not None:
            return nested
    return None


def parse_dot_notation(
    key: str,
    value: str,
    model_cls: type[BaseModel] = AppConfig,
) -> tuple[list[str], Any]:
    """
    Validate a dot-path config key and coerce its value to the target field type.

    Values are parsed with YAML semantics first, then validated with Pydantic so
    booleans, numbers, ``null``, lists, and literals behave naturally.
    """
    path = [segment.strip() for segment in key.split(".") if segment.strip()]
    if not path:
        raise ValueError("Invalid override: config key cannot be empty.")

    current_model = model_cls
    target_annotation: Any = None

    for idx, segment in enumerate(path):
        available = sorted(current_model.model_fields)
        if segment not in current_model.model_fields:
            matches = get_close_matches(segment, available, n=3)
            scope = ".".join(path[:idx]) or current_model.__name__
            suggestion = f" Did you mean: {', '.join(matches)}?" if matches else ""
            raise ValueError(
                f"Unknown config key '{segment}' in override '{key}' under '{scope}'.{suggestion}"
            )

        field_info = current_model.model_fields[segment]
        target_annotation = field_info.annotation

        if idx == len(path) - 1:
            break

        nested_model = _annotation_model_class(target_annotation)
        if nested_model is None:
            prefix = ".".join(path[: idx + 1])
            raise ValueError(
                f"Invalid override '{key}': '{prefix}' is a scalar field and cannot have nested keys."
            )
        current_model = nested_model

    raw_value: Any
    if value == "":
        raw_value = ""
    else:
        try:
            raw_value = yaml.safe_load(value)
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Invalid YAML value for override '{key}={value}': {exc}"
            ) from exc

    try:
        parsed_value = TypeAdapter(target_annotation).validate_python(raw_value)
    except ValidationError as exc:
        raise ValueError(f"Invalid value for override '{key}={value}': {exc}") from exc

    return path, parsed_value


def apply_overrides(config: AppConfig, overrides: Sequence[str]) -> AppConfig:
    """Apply ``key=value`` dot-path overrides to an existing config object."""
    merged = config.model_dump()
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid override '{override}': expected the form 'section.key=value'."
            )

        key, value = override.split("=", 1)
        path, parsed_value = parse_dot_notation(key, value, AppConfig)

        patch: Dict[str, Any] = {}
        cursor = patch
        for segment in path[:-1]:
            cursor[segment] = {}
            cursor = cursor[segment]
        cursor[path[-1]] = parsed_value

        merged = deep_merge(merged, patch)

    try:
        return AppConfig.model_validate(merged)
    except ValidationError as exc:
        raise ValueError(
            f"Invalid configuration after applying CLI overrides: {exc}"
        ) from exc


def load_config(
    path: Optional[str | Path] = None,
    *,
    config_paths: Optional[Sequence[str | Path]] = None,
    overrides: Optional[Sequence[str]] = None,
    allow_missing: bool = True,
) -> AppConfig:
    """
    Load configuration into a typed ``AppConfig`` using layered sources.

    Precedence order:
    1) ``AppConfig`` schema defaults
    2) repo_root/config/default.yaml
    3) any explicit override YAML files, in the order provided
    4) CLI ``key=value`` overrides

    Args:
        path: Backward-compatible singular override YAML path.
        config_paths: Additional override YAMLs layered after ``path``.
        overrides: Dot-path ``key=value`` overrides.
        allow_missing: If True, missing config files are ignored; otherwise raises.

    Returns:
        Fully validated ``AppConfig`` instance.
    """
    merged: Dict[str, Any] = AppConfig().model_dump()
    config_sources: list[Path] = []

    default_path = _default_config_path()
    if default_path.exists():
        merged = deep_merge(merged, _load_yaml_dict(default_path, allow_missing=False))
        config_sources.append(default_path)
    elif not allow_missing:
        raise FileNotFoundError(f"Config file not found: {default_path}")

    layered_paths: list[str | Path] = []
    if path is not None:
        layered_paths.append(path)
    if config_paths:
        layered_paths.extend(config_paths)

    for override_path in layered_paths:
        resolved_path = _resolve_config_path(override_path)
        raw = _load_yaml_dict(resolved_path, allow_missing=allow_missing)
        if raw:
            merged = deep_merge(merged, raw)
        config_sources.append(resolved_path)

    if overrides:
        config = apply_overrides(AppConfig.model_validate(merged), overrides)
        return config

    try:
        return AppConfig.model_validate(merged)
    except ValidationError as e:
        joined_sources = (
            ", ".join(str(path) for path in config_sources) or "schema defaults"
        )
        raise ValueError(
            f"Invalid configuration after merging {joined_sources}: {e}"
        ) from e
