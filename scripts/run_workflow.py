"""
Example script for complete terrain change detection workflow

This script demonstrates the full workflow from data discovery to change detection.
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path
from time import perf_counter

import numpy as np

# Add the src to the path to import modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from terrain_change_detection.acceleration import LaspyStreamReader
from terrain_change_detection.alignment import (
    apply_transform_to_files,
    save_transform_matrix,
    compute_overlap_mask,
    estimate_alignment_covariance,
)
from terrain_change_detection.alignment.coarse_registration import CoarseRegistration
from terrain_change_detection.alignment.fine_registration import ICPRegistration
from terrain_change_detection.detection import (
    ChangeDetector,
    M3C2Detector,
    M3C2Params,
    autotune_m3c2_params,
    autotune_m3c2_params_from_headers,
)
from terrain_change_detection.preprocessing.clipping import (
    AreaClipper,
    check_shapely_available,
)
from terrain_change_detection.preprocessing.data_discovery import (
    BatchLoader,
    DataDiscovery,
)
from terrain_change_detection.preprocessing.loader import PointCloudLoader
from terrain_change_detection.utils.config import AppConfig, load_config
from terrain_change_detection.utils.coordinate_transform import LocalCoordinateTransform
from terrain_change_detection.utils.export import (
    detect_crs_from_laz,
    export_distances_to_geotiff,
    export_dod_to_geotiff,
    export_points_to_laz,
)
from terrain_change_detection.utils.logging import setup_logger
from terrain_change_detection.visualization.point_cloud import PointCloudVisualizer

# Hardware optimizations
# TO DO: Implement hardware optimizations for large datasets

# Tuning knobs (now configured via YAML):
# - alignment.subsample_size: subsample size for ICP alignment
# - detection.m3c2.core_points_percent: percentage of reference ground points for M3C2 core points
# - detection.m3c2.core_points: (optional) absolute number - overrides percentage if set
# - detection.c2c.max_points: maximum points per cloud for C2C distances
# - visualization.sample_size: sample size for visualization

logging.getLogger("terrain_change_detection.preprocessing.data_discovery").setLevel(logging.ERROR)
logging.getLogger("terrain_change_detection.preprocessing.loader").setLevel(logging.ERROR)


def resolve_subsample_count(total_points: int, cfg) -> int:
    """Compute the effective subsample count from config.

    Supports both absolute-count and percentage modes, with a safety cap.

    Args:
        total_points: Total number of points in the cloud.
        cfg: ``AlignmentICPConfig`` (or similar object with the required fields).

    Returns:
        Number of points to subsample (capped by ``max_subsample_size``).
    """
    if getattr(cfg, "subsample_mode", "count") == "percent":
        n = int(total_points * cfg.subsample_percent / 100.0)
    else:
        n = cfg.subsample_size
    cap = getattr(cfg, "max_subsample_size", 500_000)
    return min(n, cap)


def build_cli_overrides(args: argparse.Namespace) -> list[str]:
    """Translate dedicated CLI flags into config-style dot-path overrides."""
    overrides = list(args.set_overrides or [])

    if args.base_dir:
        overrides.append(f"paths.base_dir={args.base_dir}")
    if args.seed is not None:
        overrides.append(f"alignment.random_seed={args.seed}")
    if args.reference is not None:
        overrides.append(f"alignment.reference={args.reference}")

    if args.m3c2_radius is not None:
        overrides.extend([
            "detection.m3c2.use_autotune=false",
            f"detection.m3c2.fixed.radius={args.m3c2_radius}",
        ])
        if args.m3c2_normal_scale is None:
            overrides.append("detection.m3c2.fixed.normal_scale=null")
        if args.m3c2_depth_factor is None:
            overrides.append("detection.m3c2.fixed.depth_factor=null")

    if args.m3c2_normal_scale is not None:
        overrides.extend([
            "detection.m3c2.use_autotune=false",
            f"detection.m3c2.fixed.normal_scale={args.m3c2_normal_scale}",
        ])
    if args.m3c2_depth_factor is not None:
        overrides.extend([
            "detection.m3c2.use_autotune=false",
            f"detection.m3c2.fixed.depth_factor={args.m3c2_depth_factor}",
        ])

    return overrides


def write_run_inputs(
    base_dir: Path,
    args: argparse.Namespace,
    cfg: AppConfig,
    *,
    config_files: list[str] | None = None,
    cli_overrides: list[str] | None = None,
):
    """
    Write the run inputs (CLI args and config) to a text file for record-keeping.
    """
    from datetime import datetime

    import yaml
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    output_path = base_dir / "logs" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("=== Command Line Arguments ===\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

        f.write("\n=== Configuration Sources ===\n")
        f.write("base_config: config/default.yaml\n")
        if config_files:
            for idx, config_file in enumerate(config_files, start=1):
                f.write(f"override_file_{idx}: {config_file}\n")
        else:
            f.write("override_files: none\n")
        if cli_overrides:
            for idx, override in enumerate(cli_overrides, start=1):
                f.write(f"cli_override_{idx}: {override}\n")
        else:
            f.write("cli_overrides: none\n")

        f.write("\n=== Configuration ===\n")

        cfg_yaml = yaml.safe_dump(cfg.model_dump(), sort_keys=False)
        f.write(cfg_yaml)


def _slice_attributes(attributes: dict, mask: np.ndarray) -> dict:
    """Apply a point mask to all attribute arrays in a dataset."""
    clipped = {}
    for name, values in attributes.items():
        arr = np.asarray(values)
        if len(arr) != len(mask):
            logger.warning(
                "Skipping attribute %s during clipping because its length (%d) does not match mask (%d)",
                name,
                len(arr),
                len(mask),
            )
            continue
        clipped[name] = arr[mask]
    return clipped


def _load_alignment_covariance_matrix(path: str) -> np.ndarray:
    """Load a persisted 12x12 covariance matrix from .npy, .txt, or .csv."""
    cov_path = Path(path)
    if not cov_path.exists():
        raise FileNotFoundError(f"Alignment covariance file not found: {cov_path}")

    if cov_path.suffix.lower() == ".npy":
        cxx = np.load(cov_path)
    else:
        delimiter = "," if cov_path.suffix.lower() == ".csv" else None
        cxx = np.loadtxt(cov_path, dtype=np.float64, delimiter=delimiter)

    cxx = np.asarray(cxx, dtype=np.float64)
    if cxx.shape != (12, 12):
        raise ValueError(f"Alignment covariance must be shape (12, 12), got {cxx.shape}")
    return cxx



def main():
    """
    Main function to run the terrain change detection workflow.
    """
    runtime_start = perf_counter()

    # CLI: allow overriding the data root and config path
    parser = argparse.ArgumentParser(description="Terrain Change Detection Workflow")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory containing area folders (e.g., data/raw or data/synthetic)",
    )
    parser.add_argument(
        "--config",
        type=str,
        action="append",
        default=None,
        help=(
            "Repeatable override YAML layered on top of config/default.yaml. "
            "Example: --config config/profiles/large_scale.yaml"
        ),
    )
    parser.add_argument(
        "--set",
        "-s",
        action="append",
        dest="set_overrides",
        default=None,
        help=(
            "Override any config value with KEY=VALUE dot notation. "
            "Example: --set discovery.source_type=drone"
        ),
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print the resolved configuration YAML and exit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for NumPy RNG to make subsampling/core selection reproducible.",
    )
    parser.add_argument(
        "--cores-file",
        type=str,
        default=None,
        help="Path to a .npy file with core points to LOAD if it exists; otherwise SAVE selected cores to this path.",
    )
    parser.add_argument(
        "--m3c2-radius",
        type=float,
        default=None,
        help="Override M3C2 radius (meters). Sets projection_scale=cylinder_radius=radius for both modes.",
    )
    parser.add_argument(
        "--m3c2-normal-scale",
        type=float,
        default=None,
        help="Override M3C2 normal_scale (meters). Defaults to radius when not set.",
    )
    parser.add_argument(
        "--m3c2-depth-factor",
        type=float,
        default=None,
        help="Override max_depth factor so that max_depth = depth_factor * radius (default from config).",
    )
    parser.add_argument(
        "--debug-m3c2-compare",
        action="store_true",
        help="Run both streaming and in-memory M3C2 on the same core points and print sign/correlation diagnostics.",
    )

    parser.add_argument(
        "--area-name",
        type=str,
        default=None,
        help="Specify the area name to process.",
    )

    parser.add_argument(
        "--show-plots",
        type=bool,
        default=False,
        help="If True, show plots interactively instead of saving to files.",
    )

    parser.add_argument(
        "--years",
        type=int,
        nargs='+',  # Acepta uno o más valores
        default=None,
        help="List of years to select for processing (e.g., --years 2020 2021).",
    )

    parser.add_argument(
        "--save-dems",
        type=bool,
        default=False,
        help="If True, generate DEMs after ICP and saves them to disk.",
    )

    parser.add_argument(
        "--reference",
        choices=["t1", "t2"],
        default=None,
        help="Which epoch is the ICP reference (t1=earlier, t2=later). Overrides config.",
    )

    args = parser.parse_args()

    if (args.m3c2_normal_scale is not None or args.m3c2_depth_factor is not None) and args.m3c2_radius is None:
        parser.error("--m3c2-normal-scale and --m3c2-depth-factor require --m3c2-radius")

    cli_overrides = build_cli_overrides(args)

    # Load configuration
    cfg: AppConfig = load_config(
        config_paths=args.config,
        overrides=cli_overrides,
        allow_missing=False,
    )

    if args.show_config:
        import yaml

        print(yaml.safe_dump(cfg.model_dump(), sort_keys=False), end="")
        return

    # Setup logging from config
    log_level = getattr(logging, cfg.logging.level.upper(), logging.INFO)
    logger = setup_logger(__name__, level=log_level, log_file=cfg.logging.file)

    show_plots: bool = args.show_plots
    selected_years: list[int] | None = args.years
    export_dem_rasters: bool = args.save_dems

    logger.info(f"selected_years: {selected_years}")

    # Deterministic RNG for reproducible subsampling
    # CLI --seed overrides config; config defaults to 42 for determinism
    _seed = cfg.alignment.random_seed
    rng = np.random.default_rng(int(_seed))
    logger.info(f"NumPy RNG seeded with {int(_seed)}")

    # Performance: set thread env vars if configured
    try:
        threads = cfg.performance.numpy_threads
        if threads == "auto":
            threads = os.cpu_count() or 1
        if isinstance(threads, int) and threads > 0:
            os.environ["OMP_NUM_THREADS"] = str(threads)
            os.environ["MKL_NUM_THREADS"] = str(threads)
            os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    except Exception:
        pass

    logger.info("Terrain Change Detection Workflow")
    logger.info("=================================")
    
    # Log GPU configuration status and check for GPU libraries
    try:
        import platform

        from terrain_change_detection.acceleration.hardware_detection import detect_gpu
        
        if getattr(cfg.gpu, 'enabled', False):
            # Check if GPU libraries are available
            # cuML is Linux-only (RAPIDS), but CuPy works on Windows too
            cupy_available = False
            cuml_available = False
            
            try:
                import cupy as cp
                cupy_available = True
            except ImportError:
                pass
            
            try:
                import cuml
                cuml_available = True
            except ImportError:
                pass
            
            is_windows = platform.system() == "Windows"
            
            # On Windows: CuPy is sufficient (cuML not available)
            # On Linux: Prefer cuML but CuPy-only is still useful
            if not cupy_available:
                # No GPU library available at all
                logger.error("=" * 80)
                logger.error("ERROR: GPU is enabled in config but CuPy is not available!")
                logger.error("")
                if is_windows:
                    logger.error("On Windows, install CuPy for GPU acceleration:")
                    logger.error("  uv add cupy-cuda12x  # or cupy-cuda11x depending on your CUDA version")
                else:
                    logger.error("To use GPU acceleration, you must activate the GPU environment:")
                    logger.error("  source activate_gpu.sh")
                logger.error("")
                logger.error("Or disable GPU in your config file:")
                logger.error("  gpu:")
                logger.error("    enabled: false")
                logger.error("=" * 80)
                logger.error("Exiting workflow. Please fix the configuration and try again.")
                return
            
            # Log GPU capability level
            if cuml_available:
                gpu_mode = "FULL (CuPy + cuML)"
            else:
                gpu_mode = "PARTIAL (CuPy only - cuML not available)"
                if not is_windows:
                    logger.warning("cuML not available. For full GPU acceleration on Linux, activate GPU environment:")
                    logger.warning("  source activate_gpu.sh")
            
            gpu_info = detect_gpu()
            if gpu_info.available:
                logger.info(f"GPU Acceleration: ENABLED - {gpu_mode}")
                logger.info(f"  Device: {gpu_info.device_name}")
                logger.info(f"  Memory: {gpu_info.memory_gb:.2f} GB")
                logger.info(f"  C2C: {'ENABLED' if getattr(cfg.gpu, 'use_for_c2c', False) else 'DISABLED'}")
                logger.info(f"  DoD: {'ENABLED' if getattr(cfg.gpu, 'use_for_dod', False) else 'DISABLED'}")
                logger.info(f"  Alignment: {'ENABLED' if getattr(cfg.gpu, 'use_for_alignment', False) else 'DISABLED'}")
                
                # Check for GPU + parallel processing incompatibility
                if getattr(cfg.parallel, 'enabled', False):
                    logger.warning("=" * 80)
                    logger.warning("WARNING: GPU and parallel processing are both enabled!")
                    logger.warning("CUDA contexts cannot survive process forking (multiprocessing limitation).")
                    logger.warning("This may cause 'CUDARuntimeError: cudaErrorInitializationError' in workers.")
                    logger.warning("Recommendation: Disable either GPU or parallel processing.")
                    logger.warning("  - To disable GPU: set gpu.enabled=false in config")
                    logger.warning("  - To disable parallel: set parallel.enabled=false in config")
                    logger.warning("=" * 80)
            else:
                logger.warning(f"GPU Acceleration: ENABLED in config but GPU not available ({gpu_info.error_message}), will use CPU fallback")
        else:
            logger.info("GPU Acceleration: DISABLED (CPU only)")
    except Exception as e:
        logger.debug(f"Could not check GPU status: {e}")

    base_dir = Path(cfg.paths.base_dir)
    # base_dir = Path(__file__).parent.parent / "tests" / "test_preprocessing" / "sample_data" / "raw"
    if not base_dir.exists():
        logger.error(f"Base directory {base_dir} does not exist.")
        return

    # ============================================================
    # Step 1: Data Preparation (Discovery & Streaming Setup)
    # ============================================================
    # Configure preprocessing and discovery according to config
    loader = PointCloudLoader(
        ground_only=cfg.preprocessing.ground_only,
        classification_filter=cfg.preprocessing.classification_filter,
    )
    data_discovery = DataDiscovery(
        base_dir,
        source_type=cfg.discovery.source_type,
        data_dir_name=cfg.discovery.data_dir_name,
        metadata_dir_name=cfg.discovery.metadata_dir_name,
        loader=loader,
    )
    areas = data_discovery.scan_areas(user_area_name=args.area_name)

    if not areas:
        logger.error("No area directories found in the base directory.")
        logger.error(f"Data source type: {cfg.discovery.source_type}")
        if cfg.discovery.source_type == 'hoydedata':
            logger.error(f"Expected structure for hoydedata: {base_dir}/<area>/<time_period>/{cfg.discovery.data_dir_name}/*.laz")
            logger.error("If your data doesn't have a 'data' subdirectory, set source_type: drone in config")
        else:
            logger.error(f"Expected structure for drone: {base_dir}/<area>/<time_period>/*.laz")
            logger.error("If your data has a 'data' subdirectory, set source_type: hoydedata in config")
        return

    if args.area_name:
        # Select specified area if it exists
        if args.area_name in areas:
            selected_area = areas[args.area_name]
        else:
            logger.error(f"Specified area '{args.area_name}' not found in base directory.")
            logger.error(f"Available areas: {list(areas.keys())}")
            return
    else:
        # Find the first area with at least two time periods
        selected_area = None
        for area_name, area_info in areas.items():
            if len(area_info.time_periods) >= 2:
                selected_area = area_info
                break

    if not selected_area:
        # Provide detailed feedback about what was found
        logger.error("Could not find an area with at least two time periods for change detection.")
        logger.error(f"Found {len(areas)} area(s):")
        for area_name, area_info in areas.items():
            logger.error(f"  - {area_name}: {len(area_info.time_periods)} time period(s) -> {area_info.time_periods}")
        logger.error("Change detection requires at least 2 time periods per area.")
        logger.error("Please organize your data with multiple time periods per area or add more data.")
        return

    
    # If selected_years is provided, filter time periods accordingly
    if selected_years is not None and len(selected_years) > 1:
        if len(selected_years) > 2:
            logger.warning("More than two selected years provided; only the first two will be used.")

        filtered_time_periods = [
            tp for tp in selected_area.time_periods
            if any(str(year) in tp for year in selected_years)
        ]
        
        if len(filtered_time_periods) < 2:
            logger.error(f"After filtering, less than two time periods remain for area '{selected_area.area_name}'.")
            logger.error(f"Filtered time periods: {filtered_time_periods}")
            return

        logger.info(f"Filtered time periods for area '{selected_area.area_name}': {filtered_time_periods}")

        t1, t2 = filtered_time_periods[:2]

    else:
    # Select the first two time periods
        t1, t2 = selected_area.time_periods[:2]
        logger.info(f"Selected time periods for area '{selected_area.area_name}': {t1}, {t2}")

    ds1 = selected_area.datasets[t1]
    ds2 = selected_area.datasets[t2]

    logger.info("=== STEP 1: Data Preparation ===")
    logger.info(f"Selected area: {selected_area.area_name}")
    logger.info(f"Time period 1: {t1} ({len(ds1.laz_files)} files)")
    logger.info(f"Time period 2: {t2} ({len(ds2.laz_files)} files)")

    # Determine if we should use streaming/out-of-core mode
    use_streaming = (
        getattr(cfg, 'outofcore', None) is not None
        and cfg.outofcore.enabled
        and cfg.outofcore.streaming_mode
        and len(ds1.laz_files) > 0
        and len(ds2.laz_files) > 0
    )

    # ============================================================
    # Local Coordinate Transform Setup (if enabled)
    # ============================================================
    # Compute local coordinate transform from T1 bounds to handle large UTM coordinates
    local_transform = None
    coord_cfg = getattr(cfg, 'coordinates', None)
    use_local_coords = coord_cfg is not None and getattr(coord_cfg, 'use_local_coordinates', True)
    
    if use_local_coords and ds1.bounds:
        # Compute transform from T1 (reference) bounds
        origin_method = getattr(coord_cfg, 'origin_method', 'min_bounds')
        include_z = getattr(coord_cfg, 'include_z_offset', False)
        
        if origin_method == 'min_bounds':
            offset_z = ds1.bounds.get('min_z', 0.0) if include_z else 0.0
            local_transform = LocalCoordinateTransform.from_bounds(
                min_x=ds1.bounds['min_x'],
                min_y=ds1.bounds['min_y'],
                min_z=offset_z,
            )
        elif origin_method == 'centroid':
            # Compute centroid from bounds
            cx = (ds1.bounds['min_x'] + ds1.bounds['max_x']) / 2
            cy = (ds1.bounds['min_y'] + ds1.bounds['max_y']) / 2
            cz = ((ds1.bounds['min_z'] + ds1.bounds['max_z']) / 2) if include_z else 0.0
            local_transform = LocalCoordinateTransform(offset_x=cx, offset_y=cy, offset_z=cz)
        else:
            # first_point not practical here, fall back to min_bounds
            offset_z = ds1.bounds.get('min_z', 0.0) if include_z else 0.0
            local_transform = LocalCoordinateTransform.from_bounds(
                min_x=ds1.bounds['min_x'],
                min_y=ds1.bounds['min_y'],
                min_z=offset_z,
            )
        
        logger.info("Local coordinate transform: offset=(%.2f, %.2f, %.2f) using %s origin",
                    local_transform.offset_x, local_transform.offset_y, local_transform.offset_z,
                    origin_method)

    try:
        # No pooled executor reuse; each phase manages its own pool
        # Step 1: Load Data (or prepare for streaming)
        if use_streaming:
            logger.info("--- Step 1: Preparing datasets for streaming/out-of-core processing ---")
            batch_loader = BatchLoader(loader=loader, streaming_mode=True)
            
            # Get file paths and metadata without loading full datasets
            pc1_data = batch_loader.load_dataset(ds1, streaming=True)
            pc2_data = batch_loader.load_dataset(ds2, streaming=True)
            
            # Log both ground and total points for clarity (derived from headers + streamed class counts)
            m1 = pc1_data['metadata']
            m2 = pc2_data['metadata']
            logger.info(
                "Dataset 1 (%s): %d files, ~%.0f ground / %.0f total (%.1f%%)",
                t1,
                len(pc1_data['file_paths']),
                float(m1.get('total_points_ground') or 0),
                float(m1.get('total_points_all') or 0),
                (float(m1.get('ground_percentage')) if m1.get('ground_percentage') is not None else float('nan')),
            )
            logger.info(
                "Dataset 2 (%s): %d files, ~%.0f ground / %.0f total (%.1f%%)",
                t2,
                len(pc2_data['file_paths']),
                float(m2.get('total_points_ground') or 0),
                float(m2.get('total_points_all') or 0),
                (float(m2.get('ground_percentage')) if m2.get('ground_percentage') is not None else float('nan')),
            )
            
            # Load samples for alignment (streaming-based reservoir sampling)
            # Estimate total ground points for percentage-based subsampling
            est_ground1 = int(m1.get('total_points_ground') or m1.get('total_points_all') or 0)
            est_ground2 = int(m2.get('total_points_ground') or m2.get('total_points_all') or 0)
            n_per_ds1 = resolve_subsample_count(max(est_ground1, 1), cfg.alignment)
            n_per_ds2 = resolve_subsample_count(max(est_ground2, 1), cfg.alignment)
            logger.info("Loading subsampled data for alignment (T1→%d, T2→%d)...", n_per_ds1, n_per_ds2)

            # Overlap-aware bounding box for streaming subsampling
            overlap_bbox = None
            if cfg.alignment.overlap_filter:
                from terrain_change_detection.acceleration.tiling import intersection_bounds
                overlap_bbox = intersection_bounds(
                    [str(p) for p in ds1.laz_files],
                    [str(p) for p in ds2.laz_files],
                    margin=cfg.alignment.overlap_margin_m,
                )
                if overlap_bbox is not None:
                    logger.info(
                        "Streaming overlap bbox: x=[%.1f, %.1f], y=[%.1f, %.1f]",
                        overlap_bbox.min_x, overlap_bbox.max_x,
                        overlap_bbox.min_y, overlap_bbox.max_y,
                    )
                else:
                    logger.warning("No overlap found between datasets; sampling full extents.")

            # T1 alignment subsample
            reader1 = LaspyStreamReader(
                [str(p) for p in ds1.laz_files],
                ground_only=cfg.preprocessing.ground_only,
                classification_filter=cfg.preprocessing.classification_filter,
                chunk_points=cfg.outofcore.chunk_points,
            )
            points1 = reader1.reservoir_sample(
                n_per_ds1, transform=local_transform, bbox=overlap_bbox,
                seed=int(_seed),
            )

            # T2 alignment subsample (use a derived seed for independence)
            reader2 = LaspyStreamReader(
                [str(p) for p in ds2.laz_files],
                ground_only=cfg.preprocessing.ground_only,
                classification_filter=cfg.preprocessing.classification_filter,
                chunk_points=cfg.outofcore.chunk_points,
            )
            points2 = reader2.reservoir_sample(
                n_per_ds2, transform=local_transform, bbox=overlap_bbox,
                seed=int(_seed) + 1,
            )

            logger.info(f"Loaded {len(points1)} sample points from T1 for alignment")
            logger.info(f"Loaded {len(points2)} sample points from T2 for alignment")
        else:
            logger.info("--- Step 1: Loading point cloud data (in-memory) ---")
            batch_loader = BatchLoader(loader=loader)
            if len(ds1.laz_files) > 1:
                logger.info(f"Batch loading {len(ds1.laz_files)} files for time period {t1}...")
                pc1_data = batch_loader.load_dataset(ds1, transform=local_transform)
            else:
                logger.info(f"Loading single file for time period {t1}...")
                pc1_data = batch_loader.loader.load(str(ds1.laz_files[0]), transform=local_transform)

            if len(ds2.laz_files) > 1:
                logger.info(f"Batch loading {len(ds2.laz_files)} files for time period {t2}...")
                pc2_data = batch_loader.load_dataset(ds2, transform=local_transform)
            else:
                logger.info(f"Loading single file for time period {t2}...")
                pc2_data = batch_loader.loader.load(str(ds2.laz_files[0]), transform=local_transform)

            logger.info(f"Dataset 1 ({t1}): {pc1_data['points'].shape[0]} points")
            logger.info(f"Dataset 2 ({t2}): {pc2_data['points'].shape[0]} points")

            points1 = pc1_data['points']
            points2 = pc2_data['points']

        # ============================================================
        # Optional: Area Clipping (before alignment)
        # ============================================================
        # Check if clipping is enabled in config
        clipping_cfg = getattr(cfg, 'clipping', None)
        clip_bounds = None  # Will store (minx, miny, maxx, maxy) if clipping is enabled
        
        if clipping_cfg is not None and clipping_cfg.enabled:
            if not check_shapely_available():
                logger.error("Clipping is enabled but shapely is not installed. Install with: uv add shapely")
                return
            
            if not clipping_cfg.boundary_file:
                logger.error("Clipping is enabled but no boundary_file is specified in config.")
                return
            
            boundary_path = Path(clipping_cfg.boundary_file)
            if not boundary_path.is_absolute():
                # Resolve relative to project root
                boundary_path = Path(__file__).parent.parent / boundary_path
            
            if not boundary_path.exists():
                logger.error(f"Clipping boundary file not found: {boundary_path}")
                return
            
            logger.info("--- Applying area clipping ---")
            
            try:
                # Load clipper with optional feature name filter
                clipper = AreaClipper.from_file(
                    str(boundary_path), 
                    feature_name=clipping_cfg.feature_name
                )
                
                # Transform clipper to local coordinates if local_transform is enabled
                # Points are loaded in local coordinates, so the clipping polygon must also be in local coordinates
                if local_transform is not None:
                    clipper = clipper.transform_to_local(local_transform)
                
                # Store clip bounds for streaming processing (DoD, C2C)
                clip_bounds = clipper.bounds
                
                # Store original counts
                original_count_1 = len(points1)
                original_count_2 = len(points2)

                # Clip both point clouds. In in-memory mode we also preserve per-point
                # attributes such as point_source_id for downstream M3C2-EP metadata.
                points1, mask1 = clipper.clip(points1, return_mask=True)
                points2, mask2 = clipper.clip(points2, return_mask=True)
                if not use_streaming:
                    pc1_data['points'] = points1
                    pc2_data['points'] = points2
                    pc1_data['attributes'] = _slice_attributes(pc1_data.get('attributes', {}), mask1)
                    pc2_data['attributes'] = _slice_attributes(pc2_data.get('attributes', {}), mask2)
                
                # Summary log
                pct1 = 100.0 * len(points1) / original_count_1 if original_count_1 > 0 else 0
                pct2 = 100.0 * len(points2) / original_count_2 if original_count_2 > 0 else 0
                logger.info(
                    f"Clipping complete: T1 {len(points1):,} pts ({pct1:.1f}%), "
                    f"T2 {len(points2):,} pts ({pct2:.1f}%)"
                )
                
                if len(points1) == 0 or len(points2) == 0:
                    logger.error("Clipping resulted in empty point clouds. Check your boundary file.")
                    return
                    
            except Exception as e:
                logger.error(f"Clipping failed: {e}")
                return

        # Instantiate the visualizer (choose backend)
        VIS_BACKEND = cfg.visualization.backend
        visualizer = PointCloudVisualizer(backend=VIS_BACKEND)

        vis_points1 = local_transform.to_global(points1) if local_transform else points1
        vis_points2 = local_transform.to_global(points2) if local_transform else points2
        if show_plots:
            # Visualize the original point clouds
            # Revert to global coordinates for visualization (users expect UTM coordinates)
            logger.info("--- Visualizing original point clouds ---")
            visualizer.visualize_clouds(
                point_clouds=[vis_points1, vis_points2],
                names=[f"PC from {t1}", f"PC from {t2}"],
                sample_size=cfg.visualization.sample_size  # Downsample for visualization
            )

        if export_dem_rasters:
            #TODO: Export DEM rasters before alignment
            logger.info("--- Exporting DEM rasters before alignment ---")
            try:
                
                
                for yy_i, pc_i in zip([t1, t2], [vis_points1, vis_points2]):
                    
                    crs = cfg.paths.output_crs
                    try:
                        detected_crs = detect_crs_from_laz(str(ds1.laz_files[0]))
                        if detected_crs:
                            crs = detected_crs
                    except Exception:
                        pass

                    if cfg.paths.output_dir:
                        export_dir = Path(cfg.paths.output_dir)
                    else:
                        export_dir = Path(cfg.paths.base_dir) / "output" / selected_area.area_name

                    dem_tif = export_dir / f"dem_{yy_i}_raw.tif"

                    try:
                        export_distances_to_geotiff(
                            pc_i[:, :2], pc_i[:, 2], str(dem_tif),
                            cell_size=cfg.detection.dod.cell_size, crs=crs,
                            local_transform=None
                        )
                        logger.info(f"Exported Dem raster: {dem_tif}")
                    except Exception as export_e:
                        logger.error(f"Failed to export raw DEM raster for point cloud {yy_i}: {export_e}")
                        continue
            except Exception as e:
                logger.error(f"Failed to export DEM rasters before alignment: {e}")


        # ============================================================
        # Step 2: Spatial Alignment
        # ============================================================
        # Check if alignment is enabled (default: True for backward compatibility)
        alignment_enabled = getattr(cfg.alignment, 'enabled', True)
        alignment_cxx = None

        if alignment_enabled:
            logger.info("=== STEP 2: Spatial Alignment ===")
            step2_start = time.time()

            # Optional coarse registration to initialize ICP
            initial_transform = None
            try:
                if getattr(cfg.alignment, "coarse", None) and cfg.alignment.coarse.enabled:
                    coarse = CoarseRegistration(
                        method=cfg.alignment.coarse.method,
                        voxel_size=cfg.alignment.coarse.voxel_size,
                        phase_grid_cell=cfg.alignment.coarse.phase_grid_cell,
                    )
                    initial_transform = coarse.compute_initial_transform(points2, points1)
                    # Optional pre-ICP error report without mutating points2
                    try:
                        points2_init = coarse.apply_transformation(points2, initial_transform)
                        tmp_icp = ICPRegistration(
                            max_iterations=1,
                            tolerance=cfg.alignment.tolerance,
                            max_correspondence_distance=cfg.alignment.max_correspondence_distance,
                            use_gpu=(cfg.gpu.enabled and cfg.gpu.use_for_alignment),
                            convergence_translation_epsilon=cfg.alignment.convergence_translation_epsilon,
                            convergence_rotation_epsilon_deg=cfg.alignment.convergence_rotation_epsilon_deg,
                        )
                        pre_err = tmp_icp.compute_registration_error(points2_init, points1)
                        logger.info("Alignment validation (pre-ICP): RMSE=%.6f m", pre_err)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Coarse registration failed: {e}")

            # Optional multi-scale ICP: coarse refinement followed by fine ICP
            transform_matrix = initial_transform if initial_transform is not None else np.eye(4)

            if getattr(cfg.alignment, "multiscale", None) and cfg.alignment.multiscale.enabled:
                logger.info("Running multi-scale ICP refinement...")

                # Coarse subsampling (deterministic via rng)
                n_coarse = cfg.alignment.multiscale.coarse_subsample_size
                n1c = min(len(points1), n_coarse)
                n2c = min(len(points2), n_coarse)
                idx1c = rng.choice(len(points1), n1c, replace=False) if len(points1) > n1c else np.arange(len(points1))
                idx2c = rng.choice(len(points2), n2c, replace=False) if len(points2) > n2c else np.arange(len(points2))
                points1_coarse = points1[idx1c]
                points2_coarse = points2[idx2c]

                coarse_max_corr = (
                    cfg.alignment.multiscale.coarse_max_correspondence_distance
                    if cfg.alignment.multiscale.coarse_max_correspondence_distance is not None
                    else cfg.alignment.max_correspondence_distance
                )

                icp_coarse = ICPRegistration(
                    max_iterations=cfg.alignment.multiscale.coarse_max_iterations,
                    tolerance=cfg.alignment.tolerance,
                    max_correspondence_distance=coarse_max_corr,
                    convergence_translation_epsilon=cfg.alignment.convergence_translation_epsilon,
                    convergence_rotation_epsilon_deg=cfg.alignment.convergence_rotation_epsilon_deg,
                )

                # Measure RMSE on the subsampled pair before the refinement pass so we can
                # discard a multi-scale refinement step that makes things worse.
                try:
                    points2_coarse_init = icp_coarse.apply_transformation(points2_coarse, transform_matrix)
                    pre_coarse_err = icp_coarse.compute_registration_error(
                        source=points2_coarse_init,
                        target=points1_coarse,
                    )
                except Exception:
                    pre_coarse_err = None

                _, T_coarse, coarse_err = icp_coarse.align_point_clouds(
                    source=points2_coarse,
                    target=points1_coarse,
                    initial_transform=transform_matrix,
                )

                if pre_coarse_err is not None and coarse_err > pre_coarse_err:
                    logger.info(
                        "Multi-scale refinement unchanged (no improvement): RMSE %.6f m → %.6f m",
                        pre_coarse_err, coarse_err
                    )
                else:
                    transform_matrix = T_coarse
                    if pre_coarse_err is not None:
                        logger.info("Multi-scale refinement improved: RMSE %.6f m → %.6f m", pre_coarse_err, coarse_err)
                    else:
                        logger.info("Multi-scale refinement completed: RMSE=%.6f m", coarse_err)

            # --- ICP backend selection (Issue 6) ---
            if cfg.alignment.icp_backend == "open3d":
                try:
                    from terrain_change_detection.alignment.open3d_icp import Open3DICP
                    icp = Open3DICP(
                        max_iterations=cfg.alignment.max_iterations,
                        tolerance=cfg.alignment.tolerance,
                        max_correspondence_distance=cfg.alignment.max_correspondence_distance,
                        convergence_translation_epsilon=cfg.alignment.convergence_translation_epsilon,
                        convergence_rotation_epsilon_deg=cfg.alignment.convergence_rotation_epsilon_deg,
                    )
                    logger.info("Using Open3D ICP backend")
                except ImportError:
                    logger.warning("Open3D not available; falling back to custom ICP backend")
                    icp = ICPRegistration(
                        max_iterations=cfg.alignment.max_iterations,
                        tolerance=cfg.alignment.tolerance,
                        max_correspondence_distance=cfg.alignment.max_correspondence_distance,
                        use_gpu=(cfg.gpu.enabled and cfg.gpu.use_for_alignment),
                        convergence_translation_epsilon=cfg.alignment.convergence_translation_epsilon,
                        convergence_rotation_epsilon_deg=cfg.alignment.convergence_rotation_epsilon_deg,
                    )
            else:
                icp = ICPRegistration(
                    max_iterations=cfg.alignment.max_iterations,
                    tolerance=cfg.alignment.tolerance,
                    max_correspondence_distance=cfg.alignment.max_correspondence_distance,
                    use_gpu=(cfg.gpu.enabled and cfg.gpu.use_for_alignment),
                    convergence_translation_epsilon=cfg.alignment.convergence_translation_epsilon,
                    convergence_rotation_epsilon_deg=cfg.alignment.convergence_rotation_epsilon_deg,
                )

            # --- Overlap filtering (Issue 5) ---
            points1_for_icp = points1
            points2_for_icp = points2
            if cfg.alignment.overlap_filter and not use_streaming:
                mask1, mask2 = compute_overlap_mask(
                    points1, points2, margin=cfg.alignment.overlap_margin_m,
                )
                n1_overlap = int(mask1.sum())
                n2_overlap = int(mask2.sum())
                if n1_overlap >= 100 and n2_overlap >= 100:
                    points1_for_icp = points1[mask1]
                    points2_for_icp = points2[mask2]
                    logger.info(
                        "Overlap filter: T1 %d/%d, T2 %d/%d points in overlap region",
                        n1_overlap, len(points1), n2_overlap, len(points2),
                    )
                else:
                    logger.warning(
                        "Overlap filter: too few points in overlap (%d, %d); using full clouds",
                        n1_overlap, n2_overlap,
                    )

            # --- Subsample for fine alignment (Issues 3 & 4) ---
            n1_target = resolve_subsample_count(len(points1_for_icp), cfg.alignment)
            if len(points1_for_icp) > n1_target:
                indices1 = rng.choice(len(points1_for_icp), n1_target, replace=False)
                points1_subsampled = points1_for_icp[indices1]
            else:
                points1_subsampled = points1_for_icp

            n2_target = resolve_subsample_count(len(points2_for_icp), cfg.alignment)
            if len(points2_for_icp) > n2_target:
                indices2 = rng.choice(len(points2_for_icp), n2_target, replace=False)
                points2_subsampled = points2_for_icp[indices2]
            else:
                points2_subsampled = points2_for_icp

            # --- Reference / target selection (Issue 2) ---
            if cfg.alignment.reference == "t2":
                icp_source = points1_subsampled
                icp_target = points2_subsampled
                icp_source_full = points1
                logger.info("ICP direction: aligning T1 (%s) to T2 (%s) reference", t1, t2)
            else:
                icp_source = points2_subsampled
                icp_target = points1_subsampled
                icp_source_full = points2
                logger.info("ICP direction: aligning T2 (%s) to T1 (%s) reference", t2, t1)

            # Perform ICP alignment
            _, transform_matrix, final_error = icp.align_point_clouds(
                source=icp_source,
                target=icp_target,
                initial_transform=transform_matrix,
            )

            # Apply the transformation to the correct full point cloud
            source_full_aligned = icp.apply_transformation(icp_source_full, transform_matrix)

            # Assign aligned results for downstream use
            if cfg.alignment.reference == "t2":
                points1_full_aligned = source_full_aligned
                points2_full_aligned = points2
                aligned_epoch = t1
            else:
                points1_full_aligned = points1
                points2_full_aligned = source_full_aligned
                aligned_epoch = t2

            # Compute the registration error (RMSE) on a potentially downsampled subset of full data
            src_err = source_full_aligned
            tgt_err = icp_target  # use the reference subsample for validation
            max_err_points = 200_000
            if len(src_err) > max_err_points:
                idx_s_err = rng.choice(len(src_err), max_err_points, replace=False)
                src_err = src_err[idx_s_err]
            if len(tgt_err) > max_err_points:
                idx_t_err = rng.choice(len(tgt_err), max_err_points, replace=False)
                tgt_err = tgt_err[idx_t_err]

            alignment_error = icp.compute_registration_error(
                source=src_err,
                target=tgt_err,
            )

            # Log validation error
            logger.info("Alignment validation (post-ICP): RMSE=%.6f m", alignment_error)

            # --- Export aligned point cloud (Issue 1) ---
            if cfg.alignment.export_aligned_pc:
                crs = cfg.paths.output_crs
                try:
                    detected_crs = detect_crs_from_laz(str(ds1.laz_files[0]))
                    if detected_crs:
                        crs = detected_crs
                except Exception:
                    pass

                if cfg.paths.output_dir:
                    export_dir = Path(cfg.paths.output_dir)
                else:
                    export_dir = Path(cfg.paths.base_dir) / "output" / selected_area.area_name
                export_dir.mkdir(parents=True, exist_ok=True)

                aligned_pc_path = export_dir / f"aligned_{aligned_epoch}.laz"
                export_points = (
                    local_transform.to_global(source_full_aligned) if local_transform else source_full_aligned
                )
                source_laz = ds1.laz_files[0] if cfg.alignment.reference == "t2" else ds2.laz_files[0]
                export_points_to_laz(
                    export_points, None, str(aligned_pc_path),
                    crs=crs, source_laz_path=str(source_laz),
                )
                logger.info("Aligned point cloud exported to: %s", aligned_pc_path)

            # If streaming mode, optionally apply transform to original files
            if use_streaming and cfg.outofcore.save_transformed_files:
                logger.info("--- Applying transformation to full datasets (streaming) ---")

                # Determine which files to transform based on reference direction
                if cfg.alignment.reference == "t2":
                    files_to_transform = pc1_data['file_paths']
                    aligned_label = f"{t1}_aligned"
                else:
                    files_to_transform = pc2_data['file_paths']
                    aligned_label = f"{t2}_aligned"

                # Determine output directory
                if cfg.outofcore.output_dir:
                    output_dir = Path(cfg.outofcore.output_dir) / selected_area.area_name / aligned_label
                else:
                    output_dir = Path(cfg.paths.base_dir).parent / "processed" / selected_area.area_name / aligned_label

                try:
                    aligned_files = apply_transform_to_files(
                        input_files=files_to_transform,
                        output_dir=str(output_dir),
                        transform=transform_matrix,
                        ground_only=cfg.preprocessing.ground_only,
                        classification_filter=cfg.preprocessing.classification_filter,
                        chunk_points=cfg.outofcore.chunk_points,
                    )
                    # Store aligned file paths for later use
                    if cfg.alignment.reference == "t2":
                        pc1_data['aligned_file_paths'] = aligned_files
                    else:
                        pc2_data['aligned_file_paths'] = aligned_files

                    # Save transformation matrix for reference
                    transform_file = output_dir / "transformation_matrix.txt"
                    save_transform_matrix(transform_matrix, str(transform_file))

                    logger.info(f"Transformed {len(aligned_files)} files saved to {output_dir}")
                except Exception as e:
                    logger.error(f"Failed to apply transformation to files: {e}")
                    logger.info("Falling back to in-memory aligned points for DoD")

            step2_end = time.time()
            logger.info("Spatial alignment completed in %.2f seconds", step2_end - step2_start)

            # Revert to global coordinates for visualization (users expect UTM coordinates)
            vis_points1_aligned = local_transform.to_global(points1_full_aligned) if local_transform else points1_full_aligned
            vis_points2_aligned = local_transform.to_global(points2_full_aligned) if local_transform else points2_full_aligned
            if show_plots:
                # Visualize the aligned point clouds
                logger.info("--- Visualizing aligned point clouds ---")
                visualizer.visualize_clouds(
                    point_clouds=[vis_points1_aligned, vis_points2_aligned],
                    names=[f"PC from {t1} (Target)", f"PC from {t2} (Aligned)"],
                    sample_size=cfg.visualization.sample_size  # Downsample for visualization
                )
            
            if export_dem_rasters:
                #TODO: Export DEM rasters after alignment
                logger.info("--- Exporting DEM rasters after alignment ---")
                try:
                    
                    
                    for yy_i, pc_i in zip([t1, t2], [vis_points1_aligned, vis_points2_aligned]):
                        
                        crs = cfg.paths.output_crs
                        try:
                            detected_crs = detect_crs_from_laz(str(ds1.laz_files[0]))
                            if detected_crs:
                                crs = detected_crs
                        except Exception:
                            pass

                        if cfg.paths.output_dir:
                            export_dir = Path(cfg.paths.output_dir)
                        else:
                            export_dir = Path(cfg.paths.base_dir) / "output" / selected_area.area_name

                        dem_tif = export_dir / f"dem_{yy_i}_icp.tif"
                        try:
                            export_distances_to_geotiff(
                                pc_i[:, :2], pc_i[:, 2], str(dem_tif),
                                cell_size=cfg.detection.dod.cell_size, crs=crs,
                                local_transform=None
                            )
                            logger.info(f"Exported Dem raster: {dem_tif}")
                        except Exception as export_e:
                            logger.error(f"Failed to export DEM raster for {yy_i}: {export_e}")
                            continue
                except Exception as e:
                    logger.error(f"Failed to export DEM rasters after alignment: {e}")


        else:
            # Alignment disabled - skip ICP and use original point clouds
            logger.info("=== STEP 2: Spatial Alignment (SKIPPED) ===")
            logger.info("ICP alignment disabled in config; using original point clouds.")
            transform_matrix = np.eye(4)
            points1_full_aligned = points1
            points2_full_aligned = points2
            alignment_error = None

        # Common aligned views for downstream in-memory analysis
        points1_analysis = points1_full_aligned
        points2_analysis = points2_full_aligned

        # ============================================================
        # Step 3: Change Detection
        # ============================================================
        logger.info("=== STEP 3: Change Detection ===")

        # 3a) DEM of Difference (DoD)
        if getattr(cfg.detection.dod, "enabled", True):
            try:
                logger.info("Computing DEM of Difference (DoD)...")

                # Determine which DoD method to use
                can_use_streaming = (
                    use_streaming
                    and cfg.detection.dod.aggregator == 'mean'
                    and 'file_paths' in pc1_data
                )

                if can_use_streaming:
                    # Use original file paths for T1, transformed file paths for T2 if available
                    files_t1 = pc1_data['file_paths']

                    # Use aligned files if they were created, otherwise fall back to original
                    if 'aligned_file_paths' in pc2_data and pc2_data['aligned_file_paths']:
                        files_t2 = pc2_data['aligned_file_paths']
                        logger.info(f"Using pre-transformed files for T2: {len(files_t2)} files")
                    else:
                        files_t2 = pc2_data['file_paths']
                        logger.info("Using original T2 files with on-the-fly alignment transform")

                    mode = "parallel" if cfg.parallel.enabled else "sequential"
                    logger.info(f"Using streaming DoD ({mode}, tiled)...")

                    try:
                        # Choose parallel or sequential based on config
                        if cfg.parallel.enabled:
                            dod_res = ChangeDetector.compute_dod_streaming_files_tiled_parallel(
                                files_t1=files_t1,
                                files_t2=files_t2,
                                cell_size=cfg.detection.dod.cell_size,
                                tile_size=cfg.outofcore.tile_size_m,
                                halo=cfg.outofcore.halo_m,
                                ground_only=cfg.preprocessing.ground_only,
                                classification_filter=cfg.preprocessing.classification_filter,
                                chunk_points=cfg.outofcore.chunk_points,
                                transform_t2=(None if ('aligned_file_paths' in pc2_data and pc2_data['aligned_file_paths']) else transform_matrix),
                                n_workers=cfg.parallel.n_workers,
                                threads_per_worker=getattr(cfg.parallel, 'threads_per_worker', 1),
                                config=cfg,
                                clip_bounds=clip_bounds,
                                local_transform=local_transform,
                            )
                        else:
                            dod_res = ChangeDetector.compute_dod_streaming_files_tiled(
                                files_t1=files_t1,
                                files_t2=files_t2,
                                cell_size=cfg.detection.dod.cell_size,
                                tile_size=cfg.outofcore.tile_size_m,
                                halo=cfg.outofcore.halo_m,
                                ground_only=cfg.preprocessing.ground_only,
                                classification_filter=cfg.preprocessing.classification_filter,
                                chunk_points=cfg.outofcore.chunk_points,
                                transform_t2=(None if ('aligned_file_paths' in pc2_data and pc2_data['aligned_file_paths']) else transform_matrix),
                                config=cfg,
                            )
                    except Exception as stream_error:
                        logger.error(f"Streaming DoD failed: {stream_error}", exc_info=True)
                        logger.info("Falling back to in-memory DoD computation...")
                        # Fallback to in-memory
                        dod_res = ChangeDetector.compute_dod(
                            points_t1=points1_analysis,
                            points_t2=points2_analysis,
                            cell_size=cfg.detection.dod.cell_size,
                            aggregator=cfg.detection.dod.aggregator,
                            config=cfg,
                        )
                else:
                    # In-memory DoD computation
                    logger.info("Using in-memory DoD...")
                    dod_res = ChangeDetector.compute_dod(
                        points_t1=points1_analysis,
                        points_t2=points2_analysis,
                        cell_size=cfg.detection.dod.cell_size,
                        aggregator=cfg.detection.dod.aggregator,
                        config=cfg,
                    )
                if show_plots:
                    # Visualize DoD
                    # Revert grid coordinates to global for visualization (users expect UTM coordinates)
                    if local_transform is not None:
                        dod_res.grid_x = dod_res.grid_x + local_transform.offset_x
                        dod_res.grid_y = dod_res.grid_y + local_transform.offset_y
                    visualizer.visualize_dod_heatmap(dod_res, title="DEM of Difference (m)")
                    
                # Export DoD to GeoTIFF if enabled
                if getattr(cfg.detection.dod, 'export_raster', False):
                    try:
                        
                        # Determine output directory (flat structure, area name in filename)
                        if cfg.paths.output_dir:
                            export_dir = Path(cfg.paths.output_dir)
                        else:
                            export_dir = Path(cfg.paths.base_dir) / "output"
                        export_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Try to auto-detect CRS from input files
                        crs = cfg.paths.output_crs
                        try:
                            detected_crs = detect_crs_from_laz(str(ds1.laz_files[0]))
                            if detected_crs:
                                crs = detected_crs
                                logger.info(f"Auto-detected CRS from input: {crs}")
                        except Exception:
                            pass
                        
                        area_prefix = selected_area.area_name
                        dod_output = export_dir / f"dod_{area_prefix}_{t1}_{t2}.tif"
                        export_dod_to_geotiff(dod_res, str(dod_output), crs=crs)
                        logger.info(f"Exported DoD raster: {dod_output}")
                    except Exception as export_err:
                        logger.error(f"DoD export failed: {export_err}")
            except Exception as e:
                logger.error(f"DoD computation failed: {e}")
        else:
            logger.info("Skipping DoD (disabled in config).")

        # 3b) Cloud-to-Cloud (C2C)
        if getattr(cfg.detection.c2c, "enabled", True):
            try:
                logger.info("Computing Cloud-to-Cloud (C2C) distances...")

                use_streaming_c2c = (
                    use_streaming and cfg.detection.c2c.max_distance is not None and 'file_paths' in pc1_data
                )
                if use_streaming_c2c:
                    files_src = pc2_data.get('aligned_file_paths') or pc2_data['file_paths']
                    files_tgt = pc1_data['file_paths']
                    
                    # Check if parallel processing is enabled
                    use_parallel = getattr(cfg.parallel, 'enabled', False)
                    mode = "parallel" if use_parallel else "sequential"
                    logger.info(f"Using streaming C2C ({mode}, tiled)...")
                    
                    if use_parallel:
                        c2c_res = ChangeDetector.compute_c2c_streaming_files_tiled_parallel(
                            files_src=files_src,
                            files_tgt=files_tgt,
                            tile_size=cfg.outofcore.tile_size_m,
                            max_distance=float(cfg.detection.c2c.max_distance),
                            ground_only=cfg.preprocessing.ground_only,
                            classification_filter=cfg.preprocessing.classification_filter,
                            chunk_points=cfg.outofcore.chunk_points,
                            transform_src=(None if ('aligned_file_paths' in pc2_data and pc2_data['aligned_file_paths']) else transform_matrix),
                            n_workers=None,  # auto-detect
                            threads_per_worker=getattr(cfg.parallel, 'threads_per_worker', 1),
                            config=cfg,  # Pass config for GPU acceleration
                            clip_bounds=clip_bounds,
                            local_transform=local_transform,
                        )
                    else:
                        if getattr(cfg.detection.c2c, 'mode', 'euclidean') != 'euclidean':
                            logger.warning("C2C mode '%s' not supported in streaming; using euclidean.", cfg.detection.c2c.mode)
                        c2c_res = ChangeDetector.compute_c2c_streaming_files_tiled(
                            files_src=files_src,
                            files_tgt=files_tgt,
                            tile_size=cfg.outofcore.tile_size_m,
                            max_distance=float(cfg.detection.c2c.max_distance),
                            ground_only=cfg.preprocessing.ground_only,
                            classification_filter=cfg.preprocessing.classification_filter,
                            chunk_points=cfg.outofcore.chunk_points,
                            transform_src=(None if ('aligned_file_paths' in pc2_data and pc2_data['aligned_file_paths']) else transform_matrix),
                            config=cfg,  # Pass config for GPU acceleration
                            local_transform=local_transform,
                        )
                    # 3D scatter not supported in streaming; fallback to histogram if plotly
                    try:
                        if show_plots:
                            if cfg.visualization.backend == 'plotly':
                                visualizer.visualize_distance_histogram(
                                    c2c_res.distances, title="C2C distances (m)", bins=60
                                )
                    except Exception:
                        pass
                else:
                    c2c_mode = getattr(cfg.detection.c2c, 'mode', 'euclidean')
                    logger.info(f"Using in-memory C2C ({c2c_mode})...")
                    # Downsample to keep pairwise search manageable if sklearn is unavailable
                    max_points = cfg.detection.c2c.max_points
                    src = points2_analysis
                    tgt = points1_analysis
                    if len(src) > max_points:
                        idx = np.random.choice(len(src), max_points, replace=False)
                        src = src[idx]
                    if len(tgt) > max_points:
                        idx = np.random.choice(len(tgt), max_points, replace=False)
                        tgt = tgt[idx]
                    # Choose algorithm based on config
                    if getattr(cfg.detection.c2c, 'mode', 'euclidean') == 'vertical_plane':
                        c2c_res = ChangeDetector.compute_c2c_vertical_plane(
                            src,
                            tgt,
                            radius=cfg.detection.c2c.radius,
                            k_neighbors=cfg.detection.c2c.k_neighbors,
                            min_neighbors=cfg.detection.c2c.min_neighbors,
                            config=cfg,  # Pass config for GPU acceleration
                        )
                    else:
                        c2c_res = ChangeDetector.compute_c2c(src, tgt, max_distance=cfg.detection.c2c.max_distance, config=cfg)  # Pass config for GPU acceleration
                    
                    if show_plots:
                        # Visualize 3D per-point distances on the source cloud (like M3C2)
                        try:
                            visualizer.visualize_c2c_points(
                                src,
                                c2c_res.distances,
                                sample_size=cfg.visualization.sample_size,
                                title="C2C distances (m)",
                            )
                        except Exception:
                            pass
                    
                    # Export C2C results if enabled
                    export_c2c_pc = getattr(cfg.detection.c2c, 'export_pc', False)
                    export_c2c_raster = getattr(cfg.detection.c2c, 'export_raster', False)
                    if export_c2c_pc or export_c2c_raster:
                        try:
                            
                            # Determine output directory (flat structure, area name in filename)
                            if cfg.paths.output_dir:
                                export_dir = Path(cfg.paths.output_dir)
                            else:
                                export_dir = Path(cfg.paths.base_dir) / "output"
                            export_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Try to auto-detect CRS from input files
                            crs = cfg.paths.output_crs
                            try:
                                detected_crs = detect_crs_from_laz(str(ds1.laz_files[0]))
                                if detected_crs:
                                    crs = detected_crs
                            except Exception:
                                pass
                            
                            area_prefix = selected_area.area_name
                            if export_c2c_pc:
                                c2c_laz = export_dir / f"c2c_{area_prefix}_{t1}_{t2}.laz"
                                export_points_to_laz(
                                    src, c2c_res.distances, str(c2c_laz),
                                    crs=crs, source_laz_path=str(ds1.laz_files[0]),
                                    local_transform=local_transform
                                )
                                logger.info(f"Exported C2C point cloud: {c2c_laz}")
                            
                            if export_c2c_raster:
                                c2c_tif = export_dir / f"c2c_{area_prefix}_{t1}_{t2}.tif"
                                export_distances_to_geotiff(
                                    src, c2c_res.distances, str(c2c_tif),
                                    cell_size=cfg.detection.dod.cell_size, crs=crs,
                                    local_transform=local_transform
                                )
                                logger.info(f"Exported C2C raster: {c2c_tif}")
                        except Exception as export_err:
                            logger.error(f"C2C export failed: {export_err}")
            except Exception as e:
                logger.error(f"C2C computation failed: {e}")
        else:
            logger.info("Skipping C2C (disabled in config).")

    # 3c) M3C2
        if getattr(cfg.detection.m3c2, "enabled", True):
            try:
                m3c2_variant = getattr(cfg.detection.m3c2, "variant", "original")
                m3c2_label = "M3C2-EP" if m3c2_variant == "ep" else "M3C2"
                logger.info("Computing %s distances...", m3c2_label)

                if m3c2_variant == "ep":
                    if use_streaming:
                        raise ValueError("M3C2-EP is currently supported only in in-memory mode")
                    if getattr(cfg.parallel, "enabled", False):
                        raise ValueError("M3C2-EP does not support repository-level parallel mode")
                    if cfg.alignment.reference != "t1":
                        raise ValueError(
                            "M3C2-EP currently requires alignment.reference='t1' because py4dgeo "
                            "applies the transform to epoch 2 internally"
                        )

                points1_m3c2 = points1_analysis
                points2_m3c2 = points2_analysis

                # Core points selection or load from file for reproducibility across runs
                # Determine the total number of reference ground points for percentage calculation
                if use_streaming and 'metadata' in pc1_data:
                    # Streaming mode: use metadata ground point count collected during discovery
                    m1 = pc1_data['metadata']
                    total_ref_points = m1.get('total_points_ground')
                    if total_ref_points is None or total_ref_points == 0:
                        # Fallback to total_points_all if ground count not available
                        total_ref_points = m1.get('total_points_all', 0)
                        if total_ref_points > 0:
                            logger.warning(
                                f"No ground point count in metadata; using total points ({total_ref_points:,}). "
                                "M3C2 core point percentage may be inaccurate."
                            )
                    if total_ref_points == 0:
                        logger.error("Cannot determine reference point count for M3C2 core points percentage.")
                        raise ValueError("No reference point count available for M3C2")
                    logger.debug(f"Metadata-based T1 ground point count: {total_ref_points:,}")
                else:
                    # In-memory mode: use loaded array length (already filtered to ground)
                    total_ref_points = len(points1_m3c2)

                # Determine number of core points (percentage-based or absolute override)
                if cfg.detection.m3c2.core_points is not None:
                    # Backward compatibility: use absolute number if explicitly set
                    max_core = cfg.detection.m3c2.core_points
                    logger.info(f"M3C2 core points: {max_core:,} (absolute override)")
                else:
                    # New behavior: calculate from percentage of reference ground points
                    pct = cfg.detection.m3c2.core_points_percent
                    if pct is None:
                        pct = 10.0  # Default fallback
                    pct = max(0.1, min(100.0, pct))  # Clamp to valid range [0.1%, 100%]
                    max_core = max(1, int(total_ref_points * pct / 100.0))
                    logger.info(
                        f"M3C2 core points: {max_core:,} ({pct:.1f}% of {total_ref_points:,} reference ground points)"
                    )

                cores_path = Path(args.cores_file) if args.cores_file else None
                core_src = None
                if cores_path is not None and cores_path.exists():
                    try:
                        core_loaded = np.load(str(cores_path))
                        if core_loaded.ndim != 2 or core_loaded.shape[1] != 3:
                            raise ValueError("cores-file must contain an array of shape (N,3)")
                        core_src = core_loaded.astype(np.float64, copy=False)
                        logger.info(f"Loaded {len(core_src)} core points from {cores_path}")
                    except Exception as e:
                        logger.warning(f"Failed to load cores from {cores_path}: {e}; falling back to selection")
                        core_src = None

                if core_src is None:
                    # Select core points using appropriate method
                    # Check if we'll use per-tile M3C2 (parallel + streaming)
                    use_parallel_streaming = (
                        m3c2_variant == "original"
                        and use_streaming and 'file_paths' in pc1_data
                        and getattr(cfg.parallel, 'enabled', False)
                    )

                    if use_parallel_streaming:
                        # Skip global core selection - per-tile M3C2 handles it internally
                        logger.info(
                            f"Per-tile M3C2 will select {cfg.detection.m3c2.core_points_percent or 10.0:.1f}% "
                            "core points per tile (no global core selection needed)"
                        )
                        core_src = None
                    elif m3c2_variant == "original" and use_streaming and 'file_paths' in pc1_data:
                        # Sequential streaming mode: still needs global core selection
                        logger.info(f"Selecting {max_core:,} core points via streaming from T1 files...")
                        core_reader = LaspyStreamReader(
                            [str(p) for p in pc1_data['file_paths']],
                            ground_only=cfg.preprocessing.ground_only,
                            classification_filter=cfg.preprocessing.classification_filter,
                            chunk_points=cfg.outofcore.chunk_points,
                        )
                        core_src = core_reader.reservoir_sample(max_core, transform=local_transform)
                        logger.info(f"Selected {len(core_src):,} core points from T1 via streaming")
                    else:
                        # In-memory mode: subsample from loaded points1
                        if len(points1_m3c2) > max_core:
                            idx = np.random.choice(len(points1_m3c2), max_core, replace=False)
                            core_src = points1_m3c2[idx]
                        else:
                            core_src = points1_m3c2

                    # Save if a path was provided but did not exist
                    if cores_path is not None and core_src is not None:
                        try:
                            cores_path.parent.mkdir(parents=True, exist_ok=True)
                            np.save(str(cores_path), core_src)
                            logger.info(f"Saved {len(core_src)} core points to {cores_path}")
                        except Exception as e:
                            logger.warning(f"Could not save cores to {cores_path}: {e}")

                # Auto-tune M3C2 parameters based on point density
                # Select M3C2 parameters: fixed from config or autotuned from data
                if getattr(cfg.detection.m3c2, 'use_autotune', True) is False and getattr(cfg.detection.m3c2, 'fixed', None) is not None and cfg.detection.m3c2.fixed.radius is not None:
                    r = float(cfg.detection.m3c2.fixed.radius)
                    depth_factor = (
                        float(cfg.detection.m3c2.fixed.depth_factor)
                        if cfg.detection.m3c2.fixed.depth_factor is not None
                        else float(cfg.detection.m3c2.autotune.max_depth_factor)
                    )
                    normal_scale = (
                        float(cfg.detection.m3c2.fixed.normal_scale)
                        if cfg.detection.m3c2.fixed.normal_scale is not None
                        else r
                    )
                    m3c2_params = M3C2Params(
                        projection_scale=r,
                        cylinder_radius=r,
                        max_depth=r * depth_factor,
                        min_neighbors=10,
                        normal_scale=normal_scale,
                        confidence=0.95,
                    )
                    logger.info(
                        "M3C2 fixed params from config: radius=%.2f, normal_scale=%.2f, max_depth=%.2f (factor=%.2f)",
                        r, normal_scale, r * depth_factor, depth_factor,
                    )
                else:
                    # Optional header-based density for autotune (mode-agnostic)
                    at = cfg.detection.m3c2.autotune
                    use_header = getattr(at, 'source', 'header') == 'header'
                    # Build file lists for header-based density whether streaming or not
                    if use_streaming:
                        files_t1_params = pc1_data['file_paths'] if 'file_paths' in pc1_data else []
                        files_t2_params = pc2_data.get('aligned_file_paths') or pc2_data.get('file_paths') or []
                    else:
                        files_t1_params = [str(p) for p in ds1.laz_files]
                        files_t2_params = [str(p) for p in ds2.laz_files]

                    if use_header and files_t1_params:
                        m3c2_params = None
                        try:
                            m3c2_params = autotune_m3c2_params_from_headers(
                                files_t1=files_t1_params,
                                files_t2=files_t2_params,
                                target_neighbors=at.target_neighbors,
                                max_depth_factor=at.max_depth_factor,
                                min_radius=at.min_radius,
                                max_radius=at.max_radius,
                            )
                        except Exception as _e:
                            logger.warning(
                                f"Header-based autotune failed ({_e}); falling back to sample-based."
                            )
                        if m3c2_params is None:
                            # Fallback to sample-based
                            m3c2_params = autotune_m3c2_params(
                                points1_m3c2,
                                target_neighbors=at.target_neighbors,
                                max_depth_factor=at.max_depth_factor,
                                min_radius=at.min_radius,
                                max_radius=at.max_radius,
                            )
                    else:
                        # Sample-based (current behavior)
                        m3c2_params = autotune_m3c2_params(
                            points1_m3c2,
                            target_neighbors=at.target_neighbors,
                            max_depth_factor=at.max_depth_factor,
                            min_radius=at.min_radius,
                            max_radius=at.max_radius,
                        )

                # Prefer streaming tiled M3C2 when out-of-core is enabled and file paths are available
                use_streaming_m3c2 = (
                    m3c2_variant == "original"
                    and use_streaming and 'file_paths' in pc1_data and (
                        ('aligned_file_paths' in pc2_data and pc2_data['aligned_file_paths'])
                        or pc2_data.get('file_paths')
                    ) is not None
                )

                if use_streaming_m3c2:
                    files_t1 = pc1_data['file_paths']
                    files_t2 = pc2_data.get('aligned_file_paths') or pc2_data['file_paths']

                    # Check if parallel processing is enabled
                    use_parallel = getattr(cfg.parallel, 'enabled', False)
                    mode = "parallel" if use_parallel else "sequential"
                    logger.info(f"Using streaming M3C2 ({mode}, tiled)...")

                    try:
                        if use_parallel:
                            # Use per-tile core selection for truly out-of-core processing
                            m3c2_res_stream = ChangeDetector.compute_m3c2_streaming_pertile_parallel(
                                files_t1=files_t1,
                                files_t2=files_t2,
                                params=m3c2_params,
                                core_points_percent=cfg.detection.m3c2.core_points_percent or 10.0,
                                tile_size=cfg.outofcore.tile_size_m,
                                halo=None,
                                ground_only=cfg.preprocessing.ground_only,
                                classification_filter=cfg.preprocessing.classification_filter,
                                chunk_points=cfg.outofcore.chunk_points,
                                transform_t2=(None if ('aligned_file_paths' in pc2_data and pc2_data['aligned_file_paths']) else transform_matrix),
                                n_workers=None,
                                threads_per_worker=getattr(cfg.parallel, 'threads_per_worker', 1),
                                local_transform=local_transform,
                            )
                        else:
                            m3c2_res_stream = ChangeDetector.compute_m3c2_streaming_files_tiled(
                                core_points=core_src,
                                files_t1=files_t1,
                                files_t2=files_t2,
                                params=m3c2_params,
                                tile_size=cfg.outofcore.tile_size_m,
                                halo=None,
                                ground_only=cfg.preprocessing.ground_only,
                                classification_filter=cfg.preprocessing.classification_filter,
                                chunk_points=cfg.outofcore.chunk_points,
                                transform_t2=(None if ('aligned_file_paths' in pc2_data and pc2_data['aligned_file_paths']) else transform_matrix),
                                local_transform=local_transform,
                            )
                        # Optionally compute in-memory M3C2 using the same core points and compare
                        if args.debug_m3c2_compare:
                            logger.info("Debug: also running in-memory M3C2 for comparison...")
                            m3c2_res_mem = ChangeDetector.compute_m3c2_original(
                                core_points=core_src,
                                cloud_t1=points1_m3c2,
                                cloud_t2=points2_m3c2,
                                params=m3c2_params,
                            )
                            import numpy as _np
                            def _pearson(a, b):
                                a = _np.asarray(a, dtype=float).ravel()
                                b = _np.asarray(b, dtype=float).ravel()
                                m = _np.isfinite(a) & _np.isfinite(b)
                                if not _np.any(m):
                                    return float('nan')
                                a = a[m]; b = b[m]
                                n = a.size
                                if n < 2:
                                    return float('nan')
                                a = a - a.mean(); b = b - b.mean()
                                sa = a.std(); sb = b.std()
                                denom = sa * sb * n
                                num = float(a.dot(b))
                                return float(num / denom) if denom > 0 else float('nan')
                            def _summary(name: str, arr):
                                d = _np.asarray(arr, dtype=float).ravel()
                                d = d[_np.isfinite(d)]
                                n = d.size
                                if n == 0:
                                    logger.info("Debug M3C2 summary (%s): n=0", name)
                                    return
                                pos = float(_np.sum(d > 0)) / n * 100.0
                                neg = float(_np.sum(d < 0)) / n * 100.0
                                med = float(_np.median(d))
                                p5 = float(_np.percentile(d, 5))
                                p95 = float(_np.percentile(d, 95))
                                a95 = float(_np.percentile(_np.abs(d), 95))
                                logger.info(
                                    "Debug M3C2 summary (%s): n=%d, pos=%.1f%%, neg=%.1f%%, med=%.4f, p5=%.4f, p95=%.4f, abs_p95=%.4f",
                                    name, n, pos, neg, med, p5, p95, a95,
                                )
                            r_same = _pearson(m3c2_res_stream.distances, m3c2_res_mem.distances)
                            r_flip = _pearson(m3c2_res_stream.distances, -_np.asarray(m3c2_res_mem.distances))
                            logger.info("Debug M3C2: corr(stream, inmem)=%.6f, corr(stream, -inmem)=%.6f", r_same, r_flip)
                            try:
                                # Vertical proxy using nearest neighbors in each epoch
                                from sklearn.neighbors import NearestNeighbors as _NN
                                nn1 = _NN(n_neighbors=1, algorithm='kd_tree').fit(points1_m3c2)
                                nn2 = _NN(n_neighbors=1, algorithm='kd_tree').fit(points2_m3c2)
                                i1 = nn1.kneighbors(core_src, return_distance=False).ravel()
                                i2 = nn2.kneighbors(core_src, return_distance=False).ravel()
                                dz = points2_m3c2[i2, 2] - points1_m3c2[i1, 2]
                                rz_stream = _pearson(m3c2_res_stream.distances, dz)
                                rz_mem = _pearson(m3c2_res_mem.distances, dz)
                                rz_mem_flip = _pearson(-_np.asarray(m3c2_res_mem.distances), dz)
                                logger.info(
                                    "Debug M3C2: corr(stream, dZ)=%.6f, corr(inmem, dZ)=%.6f, corr(-inmem, dZ)=%.6f",
                                    rz_stream, rz_mem, rz_mem_flip,
                                )
                            except Exception as _e:
                                logger.warning(f"Debug M3C2: dZ proxy check skipped ({_e})")
                            # Sign/quantile summaries
                            _summary("stream", m3c2_res_stream.distances)
                            _summary("inmem", m3c2_res_mem.distances)
                        # Use streaming result for downstream visualization/output
                        m3c2_res = m3c2_res_stream
                    except Exception as stream_err:
                        logger.error(f"Streaming M3C2 failed: {stream_err}")
                        logger.info("Falling back to in-memory M3C2...")
                        m3c2_res = ChangeDetector.compute_m3c2_original(
                            core_points=core_src,
                            cloud_t1=points1_m3c2,
                            cloud_t2=points2_m3c2,
                            params=m3c2_params,
                        )
                elif m3c2_variant == "ep":
                    ep_cfg = cfg.detection.m3c2.ep
                    t1_point_source = pc1_data.get("attributes", {}).get("point_source_id")
                    t2_point_source = pc2_data.get("attributes", {}).get("point_source_id")
                    if t1_point_source is None or t2_point_source is None:
                        raise ValueError(
                            "M3C2-EP requires point_source_id on both epochs. "
                            "Ensure the input data contains it and clipping preserves attributes."
                        )

                    scan_metadata_t1 = M3C2Detector.resolve_scan_metadata(
                        points=points1_m3c2,
                        raw_scan_ids=t1_point_source,
                        scan_metadata_source=ep_cfg.scan_metadata_source,
                        explicit_path=ep_cfg.scan_positions_t1_path,
                        metadata_dir=ds1.metadata_dir,
                        auto_discover_from_metadata_dir=ep_cfg.auto_discover_from_metadata_dir,
                        synthetic_sigma_range=ep_cfg.synthetic_sigma_range,
                        synthetic_sigma_scan=ep_cfg.synthetic_sigma_scan,
                        synthetic_sigma_yaw=ep_cfg.synthetic_sigma_yaw,
                        synthetic_origin_height=ep_cfg.synthetic_origin_height,
                        epoch_label="T1",
                    )
                    scan_metadata_t2 = M3C2Detector.resolve_scan_metadata(
                        points=points2,
                        raw_scan_ids=t2_point_source,
                        scan_metadata_source=ep_cfg.scan_metadata_source,
                        explicit_path=ep_cfg.scan_positions_t2_path,
                        metadata_dir=ds2.metadata_dir,
                        auto_discover_from_metadata_dir=ep_cfg.auto_discover_from_metadata_dir,
                        synthetic_sigma_range=ep_cfg.synthetic_sigma_range,
                        synthetic_sigma_scan=ep_cfg.synthetic_sigma_scan,
                        synthetic_sigma_yaw=ep_cfg.synthetic_sigma_yaw,
                        synthetic_origin_height=ep_cfg.synthetic_origin_height,
                        epoch_label="T2",
                    )

                    if local_transform is not None:
                        for scan_metadata in (scan_metadata_t1, scan_metadata_t2):
                            if scan_metadata.source.startswith("sidecar:"):
                                for entry in scan_metadata.scanpos_info.values():
                                    entry["origin"][0] -= local_transform.offset_x
                                    entry["origin"][1] -= local_transform.offset_y
                                    entry["origin"][2] -= local_transform.offset_z

                    if ep_cfg.cxx_source == "icp_estimate":
                        alignment_cxx = estimate_alignment_covariance(
                            source=points2,
                            target=points1_m3c2,
                            transform=transform_matrix,
                            max_correspondence_distance=cfg.alignment.max_correspondence_distance,
                            reduction_point=np.zeros(3, dtype=np.float64),
                        )
                    elif ep_cfg.cxx_source == "file":
                        if not ep_cfg.alignment_covariance_path:
                            raise ValueError(
                                "detection.m3c2.ep.alignment_covariance_path must be set when cxx_source='file'"
                            )
                        alignment_cxx = _load_alignment_covariance_matrix(ep_cfg.alignment_covariance_path)
                    elif ep_cfg.cxx_source == "zero":
                        alignment_cxx = np.zeros((12, 12), dtype=np.float64)
                    else:
                        raise ValueError(f"Unsupported M3C2-EP cxx_source: {ep_cfg.cxx_source}")

                    logger.info("Using in-memory M3C2-EP...")
                    m3c2_res = ChangeDetector.compute_m3c2_ep(
                        core_points=core_src,
                        cloud_t1=points1_m3c2,
                        cloud_t2=points2,
                        params=m3c2_params,
                        transform_matrix=transform_matrix,
                        cxx=alignment_cxx,
                        scan_metadata_t1=scan_metadata_t1,
                        scan_metadata_t2=scan_metadata_t2,
                        reduction_point=np.zeros(3, dtype=np.float64),
                        perform_transform=True,
                    )
                else:
                    logger.info("Using in-memory M3C2...")
                    m3c2_res = ChangeDetector.compute_m3c2_original(
                        core_points=core_src,
                        cloud_t1=points1_m3c2,
                        cloud_t2=points2_m3c2,
                        params=m3c2_params,
                    )

                plot_title = "M3C2-EP distances (m)" if m3c2_variant == "ep" else "M3C2 distances (m)"
                if show_plots:
                    if cfg.visualization.backend == 'plotly':
                        visualizer.visualize_distance_histogram(m3c2_res.distances, title=plot_title, bins=60)

                    # Visualize M3C2 core points in 3D.
                    # Revert to global coordinates for visualization (users expect UTM coordinates).
                    vis_core_points = local_transform.to_global(m3c2_res.core_points) if local_transform else m3c2_res.core_points
                    visualizer.visualize_m3c2_corepoints(
                        vis_core_points,
                        m3c2_res.distances,
                        sample_size=cfg.visualization.sample_size,
                        title=plot_title,
                    )

                # Export M3C2 results if enabled
                export_m3c2_pc = getattr(cfg.detection.m3c2, 'export_pc', True)
                export_m3c2_raster = getattr(cfg.detection.m3c2, 'export_raster', True)
                if export_m3c2_pc or export_m3c2_raster or m3c2_variant == "ep":
                    try:
                        
                        # Determine output directory (flat structure, area name in filename)
                        if cfg.paths.output_dir:
                            export_dir = Path(cfg.paths.output_dir)
                        else:
                            export_dir = Path(cfg.paths.base_dir) / "output" / selected_area.area_name
                        export_dir.mkdir(parents=True, exist_ok=True)

                        # Try to auto-detect CRS from input files
                        crs = cfg.paths.output_crs
                        try:
                            detected_crs = detect_crs_from_laz(str(ds1.laz_files[0]))
                            if detected_crs:
                                crs = detected_crs
                        except Exception:
                            pass

                        area_prefix = selected_area.area_name
                        output_stem = "m3c2_ep" if m3c2_variant == "ep" else "m3c2"
                        if m3c2_variant == "ep" and alignment_cxx is not None:
                            cxx_output = export_dir / f"alignment_cxx_{area_prefix}_{t1}_{t2}.npy"
                            np.save(cxx_output, alignment_cxx)
                            logger.info(f"Exported alignment covariance matrix: {cxx_output}")
                        if export_m3c2_pc:
                            m3c2_laz = export_dir / f"{output_stem}_{area_prefix}_{t1}_{t2}.laz"
                            extra_dims = {}
                            if m3c2_res.uncertainty is not None:
                                extra_dims['uncertainty'] = m3c2_res.uncertainty
                            if m3c2_res.significant is not None:
                                extra_dims['significant'] = m3c2_res.significant
                            if m3c2_variant == "ep" and cfg.detection.m3c2.ep.export_scalar_fields and m3c2_res.ep_details is not None:
                                extra_dims['spread1'] = m3c2_res.ep_details.spread1
                                extra_dims['spread2'] = m3c2_res.ep_details.spread2
                                extra_dims['num_samples1'] = m3c2_res.ep_details.num_samples1
                                extra_dims['num_samples2'] = m3c2_res.ep_details.num_samples2
                            export_points_to_laz(
                                m3c2_res.core_points, m3c2_res.distances, str(m3c2_laz),
                                crs=crs, extra_dims=extra_dims if extra_dims else None,
                                source_laz_path=str(ds1.laz_files[0]),
                                local_transform=local_transform
                            )
                            logger.info(f"Exported {m3c2_label} point cloud: {m3c2_laz}")

                        if export_m3c2_raster:
                            m3c2_tif = export_dir / f"{output_stem}_{area_prefix}_{t1}_{t2}.tif"
                            export_distances_to_geotiff(
                                m3c2_res.core_points, m3c2_res.distances, str(m3c2_tif),
                                cell_size=cfg.detection.dod.cell_size, crs=crs,
                                local_transform=local_transform
                            )
                            logger.info(f"Exported {m3c2_label} raster: {m3c2_tif}")
                            if m3c2_variant == "ep" and m3c2_res.uncertainty is not None:
                                lod_tif = export_dir / f"{output_stem}_lod_{area_prefix}_{t1}_{t2}.tif"
                                export_distances_to_geotiff(
                                    m3c2_res.core_points,
                                    m3c2_res.uncertainty,
                                    str(lod_tif),
                                    cell_size=cfg.detection.dod.cell_size,
                                    crs=crs,
                                    local_transform=local_transform,
                                )
                                logger.info(f"Exported {m3c2_label} LoD raster: {lod_tif}")
                    except Exception as export_err:
                        logger.error(f"{m3c2_label} export failed: {export_err}")
            except Exception as e:
                logger.error(f"M3C2 computation failed: {e}", exc_info=True)
        else:
            logger.info("Skipping M3C2 (disabled in config).")

        if cfg.paths.output_dir:
            export_dir = Path(cfg.paths.output_dir)
        else:
            export_dir = Path(cfg.paths.base_dir) / "output" / selected_area.area_name

        write_run_inputs(
            export_dir,
            args,
            cfg,
            config_files=list(args.config or []),
            cli_overrides=cli_overrides,
        )

        runtime_end = perf_counter()
        runtime_elapsed = runtime_end - runtime_start
        logger.info(f"Total workflow runtime: {runtime_elapsed:.2f} seconds")

        # 3d) M3C2 with Error Propagation (EP)
        # try:
        #     logger.info("Computing M3C2 with Error Propagation (EP) and significance flags...")
        #     import platform
        #     if cfg.detection.m3c2.ep.workers is not None:
        #         workers = int(cfg.detection.m3c2.ep.workers)
        #     else:
        #         workers = 1 if platform.system().lower().startswith('win') else 4
        #     m3c2_ep = ChangeDetector.compute_m3c2_error_propagation(
        #         core_points=core_src,
        #         cloud_t1=points1,
        #         cloud_t2=points2_full_aligned,
        #         params=m3c2_params,
        #         workers=workers,
        #     )
        #     sig_count = int(np.sum(m3c2_ep.significant)) if m3c2_ep.significant is not None else 0
        #     logger.info(
        #         "M3C2-EP: significant=%d of %d (%.1f%%)",
        #         sig_count,
        #         m3c2_ep.distances.size,
        #         100.0 * sig_count / max(1, m3c2_ep.distances.size),
        #     )
        #     # Optional: visualize EP distributions
        #     # visualizer.visualize_distance_histogram(m3c2_ep.distances, title="M3C2-EP distances (m)", bins=60)
        #     if m3c2_ep.significant is not None:
        #         visualizer.visualize_distance_histogram(m3c2_ep.distances[m3c2_ep.significant], title="M3C2-EP distances (significant)", bins=60)
        # except Exception as e:
        #     logger.error(f"M3C2-EP computation failed: {e}")

    except Exception as e:
        logger.error(f"Change detection workflow failed: {e}")
        return
    finally:
        pass

if __name__ == "__main__":
    main()
