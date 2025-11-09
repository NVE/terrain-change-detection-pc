"""
Example script for complete terrain change detection workflow

This script demonstrates the full workflow from data discovery to change detection.
"""

import sys
import argparse
import logging
import os
import numpy as np
from pathlib import Path

# Add the src to the path to import modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from terrain_change_detection.preprocessing.loader import PointCloudLoader
from terrain_change_detection.preprocessing.data_discovery import DataDiscovery, BatchLoader
from terrain_change_detection.alignment.fine_registration import ICPRegistration
from terrain_change_detection.alignment.coarse_registration import CoarseRegistration
from terrain_change_detection.alignment import apply_transform_to_files, save_transform_matrix
from terrain_change_detection.utils.logging import setup_logger
from terrain_change_detection.detection import ChangeDetector, autotune_m3c2_params
from terrain_change_detection.visualization.point_cloud import PointCloudVisualizer
from terrain_change_detection.utils.config import load_config, AppConfig
from terrain_change_detection.acceleration import LaspyStreamReader

# Hardware optimizations
# TO DO: Implement hardware optimizations for large datasets

# Tuning knobs (now configured via YAML):
# - alignment.subsample_size: subsample size for ICP alignment
# - detection.m3c2.core_points: number of core points for M3C2
# - detection.c2c.max_points: maximum points per cloud for C2C distances
# - visualization.sample_size: sample size for visualization

def main():
    """
    Main function to run the terrain change detection workflow.
    """
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
        default=None,
        help="Path to YAML configuration file (defaults to config/default.yaml)",
    )
    args, unknown = parser.parse_known_args()

    # Load configuration
    cfg: AppConfig = load_config(args.config)
    if args.base_dir:
        cfg.paths.base_dir = args.base_dir

    # Setup logging from config
    log_level = getattr(logging, cfg.logging.level.upper(), logging.INFO)
    logger = setup_logger(__name__, level=log_level, log_file=cfg.logging.file)

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

    print("Terrain Change Detection Workflow")
    print("=================================")

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
        data_dir_name=cfg.discovery.data_dir_name,
        metadata_dir_name=cfg.discovery.metadata_dir_name,
        loader=loader,
    )
    areas = data_discovery.scan_areas()

    if not areas:
        logger.error("No area/area directory found in the base directory.")
        return

    # Find the first area with at least two time periods
    selected_area = None
    for area_name, area_info in areas.items():
        if len(area_info.time_periods) >= 2:
            selected_area = area_info
            break

    if not selected_area:
        logger.error("Could not find an area with at least two time periods for change detection.")
        return

    # Select the first two time periods
    t1, t2 = selected_area.time_periods[:2]
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

    try:
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
            logger.info("Loading subsampled data for alignment (subsample_size=%d)...", cfg.alignment.subsample_size)

            def stream_sample(files: list[str], n: int) -> np.ndarray:
                reader = LaspyStreamReader(
                    files,
                    ground_only=cfg.preprocessing.ground_only,
                    classification_filter=cfg.preprocessing.classification_filter,
                    chunk_points=cfg.outofcore.chunk_points,
                )
                reservoir = None
                filled = 0
                seen = 0
                for chunk in reader.stream_points():
                    if chunk.size == 0:
                        continue
                    m = len(chunk)
                    if reservoir is None:
                        reservoir = np.empty((n, 3), dtype=np.float64)
                    # Fill reservoir first
                    take = min(n - filled, m)
                    if take > 0:
                        reservoir[filled:filled + take] = chunk[:take]
                        filled += take
                        seen += take
                        start = take
                    else:
                        start = 0
                    # Replacement
                    for k in range(start, m):
                        j = seen + (k - start)
                        r = np.random.randint(0, j + 1)
                        if r < n:
                            reservoir[r] = chunk[k]
                    seen += (m - start)
                if reservoir is None:
                    return np.empty((0, 3), dtype=np.float64)
                return reservoir

            n_per_dataset = cfg.alignment.subsample_size
            points1 = stream_sample([str(p) for p in ds1.laz_files], n_per_dataset)
            points2 = stream_sample([str(p) for p in ds2.laz_files], n_per_dataset)
            
            logger.info(f"Loaded {len(points1)} sample points from T1 for alignment")
            logger.info(f"Loaded {len(points2)} sample points from T2 for alignment")
        else:
            logger.info("--- Step 1: Loading point cloud data (in-memory) ---")
            batch_loader = BatchLoader(loader=loader)
            if len(ds1.laz_files) > 1:
                logger.info(f"Batch loading {len(ds1.laz_files)} files for time period {t1}...")
                pc1_data = batch_loader.load_dataset(ds1)
            else:
                logger.info(f"Loading single file for time period {t1}...")
                pc1_data = batch_loader.loader.load(str(ds1.laz_files[0]))

            if len(ds2.laz_files) > 1:
                logger.info(f"Batch loading {len(ds2.laz_files)} files for time period {t2}...")
                pc2_data = batch_loader.load_dataset(ds2)
            else:
                logger.info(f"Loading single file for time period {t2}...")
                pc2_data = batch_loader.loader.load(str(ds2.laz_files[0]))

            logger.info(f"Dataset 1 ({t1}): {pc1_data['points'].shape[0]} points")
            logger.info(f"Dataset 2 ({t2}): {pc2_data['points'].shape[0]} points")

            points1 = pc1_data['points']
            points2 = pc2_data['points']

        # Instantiate the visualizer (choose backend)
        VIS_BACKEND = cfg.visualization.backend
        visualizer = PointCloudVisualizer(backend=VIS_BACKEND)

        # Visualize the original point clouds
        logger.info("--- Visualizing original point clouds ---")
        visualizer.visualize_clouds(
            point_clouds=[points1, points2],
            names=[f"PC from {t1}", f"PC from {t2}"],
            sample_size=cfg.visualization.sample_size  # Downsample for visualization
        )

        # ============================================================
        # Step 2: Spatial Alignment
        # ============================================================
        logger.info("=== STEP 2: Spatial Alignment ===")

        # Optional coarse registration to initialize ICP
        initial_transform = None
        try:
            if getattr(cfg.alignment, 'coarse', None) and cfg.alignment.coarse.enabled:
                logger.info(
                    "Coarse registration enabled (method=%s)...",
                    cfg.alignment.coarse.method,
                )
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
                    )
                    pre_err = tmp_icp.compute_registration_error(points2_init, points1)
                    logger.info("Pre-ICP RMSE after coarse registration: %.6f", pre_err)
                except Exception:
                    pass
            else:
                logger.info("Coarse registration disabled.")
        except Exception as e:
            logger.warning(f"Coarse registration failed: {e}")

        icp = ICPRegistration(
            max_iterations=cfg.alignment.max_iterations,
            tolerance=cfg.alignment.tolerance,
            max_correspondence_distance=cfg.alignment.max_correspondence_distance,
        )

        # Subsample for alignment if datasets are large
        if len(points1) > cfg.alignment.subsample_size:
            n1 = min(len(points1), cfg.alignment.subsample_size)
            indices1 = np.random.choice(len(points1), n1, replace=False)
            points1_subsampled = points1[indices1]
        else:
            points1_subsampled = points1

        if len(points2) > cfg.alignment.subsample_size:
            n2 = min(len(points2), cfg.alignment.subsample_size)
            indices2 = np.random.choice(len(points2), n2, replace=False)
            points2_subsampled = points2[indices2]
        else:
            points2_subsampled = points2

        # Perform ICP alignment
        points2_subsampled_aligned, transform_matrix, final_error = icp.align_point_clouds(
            source=points2_subsampled,
            target=points1_subsampled,
            initial_transform=initial_transform,
        )

        # Apply the transformation to the original points2
        if len(points2) > cfg.alignment.subsample_size:
            points2_full_aligned = icp.apply_transformation(points2, transform_matrix)
        else:
            points2_full_aligned = points2_subsampled_aligned

        # Compute the registration error
        alignment_error = icp.compute_registration_error(
            source=points2_full_aligned,
            target=points1
        )

        print(f"ICP Alignment completed with final error: {alignment_error:.6f}")

        # If streaming mode, optionally apply transform to original files
        if use_streaming and cfg.outofcore.save_transformed_files:
            logger.info("--- Applying transformation to full datasets (streaming) ---")
            
            # Determine output directory
            if cfg.outofcore.output_dir:
                output_dir = Path(cfg.outofcore.output_dir) / selected_area.area_name / f"{t2}_aligned"
            else:
                output_dir = Path(cfg.paths.base_dir).parent / "processed" / selected_area.area_name / f"{t2}_aligned"
            
            try:
                aligned_files = apply_transform_to_files(
                    input_files=pc2_data['file_paths'],
                    output_dir=str(output_dir),
                    transform=transform_matrix,
                    ground_only=cfg.preprocessing.ground_only,
                    classification_filter=cfg.preprocessing.classification_filter,
                    chunk_points=cfg.outofcore.chunk_points,
                )
                # Store aligned file paths for later use
                pc2_data['aligned_file_paths'] = aligned_files
                
                # Save transformation matrix for reference
                transform_file = output_dir / "transformation_matrix.txt"
                save_transform_matrix(transform_matrix, str(transform_file))
                
                logger.info(f"Transformed {len(aligned_files)} files saved to {output_dir}")
            except Exception as e:
                logger.error(f"Failed to apply transformation to files: {e}")
                logger.info("Falling back to in-memory aligned points for DoD")
                # Keep points2_full_aligned for DoD computation

        # Visualize the aligned point clouds
        logger.info("--- Visualizing aligned point clouds ---")
        visualizer.visualize_clouds(
            point_clouds=[points1, points2_full_aligned],
            names=[f"PC from {t1} (Target)", f"PC from {t2} (Aligned)"],
            sample_size=cfg.visualization.sample_size  # Downsample for visualization
        )

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
                        logger.info(f"Using transformed files for T2: {len(files_t2)} files")
                    else:
                        files_t2 = pc2_data['file_paths']
                        logger.warning("Transformed files not available, using original T2 files (misalignment may affect results)")

                    logger.info("Using out-of-core streaming DoD (mean aggregator, tiled)...")
                    logger.info(f"T1 files: {files_t1}")
                    logger.info(f"T2 files: {files_t2}")
                    logger.info(f"Tile size: {cfg.outofcore.tile_size_m}m, Halo: {cfg.outofcore.halo_m}m")

                    try:
                        # Choose parallel or sequential based on config
                        if cfg.parallel.enabled:
                            logger.info(f"Using PARALLEL tile processing (workers={cfg.parallel.n_workers or 'auto'})")
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
                            )
                        else:
                            logger.info("Using SEQUENTIAL tile processing")
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
                            )
                    except Exception as stream_error:
                        logger.error(f"Streaming DoD failed: {stream_error}")
                        logger.info("Falling back to in-memory DoD computation...")
                        # Fallback to in-memory
                        dod_res = ChangeDetector.compute_dod(
                            points_t1=points1,
                            points_t2=points2_full_aligned,
                            cell_size=cfg.detection.dod.cell_size,
                            aggregator=cfg.detection.dod.aggregator,
                        )
                else:
                    # In-memory DoD computation
                    logger.info("Using in-memory DoD computation...")
                    dod_res = ChangeDetector.compute_dod(
                        points_t1=points1,
                        points_t2=points2_full_aligned,
                        cell_size=cfg.detection.dod.cell_size,
                        aggregator=cfg.detection.dod.aggregator,
                    )
                logger.info(
                    "DoD stats: n_cells=%d, mean=%.4f m, median=%.4f m, rmse=%.4f m, min=%.4f m, max=%.4f m",
                    dod_res.stats.get("n_cells", 0),
                    dod_res.stats.get("mean_change", float('nan')),
                    dod_res.stats.get("median_change", float('nan')),
                    dod_res.stats.get("rmse", float('nan')),
                    dod_res.stats.get("min_change", float('nan')),
                    dod_res.stats.get("max_change", float('nan')),
                )
                # Visualize DoD immediately
                visualizer.visualize_dod_heatmap(dod_res, title="DEM of Difference (m)")
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
                    if use_parallel:
                        logger.info("Using PARALLEL streaming tiled C2C...")
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
                        )
                    else:
                        logger.info("Using streaming tiled C2C...")
                        if getattr(cfg.detection.c2c, 'mode', 'euclidean') != 'euclidean':
                            logger.warning("C2C mode '%s' not supported in streaming; falling back to euclidean distances.", cfg.detection.c2c.mode)
                        c2c_res = ChangeDetector.compute_c2c_streaming_files_tiled(
                            files_src=files_src,
                            files_tgt=files_tgt,
                            tile_size=cfg.outofcore.tile_size_m,
                            max_distance=float(cfg.detection.c2c.max_distance),
                            ground_only=cfg.preprocessing.ground_only,
                            classification_filter=cfg.preprocessing.classification_filter,
                            chunk_points=cfg.outofcore.chunk_points,
                            transform_src=(None if ('aligned_file_paths' in pc2_data and pc2_data['aligned_file_paths']) else transform_matrix),
                        )
                    # 3D scatter not supported in streaming; fallback to histogram if plotly
                    try:
                        if cfg.visualization.backend == 'plotly':
                            visualizer.visualize_distance_histogram(
                                c2c_res.distances, title="C2C distances (m)", bins=60
                            )
                    except Exception:
                        pass
                else:
                    logger.info("Using in-memory C2C with downsampling for speed...")
                    # Downsample to keep pairwise search manageable if sklearn is unavailable
                    max_points = cfg.detection.c2c.max_points
                    src = points2_full_aligned
                    tgt = points1
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
                        )
                    else:
                        c2c_res = ChangeDetector.compute_c2c(src, tgt, max_distance=cfg.detection.c2c.max_distance)
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
                logger.info(
                    "C2C stats: n=%d, mean=%.4f m, median=%.4f m, rmse=%.4f m",
                    c2c_res.n,
                    c2c_res.mean,
                    c2c_res.median,
                    c2c_res.rmse,
                )
            except Exception as e:
                logger.error(f"C2C computation failed: {e}")
        else:
            logger.info("Skipping C2C (disabled in config).")

        # 3c) M3C2
        if getattr(cfg.detection.m3c2, "enabled", True):
            try:
                logger.info("Computing M3C2 distances on core points (downsampled)...")

                # Generate core points by uniform subsampling of target (T1)
                max_core = cfg.detection.m3c2.core_points
                core_src = points1
                if len(core_src) > max_core:
                    idx = np.random.choice(len(core_src), max_core, replace=False)
                    core_src = core_src[idx]

                # Auto-tune M3C2 parameters based on point density
                m3c2_params = autotune_m3c2_params(
                    points1,
                    target_neighbors=cfg.detection.m3c2.autotune.target_neighbors,
                    max_depth_factor=cfg.detection.m3c2.autotune.max_depth_factor,
                    min_radius=cfg.detection.m3c2.autotune.min_radius,
                    max_radius=cfg.detection.m3c2.autotune.max_radius,
                )
                logger.info(
                    "M3C2 auto-tuned params: proj_scale=%.2f, cyl_radius=%.2f, max_depth=%.2f, min_neighbors=%d",
                    m3c2_params.projection_scale,
                    m3c2_params.cylinder_radius,
                    m3c2_params.max_depth,
                    m3c2_params.min_neighbors,
                )

                # Prefer streaming tiled M3C2 when out-of-core is enabled and file paths are available
                use_streaming_m3c2 = (
                    use_streaming and 'file_paths' in pc1_data and (
                        'aligned_file_paths' in pc2_data and pc2_data['aligned_file_paths'] or pc2_data.get('file_paths')
                    ) is not None
                )

                if use_streaming_m3c2:
                    files_t1 = pc1_data['file_paths']
                    files_t2 = pc2_data.get('aligned_file_paths') or pc2_data['file_paths']
                    
                    # Check if parallel processing is enabled
                    use_parallel = getattr(cfg.parallel, 'enabled', False)
                    if use_parallel:
                        logger.info("Using PARALLEL streaming tiled M3C2...")
                    else:
                        logger.info("Using streaming tiled M3C2...")
                    logger.info(f"T1 files: {files_t1}")
                    logger.info(f"T2 files: {files_t2}")
                    try:
                        if use_parallel:
                            m3c2_res = ChangeDetector.compute_m3c2_streaming_files_tiled_parallel(
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
                                n_workers=None,  # auto-detect
                            )
                        else:
                            m3c2_res = ChangeDetector.compute_m3c2_streaming_files_tiled(
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
                            )
                    except Exception as stream_err:
                        logger.error(f"Streaming M3C2 failed: {stream_err}")
                        logger.info("Falling back to in-memory M3C2 (Original)...")
                        m3c2_res = ChangeDetector.compute_m3c2_original(
                            core_points=core_src,
                            cloud_t1=points1,
                            cloud_t2=points2_full_aligned,
                            params=m3c2_params,
                        )
                else:
                    m3c2_res = ChangeDetector.compute_m3c2_original(
                        core_points=core_src,
                        cloud_t1=points1,
                        cloud_t2=points2_full_aligned,
                        params=m3c2_params,
                    )
                logger.info(
                    "M3C2 stats: n=%d, mean=%.4f m, median=%.4f m, std=%.4f m",
                    m3c2_res.distances.size,
                    float(np.mean(m3c2_res.distances)),
                    float(np.median(m3c2_res.distances)),
                    float(np.std(m3c2_res.distances)),
                )
                # Visualize M3C2 core points immediately
                visualizer.visualize_m3c2_corepoints(
                    m3c2_res.core_points,
                    m3c2_res.distances,
                    sample_size=cfg.visualization.sample_size,
                    title="M3C2 distances (m)",
                )
                # Visualize M3C2 distance histogram
                # visualizer.visualize_distance_histogram(m3c2_res.distances, title="M3C2 distances (m)", bins=60)
            except Exception as e:
                logger.error(f"M3C2 computation failed: {e}")
        else:
            logger.info("Skipping M3C2 (disabled in config).")

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

if __name__ == "__main__":
    main()
