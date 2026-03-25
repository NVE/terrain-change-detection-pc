"""
Cloud-to-Cloud (C2C) distance computation for the terrain change detection workflow.

Handles streaming (sequential/parallel) and in-memory code paths, plus C2C
LAZ and GeoTIFF export.
"""

from __future__ import annotations

import logging

import numpy as np

from terrain_change_detection.detection import ChangeDetector
from terrain_change_detection.utils.config import AppConfig
from terrain_change_detection.utils.export import (
    export_distances_to_geotiff,
    export_points_to_laz,
)
from terrain_change_detection.visualization.point_cloud import PointCloudVisualizer

from .export_helpers import detect_output_crs, resolve_output_dir
from .types import AlignmentResult, PreparedData

logger = logging.getLogger(__name__)


def run_c2c(
    cfg: AppConfig,
    data: PreparedData,
    alignment: AlignmentResult,
    *,
    visualizer: PointCloudVisualizer | None = None,
    show_plots: bool = False,
) -> None:
    """Compute Cloud-to-Cloud distances (Step 3b).

    Args:
        cfg: Application configuration.
        data: Prepared dataset.
        alignment: Alignment results.
        visualizer: Optional visualizer instance.
        show_plots: Whether to display interactive plots.
    """
    if not getattr(cfg.detection.c2c, "enabled", True):
        logger.info("Skipping C2C (disabled in config).")
        return

    try:
        logger.info("Computing Cloud-to-Cloud (C2C) distances...")

        use_streaming_c2c = (
            data.use_streaming
            and cfg.detection.c2c.max_distance is not None
            and data.pc1_data is not None
            and 'file_paths' in data.pc1_data
        )

        if use_streaming_c2c:
            _run_streaming_c2c(cfg, data, alignment, visualizer=visualizer, show_plots=show_plots)
        else:
            _run_inmemory_c2c(cfg, data, alignment, visualizer=visualizer, show_plots=show_plots)

    except Exception as e:
        logger.error("C2C computation failed: %s", e)


def _run_streaming_c2c(cfg, data, alignment, *, visualizer, show_plots):
    """Streaming C2C with parallel/sequential paths."""
    if cfg.alignment.reference == "t2":
        files_tgt = data.pc1_data.get('aligned_file_paths') if data.pc1_data else None
        if files_tgt:
            files_src = data.pc2_data['file_paths']
            logger.info("Using pre-transformed files for T1 target: %d files", len(files_tgt))
            transform_src = None
        else:
            logger.info(
                "Falling back to in-memory C2C because streaming C2C can only apply "
                "an on-the-fly transform to the source cloud; reference=t2 requires "
                "aligned T1 files to keep T2 as the source cloud."
            )
            return _run_inmemory_c2c(
                cfg, data, alignment, visualizer=visualizer, show_plots=show_plots,
            )
    else:
        files_src = (data.pc2_data.get('aligned_file_paths') or data.pc2_data['file_paths'])
        files_tgt = data.pc1_data['file_paths']

        has_aligned_files = 'aligned_file_paths' in data.pc2_data and data.pc2_data['aligned_file_paths']
        transform_src = None if has_aligned_files else alignment.transform_matrix

    use_parallel = getattr(cfg.parallel, 'enabled', False)
    mode = "parallel" if use_parallel else "sequential"
    logger.info("Using streaming C2C (%s, tiled)...", mode)

    if use_parallel:
        c2c_res = ChangeDetector.compute_c2c_streaming_files_tiled_parallel(
            files_src=files_src,
            files_tgt=files_tgt,
            tile_size=cfg.outofcore.tile_size_m,
            max_distance=float(cfg.detection.c2c.max_distance),
            ground_only=cfg.preprocessing.ground_only,
            classification_filter=cfg.preprocessing.classification_filter,
            chunk_points=cfg.outofcore.chunk_points,
            transform_src=transform_src,
            n_workers=None,
            threads_per_worker=getattr(cfg.parallel, 'threads_per_worker', 1),
            config=cfg,
            clip_bounds=data.clip_bounds,
            local_transform=data.local_transform,
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
            transform_src=transform_src,
            config=cfg,
            local_transform=data.local_transform,
        )

    # Streaming visualization fallback
    try:
        if show_plots and visualizer is not None and cfg.visualization.backend == 'plotly':
            visualizer.visualize_distance_histogram(
                c2c_res.distances, title="C2C distances (m)", bins=60,
            )
    except Exception:
        pass


def _run_inmemory_c2c(cfg, data, alignment, *, visualizer, show_plots):
    """In-memory C2C computation."""
    c2c_mode = getattr(cfg.detection.c2c, 'mode', 'euclidean')
    logger.info("Using in-memory C2C (%s)...", c2c_mode)

    max_points = cfg.detection.c2c.max_points
    src = alignment.points2_aligned
    tgt = alignment.points1_aligned

    if len(src) > max_points:
        idx = np.random.choice(len(src), max_points, replace=False)
        src = src[idx]
    if len(tgt) > max_points:
        idx = np.random.choice(len(tgt), max_points, replace=False)
        tgt = tgt[idx]

    if c2c_mode == 'vertical_plane':
        c2c_res = ChangeDetector.compute_c2c_vertical_plane(
            src, tgt,
            radius=cfg.detection.c2c.radius,
            k_neighbors=cfg.detection.c2c.k_neighbors,
            min_neighbors=cfg.detection.c2c.min_neighbors,
            config=cfg,
        )
    else:
        c2c_res = ChangeDetector.compute_c2c(
            src, tgt,
            max_distance=cfg.detection.c2c.max_distance,
            config=cfg,
        )

    if show_plots and visualizer is not None:
        try:
            visualizer.visualize_c2c_points(
                src, c2c_res.distances,
                sample_size=cfg.visualization.sample_size,
                title="C2C distances (m)",
            )
        except Exception:
            pass

    # Export C2C results
    export_c2c_pc = getattr(cfg.detection.c2c, 'export_pc', False)
    export_c2c_raster = getattr(cfg.detection.c2c, 'export_raster', False)
    if export_c2c_pc or export_c2c_raster:
        _export_c2c(cfg, data, src, c2c_res, export_pc=export_c2c_pc, export_raster=export_c2c_raster)


def _export_c2c(cfg, data, src, c2c_res, *, export_pc, export_raster):
    """Export C2C results as LAZ and/or GeoTIFF."""
    try:
        # C2C uses flat output root (no area subdirectory) when output_dir is not set
        export_dir = resolve_output_dir(cfg, data.selected_area.area_name, area_scoped=False)
        crs = detect_output_crs(cfg, str(data.ds1.laz_files[0]))

        area_prefix = data.selected_area.area_name
        if export_pc:
            c2c_laz = export_dir / f"c2c_{area_prefix}_{data.t1}_{data.t2}.laz"
            export_points_to_laz(
                src, c2c_res.distances, str(c2c_laz),
                crs=crs, source_laz_path=str(data.ds1.laz_files[0]),
                local_transform=data.local_transform,
            )
            logger.info("Exported C2C point cloud: %s", c2c_laz)

        if export_raster:
            c2c_tif = export_dir / f"c2c_{area_prefix}_{data.t1}_{data.t2}.tif"
            export_distances_to_geotiff(
                src, c2c_res.distances, str(c2c_tif),
                cell_size=cfg.detection.dod.cell_size, crs=crs,
                local_transform=data.local_transform,
            )
            logger.info("Exported C2C raster: %s", c2c_tif)
    except Exception as export_err:
        logger.error("C2C export failed: %s", export_err)
