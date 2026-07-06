"""
DEM of Difference (DoD) computation for the terrain change detection workflow.

Handles streaming (sequential/parallel), in-memory, and fallback code paths,
plus DoD GeoTIFF export.
"""

from __future__ import annotations

import logging

import numpy as np

from terrain_change_detection.detection import ChangeDetector
from terrain_change_detection.utils.config import AppConfig
from terrain_change_detection.utils.export import export_dod_to_geotiff
from terrain_change_detection.visualization.point_cloud import PointCloudVisualizer

from .export_helpers import resolve_output_dir, resolve_workflow_crs
from .clipping import clipping_export_suffix, resolve_clipper
from .types import AlignmentResult, PreparedData

logger = logging.getLogger(__name__)


def run_dod(
    cfg: AppConfig,
    data: PreparedData,
    alignment: AlignmentResult,
    *,
    run_id: str | None = None,
    visualizer: PointCloudVisualizer | None = None,
    show_plots: bool = False,
) -> None:
    """Compute DEM of Difference (Step 3a).

    Args:
        cfg: Application configuration.
        data: Prepared dataset.
        alignment: Alignment results.
        run_id: Optional run identifier appended to output filenames.
        visualizer: Optional visualizer instance.
        show_plots: Whether to display interactive plots.
    """
    if not getattr(cfg.detection.dod, "enabled", True):
        logger.info("Skipping DoD (disabled in config).")
        return

    try:
        logger.info("Computing DEM of Difference (DoD)...")

        can_use_streaming = (
            data.use_streaming
            and cfg.detection.dod.aggregator == 'mean'
            and data.pc1_data is not None
            and 'file_paths' in data.pc1_data
        )

        if can_use_streaming:
            dod_res = _compute_streaming_dod(cfg, data, alignment)
        else:
            logger.info("Using in-memory DoD...")
            dod_res = _compute_inmemory_dod(cfg, alignment)

        if show_plots and visualizer is not None:
            if data.local_transform is not None:
                dod_res.grid_x = dod_res.grid_x + data.local_transform.offset_x
                dod_res.grid_y = dod_res.grid_y + data.local_transform.offset_y
            visualizer.visualize_dod_heatmap(dod_res, title="DEM of Difference (m)")

        # Export DoD GeoTIFF
        if getattr(cfg.detection.dod, 'export_raster', False):
            _export_dod(cfg, data, dod_res, run_id=run_id)

    except Exception as e:
        logger.error("DoD computation failed: %s", e)


def _compute_streaming_dod(cfg, data, alignment):
    """Streaming DoD with parallel/sequential/fallback paths."""
    if cfg.alignment.reference == "t2":
        files_t1 = data.pc1_data.get('aligned_file_paths') if data.pc1_data else None
        if files_t1:
            files_t2 = data.pc2_data['file_paths']
            logger.info("Using pre-transformed files for T1: %d files", len(files_t1))
            transform_t2 = None
        else:
            logger.info(
                "Falling back to in-memory DoD because streaming DoD can only apply "
                "an on-the-fly transform to T2; reference=t2 requires aligned T1 files."
            )
            return _compute_inmemory_dod(cfg, alignment)
    else:
        files_t1 = data.pc1_data['file_paths']
        if data.pc2_data and 'aligned_file_paths' in data.pc2_data and data.pc2_data['aligned_file_paths']:
            files_t2 = data.pc2_data['aligned_file_paths']
            logger.info("Using pre-transformed files for T2: %d files", len(files_t2))
            transform_t2 = None
        else:
            files_t2 = data.pc2_data['file_paths']
            logger.info("Using original T2 files with on-the-fly alignment transform")
            transform_t2 = alignment.transform_matrix

    mode = "parallel" if cfg.parallel.enabled else "sequential"
    logger.info("Using streaming DoD (%s, tiled)...", mode)

    try:
        if cfg.parallel.enabled:
            return ChangeDetector.compute_dod_streaming_files_tiled_parallel(
                files_t1=files_t1,
                files_t2=files_t2,
                cell_size=cfg.detection.dod.cell_size,
                tile_size=cfg.outofcore.tile_size_m,
                halo=cfg.outofcore.halo_m,
                ground_only=cfg.preprocessing.ground_only,
                classification_filter=cfg.preprocessing.classification_filter,
                chunk_points=cfg.outofcore.chunk_points,
                transform_t2=transform_t2,
                n_workers=cfg.parallel.n_workers,
                threads_per_worker=getattr(cfg.parallel, 'threads_per_worker', 1),
                config=cfg,
                clip_bounds=data.clip_bounds,
                local_transform=data.local_transform,
            )
        else:
            return ChangeDetector.compute_dod_streaming_files_tiled(
                files_t1=files_t1,
                files_t2=files_t2,
                cell_size=cfg.detection.dod.cell_size,
                tile_size=cfg.outofcore.tile_size_m,
                halo=cfg.outofcore.halo_m,
                ground_only=cfg.preprocessing.ground_only,
                classification_filter=cfg.preprocessing.classification_filter,
                chunk_points=cfg.outofcore.chunk_points,
                transform_t2=transform_t2,
                config=cfg,
            )
    except Exception as stream_error:
        logger.error("Streaming DoD failed: %s", stream_error, exc_info=True)
        logger.info("Falling back to in-memory DoD computation...")
        return _compute_inmemory_dod(cfg, alignment)


def _compute_inmemory_dod(cfg, alignment):
    """In-memory DoD using the alignment-corrected epoch clouds."""
    return ChangeDetector.compute_dod(
        points_t1=alignment.points1_aligned,
        points_t2=alignment.points2_aligned,
        cell_size=cfg.detection.dod.cell_size,
        aggregator=cfg.detection.dod.aggregator,
        config=cfg,
    )


def _export_dod(cfg, data, dod_res, *, run_id: str | None = None):
    """Export DoD as GeoTIFF."""
    try:
        export_dir = resolve_output_dir(cfg, data.selected_area.area_name, area_scoped=True)
        crs = resolve_workflow_crs(cfg, data.ds1.laz_files[0], data.ds2.laz_files[0])

        area_prefix = data.selected_area.area_name
        clip_suffix = clipping_export_suffix(cfg)
        run_suffix = f"_{run_id}" if run_id else ""
        dod_output = export_dir / f"dod_{area_prefix}_{data.t1}_{data.t2}{clip_suffix}{run_suffix}.tif"
        _mask_dod_to_clipping_geometry(cfg, data, dod_res, workflow_crs=crs)
        export_dod_to_geotiff(
            dod_res, str(dod_output), crs=crs,
            local_transform=data.local_transform,
        )
        logger.info("Exported DoD raster: %s", dod_output)
    except Exception as export_err:
        logger.error("DoD export failed: %s", export_err)


def _mask_dod_to_clipping_geometry(cfg, data, dod_res, *, workflow_crs: str | None = None) -> None:
    """Set DoD cells outside clipping polygon to NaN before export."""
    clipper = resolve_clipper(cfg, data.local_transform, workflow_crs=workflow_crs)
    if clipper is None or dod_res.dod.size == 0:
        return

    grid_points = np.column_stack([
        np.asarray(dod_res.grid_x).ravel(),
        np.asarray(dod_res.grid_y).ravel(),
        np.zeros(dod_res.grid_x.size, dtype=float),
    ])
    _, mask = clipper.clip(grid_points, return_mask=True)
    mask_2d = mask.reshape(dod_res.dod.shape)
    dod_res.dod = np.where(mask_2d, dod_res.dod, np.nan)
    dod_res.dem1 = np.where(mask_2d, dod_res.dem1, np.nan)
    dod_res.dem2 = np.where(mask_2d, dod_res.dem2, np.nan)
    logger.info("Masked DoD raster to clipping geometry: %d/%d cells kept", int(mask.sum()), int(mask.size))
