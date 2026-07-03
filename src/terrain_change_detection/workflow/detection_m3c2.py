"""
M3C2 distance computation for the terrain change detection workflow.

Handles core-point selection (file-based, streaming, in-memory), parameter
autotuning (header-based, sample-based, fixed), streaming/parallel/in-memory
M3C2 computation, debug comparison, and export.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from terrain_change_detection.acceleration import LaspyStreamReader
from terrain_change_detection.detection import (
    ChangeDetector,
    M3C2Params,
    autotune_m3c2_params,
    autotune_m3c2_params_from_headers,
)
from terrain_change_detection.utils.config import AppConfig
from terrain_change_detection.utils.export import (
    export_distances_to_geotiff,
    export_erosion_polygons_geojson,
    export_points_to_laz,
)
from terrain_change_detection.visualization.point_cloud import PointCloudVisualizer

from .export_helpers import detect_output_crs, resolve_output_dir
from .clipping import clipping_export_suffix, resolve_clipper
from .types import AlignmentResult, PreparedData
from .visualization_helpers import to_global_for_vis

logger = logging.getLogger(__name__)


def run_m3c2(
    cfg: AppConfig,
    data: PreparedData,
    alignment: AlignmentResult,
    args: argparse.Namespace,
    *,
    run_id: str | None = None,
    visualizer: PointCloudVisualizer | None = None,
    show_plots: bool = False,
) -> dict:
    """Compute M3C2 distances (Step 3c).

    Args:
        cfg: Application configuration.
        data: Prepared dataset.
        alignment: Alignment results.
        args: CLI namespace (for ``cores_file`` and ``debug_m3c2_compare``).
        visualizer: Optional visualizer instance.
        show_plots: Whether to display interactive plots.
    """
    if not getattr(cfg.detection.m3c2, "enabled", True):
        logger.info("Skipping M3C2 (disabled in config).")
        return {"enabled": False}

    try:
        logger.info("Computing M3C2 distances...")
        points_t1 = alignment.points1_aligned
        points_t2 = alignment.points2_aligned
        evaluation_source = cfg.detection.m3c2.evaluation_source
        evaluation_points = points_t2 if evaluation_source == "t2" else points_t1

        # ----------------------------------------------------------------
        # Core-point count
        # ----------------------------------------------------------------
        total_ref_points = _determine_ref_point_count(data, evaluation_points, evaluation_source)

        # ----------------------------------------------------------------
        # Resolve streaming inputs up front so reference=t2 can fall back
        # cleanly when aligned T1 files are unavailable.
        # ----------------------------------------------------------------
        streaming_inputs = _resolve_streaming_m3c2_inputs(cfg, data, alignment)

        if cfg.detection.m3c2.core_points is not None:
            max_core = cfg.detection.m3c2.core_points
            logger.info("M3C2 core points: %s (absolute override)", f"{max_core:,}")
        else:
            pct = cfg.detection.m3c2.core_points_percent
            if pct is None:
                pct = 10.0
            pct = max(0.1, min(100.0, pct))
            max_core = max(1, int(total_ref_points * pct / 100.0))
            logger.info(
                "M3C2 core points: %s (%.1f%% of %s reference ground points)",
                f"{max_core:,}", pct, f"{total_ref_points:,}",
            )

        # ----------------------------------------------------------------
        # Core-point selection (or load from file)
        # ----------------------------------------------------------------
        core_src = _select_core_points(
            cfg,
            data,
            args,
            max_core,
            reference_points=evaluation_points,
            streaming_core_files=(
                _streaming_core_files(streaming_inputs, evaluation_source)
                if streaming_inputs is not None else None
            ),
            streaming_enabled=streaming_inputs is not None,
        )

        # ----------------------------------------------------------------
        # M3C2 parameters (fixed or autotuned)
        # ----------------------------------------------------------------
        m3c2_params, params_source = _resolve_m3c2_params(
            cfg,
            data,
            reference_points=evaluation_points,
            streaming_inputs=streaming_inputs,
        )

        # ----------------------------------------------------------------
        # Compute M3C2
        # ----------------------------------------------------------------
        if streaming_inputs is not None:
            m3c2_res = _compute_streaming_m3c2(
                cfg,
                data,
                alignment,
                core_src,
                m3c2_params,
                args,
                streaming_inputs=streaming_inputs,
                points_t1=points_t1,
                points_t2=points_t2,
            )
        else:
            logger.info("Using in-memory M3C2...")
            m3c2_res = ChangeDetector.compute_m3c2_original(
                core_points=core_src,
                cloud_t1=points_t1,
                cloud_t2=points_t2,
                params=m3c2_params,
            )

        # ----------------------------------------------------------------
        # Visualization
        # ----------------------------------------------------------------
        if show_plots and visualizer is not None:
            visualizer.visualize_distance_histogram(
                m3c2_res.distances, title="M3C2 distances (m)", bins=60,
            )
            vis_core_points = to_global_for_vis(m3c2_res.core_points, data.local_transform)
            visualizer.visualize_m3c2_corepoints(
                vis_core_points, m3c2_res.distances,
                sample_size=cfg.visualization.sample_size,
                title="M3C2 distances (m)",
            )

        # ----------------------------------------------------------------
        # Export
        # ----------------------------------------------------------------
        output_paths = _export_m3c2(cfg, data, m3c2_res, run_id=run_id)
        return _build_m3c2_summary(cfg, m3c2_params, params_source, max_core, m3c2_res, output_paths)

    except Exception as e:
        logger.error("M3C2 computation failed: %s", e)
        return {"enabled": True, "error": str(e)}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _determine_ref_point_count(data: PreparedData, reference_points: np.ndarray, evaluation_source: str) -> int:
    """Determine total reference ground points for core-point percentage."""
    pc_data = data.pc2_data if evaluation_source == "t2" else data.pc1_data
    if data.use_streaming and pc_data and 'metadata' in pc_data:
        metadata = pc_data['metadata']
        total = metadata.get('total_points_ground')
        if total is None or total == 0:
            total = metadata.get('total_points_all', 0)
            if total > 0:
                logger.warning(
                    "No ground point count in metadata; using total points (%s). "
                    "M3C2 core point percentage may be inaccurate.",
                    f"{total:,}",
                )
        if total == 0:
            raise ValueError("No reference point count available for M3C2")
        logger.debug("Metadata-based %s ground point count: %s", evaluation_source.upper(), f"{total:,}")
        return total
    return len(reference_points)


def _streaming_core_files(streaming_inputs: dict, evaluation_source: str) -> list[str] | None:
    """Return streaming files used for core-point selection."""
    if evaluation_source == "t2" and streaming_inputs.get("transform_t2") is not None:
        logger.warning(
            "Selecting M3C2 core points from aligned in-memory T2 points because T2 streaming files "
            "require an on-the-fly transform."
        )
        return None
    return streaming_inputs["files_t2"] if evaluation_source == "t2" else streaming_inputs["files_t1"]


def _select_core_points(
    cfg: AppConfig,
    data: PreparedData,
    args: argparse.Namespace,
    max_core: int,
    *,
    reference_points: np.ndarray,
    streaming_core_files: list[str] | None = None,
    streaming_enabled: bool = False,
) -> np.ndarray | None:
    """Select or load M3C2 core points."""
    cores_path = Path(args.cores_file) if args.cores_file else None
    core_src = None

    # Try to load from file
    if cores_path is not None and cores_path.exists():
        try:
            core_loaded = np.load(str(cores_path))
            if core_loaded.ndim != 2 or core_loaded.shape[1] != 3:
                raise ValueError("cores-file must contain an array of shape (N,3)")
            core_src = core_loaded.astype(np.float64, copy=False)
            logger.info("Loaded %d core points from %s", len(core_src), cores_path)
            return core_src
        except Exception as e:
            logger.warning("Failed to load cores from %s: %s; falling back to selection", cores_path, e)
            core_src = None

    # Select core points
    use_parallel_streaming = (
        streaming_enabled
        and getattr(cfg.parallel, 'enabled', False)
    )

    if use_parallel_streaming:
        logger.info(
            "Per-tile M3C2 will select %.1f%% core points per tile (no global core selection needed)",
            cfg.detection.m3c2.core_points_percent or 10.0,
        )
        core_src = None
    elif streaming_core_files is not None:
        logger.info(
            "Selecting %s core points via streaming from %s files...",
            f"{max_core:,}", cfg.detection.m3c2.evaluation_source.upper(),
        )
        core_reader = LaspyStreamReader(
            [str(p) for p in streaming_core_files],
            ground_only=cfg.preprocessing.ground_only,
            classification_filter=cfg.preprocessing.classification_filter,
            chunk_points=cfg.outofcore.chunk_points,
        )
        core_src = core_reader.reservoir_sample(max_core, transform=data.local_transform)
        logger.info(
            "Selected %s core points from %s via streaming",
            f"{len(core_src):,}", cfg.detection.m3c2.evaluation_source.upper(),
        )
    else:
        if len(reference_points) > max_core:
            idx = np.random.choice(len(reference_points), max_core, replace=False)
            core_src = reference_points[idx]
        else:
            core_src = reference_points

    # Save if path was provided but did not exist
    if cores_path is not None and core_src is not None:
        try:
            cores_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(cores_path), core_src)
            logger.info("Saved %d core points to %s", len(core_src), cores_path)
        except Exception as e:
            logger.warning("Could not save cores to %s: %s", cores_path, e)

    return core_src


def _resolve_m3c2_params(
    cfg: AppConfig,
    data: PreparedData,
    *,
    reference_points: np.ndarray,
    streaming_inputs: dict | None = None,
) -> tuple[M3C2Params, str]:
    """Resolve M3C2 parameters from config (fixed or autotuned)."""
    m3c2_cfg = cfg.detection.m3c2

    if (
        getattr(m3c2_cfg, 'use_autotune', True) is False
        and getattr(m3c2_cfg, 'fixed', None) is not None
        and m3c2_cfg.fixed.radius is not None
    ):
        r = float(m3c2_cfg.fixed.radius)
        depth_factor = (
            float(m3c2_cfg.fixed.depth_factor)
            if m3c2_cfg.fixed.depth_factor is not None
            else float(m3c2_cfg.autotune.max_depth_factor)
        )
        normal_scale = (
            float(m3c2_cfg.fixed.normal_scale)
            if m3c2_cfg.fixed.normal_scale is not None
            else r
        )
        params = M3C2Params(
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
        return params, "fixed"

    # Autotuning
    at = m3c2_cfg.autotune
    use_header = getattr(at, 'source', 'header') == 'header'

    if streaming_inputs is not None:
        files_t1 = streaming_inputs["files_t1"]
        files_t2 = streaming_inputs["files_t2"]
    else:
        files_t1 = [str(p) for p in data.ds1.laz_files]
        files_t2 = [str(p) for p in data.ds2.laz_files]

    if use_header and files_t1:
        params = None
        source = "autotune_header"
        try:
            params = autotune_m3c2_params_from_headers(
                files_t1=files_t1,
                files_t2=files_t2,
                target_neighbors=at.target_neighbors,
                max_depth_factor=at.max_depth_factor,
                min_radius=at.min_radius,
                max_radius=at.max_radius,
            )
        except Exception as _e:
            logger.warning("Header-based autotune failed (%s); falling back to sample-based.", _e)
            source = "autotune_sample"

        if params is None:
            params = autotune_m3c2_params(
                reference_points,
                target_neighbors=at.target_neighbors,
                max_depth_factor=at.max_depth_factor,
                min_radius=at.min_radius,
                max_radius=at.max_radius,
            )
            source = "autotune_sample"
        return params, source

    params = autotune_m3c2_params(
        reference_points,
        target_neighbors=at.target_neighbors,
        max_depth_factor=at.max_depth_factor,
        min_radius=at.min_radius,
        max_radius=at.max_radius,
    )
    return params, "autotune_sample"


def _compute_streaming_m3c2(
    cfg,
    data,
    alignment,
    core_src,
    m3c2_params,
    args,
    *,
    streaming_inputs,
    points_t1: np.ndarray,
    points_t2: np.ndarray,
):
    """Compute streaming M3C2 with parallel/sequential/fallback paths."""
    files_t1 = streaming_inputs["files_t1"]
    files_t2 = streaming_inputs["files_t2"]
    transform_t2 = streaming_inputs["transform_t2"]

    use_parallel = getattr(cfg.parallel, 'enabled', False)
    mode = "parallel" if use_parallel else "sequential"
    logger.info("Using streaming M3C2 (%s, tiled)...", mode)

    try:
        if use_parallel:
            m3c2_res_stream = ChangeDetector.compute_m3c2_streaming_pertile_parallel(
                files_t1=files_t1,
                files_t2=files_t2,
                params=m3c2_params,
                core_points_percent=cfg.detection.m3c2.core_points_percent or 10.0,
                evaluation_source=cfg.detection.m3c2.evaluation_source,
                tile_size=cfg.outofcore.tile_size_m,
                halo=None,
                ground_only=cfg.preprocessing.ground_only,
                classification_filter=cfg.preprocessing.classification_filter,
                chunk_points=cfg.outofcore.chunk_points,
                transform_t2=transform_t2,
                n_workers=None,
                threads_per_worker=getattr(cfg.parallel, 'threads_per_worker', 1),
                local_transform=data.local_transform,
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
                transform_t2=transform_t2,
                local_transform=data.local_transform,
            )

        # Debug comparison
        if args.debug_m3c2_compare:
            _debug_m3c2_compare(
                m3c2_res_stream, core_src,
                points_t1, points_t2,
                m3c2_params,
            )

        return m3c2_res_stream

    except Exception as stream_err:
        logger.error("Streaming M3C2 failed: %s", stream_err)
        logger.info("Falling back to in-memory M3C2...")
        return ChangeDetector.compute_m3c2_original(
            core_points=core_src,
            cloud_t1=points_t1,
            cloud_t2=points_t2,
            params=m3c2_params,
        )


def _resolve_streaming_m3c2_inputs(cfg, data, alignment):
    """Resolve a reference-aware streaming M3C2 input plan.

    Returns ``None`` when streaming should be skipped in favor of in-memory
    computation, most notably when ``reference=t2`` but aligned T1 files are
    not available on disk.
    """
    can_use_streaming = (
        data.use_streaming
        and data.pc1_data is not None
        and 'file_paths' in data.pc1_data
        and data.pc2_data is not None
        and (
            ('aligned_file_paths' in data.pc2_data and data.pc2_data['aligned_file_paths'])
            or data.pc2_data.get('file_paths')
        )
    )
    if not can_use_streaming:
        return None

    if cfg.alignment.reference == "t2":
        files_t1 = data.pc1_data.get('aligned_file_paths') if data.pc1_data else None
        if not files_t1:
            logger.info(
                "Falling back to in-memory M3C2 because streaming M3C2 can only apply "
                "an on-the-fly transform to T2; reference=t2 requires aligned T1 files."
            )
            return None
        return {
            "files_t1": files_t1,
            "files_t2": data.pc2_data['file_paths'],
            "transform_t2": None,
        }

    has_aligned_t2 = (
        data.pc2_data
        and 'aligned_file_paths' in data.pc2_data
        and data.pc2_data['aligned_file_paths']
    )
    return {
        "files_t1": data.pc1_data['file_paths'],
        "files_t2": data.pc2_data.get('aligned_file_paths') or data.pc2_data['file_paths'],
        "transform_t2": None if has_aligned_t2 else alignment.transform_matrix,
    }


def _debug_m3c2_compare(m3c2_res_stream, core_src, points1, points2_aligned, m3c2_params):
    """Run in-memory M3C2 and compare with streaming results."""
    logger.info("Debug: also running in-memory M3C2 for comparison...")
    m3c2_res_mem = ChangeDetector.compute_m3c2_original(
        core_points=core_src,
        cloud_t1=points1,
        cloud_t2=points2_aligned,
        params=m3c2_params,
    )

    def _pearson(a, b):
        a = np.asarray(a, dtype=float).ravel()
        b = np.asarray(b, dtype=float).ravel()
        m = np.isfinite(a) & np.isfinite(b)
        if not np.any(m):
            return float('nan')
        a, b = a[m], b[m]
        n = a.size
        if n < 2:
            return float('nan')
        a = a - a.mean()
        b = b - b.mean()
        sa, sb = a.std(), b.std()
        denom = sa * sb * n
        num = float(a.dot(b))
        return float(num / denom) if denom > 0 else float('nan')

    def _summary(name: str, arr):
        d = np.asarray(arr, dtype=float).ravel()
        d = d[np.isfinite(d)]
        n = d.size
        if n == 0:
            logger.info("Debug M3C2 summary (%s): n=0", name)
            return
        pos = float(np.sum(d > 0)) / n * 100.0
        neg = float(np.sum(d < 0)) / n * 100.0
        med = float(np.median(d))
        p5 = float(np.percentile(d, 5))
        p95 = float(np.percentile(d, 95))
        a95 = float(np.percentile(np.abs(d), 95))
        logger.info(
            "Debug M3C2 summary (%s): n=%d, pos=%.1f%%, neg=%.1f%%, med=%.4f, p5=%.4f, p95=%.4f, abs_p95=%.4f",
            name, n, pos, neg, med, p5, p95, a95,
        )

    r_same = _pearson(m3c2_res_stream.distances, m3c2_res_mem.distances)
    r_flip = _pearson(m3c2_res_stream.distances, -np.asarray(m3c2_res_mem.distances))
    logger.info("Debug M3C2: corr(stream, inmem)=%.6f, corr(stream, -inmem)=%.6f", r_same, r_flip)

    try:
        from sklearn.neighbors import NearestNeighbors as _NN
        nn1 = _NN(n_neighbors=1, algorithm='kd_tree').fit(points1)
        nn2 = _NN(n_neighbors=1, algorithm='kd_tree').fit(points2_aligned)
        i1 = nn1.kneighbors(core_src, return_distance=False).ravel()
        i2 = nn2.kneighbors(core_src, return_distance=False).ravel()
        dz = points2_aligned[i2, 2] - points1[i1, 2]
        rz_stream = _pearson(m3c2_res_stream.distances, dz)
        rz_mem = _pearson(m3c2_res_mem.distances, dz)
        rz_mem_flip = _pearson(-np.asarray(m3c2_res_mem.distances), dz)
        logger.info(
            "Debug M3C2: corr(stream, dZ)=%.6f, corr(inmem, dZ)=%.6f, corr(-inmem, dZ)=%.6f",
            rz_stream, rz_mem, rz_mem_flip,
        )
    except Exception as _e:
        logger.warning("Debug M3C2: dZ proxy check skipped (%s)", _e)

    _summary("stream", m3c2_res_stream.distances)
    _summary("inmem", m3c2_res_mem.distances)


def _export_m3c2(cfg, data, m3c2_res, *, run_id: str | None = None):
    """Export M3C2 results as LAZ and/or GeoTIFF."""
    export_m3c2_pc = getattr(cfg.detection.m3c2, 'export_pc', True)
    export_m3c2_raster = getattr(cfg.detection.m3c2, 'export_raster', True)

    if not (export_m3c2_pc or export_m3c2_raster):
        return []

    output_paths = []

    try:
        # M3C2 uses area-scoped output directory
        export_dir = resolve_output_dir(cfg, data.selected_area.area_name, area_scoped=True)
        crs = detect_output_crs(cfg, str(data.ds1.laz_files[0]))

        area_prefix = data.selected_area.area_name
        clip_suffix = clipping_export_suffix(cfg)
        run_suffix = f"_{run_id}" if run_id else ""
        export_points, export_distances, export_uncertainty, export_significant = _clip_m3c2_export_to_geometry(
            cfg, data, m3c2_res,
        )

        if export_m3c2_pc:
            m3c2_laz = export_dir / f"m3c2_{area_prefix}_{data.t1}_{data.t2}{clip_suffix}{run_suffix}.laz"
            extra_dims = {}
            if export_uncertainty is not None:
                extra_dims['uncertainty'] = export_uncertainty
            if export_significant is not None:
                extra_dims['significant'] = export_significant
            export_points_to_laz(
                export_points, export_distances, str(m3c2_laz),
                crs=crs, extra_dims=extra_dims if extra_dims else None,
                source_laz_path=str(data.ds1.laz_files[0]),
                local_transform=data.local_transform,
            )
            logger.info("Exported M3C2 point cloud: %s", m3c2_laz)
            output_paths.append(str(m3c2_laz))

        if export_m3c2_raster:
            m3c2_tif = export_dir / f"m3c2_{area_prefix}_{data.t1}_{data.t2}{clip_suffix}{run_suffix}.tif"
            export_distances_to_geotiff(
                export_points, export_distances, str(m3c2_tif),
                cell_size=cfg.detection.dod.cell_size, crs=crs,
                local_transform=data.local_transform,
            )
            logger.info("Exported M3C2 raster: %s", m3c2_tif)
            output_paths.append(str(m3c2_tif))

            erosion_cfg = cfg.detection.m3c2.erosion_polygons
            if erosion_cfg.enabled:
                m3c2_geojson = export_dir / f"m3c2_erosion_polygons_{area_prefix}_{data.t1}_{data.t2}{clip_suffix}{run_suffix}.geojson"
                significant_values = None
                if export_significant is not None and len(export_significant) == len(export_distances):
                    significant_values = export_significant
                elif erosion_cfg.use_significance:
                    logger.warning("M3C2 significance unavailable or incomplete; exporting erosion polygons without significance mask")

                summary = export_erosion_polygons_geojson(
                    str(m3c2_tif),
                    str(m3c2_geojson),
                    peak_threshold_m=erosion_cfg.peak_threshold_m,
                    outline_threshold_m=erosion_cfg.outline_threshold_m,
                    use_significance=erosion_cfg.use_significance,
                    significant_points=export_points if significant_values is not None else None,
                    significant_values=significant_values,
                    closing_iterations=erosion_cfg.closing_iterations,
                    opening_iterations=erosion_cfg.opening_iterations,
                    structure_radius_cells=erosion_cfg.structure_radius_cells,
                    min_area_m2=erosion_cfg.min_area_m2,
                    min_cells=erosion_cfg.min_cells,
                    simplify_tolerance_m=erosion_cfg.simplify_tolerance_m,
                    local_transform=data.local_transform,
                )
                logger.info(
                    "Exported M3C2 erosion polygons: %s (%d polygons)",
                    m3c2_geojson, summary["polygon_count"],
                )
                output_paths.append(str(m3c2_geojson))

        return output_paths

    except Exception as export_err:
        logger.error("M3C2 export failed: %s", export_err)
        return output_paths


def _clip_m3c2_export_to_geometry(cfg, data, m3c2_res):
    """Filter M3C2 exports to the exact clipping polygon."""
    points = m3c2_res.core_points
    distances = m3c2_res.distances
    uncertainty = m3c2_res.uncertainty
    significant = m3c2_res.significant

    clipper = resolve_clipper(cfg, data.local_transform)
    if clipper is None or len(points) == 0:
        return points, distances, uncertainty, significant

    clipped_points, mask = clipper.clip(points, return_mask=True)
    clipped_distances = np.asarray(distances)[mask]
    clipped_uncertainty = np.asarray(uncertainty)[mask] if uncertainty is not None else None
    clipped_significant = np.asarray(significant)[mask] if significant is not None else None
    logger.info(
        "Masked M3C2 export to clipping geometry: %d/%d core points kept",
        len(clipped_points), len(points),
    )
    return clipped_points, clipped_distances, clipped_uncertainty, clipped_significant


def _build_m3c2_summary(cfg, params, params_source, max_core, m3c2_res, output_paths):
    distances = np.asarray(m3c2_res.distances)
    valid = np.isfinite(distances)
    total = int(distances.size)
    valid_count = int(valid.sum())
    at = cfg.detection.m3c2.autotune
    return {
        "enabled": True,
        "params_source": params_source,
        "target_neighbors": getattr(at, "target_neighbors", None),
        "core_points_percent": cfg.detection.m3c2.core_points_percent,
        "evaluation_source": cfg.detection.m3c2.evaluation_source,
        "core_points_requested": int(max_core),
        "core_points_result": int(len(m3c2_res.core_points)),
        "valid_distances": valid_count,
        "valid_percent": (valid_count / total * 100.0) if total else 0.0,
        "params": {
            "projection_scale": params.projection_scale,
            "cylinder_radius": params.cylinder_radius,
            "normal_scale": params.normal_scale,
            "max_depth": params.max_depth,
            "min_neighbors": params.min_neighbors,
            "confidence": params.confidence,
        },
        "output_paths": output_paths,
    }
