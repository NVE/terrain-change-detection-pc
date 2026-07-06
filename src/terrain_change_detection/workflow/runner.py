"""
Workflow orchestrator for the terrain change detection workflow.

Calls each phase in order and catches :class:`WorkflowAbort` to preserve the
current "log error and stop" behavior without changing exit semantics.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from pprint import pformat
from time import perf_counter

from terrain_change_detection.utils.config import AppConfig
from terrain_change_detection.visualization.point_cloud import PointCloudVisualizer
from terrain_change_detection.preprocessing.data_discovery import (
    dataset_size_bytes,
    filter_dataset_by_bounds,
)

from .alignment import run_alignment
from .bootstrap import setup_runtime
from .clipping import apply_clipping, resolve_clipping_bounds, split_clipping_features
from .coordinate_setup import setup_local_transform
from .data_loading import discover, load_data
from .detection_c2c import run_c2c
from .detection_dod import run_dod
from .detection_m3c2 import run_m3c2
from .export_helpers import (
    export_dem_rasters,
    make_run_id,
    reset_crs_cache,
    resolve_workflow_crs,
    resolve_output_dir,
    write_run_inputs,
)
from .types import WorkflowAbort, WorkflowResult
from .visualization_helpers import to_global_for_vis

logger_module = logging.getLogger(__name__)


def _prefilter_discovery_by_clipping(
    cfg: AppConfig,
    discovery,
    *,
    workflow_crs: str | None = None,
) -> None:
    """Reduce discovered tile lists using clipping bounds before point loading."""
    clip_bounds = resolve_clipping_bounds(cfg, workflow_crs=workflow_crs)
    if clip_bounds is None:
        return

    for attr_name, time_period in (("ds1", discovery.t1), ("ds2", discovery.t2)):
        original = getattr(discovery, attr_name)
        original_count = len(original.laz_files)
        original_size = dataset_size_bytes(original)
        filtered = filter_dataset_by_bounds(original, clip_bounds)
        filtered_count = len(filtered.laz_files)
        filtered_size = dataset_size_bytes(filtered)

        if filtered_count == 0:
            raise WorkflowAbort(
                f"Clipping bounds prefilter removed all tiles for {time_period}. "
                "Check clipping boundary CRS and dataset extent."
            )

        setattr(discovery, attr_name, filtered)
        discovery.selected_area.datasets[time_period] = filtered
        logger_module.info(
            "Tile prefilter (%s): %d -> %d files, %.2f -> %.2f GiB",
            time_period,
            original_count,
            filtered_count,
            original_size / (1024 ** 3),
            filtered_size / (1024 ** 3),
        )


def run(
    args: argparse.Namespace,
    cfg: AppConfig,
    cli_overrides: list[str],
) -> WorkflowResult | None:
    """Execute the full terrain change detection workflow.

    This is the main orchestrator that is called by :func:`cli.main` after
    CLI parsing.  Each phase is delegated to a dedicated module.

    Args:
        args: Parsed CLI arguments.
        cfg: Fully resolved application configuration.
        cli_overrides: Dot-path overrides applied to the config.

    Returns:
        A :class:`WorkflowResult` summary, or ``None`` if the workflow
        was aborted.
    """
    if getattr(cfg.clipping, 'split_features', False):
        return _run_split_features(args, cfg, cli_overrides)

    runtime_start = perf_counter()
    reset_crs_cache()

    try:
        # ----------------------------------------------------------------
        # Bootstrap
        # ----------------------------------------------------------------
        logger, rng = setup_runtime(cfg)
        run_id = make_run_id()
        logger.info("run_id: %s", run_id)

        show_plots: bool = args.show_plots
        selected_years: list[int] | None = args.years
        export_dems: bool = args.save_dems

        logger.info("selected_years: %s", selected_years)
        # ----------------------------------------------------------------
        # Step 1: Data Preparation
        # ----------------------------------------------------------------
        # Discovery is cheap (file scanning only).  We run it first to get
        # ds1 bounds, then set up the coordinate transform, then do the
        # expensive loading step exactly once with the correct transform.
        discovery = discover(
            cfg,
            area_name=args.area_name,
            selected_years=selected_years,
        )

        workflow_crs = resolve_workflow_crs(cfg, discovery.ds1.laz_files[0], discovery.ds2.laz_files[0])

        _prefilter_discovery_by_clipping(cfg, discovery, workflow_crs=workflow_crs)

        # Setup local coordinate transform from T1 bounds
        local_transform = setup_local_transform(cfg, discovery.ds1.bounds)

        # Load point-cloud data (expensive I/O — done once with transform)
        data = load_data(cfg, discovery, rng, local_transform)

        # ----------------------------------------------------------------
        # Optional: Area Clipping
        # ----------------------------------------------------------------
        points1, points2, clip_bounds = apply_clipping(
            cfg, data.points1, data.points2, data.local_transform,
            workflow_crs=workflow_crs,
        )
        data.points1 = points1
        data.points2 = points2
        data.clip_bounds = clip_bounds

        # ----------------------------------------------------------------
        # Visualization: original clouds
        # ----------------------------------------------------------------
        visualizer = PointCloudVisualizer(backend=cfg.visualization.backend)

        vis_points1 = to_global_for_vis(data.points1, data.local_transform)
        vis_points2 = to_global_for_vis(data.points2, data.local_transform)
        if show_plots:
            logger.info("--- Visualizing original point clouds ---")
            visualizer.visualize_clouds(
                point_clouds=[vis_points1, vis_points2],
                names=[f"PC from {data.t1}", f"PC from {data.t2}"],
                sample_size=cfg.visualization.sample_size,
            )

        if export_dems:
            export_dem_rasters(
                cfg=cfg,
                time_labels=[data.t1, data.t2],
                point_clouds=[vis_points1, vis_points2],
                area_name=data.selected_area.area_name,
                laz_file=str(data.ds1.laz_files[0]),
                laz_file_epoch2=str(data.ds2.laz_files[0]),
                suffix="raw",
            )

        # ----------------------------------------------------------------
        # Step 2: Spatial Alignment
        # ----------------------------------------------------------------
        alignment = run_alignment(cfg, data, rng)

        # Visualization: aligned clouds
        vis_p1_aligned = to_global_for_vis(alignment.points1_aligned, data.local_transform)
        vis_p2_aligned = to_global_for_vis(alignment.points2_aligned, data.local_transform)
        if show_plots:
            logger.info("--- Visualizing aligned point clouds ---")
            visualizer.visualize_clouds(
                point_clouds=[vis_p1_aligned, vis_p2_aligned],
                names=[f"PC from {data.t1} (Target)", f"PC from {data.t2} (Aligned)"],
                sample_size=cfg.visualization.sample_size,
            )

        if export_dems:
            export_dem_rasters(
                cfg=cfg,
                time_labels=[data.t1, data.t2],
                point_clouds=[vis_p1_aligned, vis_p2_aligned],
                area_name=data.selected_area.area_name,
                laz_file=str(data.ds1.laz_files[0]),
                laz_file_epoch2=str(data.ds2.laz_files[0]),
                suffix="icp",
            )

        # ----------------------------------------------------------------
        # Step 3: Change Detection
        # ----------------------------------------------------------------
        logger.info("=== STEP 3: Change Detection ===")

        run_dod(cfg, data, alignment, run_id=run_id, visualizer=visualizer, show_plots=show_plots)
        run_c2c(cfg, data, alignment, run_id=run_id, visualizer=visualizer, show_plots=show_plots)
        m3c2_summary = run_m3c2(
            cfg, data, alignment, args,
            run_id=run_id,
            visualizer=visualizer,
            show_plots=show_plots,
        )

        evaluation_summary = _build_evaluation_summary(cfg, data, alignment, run_id, m3c2_summary)
        logger.info("Run evaluation summary:\n%s", pformat(evaluation_summary, sort_dicts=False))

        # ----------------------------------------------------------------
        # Write run manifest
        # ----------------------------------------------------------------
        export_dir = resolve_output_dir(cfg, data.selected_area.area_name)
        write_run_inputs(
            export_dir, args, cfg,
            run_id=run_id,
            evaluation_summary=evaluation_summary,
            config_files=list(args.config or []),
            cli_overrides=cli_overrides,
        )

        runtime_end = perf_counter()
        logger.info("Total workflow runtime: %.2f seconds", runtime_end - runtime_start)

        return WorkflowResult(
            selected_area=data.selected_area.area_name,
            epochs=(data.t1, data.t2),
            streaming_used=data.use_streaming,
            alignment_enabled=getattr(cfg.alignment, 'enabled', True),
            alignment_error=alignment.alignment_error,
        )

    except WorkflowAbort as e:
        logger_module.log(e.level, "%s", e)
        return None
    except Exception as e:
        logger_module.error("Change detection workflow failed: %s", e)
        return None


def _run_split_features(
    args: argparse.Namespace,
    cfg: AppConfig,
    cli_overrides: list[str],
) -> WorkflowResult | None:
    """Run workflow once per GeoJSON feature when split_features is enabled."""
    split_root = _split_features_dir(cfg, args.area_name)

    try:
        features = split_clipping_features(cfg, split_root)
    except WorkflowAbort as e:
        logger_module.log(e.level, "%s", e)
        return None

    logger_module.info("Running split_features workflow for %d clipping features", len(features))

    last_result = None
    for feature in features:
        logger_module.info(
            "=== split_features %03d/%03d: %s ===",
            feature.index,
            len(features),
            feature.label,
        )
        feature_cfg = cfg.model_copy(deep=True)
        feature_cfg.clipping.split_features = False
        feature_cfg.clipping.boundary_file = str(feature.boundary_file)
        feature_cfg.clipping.feature_name = None

        result = run(args, feature_cfg, cli_overrides)
        if result is not None:
            last_result = result

    return last_result


def _split_features_dir(cfg: AppConfig, area_name: str | None) -> Path:
    """Resolve persistent directory for per-feature clipping GeoJSONs."""
    split_root = Path(cfg.paths.output_dir or (Path(cfg.paths.base_dir) / "output"))
    if area_name:
        split_root = split_root / area_name
    return split_root / "_split_features"


def _build_evaluation_summary(
    cfg: AppConfig,
    data,
    alignment,
    run_id: str,
    m3c2_summary: dict | None,
) -> dict:
    """Collect concise run metrics for logs/manifest."""
    return {
        "run_id": run_id,
        "area": data.selected_area.area_name,
        "epochs": {"t1": data.t1, "t2": data.t2},
        "counts": {
            "points_t1": int(len(data.points1)),
            "points_t2": int(len(data.points2)),
        },
        "alignment": {
            "enabled": bool(getattr(cfg.alignment, "enabled", True)),
            "reference": cfg.alignment.reference,
            "backend": cfg.alignment.icp_backend,
            "coarse_enabled": bool(getattr(cfg.alignment.coarse, "enabled", False)),
            "coarse_method": getattr(cfg.alignment.coarse, "method", None),
            "max_iterations": cfg.alignment.max_iterations,
            "aligned_epoch": alignment.aligned_epoch,
            "rmse": alignment.alignment_error,
            "transform_matrix": alignment.transform_matrix.tolist(),
        },
        "m3c2": m3c2_summary or {"enabled": False},
    }
