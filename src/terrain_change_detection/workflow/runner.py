"""
Workflow orchestrator for the terrain change detection workflow.

Calls each phase in order and catches :class:`WorkflowAbort` to preserve the
current "log error and stop" behavior without changing exit semantics.
"""

from __future__ import annotations

import argparse
import logging
from time import perf_counter

from terrain_change_detection.utils.config import AppConfig
from terrain_change_detection.visualization.point_cloud import PointCloudVisualizer

from .alignment import run_alignment
from .bootstrap import setup_runtime
from .clipping import apply_clipping
from .coordinate_setup import setup_local_transform
from .data_loading import discover, load_data
from .detection_c2c import run_c2c
from .detection_dod import run_dod
from .detection_m3c2 import run_m3c2
from .export_helpers import (
    export_dem_rasters,
    make_run_id,
    reset_crs_cache,
    resolve_output_dir,
    write_run_inputs,
)
from .types import WorkflowAbort, WorkflowResult
from .visualization_helpers import to_global_for_vis

logger_module = logging.getLogger(__name__)


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

        # Setup local coordinate transform from T1 bounds
        local_transform = setup_local_transform(cfg, discovery.ds1.bounds)

        # Load point-cloud data (expensive I/O — done once with transform)
        data = load_data(cfg, discovery, rng, local_transform)

        # ----------------------------------------------------------------
        # Optional: Area Clipping
        # ----------------------------------------------------------------
        points1, points2, clip_bounds = apply_clipping(
            cfg, data.points1, data.points2, data.local_transform,
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
                suffix="icp",
            )

        # ----------------------------------------------------------------
        # Step 3: Change Detection
        # ----------------------------------------------------------------
        logger.info("=== STEP 3: Change Detection ===")

        run_dod(cfg, data, alignment, visualizer=visualizer, show_plots=show_plots)
        run_c2c(cfg, data, alignment, visualizer=visualizer, show_plots=show_plots)
        run_m3c2(cfg, data, alignment, args, run_id=run_id, visualizer=visualizer, show_plots=show_plots)

        # ----------------------------------------------------------------
        # Write run manifest
        # ----------------------------------------------------------------
        export_dir = resolve_output_dir(cfg, data.selected_area.area_name)
        write_run_inputs(
            export_dir, args, cfg,
            run_id=run_id,
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
