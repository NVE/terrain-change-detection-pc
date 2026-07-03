"""
Data discovery and loading for the terrain change detection workflow.

Handles area scanning, time-period selection, year filtering, and both
streaming (out-of-core) and in-memory dataset loading paths.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from terrain_change_detection.acceleration import LaspyStreamReader
from terrain_change_detection.preprocessing.data_discovery import (
    BatchLoader,
    DataDiscovery,
)
from terrain_change_detection.preprocessing.loader import PointCloudLoader
from terrain_change_detection.utils.config import AppConfig
from terrain_change_detection.utils.coordinate_transform import LocalCoordinateTransform

from .types import DiscoveryResult, PreparedData, WorkflowAbort

logger = logging.getLogger(__name__)


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


def discover(
    cfg: AppConfig,
    *,
    area_name: str | None = None,
    selected_years: list[int] | None = None,
) -> DiscoveryResult:
    """Run data discovery, area/year selection — no I/O-heavy loading.

    This is the cheap first half of the workflow data-preparation phase.
    Call :func:`load_data` afterwards to perform the actual point-cloud
    loading with the correct local transform.

    Args:
        cfg: Application configuration.
        area_name: Explicit area name to process (or ``None`` for auto-select).
        selected_years: Years to filter time periods (or ``None``).

    Returns:
        A :class:`DiscoveryResult` with the selected area and datasets.

    Raises:
        WorkflowAbort: If no suitable area or time periods are found.
    """
    base_dir = Path(cfg.paths.base_dir)
    if not base_dir.exists():
        raise WorkflowAbort(f"Base directory {base_dir} does not exist.")

    # ----------------------------------------------------------------
    # Discovery
    # ----------------------------------------------------------------
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
        include_classification_stats=False,
    )
    areas = data_discovery.scan_areas(user_area_name=area_name)

    if not areas:
        msg = "No area directories found in the base directory.\n"
        msg += f"Data source type: {cfg.discovery.source_type}\n"
        if cfg.discovery.source_type == 'hoydedata':
            msg += f"Expected structure: {base_dir}/<area>/<time_period>/{cfg.discovery.data_dir_name}/*.laz\n"
            msg += "If your data doesn't have a 'data' subdirectory, set source_type: drone in config"
        else:
            msg += f"Expected structure: {base_dir}/<area>/<time_period>/*.laz\n"
            msg += "If your data has a 'data' subdirectory, set source_type: hoydedata in config"
        raise WorkflowAbort(msg)

    # ----------------------------------------------------------------
    # Area selection
    # ----------------------------------------------------------------
    if area_name:
        if area_name in areas:
            selected_area = areas[area_name]
        else:
            raise WorkflowAbort(
                f"Specified area '{area_name}' not found. Available: {list(areas.keys())}"
            )
    else:
        selected_area = None
        for name, info in areas.items():
            if len(info.time_periods) >= 2:
                selected_area = info
                break

    if not selected_area:
        details = "\n".join(
            f"  - {name}: {len(info.time_periods)} time period(s) -> {info.time_periods}"
            for name, info in areas.items()
        )
        raise WorkflowAbort(
            "Could not find an area with at least two time periods.\n"
            f"Found {len(areas)} area(s):\n{details}\n"
            "Change detection requires at least 2 time periods per area."
        )

    # ----------------------------------------------------------------
    # Year filtering
    # ----------------------------------------------------------------
    if selected_years is not None and len(selected_years) > 1:
        if len(selected_years) > 2:
            logger.warning("More than two selected years provided; only the first two will be used.")

        filtered = [
            tp for tp in selected_area.time_periods
            if any(str(year) in tp for year in selected_years)
        ]

        if len(filtered) < 2:
            raise WorkflowAbort(
                f"After filtering, less than two time periods remain for area "
                f"'{selected_area.area_name}'. Filtered: {filtered}"
            )

        logger.info("Filtered time periods for area '%s': %s", selected_area.area_name, filtered)
        t1, t2 = filtered[:2]
    else:
        t1, t2 = selected_area.time_periods[:2]
        logger.info("Selected time periods for area '%s': %s, %s", selected_area.area_name, t1, t2)

    ds1 = selected_area.datasets[t1]
    ds2 = selected_area.datasets[t2]

    # ----------------------------------------------------------------
    # Determine streaming vs in-memory
    # ----------------------------------------------------------------
    use_streaming = (
        getattr(cfg, 'outofcore', None) is not None
        and cfg.outofcore.enabled
        and cfg.outofcore.streaming_mode
        and len(ds1.laz_files) > 0
        and len(ds2.laz_files) > 0
    )

    return DiscoveryResult(
        selected_area=selected_area,
        t1=t1,
        t2=t2,
        ds1=ds1,
        ds2=ds2,
        use_streaming=use_streaming,
    )


def load_data(
    cfg: AppConfig,
    discovery: DiscoveryResult,
    rng: np.random.Generator,
    local_transform: LocalCoordinateTransform | None,
) -> PreparedData:
    """Load point-cloud data for the discovered area and time periods.

    This is the I/O-heavy second half — call :func:`discover` first
    to obtain a :class:`DiscoveryResult`.

    Args:
        cfg: Application configuration.
        discovery: Result from :func:`discover`.
        rng: Seeded NumPy RNG for reproducible subsampling.
        local_transform: Active local coordinate transform (or ``None``).

    Returns:
        A fully populated :class:`PreparedData` instance.
    """
    logger.info("=== STEP 1: Data Preparation ===")
    logger.info("Selected area: %s", discovery.selected_area.area_name)
    logger.info("Time period 1: %s (%d files)", discovery.t1, len(discovery.ds1.laz_files))
    logger.info("Time period 2: %s (%d files)", discovery.t2, len(discovery.ds2.laz_files))

    loader = PointCloudLoader(
        ground_only=cfg.preprocessing.ground_only,
        classification_filter=cfg.preprocessing.classification_filter,
    )

    pc1_data = None
    pc2_data = None

    if discovery.use_streaming:
        points1, points2, pc1_data, pc2_data = _load_streaming(
            cfg, loader, discovery.ds1, discovery.ds2,
            discovery.t1, discovery.t2, local_transform, rng,
        )
    else:
        points1, points2, pc1_data, pc2_data = _load_inmemory(
            cfg, loader, discovery.ds1, discovery.ds2,
            discovery.t1, discovery.t2, local_transform,
        )

    return PreparedData(
        selected_area=discovery.selected_area,
        t1=discovery.t1,
        t2=discovery.t2,
        ds1=discovery.ds1,
        ds2=discovery.ds2,
        points1=points1,
        points2=points2,
        pc1_data=pc1_data,
        pc2_data=pc2_data,
        use_streaming=discovery.use_streaming,
        local_transform=local_transform,
        clip_bounds=None,
    )


def discover_and_load(
    cfg: AppConfig,
    rng: np.random.Generator,
    local_transform: LocalCoordinateTransform | None,
    *,
    area_name: str | None = None,
    selected_years: list[int] | None = None,
) -> PreparedData:
    """Convenience wrapper: :func:`discover` + :func:`load_data` in one call.

    Prefer calling :func:`discover` and :func:`load_data` separately when
    you need to set up the local coordinate transform between the two steps.
    """
    discovery = discover(cfg, area_name=area_name, selected_years=selected_years)
    return load_data(cfg, discovery, rng, local_transform)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_streaming(
    cfg: AppConfig,
    loader: PointCloudLoader,
    ds1, ds2,
    t1: str, t2: str,
    local_transform: LocalCoordinateTransform | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Load data in streaming/out-of-core mode."""
    logger.info("--- Step 1: Preparing datasets for streaming/out-of-core processing ---")
    batch_loader = BatchLoader(loader=loader, streaming_mode=True)

    pc1_data = batch_loader.load_dataset(ds1, streaming=True)
    pc2_data = batch_loader.load_dataset(ds2, streaming=True)

    _seed = int(cfg.alignment.random_seed)

    # Log metadata
    m1 = pc1_data['metadata']
    m2 = pc2_data['metadata']
    _log_streaming_dataset_summary(t1, pc1_data)
    _log_streaming_dataset_summary(t2, pc2_data)

    # Subsample counts
    est_ground1 = int(m1.get('total_points_ground') or m1.get('total_points_all') or 0)
    est_ground2 = int(m2.get('total_points_ground') or m2.get('total_points_all') or 0)
    n_per_ds1 = resolve_subsample_count(max(est_ground1, 1), cfg.alignment)
    n_per_ds2 = resolve_subsample_count(max(est_ground2, 1), cfg.alignment)
    logger.info("Loading subsampled data for alignment (T1->%d, T2->%d)...", n_per_ds1, n_per_ds2)

    # Overlap-aware bounding box
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

    # T1 subsample
    reader1 = LaspyStreamReader(
        [str(p) for p in ds1.laz_files],
        ground_only=cfg.preprocessing.ground_only,
        classification_filter=cfg.preprocessing.classification_filter,
        chunk_points=cfg.outofcore.chunk_points,
    )
    points1 = reader1.reservoir_sample(
        n_per_ds1, transform=local_transform, bbox=overlap_bbox,
        seed=_seed,
    )

    # T2 subsample (derived seed for independence)
    reader2 = LaspyStreamReader(
        [str(p) for p in ds2.laz_files],
        ground_only=cfg.preprocessing.ground_only,
        classification_filter=cfg.preprocessing.classification_filter,
        chunk_points=cfg.outofcore.chunk_points,
    )
    points2 = reader2.reservoir_sample(
        n_per_ds2, transform=local_transform, bbox=overlap_bbox,
        seed=_seed + 1,
    )

    logger.info("Loaded %d sample points from T1 for alignment", len(points1))
    logger.info("Loaded %d sample points from T2 for alignment", len(points2))

    return points1, points2, pc1_data, pc2_data


def _log_streaming_dataset_summary(time_period: str, pc_data: dict) -> None:
    """Log streaming dataset stats without implying unknown ground counts are zero."""
    metadata = pc_data['metadata']
    total_ground = metadata.get('total_points_ground')
    total_all = float(metadata.get('total_points_all') or 0)
    ground_percentage = metadata.get('ground_percentage')

    if total_ground:
        logger.info(
            "Dataset (%s): %d files, ~%.0f ground / %.0f total (%.1f%%)",
            time_period,
            len(pc_data['file_paths']),
            float(total_ground),
            total_all,
            float(ground_percentage) if ground_percentage is not None else float('nan'),
        )
    else:
        logger.info(
            "Dataset (%s): %d files, %.0f total points (ground count not precomputed)",
            time_period,
            len(pc_data['file_paths']),
            total_all,
        )


def _load_inmemory(
    cfg: AppConfig,
    loader: PointCloudLoader,
    ds1, ds2,
    t1: str, t2: str,
    local_transform: LocalCoordinateTransform | None,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Load data fully in memory."""
    logger.info("--- Step 1: Loading point cloud data (in-memory) ---")
    batch_loader = BatchLoader(loader=loader)

    if len(ds1.laz_files) > 1:
        logger.info("Batch loading %d files for time period %s...", len(ds1.laz_files), t1)
        pc1_data = batch_loader.load_dataset(ds1, transform=local_transform)
    else:
        logger.info("Loading single file for time period %s...", t1)
        pc1_data = batch_loader.loader.load(str(ds1.laz_files[0]), transform=local_transform)

    if len(ds2.laz_files) > 1:
        logger.info("Batch loading %d files for time period %s...", len(ds2.laz_files), t2)
        pc2_data = batch_loader.load_dataset(ds2, transform=local_transform)
    else:
        logger.info("Loading single file for time period %s...", t2)
        pc2_data = batch_loader.loader.load(str(ds2.laz_files[0]), transform=local_transform)

    logger.info("Dataset 1 (%s): %d points", t1, pc1_data['points'].shape[0])
    logger.info("Dataset 2 (%s): %d points", t2, pc2_data['points'].shape[0])

    return pc1_data['points'], pc2_data['points'], pc1_data, pc2_data
