"""
Area clipping logic for the terrain change detection workflow.

Validates clipping configuration, loads boundary files, and clips point clouds
using :class:`~terrain_change_detection.preprocessing.clipping.AreaClipper`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from terrain_change_detection.preprocessing.clipping import (
    AreaClipper,
    check_shapely_available,
)
from terrain_change_detection.utils.config import AppConfig
from terrain_change_detection.utils.coordinate_transform import LocalCoordinateTransform

from .types import WorkflowAbort

logger = logging.getLogger(__name__)


def apply_clipping(
    cfg: AppConfig,
    points1: np.ndarray,
    points2: np.ndarray,
    local_transform: LocalCoordinateTransform | None,
    *,
    project_root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple | None]:
    """Apply optional area clipping to both point clouds.

    Args:
        cfg: Application configuration.
        points1: T1 point cloud (N×3).
        points2: T2 point cloud (N×3).
        local_transform: Active local transform (or ``None``).
        project_root: Repository root for resolving relative boundary paths.
            Defaults to the parent of the ``scripts/`` directory.

    Returns:
        ``(points1_clipped, points2_clipped, clip_bounds)`` — the clipped
        point clouds and the clipper bounds tuple (or ``None`` if clipping
        was not enabled).

    Raises:
        WorkflowAbort: If clipping is enabled but prerequisites are missing
            or clipping results in empty point clouds.
    """
    clipping_cfg = getattr(cfg, 'clipping', None)

    if clipping_cfg is None or not clipping_cfg.enabled:
        return points1, points2, None

    if not check_shapely_available():
        raise WorkflowAbort(
            "Clipping is enabled but shapely is not installed. Install with: uv add shapely"
        )

    if not clipping_cfg.boundary_file:
        raise WorkflowAbort(
            "Clipping is enabled but no boundary_file is specified in config."
        )

    boundary_path = Path(clipping_cfg.boundary_file)
    if not boundary_path.is_absolute():
        if project_root is None:
            project_root = Path(__file__).resolve().parents[3]
        boundary_path = project_root / boundary_path

    if not boundary_path.exists():
        raise WorkflowAbort(f"Clipping boundary file not found: {boundary_path}")

    logger.info("--- Applying area clipping ---")

    try:
        clipper = AreaClipper.from_file(
            str(boundary_path),
            feature_name=clipping_cfg.feature_name,
        )

        # Transform clipper to local coordinates if needed
        if local_transform is not None:
            clipper = clipper.transform_to_local(local_transform)

        clip_bounds = clipper.bounds

        original_count_1 = len(points1)
        original_count_2 = len(points2)

        points1 = clipper.clip(points1)
        points2 = clipper.clip(points2)

        pct1 = 100.0 * len(points1) / original_count_1 if original_count_1 > 0 else 0
        pct2 = 100.0 * len(points2) / original_count_2 if original_count_2 > 0 else 0
        logger.info(
            "Clipping complete: T1 %s pts (%.1f%%), T2 %s pts (%.1f%%)",
            f"{len(points1):,}", pct1,
            f"{len(points2):,}", pct2,
        )

        if len(points1) == 0 or len(points2) == 0:
            raise WorkflowAbort(
                "Clipping resulted in empty point clouds. Check your boundary file."
            )

        return points1, points2, clip_bounds

    except WorkflowAbort:
        raise
    except Exception as e:
        raise WorkflowAbort(f"Clipping failed: {e}") from e
