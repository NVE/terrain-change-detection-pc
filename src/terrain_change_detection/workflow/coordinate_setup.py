"""
Local coordinate transform setup for the terrain change detection workflow.

Extracts the coordinate-transform initialization logic from the monolithic
``main()`` function.
"""

from __future__ import annotations

import logging

from terrain_change_detection.utils.config import AppConfig
from terrain_change_detection.utils.coordinate_transform import LocalCoordinateTransform

logger = logging.getLogger(__name__)


def setup_local_transform(
    cfg: AppConfig,
    ds1_bounds: dict | None,
) -> LocalCoordinateTransform | None:
    """Compute a local coordinate transform from T1 bounds (if enabled).

    Args:
        cfg: Application configuration.
        ds1_bounds: Bounding-box dict from the T1 dataset (keys:
            ``min_x``, ``min_y``, ``min_z``, ``max_x``, ``max_y``, ``max_z``).
            May be ``None`` if bounds are unavailable.

    Returns:
        A :class:`LocalCoordinateTransform` instance, or ``None`` if local
        coordinates are disabled or bounds are not available.
    """
    coord_cfg = getattr(cfg, "coordinates", None)
    use_local_coords = coord_cfg is not None and getattr(
        coord_cfg, "use_local_coordinates", True
    )

    if not use_local_coords or not ds1_bounds:
        return None

    origin_method = getattr(coord_cfg, "origin_method", "min_bounds")
    include_z = getattr(coord_cfg, "include_z_offset", False)

    if origin_method == "min_bounds":
        offset_z = ds1_bounds.get("min_z", 0.0) if include_z else 0.0
        local_transform = LocalCoordinateTransform.from_bounds(
            min_x=ds1_bounds["min_x"],
            min_y=ds1_bounds["min_y"],
            min_z=offset_z,
        )
    elif origin_method == "centroid":
        cx = (ds1_bounds["min_x"] + ds1_bounds["max_x"]) / 2
        cy = (ds1_bounds["min_y"] + ds1_bounds["max_y"]) / 2
        cz = ((ds1_bounds["min_z"] + ds1_bounds["max_z"]) / 2) if include_z else 0.0
        local_transform = LocalCoordinateTransform(
            offset_x=cx, offset_y=cy, offset_z=cz
        )
    else:
        # first_point not practical here, fall back to min_bounds
        offset_z = ds1_bounds.get("min_z", 0.0) if include_z else 0.0
        local_transform = LocalCoordinateTransform.from_bounds(
            min_x=ds1_bounds["min_x"],
            min_y=ds1_bounds["min_y"],
            min_z=offset_z,
        )

    logger.info(
        "Local coordinate transform: offset=(%.2f, %.2f, %.2f) using %s origin",
        local_transform.offset_x,
        local_transform.offset_y,
        local_transform.offset_z,
        origin_method,
    )

    return local_transform
