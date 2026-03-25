"""
Visualization helpers for the terrain change detection workflow.

Provides coordinate-system conversion for visualization (local → global) and
conditional plotting wrappers.
"""

from __future__ import annotations

import numpy as np

from terrain_change_detection.utils.coordinate_transform import LocalCoordinateTransform


def to_global_for_vis(
    points: np.ndarray,
    local_transform: LocalCoordinateTransform | None,
) -> np.ndarray:
    """Convert *points* back to global coordinates for visualization.

    Users expect UTM coordinates in plots.  If *local_transform* is ``None``
    the points are returned unchanged.
    """
    if local_transform is not None:
        return local_transform.to_global(points)
    return points
