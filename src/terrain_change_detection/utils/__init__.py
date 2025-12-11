"""
Utility Functions Module

This module provides common utility functions used across the terrain change detection project.
- File I/O operations
- Logging and progress tracking
- Point cloud filtering utilities
- Export utilities for LAZ and GeoTIFF
- Coordinate transformation utilities
"""

from .logging import setup_logger
from .point_cloud_filters import (
    create_classification_mask,
    apply_classification_filter,
    get_filter_statistics,
)
from .export import (
    export_points_to_laz,
    export_dod_to_geotiff,
    export_distances_to_geotiff,
    detect_crs_from_laz,
)
from .coordinate_transform import LocalCoordinateTransform

__all__ = [
    "setup_logger",
    "create_classification_mask",
    "apply_classification_filter",
    "get_filter_statistics",
    "export_points_to_laz",
    "export_dod_to_geotiff",
    "export_distances_to_geotiff",
    "detect_crs_from_laz",
    "LocalCoordinateTransform",
]