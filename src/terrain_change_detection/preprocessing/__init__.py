"""
Point Cloud Data Preprocessing Module

This module contains functions for preprocessing multi-temporal point cloud data.
It includes methods for:
- Data loading and validation
- Data quality assessment
- Data exploration and discovery
- Support for multiple data sources (hoydedata.no and drone scanning)
- Area clipping for focused analysis on regions of interest
"""

from .loader import PointCloudLoader
from .data_discovery import DataDiscovery, BatchLoader, DatasetInfo, AreaInfo
from .clipping import (
    AreaClipper,
    clip_point_cloud_files,
    check_shapely_available,
    check_fiona_available,
)

__all__ = [
    "PointCloudLoader",
    "DataDiscovery",
    "BatchLoader",
    "DatasetInfo",
    "AreaInfo",
    "AreaClipper",
    "clip_point_cloud_files",
    "check_shapely_available",
    "check_fiona_available",
]