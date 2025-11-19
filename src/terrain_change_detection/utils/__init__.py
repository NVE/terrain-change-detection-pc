"""
Utility Functions Module

This module provides common utility functions used across the terrain change detection project.
- File I/O operations
- Logging and progress tracking
- Point cloud filtering utilities
"""

from .logging import setup_logger
from .point_cloud_filters import (
    create_classification_mask,
    apply_classification_filter,
    get_filter_statistics,
)

__all__ = [
    "setup_logger",
    "create_classification_mask",
    "apply_classification_filter",
    "get_filter_statistics",
]