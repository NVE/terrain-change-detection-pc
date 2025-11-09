"""
Spatial Alignment Module

This module provides tools for aligning multi-temporal point cloud datasets
using the ICP (Iterative Closest Point) algorithm, with support for both
in-memory and streaming/out-of-core processing.
"""

from .fine_registration import ICPRegistration
from .coarse_registration import CoarseRegistration
from .streaming_alignment import (
    apply_transform_to_files,
    save_transform_matrix,
    load_transform_matrix,
)

__all__ = [
    "ICPRegistration",
    "CoarseRegistration",
    "apply_transform_to_files",
    "save_transform_matrix",
    "load_transform_matrix",
]
