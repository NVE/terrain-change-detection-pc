"""
Spatial Alignment Module

This module provides tools for aligning multi-temporal point cloud datasets
using the ICP (Iterative Closest Point) algorithm, with support for both
in-memory and streaming/out-of-core processing.
"""

from .fine_registration import (
    ICPRegistration,
    compute_overlap_mask,
    estimate_alignment_covariance,
)
from .coarse_registration import CoarseRegistration
from .streaming_alignment import (
    apply_transform_to_files,
    save_transform_matrix,
    load_transform_matrix,
)

# Open3D ICP is optional – only available when open3d is installed
try:
    from .open3d_icp import Open3DICP
except ImportError:
    Open3DICP = None  # type: ignore[assignment,misc]

__all__ = [
    "ICPRegistration",
    "Open3DICP",
    "CoarseRegistration",
    "compute_overlap_mask",
    "estimate_alignment_covariance",
    "apply_transform_to_files",
    "save_transform_matrix",
    "load_transform_matrix",
]
