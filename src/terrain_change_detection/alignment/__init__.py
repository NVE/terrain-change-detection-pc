"""
Spatial Alignment Module

This module provides tools for aligning multi-temporal point cloud datasets
using the ICP (Iterative Closest Point) algorithm.
"""

from .fine_registration import ICPRegistration

__all__ = [
    "ICPRegistration"
]