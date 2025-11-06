"""
Acceleration Module

This module provides performance optimization infrastructure including:
- Out-of-core tiling for large datasets
- GPU acceleration utilities (future)
- JIT-compiled kernels (future)
- Parallel processing utilities (future)
"""

from .tiling import (
    Bounds2D,
    GridAccumulator,
    Tile,
    Tiler,
    MosaicAccumulator,
    LaspyStreamReader,
    union_bounds,
)

__all__ = [
    "Bounds2D",
    "GridAccumulator",
    "Tile",
    "Tiler",
    "MosaicAccumulator",
    "LaspyStreamReader",
    "union_bounds",
]
