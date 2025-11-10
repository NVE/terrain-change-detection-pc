"""
Acceleration Module

This module provides performance optimization infrastructure including:
- Out-of-core tiling for large datasets
- Parallel processing for tile-level parallelization
- GPU acceleration utilities (future)
- JIT-compiled kernels (future)
"""

from .parallel_executor import TileParallelExecutor, estimate_speedup_factor
from .tile_workers import process_c2c_tile, process_dod_tile, process_m3c2_tile
from .tiling import (
    Bounds2D,
    GridAccumulator,
    LaspyStreamReader,
    MosaicAccumulator,
    Tile,
    Tiler,
    union_bounds,
    scan_las_bounds,
    bounds_intersect,
)

__all__ = [
    # Tiling primitives
    "Bounds2D",
    "GridAccumulator",
    "Tile",
    "Tiler",
    "MosaicAccumulator",
    "LaspyStreamReader",
    "union_bounds",
    "scan_las_bounds",
    "bounds_intersect",
    # Parallel processing
    "TileParallelExecutor",
    "estimate_speedup_factor",
    "process_dod_tile",
    "process_c2c_tile",
    "process_m3c2_tile",
]
