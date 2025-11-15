"""
Acceleration Module

This module provides performance optimization infrastructure including:
- Out-of-core tiling for large datasets
- Parallel processing for tile-level parallelization
- GPU acceleration utilities (hardware_detection.py, gpu_neighbors.py, gpu_array_ops.py)
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
from .hardware_detection import (
    GPUInfo,
    detect_gpu,
    get_gpu_info,
    check_gpu_memory,
    get_optimal_batch_size,
)
from .gpu_array_ops import (
    ArrayBackend,
    get_array_backend,
    reset_array_backend,
    ensure_cpu_array,
    ensure_gpu_array,
    is_gpu_array,
)
from .gpu_neighbors import (
    GPUNearestNeighbors,
    create_gpu_neighbors,
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
    # GPU acceleration
    "GPUInfo",
    "detect_gpu",
    "get_gpu_info",
    "check_gpu_memory",
    "get_optimal_batch_size",
]
