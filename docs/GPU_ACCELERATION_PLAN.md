# GPU Acceleration Plan (Phase 2)

**Date**: November 9, 2025  
**Branch**: TBD (after `feat/gpu-acceleration` Phase 1 complete)  
**Phase**: 2 of 2 (Parallelization → GPU Acceleration)  
**Prerequisite**: Phase 1 (CPU Parallelization) must be complete

## Executive Summary

This document outlines **Phase 2: GPU Acceleration** of the terrain change detection pipeline. After completing CPU-based parallelization in Phase 1, Phase 2 focuses on leveraging GPU compute for the most intensive operations: **nearest neighbor searches** and **array operations**.

**Key Strategy**: Target GPU acceleration at the **operation level**, not the tile level. Each parallel worker can use GPU for its compute kernels while tiles process in parallel across CPU cores.

**Expected Combined Impact**: 
- Phase 1: 6-12x speedup (CPU parallelization)
- Phase 2: Additional 5-15x speedup (GPU acceleration)
- **Total: 30-180x speedup** over original sequential CPU implementation

## Current State (After Phase 1)

### What We Have from Phase 1 ✅

1. **Parallel Tile Processing**:
   - `TileParallelExecutor`: Multi-process tile distribution
   - Worker functions for DoD, C2C, M3C2
   - 6-12x speedup from CPU parallelization

2. **Production-Ready Infrastructure**:
   - Robust error handling and recovery
   - Progress reporting and monitoring
   - Memory-aware worker management
   - Spatial indexing for efficient I/O

3. **Performance Baseline**:
   - Profiling data showing remaining CPU bottlenecks
   - Identified hotspots: NN searches (60-70%), array ops (20-30%)

### What's Missing (Phase 2 Target) 🎯

**GPU-Accelerated Operations**:
- Nearest neighbor searches (KD-tree queries) - **60-70% of worker compute time**
- Grid accumulation and DEM operations - **15-20% of worker compute time**
- Distance calculations and transformations - **10-15% of worker compute time**

**Current Bottleneck**: Even with parallel CPU processing, individual workers are CPU-bound on nearest neighbor searches. GPU acceleration can make each worker 10-50x faster on these operations.

## GPU Acceleration Strategy

### Design Principles

1. **Worker-Level GPU Usage**: Each parallel worker can use GPU for compute kernels
2. **Graceful Degradation**: Automatic fallback to CPU if GPU unavailable or out of memory
3. **Hybrid Execution**: CPU for control flow, GPU for compute kernels
4. **Minimal Code Changes**: Abstraction layer for seamless CPU/GPU switching
5. **Memory Aware**: Smart data transfer between CPU and GPU memory

### Technology Stack

**Primary Framework: CuPy + cuML**
- **CuPy**: NumPy-compatible GPU arrays (CUDA backend)
- **cuML**: GPU-accelerated ML algorithms (NearestNeighbors)
- **Numba**: JIT compilation with CUDA kernel support

**Why This Stack**:
- CuPy provides drop-in replacement for NumPy operations
- cuML's NearestNeighbors matches sklearn API
- Both handle memory management automatically
- Mature ecosystem with good documentation
- Active development and community support

**Alternative/Future**:
- **PyTorch3D**: Advanced 3D operations (normals, ICP)
- **Open3D-GPU**: GPU-accelerated ICP (requires compilation)
- **RAPIDS cuSpatial**: GPU spatial operations

### Architecture Overview

```
GPU Acceleration Layer (new)
├── hardware_detection.py     # GPU capability detection
├── gpu_array_ops.py          # NumPy/CuPy abstraction
├── gpu_neighbors.py          # sklearn/cuML NN wrapper
├── jit_kernels.py            # Numba JIT/CUDA kernels
└── memory_manager.py         # GPU memory management

Integration Points
├── acceleration/tile_workers.py    # Update workers to use GPU ops
├── alignment/fine_registration.py  # GPU-accelerated ICP
└── detection/change_detection.py   # GPU-accelerated C2C/M3C2
```

## Phase 2.1: GPU Infrastructure (Week 1)

### Goal
Set up GPU detection, abstraction layer, and testing infrastructure.

### Implementation Tasks

#### Task 1.1: Hardware Detection Module

**File**: `src/terrain_change_detection/acceleration/hardware_detection.py`

```python
"""GPU hardware detection and capability assessment."""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """GPU device information."""
    available: bool
    device_count: int
    device_name: Optional[str] = None
    memory_gb: Optional[float] = None
    cuda_version: Optional[str] = None
    compute_capability: Optional[tuple] = None


def detect_gpu() -> GPUInfo:
    """
    Detect GPU availability and capabilities.
    
    Returns:
        GPUInfo with device details, or unavailable marker
    """
    try:
        import cupy as cp
        
        if not cp.cuda.is_available():
            logger.info("CUDA not available - GPU acceleration disabled")
            return GPUInfo(available=False, device_count=0)
        
        device_count = cp.cuda.runtime.getDeviceCount()
        device = cp.cuda.Device(0)
        
        info = GPUInfo(
            available=True,
            device_count=device_count,
            device_name=device.name,
            memory_gb=device.mem_info[1] / 1024**3,  # Total memory in GB
            cuda_version=cp.cuda.runtime.runtimeGetVersion(),
            compute_capability=device.compute_capability,
        )
        
        logger.info(
            f"GPU detected: {info.device_name} "
            f"({info.memory_gb:.1f} GB, CUDA {info.cuda_version})"
        )
        return info
        
    except ImportError:
        logger.info("CuPy not installed - GPU acceleration disabled")
        return GPUInfo(available=False, device_count=0)
    except Exception as e:
        logger.warning(f"GPU detection failed: {e}")
        return GPUInfo(available=False, device_count=0)


class GPUContext:
    """
    GPU context manager for safe resource management.
    
    Usage:
        with GPUContext() as gpu:
            if gpu.available:
                # Use GPU
            else:
                # Fallback to CPU
    """
    
    def __init__(self):
        self.info = detect_gpu()
        self.available = self.info.available
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        if self.available:
            import cupy as cp
            # Free unused memory
            cp.get_default_memory_pool().free_all_blocks()


# Global singleton
_gpu_info: Optional[GPUInfo] = None


def get_gpu_info() -> GPUInfo:
    """Get cached GPU info (detect once)."""
    global _gpu_info
    if _gpu_info is None:
        _gpu_info = detect_gpu()
    return _gpu_info
```

#### Task 1.2: Array Operations Abstraction

**File**: `src/terrain_change_detection/acceleration/gpu_array_ops.py`

```python
"""
NumPy/CuPy abstraction layer for transparent CPU/GPU array operations.

Usage:
    from acceleration import gpu_array_ops as xp
    
    # Automatically uses GPU if available, CPU otherwise
    arr = xp.array([1, 2, 3])
    result = xp.sum(arr ** 2)
"""

from typing import Union
import numpy as np

from .hardware_detection import get_gpu_info


# Select backend based on GPU availability
_gpu_info = get_gpu_info()

if _gpu_info.available:
    try:
        import cupy as cp
        _xp = cp
        _backend = "cupy"
    except ImportError:
        _xp = np
        _backend = "numpy"
else:
    _xp = np
    _backend = "numpy"


# Expose unified interface
array = _xp.array
zeros = _xp.zeros
ones = _xp.ones
empty = _xp.empty
arange = _xp.arange
linspace = _xp.linspace
meshgrid = _xp.meshgrid
sum = _xp.sum
mean = _xp.mean
std = _xp.std
min = _xp.min
max = _xp.max
sqrt = _xp.sqrt
exp = _xp.exp
log = _xp.log
abs = _xp.abs
ceil = _xp.ceil
floor = _xp.floor
where = _xp.where
vstack = _xp.vstack
hstack = _xp.hstack
concatenate = _xp.concatenate


def asarray(arr, dtype=None):
    """Convert to array (GPU if available)."""
    return _xp.asarray(arr, dtype=dtype)


def asnumpy(arr):
    """Convert to NumPy array (transfer from GPU if needed)."""
    if _backend == "cupy":
        import cupy as cp
        return cp.asnumpy(arr)
    else:
        return np.asarray(arr)


def get_backend():
    """Get current backend name ('numpy' or 'cupy')."""
    return _backend


def is_gpu_array(arr):
    """Check if array is on GPU."""
    if _backend == "cupy":
        import cupy as cp
        return isinstance(arr, cp.ndarray)
    return False


def synchronize():
    """Synchronize GPU operations (no-op on CPU)."""
    if _backend == "cupy":
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
```

#### Task 1.3: Configuration Updates

**File**: `config/default.yaml`

```yaml
gpu:
  # Master switch for GPU acceleration
  enabled: true
  
  # GPU memory management
  memory_pool_limit_gb: null    # null = use all available
  transfer_batch_size: 100000   # Points per CPU->GPU transfer batch
  
  # Fallback behavior
  fallback_to_cpu: true          # Auto-fallback if GPU fails
  min_points_for_gpu: 10000      # Use CPU for small datasets
  
  # Operations to accelerate (fine-grained control)
  use_gpu_for:
    nearest_neighbors: true      # KD-tree queries
    grid_operations: true        # DEM accumulation
    transformations: true        # Point transformations
    distance_calculations: true  # C2C distances
```

### Testing & Validation

**Unit Tests** (`tests/test_gpu_hardware.py`):
- GPU detection works correctly
- Graceful handling of missing CuPy
- Backend selection logic
- Array transfers CPU ↔ GPU

**Integration Tests** (`tests/test_gpu_array_ops.py`):
- NumPy/CuPy API parity
- Results match between backends
- Performance improvement measured

## Phase 2.2: GPU Nearest Neighbors (Week 2-3)

### Goal
Implement GPU-accelerated nearest neighbor search for C2C, M3C2, and ICP.

**Target Impact**: This is the **highest-value optimization** - NN searches are 60-70% of worker compute time.

### Implementation Tasks

#### Task 2.1: GPU Nearest Neighbors Wrapper

**File**: `src/terrain_change_detection/acceleration/gpu_neighbors.py`

```python
"""
Unified nearest neighbors interface supporting CPU (sklearn) and GPU (cuML).
"""

import logging
from typing import Tuple, Optional
import numpy as np

from .hardware_detection import get_gpu_info
from .gpu_array_ops import asnumpy, is_gpu_array

logger = logging.getLogger(__name__)


class NearestNeighbors:
    """
    Unified NN interface with automatic CPU/GPU selection.
    
    API matches sklearn.neighbors.NearestNeighbors for drop-in compatibility.
    """
    
    def __init__(
        self,
        n_neighbors: int = 1,
        algorithm: str = "auto",
        metric: str = "euclidean",
        use_gpu: bool = True,
        fallback_to_cpu: bool = True,
    ):
        """
        Args:
            n_neighbors: Number of neighbors to find
            algorithm: Algorithm hint ('auto', 'kd_tree', 'brute')
            metric: Distance metric
            use_gpu: Try to use GPU if available
            fallback_to_cpu: Fallback to CPU on GPU failure
        """
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric
        self.use_gpu = use_gpu and get_gpu_info().available
        self.fallback_to_cpu = fallback_to_cpu
        
        self._model = None
        self._backend = None
    
    def fit(self, X: np.ndarray) -> "NearestNeighbors":
        """
        Fit the model with target points.
        
        Args:
            X: Target points (N, D) array
            
        Returns:
            Self for chaining
        """
        if self.use_gpu:
            try:
                from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
                import cupy as cp
                
                # Transfer to GPU
                X_gpu = cp.asarray(X)
                
                # Fit cuML model
                self._model = cuNearestNeighbors(
                    n_neighbors=self.n_neighbors,
                    metric=self.metric,
                )
                self._model.fit(X_gpu)
                self._backend = "cuml"
                
                logger.debug(f"Fitted GPU NN model with {len(X)} points")
                return self
                
            except Exception as e:
                if self.fallback_to_cpu:
                    logger.warning(f"GPU NN failed, falling back to CPU: {e}")
                else:
                    raise
        
        # CPU fallback
        from sklearn.neighbors import NearestNeighbors as skNearestNeighbors
        
        self._model = skNearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm=self.algorithm,
            metric=self.metric,
        )
        self._model.fit(X)
        self._backend = "sklearn"
        
        logger.debug(f"Fitted CPU NN model with {len(X)} points")
        return self
    
    def kneighbors(
        self,
        X: Optional[np.ndarray] = None,
        n_neighbors: Optional[int] = None,
        return_distance: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find K nearest neighbors.
        
        Args:
            X: Query points (M, D) array (None = use fit data)
            n_neighbors: Number of neighbors (None = use constructor value)
            return_distance: Whether to return distances
            
        Returns:
            Tuple of (distances, indices) as NumPy arrays
        """
        if self._model is None:
            raise ValueError("Must call fit() before kneighbors()")
        
        n_neighbors = n_neighbors or self.n_neighbors
        
        if self._backend == "cuml":
            import cupy as cp
            
            # Transfer to GPU if needed
            if X is not None and not is_gpu_array(X):
                X = cp.asarray(X)
            
            # Query
            distances, indices = self._model.kneighbors(
                X, n_neighbors=n_neighbors, return_distance=return_distance
            )
            
            # Transfer back to CPU
            distances = asnumpy(distances)
            indices = asnumpy(indices)
            
        else:  # sklearn
            result = self._model.kneighbors(
                X, n_neighbors=n_neighbors, return_distance=return_distance
            )
            if return_distance:
                distances, indices = result
            else:
                indices = result
                distances = None
        
        return (distances, indices) if return_distance else indices
    
    def get_backend(self) -> str:
        """Get active backend ('cuml' or 'sklearn')."""
        return self._backend
```

#### Task 2.2: Update C2C Worker for GPU

**File**: `src/terrain_change_detection/acceleration/tile_workers.py`

Update `process_c2c_tile()`:

```python
def process_c2c_tile(
    tile: Tile,
    files_source: List[Path],
    files_target: List[Path],
    max_distance: float,
    chunk_points: int,
    classification_filter: Optional[List[int]],
    transform_matrix: Optional[np.ndarray] = None,
    k_neighbors: int = 1,
    use_gpu: bool = True,  # NEW
) -> Tuple[Tile, np.ndarray]:
    """Process single C2C tile with optional GPU acceleration."""
    from ..acceleration.gpu_neighbors import NearestNeighbors
    
    # Load points (same as before)
    source = load_tile_points(files_source, tile.inner, ...)
    target = load_tile_points(files_target, tile.outer, ...)
    
    if len(source) == 0 or len(target) == 0:
        return (tile, np.array([]))
    
    # Use GPU-capable NN (automatically falls back to CPU if needed)
    nbrs = NearestNeighbors(
        n_neighbors=k_neighbors,
        use_gpu=use_gpu,
        fallback_to_cpu=True,
    )
    nbrs.fit(target)
    distances, _ = nbrs.kneighbors(source)
    
    # Rest of processing...
    distances = distances.flatten()
    valid = distances <= max_distance
    return (tile, distances[valid])
```

#### Task 2.3: Update ICP for GPU

**File**: `src/terrain_change_detection/alignment/fine_registration.py`

Update ICP implementation to use GPU NN:

```python
def icp_align_gpu(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    max_correspondence_distance: Optional[float] = None,
    use_gpu: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    ICP alignment with GPU-accelerated nearest neighbor search.
    """
    from ..acceleration.gpu_neighbors import NearestNeighbors
    
    current_transform = np.eye(4)
    current_source = source.copy()
    
    for iteration in range(max_iterations):
        # GPU-accelerated nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=1, use_gpu=use_gpu)
        nbrs.fit(target)
        distances, indices = nbrs.kneighbors(current_source)
        
        # Filter correspondences by distance
        if max_correspondence_distance:
            valid = distances.flatten() < max_correspondence_distance
            src_matched = current_source[valid]
            tgt_matched = target[indices[valid].flatten()]
        else:
            src_matched = current_source
            tgt_matched = target[indices.flatten()]
        
        # Compute transformation (CPU - small matrices)
        delta_transform = compute_transformation(src_matched, tgt_matched)
        
        # Check convergence
        if np.linalg.norm(delta_transform - np.eye(4)) < tolerance:
            break
        
        # Update
        current_transform = delta_transform @ current_transform
        current_source = apply_transform(current_source, delta_transform)
    
    return current_transform, {"iterations": iteration + 1}
```

### Testing & Validation

**Benchmarks** (`tests/benchmarks/test_gpu_neighbors.py`):
- Compare GPU vs CPU NN search speeds
- Measure speedup factors for various point counts
- Validate numerical accuracy (distances should match)

**Expected Results**:
- **10-50x speedup** for NN searches (larger datasets = bigger gains)
- C2C: 60-70% faster per tile
- ICP: 15-30x faster alignment
- Automatic fallback works correctly

## Phase 2.3: GPU Grid Operations (Week 4)

### Goal
Accelerate DoD grid accumulation and DEM operations using GPU.

### Implementation Tasks

#### Task 3.1: GPU Grid Accumulator

Update `GridAccumulator` to support GPU arrays:

**File**: `src/terrain_change_detection/acceleration/tiling.py`

```python
class GridAccumulator:
    """Streaming mean aggregator with optional GPU acceleration."""
    
    def __init__(
        self,
        bounds: Bounds2D,
        cell_size: float,
        use_gpu: bool = False,
    ):
        from . import gpu_array_ops as xp
        
        self.bounds = bounds
        self.cell = float(cell_size)
        self.use_gpu = use_gpu
        
        # Grid sizes
        self.nx = int(np.ceil((bounds.max_x - bounds.min_x) / self.cell)) + 1
        self.ny = int(np.ceil((bounds.max_y - bounds.min_y) / self.cell)) + 1
        
        # Accumulators (GPU or CPU arrays)
        if use_gpu and xp.get_backend() == "cupy":
            self.sum = xp.zeros((self.ny, self.nx), dtype=xp.float64)
            self.cnt = xp.zeros((self.ny, self.nx), dtype=xp.int64)
            self._xp = xp
        else:
            self.sum = np.zeros((self.ny, self.nx), dtype=np.float64)
            self.cnt = np.zeros((self.ny, self.nx), dtype=np.int64)
            self._xp = np
        
        # Precompute grid centers (always on CPU, small arrays)
        x_edges = np.linspace(bounds.min_x, bounds.min_x + self.nx * self.cell, self.nx + 1)
        y_edges = np.linspace(bounds.min_y, bounds.min_y + self.ny * self.cell, self.ny + 1)
        self.grid_x, self.grid_y = np.meshgrid(
            (x_edges[:-1] + x_edges[1:]) / 2.0,
            (y_edges[:-1] + y_edges[1:]) / 2.0,
        )
    
    def accumulate(self, points: np.ndarray) -> None:
        """Accumulate points (transfers to GPU if enabled)."""
        if points.size == 0:
            return
        
        xp = self._xp
        
        # Transfer to GPU if needed
        if self.use_gpu and xp.get_backend() == "cupy":
            points_device = xp.asarray(points)
        else:
            points_device = points
        
        # Compute grid indices (vectorized on GPU/CPU)
        x, y, z = points_device[:, 0], points_device[:, 1], points_device[:, 2]
        col = ((x - self.bounds.min_x) / self.cell).astype(xp.int32)
        row = ((y - self.bounds.min_y) / self.cell).astype(xp.int32)
        
        # Filter valid indices
        valid = (row >= 0) & (row < self.ny) & (col >= 0) & (col < self.nx)
        row_valid = row[valid]
        col_valid = col[valid]
        z_valid = z[valid]
        
        # Accumulate using atomic operations (GPU) or bincount (CPU)
        if self.use_gpu and xp.get_backend() == "cupy":
            # Use CuPy's atomic operations or custom CUDA kernel
            self._accumulate_gpu(row_valid, col_valid, z_valid)
        else:
            # CPU: use numpy bincount
            self._accumulate_cpu(row_valid, col_valid, z_valid)
    
    def compute_mean(self) -> np.ndarray:
        """Compute mean DEM (transfers back from GPU if needed)."""
        xp = self._xp
        
        # Compute mean
        mean_dem = xp.where(self.cnt > 0, self.sum / self.cnt, xp.nan)
        
        # Transfer back to CPU if on GPU
        if self.use_gpu and xp.get_backend() == "cupy":
            mean_dem = xp.asnumpy(mean_dem)
        
        return mean_dem
```

#### Task 3.2: JIT Kernels for Remaining Operations

**File**: `src/terrain_change_detection/acceleration/jit_kernels.py`

Add Numba JIT compilation for CPU-bound operations:

```python
"""JIT-compiled kernels for performance-critical operations."""

import numba
import numpy as np


@numba.jit(nopython=True, parallel=True)
def apply_transform_jit(
    points: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    """
    Apply 4x4 transformation matrix to points (JIT-compiled).
    
    Args:
        points: (N, 3) array
        matrix: (4, 4) transformation matrix
        
    Returns:
        Transformed points (N, 3)
    """
    n = points.shape[0]
    result = np.empty_like(points)
    
    for i in numba.prange(n):
        x, y, z = points[i]
        result[i, 0] = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2] * z + matrix[0, 3]
        result[i, 1] = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2] * z + matrix[1, 3]
        result[i, 2] = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2] * z + matrix[2, 3]
    
    return result


@numba.jit(nopython=True, parallel=True)
def compute_distances_jit(
    points1: np.ndarray,
    points2: np.ndarray,
) -> np.ndarray:
    """
    Compute Euclidean distances between corresponding points (JIT-compiled).
    """
    n = points1.shape[0]
    distances = np.empty(n, dtype=np.float64)
    
    for i in numba.prange(n):
        dx = points1[i, 0] - points2[i, 0]
        dy = points1[i, 1] - points2[i, 1]
        dz = points1[i, 2] - points2[i, 2]
        distances[i] = np.sqrt(dx*dx + dy*dy + dz*dz)
    
    return distances
```

### Testing & Validation

**Benchmarks**:
- GPU grid accumulation vs CPU
- JIT transformation vs vanilla NumPy
- End-to-end DoD speedup

**Expected Results**:
- Grid operations: 5-10x speedup on GPU
- JIT transformations: 2-5x speedup on CPU
- DoD total: 8-15x per tile (combined with Phase 1)

## Phase 2.4: Production Readiness (Week 5)

### Goal
Polish, optimize, and prepare GPU acceleration for production use.

### Implementation Tasks

#### Task 4.1: Memory Management

Implement smart GPU memory management:

```python
class GPUMemoryManager:
    """
    Manages GPU memory allocation and transfers.
    
    Features:
    - Batch transfers to amortize overhead
    - LRU cache for frequently accessed data
    - Automatic fallback if GPU memory full
    """
    
    def __init__(self, memory_limit_gb: Optional[float] = None):
        self.memory_limit = memory_limit_gb
        self.cache = {}
    
    def transfer_points(self, points: np.ndarray, batch_size: int = 100000):
        """Transfer points to GPU in batches."""
        pass
    
    def free_memory(self):
        """Free unused GPU memory."""
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
```

#### Task 4.2: Performance Tuning

- Profile GPU utilization with `nvprof` or `nsight`
- Optimize batch sizes for CPU↔GPU transfers
- Tune worker count vs GPU memory availability
- Benchmark on real datasets

#### Task 4.3: Documentation

- Update configuration guide with GPU settings
- Add GPU troubleshooting guide
- Document performance characteristics
- Create user guide for GPU setup

#### Task 4.4: Comprehensive Testing

**Test Suite**:
- Unit tests for all GPU operations
- Integration tests with parallelization
- Stress tests on large datasets
- Multi-GPU support (future)

**Validation**:
- Results match CPU implementation (numerical parity)
- Performance targets met (5-15x additional speedup)
- Fallback mechanisms work correctly
- No memory leaks or GPU hangs

## Performance Targets

### Expected Combined Speedup (Phase 1 + Phase 2)

| Operation | CPU Sequential | Phase 1 (Parallel) | Phase 2 (Parallel + GPU) | Total Speedup |
|-----------|----------------|-------------------|-------------------------|---------------|
| DoD | 100 min | 12 min (8x) | 7 min (2x) | **14x** |
| C2C | 200 min | 25 min (8x) | 5 min (5x) | **40x** |
| ICP Alignment | 60 min | 30 min (2x) | 2 min (15x) | **30x** |
| M3C2 | 300 min | 37 min (8x) | 10 min (4x) | **30x** |

**Overall Pipeline**: **30-50x faster** than original implementation

### Hardware Requirements

**Minimum**:
- NVIDIA GPU with CUDA support (compute capability 6.0+)
- 4 GB GPU memory
- 16 GB system RAM
- CUDA Toolkit 11.x or 12.x

**Recommended**:
- NVIDIA RTX 3000/4000 series or better
- 8+ GB GPU memory
- 32+ GB system RAM
- NVMe SSD for fast I/O

**Optimal**:
- NVIDIA A100 or H100 (data center GPU)
- 24+ GB GPU memory
- 64+ GB system RAM
- Multi-GPU support (future phase)

## Dependencies

### Required Packages

```toml
# pyproject.toml

[project.optional-dependencies]
gpu = [
    "cupy-cuda12x>=12.0.0",  # or cupy-cuda11x for CUDA 11
    "cuml-cu12>=23.10.0",    # RAPIDS cuML
    "numba>=0.58.0",         # JIT compilation
]

[tool.uv]
# Auto-detect CUDA version
cuda-version = "12.x"
```

### Installation

```powershell
# Install GPU dependencies
uv pip install -e ".[gpu]"

# Verify installation
python -c "import cupy; print(cupy.cuda.is_available())"
python -c "from cuml.neighbors import NearestNeighbors"
```

## Success Criteria

Phase 2 is complete when:

1. ✅ GPU acceleration working for all major operations (NN, grid ops, transforms)
2. ✅ 5-15x additional speedup demonstrated on GPU hardware
3. ✅ Combined Phase 1 + Phase 2 achieves 30-50x total speedup
4. ✅ Graceful fallback to CPU works correctly
5. ✅ Memory management prevents OOM on GPU
6. ✅ All tests passing (CPU and GPU modes)
7. ✅ Production-ready configuration and documentation
8. ✅ User guide for GPU setup complete

## Future Work (Phase 3+)

### Potential Enhancements

1. **Multi-GPU Support**:
   - Distribute tiles across multiple GPUs
   - Data parallelism for large-scale processing
   - Expected: 2-4x additional speedup per GPU

2. **Custom CUDA Kernels**:
   - Fused operations for grid accumulation
   - Specialized M3C2 cylinder search kernels
   - Expected: 2-3x improvement over CuPy

3. **Distributed Computing**:
   - Dask/Ray integration for HPC clusters
   - Multi-node processing with GPU workers
   - Scale to national/continental datasets

4. **Advanced Optimizations**:
   - Mixed precision (FP16/FP32) where appropriate
   - Tensor core utilization for matrix ops
   - Persistent GPU kernels to reduce overhead

## References

- Phase 1 Plan: `docs/PARALLELIZATION_PLAN.md`
- CuPy Documentation: https://docs.cupy.dev/
- RAPIDS cuML: https://docs.rapids.ai/api/cuml/
- Numba CUDA: https://numba.readthedocs.io/en/stable/cuda/
- Configuration Guide: `docs/CONFIGURATION_GUIDE.md`
