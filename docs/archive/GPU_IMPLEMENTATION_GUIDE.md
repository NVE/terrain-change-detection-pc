# GPU Implementation Guide

Quick reference for implementing GPU acceleration in the terrain change detection pipeline.

## Quick Start: GPU Setup

### 1. Install CUDA Toolkit

**Windows:**
```powershell
# Download from NVIDIA website
# https://developer.nvidia.com/cuda-downloads
# Recommended: CUDA 12.x for latest GPUs, 11.8 for broader compatibility
```

**Linux:**
```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda
```

### 2. Verify GPU is Available

```powershell
# Check NVIDIA GPU
nvidia-smi

# Expected output: GPU info, memory, CUDA version
```

### 3. Install Python GPU Libraries

```powershell
# Core dependencies
uv pip install cupy-cuda12x  # or cupy-cuda11x for CUDA 11
uv pip install numba

# Optional: RAPIDS cuML (requires compatible CUDA version)
# conda install -c rapidsai -c conda-forge -c nvidia cuml
```

### 4. Test GPU in Python

```python
import cupy as cp
import numpy as np

# Check CuPy can access GPU
print("CuPy version:", cp.__version__)
print("CUDA available:", cp.cuda.is_available())
print("Device count:", cp.cuda.runtime.getDeviceCount())

# Simple test
x_gpu = cp.array([1, 2, 3])
y_gpu = x_gpu * 2
print("GPU result:", y_gpu)

# Benchmark
n = 10_000_000
x_cpu = np.random.rand(n)
x_gpu = cp.random.rand(n)

# CPU
%timeit np.sum(x_cpu ** 2)

# GPU
%timeit cp.sum(x_gpu ** 2)
```

## Architecture: Hardware Abstraction Layer

### Module Structure

```
src/terrain_change_detection/
└── acceleration/
    ├── __init__.py              # Public API
    ├── hardware.py              # GPU detection and configuration
    ├── array_ops.py             # NumPy/CuPy abstraction
    ├── nearest_neighbors.py     # GPU-accelerated KD-tree
    ├── jit_kernels.py           # Numba JIT functions
    └── benchmarks.py            # Performance testing
```

### Key Design Principles

1. **Graceful Degradation**: Always fallback to CPU if GPU unavailable
2. **Transparent API**: Minimal code changes in existing modules
3. **Memory Aware**: Automatic chunking for large datasets
4. **Configurable**: Runtime selection of backend via config

## Implementation Examples

### 1. Hardware Detection

```python
# src/terrain_change_detection/acceleration/hardware.py
import logging
from dataclasses import dataclass
from typing import Optional, List

logger = logging.getLogger(__name__)

@dataclass
class GPUInfo:
    """Information about a single GPU device."""
    device_id: int
    name: str
    memory_total_gb: float
    memory_free_gb: float
    compute_capability: tuple
    
@dataclass
class HardwareConfig:
    """Runtime hardware configuration."""
    use_gpu: bool
    gpu_devices: List[GPUInfo]
    cpu_count: int
    available_memory_gb: float

class HardwareDetector:
    """Detect and configure available hardware resources."""
    
    @staticmethod
    def detect() -> HardwareConfig:
        """Detect available hardware."""
        import os
        import psutil
        
        cpu_count = os.cpu_count() or 1
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Try to detect GPUs
        gpu_devices = []
        use_gpu = False
        
        try:
            import cupy as cp
            if cp.cuda.is_available():
                device_count = cp.cuda.runtime.getDeviceCount()
                for i in range(device_count):
                    with cp.cuda.Device(i):
                        props = cp.cuda.runtime.getDeviceProperties(i)
                        mem_info = cp.cuda.runtime.memGetInfo()
                        gpu_devices.append(GPUInfo(
                            device_id=i,
                            name=props['name'].decode('utf-8'),
                            memory_total_gb=props['totalGlobalMem'] / (1024**3),
                            memory_free_gb=mem_info[0] / (1024**3),
                            compute_capability=(props['major'], props['minor'])
                        ))
                use_gpu = len(gpu_devices) > 0
                logger.info(f"Detected {len(gpu_devices)} GPU(s)")
        except (ImportError, Exception) as e:
            logger.warning(f"GPU detection failed: {e}. Using CPU only.")
        
        return HardwareConfig(
            use_gpu=use_gpu,
            gpu_devices=gpu_devices,
            cpu_count=cpu_count,
            available_memory_gb=available_memory_gb
        )
    
    @staticmethod
    def select_device(device_id: int = 0) -> None:
        """Select a specific GPU device."""
        try:
            import cupy as cp
            cp.cuda.Device(device_id).use()
            logger.info(f"Using GPU device {device_id}")
        except Exception as e:
            logger.error(f"Failed to select GPU device {device_id}: {e}")

# Global hardware config (lazy initialization)
_hardware_config: Optional[HardwareConfig] = None

def get_hardware_config() -> HardwareConfig:
    """Get or initialize hardware configuration."""
    global _hardware_config
    if _hardware_config is None:
        _hardware_config = HardwareDetector.detect()
    return _hardware_config
```

### 2. Array Backend Abstraction

```python
# src/terrain_change_detection/acceleration/array_ops.py
"""
Unified array operations supporting both NumPy (CPU) and CuPy (GPU).

Usage:
    import terrain_change_detection.acceleration.array_ops as xp
    
    # xp automatically dispatches to NumPy or CuPy
    x = xp.array([1, 2, 3])
    y = xp.sum(x ** 2)
"""

import numpy as np
from typing import Any, Optional
from .hardware import get_hardware_config

# Determine backend
_use_cupy = False
_xp = np

try:
    import cupy as cp
    if get_hardware_config().use_gpu:
        _xp = cp
        _use_cupy = True
except ImportError:
    pass

# Export unified interface
array = _xp.array
asarray = _xp.asarray
zeros = _xp.zeros
ones = _xp.ones
empty = _xp.empty
arange = _xp.arange
linspace = _xp.linspace

# Math operations
sum = _xp.sum
mean = _xp.mean
std = _xp.std
sqrt = _xp.sqrt
square = _xp.square
abs = _xp.abs
min = _xp.min
max = _xp.max

# Array operations
concatenate = _xp.concatenate
stack = _xp.stack
reshape = _xp.reshape
transpose = _xp.transpose
clip = _xp.clip

# Statistical
nanmean = _xp.nanmean
nanmedian = _xp.nanmedian
nanstd = _xp.nanstd

# Utilities
def is_gpu_array(arr: Any) -> bool:
    """Check if array is on GPU."""
    if not _use_cupy:
        return False
    import cupy as cp
    return isinstance(arr, cp.ndarray)

def to_cpu(arr: Any) -> np.ndarray:
    """Transfer array from GPU to CPU if needed."""
    if is_gpu_array(arr):
        return arr.get()  # CuPy .get() transfers to host
    return np.asarray(arr)

def to_gpu(arr: Any) -> Any:
    """Transfer array from CPU to GPU if GPU is available."""
    if _use_cupy and not is_gpu_array(arr):
        import cupy as cp
        return cp.asarray(arr)
    return arr

def get_backend_name() -> str:
    """Get name of current backend."""
    return "cupy" if _use_cupy else "numpy"

def using_gpu() -> bool:
    """Check if GPU backend is active."""
    return _use_cupy

# Memory management
def clear_gpu_memory():
    """Clear GPU memory pool."""
    if _use_cupy:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
```

### 3. GPU-Accelerated Nearest Neighbors

```python
# src/terrain_change_detection/acceleration/nearest_neighbors.py
"""
GPU-accelerated nearest neighbor search with CPU fallback.
"""

import numpy as np
from typing import Tuple, Optional
import logging

from .hardware import get_hardware_config
from . import array_ops as xp

logger = logging.getLogger(__name__)

class GPUNearestNeighbors:
    """
    GPU-accelerated nearest neighbor search compatible with sklearn API.
    
    Automatically falls back to CPU if GPU unavailable or data too large.
    """
    
    def __init__(self, 
                 n_neighbors: int = 1,
                 algorithm: str = 'auto',
                 use_gpu: bool = True,
                 max_gpu_points: int = 10_000_000):
        """
        Args:
            n_neighbors: Number of neighbors to query
            algorithm: 'auto', 'gpu', or 'cpu'
            use_gpu: Enable GPU acceleration if available
            max_gpu_points: Maximum points to fit on GPU before chunking
        """
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.use_gpu = use_gpu and get_hardware_config().use_gpu
        self.max_gpu_points = max_gpu_points
        
        self._target = None
        self._nbrs_cpu = None
        self._use_cuml = False
        
    def fit(self, X: np.ndarray) -> 'GPUNearestNeighbors':
        """
        Fit the nearest neighbor index on target points.
        
        Args:
            X: Target points array (N, D)
        """
        if self.use_gpu and self._should_use_gpu(len(X)):
            self._fit_gpu(X)
        else:
            self._fit_cpu(X)
        return self
    
    def _should_use_gpu(self, n_points: int) -> bool:
        """Decide whether to use GPU based on data size."""
        if not self.use_gpu:
            return False
        
        hw = get_hardware_config()
        if not hw.use_gpu or not hw.gpu_devices:
            return False
        
        # Estimate memory requirement (rough: 24 bytes per point for tree)
        required_gb = n_points * 24 / (1024**3)
        available_gb = hw.gpu_devices[0].memory_free_gb * 0.8  # 80% headroom
        
        return required_gb < available_gb
    
    def _fit_gpu(self, X: np.ndarray) -> None:
        """Fit using GPU (cuML)."""
        try:
            from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
            import cupy as cp
            
            X_gpu = cp.asarray(X, dtype=cp.float32)
            self._nbrs_gpu = cuNearestNeighbors(
                n_neighbors=self.n_neighbors,
                algorithm='brute'  # cuML uses brute force (still fast on GPU)
            )
            self._nbrs_gpu.fit(X_gpu)
            self._target = X_gpu
            self._use_cuml = True
            logger.info(f"Fitted GPU nearest neighbors on {len(X)} points")
            
        except (ImportError, Exception) as e:
            logger.warning(f"GPU fit failed: {e}. Falling back to CPU.")
            self._fit_cpu(X)
    
    def _fit_cpu(self, X: np.ndarray) -> None:
        """Fit using CPU (sklearn)."""
        from sklearn.neighbors import NearestNeighbors
        
        self._nbrs_cpu = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm='kd_tree'
        )
        self._nbrs_cpu.fit(X)
        self._target = X
        self._use_cuml = False
        logger.info(f"Fitted CPU nearest neighbors on {len(X)} points")
    
    def kneighbors(self, X: Optional[np.ndarray] = None, 
                   return_distance: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query k-nearest neighbors.
        
        Args:
            X: Query points (M, D). If None, use training points.
            return_distance: Return distances in addition to indices
            
        Returns:
            (distances, indices) if return_distance else indices
        """
        if self._use_cuml:
            return self._kneighbors_gpu(X, return_distance)
        else:
            return self._kneighbors_cpu(X, return_distance)
    
    def _kneighbors_gpu(self, X: Optional[np.ndarray], 
                       return_distance: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Query using GPU."""
        import cupy as cp
        
        if X is None:
            X_gpu = self._target
        else:
            X_gpu = cp.asarray(X, dtype=cp.float32)
        
        distances, indices = self._nbrs_gpu.kneighbors(X_gpu)
        
        # Transfer back to CPU
        distances = cp.asnumpy(distances)
        indices = cp.asnumpy(indices).astype(np.int64)
        
        if return_distance:
            return distances, indices
        return indices
    
    def _kneighbors_cpu(self, X: Optional[np.ndarray], 
                       return_distance: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Query using CPU."""
        result = self._nbrs_cpu.kneighbors(X, return_distance=return_distance)
        if return_distance:
            return result
        return result
```

### 4. JIT-Compiled Grid Accumulation

```python
# src/terrain_change_detection/acceleration/jit_kernels.py
"""
Numba JIT-compiled kernels for performance-critical operations.
"""

import numpy as np
import numba

@numba.jit(nopython=True, parallel=True, cache=True)
def accumulate_grid_mean_jit(points: np.ndarray,
                             min_x: float, min_y: float,
                             cell_size: float,
                             sum_grid: np.ndarray,
                             cnt_grid: np.ndarray) -> None:
    """
    JIT-compiled grid accumulation for mean DEM.
    
    Args:
        points: Point cloud (N, 3)
        min_x, min_y: Grid origin
        cell_size: Cell size
        sum_grid: Accumulator for Z sums (ny, nx)
        cnt_grid: Accumulator for counts (ny, nx)
    """
    ny, nx = sum_grid.shape
    n_points = points.shape[0]
    
    for i in numba.prange(n_points):
        x, y, z = points[i, 0], points[i, 1], points[i, 2]
        
        # Compute grid indices
        xi = int((x - min_x) / cell_size)
        yi = int((y - min_y) / cell_size)
        
        # Check bounds
        if 0 <= xi < nx and 0 <= yi < ny:
            sum_grid[yi, xi] += z
            cnt_grid[yi, xi] += 1


@numba.jit(nopython=True, parallel=True, cache=True)
def apply_transform_jit(points: np.ndarray,
                       R: np.ndarray,
                       t: np.ndarray) -> np.ndarray:
    """
    Apply rigid transformation: R @ points.T + t
    
    Args:
        points: Points (N, 3)
        R: Rotation matrix (3, 3)
        t: Translation vector (3,)
    
    Returns:
        Transformed points (N, 3)
    """
    n = points.shape[0]
    result = np.empty_like(points)
    
    for i in numba.prange(n):
        for j in range(3):
            result[i, j] = (R[j, 0] * points[i, 0] +
                           R[j, 1] * points[i, 1] +
                           R[j, 2] * points[i, 2] +
                           t[j])
    
    return result


@numba.jit(nopython=True, cache=True)
def compute_centroid_jit(points: np.ndarray) -> np.ndarray:
    """Compute centroid of points."""
    n = points.shape[0]
    centroid = np.zeros(3, dtype=np.float64)
    
    for i in range(n):
        for j in range(3):
            centroid[j] += points[i, j]
    
    for j in range(3):
        centroid[j] /= n
    
    return centroid
```

## Integration into Existing Code

### Step 1: Update ICP Registration

```python
# src/terrain_change_detection/alignment/fine_registration.py

# Add at top of file
try:
    from ..acceleration.nearest_neighbors import GPUNearestNeighbors
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class ICPRegistration:
    def __init__(self,
                 max_iterations: int = 100,
                 tolerance: float = 1e-6,
                 max_correspondence_distance: float = 1.0,
                 use_gpu: bool = True):  # New parameter
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.max_correspondence_distance = max_correspondence_distance
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
    def align_point_clouds(self, source, target, initial_transform=None):
        # ... existing code ...
        
        # Build KD-tree once (GPU or CPU)
        if self.use_gpu:
            nbrs = GPUNearestNeighbors(n_neighbors=1)
        else:
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
        
        nbrs.fit(target)
        
        # Rest of ICP loop unchanged!
        for iteration in range(self.max_iterations):
            correspondences, distances = self.find_correspondences(current_source, nbrs)
            # ... rest of code ...
```

### Step 2: Update DoD Computation

```python
# src/terrain_change_detection/detection/change_detection.py

try:
    from ..acceleration.jit_kernels import accumulate_grid_mean_jit
    JIT_AVAILABLE = True
except ImportError:
    JIT_AVAILABLE = False

class ChangeDetector:
    @staticmethod
    def compute_dod(points_t1, points_t2, cell_size=1.0, bounds=None, 
                   aggregator="mean", use_jit=True):
        # ... bounds setup ...
        
        def grid_dem(points: np.ndarray) -> np.ndarray:
            if use_jit and JIT_AVAILABLE and aggregator == "mean":
                # Fast path: JIT-compiled accumulation
                sum_grid = np.zeros((ny, nx), dtype=np.float64)
                cnt_grid = np.zeros((ny, nx), dtype=np.int64)
                accumulate_grid_mean_jit(points, min_x, min_y, cell_size, 
                                        sum_grid, cnt_grid)
                dem = np.full((ny, nx), np.nan, dtype=np.float64)
                mask = cnt_grid > 0
                dem[mask] = sum_grid[mask] / cnt_grid[mask]
                return dem
            else:
                # Original implementation for other aggregators
                # ... existing code ...
        
        # Rest unchanged
        dem1 = grid_dem(points_t1)
        dem2 = grid_dem(points_t2)
        # ...
```

## Configuration Integration

### Update YAML Schema

```yaml
# config/default.yaml
performance:
  numpy_threads: auto
  cpu_workers: auto
  
  gpu:
    enabled: true  # Auto-detect and use GPU if available
    backend: cuda  # cuda | rocm | cpu
    device_id: 0   # Which GPU to use (if multiple)
    memory_limit_gb: null  # Auto-detect if null
    fallback_to_cpu: true  # Fallback if GPU fails
  
  jit:
    enabled: true  # Enable Numba JIT
    cache: true    # Cache compiled functions
    parallel: true # Enable parallel execution
```

### Load Configuration

```python
# src/terrain_change_detection/utils/config.py

from pydantic import BaseModel

class GPUConfig(BaseModel):
    enabled: bool = True
    backend: str = "cuda"
    device_id: int = 0
    memory_limit_gb: Optional[float] = None
    fallback_to_cpu: bool = True

class JITConfig(BaseModel):
    enabled: bool = True
    cache: bool = True
    parallel: bool = True

class PerformanceConfig(BaseModel):
    numpy_threads: Union[int, str] = "auto"
    cpu_workers: Union[int, str] = "auto"
    gpu: GPUConfig = GPUConfig()
    jit: JITConfig = JITConfig()
    # ... existing fields ...
```

## Testing Strategy

### Unit Tests for GPU Operations

```python
# tests/test_gpu_acceleration.py
import pytest
import numpy as np

def test_gpu_nearest_neighbors():
    """Test GPU nearest neighbor search matches CPU results."""
    from terrain_change_detection.acceleration import GPUNearestNeighbors
    from sklearn.neighbors import NearestNeighbors
    
    # Create test data
    np.random.seed(42)
    target = np.random.rand(1000, 3)
    query = np.random.rand(100, 3)
    
    # CPU baseline
    nbrs_cpu = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
    nbrs_cpu.fit(target)
    dist_cpu, idx_cpu = nbrs_cpu.kneighbors(query)
    
    # GPU version
    nbrs_gpu = GPUNearestNeighbors(n_neighbors=1, use_gpu=True)
    nbrs_gpu.fit(target)
    dist_gpu, idx_gpu = nbrs_gpu.kneighbors(query)
    
    # Results should match (within floating point tolerance)
    np.testing.assert_allclose(dist_cpu, dist_gpu, rtol=1e-5)
    np.testing.assert_array_equal(idx_cpu, idx_gpu)

def test_jit_grid_accumulation():
    """Test JIT-compiled grid accumulation matches NumPy version."""
    # ... similar structure ...
```

### Benchmarking

```python
# scripts/benchmark_gpu.py
import time
import numpy as np
from terrain_change_detection.acceleration import GPUNearestNeighbors
from sklearn.neighbors import NearestNeighbors

def benchmark_nn_search(n_target, n_query, dimensions=3):
    """Benchmark nearest neighbor search."""
    target = np.random.rand(n_target, dimensions).astype(np.float32)
    query = np.random.rand(n_query, dimensions).astype(np.float32)
    
    # CPU
    t0 = time.time()
    nbrs_cpu = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
    nbrs_cpu.fit(target)
    dist_cpu, idx_cpu = nbrs_cpu.kneighbors(query)
    cpu_time = time.time() - t0
    
    # GPU
    t0 = time.time()
    nbrs_gpu = GPUNearestNeighbors(n_neighbors=1, use_gpu=True)
    nbrs_gpu.fit(target)
    dist_gpu, idx_gpu = nbrs_gpu.kneighbors(query)
    gpu_time = time.time() - t0
    
    speedup = cpu_time / gpu_time
    print(f"n_target={n_target:,}, n_query={n_query:,}")
    print(f"  CPU: {cpu_time:.3f}s")
    print(f"  GPU: {gpu_time:.3f}s")
    print(f"  Speedup: {speedup:.2f}x")
    return speedup

if __name__ == "__main__":
    # Run benchmarks at different scales
    for n in [1_000, 10_000, 100_000, 1_000_000]:
        benchmark_nn_search(n, n//10)
        print()
```

## Common Issues and Solutions

### Issue 1: CUDA Out of Memory

**Symptom**: `CuPy: out of memory` error

**Solution**:
```python
# Reduce chunk size or enable CPU fallback
from terrain_change_detection.acceleration import clear_gpu_memory

clear_gpu_memory()  # Free unused memory

# Or process in smaller chunks
chunk_size = 100_000  # Reduce if OOM
for i in range(0, len(points), chunk_size):
    chunk = points[i:i+chunk_size]
    # process chunk
```

### Issue 2: Slow First Run

**Symptom**: First GPU operation is very slow

**Solution**: JIT compilation and GPU initialization occur on first run
```python
# Warmup GPU
import cupy as cp
_ = cp.sum(cp.array([1, 2, 3]))  # Trigger initialization
```

### Issue 3: Numerical Differences GPU vs CPU

**Symptom**: Results differ slightly between GPU and CPU

**Solution**: Use appropriate tolerances in tests
```python
# Don't expect exact equality
np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-5, atol=1e-8)
```

## Next Steps

1. Implement hardware detection module
2. Create array backend abstraction
3. Port nearest neighbor search to GPU
4. Add JIT compilation to grid operations
5. Integrate into ICP and DoD workflows
6. Benchmark and optimize
7. Document performance gains

---

**Status**: Implementation guide v1.0  
**Last Updated**: 2025-01-05
