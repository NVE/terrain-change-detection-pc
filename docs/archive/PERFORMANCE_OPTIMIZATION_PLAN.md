# Performance Optimization Plan for Terrain Change Detection

**Date**: January 2025  
**Branch**: `feat/gpu-acceleration`  
**Goal**: Enable country/region-scale terrain change detection with GPU acceleration and scalable out-of-core processing

## Executive Summary

This document outlines a comprehensive strategy to optimize the terrain change detection pipeline for large-scale operations (regional to national scale). The optimization focuses on three main pillars:

1. **GPU Acceleration**: Leverage CUDA-enabled GPUs for compute-intensive operations
2. **Out-of-Core Processing**: Enhance tiling system for memory-efficient processing of massive datasets
3. **Parallel Processing**: Utilize multi-core CPUs and distributed computing frameworks

## 1. Current Architecture Analysis

### 1.1 Identified Bottlenecks

**Critical Performance Bottlenecks:**

1. **Nearest Neighbor Searches (70-80% of compute time)**
   - ICP alignment: Building KD-trees and iterative NN queries on millions of points
   - C2C distance computation: NN search for every point in source cloud
   - M3C2: Cylindrical neighborhood searches for each core point
   - Current: CPU-only with sklearn's KD-tree

2. **Array Operations (15-20% of compute time)**
   - DoD gridding: Binning millions of points into grid cells
   - Distance calculations: Euclidean distances, squared differences
   - Statistical aggregations: Mean, median, percentiles per cell
   - Current: Pure NumPy without vectorization optimization

3. **Sequential Processing (5-10% overhead)**
   - Tiles processed one-by-one despite independence
   - No parallel I/O for LAZ file reading
   - Current: Single-threaded tiling loop

4. **Memory Limitations**
   - Loading entire point clouds into RAM
   - Limited by available system memory (~32-64 GB typical)
   - Current streaming DoD is prototype-only

### 1.2 Hardware Resource Utilization

**Current State:**
- CPU: Multi-core available but underutilized (sequential loops)
- GPU: Not utilized at all
- Memory: Conservative approach, loads full datasets when possible
- Disk I/O: Sequential LAZ reading, no prefetching

**Target State:**
- CPU: All cores active for parallel tile processing
- GPU: 80%+ utilization for NN searches and array operations
- Memory: Constant footprint via streaming + tiling
- Disk I/O: Parallel prefetching with multiple workers

## 2. GPU Acceleration Strategy

### 2.1 Technology Stack Selection

**Primary GPU Framework: CuPy + RAPIDS cuML**
- **CuPy**: NumPy-compatible GPU arrays, minimal code changes
- **cuML**: GPU-accelerated nearest neighbor algorithms (NearestNeighbors)
- **cuSpatial**: GPU point cloud spatial operations (future)

**Alternative/Complementary:**
- **PyTorch3D**: For advanced 3D operations (M3C2 normal estimation)
- **Open3D-GPU**: GPU-accelerated ICP (compile from source with CUDA)
- **Numba CUDA**: Custom CUDA kernels for specialized operations

**Rationale:**
- CuPy provides drop-in replacement for NumPy operations
- cuML's NearestNeighbors matches sklearn API
- RAPIDS ecosystem well-supported for data science workloads
- Gradual migration: start with CuPy, add cuML, then custom kernels

### 2.2 GPU-Accelerated Operations

#### Priority 1: Nearest Neighbor Searches
```python
# Current (CPU)
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target)
distances, indices = nbrs.kneighbors(source)

# Target (GPU with fallback)
from terrain_change_detection.acceleration import GPUNearestNeighbors
nbrs = GPUNearestNeighbors(n_neighbors=1, use_gpu=True, fallback_to_cpu=True)
nbrs.fit(target)
distances, indices = nbrs.kneighbors(source)
```

**Implementation:**
- Abstraction layer with unified API
- Automatic GPU detection and memory management
- Chunked processing if data > GPU memory
- Transparent CPU fallback

#### Priority 2: Array Operations (DoD, Distance Calculations)
```python
# Target: Backend-agnostic array operations
import terrain_change_detection.acceleration.array_ops as xp

def compute_distances_gpu(source, target):
    # xp automatically dispatches to CuPy or NumPy
    diff = xp.subtract(source[:, None, :], target[None, :, :])
    distances = xp.sqrt(xp.sum(diff ** 2, axis=2))
    return distances
```

**Key Operations for GPU:**
- Point cloud transformations (ICP, coarse alignment)
- Grid binning and aggregation (DoD)
- Distance matrix computations
- Statistical reductions (mean, median, percentile)

#### Priority 3: M3C2 Acceleration
- Normal estimation: GPU-accelerated PCA on neighborhoods
- Cylindrical search: Custom CUDA kernel or cuML radius search
- Distance aggregation: CuPy array operations

**Note:** py4dgeo doesn't support GPU natively; we'll need custom implementation or contribute upstream.

### 2.3 Memory Management Strategy

**GPU Memory Hierarchy:**
```
┌─────────────────────────────────────┐
│   Unified Memory Manager            │
├─────────────────────────────────────┤
│ GPU Memory Pool (e.g., 8-24 GB)     │
│ - KD-tree structures                │
│ - Active tile data                  │
│ - Intermediate results              │
├─────────────────────────────────────┤
│ CPU Memory (32-128 GB)              │
│ - File I/O buffers                  │
│ - Tile queue                        │
│ - Final results                     │
├─────────────────────────────────────┤
│ Disk (TB scale)                     │
│ - LAZ files                         │
│ - Intermediate checkpoints          │
└─────────────────────────────────────┘
```

**Strategies:**
1. **Chunked Transfer**: Stream data to GPU in manageable chunks
2. **Pinned Memory**: Use page-locked memory for faster CPU-GPU transfers
3. **Double Buffering**: Overlap computation with data transfer
4. **Memory Pooling**: Reuse GPU allocations across tiles

## 3. Enhanced Out-of-Core Processing

### 3.1 Improved Tiling Architecture

**Current System:**
- Sequential tile processing
- Fixed tile size
- Halo for overlap handling

**Enhanced System:**
```
┌──────────────────────────────────────────┐
│         Tile Scheduler                   │
│  - Dynamic tile sizing based on density  │
│  - Priority queue (high-change areas)    │
│  - Load balancing across workers         │
└──────────────────────────────────────────┘
           │
           ├─────────┬─────────┬─────────┐
           ▼         ▼         ▼         ▼
      Worker 1   Worker 2   Worker 3  Worker N
      (GPU 0)    (GPU 0)    (GPU 1)   (CPU)
```

**Features:**
1. **Adaptive Tiling**: Smaller tiles in dense areas, larger in sparse
2. **Parallel Execution**: Process multiple independent tiles simultaneously
3. **GPU Affinity**: Assign tiles to specific GPU devices
4. **Checkpoint/Resume**: Save intermediate results for fault tolerance

### 3.2 I/O Optimization

**Parallel LAZ Reading:**
```python
# Use concurrent futures for parallel file reading
from concurrent.futures import ThreadPoolExecutor

def prefetch_tiles(tile_queue, files, workers=4):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for tile in tile_queue:
            future = executor.submit(read_tile_data, files, tile.bounds)
            futures.append((tile, future))
        for tile, future in futures:
            yield tile, future.result()
```

**Optimizations:**
- **PDAL Integration**: Use PDAL for efficient spatial filtering
- **LAZ Streaming**: Read only points within tile bounds (avoid full decompression)
- **Caching**: Cache frequently accessed tiles (border tiles)
- **SSD/NVMe**: Recommend fast storage for intermediate results

### 3.3 Memory-Mapped Processing

For extremely large rasters (DoD outputs):
```python
import numpy as np

# Memory-mapped array for out-of-core aggregation
mosaic = np.memmap('temp_dod.dat', dtype=np.float64, 
                   mode='w+', shape=(global_ny, global_nx))
```

## 4. Parallel Processing Strategy

### 4.1 Multi-Level Parallelism

**Level 1: Tile-Level Parallelism**
```python
from multiprocessing import Pool
from functools import partial

def process_tile_worker(tile, files_t1, files_t2, config):
    # Each tile is independent
    result = process_single_tile(tile, files_t1, files_t2, config)
    return tile.id, result

def process_all_tiles_parallel(tiles, files_t1, files_t2, config, workers=None):
    workers = workers or os.cpu_count()
    with Pool(processes=workers) as pool:
        worker_fn = partial(process_tile_worker, 
                           files_t1=files_t1, files_t2=files_t2, config=config)
        results = pool.map(worker_fn, tiles)
    return dict(results)
```

**Level 2: Operation-Level Parallelism (Dask)**
```python
import dask.array as da

# Dask for large array operations
large_array = da.from_array(numpy_array, chunks=(10000, 3))
result = large_array.mean(axis=0).compute()
```

**Level 3: GPU-Level Parallelism**
- Multi-GPU support: Distribute tiles across available GPUs
- GPU streams: Overlap kernel execution with memory transfers

### 4.2 Process vs Thread Selection

**Use Multiprocessing for:**
- Tile processing (CPU-bound, independent)
- File I/O (parallel decompression)
- M3C2 computations (py4dgeo workers)

**Use Threading for:**
- I/O operations (reading LAZ in background)
- GPU kernel launches (already async)
- Lightweight coordination tasks

**Avoid:**
- Mixing processes and CUDA (careful with fork/spawn on Linux/Windows)
- Thread pool for CPU-heavy operations (GIL limitations)

## 5. JIT Compilation with Numba

### 5.1 Target Functions for JIT

**Grid Accumulation (DoD):**
```python
import numba

@numba.jit(nopython=True, parallel=True, cache=True)
def accumulate_points_jit(points, bounds, cell_size, sum_grid, cnt_grid):
    min_x, min_y = bounds[0], bounds[1]
    nx, ny = sum_grid.shape[1], sum_grid.shape[0]
    
    for i in numba.prange(len(points)):
        x, y, z = points[i, 0], points[i, 1], points[i, 2]
        xi = int((x - min_x) / cell_size)
        yi = int((y - min_y) / cell_size)
        
        if 0 <= xi < nx and 0 <= yi < ny:
            sum_grid[yi, xi] += z
            cnt_grid[yi, xi] += 1
```

**Distance Calculations:**
```python
@numba.jit(nopython=True, parallel=True)
def compute_distances_jit(source, target_kdtree_indices, target_points):
    n = len(source)
    distances = np.empty(n, dtype=np.float64)
    
    for i in numba.prange(n):
        idx = target_kdtree_indices[i]
        dx = source[i, 0] - target_points[idx, 0]
        dy = source[i, 1] - target_points[idx, 1]
        dz = source[i, 2] - target_points[idx, 2]
        distances[i] = math.sqrt(dx*dx + dy*dy + dz*dz)
    
    return distances
```

**Transformation Application:**
```python
@numba.jit(nopython=True, parallel=True)
def apply_transform_jit(points, R, t):
    n = len(points)
    result = np.empty_like(points)
    
    for i in numba.prange(n):
        for j in range(3):
            result[i, j] = (R[j, 0] * points[i, 0] + 
                           R[j, 1] * points[i, 1] + 
                           R[j, 2] * points[i, 2] + t[j])
    return result
```

**Expected Speedup:** 2-10x for these operations on multi-core CPUs

### 5.2 Numba CUDA Kernels (Advanced)

For custom GPU operations not available in CuPy/cuML:

```python
from numba import cuda

@cuda.jit
def cylindrical_search_kernel(core_points, cloud, normals, radius, max_depth, 
                               output_distances):
    idx = cuda.grid(1)
    if idx < core_points.shape[0]:
        # Custom M3C2 cylindrical search logic
        # ... (implementation details)
        pass
```

## 6. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
**Goal**: Establish GPU infrastructure and abstraction layer

1. ✅ Create performance optimization plan document
2. ⬜ Implement GPU detection and configuration system
   - Add `gpu` section to YAML config
   - Device selection, memory limits
   - Backend preference order (CUDA > CPU)
3. ⬜ Create array backend abstraction (`array_ops.py`)
   - Unified API for NumPy/CuPy
   - Automatic dispatch based on config
   - Memory transfer helpers
4. ⬜ Add dependencies to `pyproject.toml`
   - CuPy (optional, CUDA 11.x/12.x)
   - cuML (optional)
   - Numba (required for JIT)
   - Dask (optional for distributed)

### Phase 2: GPU Nearest Neighbor (Weeks 3-4)
**Goal**: Accelerate KD-tree operations

1. ⬜ Implement `GPUNearestNeighbors` class
   - Wrapper for cuML's NearestNeighbors
   - Automatic chunking for large datasets
   - CPU fallback logic
2. ⬜ Integrate into ICP registration
   - Replace sklearn in `fine_registration.py`
   - Benchmark CPU vs GPU performance
3. ⬜ Integrate into C2C distance computation
   - Update `change_detection.py`
   - Test on large point clouds (10M+ points)
4. ⬜ Performance testing and tuning
   - Measure speedup on various GPU types
   - Optimize chunk sizes

### Phase 3: JIT Compilation (Week 5)
**Goal**: Accelerate CPU-bound array operations

1. ⬜ Add Numba JIT to grid accumulation
   - Update `GridAccumulator` in `tiling.py`
   - Benchmark vs current NumPy version
2. ⬜ JIT-compile distance calculations
   - C2C fallback path
   - Transform applications
3. ⬜ JIT-compile statistical aggregators
   - Median, percentile reducers
   - Custom binning operations

### Phase 4: Parallel Tiling (Weeks 6-7)
**Goal**: Enable distributed tile processing

1. ⬜ Implement parallel tile scheduler
   - Process pool with configurable workers
   - GPU affinity assignment
2. ⬜ Add checkpoint/resume capability
   - Save intermediate tile results
   - Resume from partial runs
3. ⬜ Integrate Dask for large-scale operations
   - Optional backend for massive datasets
   - Distributed processing on clusters
4. ⬜ Optimize I/O pipeline
   - Parallel LAZ reading
   - Prefetching and caching

### Phase 5: GPU Array Operations (Week 8)
**Goal**: Accelerate DoD and distance computations

1. ⬜ Port DoD gridding to CuPy
   - GPU-accelerated binning
   - Statistical aggregations on GPU
2. ⬜ GPU-based distance matrix computations
   - Alternative to NN search for small clouds
   - Memory-efficient chunked approach
3. ⬜ Integration testing
   - End-to-end workflow on GPU
   - Memory profiling

### Phase 6: M3C2 Optimization (Weeks 9-10)
**Goal**: Accelerate M3C2 computations

1. ⬜ GPU-accelerated normal estimation
   - PyTorch3D or custom CUDA kernel
   - Batch processing of neighborhoods
2. ⬜ Custom cylindrical search on GPU
   - Numba CUDA or CuPy implementation
   - Compare with py4dgeo performance
3. ⬜ Evaluate full GPU M3C2 pipeline
   - Consider contributing to py4dgeo
   - Or maintain separate GPU implementation

### Phase 7: Testing & Benchmarking (Week 11)
**Goal**: Validate performance gains

1. ⬜ Create comprehensive benchmark suite
   - Various point cloud sizes (1K to 100M points)
   - Different hardware configs (CPU-only, single GPU, multi-GPU)
   - Measure: throughput, memory usage, time-to-result
2. ⬜ Regression testing
   - Ensure numerical accuracy matches CPU version
   - Tolerance levels for floating-point differences
3. ⬜ Real-world testing
   - Process Akershus dataset (full region)
   - Compare with original workflow
   - Document speedup factors

### Phase 8: Documentation & Deployment (Week 12)
**Goal**: Production-ready optimization

1. ⬜ Update user documentation
   - Hardware requirements
   - GPU setup guides (CUDA installation)
   - Performance tuning recommendations
2. ⬜ Create optimization guide
   - Configuration best practices
   - Troubleshooting common issues
3. ⬜ Package and release
   - Merge to main branch
   - Tag release version
   - Publish benchmarks

## 7. Configuration Schema

### 7.1 Extended YAML Configuration

```yaml
performance:
  # CPU settings
  numpy_threads: auto  # int or 'auto'
  cpu_workers: auto    # for multiprocessing

  # GPU settings
  gpu:
    enabled: true
    backend: cuda  # cuda | rocm | cpu
    device_ids: [0]  # list of GPU devices to use
    memory_limit_gb: 8  # per-device limit
    fallback_to_cpu: true
    
  # JIT compilation
  jit:
    enabled: true
    cache: true
    parallel: true
    
  # Out-of-core tiling
  outofcore:
    enabled: true
    tile_size_m: 500.0
    halo_m: 20.0
    chunk_points: 1000000
    parallel_tiles: 4  # number of tiles to process concurrently
    checkpoint_dir: null  # optional checkpoint directory
    
  # Advanced
  prefetch_tiles: 2  # number of tiles to prefetch
  use_dask: false    # enable Dask for large-scale operations
```

### 7.2 Runtime Hardware Detection

```python
from terrain_change_detection.acceleration import HardwareInfo

info = HardwareInfo.detect()
print(f"GPUs available: {info.gpu_count}")
print(f"GPU memory: {info.gpu_memory_gb}")
print(f"CUDA version: {info.cuda_version}")
print(f"CPU cores: {info.cpu_count}")
```

## 8. Expected Performance Gains

### 8.1 Theoretical Speedup Estimates

Based on literature and similar GPU-accelerated point cloud projects:

| Operation | Current (CPU) | GPU Target | JIT Target | Combined |
|-----------|--------------|-----------|-----------|----------|
| KD-tree NN search | 1x | 10-50x | 2-3x | 50x |
| ICP alignment | 1x | 15-30x | 2x | 30x |
| DoD gridding | 1x | 5-10x | 3-5x | 15x |
| C2C distances | 1x | 20-40x | 2x | 40x |
| M3C2 (custom) | 1x | 10-20x | 2x | 20x |
| **Overall pipeline** | 1x | **10-25x** | **2-4x** | **25x** |

### 8.2 Real-World Scenarios

**Scenario 1: Small Area (1 km²)**
- Point density: 2 pts/m² → 2M points per epoch
- Current time: ~5 minutes
- GPU-accelerated: ~20 seconds
- **Speedup: 15x**

**Scenario 2: Municipality (100 km²)**
- Point density: 2 pts/m² → 200M points per epoch
- Current time: ~8 hours (out-of-memory or crashes)
- GPU-accelerated + tiling: ~25 minutes
- **Speedup: 20x (with improved memory efficiency)**

**Scenario 3: County/Region (10,000 km²)**
- Point density: 2 pts/m² → 20B points per epoch
- Current time: Infeasible (memory exhaustion)
- GPU-accelerated + distributed: ~40 hours on 4-GPU workstation
- **Enables previously impossible analysis**

**Scenario 4: Country (Norway, 385,000 km²)**
- Point density: 2 pts/m² → 770B points per epoch
- GPU-accelerated + HPC cluster (32 GPUs): ~120 hours (~5 days)
- **Achievable with proper infrastructure**

## 9. Hardware Recommendations

### 9.1 Development Workstation

**Minimum:**
- GPU: NVIDIA RTX 3060 (12 GB VRAM)
- CPU: 8-core (Intel i7/AMD Ryzen 7)
- RAM: 32 GB
- Storage: 1 TB NVMe SSD

**Recommended:**
- GPU: NVIDIA RTX 4080/4090 (16-24 GB VRAM)
- CPU: 16-core (Intel i9/AMD Ryzen 9)
- RAM: 64 GB
- Storage: 2 TB NVMe SSD + 10 TB HDD

**Optimal (Multi-GPU):**
- GPUs: 2x NVIDIA A6000 (48 GB VRAM each)
- CPU: 32-core Threadripper/Xeon
- RAM: 128 GB
- Storage: 4 TB NVMe RAID + 50 TB storage array

### 9.2 Production HPC Cluster

For national-scale processing:
- 8-16 compute nodes
- 4 GPUs per node (NVIDIA A100/H100)
- 80 GB VRAM per GPU
- High-bandwidth interconnect (InfiniBand)
- Parallel filesystem (Lustre/GPFS)

## 10. Risk Mitigation

### 10.1 Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| GPU memory exhaustion | High | Implement automatic chunking, fallback to CPU |
| Numerical precision differences | Medium | Extensive validation, tolerance thresholds |
| CUDA/CuPy installation complexity | Medium | Provide Docker containers, detailed setup docs |
| Performance not meeting expectations | Medium | Incremental development, continuous benchmarking |
| py4dgeo compatibility issues | Low | Maintain separate GPU M3C2 implementation |

### 10.2 Compatibility Risks

**Platform Support:**
- Windows: Full support (CUDA, CPU)
- Linux: Full support (CUDA, ROCm, CPU)
- macOS: CPU-only (no CUDA), limited testing

**Python Version:**
- Target: Python 3.10+ (current: 3.13)
- CuPy/cuML: May not support latest Python immediately
- Strategy: Support Python 3.10-3.12 range

## 11. Success Metrics

### 11.1 Quantitative Metrics

1. **Throughput**: Points processed per second
   - Target: >1M points/sec (vs current ~50K pts/sec)

2. **Memory Efficiency**: Maximum dataset size
   - Target: Process 1B+ points with 64 GB RAM + 24 GB VRAM

3. **Speedup Factor**: GPU vs CPU for key operations
   - Target: 10-25x overall pipeline speedup

4. **Scalability**: Performance vs data size
   - Target: Sub-linear time complexity with tiling

### 11.2 Qualitative Metrics

1. **Usability**: Easy configuration, automatic hardware detection
2. **Reliability**: Graceful degradation, no crashes
3. **Maintainability**: Clean abstractions, well-documented code
4. **Reproducibility**: Deterministic results across hardware

## 12. References and Resources

### 12.1 Academic References

1. **GPU-Accelerated Point Cloud Processing:**
   - Bolitho et al. (2009): "Parallel Poisson Surface Reconstruction"
   - Zhou & Neumann (2010): "Fast and Extensible Building Modeling from Airborne LiDAR Data"
   
2. **Change Detection Algorithms:**
   - Lague et al. (2013): "Accurate 3D comparison of complex topography with terrestrial laser scanner"
   - Winiwarter et al. (2021): "M3C2-EP: Pushing the limits of 3D topographic point cloud change detection"

3. **Out-of-Core Processing:**
   - Schütz et al. (2020): "Fast Out-of-Core Octree Generation for Massive Point Clouds"
   - Elseberg et al. (2013): "One billion points in the cloud"

### 12.2 Software Documentation

- RAPIDS cuML: https://docs.rapids.ai/api/cuml/stable/
- CuPy: https://docs.cupy.dev/
- Numba: https://numba.pydata.org/
- PDAL: https://pdal.io/
- py4dgeo: https://py4dgeo.readthedocs.io/

### 12.3 Benchmark Datasets

- **Synthetic dataset** (current): Testing infrastructure
- **Akershus 2015-2020** (current): Real-world validation
- **OpenTopography**: Public LiDAR datasets for benchmarking
- **ISPRS Benchmark**: Standardized point cloud processing tests

## 13. Next Steps

### Immediate Actions (This Week)

1. ✅ **Complete this planning document**
2. ⬜ **Set up GPU development environment**
   - Install CUDA toolkit
   - Install CuPy, cuML
   - Verify GPU is accessible from Python

3. ⬜ **Create feature branch: `feat/gpu-acceleration`**
4. ⬜ **Implement basic GPU detection module**
5. ⬜ **Create initial benchmarking harness**

### First Sprint (Next 2 Weeks)

- Implement array backend abstraction
- Port nearest neighbor search to GPU
- Benchmark ICP with GPU acceleration
- Document performance gains

---

**Document Status**: Draft v1.0  
**Last Updated**: 2025-01-05  
**Author**: AI Assistant + Yared  
**Review Status**: Pending technical review
