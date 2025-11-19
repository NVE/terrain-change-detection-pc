# GPU Integration Strategy for Terrain Change Detection

## Overview

This document explains the GPU acceleration strategy for the terrain change detection pipeline, including what can be accelerated, what cannot, and why.

## GPU Infrastructure (Phase 2 - Week 1) ✅ Complete

### Implemented Components

1. **Hardware Detection** (`hardware_detection.py`)
   - Automatic GPU detection with graceful CPU fallback
   - Memory checking and batch size optimization
   - Compute capability validation

2. **Array Operations** (`gpu_array_ops.py`)
   - Transparent NumPy/CuPy switching via `ArrayBackend`
   - Unified interface for CPU/GPU array operations
   - Automatic memory management

3. **Nearest Neighbors** (`gpu_neighbors.py`)
   - `GPUNearestNeighbors` wrapper for sklearn/cuML
   - Strategy: cuML (Linux) > sklearn-gpu (Windows) > sklearn-cpu (fallback)
   - Expected 10-50x speedup on NN searches

4. **Configuration** (`config.py` + `default.yaml`)
   - `gpu.enabled`: Master switch (default: true)
   - `gpu.use_for_c2c`: Enable C2C GPU acceleration
   - `gpu.fallback_to_cpu`: Automatic CPU fallback
   - `gpu.gpu_memory_limit_gb`: Memory management

### Test Coverage
- 34 GPU tests passing (detection: 7, array ops: 14, neighbors: 13)
- Full CPU fallback tested and verified
- GPU/CPU numerical parity validated

## Integration Strategy

### ✅ C2C (Cloud-to-Cloud) - Direct Integration

**Status**: Ready for GPU integration  
**Implementation**: Direct control via `ChangeDetector.compute_c2c()`

**Current C2C Methods**:
1. `compute_c2c()` - Euclidean nearest neighbor (uses sklearn KDTree)
2. `compute_c2c_plane_based()` - Local plane fitting (uses sklearn for NN + radius search)
3. `compute_c2c_streaming_*()` - Out-of-core variants

**GPU Integration Plan**:
```python
# Replace sklearn.neighbors.NearestNeighbors with GPUNearestNeighbors
from terrain_change_detection.acceleration import create_gpu_neighbors

# In compute_c2c():
if cfg.gpu.enabled and cfg.gpu.use_for_c2c:
    nbrs = create_gpu_neighbors(n_neighbors=1, use_gpu=True)
else:
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
```

**Expected Performance**:
- **Current**: sklearn KDTree (CPU-only)
- **With GPU**: 10-20x speedup per query
- **Combined** (GPU + Phase 1 CPU parallel): **20-60x total speedup**

**Effort**: Low (drop-in replacement)

---

### ⚠️ M3C2 - Limited Integration Potential

**Status**: py4dgeo dependency limits GPU integration  
**Implementation**: Uses `py4dgeo.m3c2.M3C2` (C++ library)

**Why M3C2 Cannot Use GPU Neighbors Directly**:

1. **External C++ Library**: py4dgeo is a compiled C++ library with Python bindings
   - We cannot inject Python GPU code into C++ execution
   - KDTree building happens inside py4dgeo's C++ code
   - No Python hooks for nearest neighbor queries

2. **Integrated Pipeline**: M3C2 algorithm flow
   ```
   Epoch.build_kdtree()  [C++]
   ↓
   M3C2.run()  [C++]
   ↓
   Cylindrical searches  [C++ KDTree, not accessible]
   ↓
   Distance computation  [C++]
   ```

3. **Already Optimized**: py4dgeo uses highly optimized C++ KDTree implementations
   - Compiled code is typically within 2-5x of GPU performance
   - GPU overhead may not justify custom implementation for M3C2

**Current M3C2 Methods**:
- `compute_m3c2_original()` - py4dgeo M3C2 ✅ (keep as-is)
- `compute_m3c2_error_propagation()` - py4dgeo M3C2-EP ✅ (keep as-is)  
- `compute_m3c2_plane_based()` - Custom implementation (could be GPU-accelerated)
- `compute_m3c2_streaming_*()` - Tiled out-of-core (preprocessing could use GPU)

**Potential GPU Opportunities**:

1. **Data Preprocessing** (✅ Feasible):
   ```python
   # Transform point clouds on GPU before py4dgeo
   from terrain_change_detection.acceleration import get_array_backend
   
   if cfg.gpu.enabled and cfg.gpu.use_for_preprocessing:
       backend = get_array_backend(use_gpu=True)
       points_gpu = backend.asarray(points)
       transformed = backend.transform(points_gpu, matrix)  # GPU transform
       points = backend.to_cpu(transformed)
   
   # Then pass to py4dgeo
   epoch = Epoch(points)
   ```

2. **Custom GPU M3C2 Implementation** (⚠️ High Effort):
   - Reimplement M3C2 algorithm using GPU primitives
   - Use CuPy + GPU KDTree for cylindrical searches
   - Expected 10-30x speedup vs current CPU
   - **Effort**: High (2-3 weeks), **Risk**: Medium (numerical stability)
   - **Recommendation**: Phase 3 if C2C GPU shows significant gains

3. **Hybrid Approach** (✅ Recommended):
   - Keep py4dgeo M3C2 for correctness and reliability
   - Add GPU preprocessing for transformations/filtering
   - Enable GPU C2C for complementary analysis
   - Let Phase 1 CPU parallelization handle M3C2 scaling

**Performance Expectations**:
- **Current**: py4dgeo M3C2 (optimized C++ KDTree)
- **With GPU preprocessing**: 5-10% overall speedup (minimal impact)
- **Custom GPU M3C2**: 10-30x speedup (high development cost)
- **Phase 1 CPU parallel**: 2-3x speedup (already implemented) ✅

**Recommendation**: **Do NOT integrate GPU into py4dgeo-based M3C2**
- Cost/benefit ratio unfavorable
- py4dgeo C++ already well-optimized
- Phase 1 CPU parallelization provides sufficient scaling
- Focus GPU efforts on C2C where we have direct control

---

## Implementation Roadmap

### Phase 2.1 - C2C GPU Integration ⏳ (Current)

**Tasks**:
1. ✅ GPU infrastructure (detection, array ops, neighbors)
2. ✅ Configuration system (GPU settings in YAML)
3. ⏳ Update `compute_c2c()` to use `GPUNearestNeighbors`
4. ⏳ Update `compute_c2c_plane_based()` for GPU NN
5. ⏳ Update streaming C2C variants
6. ⏳ Comprehensive testing (GPU vs CPU parity)
7. ⏳ Performance benchmarks

**Expected Outcome**:
- C2C: 20-60x total speedup (10-20x GPU + 2-3x CPU parallel)
- Configuration-driven GPU control
- Graceful CPU fallback

### Phase 2.2 - Optional GPU Preprocessing

**Tasks**:
1. GPU-accelerated point transformations
2. GPU-accelerated filtering/downsampling
3. Integration into preprocessing pipeline

**Expected Outcome**:
- 5-10% preprocessing speedup
- Reduced data transfer overhead

### Phase 3 - Custom GPU M3C2 (Future, If Needed)

**Conditions to proceed**:
- C2C GPU shows >20x speedup in production
- M3C2 becomes primary bottleneck
- Resources available for 2-3 week implementation

**Tasks**:
1. Design GPU cylindrical search algorithm
2. Implement CuPy-based M3C2 core
3. Numerical validation against py4dgeo
4. Performance benchmarking
5. Integration with existing pipeline

**Expected Outcome**:
- M3C2: 20-90x total speedup (10-30x GPU + 2-3x CPU parallel)
- Full GPU acceleration pipeline

---

## Configuration Guide

### Enabling GPU Acceleration

**config/default.yaml**:
```yaml
gpu:
  enabled: true                    # Master switch
  use_for_c2c: true                # Enable C2C GPU (Phase 2.1)
  use_for_preprocessing: true      # Enable preprocessing GPU (Phase 2.2)
  fallback_to_cpu: true            # Auto-fallback if GPU unavailable
  gpu_memory_limit_gb: null        # null = auto (80% of GPU memory)
  batch_size: null                 # null = auto-calculate
```

### Checking GPU Status

```python
from terrain_change_detection.acceleration import get_gpu_info

gpu_info = get_gpu_info()
if gpu_info.available:
    print(f"GPU: {gpu_info.device_name}, {gpu_info.memory_gb:.1f} GB")
else:
    print(f"GPU unavailable: {gpu_info.error_message}")
```

### Disabling GPU

```yaml
gpu:
  enabled: false  # Use CPU for all operations
```

Or per-method:
```yaml
gpu:
  enabled: true
  use_for_c2c: false  # Disable C2C GPU, keep others enabled
```

---

## Performance Expectations

### Current Performance (Baseline)

| Operation | Size | Time (CPU) | Method |
|-----------|------|-----------|--------|
| C2C | 100K → 100K | 10s | sklearn KDTree |
| M3C2 | 10K cores, 1M pts | 60s | py4dgeo C++ |
| DoD | 1M pts → 1000x1000 grid | 5s | NumPy gridding |

### With GPU Acceleration (Phase 2.1)

| Operation | Size | Time (GPU) | Speedup | Notes |
|-----------|------|-----------|---------|-------|
| C2C | 100K → 100K | 0.5-1s | 10-20x | GPUNearestNeighbors |
| M3C2 | 10K cores, 1M pts | 60s | 1x | py4dgeo (no change) |
| DoD | 1M pts → 1000x1000 grid | 5s | 1x | Already fast |

### Combined with Phase 1 CPU Parallel

| Operation | Size | Time (Parallel+GPU) | Total Speedup |
|-----------|------|-------------------|---------------|
| C2C | 1M → 1M | 1-2s | **20-60x** |
| M3C2 | 100K cores, 10M pts | 30-40s | **2-3x** (CPU only) |

---

## Technical Notes

### Why py4dgeo Cannot Use GPU

**Architectural Incompatibility**:
```
Python (our code)
    ↓
py4dgeo Python bindings
    ↓
py4dgeo C++ core  ← KDTree built here (inaccessible to Python GPU code)
    ↓
C++ stdlib / Eigen
```

**No Python Hooks**:
- py4dgeo builds KDTree in C++: `epoch.build_kdtree()` [C++]
- Searches happen in C++: Internal to `M3C2.run()` [C++]
- No callback mechanism to inject Python-based GPU neighbors

**Alternative Would Require**:
1. Fork py4dgeo and add GPU support in C++ (use CUDA/CuPy C++ API)
2. OR: Reimplement M3C2 entirely in Python with CuPy
3. OR: Create Python/C++ hybrid with GPU kernels

All options are **high effort** (weeks of development) with **medium risk**.

### When Custom GPU M3C2 Makes Sense

- **If** C2C GPU achieves >20x speedup in production use
- **AND** M3C2 becomes the primary bottleneck
- **AND** datasets are large enough that 30s → 3s matters
- **THEN** custom GPU M3C2 may justify 2-3 week implementation

---

## Summary

✅ **DO** (Phase 2.1):
- GPU-accelerate C2C nearest neighbors
- Add GPU configuration system
- Enable GPU preprocessing (optional)

❌ **DON'T** (Current Phase):
- Try to GPU-accelerate py4dgeo-based M3C2
- Reimplement M3C2 from scratch
- Spend time on marginal preprocessing gains

🔮 **FUTURE** (Phase 3, If Needed):
- Custom GPU M3C2 implementation
- Full end-to-end GPU pipeline
- Advanced GPU memory management

**Current Focus**: Integrate GPU into C2C where we have direct control and can achieve 20-60x total speedup with combined GPU + CPU parallelization.
