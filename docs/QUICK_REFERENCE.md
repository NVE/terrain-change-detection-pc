# GPU Acceleration Quick Reference

One-page reference for the performance optimization project.

## Current Bottlenecks

| Operation | % of Time | Current Impl | Target |
|-----------|-----------|--------------|--------|
| Nearest Neighbor | 70-80% | sklearn CPU | cuML GPU |
| Array Ops (DoD) | 15-20% | NumPy | CuPy + Numba JIT |
| Tiling | 5-10% | Sequential | Parallel |

## Expected Speedups

| Operation | GPU | JIT | Combined |
|-----------|-----|-----|----------|
| NN Search | 10-50x | 2-3x | 50x |
| ICP | 15-30x | 2x | 30x |
| DoD | 5-10x | 3-5x | 15x |
| C2C | 20-40x | 2x | 40x |
| **Pipeline** | **10-25x** | **2-4x** | **25x** |

## Technology Stack

```
Primary:
  - CuPy: NumPy → GPU arrays
  - cuML: sklearn → GPU ML (NN search)
  - Numba: JIT compilation + CUDA kernels

Optional:
  - PyTorch3D: Advanced 3D ops
  - Open3D-GPU: GPU ICP
  - Dask: Distributed computing
```

## Module Architecture

```
acceleration/
├── hardware.py          # GPU detection & config
├── array_ops.py         # NumPy/CuPy abstraction  
├── nearest_neighbors.py # sklearn/cuML wrapper
└── jit_kernels.py       # Numba JIT functions
```

## Quick Start Commands

```bash
# 1. Install CUDA Toolkit
# Download from https://developer.nvidia.com/cuda-downloads

# 2. Verify GPU
nvidia-smi

# 3. Install Python packages
uv pip install cupy-cuda12x numba

# 4. Test GPU
python -c "import cupy as cp; print('GPU OK:', cp.cuda.is_available())"

# 5. Create branch
git checkout -b feat/gpu-acceleration
```

## Configuration

```yaml
performance:
  gpu:
    enabled: true
    device_id: 0
    fallback_to_cpu: true
  jit:
    enabled: true
    cache: true
  outofcore:
    parallel_tiles: 4
```

## Code Changes Required

### ICP Registration (fine_registration.py)

```python
# Before
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')

# After  
from ..acceleration import GPUNearestNeighbors
nbrs = GPUNearestNeighbors(n_neighbors=1, use_gpu=True)

# API stays the same!
nbrs.fit(target)
distances, indices = nbrs.kneighbors(source)
```

### DoD Computation (change_detection.py)

```python
# Before
dem = np.full((ny, nx), np.nan)
# ... manual binning loop ...

# After
from ..acceleration.jit_kernels import accumulate_grid_mean_jit
sum_grid = np.zeros((ny, nx))
cnt_grid = np.zeros((ny, nx))
accumulate_grid_mean_jit(points, min_x, min_y, cell_size, sum_grid, cnt_grid)
dem = sum_grid / cnt_grid
```

## Hardware Requirements

### Minimum Dev Setup
- GPU: RTX 3060 (12 GB)
- CPU: 8-core
- RAM: 32 GB
- Storage: 1 TB SSD

### Recommended Dev Setup
- GPU: RTX 4080/4090 (16-24 GB)
- CPU: 16-core  
- RAM: 64 GB
- Storage: 2 TB SSD + 10 TB HDD

### Production (National Scale)
- 8-16 nodes × 4 GPUs (A100 80GB)
- Parallel filesystem
- InfiniBand interconnect

## Performance Estimates

| Dataset | Size | Current | With GPU | Speedup |
|---------|------|---------|----------|---------|
| Small area | 2M pts | 5 min | 20 sec | 15x |
| Municipality | 200M pts | 8 hrs | 25 min | 20x |
| County | 20B pts | N/A | 40 hrs | New |
| Country | 770B pts | N/A | 5 days* | New |

*With 32 GPUs

## Implementation Checklist

### Week 1-2: Foundation
- [ ] Install CUDA & CuPy
- [ ] Create `acceleration/hardware.py`
- [ ] Create `acceleration/array_ops.py`
- [ ] Test GPU detection

### Week 3-4: GPU NN
- [ ] Create `acceleration/nearest_neighbors.py`
- [ ] Integrate into ICP
- [ ] Benchmark & validate
- [ ] Integrate into C2C

### Week 5: JIT
- [ ] Create `acceleration/jit_kernels.py`
- [ ] JIT grid accumulation
- [ ] JIT transforms
- [ ] Benchmark

### Week 6-7: Parallel
- [ ] Parallel tile processing
- [ ] GPU affinity
- [ ] Checkpoint/resume
- [ ] I/O optimization

### Week 8: GPU Arrays
- [ ] CuPy DoD gridding
- [ ] Distance computations
- [ ] Integration tests

### Week 9-10: M3C2
- [ ] GPU normal estimation
- [ ] Cylindrical search
- [ ] Full pipeline

### Week 11: Testing
- [ ] Unit tests
- [ ] Benchmark suite
- [ ] Numerical validation
- [ ] Real-world testing

### Week 12: Docs
- [ ] User guide
- [ ] Performance tuning
- [ ] Deployment
- [ ] Release

## Common Issues

### CUDA OOM
```python
# Reduce chunk size
chunk_size = 100_000  # down from 1M

# Or clear memory
from terrain_change_detection.acceleration import clear_gpu_memory
clear_gpu_memory()
```

### Numerical Differences
```python
# Use tolerances in tests
np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-5)
```

### Slow First Run
```python
# Warmup GPU
import cupy as cp
_ = cp.sum(cp.array([1, 2, 3]))  # Trigger init
```

## Key Design Principles

1. **Graceful Degradation**: Always fallback to CPU
2. **Transparent API**: No user code changes required  
3. **Memory Aware**: Automatic chunking
4. **Configurable**: Runtime backend selection

## Resources

- [PERFORMANCE_OPTIMIZATION_PLAN.md](PERFORMANCE_OPTIMIZATION_PLAN.md) - Full strategy
- [GPU_IMPLEMENTATION_GUIDE.md](GPU_IMPLEMENTATION_GUIDE.md) - Code examples
- [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md) - Executive summary

## Contact Points

- RAPIDS: https://rapids.ai/
- CuPy: https://docs.cupy.dev/
- Numba: https://numba.pydata.org/

---

**Last Updated**: 2025-01-05  
**Status**: Ready to implement
