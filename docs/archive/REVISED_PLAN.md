# Revised GPU Acceleration Plan
## Adjusting for Existing Tiling Implementation

**Date**: November 5, 2025  
**Status**: Plan Updated - Existing tiling is solid foundation

## Key Realization

The `feat/outofcore-tiling` branch contains a **well-designed tiling system** that's perfect for GPU acceleration. Rather than starting over, we'll enhance it.

## What We Have (Assessment)

### Existing Tiling Code ✅ GOOD
Located in `src/terrain_change_detection/pipeline/tiling.py`:

**Strengths:**
- `GridAccumulator`: Efficient streaming mean aggregation
- `LaspyStreamReader`: Chunked LAZ reading with filtering
- `Tiler`: Grid-aligned tile generation with halo
- `MosaicAccumulator`: Tile assembly with overlap handling
- Already integrated into `ChangeDetector.compute_dod_streaming_files_tiled()`

**What's Missing (Our Opportunity):**
- Sequential processing (line 424 in change_detection.py: `for tile in tiler.tiles()`)
- No GPU acceleration in `GridAccumulator.accumulate()`
- No JIT compilation
- No parallel I/O

## Revised Implementation Strategy

### Phase 1: Add GPU/JIT to Existing Tiling ⚡

**Goal**: Accelerate existing tiling without major refactoring

#### 1.1 JIT-Compile GridAccumulator (Week 1)
**Target**: `GridAccumulator.accumulate()` method

Current code (lines 54-77 in tiling.py):
```python
def accumulate(self, points: np.ndarray) -> None:
    # ... vectorized but not JIT compiled
```

**Enhancement**:
```python
# Add JIT-compiled version
from ..acceleration.jit_kernels import accumulate_grid_jit

def accumulate(self, points: np.ndarray, use_jit: bool = True) -> None:
    if use_jit:
        accumulate_grid_jit(points, self.bounds, self.cell, 
                          self.sum, self.cnt)
    else:
        # Original vectorized code as fallback
        ...
```

**Expected Gain**: 2-5x speedup on grid accumulation

#### 1.2 GPU-Accelerate GridAccumulator (Week 2)
**Option A**: CuPy arrays for grid operations
```python
import acceleration.array_ops as xp

class GridAccumulator:
    def __init__(self, bounds, cell_size, use_gpu=False):
        # Use xp.zeros instead of np.zeros
        # Automatically uses GPU if available
        self.sum = xp.zeros((self.ny, self.nx), dtype=xp.float64)
        self.cnt = xp.zeros((self.ny, self.nx), dtype=xp.int64)
```

**Expected Gain**: 5-10x speedup when points fit in GPU memory

#### 1.3 Parallelize Tile Processing (Week 3)
**Target**: The sequential loop in `compute_dod_streaming_files_tiled()`

Current code (change_detection.py, around line 424):
```python
for tile in tiler.tiles():
    acc1 = GridAccumulator(tile.inner, cell_size)
    acc2 = GridAccumulator(tile.inner, cell_size)
    # ... process tile
```

**Enhancement**:
```python
from multiprocessing import Pool
from functools import partial

def _process_tile(tile, files_t1, files_t2, config):
    # Each worker processes one tile independently
    acc1 = GridAccumulator(tile.inner, cell_size)
    acc2 = GridAccumulator(tile.inner, cell_size)
    # ... process and return results
    return (tile, dem1, dem2)

# Parallel processing
with Pool(processes=n_workers) as pool:
    results = pool.map(worker_fn, list(tiler.tiles()))
```

**Expected Gain**: Near-linear scaling with CPU cores

### Phase 2: GPU for Non-Tiled Operations (Weeks 4-5)

Focus on operations that don't use tiling yet:

#### 2.1 ICP Registration (High Priority)
- GPU nearest neighbor search
- Most compute-intensive single operation
- Expected: 15-30x speedup

#### 2.2 C2C Distances  
- GPU KD-tree for all-pairs NN
- Expected: 20-40x speedup

#### 2.3 M3C2
- GPU cylindrical search
- Expected: 10-20x speedup

### Phase 3: Advanced Optimizations (Weeks 6-8)

- Multi-GPU tile distribution
- Asynchronous I/O prefetching
- Custom CUDA kernels for M3C2
- Adaptive tile sizing based on density

## Revised TODO Priorities

### Immediate (Week 1) 🔥
1. ✅ Create branch (done)
2. ⬜ **Add Numba JIT to GridAccumulator** ← START HERE
   - Minimal risk, good ROI
   - Tests existing tiling
3. ⬜ **Benchmark current tiling**
   - Establish baseline
   - Measure JIT improvement

### Week 2
4. ⬜ Implement hardware detection
5. ⬜ Implement array backend (xp module)
6. ⬜ GPU-accelerate GridAccumulator

### Week 3
7. ⬜ Implement parallel tile processing
8. ⬜ Test on Akershus dataset
9. ⬜ Measure speedup vs sequential

### Weeks 4-5
10. ⬜ GPU nearest neighbors
11. ⬜ GPU ICP
12. ⬜ GPU C2C

### Weeks 6-8
13. ⬜ Advanced optimizations
14. ⬜ Testing and validation
15. ⬜ Documentation

## Integration Points

### Existing Code to Modify

1. **`tiling.py`**: Add JIT/GPU support
   - `GridAccumulator.accumulate()` - JIT compile
   - `GridAccumulator.__init__()` - Support GPU arrays
   
2. **`change_detection.py`**: Add parallel processing
   - `compute_dod_streaming_files_tiled()` - Parallelize tile loop
   - Add `use_jit` and `use_gpu` parameters
   
3. **`fine_registration.py`**: GPU acceleration
   - `ICPRegistration.align_point_clouds()` - GPU NN search
   
4. **`config.py`**: Extend configuration
   - Add GPU/JIT settings
   - Parallel worker count

### New Modules to Create

```
src/terrain_change_detection/acceleration/
├── __init__.py
├── hardware.py        # GPU detection
├── array_ops.py       # NumPy/CuPy abstraction
├── jit_kernels.py     # Numba kernels for GridAccumulator
└── nearest_neighbors.py  # GPU NN search
```

## Success Metrics (Revised)

### DoD with Tiling (Current Focus)
- **Baseline**: Current sequential tiling
- **Target 1**: 2-5x with JIT (Week 1)
- **Target 2**: 5-10x with GPU grids (Week 2)
- **Target 3**: 10-20x with parallel + GPU (Week 3)

### ICP Registration
- **Baseline**: Current CPU sklearn
- **Target**: 15-30x with GPU NN search

### Overall Pipeline
- **Target**: 10-25x end-to-end speedup

## Key Insight

**The existing tiling code is well-designed and ready for GPU/parallel enhancement!** 

We don't need to rebuild it - just add:
1. JIT compilation to hot spots
2. GPU array backend
3. Parallel tile processing
4. GPU nearest neighbors for ICP/C2C

This is actually **better** than starting from scratch because:
- The architecture is proven
- It's already integrated
- We can make incremental improvements
- Each enhancement is independently testable

## Next Steps

### Recommended Starting Point

**Start with JIT compilation of GridAccumulator** because:
- Lowest risk (pure addition, no refactoring)
- Fast to implement (1-2 hours)
- Immediate measurable benefit (2-5x)
- Validates our approach
- Requires no GPU hardware

Code example:
```python
# acceleration/jit_kernels.py
import numba
import numpy as np

@numba.jit(nopython=True, parallel=True)
def accumulate_grid_jit(points, bounds, cell_size, sum_grid, cnt_grid):
    """JIT-compiled grid accumulation."""
    min_x, min_y = bounds.min_x, bounds.min_y
    nx, ny = sum_grid.shape[1], sum_grid.shape[0]
    
    for i in numba.prange(len(points)):
        x, y, z = points[i, 0], points[i, 1], points[i, 2]
        xi = int((x - min_x) / cell_size)
        yi = int((y - min_y) / cell_size)
        
        if 0 <= xi < nx and 0 <= yi < ny:
            sum_grid[yi, xi] += z
            cnt_grid[yi, xi] += 1
```

**Want me to implement this first JIT optimization?** It's a great way to:
1. Test the workflow
2. Get immediate results
3. Build confidence in the approach
4. Establish benchmarking methodology

---

**Status**: Ready to proceed with incremental enhancements to existing tiling  
**Risk Level**: Low (additive changes, not refactoring)  
**Expected Timeline**: Can start seeing results this week!
