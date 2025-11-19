# GPU C2C Performance Analysis - November 15, 2025

## Test Configuration
- **Hardware**: NVIDIA GeForce RTX 3050, 8GB VRAM, CUDA 13.0
- **Dataset**: Norwegian terrain (eksport_1225654_20250602)
  - Reference (2015): 5,937,325 ground points
  - Comparison (2020): 9,061,457 ground points
- **Test Sizes**: 1K → 9M points (10 different sizes)

## Performance Results

| Points      | GPU Time | CPU Time | Speedup |
|-------------|----------|----------|---------|
| 1,000       | 0.003s   | 0.001s   | 0.54x   |
| 5,000       | 0.005s   | 0.005s   | 0.88x   |
| 10,000      | 0.011s   | 0.010s   | 0.93x   |
| 20,000      | 0.021s   | 0.021s   | 0.98x   |
| 50,000      | 0.058s   | 0.059s   | 1.01x   |
| 100,000     | 0.127s   | 0.126s   | 1.00x   |
| 200,000     | 0.276s   | 0.274s   | 0.99x   |
| 500,000     | 0.805s   | 0.797s   | 0.99x   |
| 1,000,000   | 2.209s   | 2.228s   | 1.01x   |
| **9,061,457** | **11.311s** | **11.309s** | **1.00x** |

**Average Speedup (10K+ points)**: 0.99x (essentially no speedup)

## Root Cause Analysis

### Issue: Windows GPU Backend Limitation
The current implementation uses three backends:
1. **cuML** (Linux only) - True GPU acceleration (10-50x expected)
2. **sklearn-gpu** (Windows) - **sklearn CPU KDTree with CuPy array wrappers**
3. **sklearn-cpu** (Fallback) - Pure CPU implementation

On Windows, the code falls back to `sklearn-gpu` which:
- ❌ Still uses sklearn's **CPU-based KDTree** for nearest neighbor search
- ❌ Only wraps data in CuPy arrays (no compute benefit)
- ❌ Cannot pass CuPy arrays to sklearn (it doesn't support GPU)
- ✓ Metadata correctly reports `gpu_used=True` (misleading)

### Evidence from Code
```python
# From gpu_neighbors.py line 122-130
elif self.backend_ == 'sklearn-gpu':
    # sklearn + CuPy path: fit on CPU (sklearn doesn't accept CuPy directly)
    # The benefit comes from keeping data on GPU for queries
    self._model.fit(X)  # <-- Still fitting on CPU!
```

The comment reveals the issue: **"sklearn doesn't accept CuPy directly"**

## Implications

### Current State
- ✅ GPU integration infrastructure complete and working
- ✅ Configuration propagation correct
- ✅ Graceful fallback functioning
- ❌ **No actual GPU compute happening on Windows**
- ❌ Performance results show this clearly (1.00x "speedup")

### Why No Speedup
1. Nearest neighbor search still runs on CPU via sklearn
2. CuPy array wrapping adds slight overhead
3. Data transfer CPU↔GPU adds latency
4. Result: **GPU ≈ CPU** (as observed in tests)

## Solutions

### Option 1: Accept Current State (Recommended for Now)
**Status**: Document limitation, focus on Linux deployment

**Pros**:
- Already have working CPU implementation
- Linux servers get true GPU speedup with cuML
- No additional development needed

**Cons**:
- No Windows GPU acceleration
- Documented 20-60x speedup not achievable on Windows

### Option 2: Implement True GPU k-NN on Windows
**Effort**: Medium-High (1-2 weeks)

**Approach**: Use RAPIDS/CuPy to implement GPU k-NN:
```python
# Pseudo-code for GPU k-NN
import cupy as cp

def gpu_kneighbors(query_points_gpu, reference_points_gpu, k):
    # Compute pairwise distances on GPU
    distances = cp.sqrt(cp.sum((query_points_gpu[:, None, :] - 
                                 reference_points_gpu[None, :, :]) ** 2, axis=2))
    # Find k smallest distances per query
    indices = cp.argpartition(distances, k, axis=1)[:, :k]
    nearest_dists = cp.take_along_axis(distances, indices, axis=1)
    return nearest_dists, indices
```

**Pros**:
- True GPU acceleration on Windows
- 10-30x speedup potential for large datasets
- Self-contained solution

**Cons**:
- Custom implementation complexity
- Memory management challenges
- Batch processing required for very large datasets
- May not match cuML performance

### Option 3: Use FAISS Library
**Effort**: Low-Medium (2-3 days)

**Approach**: Replace sklearn with Facebook's FAISS:
```python
import faiss

# GPU index
gpu_index = faiss.index_cpu_to_gpu(
    faiss.StandardGpuResources(), 0,
    faiss.IndexFlatL2(dim)
)
gpu_index.add(reference_points)
distances, indices = gpu_index.search(query_points, k)
```

**Pros**:
- Proven GPU-accelerated nearest neighbors
- Works on Windows and Linux
- Excellent performance (competitive with cuML)
- Well-maintained library

**Cons**:
- Additional dependency
- Different API than sklearn
- May require code refactoring

## Recommendations

### Short Term (This Week)
1. ✅ **Document Windows GPU limitation** in CHANGELOG and README
2. ✅ **Update performance expectations** to reflect Linux/Windows differences
3. ✅ **Keep current implementation** (infrastructure is solid)
4. ⚠️ **Change `gpu_used` reporting** to distinguish between GPU backend vs GPU compute

### Medium Term (Next Sprint)
1. Evaluate FAISS integration for cross-platform GPU acceleration
2. Benchmark FAISS vs cuML on Linux
3. Consider FAISS as unified GPU backend if performance comparable

### Long Term (Production)
1. Deploy on Linux servers with cuML for true GPU speedup
2. Keep Windows as development environment (CPU acceptable)
3. Document platform-specific performance characteristics

## Updated Documentation Needed

### 1. CHANGELOG.md
Add Windows GPU limitation section:
- Clarify Linux (cuML) vs Windows (sklearn) backends
- Update expected speedup based on platform
- Explain why Windows shows no speedup

### 2. README.md
Platform-specific performance table:
```markdown
| Platform | Backend | Expected Speedup |
|----------|---------|------------------|
| Linux    | cuML    | 10-50x           |
| Windows  | sklearn | ~1x (CPU)        |
```

### 3. GPU_SETUP_GUIDE.md
Add troubleshooting section:
- Windows users: No GPU speedup expected (infrastructure only)
- Linux users: Install cuML for true acceleration
- Check `backend_` attribute to verify GPU compute

## Conclusion

The **comprehensive performance test successfully revealed** that:
1. ✅ GPU integration infrastructure works correctly
2. ✅ Configuration and fallback mechanisms function properly
3. ❌ **Windows does not achieve GPU compute acceleration** (as designed)
4. ⚠️ Current implementation is **Linux-only for GPU performance**
5. ✅ C2C results now expose `metadata["gpu_backend"]` alongside `metadata["gpu_used"]` so you can see whether cuML (`"cuml"`) or a CPU fallback backend (`"sklearn-gpu"`/`"none"`) was actually used.

The test was valuable - it confirmed our infrastructure is solid but identified the platform limitation that needs documentation.

**Next action**: Update documentation to reflect Windows/Linux performance differences, then evaluate FAISS for cross-platform GPU acceleration in a future sprint.
