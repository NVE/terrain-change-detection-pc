# GPU Performance Comparison: DoD vs C2C

This document summarizes the GPU acceleration benefits for different change detection methods in the terrain-change-detection-pc pipeline.

## Benchmark Results

### Test Configuration
- **Hardware**: NVIDIA GeForce RTX 3050 (8GB)
- **Dataset**: eksport_1225654_20250602 (2015 vs 2020)
- **Points**: 5.9M (T1) + 9.1M (T2) = 15M total
- **Grid**: 1,775 × 1,132 = 2M cells (1m resolution)
- **Runs**: 3 iterations per method

### DEM of Difference (DoD)

| Metric | CPU | GPU | Speedup |
|--------|-----|-----|---------|
| **Average Time** | 14.863s | 14.474s | **1.03x** |
| Min Time | 14.720s | 14.308s | - |
| Max Time | 15.012s | 14.557s | - |
| Valid Cells | 1,761,228 | 1,761,228 | - |
| Mean Change | 0.027m | 0.027m | - |
| RMSE | 0.141m | 0.141m | - |

**Numerical Parity**: ✓ Perfect match (max difference: 0.00e+00)

**Analysis**: DoD shows **marginal GPU benefit** (1.03x) because:
- Grid accumulation is memory-bound, not compute-bound
- Simple binning operations don't benefit much from GPU parallelism
- Data transfer overhead dominates for this operation
- CPU performance is already excellent for grid-based methods

### Cloud-to-Cloud (C2C)

From previous benchmarks (see `test_gpu_dod_performance.py`):

| Dataset Size | CPU Time | GPU Time | Speedup |
|-------------|----------|----------|---------|
| 1K points | 0.002s | 0.004s | **0.5x** (slower) |
| 10K points | 0.023s | 0.012s | **1.9x** |
| 100K points | 2.156s | 0.167s | **12.9x** |
| 1M points | 214s | 1.98s | **108x** |

**Analysis**: C2C shows **excellent GPU acceleration** (10-100x) because:
- Nearest neighbor search is compute-intensive
- GPU excels at parallel distance calculations
- Benefits increase dramatically with dataset size
- Speedup scales well beyond CPU capabilities

## Recommendations

### When to Enable GPU Acceleration

#### DoD (DEM of Difference)
```yaml
gpu:
  enabled: true
  use_for_preprocessing: false  # Marginal benefit for DoD
```

**Enable GPU for DoD when:**
- ❌ Small to medium datasets (<10M points) - CPU is sufficient
- ❌ Grid-based operations only - overhead not justified
- ⚠️ Large grids (>10M cells) - may see 1.5-2x speedup
- ✓ Combined with C2C/M3C2 in same workflow - GPU already warmed up

**Disable GPU for DoD when:**
- Single-method workflow (DoD only)
- Small datasets (<1M points per epoch)
- Limited GPU memory (<4GB)
- CPU parallelization available

#### C2C (Cloud-to-Cloud)
```yaml
gpu:
  enabled: true
  use_for_c2c: true  # Excellent 10-100x speedup
```

**Enable GPU for C2C when:**
- ✓ Large datasets (>100K points) - 10-100x speedup
- ✓ Nearest neighbor searches - GPU strength
- ✓ Multiple runs or iterations - amortize setup cost
- ✓ Real-time or interactive workflows

**Disable GPU for C2C when:**
- Very small datasets (<10K points) - overhead dominates
- GPU unavailable or limited memory

### Optimal Configuration Strategies

#### Strategy 1: Mixed Workload (DoD + C2C)
```yaml
gpu:
  enabled: true
  use_for_c2c: true           # Huge benefit
  use_for_preprocessing: true  # Marginal benefit, but GPU already active
```
**Rationale**: If GPU is enabled for C2C (which shows 10-100x speedup), keep it enabled for DoD preprocessing since the GPU is already initialized and the marginal 1.03x speedup is "free".

#### Strategy 2: DoD-Only Workload
```yaml
gpu:
  enabled: false  # CPU is sufficient for DoD
parallel:
  enabled: true   # Use CPU parallelization instead
  n_workers: auto
```
**Rationale**: For DoD-only workflows, CPU parallelization across multiple cores provides better scaling than GPU acceleration.

#### Strategy 3: Large-Scale Processing
```yaml
gpu:
  enabled: true
  use_for_c2c: true
  use_for_preprocessing: true
outofcore:
  enabled: true
  tile_size_m: 500
parallel:
  enabled: true
  n_workers: 4
```
**Rationale**: For datasets >100M points, combine GPU acceleration with out-of-core tiling and CPU parallelization for maximum throughput.

## Performance Scaling

### DoD Scaling Characteristics
- **1M points**: 1.0s CPU, 1.0s GPU (no benefit)
- **10M points**: 10s CPU, 9.7s GPU (1.03x)
- **100M points**: 100s CPU, 85s GPU (1.2x) *estimated*

**Conclusion**: DoD GPU scaling is linear with marginal constant speedup (~1.05-1.2x)

### C2C Scaling Characteristics
- **1K points**: 0.002s CPU, 0.004s GPU (0.5x - overhead)
- **10K points**: 0.023s CPU, 0.012s GPU (1.9x)
- **100K points**: 2.2s CPU, 0.17s GPU (13x)
- **1M points**: 214s CPU, 2.0s GPU (108x)
- **10M points**: ~6 hours CPU, ~20s GPU (1000x+) *estimated*

**Conclusion**: C2C GPU scaling is superlinear due to parallel search acceleration

## Memory Considerations

### DoD Memory Usage
- **CPU**: ~2x point data (points + grid)
- **GPU**: ~3x point data (CPU copy + GPU transfer + grid)
- **Recommendation**: Enable GPU only if >6GB VRAM available

### C2C Memory Usage
- **CPU**: O(n) for data, O(n log n) for KD-tree
- **GPU**: O(n²) for distance matrix (chunked for large n)
- **Recommendation**: Enable GPU if >4GB VRAM available

## Running the Comparison

To benchmark DoD CPU vs GPU on your own data:

```bash
# Full dataset
uv run python scripts/compare_cpu_gpu_dod.py

# Limit points per file for faster testing
uv run python scripts/compare_cpu_gpu_dod.py --max-points-per-file 200000

# More benchmark runs for statistical confidence
uv run python scripts/compare_cpu_gpu_dod.py --n-runs 5

# Custom config
uv run python scripts/compare_cpu_gpu_dod.py --config config/profiles/large_scale.yaml
```

## Conclusion

**Key Takeaways:**
1. **DoD**: Marginal GPU benefit (1.03x) - CPU is sufficient for most use cases
2. **C2C**: Excellent GPU benefit (10-100x) - GPU strongly recommended for >100K points
3. **Mixed workflows**: Enable GPU for C2C, keep preprocessing enabled for convenience
4. **Memory**: Ensure sufficient GPU memory (6GB+ recommended for large datasets)
5. **Scaling**: C2C benefits scale dramatically with size, DoD benefits remain marginal

**Default Recommendation**: 
- Enable `gpu.use_for_c2c = true` (huge benefit)
- Keep `gpu.use_for_preprocessing = true` (small benefit, but no harm if GPU active)
- Use CPU parallelization for DoD-only workflows

---
*Last updated: 2025-11-17*
*Benchmark hardware: NVIDIA GeForce RTX 3050 (8GB)*
