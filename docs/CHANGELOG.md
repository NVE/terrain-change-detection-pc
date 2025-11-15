# Changelog and Implementation Notes

## 2025-11-15 - Phase 2.1: C2C GPU Acceleration - Workflow Integration & Verification

### Summary
Integrated GPU C2C acceleration into the main workflow script (`run_workflow.py`) and verified end-to-end functionality on real production data. GPU acceleration is now fully operational in the complete terrain change detection pipeline with proper configuration propagation, status logging, and usage reporting.

### What Changed

- **Workflow Integration (`run_workflow.py`)**
  - Updated all C2C method calls to pass `config` parameter:
    - `compute_c2c()` - basic Euclidean C2C distances
    - `compute_c2c_vertical_plane()` - plane-based C2C with local fitting
    - `compute_c2c_streaming_files_tiled()` - streaming tiled C2C
    - `compute_c2c_streaming_files_tiled_parallel()` - parallel streaming C2C
  - GPU configuration now properly propagated through entire workflow
  - Backward compatible (config parameter is optional)

- **GPU Status Logging**
  - Added GPU hardware detection at workflow startup:
    - Displays GPU device name, memory, and CUDA version
    - Shows which GPU features are enabled (C2C, preprocessing)
    - Warns if GPU enabled in config but hardware unavailable
  - Example output:
    ```
    GPU Acceleration: ENABLED
      Device: NVIDIA GeForce RTX 3050
      Memory: 8.00 GB
      C2C: ENABLED
      Preprocessing: ENABLED
    ```

- **GPU Usage Reporting**
  - Added post-C2C computation reporting:
    - Logs whether GPU or CPU was used for computation
    - Helps verify GPU acceleration is working as expected
  - Example: `C2C computation used: GPU`

- **Performance Benchmark Script**
  - Added `scripts/test_gpu_c2c_performance.py`:
    - Standalone GPU vs CPU benchmark on real data
    - Tests multiple point cloud sizes (1K, 5K, 10K points)
    - Calculates speedups and provides performance insights
    - Uses data from `config/default.yaml` for easy testing

### Verification Results

**Tested on Real Data:**
- Dataset: `eksport_1225654_20250602` (Norwegian terrain)
- Time periods: 2015 (5.9M ground points) and 2020 (9.1M ground points)
- Hardware: NVIDIA GeForce RTX 3050, 8GB VRAM, CUDA 13.0
- **✅ GPU Successfully Used**: Confirmed via workflow logs
- Workflow completed successfully with GPU acceleration

**Performance Observations:**
- Small datasets (<1K points): CPU faster due to GPU transfer overhead
- Medium datasets (1K-10K): GPU benefits begin to show
- Large datasets (10K+ points): Significant GPU speedup expected
- Configuration-driven: Easy to enable/disable for testing

### Usage Examples

**Run complete workflow with GPU:**
```bash
uv run python scripts/run_workflow.py --config config/default.yaml
```

**Benchmark GPU vs CPU performance:**
```bash
uv run python scripts/test_gpu_c2c_performance.py
```

**Enable/Disable GPU in config:**
```yaml
gpu:
  enabled: true              # Master GPU switch
  use_for_c2c: true         # Use GPU for C2C distances
  fallback_to_cpu: true     # Auto-fallback if GPU fails
```

### Technical Notes

- GPU detection uses `detect_gpu()` from `hardware_detection.py`
- Returns `GPUInfo` dataclass with availability, device info, and error messages
- Graceful degradation: If GPU unavailable, falls back to CPU automatically
- All GPU usage properly logged for monitoring and debugging

### Files Changed
- `scripts/run_workflow.py`: GPU integration, status logging, usage reporting
- `scripts/test_gpu_c2c_performance.py`: New benchmark script
- `docs/CHANGELOG.md`: This entry

### Migration Notes
- No breaking changes - config parameter is optional
- Existing workflows continue to work without modification
- To enable GPU: Set `gpu.enabled=true` and `gpu.use_for_c2c=true` in config
- GPU status logged at workflow startup for visibility

---

## 2025-11-15 - Phase 2.1: C2C GPU Acceleration Integration (Complete)

### Summary
Successfully integrated GPU acceleration into all C2C (Cloud-to-Cloud) change detection variants. GPU-accelerated nearest neighbor searches now available for basic C2C, vertical plane C2C, and streaming/tiled C2C (both sequential and parallel). All implementations maintain GPU/CPU numerical parity with graceful CPU fallback. Configuration-driven GPU enable/disable allows flexible deployment across different hardware environments.

### What Changed

- **GPU-Accelerated C2C Variants**
  - Updated `compute_c2c()`: GPU nearest neighbors for basic C2C
    - Automatic GPU detection and fallback to CPU
    - Config-driven GPU enable (`config.gpu.enabled` and `config.gpu.use_for_c2c`)
    - Metadata includes `gpu_used` flag for observability
  - Updated `compute_c2c_vertical_plane()`: GPU for local plane fitting
    - Supports both k-NN and radius-based neighborhood searches on GPU
    - GPU-accelerated neighbor queries with CPU plane fitting
    - Graceful fallback maintains numerical consistency
  - Updated `compute_c2c_streaming_files_tiled()`: GPU for streaming C2C
    - Per-tile GPU nearest neighbor searches
    - Automatic memory management for large datasets
    - Log messages indicate GPU usage per tile
  - Updated `compute_c2c_streaming_files_tiled_parallel()`: GPU + multicore
    - Each worker process can use GPU independently
    - Combined CPU parallelization (2-3x) + GPU acceleration (10-20x)
    - Expected total speedup: **20-60x** on large datasets

- **Tile Worker GPU Support**
  - Updated `process_c2c_tile()` in `tile_workers.py`:
    - Added `use_gpu` parameter for worker functions
    - GPU-accelerated nearest neighbors in tile processing
    - Per-tile GPU/CPU fallback with debug logging
  - Workers automatically use GPU when available and configured

- **GPU Neighbors Enhancement**
  - Updated `create_gpu_neighbors()`: Added `radius` parameter
    - Now supports both k-NN and radius-based queries
    - Unified factory function for all neighbor search types

- **Comprehensive Testing**
  - Added `tests/test_gpu_c2c_integration.py`: 16 GPU C2C tests
    - GPU/CPU numerical parity validation
    - Configuration propagation tests
    - Edge cases (empty clouds, identical clouds, single points)
    - Performance benchmarks (10K+ points)
    - max_distance filtering verification
    - Vertical plane variant testing
  - **All 16 C2C integration tests passing**
  - **Total GPU tests: 50 passing** (34 infrastructure + 16 C2C)

- **Configuration Integration**
  - C2C methods now accept optional `config` parameter
  - GPU usage respects configuration hierarchy:
    1. `config.gpu.enabled` must be `True`
    2. `config.gpu.use_for_c2c` must be `True`
    3. GPU hardware must be available
  - If any condition fails, automatic CPU fallback
  - No config provided → defaults to CPU (safe default)

### GPU/CPU Numerical Parity

All C2C variants produce numerically identical results between GPU and CPU:
- Basic C2C: distances match within `rtol=1e-5`
- Vertical plane C2C: distances match within `rtol=1e-4` (plane fitting tolerance)
- Statistics (RMSE, mean, median) match within floating point precision
- Verified on point clouds from 100 to 100,000 points

### Performance Characteristics

**Expected Speedups** (based on GPU architecture analysis):
- **Small clouds (<1K points)**: 1-2x (GPU overhead dominant)
- **Medium clouds (1K-10K)**: 5-10x (GPU acceleration starts)
- **Large clouds (10K-100K)**: 10-20x (full GPU utilization)
- **Very large (100K+)**: 15-30x (GPU memory bandwidth saturated)
- **Parallel + GPU**: 20-60x total (2-3x CPU parallel × 10-20x GPU)

**Memory Efficiency**:
- GPU operations use automatic batching
- Graceful handling of out-of-memory conditions
- Memory limit configurable via `config.gpu.gpu_memory_limit_gb`

### Integration Examples

```python
from terrain_change_detection.detection import ChangeDetector
from terrain_change_detection.utils.config import AppConfig

# Load configuration with GPU enabled
config = AppConfig.from_yaml("config/default.yaml")

# Basic C2C with GPU
result = ChangeDetector.compute_c2c(
    source_points, 
    target_points, 
    config=config  # GPU used if available
)

# Check if GPU was used
print(f"GPU used: {result.metadata.get('gpu_used', False)}")

# Vertical plane C2C with GPU
result = ChangeDetector.compute_c2c_vertical_plane(
    source_points,
    target_points,
    k_neighbors=20,
    config=config
)

# Streaming tiled C2C with GPU + parallel
result = ChangeDetector.compute_c2c_streaming_files_tiled_parallel(
    files_src=source_files,
    files_tgt=target_files,
    tile_size=500.0,
    max_distance=2.0,
    n_workers=None,  # Auto-detect CPU cores
    config=config,   # GPU per worker if available
)
```

### Observability and Debugging

All C2C methods now provide GPU usage metadata:
- `result.metadata["gpu_used"]`: Boolean indicating GPU usage
- Log messages indicate GPU acceleration status
- Per-tile logging in streaming variants: `"Tile (2,3): src=5000, tgt=5200 - building NN (GPU)"`
- Automatic warning logs on GPU fallback

### Migration Notes

**No Breaking Changes**: 
- All existing C2C calls remain compatible (config parameter optional)
- Without config, behavior defaults to CPU (backward compatible)
- GPU acceleration is opt-in via configuration

**To Enable GPU**:
1. Install GPU dependencies: `uv sync --extra gpu`
2. Ensure CUDA Toolkit installed (12.x or 13.x)
3. Set `config.gpu.enabled: true` and `config.gpu.use_for_c2c: true` in YAML
4. Pass config to C2C methods: `compute_c2c(..., config=config)`

**Verification**:
- Check result metadata: `result.metadata.get("gpu_used")`
- Run tests: `pytest tests/test_gpu_c2c_integration.py -v`

### Next Steps

**Phase 2.2** (Optional - GPU Preprocessing):
- GPU-accelerated point cloud transformations
- GPU filtering and downsampling
- 5-10% preprocessing speedup expected

**Phase 3** (Deferred - Custom GPU M3C2):
- Only if C2C GPU shows >20x production speedup
- Only if M3C2 becomes bottleneck
- Requires 2-3 weeks full reimplementation

### Technical Debt

- Windows cuML support limited (Linux-only native GPU ML)
- cuML radius_neighbors returns different format than sklearn (requires CPU conversion)
- GPU memory management heuristics need production tuning

---

## 2025-11-15 - Phase 2: GPU Acceleration Infrastructure (Week 1 Foundation)

### Summary
Implemented comprehensive GPU acceleration infrastructure including hardware detection, array operations abstraction, GPU-accelerated nearest neighbors, and configuration system. This foundational work enables GPU acceleration for change detection algorithms, with C2C identified as the primary target (20-60x total speedup potential). M3C2 integration analysis revealed py4dgeo's C++ implementation cannot be directly GPU-accelerated, recommending focus on C2C where we have direct control.

### What Changed

- **GPU Hardware Detection** (`src/terrain_change_detection/acceleration/hardware_detection.py`)
  - Added `detect_gpu()`: Auto-detects NVIDIA GPU with CUDA support
  - Added `get_gpu_info()`: Cached GPU information (device name, memory, compute capability)
  - Added `check_gpu_memory()`: Memory availability validation
  - Added `get_optimal_batch_size()`: Automatic batch size calculation
  - Graceful CPU fallback when GPU unavailable
  - All 7 detection tests passing

- **GPU Array Operations** (`src/terrain_change_detection/acceleration/gpu_array_ops.py`)
  - Added `ArrayBackend`: Unified NumPy/CuPy interface for transparent CPU/GPU switching
  - Added `get_array_backend()`: Global backend instance management
  - Added convenience functions: `ensure_cpu_array()`, `ensure_gpu_array()`, `is_gpu_array()`
  - Supports all standard array operations (creation, math, manipulation, logical)
  - Automatic memory management between CPU/GPU
  - All 14 array operation tests passing

- **GPU Nearest Neighbors** (`src/terrain_change_detection/acceleration/gpu_neighbors.py`)
  - Added `GPUNearestNeighbors`: sklearn/cuML wrapper with automatic backend selection
  - Strategy: cuML (Linux native GPU) > sklearn-gpu (Windows hybrid) > sklearn-cpu (fallback)
  - Added `create_gpu_neighbors()`: Factory function for easy instantiation
  - Supports k-neighbors and radius neighbors queries
  - 10-50x speedup potential on nearest neighbor searches
  - All 13 neighbor tests passing

- **GPU Configuration System**
  - Added `gpu` section to configuration model (`src/terrain_change_detection/utils/config.py`)
  - Configuration options:
    - `gpu.enabled`: Master switch (default: true, with graceful fallback)
    - `gpu.use_for_c2c`: Enable C2C GPU acceleration (default: true)
    - `gpu.use_for_preprocessing`: Enable preprocessing GPU (default: true)
    - `gpu.fallback_to_cpu`: Automatic CPU fallback (default: true)
    - `gpu.gpu_memory_limit_gb`: GPU memory management (default: auto 80%)
    - `gpu.batch_size`: Operation batch sizing (default: auto-calculate)
  - Updated `config/default.yaml` with GPU settings

- **Dependencies**
  - Added optional `[gpu]` dependency group in `pyproject.toml`:
    - `cupy-cuda13x>=13.0.0`: NumPy-compatible GPU arrays (CUDA 13.x)
    - `numba>=0.59.0`: JIT compilation with CUDA support
    - `cuml-cu12>=24.0.0`: GPU ML library (Linux-only)
  - Installation: `uv sync --extra gpu`

- **Documentation**
  - Added `docs/GPU_SETUP_GUIDE.md`: Comprehensive GPU setup instructions
    - Prerequisites (hardware, CUDA Toolkit)
    - Installation steps (Windows/Linux)
    - Troubleshooting (NVRTC DLL errors, memory issues)
    - Configuration guide
  - Added `docs/GPU_INTEGRATION_STRATEGY.md`: Integration architecture document
    - Analysis of C2C vs M3C2 GPU integration feasibility
    - Technical explanation of py4dgeo C++ limitations
    - Performance expectations and benchmarks
    - Phase 2.1 roadmap (C2C integration)
  - Added `docs/GPU_CUDA_TOOLKIT_REQUIRED.md`: CUDA Toolkit installation guide
  - Updated `docs/ROADMAP.md`: Phase 1 marked complete, Phase 2 initiated

- **Testing**
  - Added `tests/test_gpu_detection.py`: 7 hardware detection tests
  - Added `tests/test_gpu_array_ops.py`: 14 array operations tests
  - Added `tests/test_gpu_neighbors.py`: 13 nearest neighbors tests
  - **Total: 34 GPU tests passing**
  - Validated GPU/CPU numerical parity
  - Verified graceful CPU fallback

- **Module Exports**
  - Updated `src/terrain_change_detection/acceleration/__init__.py`:
    - Exported GPU detection functions
    - Exported array backend classes
    - Exported GPU neighbors wrapper

### GPU Integration Analysis

**C2C (Cloud-to-Cloud)** - ✅ Direct GPU Integration Recommended
- Current: Uses `sklearn.neighbors.NearestNeighbors` (CPU-only)
- Strategy: Drop-in replacement with `GPUNearestNeighbors`
- Expected: 10-20x GPU speedup + 2-3x CPU parallel = **20-60x total**
- Effort: Low (2-3 hours)
- Status: Ready for Phase 2.1 integration

**M3C2** - ⚠️ Limited GPU Integration Potential
- Current: Uses `py4dgeo.m3c2.M3C2` (external C++ library)
- Limitation: KDTree built inside C++ code, no Python hooks for GPU injection
- py4dgeo already uses optimized C++ KDTree (within 2-5x of GPU performance)
- Phase 1 CPU parallelization (2-3x) remains primary M3C2 optimization
- Custom GPU M3C2: Possible but requires full reimplementation (2-3 weeks effort)
- Recommendation: **Keep py4dgeo M3C2 as-is**, focus GPU on C2C

### Performance Expectations

**Current Baseline**:
- C2C (100K pts): ~10s (sklearn KDTree CPU)
- M3C2 (10K cores, 1M pts): ~60s (py4dgeo C++)

**With GPU (Phase 2.1 - C2C only)**:
- C2C (100K pts): ~0.5-1s (**10-20x speedup**)
- C2C combined (GPU + CPU parallel): **20-60x total speedup**
- M3C2: ~60s (no change, py4dgeo C++ already optimized)

**Hardware Verified**:
- GPU: NVIDIA GeForce RTX 3050, 8.0 GB, Compute Capability 8.6
- CUDA: 13.0 (Toolkit installed and verified)
- Driver: NVIDIA 581.57

### Migration Notes

**To Enable GPU Acceleration**:
```yaml
# config/default.yaml (already updated)
gpu:
  enabled: true
  use_for_c2c: true
  fallback_to_cpu: true
```

**To Disable GPU**:
```yaml
gpu:
  enabled: false
```

**CUDA Toolkit Requirement** (Windows):
- CuPy requires CUDA Toolkit for JIT compilation (NVRTC libraries)
- GPU drivers alone insufficient
- Installation: https://developer.nvidia.com/cuda-downloads
- See `docs/GPU_SETUP_GUIDE.md` for detailed instructions

**Python Package Installation**:
```bash
# Install with GPU support
uv sync --extra gpu

# Or for development
uv sync --extra gpu --group dev
```

### Next Steps (Phase 2.1)

1. Integrate `GPUNearestNeighbors` into `ChangeDetector.compute_c2c()`
2. Update streaming/tiled C2C variants for GPU
3. Performance benchmarking (GPU vs CPU)
4. Numerical parity validation
5. Production testing on large datasets

### Technical Debt / Known Issues

- M3C2 GPU integration requires custom implementation (deferred to Phase 3)
- cuML only available on Linux (Windows uses sklearn-gpu hybrid)
- CUDA Toolkit installation required on Windows (documented)

---

## 2025-11-13 - Alignment, Tiling, and M3C2 Autotune Consistency

### Summary
This release resolves cross-mode differences between in-memory and streaming/tiled analyses by fixing a DoD mosaicking bug, aligning filtering/transform behavior across paths, and making M3C2 parameter selection consistent and reproducible. A new header-based autotune option and a fixed-parameter mode make results mode-agnostic when desired.

### What Changed

- DoD mosaicking
  - Fixed masked writes on a view that could drop values when assembling tiles.
  - File: `src/terrain_change_detection/acceleration/tiling.py` (MosaicAccumulator.add_tile)

- Parallel workers (tiling)
  - Pass `ground_only` through all worker paths (DoD, C2C, M3C2).
  - Apply epoch-2 transform per chunk where applicable to keep parity with in-memory.
  - File: `src/terrain_change_detection/acceleration/tile_workers.py`.

- M3C2 consistency and robustness
  - Added header-based autotune (`autotune_m3c2_params_from_headers`) for density derived from LAS headers and union extent (mode-agnostic).
  - Kept sample-based autotune; selection is controlled by config (`detection.m3c2.autotune.source: header | sample`).
  - Added fixed-parameter mode via YAML (`detection.m3c2.use_autotune: false` with `detection.m3c2.fixed.{radius, normal_scale, depth_factor}`).
  - Exposed CLI overrides for experiments: `--m3c2-radius`, `--m3c2-normal-scale`, `--m3c2-depth-factor`.
  - Added reproducibility and diagnostics: `--seed`, `--cores-file`, and `--debug-m3c2-compare` (correlations and sign/quantile summaries).
  - Files:
    - `src/terrain_change_detection/detection/change_detection.py` (new header-based autotune API)
    - `src/terrain_change_detection/detection/__init__.py` (exports)
    - `scripts/run_workflow.py` (config plumbing, CLI flags, diagnostics)

- Configuration
  - Config model additions: `detection.m3c2.use_autotune`, `detection.m3c2.fixed`, `detection.m3c2.autotune.source`.
  - YAML profiles updated to include the new keys and default `autotune.source: header`:
    - `config/default.yaml`
    - `config/profiles/synthetic.yaml`
    - `config/profiles/large_synthetic.yaml`
    - `config/profiles/large_scale.yaml`
  - File: `src/terrain_change_detection/utils/config.py`.

### Expected Impact

- Streaming/tiled and in-memory runs produce consistent M3C2 results when using the same parameters. Header-based autotune yields mode-agnostic parameter selection.
- DoD mosaics no longer lose contributions in overlapped areas.
- Class filtering and transform handling are consistent across execution modes.

### Migration Notes

- For reproducible production runs, set fixed M3C2 parameters in YAML.
- To keep autotune but avoid mode sensitivity, set `detection.m3c2.autotune.source: header` (profiles already default to this).
- Use `--cores-file` to compare streaming and in-memory on identical core sets during validation.

## 2025-11-11 - M3C2 Stats Robustness and Streaming Consistency

### Summary
Improved consistency between in-memory and streaming M3C2 reporting and fixed
confusing NaN statistics when a subset of core points has undefined distances.
This change does not alter the underlying M3C2 distance calculations; it makes
summary statistics robust and consistent across execution modes and ensures the
parallel streaming path respects classification filtering like other paths.

### What Changed

- NaNâ€‘robust stats and valid counts for M3C2:
  - In the workflow runner, M3C2 logs now compute mean/median/std with
    NaNâ€‘aware reducers and include a count of valid distances.
  - File: `scripts/run_workflow.py`.

- Enriched M3C2 metadata:
  - `ChangeDetector.compute_m3c2_original` now adds `n_valid`, `mean`,
    `median`, and `std` to `M3C2Result.metadata` using NaNâ€‘robust reducers.
  - File: `src/terrain_change_detection/detection/change_detection.py`.

- Streaming/parallel M3C2 respects `ground_only`:
  - Wired `ground_only` through the parallel tiled M3C2 worker path and reader,
    aligning class filtering behavior with inâ€‘memory and sequential streaming.
  - Files:
    - `src/terrain_change_detection/detection/change_detection.py`
    - `src/terrain_change_detection/acceleration/tile_workers.py`

### Expected Impact

- In-memory runs no longer report allâ€‘NaN summary stats if only a fraction of
  cores are undefined; summaries are computed over valid distances only.
- Streaming and inâ€‘memory runs present comparable summary logs
  (`n` and `valid` counts), reducing confusion across modes.
- Parallel streaming M3C2 applies the same ground/class filtering as other
  paths, improving reproducibility.

### Notes

- If many core points yield NaN distances, consider increasing effective
  neighborhoods via `detection.m3c2.autotune` (e.g., larger `min_radius`,
  `target_neighbors`, or `max_depth_factor`).
- The DoD warning about missing transformed files is unrelated to M3C2: for
  streaming M3C2, the ICP transform is applied onâ€‘theâ€‘fly when aligned files
  are not written.

## 2025-11-11 - Logging + Progress UX Overhaul

### Summary
Significant improvements to runtime logging and user feedback during long operations. Console logs are cleaner and more focused; parallel and sequential tile processing now display a Rich progress bar with elapsed time and ETA. Noisy thirdâ€‘party prints (KDTree builds) are captured and demoted to DEBUG, including Windowsâ€‘safe handling.

### What Changed

- Console/file log formats:
  - Console: simplified to `time | level | logger | message` (no process name/PID).
  - File logs (when configured): keep detailed `processName[pid]` for debugging.
  - File: `src/terrain_change_detection/utils/logging.py`.

- Rich progress bars for tile processing (DoD, C2C, M3C2):
  - Added progress bars in `TileParallelExecutor` for both sequential and parallel paths.
  - Fallback to adaptive interval INFO logs when Rich is unavailable.
  - File: `src/terrain_change_detection/acceleration/parallel_executor.py`.

- Less noisy INFO logs:
  - Demoted perâ€‘run M3C2 start/finish lines from INFO â†’ DEBUG.
  - Shortened file list logs to counts + a few basenames.
  - Replaced banner `print()` with `logger.info()` for consistent formatting.
  - Files: `src/terrain_change_detection/detection/change_detection.py`, `scripts/run_workflow.py`.

- Capture and demote thirdâ€‘party stdout/stderr:
  - New helpers to redirect/capture Python stdout/stderr and Câ€‘level fd(1/2) output.
  - Filtered to only log KDTree build messages at DEBUG.
  - Windowsâ€‘safe fd redirection to prevent `OSError: [WinError 6] Handle is invalid` in workers.
  - Files: `src/terrain_change_detection/utils/logging.py`, `src/terrain_change_detection/detection/change_detection.py`.

- Dependency updates:
  - Added `rich>=13.7.0` for progress bars.
  - File: `pyproject.toml`.

### Expected Impact

- Cleaner console logs by default; detailed process information still available in file logs.
- Better user feedback during long, parallel operations with minimal log spam.
- KDTree and similar library prints no longer clutter INFO output; visible only at DEBUG.

### Notes

- If you want to surface the KDTree messages, set `logging.level: DEBUG` in your profile.
- Progress bars are transient in the console and complemented by occasional summary INFO lines.

## 2025-11-10 - CPU Parallelization Refinements and I/O Pruning

### Summary
Focused improvements to the CPU parallel path: reduced redundant I/O per tile, prevented BLAS/NumPy thread oversubscription in workers, improved scheduling, and corrected misleading speedup logs. Pool reuse and failâ€‘fast were explored but intentionally not shipped to avoid regressions.

### What Changed

- Parallel executor (CPU):
  - Clamp effective workers to `min(n_workers, n_tiles)` to avoid overspawn.
  - Pin BLAS/NumPy perâ€‘process threads via pool initializer (default `threads_per_worker=1`).
  - Use a small `chunksize` heuristic for `imap_unordered` to lower IPC overhead on many tiles.
  - Replace misleading â€œtheoretical speedupâ€ log with throughput plus heuristic expected speedup.
  - File: `src/terrain_change_detection/acceleration/parallel_executor.py`

- Tiled I/O pruning:
  - Added `scan_las_bounds(files)` and `bounds_intersect(a, b)` utilities with a simple headerâ€‘bounds cache to avoid rescanning the same files.
  - DoD/C2C/M3C2 parallel methods now pass only perâ€‘tile intersecting files to workers, cutting repeated reads/decompression.
  - Files:
    - `src/terrain_change_detection/acceleration/tiling.py`
    - `src/terrain_change_detection/detection/change_detection.py`
    - `src/terrain_change_detection/acceleration/__init__.py`

- Configuration and workflow:
  - Added `parallel.threads_per_worker` (default 1) to config model and wired through runner and compute APIs.
  - Present in `config/default.yaml` and `config/profiles/large_scale.yaml`; optional in other profiles (defaults apply).
  - Files:
    - `src/terrain_change_detection/utils/config.py`
    - `config/default.yaml`, `config/profiles/large_scale.yaml`, `config/profiles/synthetic.yaml`
    - `scripts/run_workflow.py`

### Notes

- Pool reuse and failâ€‘fast threshold were prototyped but reverted to maintain stability across platforms and to keep behavior simple (perâ€‘call pools, aggregate error reporting).
- The perâ€‘tile file pruning provides the biggest win on multiâ€‘file datasets by avoiding O(#tilesÃ—#files) rescans.

### Expected Impact

- Better scaling on medium/large datasets due to reduced redundant I/O and no thread oversubscription.
- More accurate logs for expected speedup; easier to reason about throughput.

## 2025-11-09 - Large Synthetic Dataset for Performance Testing

### Summary
Created comprehensive large-scale synthetic dataset generation tools to properly test CPU parallelization performance at scale (50M+ points). The dataset includes realistic terrain features and controlled changes without misalignment complexity.

### What Changed

**New Dataset Generation**:
- Created `scripts/generate_large_synthetic_laz.py`:
  - Generates 51.2M total points (25.6M per epoch) across 16 tiles
  - 4Ã—4 km coverage area with 1000m Ã— 1000m tiles
  - Realistic multi-scale terrain: hills, valleys, ridges, roughness
  - 6 controlled terrain change features: mounds, pits, landslides, ridges
  - No misalignment - pure terrain changes for clean testing
  - Proper Z-range overlap between epochs

**Configuration**:
- Created `config/profiles/large_synthetic.yaml`:
  - Optimized for large dataset processing (50M+ points)
  - Out-of-core and parallel processing enabled
  - All three detection methods enabled
  - 50,000 M3C2 core points for comprehensive testing
  - Alignment disabled (no misalignment in synthetic data)

**Dataset Characteristics**:
- 16 LAZ tiles per epoch (32 files total)
- Each tile: ~1.6M points, 1000m Ã— 1000m area
- Terrain changes: 1.2-3.0m magnitude across 6 locations
- Z-ranges nearly identical between epochs (terrain changes only)
- Perfect registration - no ICP alignment needed

### Rationale

Previous benchmarks on ~9M point dataset showed parallelization was slower due to overhead exceeding benefits. This larger dataset (51M points, 16 tiles) provides:
- Better tile/worker ratio for load balancing
- More computation per tile to amortize overhead
- Realistic scale where parallel benefits should materialize
- Clean test case without alignment complications

### Usage

```bash
# Generate dataset (one-time setup)
uv run scripts/generate_large_synthetic_laz.py

# Run workflow with parallel processing
uv run scripts/run_workflow.py --config config/profiles/large_synthetic.yaml
```

Expected performance improvements:
- DoD: 2-3x speedup
- C2C: 3-5x speedup
- M3C2: 4-6x speedup

---

## 2025-11-09 - CPU Parallelization Implementation Complete (Phase 1)

### Summary
Completed CPU parallelization of all three change detection methods (DoD, C2C, M3C2). Implemented tile-level parallel processing infrastructure with comprehensive benchmarking showing the optimization provides infrastructure for larger datasets but has limited benefit at current scale (~9M points).

### What Changed

**Core Parallelization Infrastructure**:
- Created `src/terrain_change_detection/acceleration/tile_parallel.py`: 
  - `TileParallelExecutor` class for managing worker pools and tile distribution
  - Adaptive worker count estimation based on system resources
  - Progress tracking and error handling for parallel tile processing
  
- Created `src/terrain_change_detection/acceleration/tile_workers.py`:
  - Module-level picklable worker functions (`process_dod_tile`, `process_c2c_tile`, `process_m3c2_tile`)
  - Handles tile data loading, processing, and result assembly
  - Proper error handling and empty tile management

**Change Detection Methods - Parallel Variants**:
- `compute_dod_streaming_files_tiled_parallel()`: Parallel DoD with grid mosaicking
- `compute_c2c_streaming_files_tiled_parallel()`: Parallel C2C with distance concatenation
- `compute_m3c2_streaming_files_tiled_parallel()`: Parallel M3C2 with core point distribution

**Workflow Integration**:
- Updated `scripts/run_workflow.py`: Added parallel routing for all three methods
- Configuration flag `parallel.enabled` controls sequential vs parallel execution
- Automatic fallback to sequential on errors

**Bug Fixes**:
- Fixed M3C2Result dataclass field names: `uncertainties` â†’ `uncertainty` (singular)
- Added missing `core_points` field to M3C2Result construction
- Removed invalid `stats` parameter from M3C2Result (moved to metadata)
- Fixed C2CResult to use individual fields instead of stats dict
- Corrected streaming API calls with proper `bbox` parameter usage

**Configuration Files**:
- Created benchmark configs under `experiments/configs`: `bench_{dod,c2c,m3c2}_{seq,par}.yaml`
- Updated `large_scale.yaml` with parallel settings

**Benchmarking**:
- Created `experiments/scripts/run_benchmark.ps1`: PowerShell script for performance testing
- Created `experiments/benchmarks/reports/BENCHMARK_RESULTS.md`: Comprehensive performance analysis
- Tested all 6 scenarios (3 methods Ã— 2 modes)

### Performance Results

**Dataset**: eksport_1225654 (~9M points, 12 tiles, 11 workers)

| Method | Sequential | Parallel | Speedup | Status |
|--------|-----------|----------|---------|---------|
| DoD    | 77.67s    | 96.37s   | 0.81x   | Slower  |
| C2C    | 149.09s   | 107.71s  | 1.38x   | Modest  |
| M3C2   | 117.43s   | 125.97s  | 0.93x   | Slower  |

**Key Finding**: Parallelization overhead (2-5s per run) exceeds benefits for small datasets (<20M points). Fixed costs (alignment ~45s, visualization ~15s) dominate total execution time.

**When Parallelization Helps**:
- Datasets > 50M points (overhead becomes <5%)
- Batch processing multiple areas
- Compute-intensive methods (M3C2-EP)
- Many tiles (50+) for better load balancing

### Architecture

**Parallel Processing Flow**:
1. Partition tiles among worker processes
2. Each worker:
   - Loads tile data via streaming (inner + halo regions)
   - Applies transformations
   - Computes change detection
   - Returns results
3. Main process assembles results (grid mosaicking or point concatenation)

**Design Decisions**:
- **Module-level workers**: Required for pickling with multiprocessing
- **Streaming per tile**: Memory-efficient, no full dataset loading
- **Result assembly varies by method**:
  - DoD: Grid mosaicking with halo blending
  - C2C: Distance array concatenation
  - M3C2: Ordered reassembly by core point indices

### Testing

**Validation**:
- âœ… DoD parallel: 12 tiles in 39.50s
- âœ… C2C parallel: 9.06M points in 46.72s
- âœ… M3C2 parallel: 20,000 cores in 46.31s
- âœ… All methods produce correct results (verified statistics)
- âœ… Error handling and empty tile management

### Rationale

**Why Implement Despite Current Scale Results**:
1. **Scalability Foundation**: Infrastructure ready for 100M+ point datasets
2. **Modular Architecture**: Clean separation of concerns for future optimization
3. **Batch Processing**: Essential for processing multiple areas
4. **GPU Preparation**: Parallel tile processing enables GPU acceleration per tile (Phase 2)

**Why Limited Current Benefit**:
1. Small dataset (12 tiles, ~750K points/tile)
2. High overhead-to-computation ratio
3. I/O bandwidth contention with 11 concurrent readers
4. Fixed costs (alignment, visualization) dominate total time

### Next Steps (Phase 1.3 - Week 3)

**I/O Optimization** (if pursuing larger datasets):
- Spatial pre-indexing to reduce redundant file reads
- Lazy tile generation (stream on-demand)
- Memory-mapped result arrays

**Production Polish** (Week 4):
- Adaptive worker count based on dataset size
- Enhanced progress reporting
- Configuration validation and tuning guide

**Alternative Path** (if staying at current scale):
- Focus on sequential optimizations
- Skip to Phase 2 (GPU) only for methods with high per-point cost (M3C2-EP)

### Files Changed
- `src/terrain_change_detection/acceleration/tile_parallel.py` (new)
- `src/terrain_change_detection/acceleration/tile_workers.py` (new)
- `src/terrain_change_detection/detection/change_detection.py` (3 parallel methods added)
- `scripts/run_workflow.py` (parallel routing added)
- `scripts/run_benchmark.ps1` (new)
- `config/profiles/bench_*.yaml` (6 new configs)
- `BENCHMARK_RESULTS.md` (new)

---

## 2025-11-09 - Performance Optimization Strategy Reassessment

### Summary
Reassessed performance optimization approach based on completed out-of-core tiling infrastructure. Created new two-phase implementation plan: CPU parallelization first, then GPU acceleration.

### What Changed
**Branch Synchronization**:
- Merged `feat/outofcore-tiling` into `feat/gpu-acceleration` branch (19 commits)
- Now have complete tiling infrastructure as foundation for optimization

**Documentation Reorganization**:
- Archived outdated planning documents to `docs/archive/`:
  - `REVISED_PLAN.md` (pre-tiling assessment)
  - `PERFORMANCE_OPTIMIZATION_PLAN.md` (assumed no tiling existed)
  - `GPU_IMPLEMENTATION_GUIDE.md` (premature GPU focus)
  - `OPTIMIZATION_SUMMARY.md` (outdated analysis)
  - `ROADMAP.md` (old roadmap)
  - `QUICK_REFERENCE.md` (old reference)

**New Planning Documents**:
- Created `PARALLELIZATION_PLAN.md`: Comprehensive Phase 1 plan for CPU parallelization
  - 4-week implementation plan
  - Tile-level multiprocessing using existing tiling infrastructure
  - Target: 6-12x speedup on typical hardware
  - Detailed implementation tasks, code examples, testing strategy
  
- Created `GPU_ACCELERATION_PLAN.md`: Phase 2 plan for GPU acceleration
  - 5-week implementation plan (after Phase 1 complete)
  - Operation-level GPU acceleration (NN searches, grid ops)
  - Target: Additional 5-15x speedup (30-50x total)
  - CuPy/cuML/Numba technology stack
  
- Created `ROADMAP.md`: High-level timeline and strategy overview
  - Two-phase approach with clear milestones
  - Performance targets per phase
  - Hardware requirements
  - Risk management and success criteria

### Rationale

**Why Two Phases**:
1. **Foundation First**: Out-of-core tiling creates naturally independent units of work (tiles) perfect for parallelization
2. **Lower Complexity**: CPU parallelization is simpler, more portable, and delivers significant gains (6-12x)
3. **Compounding Benefits**: Parallel CPU workers + GPU acceleration = multiplicative speedup (30-180x total)
4. **Risk Mitigation**: Validate parallel architecture before adding GPU complexity

**Key Insight**: The sequential tile processing loop is the primary bottleneck. On 8-core machines, we're using only 12-15% of CPU capacity. Parallelization is the highest-priority, lowest-risk optimization.

**Why Not GPU First**:
- GPU acceleration is more complex (memory management, CUDA, driver issues)
- GPU benefits are largest for operations within tiles (NN searches, grid ops)
- Better to parallelize tiles first, then accelerate per-tile operations
- Phase 1 delivers substantial value quickly; Phase 2 builds on stable foundation

### Implementation Status
- **Phase 1** (CPU Parallelization): Ready to begin Week 1
  - Target: Parallel tile processing for DoD, C2C, M3C2
  - Infrastructure: `parallel_executor.py`, `tile_workers.py`, spatial indexing
  - Expected: 6-12x speedup in 4 weeks

- **Phase 2** (GPU Acceleration): Starts after Phase 1 complete
  - Target: GPU NN searches, grid operations, JIT kernels
  - Technology: CuPy, cuML, Numba
  - Expected: Additional 5-15x speedup (30-50x total)

### Next Steps
1. Begin Phase 1 Week 1: Parallel executor infrastructure
2. Implement parallel DoD with initial benchmarks
3. Extend to C2C and M3C2 in Week 2
4. Optimize I/O in Week 3
5. Polish and production-ready in Week 4

### References
- Planning: `docs/PARALLELIZATION_PLAN.md`, `docs/GPU_ACCELERATION_PLAN.md`, `docs/ROADMAP.md`
- Foundation: `src/terrain_change_detection/acceleration/tiling.py`
- Archived docs: `docs/archive/`

---

## 2025-11-09 - Config Schema Completion

### Summary
Finalized out-of-core configuration schema with missing fields for production readiness.

### Changes
- Changed `save_transformed_files` default from `true` to `false` for more conservative behavior (opt-in file writing).
- Added `memmap_dir` field to `OutOfCoreConfig` for optional memory-mapped array backing in mosaicking operations.
- Updated all config profiles (`default.yaml`, `large_scale.yaml`, `synthetic.yaml`) with complete outofcore section.

### Why It Matters
- `save_transformed_files: false` prevents unintended disk usage; users must explicitly enable transformed file output.
- `memmap_dir` provides escape hatch for very large mosaicking operations that exceed available RAM.
- Config schema is now complete and production-ready for merge.

---

## 2025-11-06 - Out-of-Core Processing & Tiling (Complete Implementation)

### Summary
Implemented complete out-of-core processing infrastructure with tiling support for all three change detection methods (DoD, C2C, M3C2). This enables processing of datasets that exceed available memory by dividing spatial domains into tiles and streaming point data in chunks.

### Core Infrastructure
**Tiling System** (`src/terrain_change_detection/acceleration/tiling.py`):
- `Tiler`, `Tile`, `Bounds2D` classes for grid-aligned tiling with inner/outer bounds and halo support.
- `LaspyStreamReader` for chunked LAZ/LAS reading with bbox and classification filtering.
- `GridAccumulator` for streaming mean aggregation over regular XY grids (memory-efficient DEM building).
- `MosaicAccumulator` for seamlessly stitching tile DEMs with overlap averaging, includes optional memmap backing for very large grids.

### Change Detection Methods
**DoD (DEM of Difference)**:
- `compute_dod_streaming_files_tiled()` - out-of-core DoD with grid-aligned mosaicking.
- Single-pass chunk routing: each epoch scanned once; chunks routed to tile accumulators by XY position.
- Supports mean aggregator in streaming mode (median/percentiles deferred for future work).
- Optional on-disk mosaicking via `memmap_dir` parameter to reduce RAM for huge grids.
  
**C2C (Cloud-to-Cloud)**:
- `compute_c2c_streaming_files_tiled()` - tiled C2C with radius-bounded nearest-neighbor queries.
- Requires finite `max_distance` to bound search; tile halo = search radius.
- Streams source (inner tile) and target (outer tile with halo) points per tile.
- Uses sklearn k-d tree when available, with brute-force fallback.
- Added `compute_c2c_vertical_plane()` for terrain-aware distances with local plane fitting.
  
**M3C2 (Multiscale Model-to-Model)**:
- `compute_m3c2_streaming_files_tiled()` - tiled M3C2 processing with py4dgeo integration.
- Partitions core points spatially, streams tile-local points from both epochs per tile.
- Uses safe halo = `max(cylinder_radius, projection_scale)` by default for neighborhood completeness.
- Stitches per-tile results back in input order.

### Configuration & Workflow
**New `outofcore` Config Section**:
- `enabled`: Master switch for streaming/tiling workflows.
- `tile_size_m`: Tile size in meters for spatial partitioning.
- `halo_m`: Halo/buffer width around tiles to avoid edge artifacts.
- `chunk_points`: Max points per streaming chunk (memory/performance trade-off).
- `streaming_mode`: Use streaming in preprocessing when enabled.
- `save_transformed_files`: Optionally save aligned LAZ files (default: false).
- `output_dir`: Directory for transformed files (auto-generated if null).
- `memmap_dir`: Directory for memory-mapped mosaicking arrays (in-memory if null).

**Workflow Integration** (`scripts/run_workflow.py`):
- Automatically detects and routes between in-memory vs. streaming execution paths based on config.
- Streaming mode uses reservoir sampling for alignment subsampling (memory-safe).
- File-based alignment transformations with `apply_transform_to_files()` for out-of-core workflows.
- Enhanced logging: step framing, extent metrics (m/km), tile grids, chunk counts, timings.
- Fallback to in-memory mode if streaming fails.

**Configuration Profiles**:
- `config/default.yaml`: Conservative defaults, out-of-core disabled.
- `config/profiles/large_scale.yaml`: Optimized for datasets too large for memory.
- `config/profiles/synthetic.yaml`: Small datasets for development/testing.
- Config toggles to enable/disable individual detection methods (`dod.enabled`, `c2c.enabled`, `m3c2.enabled`).

### Supporting Features
- **Streaming Preprocessing**: `BatchLoader` supports `streaming_mode` returning file paths instead of loading all data.
- **File-Based Alignment**: `streaming_alignment.py` module with `apply_transform_to_files()` for transforming large datasets in chunks.
- **Shared Filtering**: `utils/point_cloud_filters.py` with reusable classification filtering logic.
- **Discovery & Metadata**: Header-only metadata + streamed classification counts for accurate totals without loading full arrays.
- **Streaming Transforms**: When transformed files aren't saved, T2 chunks are transformed on-the-fly for DoD/M3C2/C2C.

### Visualization
- In-memory C2C 3D visualization via `PointCloudVisualizer.visualize_c2c_points()` (colors by distance).
- Streaming C2C falls back to histogram visualization (point-level viz requires in-memory mode).

### Documentation
- `docs/ALGORITHMS.md`: Comprehensive guide explaining DoD/C2C/M3C2 tiling applicability, halo sizing, and when to use each method.
- All classes/methods in `tiling.py` have detailed docstrings.

### Testing & Validation
- Unit tests: grid accumulator mean parity, tiler/mosaic identity.
- Integration tests: streaming mode, config validation, point cloud filtering.
- Successfully validated with 15M + 20M point datasets using constant memory.
- Tiled DoD computes 481K cells efficiently with 1000m tiles.
- File transformation: 9M points processed in ~3 seconds.
- All 40+ tests passing.

### Bug Fixes
- Fixed laspy 2.x API incompatibility in `tiling.py` - changed `len(las.points)` to `len(las)`.
- Fixed `FileNotFoundError` when log directory doesn't exist - added `mkdir` in `setup_logger()`.
- Fixed missing `Tile` import in detection module.
- Fixed missing `LaspyStreamReader` import in workflow.
- Normalized whitespace to avoid indentation errors.
- Corrected tile grid computation order in streaming M3C2 logging.
- Coarse registration guard: validates PCA/Phase/Open3D against centroid baseline; auto-fallback if worse.

### Performance
- Successfully processed 15M + 20M point datasets using constant memory.
- Single-pass chunk routing eliminates O(#tiles Ã— #files) I/O pattern.

### Known Limitations (By Design)
- DoD streaming supports only mean aggregator (median/percentiles require different approach).
- No parallelization (CPU single-threaded) - deferred to future branch.
- No GPU acceleration - deferred to future branch.
- C2C streaming returns distances only, no target indices (memory optimization).

### Future Work
- Parallelization: multi-process tile processing, threaded chunk reading.
- GPU acceleration: CUDA-based neighborhood queries and grid operations.
- Additional aggregators: streaming median/percentiles for DoD.
- Cache per-file spatial subsets or integrate PDAL filters to reduce re-scans in tiled workflows.
- Speed up alignment subsampling while staying memory-safe (fast mode, early-stop factor, parallel sampling).

---

## 2025-11-05 - External Configuration System

### Summary
Externalized central pipeline parameters into human-readable YAML configuration with typed validation (pydantic). Makes it easier to tune analyses without code changes and prepares for future UI-based configuration.

### Features
- **YAML Configuration**: Typed validation using pydantic with sensible defaults.
- **Config Files**: `config/default.yaml` and `config/profiles/synthetic.yaml`.
- **CLI Flags**: `--config` for YAML file path, `--base-dir` to override dataset root.
- **Configurable Parameters**:
  - Preprocessing: `ground_only`, `classification_filter`
  - Discovery: `data_dir_name`, `metadata_dir_name`
  - Alignment (ICP): `max_iterations`, `tolerance`, `max_correspondence_distance`, `subsample_size`
  - Detection (DoD): `cell_size`, `aggregator`
  - Detection (C2C): `max_points`, `max_distance`
  - Detection (M3C2): `core_points`, autotune parameters, EP workers
  - Visualization: `backend` (plotly/pyvista/pyvistaqt), `sample_size`
  - Logging: `level`, `file`
  - Performance: `numpy_threads`

### Improvements
- Loader info logs explicitly state "ground points" when `ground_only=True`.
- Config-driven sample sizes eliminate hardcoded constants.
- Shared loader instance injected into discovery and batch loading for consistency.

### Backward Compatibility
- Default behavior unchanged when run without flags.
- All modules retain function defaults for direct programmatic use.

---

## 2025-11-05 - Coarse Registration & Open3D Integration

### Summary
Added coarse registration stage ahead of ICP with multiple methods (centroid, PCA, phase correlation, Open3D FPFH) and made Open3D an optional dependency.

### Features
- **Coarse Registration Module** (`alignment/coarse_registration.py`):
  - `centroid`: Simple translation alignment
  - `pca`: PCA-based orientation alignment
  - `phase`: Phase correlation for translation estimation
  - `open3d_fpfh`: FPFH feature-based RANSAC (requires Open3D)
- **Config Integration**: `alignment.coarse.enabled`, `alignment.coarse.method`, `alignment.coarse.voxel_size`, `alignment.coarse.phase_grid_cell`
- **Workflow Integration**: Coarse transform used to initialize ICP when enabled
- **Validation Guard**: Validates coarse results against centroid baseline; auto-fallback if worse

### Dependencies
- Made Open3D optional - only required for `open3d_fpfh` method
- Works without Open3D for other coarse methods and in-memory workflows

### Testing
- Added `test_coarse_registration.py` with tests for all methods
- Config integration tests validate coarse registration settings

---

## Earlier Work (Pre-Out-of-Core Branch)

### M3C2 Implementation
- Integrated py4dgeo for M3C2 and M3C2-EP (Error Propagation) computation
- Auto-tuning function for M3C2 parameters based on point density
- M3C2-EP with significance testing and level-of-detection thresholds
- Configurable via `detection.m3c2` section

### Visualization Enhancements
- Multiple backend support: plotly, pyvista, pyvistaqt
- Interactive 3D point cloud visualization with sampling
- DoD heatmaps with configurable colormaps
- M3C2 core point visualization colored by distance
- Distance histograms for C2C and M3C2 results

### Testing Infrastructure
- Comprehensive test suite covering preprocessing, alignment, detection
- Sample data for reproducible testing
- Integration tests for end-to-end workflows

### Documentation
- README with installation instructions and usage examples
- Configuration guide explaining all parameters
- Algorithm documentation (ALGORITHMS.md) explaining methods
- Bug fix documentation (BUGFIX_LASPY_API.md)
