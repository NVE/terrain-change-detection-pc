# Changelog and Implementation Notes

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
# Changelog and Implementation Notes

## 2025-11-11 â€” M3C2 Stats Robustness and Streaming Consistency

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

## 2025-11-11 â€” Logging + Progress UX Overhaul

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

## 2025-11-10 â€” CPU Parallelization Refinements and I/O Pruning

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

## 2025-11-09 â€” Large Synthetic Dataset for Performance Testing

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

## 2025-11-09 â€” CPU Parallelization Implementation Complete (Phase 1)

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
- Created benchmark configs: `bench_{dod,c2c,m3c2}_{seq,par}.yaml`
- Updated `large_scale.yaml` with parallel settings

**Benchmarking**:
- Created `scripts/run_benchmark.ps1`: PowerShell script for performance testing
- Created `BENCHMARK_RESULTS.md`: Comprehensive performance analysis
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

## 2025-11-09 â€” Performance Optimization Strategy Reassessment

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

## 2025-11-09 â€” Config Schema Completion

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

## 2025-11-06 â€” Out-of-Core Processing & Tiling (Complete Implementation)

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

## 2025-11-05 â€” External Configuration System

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

## 2025-11-05 â€” Coarse Registration & Open3D Integration

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
## 2025-11-13 – Alignment, Tiling, and M3C2 Autotune Consistency

### Highlights
- Fixed mosaic accumulation bug (masked writes on views) causing lost cells in DoD mosaics.
- Respected ground_only/classification filtering in parallel tile workers (DoD, C2C, M3C2) to match in-memory behavior.
- Ensured T2?T1 transform is applied in all streaming/tiled paths (DoD/C2C/M3C2), including per-chunk application.
- M3C2: Resolved cross-mode discrepancies by introducing consistent parameter handling and diagnostics.
  - Added header-based autotune option for density (utotune.source: header|sample).
  - Added fixed-parameter mode (detection.m3c2.use_autotune: false + ixed.{radius, normal_scale, depth_factor}).
  - Exposed CLI overrides: --m3c2-radius, --m3c2-normal-scale, --m3c2-depth-factor.
  - Added reproducibility helpers: --seed, --cores-file to save/load core sets.
  - Added --debug-m3c2-compare to print corr(stream,inmem), corr(..., dZ) and sign/quantile summaries.

### Files Touched
- DoD/tiling
  - src/terrain_change_detection/acceleration/tiling.py: fix MosaicAccumulator.add_tile to index global arrays directly.
- Parallel workers
  - src/terrain_change_detection/acceleration/tile_workers.py: pass ground_only; apply transforms per chunk.
- Change detection API
  - src/terrain_change_detection/detection/change_detection.py:
    - utotune_m3c2_params_from_headers(...) (new) for mode-agnostic autotune.
    - Minor logging/param propagation improvements.
  - src/terrain_change_detection/detection/__init__.py: export new API.
- Workflow
  - scripts/run_workflow.py:
    - Respect use_autotune/ixed params from YAML.
    - Implement header/sample autotune selection per config and log chosen params.
    - Add --seed, --cores-file, --debug-m3c2-compare, --m3c2-*.
- Config
  - src/terrain_change_detection/utils/config.py:
    - Add detection.m3c2.use_autotune, detection.m3c2.fixed, and utotune.source.
  - YAML profiles (default, synthetic, large_synthetic, large_scale): include the new keys.

### Notes for Users
- For reproducible production runs, prefer fixed M3C2 parameters via YAML.
- For mode-agnostic autotune, set detection.m3c2.autotune.source: header (now default in profiles).
- Use --cores-file to compare streaming vs in-memory on identical core points.



