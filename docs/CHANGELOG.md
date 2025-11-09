# Changelog and Implementation Notes

## 2025-11-09 — Performance Optimization Strategy Reassessment

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

## 2025-11-09 — Config Schema Completion

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

## 2025-11-06 — Out-of-Core Processing & Tiling (Complete Implementation)

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
- Single-pass chunk routing eliminates O(#tiles × #files) I/O pattern.

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

## 2025-11-05 — External Configuration System

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

## 2025-11-05 — Coarse Registration & Open3D Integration

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
