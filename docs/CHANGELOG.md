# Changelog and Implementation Notes

## 2025-11-09 — Config Schema Completion

### Summary
Finalized out-of-core configuration schema with missing fields for production readiness.

### Changes
- **Config Schema Updates**:
  - Changed `save_transformed_files` default from `true` to `false` for more conservative behavior (opt-in file writing).
  - Added `memmap_dir` field to `OutOfCoreConfig` for optional memory-mapped array backing in mosaicking operations.
  - Updated all config profiles (`default.yaml`, `large_scale.yaml`, `synthetic.yaml`) with complete outofcore section.
- **Documentation**: Updated root `CHANGELOG.md` with comprehensive out-of-core feature summary.

### Why It Matters
- `save_transformed_files: false` prevents unintended disk usage; users must explicitly enable transformed file output.
- `memmap_dir` provides escape hatch for very large mosaicking operations that exceed available RAM.
- Config schema is now complete and production-ready for merge.

---

## 2025-11-07 — Feature Branch Summary & Pre-Merge Review

### Summary
Comprehensive summary of all out-of-core processing and tiling features developed in the `feat/outofcore-tiling` branch, prepared for merge to main.

### Core Infrastructure Added
- **Tiling System**:
  - `Tiler`, `Tile`, `Bounds2D` classes for grid-aligned tiling with inner/outer bounds and halo support.
  - Enables processing large datasets by dividing into manageable spatial tiles.
  - Halo regions around tiles prevent edge artifacts in neighborhood operations.
  
- **Streaming Data Loading**:
  - `LaspyStreamReader` for chunked LAZ/LAS reading with bbox and classification filtering.
  - `GridAccumulator` for streaming mean aggregation over regular XY grids (memory-efficient DEM building).
  - `MosaicAccumulator` for seamlessly stitching tile DEMs with overlap averaging, includes optional memmap backing.

### Change Detection Methods (Streaming/Tiled)
- **DoD (DEM of Difference)**:
  - `compute_dod_streaming_files_tiled()` - out-of-core DoD with grid-aligned mosaicking.
  - Single-pass chunk routing: each epoch scanned once; chunks routed to tile accumulators.
  - Supports mean aggregator in streaming mode (median/percentiles deferred for future work).
  
- **C2C (Cloud-to-Cloud)**:
  - `compute_c2c_streaming_files_tiled()` - tiled C2C with radius-bounded nearest-neighbor queries.
  - Requires `max_distance` to bound search; tile halo = radius.
  - Streams source (inner) and target (outer) per tile; uses sklearn NN when available.
  - Also added `compute_c2c_vertical_plane()` for terrain-aware distances with local plane fitting.
  
- **M3C2**:
  - `compute_m3c2_streaming_files_tiled()` - tiled M3C2 processing core points with py4dgeo integration.
  - Uses safe halo = `max(cylinder_radius, projection_scale)` by default.
  - Per-tile streaming and execution with stitched results; clear timing and counts.

### Configuration & Workflow
- **Configuration Support**:
  - New `outofcore` config section with `enabled`, `tile_size_m`, `halo_m`, `chunk_points`, `streaming_mode`.
  - `save_transformed_files` flag (default: false) to optionally save aligned LAZ files.
  - `output_dir` for controlling transformed file output location.
  - `memmap_dir` for memory-mapped array backing in very large mosaicking operations.
  - Config toggles to enable/disable individual detection methods (`dod`, `c2c`, `m3c2`).

- **Workflow Integration**:
  - `run_workflow.py` automatically detects and routes between in-memory vs. streaming execution paths.
  - Streaming mode includes reservoir sampling for alignment subsampling.
  - File-based alignment transformations with `apply_transform_to_files()` for out-of-core workflows.
  - Enhanced logging with transparency on chunk processing, tile statistics, and processing times.
  
- **Configuration Profiles**:
  - `large_scale.yaml` profile configured for out-of-core processing of large datasets.
  - `synthetic.yaml` for small in-memory development/testing.
  - Standardized comments and structure across all YAML configs.

### Visualization Enhancements
- In-memory C2C 3D visualization similar to M3C2.
- `PointCloudVisualizer.visualize_c2c_points()` colors source points by C2C distances.
- Streaming C2C falls back to histogram visualization (point-level viz requires in-memory mode).

### Documentation
- Comprehensive `docs/ALGORITHMS.md` explaining DoD/C2C/M3C2 tiling applicability.
- Clear guidance on when to use each method and how tiling applies to each.
- All classes/methods in `tiling.py` have detailed docstrings.

### Testing & Validation
- Unit tests for grid accumulator mean parity and tiler/mosaic identity.
- Successfully validated with 15M + 20M point datasets using constant memory.
- All 40+ tests passing.

### Known Limitations (By Design)
- DoD streaming supports only mean aggregator (median/percentiles require different approach).
- No parallelization (CPU single-threaded) - deferred to future branch.
- No GPU acceleration - deferred to future branch.
- C2C streaming returns distances only, no target indices (memory optimization).

### What's Next
- Parallelization: multi-process tile processing, threaded chunk reading.
- GPU acceleration: CUDA-based neighborhood queries and grid operations.
- Additional aggregators: streaming median/percentiles for DoD.

---

## 2025-11-06 — Out-of-Core Reliability & UX

Summary
- Strengthened out-of-core operation end-to-end, clarified logs for engineers, and added streaming M3C2 with tiling. Kept discovery and alignment subsampling memory-safe while exposing accurate dataset stats (ground vs total).

Highlights
- DoD (tiled streaming)
  - Single-pass chunk routing: each epoch scanned once; chunks routed to tile accumulators.
  - Optional on-disk mosaicking (`memmap_dir`) to reduce RAM for very large grids.
  - Engineer-friendly logs: global extent in meters and km; tile grid and timing.
- M3C2 (tiled streaming)
  - New API: `ChangeDetector.compute_m3c2_streaming_files_tiled(...)`.
  - Uses safe halo = `max(cylinder_radius, projection_scale)` by default.
  - Per-tile streaming and execution with stitched results; clear timing and counts.
  - Integrated into workflow when out-of-core is enabled; fallback to in-memory when needed.
- C2C (tiled streaming)
  - New API: `ChangeDetector.compute_c2c_streaming_files_tiled(...)`.
  - Requires `max_distance` to bound search; tile halo = radius.
  - Streams source (inner) and target (outer) per tile; uses sklearn NN when available.
  - Integrated into workflow when out-of-core is enabled and a radius is set.
- Workflow logging & UX
  - Clear step framing: Step 1 (Data Prep), Step 2 (Spatial Alignment), Step 3 (Change Detection).
  - Logs core/global extents and tile grids; prints ground/total dataset points with percentage.
- Discovery & metadata
  - Header-only metadata + streamed classification counts (no full arrays) for accurate totals and ground counts.
  - Streaming-based reservoir sampler for alignment subsampling (no full-file loads).

Bug fixes & robustness
- Fixed missing `Tile` import in detection module for tiled DoD.
- Normalized whitespace to avoid indentation errors.
- Corrected tiles log order in streaming M3C2 (compute tx/ty before logging).
- Added missing `LaspyStreamReader` import in workflow; ensured streaming alignment sampling works.
- Streaming transforms: when transformed files aren’t saved, streamed T2 chunks are transformed on-the-fly for DoD/M3C2/C2C to keep parity with in-memory paths.
- Coarse registration guard: PCA/Phase/Open3D results are validated against a centroid baseline on a small NN RMSE; if clearly worse, we auto-fallback to centroid. Prefer `centroid` for synthetic/symmetric cases.

To Do
- Speed up alignment subsampling while staying memory-safe:
  - Add config flag for “fast” in-memory sampling (opt-in) for small/single-file cases.
  - Add early-stop factor for reservoir sampler (e.g., stop after ~3–5× sample_size points).
  - Explore parallel chunk sampling with a bounded reservoir and warp-level merging.

## 2025-11-06 — Out-of-Core Processing Integration & Streaming Pipeline

### Summary
Fully integrated out-of-core tiling capabilities with the main workflow, added streaming support throughout preprocessing/alignment/detection, and fixed critical laspy 2.x API compatibility issue.

### New Features
- **Streaming Preprocessing**: `BatchLoader` now supports `streaming_mode` returning file paths instead of loading all data
- **File-Based Alignment**: New `streaming_alignment.py` module with `apply_transform_to_files()` for transforming large datasets in chunks
- **Shared Filtering Utilities**: Created `utils/point_cloud_filters.py` with reusable classification filtering logic
- **Configuration Profiles**: Added `large_scale.yaml` profile for datasets too large for memory
- **Out-of-Core Config**: Extended `OutOfCoreConfig` with `streaming_mode`, `save_transformed_files`, `output_dir` parameters

### Bug Fixes
- **Critical**: Fixed laspy 2.x API incompatibility in `tiling.py` - changed `len(las.points)` to `len(las)` 
- **Logging**: Fixed `FileNotFoundError` when log directory doesn't exist - added `mkdir` in `setup_logger()`

### Improvements
- **Documentation**: Added comprehensive docstrings to all classes/methods in `tiling.py`
- **Enhanced Logging**: Added detailed logging throughout streaming pipeline for diagnostics
- **Error Handling**: Added fallback to in-memory DoD if streaming fails
- **Workflow Coordination**: Main workflow now detects streaming mode and uses appropriate code paths

### Testing
- Created 13 new tests covering streaming integration, config validation, and point cloud filtering
- Added diagnostic script `test_tiled_dod.py` for isolated DoD testing
- All 40 tests passing (6 streaming/tiling specific)

### Performance
- Successfully processed 15M + 20M point datasets using constant memory
- Tiled DoD computes 481K cells efficiently with 1000m tiles
- File transformation: 9M points processed in ~3 seconds

---

## 2025-11-06 — Tiled Streaming Optimization & Memmap Mosaic

### Summary
Improved tiled out-of-core DoD to avoid re-reading files per tile and added optional on-disk (memmap) mosaicking for very large output grids.

### What Changed
- Reworked `ChangeDetector.compute_dod_streaming_files_tiled` to stream each epoch (T1/T2) once and route chunks to tile accumulators by index. This removes the previous O(#tiles -- #files) I/O pattern.
- `MosaicAccumulator` now supports an optional `memmap_dir` parameter. When provided, the per-cell `sum` and `cnt` arrays are backed by `numpy.memmap` files on disk, reducing RAM pressure for huge grids.
- Added progress logging for tiled streaming: tiles geometry, per-epoch point counts, and timing of streaming and mosaicking phases.

### Why It Matters
- Large datasets now benefit from real out-of-core behavior: each file is scanned once per epoch, and only tile-local accumulators are kept in memory.
- Very large global grids can be mosaicked with reduced RAM usage via on-disk arrays.

### Backwards Compatibility
- API of `compute_dod_streaming_files_tiled` is preserved; a new optional argument `memmap_dir` was added (default `None`).
- Existing tests and workflows continue to function. All tests pass.

### Usage Example
```python
res = ChangeDetector.compute_dod_streaming_files_tiled(
    files_t1=files_t1,
    files_t2=files_t2,
    cell_size=2.0,
    tile_size=1000.0,
    halo=50.0,
    chunk_points=2_000_000,
    memmap_dir="./.mosaic_tmp"  # optional, enables on-disk mosaicking
)
```

Note: `grid_x`/`grid_y` are still produced in-memory; for extremely large domains prefer writing mosaics directly to formats like GeoTIFF or chunked arrays in a future step.

## 2025-11-06 --- Out-of-Core M3C2 (Tiled)

### Summary
Added tiled, streaming-based M3C2 wrapper that partitions core points spatially, streams tile-local points from files, runs py4dgeo M3C2 per tile, and stitches results back in input order.

### What Changed
- New API: `ChangeDetector.compute_m3c2_streaming_files_tiled(...)`
  - Inputs: `core_points`, `files_t1`, `files_t2`, `params`, and tiling/streaming controls.
  - Uses halo = max(cylinder_radius, projection_scale) unless overridden, ensuring neighborhood completeness.
  - Clear logging per tile with counts and timings.

### Why It Matters
- Enables M3C2 on datasets that exceed memory by avoiding full in-memory epochs; only tile-local neighborhoods are materialized at a time.

### Usage Example
```python
from terrain_change_detection.detection import ChangeDetector, M3C2Params

m3c2_res = ChangeDetector.compute_m3c2_streaming_files_tiled(
    core_points=core_src,           # Nx3 (sampled from T1 or grid)
    files_t1=files_t1,              # epoch 1 LAZ/LAS
    files_t2=files_t2_aligned,      # epoch 2 LAZ/LAS (aligned)
    params=m3c2_params,             # tuned via autotune_m3c2_params
    tile_size=1000.0,
    halo=None,                      # default uses max(cyl_radius, projection_scale)
    ground_only=True,
    classification_filter=[2],
    chunk_points=2_000_000,
)
```

### Notes
- For very large numbers of tiles, this approach re-reads epoch files per tile. A future iteration may cache per-file spatial subsets or integrate PDAL filters to reduce rescans.

## 2025-11-05 --- External Configuration System

Date: 2025-11-05

This document summarizes the new features, improvements, and behavior changes introduced with the addition of a typed, external configuration system to the terrain-change-detection project.

## Overview

We externalized central pipeline parameters into a human-readable YAML configuration, validated by a typed schema (pydantic). The main workflow now reads parameters from YAML while keeping safe defaults. This makes it easier to tune analyses without code changes and prepares the project for future UI-based configuration.

## Highlights

- YAML configuration with typed validation (pydantic) and sensible defaults.
- New config files: `config/default.yaml` and `config/profiles/synthetic.yaml`.
- Workflow accepts `--config` and `--base-dir` CLI flags.
- Preprocessing and discovery stages are configurable (ground filtering; subfolder names).
- Logging level/file, visualization backend/sample size, and performance threads are configurable.
- Clearer loader logging: explicitly reports ---ground points--- when ground filtering is enabled.

## New Features

1) External Configuration (YAML)
- Default config auto-loaded from `config/default.yaml`.
- Example profile: `config/profiles/synthetic.yaml`.
- Typed schema defined in `src/terrain_change_detection/utils/config.py` using pydantic.

2) CLI Enhancements
- `--config <path>`: specify YAML file path (optional).
- `--base-dir <path>`: override `paths.base_dir` from YAML.

3) Configurable Preprocessing and Discovery
- Preprocessing:
  - `preprocessing.ground_only` (default true).
  - `preprocessing.classification_filter` (default `[2]`).
- Data discovery:
  - `discovery.data_dir_name` and `discovery.metadata_dir_name` (defaults `data`/`metadata`).
- Shared loader instance is injected into discovery and batch loading to keep behavior consistent.

4) Configurable Algorithms and Visualization
- Alignment (ICP): `alignment.{max_iterations,tolerance,max_correspondence_distance,subsample_size}`.
- DoD: `detection.dod.{cell_size,aggregator}`.
- C2C: `detection.c2c.{max_points,max_distance}`.
- M3C2: `detection.m3c2.core_points`, `detection.m3c2.autotune.*`, `detection.m3c2.ep.workers`.
- Visualization: `visualization.{backend,sample_size}` (`plotly`/`pyvista`/`pyvistaqt`).

5) Logging and Performance
- Logging: `logging.{level,file}`. Console logging remains enabled; optional file output can be enabled.
- Performance threads: `performance.numpy_threads` sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS`.

## Improvements

- Loader info logs now explicitly state ---ground points--- when `ground_only=True`; also warns if no ground points are found.
- Aligned-cloud visualization and other steps now use config-driven sample sizes and parameters (no hardcoded constants).
- Code paths constructed from config reduce the need to edit scripts when switching datasets.

## Backward Compatibility

- Default behavior is unchanged when run without flags: the workflow reads `config/default.yaml` which mirrors prior defaults.
- CLI `--base-dir` continues to work and overrides YAML for that setting.
- All modules retain function defaults for direct programmatic use.

## Usage Changes (Summary)

Run with defaults:
```powershell
uv run scripts/run_workflow.py
```

Run with explicit config:
```powershell
uv run scripts/run_workflow.py --config config/default.yaml
```

Override dataset root via CLI:
```powershell
uv run scripts/run_workflow.py --base-dir data/synthetic
```

Use profile for synthetic data:
```powershell
uv run scripts/run_workflow.py --config config/profiles/synthetic.yaml
```

## Configuration Keys (Quick Reference)

- paths
  - base_dir: dataset root folder (default `data/raw`).
- preprocessing
  - ground_only: filter to ground classes (default true).
  - classification_filter: list of LAS class codes to keep (default `[2]`).
- discovery
  - data_dir_name: name of the subfolder that contains LAZ/LAS files (default `data`).
  - metadata_dir_name: name of the metadata subfolder (default `metadata`).
- alignment
  - max_iterations, tolerance, max_correspondence_distance, subsample_size
- detection.dod
  - cell_size, aggregator (`mean|median|p95|p5`).
- detection.c2c
  - max_points, max_distance (optional cap).
- detection.m3c2
  - core_points
  - autotune: target_neighbors, max_depth_factor, min_radius, max_radius
  - ep: workers (null = OS-specific default: 1 on Windows, 4 elsewhere)
- visualization
  - backend (`plotly|pyvista|pyvistaqt`), sample_size
- logging
  - level (`DEBUG|INFO|WARNING|ERROR|CRITICAL`), file (optional path)
- performance
  - numpy_threads: `auto` (default) or an integer

## File Changes

- pyproject.toml
  - Added dependencies: `pydantic`, `PyYAML`.
- config/default.yaml
  - New default configuration file.
- config/profiles/synthetic.yaml
  - Example profile for synthetic datasets.
- src/terrain_change_detection/utils/config.py
  - New: typed AppConfig model and `load_config` loader for YAML.
- scripts/run_workflow.py
  - Integrated config loading and CLI flags; replaced hardcoded constants with config values; applied logging/performance settings.
- src/terrain_change_detection/preprocessing/loader.py
  - Loader now accepts `ground_only` and `classification_filter`; info log explicitly mentions ---ground points--- when applicable.
- src/terrain_change_detection/preprocessing/data_discovery.py
  - Discovery accepts `data_dir_name` and `metadata_dir_name`; uses injected `PointCloudLoader`. `BatchLoader` accepts a loader instance.
- README.md
  - Updated with configuration usage, keys, and examples.

## Upgrade Guide

1) No code changes are required for typical usage --- continue running the workflow as before.
2) To customize behavior, edit `config/default.yaml` or point to an alternative file with `--config`.
3) Override dataset root at the CLI with `--base-dir` for ad hoc runs.
4) On Windows, M3C2-EP worker processes default to 1 when not specified; set `detection.m3c2.ep.workers` to override.

## Known Considerations

- Thread environment variables may be read by numerical libraries at import time. While we set them early in the script, some backends might require setting these before the Python process starts to take full effect.
- Large datasets can be memory intensive; tune `alignment.subsample_size`, `detection.c2c.max_points`, and `detection.m3c2.core_points` accordingly.

## Next Steps (Optional)

- Add environment variable overrides (e.g., `TCD_paths__base_dir`) for containerized deployments.
- Add small config parsing tests and a smoke test for config-driven parameters.
- Consider exposing config parameters through a future web UI.

---

# Changelog and Implementation Notes --- Coarse Registration, Open3D Optional, Config Fix

Date: 2025-11-05

This entry documents the addition of a coarse registration stage ahead of ICP, optional Open3D support for feature-based global matching, integration details in the workflow, and a bug fix in configuration loading.

## Overview

- New coarse registration module supports several initializers to bring clouds into rough alignment before ICP.
- Workflow integrates the coarse stage and passes the resulting transform as an initial guess to ICP for faster convergence and greater robustness on misaligned pairs.
- Optional Open3D-based global feature registration (FPFH + RANSAC) can be used when Open3D is available; otherwise we fall back gracefully to PCA.
- Fixed config root path resolution so YAML overrides are honored reliably.

## Highlights

- New: `CoarseRegistration` with methods:
  - `centroid`: translation-only by centroid alignment.
  - `pca`: rigid transform from principal axes + centroid (default).
  - `phase`: 2D phase correlation on XY occupancy grid (translation).
  - `open3d_fpfh`: feature-based global match via Open3D (optional dependency).
- Workflow improvement: computes a pre-ICP RMSE (non-destructive) and passes `initial_transform` to ICP.
- Optional dependency gating: Open3D extra is only offered for Python < 3.13 to avoid resolver failures.
- Config loader bug fix: repository root detection corrected so `config/default.yaml` and profiles are found.

## New Features

1) Coarse Registration Module
- File: `src/terrain_change_detection/alignment/coarse_registration.py`
- Public API: `CoarseRegistration(method=..., voxel_size=..., phase_grid_cell=...)`
- Returns a 4--4 transform; helper `apply_transformation(points, T)` provided.

2) Workflow Integration
- File: `scripts/run_workflow.py`
- Behavior:
  - If `alignment.coarse.enabled: true`, compute `initial_transform` using the selected method.
  - Log a pre-ICP RMSE computed on a temporary transformed copy of the moving cloud.
  - Pass `initial_transform` to `ICPRegistration.align_point_clouds`.
  - ICP returns the cumulative transform (initial -- delta); that transform is applied to the full moving cloud.

3) Configuration Additions
- File: `src/terrain_change_detection/utils/config.py`
- New model: `CoarseRegistrationConfig` (method, voxel_size, phase_grid_cell, enabled).
- New key under alignment: `alignment.coarse`.
- YAML defaults:
  - `config/default.yaml`: coarse enabled, method `pca`.
  - `config/profiles/synthetic.yaml`: coarse enabled, method `pca`.

4) Optional Dependency (Open3D)
- File: `pyproject.toml`
- `[project.optional-dependencies].open3d = ["open3d>=0.18.0; python_version < '3.13'"]`
- Rationale: Open3D wheels are not available for Python 3.13 at the time of writing; gating avoids unsatisfiable dependency errors.

## Bug Fixes

- Config loader root path resolution
  - File: `src/terrain_change_detection/utils/config.py`
  - Changed `_project_root()` from incorrect parent to the actual repo root.
  - Impact: YAML files are found correctly; edits in `config/default.yaml` (e.g., `alignment.coarse.method`) now take effect.

## Backward Compatibility

- Coarse registration is enabled by default with `method: pca`. This improves alignment without breaking existing workflows.
- If `open3d_fpfh` is selected on Python 3.13 (no Open3D), a clear warning is logged and PCA is used as a safe fallback.

## Usage Changes (Summary)

- Configure coarse registration in YAML:
```yaml
alignment:
  max_iterations: 100
  tolerance: 1.0e-6
  max_correspondence_distance: 1.0
  subsample_size: 50000
  coarse:
    enabled: true
    method: pca        # centroid | pca | phase | open3d_fpfh | none
    voxel_size: 2.0    # for open3d_fpfh only
    phase_grid_cell: 2.0
```

- To use Open3D FPFH global matching:
  - Use Python 3.12 (Open3D wheels are available up to cp312).
  - Install the optional extra: `uv run -m pip install '.[open3d]'`.
  - Set `alignment.coarse.method: open3d_fpfh`.

## Tests

- `tests/test_coarse_registration.py`:
  - Verifies centroid translation recovery and PCA error reduction on synthetic data.
- `tests/test_config_coarse_alignment.py`:
  - Sanity checks for alignment.coarse defaults via `load_config`.

## Documentation

- `README.md` updated to reflect optional coarse registration before ICP and mention Open3D extra.

## Known Limitations

- Open3D not available for Python 3.13: attempting `open3d_fpfh` on 3.13 logs a warning and falls back to PCA.
- Phase correlation estimates XY translation only; rotation still handled by ICP.

