# Changelog and Implementation Notes

## 0.2.0 - 2026-07-06

### Summary
Added structured workflow outputs, reusable run artifacts, clipping improvements, M3C2 evaluation controls, erosion polygon export, and fixes for CRS handling and raster exports.

### Changes
- Added run IDs and artifact management for tracking and reusing workflow results.
- Added M3C2 evaluation summaries and configurable evaluation source for core point selection.
- Added erosion polygon export support and related configuration.
- Improved clipping workflow performance and split clipping feature handling.
- Improved dataset and selected-area typing in workflow data structures.
- Improved CRS detection from LAZ inputs and clipping boundary reprojection.
- Fixed C2C/DoD raster exports and made run outputs unique.

---

## 2026-04-10 - Terrain Change Best Practice Guide

### Summary
Added a comprehensive best practice guide for terrain change detection workflows, covering data quality requirements, parameter selection, validation procedures, and independent verification strategies.

### Key Changes

**Best Practice Guide** (`docs/BEST_PRACTICES_GUIDE.md`):
- Practical guidance for M3C2 parameter selection (normal scale, projection scale, cylinder radius)
- Data quality checklist: point density, temporal consistency, coordinate system alignment
- Validation workflow: unchanged-area statistics, CloudCompare cross-verification
- Guidance on interpreting Level of Detection (LoD) and significance flags

**Evidence Document** (`docs/BEST_PRACTICES_EVIDENCE.md`):
- Supporting data and examples referenced by the guide
- Independent verification methodology using CloudCompare

**Documentation Updates**:
- Updated `README.md` to reference the new guide
- Updated `docs/CONFIGURATION_GUIDE.md` for consistency

---

## 2026-04-09 - Housekeeping: Archive Workflow Refactor Plan

### Summary
Moved the workflow refactor plan document to `docs/archive/` to keep the active docs directory clean. Fixed a WSL environment variable leak in the headless plotly test.

### Key Changes
- Moved `PLAN_REFACTOR_WORKFLOW.md` to `docs/archive/`
- Fixed headless plotly test on WSL: cleared `WSL_DISTRO_NAME`/`WSL_INTEROP` env vars that caused the test to take the WSL branch instead of the truly-headless path

---

## 2026-03-25 - Workflow Refactoring

### Summary
Major refactoring of `run_workflow.py` from a monolithic 1800+ line script into a modular `workflow` package with dedicated modules for each pipeline stage. Also fixed reference epoch routing for ICP alignment.

### Key Changes

**New `workflow` Package** (`src/terrain_change_detection/workflow/`):
- `cli.py`: Command-line argument parsing
- `bootstrap.py`: Configuration loading and initialization
- `data_loading.py`: Point cloud loading and discovery
- `coordinate_setup.py`: Local coordinate transform setup
- `alignment.py`: ICP alignment orchestration
- `clipping.py`: Area clipping logic
- `detection_dod.py`: DoD computation dispatch
- `detection_c2c.py`: C2C computation dispatch
- `detection_m3c2.py`: M3C2 computation dispatch
- `export_helpers.py`: Output export utilities
- `visualization_helpers.py`: Visualization dispatch
- `runner.py`: Main pipeline runner
- `types.py`: Shared dataclasses and type definitions

**Reference Routing Fix**:
- Fixed bug where `--reference t2` CLI argument was not correctly routed through the refactored workflow

**Plotly Headless Fallback**:
- Handle plotly rendering gracefully in headless WSL environments

### Files Changed
- `scripts/run_workflow.py`: Reduced from ~1800 to thin entry point
- 14 new modules in `src/terrain_change_detection/workflow/`
- New `tests/test_workflow_reference_routing.py` with 179 lines of reference routing tests

---

## 2026-03-17 - Simplify Configuration Layering and CLI Overrides

### Summary
Simplified the configuration system by making profile YAML files contain only overrides instead of full copies of all settings. Cleaned up alignment config defaults and improved ICP test assertions.

### Key Changes

**Config System** (`config.py`):
- Implemented deep-merge layering: `default.yaml` → profile YAML → CLI overrides
- Profiles now only specify values that differ from defaults
- Removed ~870 lines of duplicated config across 6 profile files

**Profile YAML Files**:
- `config/default_clipped.yaml`: Reduced to clipping-specific overrides
- `config/profiles/drone.yaml`: Reduced to drone-specific overrides
- `config/profiles/large_scale.yaml`: Reduced to scale-specific overrides
- `config/profiles/large_synthetic.yaml`: Reduced to synthetic-specific overrides
- `config/profiles/large_synthetic_clipped.yaml`: Reduced to clipped synthetic overrides
- `config/profiles/synthetic.yaml`: Reduced to synthetic-specific overrides

**Alignment Defaults** (`default.yaml`):
- Cleaned up ICP alignment default parameters for consistency

**Tests** (`test_config_integration.py`):
- Added 84-line integration test suite for config layering
- ICP tests now use baseline comparison assertions with increased `max_iterations` and `max_correspondence_distance`

### Migration Notes
- **No breaking changes**: Existing workflows continue to work. Profile files are now much smaller and easier to maintain.

---

## 2026-03-11 - ICP Alignment Improvements

### Summary
Six improvements to ICP alignment addressing reproducibility, memory safety, overlap-aware subsampling, reference selection, aligned point cloud export, and Open3D cross-validation.

### Key Changes

**Deterministic Results** (`fine_registration.py`, `config.py`):
- Seed all random operations via `np.random.default_rng` with configurable `random_seed` (default: 42)
- Fixes critical non-reproducibility issue across runs

**Percentage Subsampling**:
- Added `subsample_mode` config (`count` or `percent`)
- Safety cap via `max_subsample_size` (default: 500K) to prevent OOM on large datasets

**Overlap-Aware Subsampling**:
- Filter both clouds to XY bounding-box intersection before subsampling
- ICP only uses points with valid correspondences, improving convergence

**Reference/Target Selection**:
- Added `reference` config (`t1` or `t2`) and `--reference` CLI argument
- Choose which epoch is the fixed ICP reference

**Export Aligned Point Cloud**:
- Wired up the existing `export_aligned_pc` config option that was never implemented in the workflow

**Open3D ICP Backend**:
- Added `icp_backend` config (`custom` or `open3d`)
- New `Open3DICP` wrapper class for cross-validation against a well-known library

### Tests
- 13 new tests added (15 total ICP tests)
- Integration verification report in `docs/ICP_FIX_VERIFICATION_REPORT.md`

---

## 2026-02-26 - README Update: Conda Installation Instructions

### Summary
Updated README with conda installation instructions and usage examples to support users who prefer conda over pip/uv.

---

## 2026-01-15 - Run Input Logging and Pre-Alignment DEM Export

### Summary
Added structured logging of all run inputs at workflow start and the option to export DEM rasters before ICP alignment for quality inspection.

### Key Changes

**Run Input Logging** (`run_workflow.py`):
- Logs all configuration values, file paths, and CLI arguments at the start of each run
- Provides a clear record of inputs for reproducibility and debugging

**Pre-Alignment DEM Export**:
- New option to save DEM rasters before ICP alignment is applied
- Enables visual comparison of pre- vs post-alignment terrain surfaces

---

## 2026-01-14 - Year Selection, Area Filtering, and CLI Enhancements

### Summary
Added support for selecting specific survey years and filtering by area name, along with CLI argument improvements and export directory restructuring.

### Key Changes

**Year Selection and Area Filtering** (`run_workflow.py`, `data_loading.py`):
- Added `--years` CLI argument to select specific survey epochs (e.g., `--years 2015 2020`)
- Added `--area` CLI argument to filter to a specific survey area

**Area-Aware Data Discovery** (`preprocessing/loader.py`):
- Enhanced `scan_areas()` method to filter by user-specified area name
- Returns only matching areas when `--area` is provided

**Export Directory Structure**:
- Updated export paths to include the selected area name (e.g., `output/Doli/...`)
- Prevents output from different areas overwriting each other

**CLI Arguments** (`run_workflow.py`):
- Added `--area` for area name filtering
- Added `--no-plot` flag to suppress visualization (useful for batch/CI runs)

**Type Checking**:
- Added `ty` type checking rules to `pyproject.toml`

---

## 2025-12-18 - Local Coordinate Transform Merge & Client Delivery Prep

### Summary
Merged the `feature/local-coordinate-transform` branch into main. Rewrote README to reflect the current project state. Cleaned up code style and prepared repository for client delivery.

### Key Changes

**Local Coordinate Transform** (merged via PR #5):
- Full local coordinate transformation support across all pipeline modules
- See entries for 2025-12-12 and 2025-12-15 for detailed changes

**README Rewrite** (`README.md`):
- Complete rewrite to reflect current project capabilities and architecture
- Removed references to deleted documentation files

**Code Style**:
- Applied ruff auto-fixes across the codebase
- Fixed duplicate content in README after merge

---

## 2025-12-16 - Døli Benchmarking Data & M3C2 Histogram Comparison

### Summary
Added Døli raw survey data for client delivery and benchmarking. Created an M3C2 histogram comparison script for validating toolkit results against CloudCompare.

### Key Changes

**M3C2 Histogram Comparison** (`scripts/compare_m3c2_histograms.py`):
- New script for comparing M3C2 distance distributions between CloudCompare and the toolkit
- Publication-quality histograms with 256 bins

**Configuration Updates**:
- Updated `default.yaml` and `default_clipped.yaml` with fixed M3C2 params (radius=1.0, depth_factor=2.0)
- Updated visualization colormap to use pale green for stable areas
- Rewrote `docs/CONFIGURATION_GUIDE.md` to align with technical note Section 5

**Døli Raw Data** (`data/raw/`):
- Added Døli area 2015/2020 LAZ files and metadata
- Added river polygon GeoJSON for clipping
- Added CloudCompare M3C2 histogram CSV for benchmarking

**Repository Cleanup**:
- Updated `.gitignore` to include `data/raw/` (client sample data) and exclude generated/test data

---

## 2025-12-15 - ICP Alignment Toggle Feature

### Summary
Added the ability to enable or disable ICP fine registration for spatial alignment. This is useful for pre-aligned datasets where ICP alignment is not needed, saving processing time.

### Key Changes

**Configuration** (`config.py`):
- Added `enabled` field to `AlignmentICPConfig` (default: `true` for backward compatibility)

**YAML Configuration Files**:
- Updated all 7 config files with `alignment.enabled` option:
  - `config/default.yaml`
  - `config/default_clipped.yaml`
  - `config/profiles/drone.yaml`
  - `config/profiles/large_scale.yaml`
  - `config/profiles/large_synthetic.yaml`
  - `config/profiles/large_synthetic_clipped.yaml`
  - `config/profiles/synthetic.yaml`

**Workflow Script** (`run_workflow.py`):
- Step 2 (Spatial Alignment) now checks `alignment.enabled` before running ICP
- When disabled, skips all alignment processing (coarse registration, multi-scale ICP, fine ICP)
- Sets `transform_matrix` to identity and `points2_full_aligned` to original points
- Logs "Spatial Alignment (SKIPPED)" when alignment is disabled

### Usage

```yaml
# Disable alignment for pre-aligned datasets
alignment:
  enabled: false
  # ... other alignment parameters (ignored when disabled)
```

### Migration Notes
- **No breaking changes**: Default `enabled: true` maintains existing behavior
- Existing configs without `enabled` field continue to work (defaults to `true`)

---

## 2025-12-15 - Local Transform Integration Audit & Fixes

### Summary
Comprehensive audit and fix of LocalCoordinateTransform integration across all modules. Added local_transform support to remaining DoD streaming functions, fixed visualization to use global coordinates, and integrated clipping with local transform.

### Key Changes

**Area Clipping Integration** (`clipping.py`, `run_workflow.py`):
- Added `transform_to_local()` method to AreaClipper class
- Uses shapely's `translate()` to shift polygon coordinates by offset
- Workflow now transforms clipper when local_transform is enabled
- Fixes issue where clipping returned 0 points with local coordinates

**Visualization Fix** (`run_workflow.py`):
- All visualization calls now revert points/grids to global UTM coordinates
- Users see correct geospatial coordinates matching maps and real-world locations
- Applied to: original point clouds, aligned point clouds, M3C2 core points, DoD grid
- Enabled M3C2 distance histogram (shown before 3D visualization)

**DoD Streaming Fixes** (`dod.py`):
- Added `local_transform` parameter to `compute_dod_streaming_files_tiled()`
- Added `local_transform` parameter to `compute_dod_streaming_files()`
- Both functions now transform bounds and pass transform to stream_points()

**GPU Dependency Update** (`pyproject.toml`):
- Changed from `cupy-cuda13x` to `cupy-cuda12x` to match CUDA 12.x toolkit

### Integration Status

All modules now fully support local coordinate transformation:
- ✅ Data Loading (loader, stream reader, batch loader)
- ✅ Detection Parallel (DoD, C2C, M3C2)
- ✅ Detection Sequential (DoD, C2C, M3C2)
- ✅ Tile Workers (all 3)
- ✅ Export (LAZ, GeoTIFF)
- ✅ Clipping
- ✅ Visualization (point clouds, DoD, M3C2)

### Files Changed
- `pyproject.toml`: GPU dep to cupy-cuda12x
- `clipping.py`: Added transform_to_local() method
- `dod.py`: Added local_transform to both streaming functions
- `run_workflow.py`: Clipping transform, all visualization global coords, M3C2 histogram

---

## 2025-12-12 - Cross-Platform GPU Support & Sequential Streaming Fixes

### Summary
Enabled GPU acceleration on Windows (CuPy-only mode) and fixed coordinate transform handling in sequential streaming paths for C2C and M3C2. Previously, GPU acceleration required cuML which is Linux-only (RAPIDS). Now Windows users can use CuPy for partial GPU acceleration.

### Key Changes

**Cross-Platform GPU Library Check** (`run_workflow.py`):
- Modified GPU library check to allow CuPy-only mode on Windows
- cuML is Linux-only (RAPIDS); CuPy works on both platforms
- Now shows GPU mode: **FULL** (CuPy + cuML on Linux) or **PARTIAL** (CuPy only on Windows)
- Platform-aware error messages for missing libraries
- Added `use_for_dod` and `use_for_alignment` status to GPU info log

**GPUConfig** (`config.py`):
- Added `use_for_dod: bool` field to control GPU acceleration for DoD grid accumulation
- Default: `true` (enabled)

**Sequential C2C Streaming Fix** (`c2c.py`, `run_workflow.py`):
- Added `local_transform` parameter to `compute_c2c_streaming_files_tiled()`
- Fixed coordinate mismatch: sequential path was reading files in global coordinates but tile bounds were in local coordinates
- Transforms global file bounds to local for tile grid generation
- Converts tile bounds back to global for file bbox filtering
- Passes `local_transform` to `stream_points()` for coordinate transformation

**Sequential M3C2 Streaming Fix** (`m3c2.py`, `run_workflow.py`):
- Added `local_transform` parameter to `compute_m3c2_streaming_files_tiled()`
- **Changed sequential M3C2 to use `compute_m3c2_streaming_pertile_parallel` with `n_workers=1`**
- This enables per-tile core selection for sequential mode, making it truly out-of-core
- Removed global core point selection via reservoir sampling for streaming mode
- Both parallel and sequential now use per-tile core selection, supporting 100% core points without loading all data
- Same fix as C2C: transforms bounds appropriately between coordinate spaces
- Core points (in local coords) now correctly match file data (transformed to local)

### Verification Results

| Method | Before Fix | After Fix |
|--------|-----------|-----------|
| C2C (GPU, sequential) | valid=6,172 (0.07%) | valid=9,061,786 (100%) |
| M3C2 (sequential) | valid=0 | Expected: all valid |

### Known Issues

**CuPy NVRTC DLL on Windows**:
- DoD GPU may fail with `nvrtc64_130_0.dll` missing
- This is a CuPy/CUDA installation issue, not a code bug
- Solution: Install CUDA Toolkit or set `gpu.use_for_dod: false`
- DoD falls back to CPU successfully

### Files Changed
- `scripts/run_workflow.py`: Cross-platform GPU check, local_transform parameters
- `src/terrain_change_detection/utils/config.py`: Added `use_for_dod` to GPUConfig
- `src/terrain_change_detection/detection/c2c.py`: Added local_transform handling
- `src/terrain_change_detection/detection/m3c2.py`: Added local_transform handling

---

## 2025-12-12 - M3C2/C2C Visualization Invalid Point Filtering

### Summary
Fixed visualization issue where points with NaN/invalid distances were rendered as dark brownish spots. The `visualize_m3c2_corepoints` function now filters out non-finite distance values before plotting.

### Changes
- Added `isfinite()` mask to filter out NaN/inf distances before visualization
- Applies to both M3C2 and C2C visualizations (C2C reuses the same function)
- Raises `ValueError` if no valid distances exist to visualize

---

## 2025-12-12 - Streaming LocalCoordinateTransform Fix

### Summary
Fixed critical bug where streaming DoD, C2C, and M3C2 methods returned zero valid results when `LocalCoordinateTransform` was enabled. The issue was a coordinate space mismatch between ICP alignment (running in local coords) and streaming workers (processing files in global coords).

### Root Cause
- ICP alignment operated in local coordinates (0-2000m range)
- Streaming workers processed file headers in global coordinates (280475m+ range)
- `bounds_intersect()` comparing local tile bounds with global file bounds always returned `False`
- Result: empty file lists for all tiles → no points processed → `n_cells=0`

### Key Changes

**Detection Modules** (`dod.py`, `c2c.py`, `m3c2.py`):
- Added `local_transform` parameter to parallel streaming functions
- Transform global bounds to local for tile grid generation
- Convert tile bounds back to global for `bounds_intersect()` file filtering
- Pass `local_transform` to worker kwargs

**Tile Workers** (`tile_workers.py`):
- Added `local_transform` parameter to `process_dod_tile`, `process_c2c_tile`, `process_m3c2_tile`
- Convert tile bounds back to global for bbox filtering
- Pass `local_transform` to `stream_points()` which applies `to_local()` to points

**Workflow** (`run_workflow.py`):
- Pass `local_transform` to all three streaming parallel functions

### Verification Results

| Method | Before Fix | After Fix |
|--------|-----------|-----------|
| DoD | n_cells=0, mean=nan | n_cells=481,146, mean=-0.017m |
| C2C | valid=0, RMSE=inf | valid=9,061,786, RMSE=0.354m |
| M3C2 | valid=0, RMSE=inf | 494,325 cells exported |

### Testing
- All 178 tests pass
- Verified against main branch results (values match within expected variance)

---

## 2025-12-11 - Local Coordinate Transformation Infrastructure

### Summary
Implemented local coordinate transformation infrastructure to handle large UTM coordinates (e.g., Easting ~500,000m, Northing ~6,000,000m) and prevent floating-point precision issues during numerical computations, especially on GPUs with float32 limitations.

### Key Changes

**New Coordinate Transform Utility** (`src/terrain_change_detection/utils/coordinate_transform.py`):
- `LocalCoordinateTransform` dataclass with offset storage
- Creation methods: `from_bounds()`, `from_centroid()`, `from_first_point()`
- Transform methods: `to_local()`, `to_global()`, `transform_bounds()`
- Serialization: `to_dict()`, `from_dict()` for persistence
- Exported via `__init__.py` for project-wide access

**Configuration Updates** (`config.py`):
- New `CoordinateConfig` class with:
  - `use_local_coordinates`: Enable/disable feature (default: True)
  - `origin_method`: "min_bounds" | "centroid" | "first_point" (default: min_bounds)
  - `include_z_offset`: Whether to offset Z (default: False)

**Data Loading Integration**:
- `PointCloudLoader.load(transform=...)`: Apply transform during loading, store in metadata
- `LaspyStreamReader.stream_points(transform=...)`: Apply transform to streamed chunks
- `BatchLoader.load_dataset(transform=...)`: Pass transform through to file loading

**Export Utilities Integration**:
- `export_points_to_laz(local_transform=...)`: Reverts to global coords before writing LAZ
- `export_distances_to_geotiff(local_transform=...)`: Reverts to global coords for raster
- `apply_transform_to_files(local_transform=...)`: Reverts in streaming alignment export

### New Files
- `src/terrain_change_detection/utils/coordinate_transform.py`: Core transform utility
- `tests/test_coordinate_transform.py`: 24 unit tests covering all functionality

### Testing
- 27 tests passing
- Round-trip precision verified (to_local → to_global preserves coordinates)
- Large UTM coordinate handling validated

### Usage Notes

The infrastructure is in place but not yet wired into the main workflow. To complete integration:
1. Compute transform from T1 bounds in `run_workflow.py`
2. Pass transform to all loading/streaming calls
3. Pass transform to export calls

### Migration Notes
- No breaking changes - all new parameters are optional
- Existing code continues to work without modification
- Feature is opt-in via configuration

---

## 2025-12-11 - Output File Export (LAZ Point Clouds & GeoTIFF Rasters)

### Summary
Implemented comprehensive output file export capabilities for terrain change detection results, enabling QGIS-compatible outputs. Point cloud results (M3C2 core points, C2C source points) can now be exported as LAZ files with distance values as extra dimensions. Raster outputs (DoD, interpolated M3C2/C2C distances) can be exported as GeoTIFF files with proper CRS metadata.

### Key Changes

**New Export Module** (`src/terrain_change_detection/utils/export.py`):
- `export_points_to_laz()`: Export points with distance as extra dimension, supports uncertainty/significant flags
- `export_dod_to_geotiff()`: Export DoD result directly to GeoTIFF
- `export_distances_to_geotiff()`: Interpolate point distances to raster grid using KDTree nearest-neighbor
- `detect_crs_from_laz()`: Auto-detect CRS from LAZ file WKT VLRs
- `_epsg_to_wkt()`: Convert EPSG codes to WKT using pyproj

**Configuration Updates** (`config.py`, all YAML files):
- `paths.output_dir`: Base directory for exports (defaults to `base_dir/output/`)
- `paths.output_crs`: Fallback CRS when auto-detection fails (default: EPSG:25833)
- `alignment.export_aligned_pc`: Export aligned T2 point cloud
- `detection.dod.export_raster`: Export DoD as GeoTIFF
- `detection.c2c.export_pc`, `detection.c2c.export_raster`: C2C exports
- `detection.m3c2.export_pc`, `detection.m3c2.export_raster`: M3C2 exports (enabled by default)

**Workflow Integration** (`scripts/run_workflow.py`):
- Export calls added after DoD, C2C, and M3C2 computations
- CRS auto-detected from input LAZ files with fallback to config
- Flat output structure: files saved as `{method}_{area}_{t1}_{t2}.{ext}`

**New Dependencies**:
- `rasterio>=1.3`: Required for GeoTIFF operations

### Output Files

All exports saved to `{base_dir}/output/`:

| File | Format | Content |
|------|--------|---------|
| `dod_{area}_{t1}_{t2}.tif` | GeoTIFF | DoD raster grid |
| `c2c_{area}_{t1}_{t2}.laz` | LAZ | Source points with `distance` dimension |
| `c2c_{area}_{t1}_{t2}.tif` | GeoTIFF | C2C interpolated to raster |
| `m3c2_{area}_{t1}_{t2}.laz` | LAZ | Core points with `distance`, `uncertainty`, `significant` |
| `m3c2_{area}_{t1}_{t2}.tif` | GeoTIFF | M3C2 interpolated to raster |

### New Files
- `src/terrain_change_detection/utils/export.py`: Core export module
- `tests/test_export.py`: 9 test cases for export functionality

### Usage Example

```yaml
# Enable exports in config
paths:
  output_crs: "EPSG:25833"
detection:
  m3c2:
    export_pc: true    # LAZ with distances
    export_raster: true  # GeoTIFF
```

Run workflow normally; outputs appear in `data/raw/output/`.

---

## 2025-12-11 - M3C2 Core Points Percentage

### Summary
Added ability to specify M3C2 core points as a percentage of reference ground points instead of an absolute number. This provides more intuitive and dataset-adaptive configuration, especially when processing datasets of varying sizes.

### Key Changes

**Configuration** (`config.py`, all YAML files):
- Added `core_points_percent` field to `DetectionM3C2Config` (default: 10.0)
- Original `core_points` field remains for backward compatibility
- Percentage takes precedence; absolute count used only if percentage is null/not set

**Workflow Logic** (`scripts/run_workflow.py`):
- In-memory mode: calculates from loaded array length
- Streaming mode: uses LAS header point counts for efficiency (no data loading)
- Logs the calculated core point count for transparency

### Configuration Example

```yaml
detection:
  m3c2:
    # Use 10% of reference ground points as M3C2 core points
    core_points_percent: 10.0
    # Or specify absolute count (takes precedence if set):
    # core_points: 50000
```

### Migration Notes

- Default changed from `core_points: 50000` to `core_points_percent: 10.0`
- All YAML configs updated to use percentage-based configuration
- Backward compatible: existing configs with `core_points` still work

---

## 2025-12-03 - Area Clipping Feature

### Summary
Implemented area clipping to focus terrain change analysis on specific regions of interest. Point clouds can be clipped to polygon boundaries (GeoJSON or Shapefile) before ICP registration.

### Key Changes

**New Clipping Module** (`preprocessing/clipping.py`):
- `AreaClipper` class with `from_file()`, `from_polygon()`, `from_bounds()` constructors
- Two-stage clipping: fast bounding box pre-filter, then vectorized point-in-polygon via shapely
- Support for `feature_name` to select specific polygons from multi-feature files

**Streaming Integration**:
- `clip_bounds` parameter added to DoD and C2C parallel streaming methods
- Tiles outside clip region are skipped

**Configuration** (`config.py`):
- New `ClippingConfig`: `enabled`, `boundary_file`, `feature_name`, `save_clipped_files`

**Workflow** (`run_workflow.py`):
- Clipping applied after data loading, before ICP alignment

**New Dependencies**: `shapely>=2.0`, `fiona` (optional, for Shapefiles)

---

## 2025-11-19 - Module Refactoring, Logging Improvements, and Parallel Execution Fixes

### Summary
Split monolithic `change_detection.py` (2300+ lines) into separate modules (`dod.py`, `c2c.py`, `m3c2.py`). Improved logging by moving detection and alignment logs to their respective modules. Fixed critical bugs in DoD streaming and parallel execution paths.

### Key Changes

**Module Refactoring** (`detection/`):
- Split into `dod.py`, `c2c.py`, `m3c2.py` with `__init__.py` facade for backward compatibility
- Detection and alignment completion logs moved to their respective modules
- Added per-tile progress logging for all three methods

**DoD Streaming Bug Fixes**:
- Fixed `Tile` nx/ny calculation (changed to `ceil(...) + 1` to match `GridAccumulator`)
- Fixed sequential path `_make_tile()` to create proper `Tile` objects
- Fixed `mosaic.add_tile()` calls to pass `Tile` objects

**Parallel Execution Fixes**:
- Fixed bounds unpacking in DoD, C2C, and M3C2 parallel functions
- Fixed M3C2 tile hashability (replaced `Tile` dict keys with `(i, j)` tuples)
- Fixed DoD parallel mosaic tile object passing

---

## 2025-11-19 - Drone Scanning Data Support

### Summary
Added support for drone scanning point cloud data as an alternative to the hierarchical hoydedata.no structure. The `DataDiscovery` class now accepts a `source_type` parameter ('hoydedata' or 'drone') to handle both directory structures.

### Key Changes

**Data Discovery** (`data_discovery.py`):
- Hoydedata structure: `area/time_period/data/*.laz`
- Drone structure: `area/time_period/*.laz` (no 'data' subdirectory)
- Source-type aware error messages and configuration warnings

**Configuration**:
- Added `source_type` field to `DiscoveryConfig` (default: 'hoydedata')
- Created `config/profiles/drone.yaml` profile (higher resolution, no out-of-core)

**Documentation**: `docs/DRONE_DATA_SUPPORT.md` with setup guide and troubleshooting

---

## 2025-11-17 - DoD GPU Acceleration

### Summary
Implemented and validated GPU acceleration for DoD grid accumulation. Fixed critical config propagation bug where `config=cfg` was missing from DoD method calls. Discovered GPU+parallel incompatibility (CUDA fork limitation) and cuML ICP reliability issues — both have automatic CPU fallback.

### Key Changes

**GPU GridAccumulator** (`tiling.py`):
- GPU accumulation using CuPy `unique()` and `add.at()` for efficient binning
- Automatic GPU/CPU fallback with error handling
- 1.5-4x speedup on large datasets (100K-1M points)

**DoD Methods** (`change_detection.py`):
- Added `config` parameter to all 4 DoD methods for GPU control
- Added detailed GPU/CPU backend logging

**JIT Kernels** (`jit_kernels.py`):
- Numba-accelerated transform and distance computation with NumPy fallback

**Bug Fix** (`run_workflow.py`):
- Fixed missing `config=cfg` in DoD calls — GPU was never actually enabled

### Performance

- DoD GPU: ~1.03x speedup (memory-bound, marginal benefit)
- C2C GPU: 10-100x speedup (compute-bound, significant benefit)
- GPU+parallel incompatible (CUDA contexts corrupt in forked processes) — auto-fallback to CPU

---

## 2025-11-16 - ICP Alignment Testing & Logging Improvements

### Summary
Tested ICP alignment across all execution modes (in-memory, streaming, GPU, CPU) and coarse registration methods. Improved logging to distinguish alignment RMSE (on subsampled ICP points) from validation RMSE (on larger sample).

### Key Changes

**ICP Logging** (`fine_registration.py`, `run_workflow.py`):
- Logs now show point counts: `"Final RMSE on 50000 alignment points: X.XXX"`
- Validation log distinguishes subset from full dataset size
- Eliminates confusion from previously conflicting RMSE messages

**Configuration** (`default.yaml`):
- Added explicit `convergence_translation_epsilon` and `convergence_rotation_epsilon_deg` fields

### Testing Results

| Mode | Coarse Method | Iterations | Alignment RMSE | Validation RMSE |
|------|---------------|------------|----------------|-----------------|
| In-memory + GPU | Phase | 48-53 | 0.618-0.649 | 0.687-0.688 |
| In-memory + CPU | Phase | 19 | 0.705 | 0.687 |
| In-memory + CPU | Centroid | 32 | 0.751 | 0.755 |
| Streaming + GPU | Phase/Centroid | 100 | 0.717 | 0.746 |

All modes produce consistent alignments. CPU outperforms GPU for small subsamples (50k points).

---

## 2025-11-16 - ICP Alignment Instrumentation & Benchmarking

### Summary
Improved observability and robustness of ICP-based spatial alignment, added optional GPU-backed nearest-neighbor search with safe CPU fallback, and introduced a small real-data benchmark script for ICP alignment.

### Key Changes

**ICP Registration Enhancements**:
- Added detailed timing logs, motion-based convergence criteria (translation and rotation), and better handling of empty inputs/errors.
- Fixed KD-tree reuse and integrated optional GPU-accelerated nearest neighbors (`GPUNearestNeighbors`) with automatic CPU fallback.

**Workflow & Configuration**:
- Alignment now operates on reservoir-sampled subsets in streaming mode, with a capped random subset for the final RMSE check to avoid excessive overhead.
- Multi-scale ICP refinement supported but disabled by default.
- Added `gpu.use_for_alignment` to configuration (defaults to false).

**Tooling & Tests**:
- Added test and benchmark tools for ICP registration and alignment performance.

### Testing

- `uv run pytest -q tests/test_icp_registration.py -q` to validate ICP behavior.
- `uv run scripts/test_icp_alignment_performance.py` on the Norwegian dataset to compare CPU vs (attempted) GPU ICP; current runs show safe CPU fallback when cuML produces invalid distances.

## 2025-11-16 - GPU C2C Robustness & cuML Large-Scale Debugging

### Summary
Hardened GPU C2C pipeline against corrupted distances from cuML at large scales. Added `gpu_backend` metadata field and per-tile backend logging for observability.

### Key Changes

**Distance Sanity Checks** (`change_detection.py`):
- After GPU k-NN, checks for finite distances and `max(distance) <= 1e5 m`
- On failure, transparently recomputes on CPU
- `gpu_backend` metadata distinguishes cuML, sklearn-gpu, and CPU paths

**Streaming C2C**:
- Tracks `gpu_tiles` vs `tiles_with_src` to report actual GPU usage per tile
- Per-tile logging shows effective backend (e.g., `GPU[cuml]`, `CPU`)

**GPU Neighbor Wrapper**:
- cuML backend keeps optional CPU copy for radius-based query fallback
- Added `scripts/debug_cuml_large_c2c_issue.py` for isolating cuML scaling issues
- Added `docs/WSL2_GPU_SETUP_SUMMARY.md` and `activate_gpu.sh`


## 2025-11-15 - GPU Performance Testing & Platform Analysis

### Summary
Comprehensive GPU vs CPU testing on real data (1K–9M points) revealed that GPU C2C acceleration is Linux-only. On Windows, the `sklearn-gpu` backend wraps data in CuPy arrays but still uses CPU KDTree — no actual GPU compute occurs (0.99x average speedup).

### Key Findings

| Platform | Backend | k-NN Compute | Expected Speedup |
|----------|---------|--------------|------------------|
| Linux | cuML | GPU (CUDA) | 10-50x |
| Windows | sklearn-gpu | CPU | ~1x (no benefit) |

GPU infrastructure is correct and complete; the limitation is in the underlying libraries. Documented in `docs/GPU_PERFORMANCE_ANALYSIS.md`.

---

## 2025-11-15 - C2C GPU Acceleration & Workflow Integration

### Summary
Integrated GPU-accelerated nearest neighbor searches into all C2C variants (basic, vertical plane, streaming/tiled, parallel). Added GPU status logging at workflow startup and post-computation usage reporting.

### Key Changes

**C2C GPU Integration** (`change_detection.py`, `tile_workers.py`):
- All C2C methods accept optional `config` parameter for GPU control
- GPU usage respects hierarchy: `gpu.enabled` → `gpu.use_for_c2c` → hardware available
- Automatic CPU fallback; `gpu_used` metadata flag for observability
- GPU/CPU numerical parity verified (basic C2C: rtol=1e-5, vertical plane: rtol=1e-4)

**GPU Neighbors** (`gpu_neighbors.py`):
- `create_gpu_neighbors()` supports both k-NN and radius-based queries
- Backend selection: cuML (Linux) > sklearn-gpu (Windows) > sklearn-cpu

**Workflow** (`run_workflow.py`):
- GPU hardware detection and status logging at startup
- All C2C calls pass `config` for GPU propagation

**Testing**: 16 C2C integration tests + 34 infrastructure tests = 50 GPU tests passing

---

## 2025-11-15 - GPU Acceleration Infrastructure

### Summary
Built foundational GPU acceleration infrastructure: hardware detection, NumPy/CuPy array abstraction, GPU nearest neighbors wrapper, and configuration system. Analysis showed C2C is the primary GPU target (10-60x speedup potential); M3C2 via py4dgeo cannot be directly GPU-accelerated.

### Key Changes

**Hardware Detection** (`hardware_detection.py`):
- `detect_gpu()`, `get_gpu_info()`, `check_gpu_memory()`, `get_optimal_batch_size()`

**Array Operations** (`gpu_array_ops.py`):
- `ArrayBackend` providing unified NumPy/CuPy interface

**GPU Nearest Neighbors** (`gpu_neighbors.py`):
- `GPUNearestNeighbors` with cuML > sklearn-gpu > sklearn-cpu fallback chain

**Configuration** (`config.py`, `default.yaml`):
- `gpu.enabled`, `gpu.use_for_c2c`, `gpu.use_for_preprocessing`, `gpu.fallback_to_cpu`

**Dependencies** (`pyproject.toml`):
- Optional `[gpu]` group: `cupy-cuda13x`, `numba`, `cuml-cu12`

**Documentation**: `GPU_SETUP_GUIDE.md`, `GPU_INTEGRATION_STRATEGY.md`

---

## 2025-11-13 - Alignment, Tiling, and M3C2 Autotune Consistency

### Summary
This release resolves cross-mode differences between in-memory and streaming/tiled analyses by fixing a DoD mosaicking bug, aligning filtering/transform behavior across paths, and making M3C2 parameter selection consistent and reproducible. A new header-based autotune option and a fixed-parameter mode make results mode-agnostic when desired.

### Key Changes

**DoD Mosaicking**:
- Fixed masked writes on a view that could drop values when assembling tiles.

**Parallel Workers**:
- Pass `ground_only` through all worker paths (DoD, C2C, M3C2).
- Apply epoch-2 transform per chunk where applicable to keep parity with in-memory mode.

**M3C2 Consistency**:
- Added header-based autotune for density derived from LAS headers and union extent (mode-agnostic).
- Added fixed-parameter mode via YAML configuration.
- Exposed CLI overrides for experiments (`--m3c2-radius`, etc.).
- Added reproducibility and diagnostics (`--seed`, `--cores-file`, `--debug-m3c2-compare`).

**Configuration**:
- Added autotune options and updated default YAML profiles.

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
Improved consistency between in-memory and streaming M3C2 reporting and fixed confusing NaN statistics when a subset of core points has undefined distances. This change does not alter the underlying M3C2 distance calculations; it makes summary statistics robust and consistent across execution modes and ensures the parallel streaming path respects classification filtering like other paths.

### Key Changes

**Robust M3C2 Stats**:
- M3C2 logs now compute mean, median, and std with NaN-aware reducers and include a count of valid distances.
- Enriched M3C2 results metadata with robust statistics.

**Streaming Consistency**:
- Wired `ground_only` through the parallel tiled M3C2 worker path and reader, aligning class filtering behavior with in-memory and sequential streaming.

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
Significant improvements to runtime logging and user feedback during long operations. Console logs are cleaner and more focused; parallel and sequential tile processing now display a Rich progress bar with elapsed time and ETA. Noisy third-party prints (KDTree builds) are captured and demoted to DEBUG, including Windows-safe handling.

### Key Changes

**Cleaner Logging**:
- Simplified console logs (removed process name/PID) while keeping detail in file logs.
- Reduced noise for INFO logs (demoted M3C2 start/finish messages, shortened file lists).
- Captured and demoted third-party stdout/stderr (e.g., KDTree builds are only logged at DEBUG).

**Progress Feedback**:
- Added `rich` progress bars for sequential and parallel tile processing (DoD, C2C, M3C2).
- Graceful fallback for environments missing the `rich` library.

### Expected Impact

- Cleaner console logs by default; detailed process information still available in file logs.
- Better user feedback during long, parallel operations with minimal log spam.
- KDTree and similar library prints no longer clutter INFO output; visible only at DEBUG.

### Notes

- If you want to surface the KDTree messages, set `logging.level: DEBUG` i## 2025-11-10 - CPU Parallelization Refinements and I/O Pruning

### Summary
Focused improvements to the CPU parallel path: reduced redundant I/O per tile, prevented BLAS/NumPy thread oversubscription in workers, improved scheduling, and corrected misleading speedup logs.

### Key Changes

**Parallel Executor Optimizations**:
- Clamped effective workers to avoid overspawn and pinned per-process threads to prevent oversubscription.
- Improved chunk size heuristics for reduced IPC overhead.
- Adjusted logging to show actual throughput and realistic expected speedups.

**I/O Pruning**:
- Implemented file bounds caching to avoid rescanning unchanged files.
- Parallel worker methods now only receive intersecting files for the specific tile they process, reducing decompression overhead.

**Configuration**:
- Added `parallel.threads_per_worker` parameter for precise control over worker performance.

### Notes

- Pool reuse and fail-fast threshold were prototyped but reverted to maintain stability across platforms and to keep behavior simple (per-call pools, aggregate error reporting).
- The per-tile file pruning provides the biggest win on multi-file datasets by avoiding O(#tiles x #files) rescans.

### Expected Impact

- Better scaling on medium/large datasets due to reduced redundant I/O and no thread oversubscription.
- More accurate logs for expected speedup; easier to reason about throughput.

## 2025-11-09 - Large Synthetic Dataset for Performance Testing

### Summary
Created large-scale synthetic dataset generation for testing CPU parallelization at scale (51.2M points across 16 tiles).

### Key Changes
- `scripts/generate_large_synthetic_laz.py`: 25.6M points per epoch, 4x4 km, 6 controlled terrain change features, no misalignment
- `config/profiles/large_synthetic.yaml`: Out-of-core and parallel processing enabled, alignment disabled

---

## 2025-11-09 - CPU Parallelization Implementation

### Summary
Implemented tile-level CPU parallelization for all three change detection methods. Benchmarking showed limited benefit at current scale (~9M points) but provides infrastructure for larger datasets.

### Key Changes

**Parallelization Infrastructure** (`acceleration/`):
- `TileParallelExecutor` class for worker pool management and tile distribution
- Module-level picklable workers: `process_dod_tile`, `process_c2c_tile`, `process_m3c2_tile`
- Parallel variants: `compute_{dod,c2c,m3c2}_streaming_files_tiled_parallel()`
- `parallel.enabled` config flag with automatic fallback to sequential

**Bug Fixes**:
- Fixed M3C2Result field names (`uncertainties` → `uncertainty`)
- Fixed C2CResult to use individual fields instead of stats dict
- Corrected streaming API `bbox` parameter usage

### Performance (9M points, 12 tiles)

| Method | Sequential | Parallel | Speedup |
|--------|-----------|----------|---------|
| DoD | 77.67s | 96.37s | 0.81x |
| C2C | 149.09s | 107.71s | 1.38x |
| M3C2 | 117.43s | 125.97s | 0.93x |

Overhead exceeds benefit at this scale. Benefits expected at 50M+ points with more tiles.

---

## 2025-11-09 - Performance Optimization Strategy

### Summary
Established two-phase optimization approach: CPU parallelization first (using existing tiling infrastructure), then GPU acceleration. Archived outdated planning documents and created new phase-specific plans.

### Key Changes
- Merged `feat/outofcore-tiling` into `feat/gpu-acceleration` (19 commits)
- Archived 6 outdated planning documents to `docs/archive/`
- Created `PARALLELIZATION_PLAN.md` (Phase 1) and `GPU_ACCELERATION_PLAN.md` (Phase 2)

---

## 2025-11-09 - Config Schema Completion

### Summary
Finalized out-of-core configuration schema with missing fields for production readiness.

### Changes
- Changed `save_transformed_files` default from `true` to `false` (opt-in file writing)
- Added `memmap_dir` field to `OutOfCoreConfig` for memory-mapped mosaicking arrays
- Updated all config profiles with complete outofcore section

---

## 2025-11-06 - Out-of-Core Processing & Tiling

### Summary
Implemented complete out-of-core processing infrastructure with tiling for all three change detection methods (DoD, C2C, M3C2). Enables processing of datasets that exceed available memory by dividing spatial domains into tiles and streaming point data in chunks.

### Key Changes

**Tiling System** (`acceleration/tiling.py`):
- `Tiler`, `Tile`, `Bounds2D` for grid-aligned tiling with halo support
- `LaspyStreamReader` for chunked LAZ/LAS reading with bbox and classification filtering
- `GridAccumulator` for streaming mean aggregation over regular XY grids
- `MosaicAccumulator` for stitching tile DEMs with overlap averaging

**Change Detection Methods**:
- `compute_dod_streaming_files_tiled()`: Single-pass chunk routing with grid-aligned mosaicking
- `compute_c2c_streaming_files_tiled()`: Tiled C2C with radius-bounded NN queries + `compute_c2c_vertical_plane()`
- `compute_m3c2_streaming_files_tiled()`: Tiled M3C2 with spatial core point partitioning

**Configuration**:
- New `outofcore` config section: `enabled`, `tile_size_m`, `halo_m`, `chunk_points`, `streaming_mode`, `save_transformed_files`, `memmap_dir`
- Workflow auto-routes between in-memory and streaming paths based on config
- Reservoir sampling for memory-safe alignment subsampling

**Supporting Modules**:
- `streaming_alignment.py`: `apply_transform_to_files()` for chunk-based transformations
- `utils/point_cloud_filters.py`: Reusable classification filtering
- Header-only metadata discovery without loading full arrays

**Documentation**: `docs/ALGORITHMS.md` covering tiling applicability and halo sizing

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
