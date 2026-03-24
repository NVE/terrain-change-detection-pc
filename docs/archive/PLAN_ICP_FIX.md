# ICP Alignment Fixes — Work Plan

## Context

After delivering the terrain change detection pipeline, the client raised 6 concerns about the ICP spatial alignment stage. This document describes each issue, the root cause analysis, and the implemented solution.

---

## Issue 3: Non-Deterministic ICP Results (CRITICAL)

**Client Concern:** Repeated runs of ICP produce different alignment results each time and may lead to progressive misalignment.

**Root Cause:** Subsampling before ICP used `np.random.choice()` without a seed. Each run picked different random points, leading to different ICP convergence. The ICP algorithm itself (SVD + KD-tree) is deterministic given identical inputs — the non-determinism was entirely in the upstream subsampling.

**Affected code paths:**
- `run_workflow.py` — multi-scale and fine ICP subsampling, validation error subsampling
- `tiling.py` — reservoir sampling for streaming mode

**Solution:**
- Added `random_seed: 42` config parameter (default = deterministic out of the box)
- Created a shared `np.random.default_rng(seed)` at workflow startup
- Replaced all `np.random.choice()` and `np.random.randint()` calls with `rng.choice()` and `rng.integers()`
- The `--seed` CLI arg still works as an override
- Reservoir sampling in `tiling.py` now accepts an explicit `seed` parameter

**Key files:** `config.py`, `default.yaml`, `run_workflow.py`, `tiling.py`

---

## Issue 4: Subsample Size as Percentage

**Client Concern:** Subsample size is specified as an absolute number of points. Can it be specified as a percentage?

**Risk:** Large percentages on large datasets can cause OOM or extreme slowness.

**Solution:**
- Added `subsample_mode` config: `"count"` (default, backward-compatible) or `"percent"`
- Added `subsample_percent` (default 10%) and `max_subsample_size` (default 500,000 — safety cap)
- Helper function `resolve_subsample_count()` computes the effective count and enforces the cap
- When the cap kicks in, a log message explains why fewer points were used

**Key files:** `config.py`, `default.yaml`, `run_workflow.py`

---

## Issue 5: Overlap-Aware Subsampling for Different-Sized Point Clouds

**Client Concern:** When two point clouds have different spatial extents, how does random subsampling ensure points are selected from the overlap area?

**Root Cause:** Previously, subsampling was uniform across the entire cloud. Points outside the overlap region have no correspondences in the other cloud and waste computation during ICP.

**Solution:**
- Added `compute_overlap_mask()` function that computes the XY bounding-box intersection between two clouds
- Configurable via `overlap_filter: true` (default) and `overlap_margin_m: 5.0`
- Before subsampling, both clouds are filtered to the overlap region
- Falls back to full clouds if overlap has fewer than 100 points
- For streaming mode: uses `intersection_bounds()` from file headers to pass a bbox to reservoir sampling

**Key files:** `fine_registration.py`, `tiling.py`, `run_workflow.py`, `config.py`

---

## Issue 2: Reference vs Target Point Cloud Selection

**Client Concern:** There is no option to choose which point cloud is the ICP reference (fixed) and which is the source (aligned).

**Previous behavior:** T1 (chronologically first) was always the reference. T2 was always transformed.

**Solution:**
- Added `reference: "t1" | "t2"` config parameter (default `"t1"` — backward-compatible)
- Added `--reference` CLI argument to override config
- When `reference: "t2"`, T1 is aligned to T2 instead
- All downstream operations (export, visualization, change detection) correctly use the aligned result
- Streaming file transformation also respects the reference direction

**Key files:** `config.py`, `default.yaml`, `run_workflow.py`

---

## Issue 1: Export Aligned Point Cloud

**Client Concern:** ICP seems to overwrite one of the point clouds instead of creating a new aligned point cloud by name.

**Root Cause:** The `export_aligned_pc` config option existed but was never wired up in the workflow code. The aligned cloud existed only in memory; no distinct output file was created.

**Solution:**
- Implemented the `export_aligned_pc` feature: when enabled, writes `aligned_{time_period}.laz` to the output directory
- Made the `distances` parameter optional in `export_points_to_laz()` since aligned clouds don't have distance values
- Clear naming convention: the file is always named `aligned_` + the time period that was transformed
- Respects the `reference` direction (exports whichever cloud was aligned)

**Key files:** `export.py`, `run_workflow.py`

---

## Issue 6: Open3D ICP Backend Option

**Client Concern:** The custom ICP implementation should be validated against well-known libraries, with an option to use them instead.

**Our Position:** The custom SVD-based ICP gives fine-grained control for GPU/performance optimizations. However, providing an alternative backend builds confidence and allows cross-validation.

**Solution:**
- Added `icp_backend: "custom" | "open3d"` config parameter (default `"custom"`)
- Created `Open3DICP` wrapper class in `alignment/open3d_icp.py` with the same interface as `ICPRegistration`
- Uses Open3D's `registration_icp()` with point-to-point estimation (matching our custom implementation)
- Lazy import — Open3D is only loaded when the backend is selected
- Falls back to custom backend with a warning if Open3D is not installed
- Tests verify both backends produce similar results on synthetic data

**Key files:** new `open3d_icp.py`, `alignment/__init__.py`, `config.py`, `run_workflow.py`

---

## New Configuration Parameters

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `random_seed` | `int` | `42` | Reproducible subsampling |
| `subsample_mode` | `"count" \| "percent"` | `"count"` | How subsample size is specified |
| `subsample_percent` | `float` | `10.0` | Percentage when mode is `"percent"` |
| `max_subsample_size` | `int` | `500000` | Safety cap for either mode |
| `overlap_filter` | `bool` | `true` | Filter to overlap region before subsampling |
| `overlap_margin_m` | `float` | `5.0` | Margin around overlap box (meters) |
| `reference` | `"t1" \| "t2"` | `"t1"` | Which epoch is the ICP reference |
| `icp_backend` | `"custom" \| "open3d"` | `"custom"` | ICP implementation to use |

All parameters have defaults that preserve existing behavior (except `overlap_filter` which defaults to enabled as an improvement).

---

## Testing

The test suite was expanded from 2 to 15 tests covering:

- **Determinism:** Same seed → identical transforms; different seeds → different results
- **Percentage subsampling:** Count mode, percent mode, safety cap enforcement
- **Overlap filtering:** Full overlap, partial overlap, no overlap, margin expansion
- **Reference swap:** Forward and reverse transforms are approximate inverses
- **Open3D backend:** Recovery of known transforms, parity with custom backend (skipped if Open3D not installed)

Run tests: `.venv/bin/python -m pytest tests/test_icp_registration.py -v`

---

## Files Modified

| File | Changes |
|------|---------|
| `src/terrain_change_detection/utils/config.py` | 8 new fields in `AlignmentICPConfig` |
| `config/default.yaml` | New alignment parameters |
| `config/default_clipped.yaml` | Added `random_seed` |
| `config/profiles/*.yaml` (5 files) | Added `random_seed` |
| `scripts/run_workflow.py` | Seeding, overlap filter, subsample helper, reference swap, export wiring, backend selection, `--reference` CLI arg |
| `src/terrain_change_detection/alignment/fine_registration.py` | Added `compute_overlap_mask()` |
| `src/terrain_change_detection/alignment/open3d_icp.py` | **New:** Open3D ICP wrapper |
| `src/terrain_change_detection/alignment/__init__.py` | Exports for new functions/classes |
| `src/terrain_change_detection/acceleration/tiling.py` | `seed` param for `reservoir_sample()`, new `intersection_bounds()` |
| `src/terrain_change_detection/utils/export.py` | Made `distances` optional |
| `tests/test_icp_registration.py` | 13 new tests (15 total) |
