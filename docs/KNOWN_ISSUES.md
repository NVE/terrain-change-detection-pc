# Known Issues

This document tracks known limitations and issues in the terrain change detection workflow.

## Area Clipping

### Bounding Box vs Precise Polygon Filtering for DoD/C2C

**Issue**: When clipping is enabled with multi-polygon boundaries, DoD and C2C streaming use bounding box filtering while M3C2 uses precise core-point filtering, resulting in different tile counts.

**Example**: With `clip_areas.geojson` containing 4 polygons:
- DoD/C2C: 49/64 tiles (bounding box filter)
- M3C2: 24/64 tiles (core points filter)

**Explanation**: 
- DoD and C2C filter tiles by checking if they overlap the clip region's **bounding box** (fast but loose)
- M3C2 filters by checking which tiles contain **actual core points** (precise)
- The bounding box of scattered polygons is much larger than the polygons themselves

**Workaround**: Use single-polygon clip files (e.g., `clip_central_area.geojson`) for tighter bounding box filtering.

**Potential Fix**: Implement precise polygon-tile intersection testing, but this adds computational overhead.

---

## Performance

### Slow Subsampled Data Loading with Out-of-Core + Parallel

**Issue**: When `outofcore.enabled=true` and `parallel.enabled=true`, loading subsampled data for alignment takes significantly longer than expected (~60+ seconds for large datasets).

**Observed in**: Step 1 "Loading subsampled data for alignment"

**Root Cause**: TBD - likely related to streaming sample logic or file I/O patterns.

**Workaround**: None currently. Consider disabling streaming for smaller datasets where full in-memory loading is feasible.

**Status**: Not investigated

---

## Alignment

### ICP Can Introduce Systematic Offsets Between Airborne and Drone Epochs

**Issue**: On the Ristvassdrag dataset (hoydedata.no airborne LiDAR 2017 vs. drone LiDAR 2025), enabling ICP fine registration produced a visible, spatially systematic displacement pattern in the M3C2 output that was not present when alignment was disabled. The two epochs were already delivered in the same projected CRS (EPSG:25832), so the "correction" applied by ICP degraded rather than improved the result.

**Observed in**: Ristvassdrag runs, July 2026 (`alignment.enabled=true`, `reference=t2`, coarse `phase` + ICP, reported RMSE ~1.37 m on the ICP subsample).

**Suspected Causes**:
- Strongly unequal spatial extents and point densities between epochs (a full ALS tile set vs. a narrow drone corridor), so the nearest-neighbour correspondences used by point-to-point ICP are biased by non-overlapping terrain and vegetation-edge effects.
- Real terrain change inside the corridor is a large fraction of the overlap, which violates the ICP assumption that most correspondences are on unchanged surfaces.
- ICP RMSE is not a reliable acceptance criterion here (see the RMSE note in the Best Practice Guide).

**Workaround**: For epochs that are already georeferenced in the same CRS, run with `alignment.enabled=false` and verify alignment by checking that M3C2 distances on known stable surfaces (roads, bedrock) are centred on zero. Only enable ICP if that check fails, and then restrict it to stable reference areas (e.g., via clipping) rather than the full overlap.

**Status**: Under investigation. A stable-area-constrained registration option would address this properly.

---

## Change Detection Algorithms

### C2C Implementation Accuracy

**Issue**: The Cloud-to-Cloud (C2C) distance implementation may not be fully accurate or reliable for all use cases.

**Symptoms**: 
- Distance values may differ from reference implementations (e.g., CloudCompare)
- Edge effects at tile boundaries in tiled/streaming mode

**Status**: Needs investigation and validation against reference implementations.

**Recommendation**: Use M3C2 for production workflows requiring high accuracy. C2C is suitable for quick exploratory analysis.

### C2C Visualization Fallback to Histogram

**Issue**: When `outofcore.enabled=true` and `parallel.enabled=true`, C2C distance visualization falls back to histogram instead of 3D point cloud visualization.

**Reason**: Streaming C2C returns aggregated results without full point coordinates needed for 3D visualization.

**Workaround**: 
- Use in-memory mode for full 3D visualization
- Or use the histogram visualization which still shows distance distribution

**Status**: By design for memory efficiency in streaming mode.

---

## Erosion Polygon Export

### Volume Underestimates on Slopes (normal distance × horizontal cell area)

**Issue**: `export_erosion_polygons_geojson()` computes `volume_loss_m3 = sum(|erosion_values|) * pixel_area`, where the raster values are M3C2 distances measured along the local surface normal and `pixel_area` is the horizontal cell area. A slab of normal thickness *d* on a slope of angle θ occupies a surface area of `pixel_area / cos θ` per cell, so the exported volume equals the true volume × cos θ: 13 % low at 30°, 29 % low at 45°, 50 % low at 60°. Flat areas are correct.

**Observed in**: Ristvassdrag runs 20260819_154059 (2017→2025) and 20260820_091944 (2025→2026), verified September 2026 against exact vertical DTM differencing (2017 DTM − 2025 drone DTM) on the same cells for 232 polygons ≥ 10 m²: exported 9 184 m³ vs 9 769 m³ exact on the shared cells (+6.4 %). Per-polygon ratio followed 1/cos(mean slope) closely (slope class 15–25°: 1.08 vs 1.07 expected; 25–35°: 1.18 vs 1.14). The site total for all 654 polygons moved only +4.7 % (12 518 → 13 108 m³ after correction) because the largest volumes lie in a flat excavation pit; individual bank polygons were 5–15 % low, some up to 35 %.

**Workaround**: Post-process the exported raster: `V = Σ |d| · pixel_area / cos θ` per cell with θ from a DTM (unsmoothed 0.5 m drone DTM worked best; NaN-aware, θ capped at 70°). The DTM slope is a proxy: py4dgeo estimates the normals from the first epoch's points around each core point, which is not necessarily the core-point epoch. This reproduced the exact DTM-difference volumes within 1 % in total. Ranking of the largest polygons is barely affected (9 of the top 10 unchanged).

**Potential Fix**: Export the z-component of the M3C2 normal (`n_z`) as a second raster band or LAZ extra dimension and compute `volume = Σ |d| · pixel_area / max(|n_z|, 0.2)` (equivalently project *d* to vertical before rasterising). The normals are already computed by py4dgeo for every core point (`M3C2.directions()`), so no extra computation is needed. This is an approximation that is exact only where the two surfaces are locally parallel; where the slope changes between epochs the projected value can be off by a factor of order 2 for individual cells (a synthetic check reproduced 0.20 m for parallel slopes but gave 0.40 m for a 0.20 m vertical difference across a slope change). Vertical DTM differencing remains the exact volume integral and should be offered as an independent estimate. Add `volume_loss_flat_m3` (current behaviour) alongside the corrected value for backward comparison.

**Status**: Not fixed in 0.2.0. Corrected values for the Ristvassdrag article were produced by post-processing.

### Implausible Depths at the Edge of Coverage and Across the Channel

**Issue**: With a large `max_depth` (10–12 m from autotune, `max_depth_factor=6`) the search cylinder can reach the opposite bank, or ground outside the drone coverage, and return single cells with 8–11 m of apparent lowering. Small polygons built from such cells can dominate volume rankings: in run 20260820_091944 a 7.75 m² polygon at the coverage edge (label 142, p95 lowering 11.5 m) carried 38.8 m³: 4.8 % of all exported volume (808 m³) and 7 % of the ≥ 5 m² subset (558 m³). It is not caught by a level-of-detection test.

**Workaround**: Screen polygons with an implausible 95th-percentile lowering (> 3 m removed 19 polygons < 10 m² in 2017→2025 and 7 in 2025→2026, one of them ≥ 5 m²) and rank on `p95_erosion_m` rather than `max_erosion_m`. Treat polygons touching the coverage boundary as truncated. A blanket "within 5 m of the coverage edge" rule is too blunt: on a narrow drone corridor it removed 21 real polygons ≥ 10 m², including the two largest.

**Potential Fix**: (1) The default `autotune.max_depth_factor` is already 1.0 (0.6 in the drone profile); both cited runs overrode it to 6. Rather than changing the default, warn when the resulting depth exceeds a few times the radius or the local relief, and record the override in the run log; (2) add per-polygon quality attributes: distance to the coverage boundary, number of core points, mean level of detection, share of cells whose cylinder had neighbours in both epochs; (3) optional plausibility filter (`max_p95_depth_m`) in the polygon export.

**Status**: Open.

---

## Change Detection Algorithms (M3C2 uncertainty)

### Registration Error Fixed at Zero — Exported LoD Excludes Alignment Error

**Issue**: `M3C2Detector` constructs py4dgeo's `M3C2(..., registration_error=0.0, ...)` (detection/m3c2.py). The exported `uncertainty` field (py4dgeo `lodetection`, LoD95) therefore reflects only point spread and density in the cylinder. py4dgeo computes `LoD = 1.96 · (sqrt(σ1²/n1 + σ2²/n2) + registration_error)` (verified with a synthetic run: passing 0.10 m raises the LoD by exactly 0.196 m).

**Observed in**: Ristvassdrag: median exported LoD 0.7 cm against 8 cm noise measured on reference areas; 89 % of valid core points exceed their own LoD, including stable terrain. With the reference-area σ passed as registration error the share drops to 5 % (2017→2025) and 3 % (2025→2026).

**Workaround**: Add `1.96 · σ_ref` to the exported `uncertainty` in post-processing; this is exactly what a run with `registration_error=σ_ref` would produce.

**Potential Fix**: Expose `detection.m3c2.registration_error` in the configuration (manual value, or derived from M3C2 statistics on a user-supplied set of reference-area polygons), and log the value used. Do not feed the ICP RMSE in directly: it includes nearest-neighbour sampling distances (two perfectly aligned but differently sampled planes give a large non-zero RMSE). Note also that the reference-area spread partly double-counts the roughness term already inside the LoD, so the result is conservative. A polygon rule such as "mean lowering > mean LoD" is a screening heuristic, not a polygon-level significance test.

**Status**: Open.

### Significance Mask Is Never Produced (`use_significance` has no effect)

**Issue**: `M3C2Result.significant` is always `None` for the standard M3C2 (detection/m3c2.py, "Original M3C2 does not produce significance mask without EP"), so the `use_significance` option of the erosion polygon export never applies and the LAZ export never carries a `significant` dimension.

**Potential Fix**: Once a registration error is configured, derive `significant = |distance| > uncertainty` in the detector and pass it through; keep the option off by default.

**Status**: Documented limitation.

---

## Data Loading

### Duplicate Input Files Are Read Twice (COPC copies)

**Issue**: File discovery takes every LAZ/LAS file in an epoch folder. A `.copc.laz` copy of a cloud that is also present as plain `.laz` is treated as a second file, so its points are loaded twice.

**Observed in**: Ristvassdrag 2025 drone epoch: the run logs report 45 161 615 core points where the dataset has 23.9 million ground points; the exported LAZ files hold only 21 825 680 unique XYZ positions (17.2 million points occur twice). Consequences: the header-based autotune sums only the first epoch's headers, so the doubled 2025 count shrank the automatic radius for the 2025→2026 run (T1 = 2025; the cylinder held about half as many distinct points as the 32 targeted) but not for 2017→2025 (T1 = 2017); in the LoD term the duplicated epoch's point count is doubled (LoD understated by up to √2 for that term); LAZ exports and run-log point counts contain duplicates. The change rasters and polygons were unaffected (a re-run without the copy gave identical extents and polygons).

**Workaround**: Keep only one copy of each cloud in the epoch folder; deduplicate exported LAZ files on exact XYZ before computing point statistics.

**Potential Fix**: Skip `*.copc.laz` when a file with the same stem and `.laz` or `.las` exists (the 2025 originals were `.las`), and vice versa, warn when the share of exact-duplicate XYZ positions in an epoch exceeds a few percent, and report the deduplicated point count in the run log.

**Status**: Open.

---

## How to Report New Issues

When encountering new issues, please document:
1. Configuration used (config file, key settings)
2. Dataset characteristics (size, number of files)
3. Full log output
4. Expected vs actual behavior
