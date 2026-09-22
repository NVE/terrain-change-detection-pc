# Feature Requests

Improvements identified while preparing the Geoteknikkdagen 2026 article on the Ristvassdrag data (September 2026). Defects are tracked in `KNOWN_ISSUES.md`; this list covers additions.

## 1. Slope-aware volume in the erosion polygon export
**Why**: the current volume is true volume × cos θ (see Known Issues). **Sketch**: rasterise the normal z-component next to the distance and use `pixel_area / |n_z|` per cell (exact only for locally parallel surfaces, see Known Issues); keep the current value as `volume_loss_flat_m3`, and offer vertical DTM differencing as the exact alternative when two DTMs are available (item 7).

## 2. Configurable registration error and level-of-detection outputs
**Why**: the exported LoD is unusable as a significance test without an alignment term. **Sketch**: `detection.m3c2.registration_error` (value, or `from_reference_areas` with a polygon file; not the ICP RMSE, which includes sampling distances); export the LoD as a raster band alongside the distance raster (requested by NVE); add per-polygon `mean_lod_m`, `n_core_points`, `share_points_exceeding_lod` and a boolean `lod_pass` (mean lowering > mean LoD, a screening heuristic rather than a significance test). On Ristvassdrag this rule passed 308 of 314 polygons ≥ 10 m² and 90 of 93 ≥ 5 m²; the failures were small polygons with few or scattered points.

## 3. Reference-area noise check in the workflow
**Why**: the practical uncertainty is the spread of M3C2 distances on terrain that did not change (8 cm airborne→drone, 6 cm drone→drone on Ristvassdrag), and it should be measured for every pair. **Sketch**: accept a polygon file of stable areas (or auto-select flat cells > N m from streams and outside polygons), report mean/σ/share outside ±k σ in the run summary, and suggest display and outline thresholds of about 3 σ.

## 4. Polygon quality attributes and plausibility filter
**Why**: edge effects and opposite-bank hits create small polygons with impossible depths. **Sketch**: attributes `dist_to_coverage_edge_m`, `mean_slope_deg` (from a DTM or the normals), `p95_erosion_m` already exists; optional `max_p95_depth_m` filter; flag polygons touching the coverage boundary as truncated instead of dropping them.

## 5. Duplicate input detection
**Why**: a COPC copy in the epoch folder doubled the 2025 cloud (see Known Issues). **Sketch**: skip same-stem `.copc.laz`/`.laz` twins, warn on a high share of exact-duplicate XYZ, log deduplicated counts.

## 6. Stable-area-constrained ICP
**Why**: ICP on a narrow corridor in active change introduced a systematic offset (Known Issues, Alignment). **Sketch**: restrict the ICP sample to user-supplied reference polygons, and always report the residual on those areas.

## 7. Volume cross-check by DTM differencing (optional)
**Why**: vertical differencing of gridded surfaces is the exact volume integral and is a cheap independent check of the M3C2-based volumes where both DTMs exist. **Sketch**: optional per-polygon `dtm_check_volume_m3` when two DTMs are supplied.
