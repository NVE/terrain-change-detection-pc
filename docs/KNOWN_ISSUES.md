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

## How to Report New Issues

When encountering new issues, please document:
1. Configuration used (config file, key settings)
2. Dataset characteristics (size, number of files)
3. Full log output
4. Expected vs actual behavior
