# Bug Fix: Laspy 2.x API Compatibility

## Issue Summary
**Date:** 2025-11-06  
**Severity:** Critical  
**Component:** `acceleration/tiling.py`

### Problem
The streaming DoD computation failed with the error:
```
AttributeError: points is not a valid dimension
```

This occurred when processing large-scale datasets using the `large_scale.yaml` configuration profile with out-of-core tiled processing.

### Root Cause
The `LaspyStreamReader._mask_classes()` method used laspy 1.x API syntax:
```python
n = len(las.points)  # ❌ Invalid in laspy 2.x
```

In laspy 2.x, the `.points` attribute doesn't exist. The correct way to get the point count is:
```python
n = len(las)  # ✅ Correct for laspy 2.x
```

### Impact
- **Before Fix:** Streaming DoD failed immediately when reading transformed LAZ files
- **After Fix:** Full workflow completes successfully with streaming mode
- **Affected Workflows:** Any out-of-core processing using `LaspyStreamReader`

### Fix Details

**File:** `src/terrain_change_detection/acceleration/tiling.py`  
**Line:** 382  
**Change:**
```diff
- n = len(las.points)
+ n = len(las)  # laspy 2.x: len() works directly on point record
```

### Verification

#### 1. Diagnostic Test
Created `scripts/test_tiled_dod.py` to isolate the issue:
```bash
uv run python scripts/test_tiled_dod.py
# ✅ SUCCESS! Tiled DoD computed
# Grid shape: (567, 889)
# Stats: {'n_cells': 481207, ...}
```

#### 2. Full Workflow Test
```bash
uv run python scripts/run_workflow.py --config config/profiles/large_scale.yaml
# ✅ DoD stats: n_cells=481232, mean=-0.0135m, median=-0.0009m, rmse=0.15m
```

#### 3. Unit Tests
```bash
uv run pytest tests/ -v -k "streaming or tiling"
# ✅ 6 passed in 1.79s
```

### Performance Metrics
**Large Scale Dataset (15M + 20M points):**
- Streaming mode: ✅ Working
- Alignment: RMSE 1.325m (100K subsampled points)
- File transformation: 9M points processed in ~3 seconds
- Streaming DoD: 481K cells computed in ~6 seconds
- Memory: Constant (chunked processing)

### Related Components
- `LaspyStreamReader` class in `tiling.py`
- `compute_dod_streaming_files_tiled()` in `change_detection.py`
- `apply_transform_to_files()` in `streaming_alignment.py`

### Prevention
Added comprehensive logging throughout the streaming pipeline to catch similar issues early:
- Chunk size and point counts logged during transformation
- File creation and point counts verified
- Error handling with fallback to in-memory processing

### Testing Coverage
All streaming/tiling tests passing:
- ✅ `test_classification_filter_shared_utility`
- ✅ `test_filter_statistics`
- ✅ `test_batch_loader_streaming_mode`
- ✅ `test_laspy_stream_reader_filtering`
- ✅ `test_bounds_2d_utility`
- ✅ `test_integration_classification_consistency`

### Lessons Learned
1. **API Version Compatibility:** Always verify library API versions when working with external dependencies
2. **Isolated Testing:** Creating minimal reproduction scripts (like `test_tiled_dod.py`) helps quickly isolate issues
3. **Comprehensive Logging:** Enhanced logging throughout the streaming pipeline made debugging much faster
4. **Fallback Mechanisms:** Having in-memory fallback prevented complete workflow failure

### Dependencies
- **laspy:** 2.x (uses modern API without `.points` attribute)
- **numpy:** For array operations
- **Classification filtering:** Uses shared `utils.point_cloud_filters` module

## Summary
A single-line fix resolved the critical streaming DoD failure. The issue was caused by using deprecated laspy 1.x API syntax (`las.points`) instead of the laspy 2.x syntax (`len(las)`). All tests pass and the full workflow now completes successfully with large-scale datasets.
