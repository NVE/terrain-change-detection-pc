# Configuration Guide for Out-of-Core Processing

This guide explains how the configuration system coordinates with the streaming/out-of-core processing features.

## Configuration Files

### 1. **Default Configuration** (`config/default.yaml`)
Standard configuration with out-of-core processing **disabled**. Use for datasets that fit in memory.

```yaml
outofcore:
  enabled: false
  streaming_mode: true  # Ready when you enable it
  tile_size_m: 500.0
  halo_m: 20.0
  chunk_points: 1000000
  save_transformed_files: false  # Opt-in to writing transformed files
```

### 2. **Synthetic Profile** (`config/profiles/synthetic.yaml`)
Optimized for smaller synthetic test datasets.

```yaml
outofcore:
  enabled: false
  tile_size_m: 300.0  # Smaller tiles
  save_transformed_files: false  # No need for file transforms
```

### 3. **Large Scale Profile** (`config/profiles/large_scale.yaml`) ⭐
**Fully coordinated out-of-core configuration** for large datasets that don't fit in memory.

```yaml
outofcore:
  enabled: true  # Enables streaming throughout workflow
  tile_size_m: 500.0  # Tile size in meters for large areas
  halo_m: 30.0  # Buffer for edge effects
  chunk_points: 2000000  # 2M points per chunk
  streaming_mode: true  # Use streaming preprocessing
  save_transformed_files: false  # Enable explicitly when you want files
  output_dir: null  # Auto-generate when saving transformed files
```

## Configuration Parameters

### Out-of-Core Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | false | Master switch for out-of-core processing |
| `streaming_mode` | bool | true | Use streaming in preprocessing (when enabled) |
| `save_transformed_files` | bool | false | Save transformed LAZ files during alignment |
| `tile_size_m` | float | 500.0 | Tile size in meters for tiled processing |
| `halo_m` | float | 20.0 | Buffer width around tiles in meters |
| `chunk_points` | int | 1000000 | Points per chunk for streaming reads |
| `output_dir` | str | null | Output directory (auto-generated if null) |

## How Configuration Coordinates the Workflow

### 1. **Preprocessing Stage**

```python
# Config determines streaming mode
if cfg.outofcore.enabled and cfg.outofcore.streaming_mode:
    # Returns file paths + metadata (no data loading)
    batch_loader = BatchLoader(streaming_mode=True)
    data = batch_loader.load_dataset(dataset_info, streaming=True)
else:
    # Traditional in-memory loading
    batch_loader = BatchLoader(streaming_mode=False)
    data = batch_loader.load_dataset(dataset_info)
```

**Config Impact:**
- `outofcore.enabled` + `outofcore.streaming_mode` → File paths returned
- Otherwise → Full data loaded into memory

### 2. **Alignment Stage**

```python
# Config determines if transforms are saved to files
if cfg.outofcore.enabled and cfg.outofcore.save_transformed_files:
    # Apply transformation in streaming fashion
    output_dir = cfg.outofcore.output_dir or auto_generate_dir()
    aligned_files = apply_transform_to_files(
        input_files=file_paths,
        output_dir=output_dir,
        transform=transform_matrix,
        chunk_points=cfg.outofcore.chunk_points,
    )
```

**Config Impact:**
- `save_transformed_files=true` → Transformed LAZ files saved
- `save_transformed_files=false` → Only in-memory transformation
- `output_dir` → Controls where files are saved
- `chunk_points` → Controls streaming chunk size

### 3. **Change Detection Stage**

```python
# Config determines tiling parameters
if cfg.outofcore.enabled:
    dod_result = ChangeDetector.compute_dod_streaming_files_tiled(
        files_t1=file_paths_t1,
        files_t2=file_paths_t2,
        cell_size=cfg.detection.dod.cell_size,
        tile_size=cfg.outofcore.tile_size_m,
        halo=cfg.outofcore.halo_m,
        chunk_points=cfg.outofcore.chunk_points,
        # Advanced: optionally back mosaic arrays on disk to reduce RAM
        # memmap_dir=".mosaic_tmp",
    )
```

**Config Impact:**
- `tile_size_m` → Size of processing tiles
- `halo_m` → Overlap between tiles
- `chunk_points` → Points per streaming chunk
- `detection.dod.cell_size` → Grid resolution

For algorithm-specific guidance (DoD, C2C, and M3C2) and how tiling/halo settings differ, see docs/ALGORITHMS.md.

## Usage Examples

### Running with Default Config (In-Memory)
```bash
python scripts/run_workflow.py
```

### Running with Large-Scale Profile (Out-of-Core)
```bash
python scripts/run_workflow.py --config config/profiles/large_scale.yaml
```

### Creating Custom Profile
```yaml
# config/profiles/my_dataset.yaml
paths:
  base_dir: data/my_large_dataset

outofcore:
  enabled: true
  streaming_mode: true
  save_transformed_files: true
  tile_size_m: 2000.0  # 2km tiles
  halo_m: 100.0  # 100m overlap
  chunk_points: 5000000  # 5M points per chunk
  output_dir: /path/to/fast/storage

alignment:
  subsample_size: 200000  # More samples for large dataset

detection:
  dod:
    cell_size: 5.0  # Coarser for regional analysis
```

## Coordination Benefits

✅ **Single Source of Truth** - All modules read from same config  
✅ **Consistent Behavior** - Same settings throughout workflow  
✅ **Easy Switching** - Toggle between in-memory and streaming modes  
✅ **Profile-Based** - Different configs for different dataset sizes  
✅ **Validated** - Pydantic ensures type safety and defaults  

## Configuration Validation

The configuration system uses Pydantic for validation:

```python
from terrain_change_detection.utils.config import load_config

# Loads with validation
cfg = load_config("config/profiles/large_scale.yaml")

# Type-safe access
assert isinstance(cfg.outofcore.enabled, bool)
assert isinstance(cfg.outofcore.tile_size_m, float)
```

All configuration values are validated at load time, preventing runtime errors from misconfiguration.

## Advanced: On-Disk Mosaic (Memmap)

For very large output grids, you can reduce memory by enabling on-disk mosaicking in the tiled streaming DoD call:

```python
ChangeDetector.compute_dod_streaming_files_tiled(
    ..., memmap_dir="/fast/scratch/mosaic"
)
```

This backs the internal `sum`/`cnt` arrays with `numpy.memmap` files in the provided directory. It lowers RAM usage at the cost of extra disk I/O. Grid `X/Y` arrays are still returned as in-memory arrays.

## M3C2: Streaming + Tiling

For large datasets, you can run M3C2 without loading whole epochs into memory by tiling core points and streaming points per tile:

```python
from terrain_change_detection.detection import ChangeDetector

m3c2_res = ChangeDetector.compute_m3c2_streaming_files_tiled(
    core_points=core_src,           # Nx3, e.g., subsample of T1
    files_t1=files_t1,              # LAZ/LAS epoch 1
    files_t2=files_t2_aligned,      # LAZ/LAS epoch 2 (aligned to T1)
    params=m3c2_params,             # from autotune_m3c2_params()
    tile_size=cfg.outofcore.tile_size_m,
    halo=None,                      # default halo = max(cyl_radius, projection_scale)
    ground_only=cfg.preprocessing.ground_only,
    classification_filter=cfg.preprocessing.classification_filter,
    chunk_points=cfg.outofcore.chunk_points,
)
```

- Halo rule of thumb: use at least `max(cylinder_radius, projection_scale)` to ensure neighborhoods are contained within each tile’s outer bbox.
- If some tiles have zero points in an epoch, distances for their core points return as NaN and are logged.

## C2C: Streaming + Tiling

Cloud‑to‑Cloud (nearest neighbor) can also run out‑of‑core by tiling the XY domain and streaming per tile. This requires a finite radius to bound neighborhoods.

```python
from terrain_change_detection.detection import ChangeDetector

c2c_res = ChangeDetector.compute_c2c_streaming_files_tiled(
    files_src=files_t2_aligned,          # LAZ/LAS epoch 2 (aligned), source cloud
    files_tgt=files_t1,                  # LAZ/LAS epoch 1, target cloud
    tile_size=cfg.outofcore.tile_size_m,
    max_distance=float(cfg.detection.c2c.max_distance),  # required radius (meters)
    ground_only=cfg.preprocessing.ground_only,
    classification_filter=cfg.preprocessing.classification_filter,
    chunk_points=cfg.outofcore.chunk_points,
)
```

- Set `detection.c2c.max_distance` to enable streaming C2C; the tile halo uses this radius.
- Indices are not tracked in streaming mode (set to `-1`); distances and summary stats are returned.
- If a tile has no target points in its halo, distances for that tile’s sources are set to `inf` and logged.

See docs/ALGORITHMS.md for algorithm-specific guidance on C2C tiling, halos, and classification.
