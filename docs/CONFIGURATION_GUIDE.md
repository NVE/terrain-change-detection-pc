# Configuration Guide for Out-of-Core Processing

This guide explains how the configuration system coordinates with the streaming/out-of-core processing features.

## Configuration Files

### 1. **Default Configuration** (`config/default.yaml`)
Standard configuration with out-of-core processing **disabled**. Use for datasets that fit in memory.

```yaml
outofcore:
  enabled: false
  streaming_mode: true  # Ready when you enable it
  save_transformed_files: true
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
  tile_size_m: 1000.0  # 1km tiles for large areas
  halo_m: 50.0  # Larger buffer for edge effects
  chunk_points: 2000000  # 2M points per chunk
  streaming_mode: true  # Use streaming preprocessing
  save_transformed_files: true  # Save transformed LAZ files
  output_dir: data/processed  # Where to save outputs
```

## Configuration Parameters

### Out-of-Core Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | false | Master switch for out-of-core processing |
| `streaming_mode` | bool | true | Use streaming in preprocessing (when enabled) |
| `save_transformed_files` | bool | true | Save transformed LAZ files during alignment |
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
    )
```

**Config Impact:**
- `tile_size_m` → Size of processing tiles
- `halo_m` → Overlap between tiles
- `chunk_points` → Points per streaming chunk
- `detection.dod.cell_size` → Grid resolution

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
