# Configuration Guide

This guide documents all configuration parameters for the terrain change detection toolkit. Configuration files use YAML format and reside in the `config/` directory.

## Running with Configuration Profiles

```bash
# Default configuration (in-memory processing)
uv run scripts/run_workflow.py

# Specific configuration profile
uv run scripts/run_workflow.py --config config/profiles/large_scale.yaml
```

## Available Configuration Profiles

| Profile | Description |
| :--- | :--- |
| `default.yaml` | Base configuration with sensible defaults for all parameters |
| `default_clipped.yaml` | Default configuration with area clipping enabled |
| `profiles/large_scale.yaml` | Optimized for national-scale datasets with streaming |
| `profiles/large_synthetic.yaml` | Large synthetic data with streaming enabled |
| `profiles/drone.yaml` | Tuned for high-density drone-captured point clouds |
| `profiles/synthetic.yaml` | Settings for validation with generated test data |

---

## 1. Paths and Discovery

These settings define where input data is located and where results are stored.

```yaml
paths:
  base_dir: "data/raw"
  output_dir: null
  output_crs: "EPSG:25833"

discovery:
  source_type: "hoydedata"
  data_dir_name: "data"
  metadata_dir_name: "metadata"
```

| Section | Parameter | Default | Description |
| :--- | :--- | :--- | :--- |
| **paths** | `base_dir` | `data/raw` | Root directory containing the dataset area folders. |
| | `output_dir` | `null` | Destination for results. If `null`, defaults to `base_dir/output`. |
| | `output_crs` | `EPSG:25833` | Coordinate Reference System for exported files. |
| **discovery** | `source_type` | `hoydedata` | Directory layout: `hoydedata` (nested `data/` folder) or `drone` (flat structure). |
| | `data_dir_name` | `data` | Subfolder name for point clouds (used with `hoydedata` source). |
| | `metadata_dir_name` | `metadata` | Subfolder name for metadata files. |

---

## 2. Preprocessing and Clipping

Controls how point clouds are filtered and spatially subset before analysis.

```yaml
preprocessing:
  ground_only: true
  classification_filter: [2]

clipping:
  enabled: false
  boundary_file: "area_boundary.geojson"
  feature_name: null
  save_clipped_files: false
  output_dir: null
```

| Section | Parameter | Default | Description |
| :--- | :--- | :--- | :--- |
| **preprocessing** | `ground_only` | `true` | If true, retains only LAS Class 2 (Ground) points. |
| | `classification_filter` | `[2]` | Specific LAS classes to keep (overrides `ground_only`). |
| **clipping** | `enabled` | `false` | If true, clips point clouds to the polygon defined in `boundary_file`. |
| | `boundary_file` | `null` | Path to a GeoJSON or Shapefile polygon. |
| | `feature_name` | `null` | Name of the specific feature to use if the file contains multiple. |
| | `save_clipped_files` | `false` | If true, saves the clipped LAZ files to disk to speed up future runs. |
| | `output_dir` | `null` | Specific directory for clipped files (auto-generated if null). |

**Local Coordinate Transformation**: By default, the toolkit automatically transforms point clouds to a local coordinate system before processing to avoid floating-point precision issues with large UTM coordinates. This is controlled by the `coordinates` section:
- `use_local_coordinates` (default: `true`) — Enables/disables the feature
- `origin_method` (`min_bounds`, `centroid`, or `first_point`) — Determines how the local origin is computed
- `include_z_offset` (default: `false`) — Optionally shifts Z values as well

---

## 3. Alignment (ICP)

Settings for registering (aligning) point clouds from different time periods using ICP (Iterative Closest Point).

```yaml
alignment:
  enabled: true
  max_iterations: 100
  tolerance: 1.0e-6
  max_correspondence_distance: 1.0
  subsample_size: 50000
  export_aligned_pc: false
  
  coarse:
    enabled: false
    method: "centroid"
```

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `enabled` | `true` | Enable/disable ICP alignment step. |
| `max_iterations` | `100` | Maximum number of iterations for the ICP algorithm. |
| `tolerance` | `1e-6` | Convergence threshold based on MSE change. |
| `max_correspondence_distance` | `1.0` | Max distance (meters) to search for matching points. |
| `subsample_size` | `50000` | Number of points to sample for calculating alignment. |
| `export_aligned_pc` | `false` | If true, saves the aligned T2 point cloud as a new LAZ file. |
| `convergence_translation_epsilon` | `1e-4` | Translation threshold for early stopping. |
| `convergence_rotation_epsilon_deg` | `0.1` | Rotation threshold (degrees) for early stopping. |
| **coarse.enabled** | `false` | Enables pre-alignment. Use only if datasets are significantly misaligned (>1m). |
| **coarse.method** | `centroid` | Alignment method: `centroid`, `pca`, `phase`, or `open3d_fpfh`. |
| **coarse.voxel_size** | `2.0` | Voxel size (m) for downsampling during coarse registration. |

---

## 4. Detection Methods

Configuration for the three available change detection algorithms.

### 4.1 M3C2 (Multiscale Model to Model Cloud Comparison)

The recommended method for accurate 3D change detection.

```yaml
detection:
  m3c2:
    enabled: true
    use_autotune: true
    core_points_percent: 10.0
    export_pc: true
    export_raster: true
    
    autotune:
      source: header
      target_neighbors: 16
      max_depth_factor: 1.0
      min_radius: 1.0
      max_radius: 20.0
    
    fixed:
      radius: null
      normal_scale: null
      depth_factor: null
```

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `enabled` | `true` | Activates M3C2 processing. |
| `use_autotune` | `true` | Automatically estimates cylinder radius from point density. |
| `core_points_percent` | `10.0` | Percentage of T1 points to use as "core points". |
| `export_pc` | `true` | Saves results as a point cloud (.laz) with distance attributes. |
| `export_raster` | `true` | Interpolates results to a GeoTIFF raster. |
| **autotune.source** | `header` | Density source: `header` (fast) or `sample` (accurate). |
| **autotune.target_neighbors** | `16` | Desired number of neighbors to define the cylinder radius. |
| **autotune.max_depth_factor** | `1.0` | Cylinder depth relative to radius. |
| **autotune.min_radius** | `1.0` | Minimum allowed cylinder radius (meters). |
| **autotune.max_radius** | `20.0` | Maximum allowed cylinder radius (meters). |
| **fixed.radius** | `null` | Manual cylinder radius (used if `use_autotune: false`). |
| **fixed.normal_scale** | `null` | Manual scale for normal estimation. |
| **fixed.depth_factor** | `null` | Manual depth factor (max_depth = radius × depth_factor). |

### 4.2 DoD (DEM of Difference)

Fast, grid-based vertical change detection.

```yaml
detection:
  dod:
    enabled: false
    cell_size: 1.0
    aggregator: "mean"
    export_raster: false
```

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `enabled` | `false` | Activates DoD processing. |
| `cell_size` | `1.0` | Grid resolution in meters. |
| `aggregator` | `mean` | Statistic for cell value: `mean`, `median`, `p95`, `p5`. |
| `export_raster` | `false` | Saves result as GeoTIFF. |

### 4.3 C2C (Cloud to Cloud)

Simple absolute distance measurement.

```yaml
detection:
  c2c:
    enabled: false
    mode: "euclidean"
    max_distance: 10.0
    export_pc: false
    export_raster: false
```

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `enabled` | `false` | Activates C2C processing. |
| `mode` | `euclidean` | `euclidean` (3D distance) or `vertical_plane` (local reference). |
| `max_distance` | `10.0` | Maximum search radius for nearest neighbor. |
| `max_points` | `1000000` | Limit before switching to tiled processing. |
| `export_pc` | `false` | Saves results as LAZ point cloud. |
| `export_raster` | `false` | Interpolates results to GeoTIFF. |

---

## 5. Execution Modes

These sections control performance, scalability, and hardware acceleration.

### 5.1 Out-of-Core Processing

For datasets that exceed available system memory.

```yaml
outofcore:
  enabled: false
  tile_size_m: 500.0
  halo_m: 20.0
  chunk_points: 1000000
  streaming_mode: true
  save_transformed_files: false
  output_dir: null
  memmap_dir: null
```

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `enabled` | `false` | Master switch for out-of-core processing. |
| `tile_size_m` | `500.0` | Size of each square tile in meters. |
| `halo_m` | `20.0` | Overlay buffer to prevent edge artifacts. |
| `chunk_points` | `1000000` | Number of points to read at once during streaming. |
| `streaming_mode` | `true` | Use streaming preprocessing. |
| `save_transformed_files` | `false` | Save intermediate transformed LAZ files. |
| `output_dir` | `null` | Directory for transformed files. |
| `memmap_dir` | `null` | Directory for memory-mapped arrays. |

### 5.2 Parallel Processing

Multi-core CPU processing for tile-based workloads.

```yaml
parallel:
  enabled: false
  n_workers: null
  memory_limit_gb: null
```

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `enabled` | `false` | Process multiple tiles simultaneously. |
| `n_workers` | `null` | Number of CPU cores (null = all available - 1). |
| `memory_limit_gb` | `null` | Soft memory limit per worker. |

### 5.3 GPU Acceleration

CUDA-based acceleration for NVIDIA GPUs.

```yaml
gpu:
  enabled: false
  gpu_memory_limit_gb: null
  fallback_to_cpu: true
  use_for_c2c: true
  use_for_preprocessing: true
  use_for_alignment: false
```

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `enabled` | `false` | Enable CUDA acceleration (NVIDIA only). |
| `gpu_memory_limit_gb` | `null` | Max GPU memory (null = 80%). |
| `fallback_to_cpu` | `true` | Fall back to CPU if GPU init fails. |
| `use_for_c2c` | `true` | Use GPU for nearest neighbor search. |
| `use_for_preprocessing` | `true` | Use GPU for coordinates/filtering. |
| `use_for_alignment` | `false` | Use GPU for ICP (experimental). |

> **Note**: GPU mode and Parallel mode are mutually exclusive due to CUDA constraints.

---

## 6. Visualization and Logging

```yaml
visualization:
  backend: "plotly"
  sample_size: 100000

logging:
  level: "INFO"
  file: null

performance:
  numpy_threads: "auto"
```

| Section | Parameter | Default | Description |
| :--- | :--- | :--- | :--- |
| **visualization** | `backend` | `plotly` | Interactive viewer: `plotly` (web) or `pyvista` (desktop). |
| | `sample_size` | `100000` | Max points to render for responsiveness. |
| **logging** | `level` | `INFO` | Verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| | `file` | `null` | Optional path to write logs to a file. |
| **performance** | `numpy_threads` | `auto` | Limits BLAS/NumPy threads to prevent contention. |

---

## Example: Custom Profile

```yaml
# config/profiles/my_dataset.yaml
paths:
  base_dir: data/my_large_dataset

preprocessing:
  ground_only: true
  classification_filter: [2]

clipping:
  enabled: true
  boundary_file: data/boundary.geojson

alignment:
  enabled: true
  subsample_size: 100000

detection:
  m3c2:
    enabled: true
    use_autotune: false
    core_points_percent: 100.0
    fixed:
      radius: 1.0
      normal_scale: 1.0
      depth_factor: 2.0

outofcore:
  enabled: true
  tile_size_m: 500.0
  halo_m: 30.0

parallel:
  enabled: true
  n_workers: 8
```

---

## Configuration Validation

The configuration system uses Pydantic for validation, ensuring type safety and sensible defaults:

```python
from terrain_change_detection.utils.config import load_config

# Loads with validation
cfg = load_config("config/profiles/large_scale.yaml")

# Type-safe access
assert isinstance(cfg.outofcore.enabled, bool)
assert isinstance(cfg.detection.m3c2.fixed.radius, float | None)
```

All configuration values are validated at load time, preventing runtime errors from misconfiguration.
