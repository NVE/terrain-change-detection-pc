# Terrain Change Detection from Multi-temporal Point Clouds

A Python toolkit for detecting and quantifying terrain changes using multi-temporal LiDAR point cloud data. Built for processing large-scale datasets with support for GPU acceleration and parallel processing.

## Overview

This project provides a complete pipeline for terrain change detection:

1. **Data Discovery** — Automatically find and organize point cloud datasets
2. **Spatial Alignment** — Register point clouds using ICP with optional coarse alignment
3. **Change Detection** — Quantify changes using M3C2, DoD, or C2C methods
4. **Visualization** — Interactive 3D visualizations and heatmaps

The toolkit handles datasets of any size through out-of-core streaming and spatial tiling, with optional GPU acceleration for compute-intensive operations.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yaredwb/terrain-change-detection-pc.git
cd terrain-change-detection-pc

# Install dependencies (requires uv package manager)
uv sync
```

### Generate Test Data

```bash
uv run scripts/generate_synthetic_laz.py
```

### Run the Pipeline

```bash
uv run scripts/run_workflow.py --base-dir data/synthetic
```

This runs the complete workflow: data discovery → alignment → change detection → visualization.

## Project Structure

```
terrain-change-detection-pc/
├── config/                     # Configuration files
│   ├── default.yaml           # Default settings
│   └── profiles/              # Preset configurations
├── docs/                       # Documentation
├── scripts/                    # Executable scripts
│   ├── run_workflow.py        # Main pipeline
│   ├── explore_data.py        # Data exploration
│   └── generate_*.py          # Test data generators
├── src/terrain_change_detection/
│   ├── preprocessing/         # Data loading and filtering
│   ├── alignment/             # ICP and coarse registration
│   ├── detection/             # DoD, C2C, M3C2 algorithms
│   ├── acceleration/          # GPU and parallel processing
│   ├── visualization/         # Plotting utilities
│   └── utils/                 # Configuration and helpers
├── exploration/               # Data exploration scripts
└── tests/                     # Test suite
```

## Data Organization

The pipeline expects point cloud data organized by area and time period:

```
data/raw/
└── my_area/
    ├── 2020/
    │   └── data/
    │       └── *.laz
    └── 2023/
        └── data/
            └── *.laz
```

Two data source formats are supported:
- **hoydedata** (default): `area/time_period/data/*.laz`
- **drone**: `area/time_period/*.laz` (no `data/` subdirectory)

Set `source_type` in your configuration to match your data structure.

## Change Detection Methods

### M3C2 (Primary Method)
Multi-scale Model-to-Model Cloud Comparison — computes distances along locally estimated surface normals. Best for detecting true surface changes while minimizing noise effects.

### DoD (DEM of Difference)
Grids both point clouds into DEMs and computes elevation differences. Fast and intuitive for terrain analysis.

### C2C (Cloud-to-Cloud)
Computes nearest neighbor distances between point clouds. Useful for quick comparisons and as a baseline.

## Configuration

Configuration is managed through YAML files. Key sections:

```yaml
paths:
  base_dir: data/raw              # Data location

preprocessing:
  ground_only: true               # Filter to ground points

alignment:
  max_iterations: 100             # ICP iterations
  subsample_size: 50000           # Points for alignment

detection:
  dod:
    enabled: true
    cell_size: 1.0                # Grid resolution (m)
  c2c:
    enabled: true
    max_distance: 10.0            # Search radius (m)
  m3c2:
    enabled: true
    use_autotune: true            # Auto-calculate parameters

# For large datasets
outofcore:
  enabled: true
  tile_size_m: 500.0

# For multi-core processing
parallel:
  enabled: true
  n_workers: null                 # Auto-detect
```

See [CONFIGURATION_GUIDE.md](docs/CONFIGURATION_GUIDE.md) for complete reference.

### Configuration Profiles

| Profile | Use Case |
|---------|----------|
| `default.yaml` | In-memory processing for small-medium datasets |
| `profiles/synthetic.yaml` | Testing with synthetic data |
| `profiles/large_scale.yaml` | Large datasets with GPU acceleration |
| `profiles/drone.yaml` | Drone scanning data format |

## Processing Modes

### In-Memory (Default)
Best for datasets under ~10M points. Fast with minimal overhead.

### Out-of-Core
For datasets of any size. Streams data through spatial tiles with constant memory usage.

```yaml
outofcore:
  enabled: true
  tile_size_m: 500.0
  streaming_mode: true
```

### Parallel Processing
Multi-core CPU processing for medium-large datasets.

```yaml
parallel:
  enabled: true
  n_workers: null  # Auto-detect CPU cores
```

### GPU Acceleration
CUDA-powered acceleration for C2C nearest neighbor searches on large datasets (1M+ points).

```yaml
gpu:
  enabled: true
  use_for_c2c: true
```

> **Note**: GPU acceleration applies to C2C and DoD operations only. M3C2 uses py4dgeo's C++ backend, and it has not been possible to expose its internal operations (nearest neighbor searches, normal estimation) to GPU acceleration. GPU and parallel processing cannot be enabled simultaneously due to CUDA limitations.

## Area Clipping

Focus analysis on specific regions by clipping to polygon boundaries:

```yaml
clipping:
  enabled: true
  boundary_file: path/to/boundary.geojson
```

See [AREA_CLIPPING_GUIDE.md](docs/AREA_CLIPPING_GUIDE.md) for details.

## GPU Setup (Optional)

For GPU acceleration on large datasets:

```bash
# Install GPU dependencies
uv sync --extra gpu

# Activate GPU libraries (Linux/WSL2)
source activate_gpu.sh

# Enable in config
gpu:
  enabled: true
  use_for_c2c: true
```

Requirements:
- NVIDIA GPU (Compute Capability 6.0+)
- CUDA Toolkit 12.x
- 4+ GB GPU memory

See [GPU_SETUP_GUIDE.md](docs/GPU_SETUP_GUIDE.md) for detailed instructions.

## Running the Pipeline

### Basic Usage

```bash
# Run with default config
uv run scripts/run_workflow.py

# Specify data directory
uv run scripts/run_workflow.py --base-dir data/synthetic

# Use custom config
uv run scripts/run_workflow.py --config config/profiles/large_scale.yaml
```

### Data Exploration

```bash
# Explore data structure and statistics
uv run scripts/explore_data.py
```

### Generate Test Data

```bash
# Small synthetic dataset (~100K points)
uv run scripts/generate_synthetic_laz.py

# Large synthetic dataset (configurable size)
uv run scripts/generate_large_synthetic_laz.py
```

## Dependencies

**Core** (installed automatically):
- `laspy`, `lazrs` — Point cloud I/O
- `py4dgeo` — M3C2 algorithms
- `scikit-learn` — KD-tree operations
- `plotly`, `pyvista` — Visualization
- `numpy`, `scipy` — Numerical computing

**Optional GPU**:
- `cupy-cuda12x` — GPU arrays
- `numba` — JIT compilation
- `cuml` — GPU ML (Linux only)

## Documentation

- [CONFIGURATION_GUIDE.md](docs/CONFIGURATION_GUIDE.md) — Complete configuration reference
- [GPU_SETUP_GUIDE.md](docs/GPU_SETUP_GUIDE.md) — GPU acceleration setup
- [ALGORITHMS.md](docs/ALGORITHMS.md) — Algorithm explanations
- [DRONE_DATA_SUPPORT.md](docs/DRONE_DATA_SUPPORT.md) — Drone data integration
- [AREA_CLIPPING_GUIDE.md](docs/AREA_CLIPPING_GUIDE.md) — Area clipping guide
- [CHANGELOG.md](docs/CHANGELOG.md) — Change history

## Testing

```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest tests/test_gpu_*.py -v
uv run pytest tests/test_change_detection.py -v
```

## License

[Add license information]
