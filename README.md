# Terrain Change Detection Based on Multi-temporal Point Clouds

A Python project for detecting terrain changes using multi-temporal point cloud data. The workflow includes data discovery and preprocessing, spatial alignment using ICP (Iterative Closest Point), change detection (with a primary focus on M3C2), and interactive visualizations (Plotly default; PyVista supported). DoD and C2C are implemented for comparison/baseline purposes. PyVista can run non-blocking with the optional `pyvistaqt` dependency.

## Project Structure

```
terrain-change-detection-pc/
├── config/                        # YAML configuration files
│   ├── default.yaml               # Default configuration loaded if not overridden
│   └── profiles/
│       └── synthetic.yaml         # Example profile for synthetic data
├── main.py                          # Main entry point
├── pyproject.toml                   # UV project configuration and dependencies
├── exploration/                     # Exploration scripts and sample data
│   ├── explore.py                   # Data exploration script
│   ├── fast_explore.py              # Quick exploration script
│   └── 33-1-466-136-13.laz          # Sample LAZ point cloud file
├── scripts/                         # Workflow execution scripts
│   ├── explore_data.py              # Data exploration workflow
│   └── run_workflow.py              # Main change detection workflow
├── src/terrain_change_detection/    # Core library modules
│   ├── preprocessing/               # Data loading and preprocessing
│   │   ├── data_discovery.py        # Data discovery utilities
│   │   └── loader.py                # Point cloud data loaders
│   ├── alignment/                   # Spatial alignment algorithms
│   │   └── fine_registration.py     # ICP and fine registration
│   ├── detection/                   # Change detection algorithms
│   │   └── change_detection.py      # Main change detection logic
│   ├── visualization/               # Visualization tools
│   │   └── point_cloud.py           # Point cloud visualization
│   └── utils/                       # Utility modules
│       ├── config.py                # Configuration management
│       ├── io.py                    # Input/output utilities
│       └── logging.py               # Logging setup
└── tests/                           # Test suite
    ├── test_data_discovery.py       # Tests for data discovery script
    ├── test_loader.py               # Tests for point cloud data loader
    └── sample_data/                 # Test data
```

## Features and workflow overview

The pipeline performs:

1. Data discovery and preprocessing (optimized for LiDAR from hoydedata.no).
2. Spatial alignment via optional Coarse Registration followed by ICP. Coarse methods include centroid, PCA, 2D phase correlation, and optional Open3D FPFH/RANSAC when available.
3. Change detection methods (M3C2 is the main algorithm):
    - M3C2 (Original) via py4dgeo.
    - M3C2-EP (Error Propagation) via py4dgeo with Level of Detection (LoD) significance flags.
    - For comparison purposes, DoD (DEM of Difference) with multiple aggregators and C2C (Cloud-to-Cloud) nearest-neighbor distances are implemented.
4. Visualizations shown immediately after each computation:
    - DoD heatmap.
    - Distance histograms (C2C and M3C2/M3C2-EP).
    - 3D core-points colored by M3C2 distances.

Notes:
- M3C2 parameters (projection scale, cylinder radius, max depth) may need tuning per dataset.
- M3C2-EP uses default scan-position noise parameters if not provided; supplying real scanner metadata improves uncertainty estimates.
- Coarse registration can speed up and stabilize ICP on misaligned pairs. Configure under `alignment.coarse` in YAML.

## Getting Started

This project requires `uv` to be installed on your system for dependency management and script execution.

### Prerequisites
- Python 3.13+
- `uv` package manager

### Installation

Clone the repository:
```bash
git clone https://github.com/yaredwb/terrain-change-detection-pc.git
cd terrain-change-detection-pc
```

## Running scripts

Use `uv run` followed by the script path, as script aliases are not defined in `pyproject.toml` at the moment. The virtual environment will be created automatically on first use.

### Exploration Scripts

Explore and analyze point cloud data:

```bash
# Run data exploration from the exploration directory
uv run exploration/explore.py
uv run exploration/fast_explore.py

# Run the main data exploration workflow
uv run scripts/explore_data.py
```

### Main workflow

Execute the complete change detection pipeline. Visualizations appear after each step.

```bash
uv run scripts/run_workflow.py

# With explicit YAML config (optional)
uv run scripts/run_workflow.py --config config/default.yaml

# Override base data directory via CLI
uv run scripts/run_workflow.py --base-dir data/synthetic

# Or run the main entry point (Not ready yet)
uv run main.py
```

### Configuration

The project reads a YAML configuration (typed via pydantic) with sensible defaults.

- Default location: `config/default.yaml`
- Example profile: `config/profiles/synthetic.yaml`

Key sections and keys:
- `paths.base_dir`: dataset root folder (default: `data/raw`).
- `preprocessing.ground_only`: filter to ground classes (default true).
- `preprocessing.classification_filter`: list of LAS class codes to keep (default `[2]`).
- `discovery.data_dir_name` / `discovery.metadata_dir_name`: subfolder names under each time period (defaults `data`/`metadata`).
- `alignment.{max_iterations,tolerance,max_correspondence_distance,subsample_size}`: ICP + sampling.
- `detection.dod.{cell_size,aggregator}`: DoD grid settings.
- `detection.c2c.{max_points,max_distance}`: C2C sampling and optional distance cap.
- `detection.m3c2.core_points`: core points count; `detection.m3c2.autotune.*`: autotune knobs; `detection.m3c2.ep.workers`: worker processes (null uses OS default: 1 on Windows, 4 elsewhere).
- `visualization.{backend,sample_size}`: plotting backend and downsampling.
- `logging.{level,file}`: logging level and optional file output.
- `performance.numpy_threads`: integer thread count or `auto`.

Example run using the synthetic profile:

```powershell
uv run scripts/run_workflow.py --config config/profiles/synthetic.yaml
```

CLI overrides take precedence for values they set (e.g., `--base-dir`).

Example custom YAML snippet:

```yaml
paths:
  base_dir: data/raw
preprocessing:
  ground_only: true
  classification_filter: [2]
discovery:
  data_dir_name: data
  metadata_dir_name: metadata
alignment:
  max_iterations: 120
  tolerance: 1.0e-6
  max_correspondence_distance: 0.8
  subsample_size: 40000
detection:
  dod: { cell_size: 1.0, aggregator: median }
  c2c: { max_points: 15000, max_distance: null }
  m3c2:
    core_points: 12000
    autotune: { target_neighbors: 16, max_depth_factor: 0.6, min_radius: 1.0, max_radius: 20.0 }
    ep: { workers: null }
visualization:
  backend: plotly
  sample_size: 50000
logging:
  level: INFO
  file: null
performance:
  numpy_threads: auto
```

### Synthetic datasets

You can generate a small, controlled synthetic dataset to validate alignment and change detection. It creates two LAZ files with known deposition/erosion and intentional misalignment.

Output location:
- `data/synthetic/synthetic_area/2015/data/synthetic_tile_01.laz`
- `data/synthetic/synthetic_area/2020/data/synthetic_tile_01.laz`

Generate them:

```powershell
uv run scripts/generate_synthetic_laz.py
```

### Running the workflow on specific datasets

By default, the workflow script (`run_workflow.py`) looks for data in the `data/raw` directory. You can specify a different directory, such as the synthetic data directory, using the `--base-dir` command-line argument.

**Example (running on synthetic data):**

```powershell
uv run scripts/run_workflow.py --base-dir data/synthetic
```

This command tells the script to look for area folders inside `data/synthetic` instead of the default location.

Optional: enable non-blocking PyVista windows (BackgroundPlotter):

```powershell
uv add pyvistaqt PySide6
```

Notes:
- `pyvistaqt` uses Qt; `PySide6` provides the Qt runtime.
- On Windows, BackgroundPlotter gives stable non-blocking behavior. Without it, classic `Plotter` windows are blocking to avoid UI freezes.

Windows note (M3C2-EP):
- On Windows, the workflow automatically runs M3C2-EP in a safe single-process mode to avoid multiprocessing spawn issues.
    This has lower performance but better compatibility. On Linux/macOS it runs in parallel by default.

## Dependencies

Core libraries (managed via `uv` in `pyproject.toml`):
- `laspy` — LAZ/LAS I/O
- `py4dgeo` — M3C2 algorithms
- `plotly` — interactive plotting (default backend)
- `pyvista` — optional 3D plotting backend
- `scikit-learn` — KD-tree for C2C and utilities

Configuration:
- `pydantic` — typed validation of YAML config
- `PyYAML` — YAML parsing

Optional (recommended for non-blocking PyVista):
- `pyvistaqt` and `PySide6` — enables PyVista BackgroundPlotter (non-blocking windows)

`uv` installs the core packages automatically when running scripts. Use `uv add` for optional extras (see above).

## CLI Options

- `--config <path>`: Path to YAML configuration (defaults to `config/default.yaml`).
- `--base-dir <path>`: Overrides `paths.base_dir` in config.

## Performance

The config key `performance.numpy_threads` sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS`. Use `auto` (default) or an explicit integer. Some BLAS backends read these at import time; launching via `uv run` ensures consistent environment handling.
