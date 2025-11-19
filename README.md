# Terrain Change Detection Based on Multi-temporal Point Clouds

A high-performance Python project for detecting terrain changes using multi-temporal point cloud data. The workflow includes data discovery and preprocessing, spatial alignment using ICP (Iterative Closest Point), change detection (with a primary focus on M3C2), and interactive visualizations (Plotly default; PyVista supported). DoD and C2C are implemented for comparison/baseline purposes.

## Key Features

- ✅ **Out-of-Core Processing**: Handle datasets of any size with streaming and spatial tiling
- ✅ **GPU Acceleration**: Optional CUDA-powered nearest neighbor searches for C2C operations
- ✅ **CPU Parallelization**: Multi-core processing for all change detection methods
- ✅ **Flexible Configuration**: Multiple processing modes for different dataset sizes

See [ROADMAP.md](docs/ROADMAP.md) and [PROJECT_STATUS.md](PROJECT_STATUS.md) for complete details.

## Project Structure

```
terrain-change-detection-pc/
├── config/                          # YAML configuration files
│   ├── default.yaml                 # Default in-memory configuration
│   └── profiles/
│       ├── synthetic.yaml           # Small synthetic datasets
│       ├── large_scale.yaml         # Large-scale out-of-core processing
│       └── large_synthetic.yaml     # Performance testing datasets
├── docs/                            # Comprehensive documentation
│   ├── ROADMAP.md                   # Development roadmap and achievements
│   ├── PROJECT_STATUS.md            # Current status and merge details
│   ├── CONFIGURATION_GUIDE.md       # Complete configuration reference
│   ├── GPU_SETUP_GUIDE.md           # GPU acceleration setup
│   ├── PARALLELIZATION_PLAN.md      # CPU parallelization details
│   └── ...                          # Additional technical documentation
├── scripts/                         # Workflow execution scripts
│   ├── run_workflow.py              # Main change detection workflow
│   ├── explore_data.py              # Data exploration workflow
│   ├── generate_synthetic_laz.py    # Generate test datasets
│   └── test_*.py                    # Performance benchmarking scripts
├── src/terrain_change_detection/    # Core library modules
│   ├── acceleration/                # GPU and parallel processing
│   │   ├── gpu_array_ops.py         # CuPy GPU operations
│   │   ├── gpu_neighbors.py         # GPU nearest neighbor searches
│   │   ├── parallel_executor.py     # CPU parallelization framework
│   │   ├── tiling.py                # Spatial tiling for out-of-core
│   │   └── ...
│   ├── alignment/                   # Spatial alignment algorithms
│   │   ├── coarse_registration.py   # Coarse alignment methods
│   │   ├── fine_registration.py     # ICP registration
│   │   └── streaming_alignment.py   # Out-of-core alignment
│   ├── detection/                   # Change detection algorithms
│   │   └── change_detection.py      # DoD, C2C, M3C2 implementations
│   ├── preprocessing/               # Data loading and filtering
│   │   ├── data_discovery.py        # Dataset discovery utilities
│   │   └── loader.py                # Point cloud loaders
│   ├── visualization/               # Visualization tools
│   │   └── point_cloud.py           # Point cloud visualization
│   └── utils/                       # Configuration, logging, I/O
│       ├── config.py                # Configuration management
│       ├── io.py                    # Input/output utilities
│       ├── logging.py               # Logging setup
│       └── point_cloud_filters.py   # Point cloud filtering utilities
└── tests/                           # Comprehensive test suite (144 tests)
    ├── test_gpu_*.py                # GPU acceleration tests
    ├── test_parallel_executor.py    # Parallelization tests
    ├── test_*_integration.py        # Integration tests
    └── ...
```

## Features and Workflow Overview

### Processing Modes

The pipeline supports three processing modes:

1. **In-Memory** (default): Fast processing for small-medium datasets (< 10M points)
2. **Out-of-Core Tiled**: Constant memory usage for datasets of any size
3. **Parallel**: Multi-core processing for medium-large datasets

### Pipeline Steps

1. **Data Discovery and Preprocessing**
   - Optimized for LiDAR from hoydedata.no
   - Streaming data loading with spatial filtering
   - Ground point classification filtering

2. **Spatial Alignment**
   - Optional Coarse Registration: centroid, PCA, 2D phase correlation, FPFH/RANSAC
   - ICP fine registration with multiple convergence criteria
   - Streaming alignment for large datasets

3. **Change Detection Methods**
   - **M3C2** (primary): Multi-scale Model-to-Model Cloud Comparison via py4dgeo
   - **M3C2-EP**: Error Propagation variant with Level of Detection (LoD)
   - **DoD**: DEM of Difference with multiple aggregators (mean, median, p95, p5)
   - **C2C**: Cloud-to-Cloud nearest neighbor distances (euclidean, vertical plane)

4. **Visualizations**
   - DoD heatmaps
   - Distance histograms (C2C and M3C2/M3C2-EP)
   - 3D point clouds colored by change magnitude

### Performance Features

- **GPU Acceleration**: Optional CUDA-powered C2C nearest neighbor searches
- **CPU Parallelization**: Multi-core processing for M3C2 and DoD (medium-large datasets)
- **Out-of-Core Processing**: Handle large datasets with constant memory usage
- **Spatial Tiling**: Configurable tile size with halo for edge handling

**Notes**:
- M3C2 parameters can be auto-tuned from point cloud density or manually configured
- GPU and parallel processing cannot run simultaneously (CUDA fork limitation)
- See [CONFIGURATION_GUIDE.md](docs/CONFIGURATION_GUIDE.md) for detailed settings

## Getting Started

This project requires `uv` to be installed on your system for dependency management and script execution.

### Prerequisites

**Required**:
- Python 3.13+
- `uv` package manager (https://docs.astral.sh/uv/)

**Optional (for GPU acceleration)**:
- NVIDIA GPU with CUDA support (Compute Capability 6.0+)
- CUDA Toolkit 12.x
- 4+ GB GPU memory (8+ GB recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yaredwb/terrain-change-detection-pc.git
cd terrain-change-detection-pc
```

2. Install core dependencies (automatic on first `uv run`):
```bash
uv sync
```

3. **Optional**: Install GPU dependencies for acceleration:
```bash
uv sync --extra gpu
```

See [GPU_SETUP_GUIDE.md](docs/GPU_SETUP_GUIDE.md) for detailed GPU setup instructions.

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

The project uses YAML configuration files with sensible defaults and multiple profiles:

**Configuration Files**:
- `config/default.yaml` - In-memory processing (default)
- `config/profiles/synthetic.yaml` - Small synthetic datasets
- `config/profiles/large_scale.yaml` - Large-scale out-of-core processing with GPU
- `config/profiles/large_synthetic.yaml` - Performance testing

**Key Configuration Sections**:

- **Paths & Preprocessing**: Dataset location, ground filtering, classification codes
- **Alignment**: ICP parameters, coarse registration method, convergence criteria
- **Detection**: Enable/configure DoD, C2C, and M3C2 methods
- **Out-of-Core**: Tile size, halo width, streaming mode (for large datasets)
- **Parallel**: CPU parallelization, worker count, memory limits
- **GPU**: GPU acceleration, memory limits, which operations to accelerate
- **Visualization**: Backend (plotly/pyvista), sample size for display
- **Logging**: Level, file output

**Important**: GPU and parallel processing cannot run simultaneously due to CUDA fork limitations. See configuration warnings in the files.

For complete configuration reference, see [CONFIGURATION_GUIDE.md](docs/CONFIGURATION_GUIDE.md).

**Example Configurations**:

```bash
# Small datasets - in-memory processing
uv run scripts/run_workflow.py --config config/profiles/synthetic.yaml

# Large datasets - out-of-core with GPU acceleration
uv run scripts/run_workflow.py --config config/profiles/large_scale.yaml

# Medium datasets - CPU parallelization
uv run scripts/run_workflow.py --config config/default.yaml  # with parallel.enabled: true
```

**Quick Configuration Example**:

```yaml
# For medium datasets (10-50M points) with CPU parallelization
outofcore:
  enabled: true
  tile_size_m: 500.0
  
parallel:
  enabled: true
  n_workers: null  # auto-detect

gpu:
  enabled: false  # Cannot use with parallel

# For large datasets (50M+ points) with GPU
outofcore:
  enabled: true
  
parallel:
  enabled: false  # Cannot use with GPU
  
gpu:
  enabled: true
  use_for_c2c: true
```

### Synthetic Datasets

Generate synthetic datasets for testing and validation:

**Small Synthetic Dataset** (~100K points):
```bash
uv run scripts/generate_synthetic_laz.py
```
Output: `data/synthetic/synthetic_area/{2015,2020}/data/synthetic_tile_01.laz`

**Large Synthetic Dataset** (configurable size for performance testing):
```bash
uv run scripts/generate_large_synthetic_laz.py
```
Output: `data/large_synthetic/large_area/` with multiple time periods

These datasets include:
- Known deformation patterns (deposition/erosion)
- Intentional misalignment for testing coarse registration
- Ground truth for validation

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

### Core Dependencies (Automatic)

Installed automatically via `uv sync`:
- `laspy` ~= 2.5.4 — LAZ/LAS I/O
- `lazrs` — Fast LAZ compression
- `py4dgeo` — M3C2 algorithms
- `plotly` — Interactive plotting (default backend)
- `pyvista` — Optional 3D plotting backend
- `scikit-learn` — KD-tree and ML utilities
- `numpy`, `scipy`, `matplotlib` — Scientific computing
- `pydantic`, `PyYAML` — Configuration management

### Optional: GPU Acceleration

Install with `uv sync --extra gpu`:
- `cupy-cuda12x` >= 13.0.0 — GPU arrays (CUDA 12.x)
- `numba` >= 0.59.0 — JIT compilation with CUDA support
- `cuml` >= 24.0.0 — GPU ML algorithms (Linux/WSL2 only)

**Note**: GPU dependencies require NVIDIA GPU with CUDA 12.x toolkit. See [GPU_SETUP_GUIDE.md](docs/GPU_SETUP_GUIDE.md).

### Optional: Enhanced Visualization

For non-blocking PyVista windows:
```bash
uv add pyvistaqt PySide6
```

## GPU Acceleration (Optional)

### Overview

GPU acceleration is available for certain compute-intensive operations:

| Operation | GPU Support | Notes |
|-----------|-------------|-------|
| C2C Nearest Neighbors | ✅ Supported | Recommended for large point clouds (1M+ points) |
| DoD Grid Accumulation | ⚠️ Limited benefit | Memory-bound operation |
| ICP Alignment | ❌ Not recommended | Stability issues with cuML |

### Setup

1. **Requirements**:
   - NVIDIA GPU (Compute Capability 6.0+, Pascal or newer)
   - CUDA Toolkit 12.x
   - 4+ GB GPU memory (8+ GB recommended)

2. **Install GPU dependencies**:
   ```bash
   uv sync --extra gpu
   ```

3. **Activate GPU libraries** (required before running):
   ```bash
   source activate_gpu.sh
   ```

4. **Enable in configuration**:
   ```yaml
   gpu:
     enabled: true
     use_for_c2c: true          # Recommended for large datasets
     use_for_preprocessing: true # Optional (marginal benefit)
     use_for_alignment: false   # Not recommended (unstable)
   ```

### Important Limitations

⚠️ **GPU and parallel processing cannot run simultaneously** due to CUDA fork limitations. Choose one:
- **Medium datasets (10-50M points)**: Use `parallel.enabled: true`, GPU disabled
- **Large datasets (50M+ points)**: Use `gpu.enabled: true`, parallel disabled

### Automatic Fallback

The system automatically falls back to CPU if:
- GPU libraries not installed
- GPU memory insufficient
- CUDA initialization fails
- GPU operations produce invalid results

Results are numerically identical (CPU vs GPU) with tolerance < 1e-5.

For complete setup instructions, see [GPU_SETUP_GUIDE.md](docs/GPU_SETUP_GUIDE.md).

## CLI Options

- `--config <path>`: Path to YAML configuration (defaults to `config/default.yaml`).
- `--base-dir <path>`: Overrides `paths.base_dir` in config.

## Performance Optimization

### Scaling Recommendations

**Small Datasets (< 10M points)**:
- Use default in-memory processing
- No parallelization needed (overhead dominates)
- GPU optional (launch overhead for small data)

**Medium Datasets (10-50M points)**:
- Enable out-of-core: `outofcore.enabled: true`
- Enable CPU parallelization: `parallel.enabled: true`
- Performance improvement for parallel operations

**Large Datasets (50M+ points)**:
- Enable out-of-core: `outofcore.enabled: true`
- Enable GPU acceleration: `gpu.enabled: true`
- Significant speedup for C2C operations
- Constant memory usage regardless of size

### Thread Control

Configure NumPy/BLAS thread count:
```yaml
performance:
  numpy_threads: auto  # or explicit integer (e.g., 4)
```

Sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS`. Launching via `uv run` ensures consistent environment.

### Benchmarking

Performance testing scripts available:
```bash
# C2C GPU performance
uv run scripts/test_gpu_c2c_performance.py

# DoD CPU vs GPU comparison
uv run scripts/compare_cpu_gpu_dod.py

# ICP alignment performance
uv run scripts/test_icp_alignment_performance.py
```

See [GPU_PERFORMANCE_COMPARISON.md](docs/GPU_PERFORMANCE_COMPARISON.md) for detailed benchmarks.

## Documentation

Comprehensive documentation available in `docs/`:

- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Current status, v2.0.0 summary, merge details
- **[ROADMAP.md](docs/ROADMAP.md)** - Development roadmap and achievements
- **[CONFIGURATION_GUIDE.md](docs/CONFIGURATION_GUIDE.md)** - Complete configuration reference
- **[GPU_SETUP_GUIDE.md](docs/GPU_SETUP_GUIDE.md)** - GPU acceleration setup and troubleshooting
- **[PARALLELIZATION_PLAN.md](docs/PARALLELIZATION_PLAN.md)** - CPU parallelization architecture
- **[GPU_PERFORMANCE_COMPARISON.md](docs/GPU_PERFORMANCE_COMPARISON.md)** - Performance benchmarks
- **[ALGORITHMS.md](docs/ALGORITHMS.md)** - Algorithm explanations (DoD, C2C, M3C2)
- **[CHANGELOG.md](docs/CHANGELOG.md)** - Detailed change history

## Testing

Run the comprehensive test suite (145 tests):

```bash
# All tests (132 pass, 13 skipped integration tests)
uv run pytest

# With GPU libraries activated
source activate_gpu.sh
uv run pytest  # All tests pass

# Specific test categories
uv run pytest tests/test_gpu_*.py -v          # GPU tests
uv run pytest tests/test_parallel_*.py -v     # Parallelization tests
uv run pytest tests/test_*_integration.py -v  # Integration tests
```

**Note**: 13 integration tests are skipped when test data is not available. These tests validate workflows with real LAZ files and can be run when sample data is present in `tests/sample_data/`.

## Contributing

This project is currently in active development. For questions or issues, please open a GitHub issue.

## License

[Add license information]

## Acknowledgments

- py4dgeo for M3C2 implementation
- RAPIDS/cuML for GPU acceleration
- hoydedata.no for LiDAR data standards
