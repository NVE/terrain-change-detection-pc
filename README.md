# Python Toolkit for Terrain Change Detection from Multi-temporal Point Clouds

A robust Python toolkit for detecting and quantifying terrain changes using multi-temporal LiDAR point cloud data. Developed for the Norwegian Water Resources and Energy Directorate (NVE) as a scalable alternative to raster-based DTM differencing.

## Key Features

*   **Automated Discovery**: Tools to scan and index complex directory structures from national data providers (hoydedata.no).
*   **Targeted Analysis**: Support for polygon-based clipping to focus processing on specific regions of interest (e.g., river corridors).
*   **ICP Registration**: Precise spatial alignment of multi-temporal epochs.
*   **Robust Algorithms**:
    *   **M3C2**: Accurate, normal-oriented 3D change detection (wraps the **py4dgeo** library).
    *   **DoD & C2C**: Complementary methods for analysis and validation.
*   **Scalable Architecture**: Support for out-of-core streaming and spatial tiling to process massive national-scale datasets.
*   **Flexible Configuration**: YAML-based profile system to switch between drone surveys, national datasets, and synthetic validation data.
*   **Point Cloud Native**: Computes changes directly on 3D point clouds (LAS/LAZ) to preserve fine terrain details.

## Project Structure

```text
terrain-change-detection-pc/
├── config/                 # YAML configuration profiles
├── data/                   # Data directory (raw inputs & outputs)
├── docs/                   # Documentation and guides
├── scripts/                # Entry point scripts (workflow, generators)
├── src/                    # Source code
│   └── terrain_change_detection/
│       ├── preprocessing/  # Discovery, loading, clipping
│       ├── alignment/      # ICP and coarse registration
│       ├── detection/      # M3C2 (py4dgeo wrapper), DoD, C2C
│       ├── acceleration/   # GPU and parallel processing
│       ├── visualization/  # Plotting and 3D rendering
│       └── utils/          # Configuration and helpers
└── tests/                  # Pytest suite
```

## Quick Start

### Installation

Requires Python 3.13+. We recommend using `uv` for fast dependency management.

```bash
# Clone the repository
git clone https://github.com/yaredwb/terrain-change-detection-pc.git
cd terrain-change-detection-pc

# Install dependencies
uv sync
```

### Running the Workflow

1.  **Generate Test Data** (Optional)
    To verify installation, generate a synthetic dataset with known changes:
    ```bash
    uv run scripts/generate_synthetic_laz.py
    ```

2.  **Run Processing**
    Execute the full pipeline (Discovery → Alignment → Detection → Visualization):
    ```bash
    uv run scripts/run_workflow.py --config config/profiles/synthetic.yaml
    ```

## Usage

### Configuration Profiles

The toolkit uses YAML configuration files in the `config/` directory.

| Profile | Description |
| :--- | :--- |
| **`default.yaml`** | Standard profile for in-memory processing of small-to-medium areas. |
| **`profiles/drone.yaml`** | Optimized for high-density drone LiDAR data. |
| **`profiles/large_scale.yaml`** | Enables tiling and streaming for massive datasets. |
| **`default_clipped.yaml`** | Example of polygon clipping to restrict analysis to a specific area. |

Run with a specific profile:
```bash
uv run scripts/run_workflow.py --config config/profiles/drone.yaml
```

### Data Organization

The toolkit expects data organized by "Area" and "Time Period".

**Standard Structure (hoydedata.no):**
```
data/raw/
└── my_area_name/
    ├── 2015/
    │   └── data/
    │       └── file1.laz
    └── 2020/
        └── data/
            └── file2.laz
```

**Drone Structure:**
Set `discovery.source_type: drone` in config.
```
data/raw/
└── my_drone_site/
    ├── 2023-05-01/
    │   └── flight_line.laz
    └── 2023-09-15/
        └── flight_line.laz
```

## Documentation

*   [**Configuration Guide**](docs/CONFIGURATION_GUIDE.md): Detailed reference for all YAML parameters.
*   [**Known Issues**](docs/KNOWN_ISSUES.md): Current limitations and workarounds.
*   [**Changelog**](docs/CHANGELOG.md): History of changes and updates.

## License

[Add License Information Here]
