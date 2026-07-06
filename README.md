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
*   **Flexible Configuration**: One canonical `config/default.yaml`, repeatable `--set` overrides, and optional small override presets for common dataset types.
*   **Point Cloud Native**: Computes changes directly on 3D point clouds (LAS/LAZ) to preserve fine terrain details.

## Project Structure

```text
terrain-change-detection-pc/
├── config/                 # Canonical default config + small override presets
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
git clone https://github.com/NVE/terrain-change-detection-pc.git
cd terrain-change-detection-pc

# Install dependencies
uv sync
```

For managed devices (without admin rights) using conda/Anaconda, do the following:
```bash
# Clone the repository
git clone https://github.com/NVE/terrain-change-detection-pc.git
cd terrain-change-detection-pc
conda create --name myenv python=3.13
conda activate myenv
pip install uv
uv pip install -r pyproject.toml
# Further installs with uv pip install <package_name> as needed
```


### Running the Workflow

1.  **Generate Test Data** (Optional)
    To verify installation, generate a synthetic dataset with known changes:
    ```bash
    uv run scripts/generate_synthetic_laz.py
    ```
    or
    ```bash
    # conda activate myenv
    python scripts/generate_synthetic_laz.py
    ```

2.  **Run Processing**
    Execute the full pipeline (Discovery → Alignment → Detection → Visualization):
    ```bash
    uv run scripts/run_workflow.py --config config/profiles/synthetic.yaml
    ```
    or
    ```bash
    # conda activate myenv
    python scripts/run_workflow.py --config config/profiles/synthetic.yaml
    ```

## Usage

### Configuration

The toolkit always starts from `config/default.yaml`. You can then:

- add one or more small override YAMLs with `--config`
- override individual values with `--set section.key=value`

Common override presets live in `config/profiles/`.

`paths.output_crs` is not a reprojection target. It is only used as fallback CRS metadata when input LAZ/LAS files do not declare a CRS; outputs are otherwise written in the detected input CRS.

| Config | Description |
| :--- | :--- |
| **`default.yaml`** | Full canonical config with all runtime defaults. |
| **`profiles/drone.yaml`** | Minimal override preset for high-density drone LiDAR data. |
| **`profiles/large_scale.yaml`** | Minimal override preset for large out-of-core runs. |
| **`default_clipped.yaml`** | Minimal override that enables polygon clipping. |

Run with CLI overrides only:
```bash
uv run scripts/run_workflow.py --set paths.base_dir=data --set discovery.source_type=drone --area-name Jeksla --years 2024 2025
```

Run with an override preset:
```bash
uv run scripts/run_workflow.py --config config/profiles/drone.yaml --set paths.base_dir=data --area-name Jeksla --years 2024 2025
```

You can combine both styles:
```bash
uv run scripts/run_workflow.py --config config/profiles/drone.yaml --set alignment.enabled=false
```

Run a specific area and time period:
```bash
uv run scripts/run_workflow.py --config config/profiles/drone.yaml --area-name Ristvassdrag --years 2017 2025
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

*   [**Best Practice Guide**](docs/BEST_PRACTICES_GUIDE.md): Guidance on what settings to start with, when to override them, and how to validate results.
*   [**Best-Practice Evidence**](docs/BEST_PRACTICES_EVIDENCE.md): Short summary of the repo runs and datasets used to support the guide.
*   [**Configuration Guide**](docs/CONFIGURATION_GUIDE.md): Detailed reference for all YAML parameters.
*   [**Known Issues**](docs/KNOWN_ISSUES.md): Current limitations and workarounds.
*   [**Changelog**](docs/CHANGELOG.md): History of changes and updates.

## License

This project is licensed under the [MIT License](LICENSE).

## Citation

If you use this software in your research, please cite it using the metadata in [`CITATION.cff`](CITATION.cff), or click the **"Cite this repository"** button on GitHub.
