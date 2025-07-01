# Terrain Change Detection Based on Multi-temporal Point Clouds

A Python project for detecting terrain changes using multi-temporal point cloud data. The workflow includes data discovery and preprocessing, spatial alignment using ICP (Iterative Closest Point), change detection, and visualization capabilities.

## Project Structure

```
terrain-change-detection-pc/
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

## Workflow Overview

The terrain change detection process follows these main steps:

1. **Data Discovery and Preprocessing**: Locate and prepare multi-temporal point cloud data, optimized for LiDAR point cloud data from hoydedata.no.
2. **Spatial Alignment**: Co-register and spatially align point clouds from different time periods using the ICP algorithm.
3. **Change Detection**: Identify and quantify significant terrain changes (TODO)
4. **Visualization**: Generate visual representations of detected changes (TODO)

## Getting Started

This project requires `uv` to be installed on your system for dependency management and script execution.

### Prerequisites
- Python 3.13+
- `uv` package manager

### Installation

Clone the repository:
```bash
git clone <repository-url>
cd terrain-change-detection-pc
```

## Running Scripts

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

### Main Workflow

Execute the complete change detection pipeline:

```bash
# Run the main change detection workflow (only preprocessing and alignment are implemented, change detection and visualization are TODOs)
uv run scripts/run_workflow.py

# Or run the main entry point (Not ready yet)
uv run main.py
```

## Dependencies

- `laspy>=2.5.4` - Reading and writing LAZ/LAS point cloud files
- `py4dgeo>=0.7.0` - 4D geospatial analysis and change detection algorithms

Additional dependencies will be installed automatically by `uv` as needed.
