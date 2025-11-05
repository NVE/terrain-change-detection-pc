# Changelog and Implementation Notes — External Configuration System

Date: 2025-11-05

This document summarizes the new features, improvements, and behavior changes introduced with the addition of a typed, external configuration system to the terrain-change-detection project.

## Overview

We externalized central pipeline parameters into a human-readable YAML configuration, validated by a typed schema (pydantic). The main workflow now reads parameters from YAML while keeping safe defaults. This makes it easier to tune analyses without code changes and prepares the project for future UI-based configuration.

## Highlights

- YAML configuration with typed validation (pydantic) and sensible defaults.
- New config files: `config/default.yaml` and `config/profiles/synthetic.yaml`.
- Workflow accepts `--config` and `--base-dir` CLI flags.
- Preprocessing and discovery stages are configurable (ground filtering; subfolder names).
- Logging level/file, visualization backend/sample size, and performance threads are configurable.
- Clearer loader logging: explicitly reports “ground points” when ground filtering is enabled.

## New Features

1) External Configuration (YAML)
- Default config auto-loaded from `config/default.yaml`.
- Example profile: `config/profiles/synthetic.yaml`.
- Typed schema defined in `src/terrain_change_detection/utils/config.py` using pydantic.

2) CLI Enhancements
- `--config <path>`: specify YAML file path (optional).
- `--base-dir <path>`: override `paths.base_dir` from YAML.

3) Configurable Preprocessing and Discovery
- Preprocessing:
  - `preprocessing.ground_only` (default true).
  - `preprocessing.classification_filter` (default `[2]`).
- Data discovery:
  - `discovery.data_dir_name` and `discovery.metadata_dir_name` (defaults `data`/`metadata`).
- Shared loader instance is injected into discovery and batch loading to keep behavior consistent.

4) Configurable Algorithms and Visualization
- Alignment (ICP): `alignment.{max_iterations,tolerance,max_correspondence_distance,subsample_size}`.
- DoD: `detection.dod.{cell_size,aggregator}`.
- C2C: `detection.c2c.{max_points,max_distance}`.
- M3C2: `detection.m3c2.core_points`, `detection.m3c2.autotune.*`, `detection.m3c2.ep.workers`.
- Visualization: `visualization.{backend,sample_size}` (`plotly`/`pyvista`/`pyvistaqt`).

5) Logging and Performance
- Logging: `logging.{level,file}`. Console logging remains enabled; optional file output can be enabled.
- Performance threads: `performance.numpy_threads` sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS`.

## Improvements

- Loader info logs now explicitly state “ground points” when `ground_only=True`; also warns if no ground points are found.
- Aligned-cloud visualization and other steps now use config-driven sample sizes and parameters (no hardcoded constants).
- Code paths constructed from config reduce the need to edit scripts when switching datasets.

## Backward Compatibility

- Default behavior is unchanged when run without flags: the workflow reads `config/default.yaml` which mirrors prior defaults.
- CLI `--base-dir` continues to work and overrides YAML for that setting.
- All modules retain function defaults for direct programmatic use.

## Usage Changes (Summary)

Run with defaults:
```powershell
uv run scripts/run_workflow.py
```

Run with explicit config:
```powershell
uv run scripts/run_workflow.py --config config/default.yaml
```

Override dataset root via CLI:
```powershell
uv run scripts/run_workflow.py --base-dir data/synthetic
```

Use profile for synthetic data:
```powershell
uv run scripts/run_workflow.py --config config/profiles/synthetic.yaml
```

## Configuration Keys (Quick Reference)

- paths
  - base_dir: dataset root folder (default `data/raw`).
- preprocessing
  - ground_only: filter to ground classes (default true).
  - classification_filter: list of LAS class codes to keep (default `[2]`).
- discovery
  - data_dir_name: name of the subfolder that contains LAZ/LAS files (default `data`).
  - metadata_dir_name: name of the metadata subfolder (default `metadata`).
- alignment
  - max_iterations, tolerance, max_correspondence_distance, subsample_size
- detection.dod
  - cell_size, aggregator (`mean|median|p95|p5`).
- detection.c2c
  - max_points, max_distance (optional cap).
- detection.m3c2
  - core_points
  - autotune: target_neighbors, max_depth_factor, min_radius, max_radius
  - ep: workers (null = OS-specific default: 1 on Windows, 4 elsewhere)
- visualization
  - backend (`plotly|pyvista|pyvistaqt`), sample_size
- logging
  - level (`DEBUG|INFO|WARNING|ERROR|CRITICAL`), file (optional path)
- performance
  - numpy_threads: `auto` (default) or an integer

## File Changes

- pyproject.toml
  - Added dependencies: `pydantic`, `PyYAML`.
- config/default.yaml
  - New default configuration file.
- config/profiles/synthetic.yaml
  - Example profile for synthetic datasets.
- src/terrain_change_detection/utils/config.py
  - New: typed AppConfig model and `load_config` loader for YAML.
- scripts/run_workflow.py
  - Integrated config loading and CLI flags; replaced hardcoded constants with config values; applied logging/performance settings.
- src/terrain_change_detection/preprocessing/loader.py
  - Loader now accepts `ground_only` and `classification_filter`; info log explicitly mentions “ground points” when applicable.
- src/terrain_change_detection/preprocessing/data_discovery.py
  - Discovery accepts `data_dir_name` and `metadata_dir_name`; uses injected `PointCloudLoader`. `BatchLoader` accepts a loader instance.
- README.md
  - Updated with configuration usage, keys, and examples.

## Upgrade Guide

1) No code changes are required for typical usage — continue running the workflow as before.
2) To customize behavior, edit `config/default.yaml` or point to an alternative file with `--config`.
3) Override dataset root at the CLI with `--base-dir` for ad hoc runs.
4) On Windows, M3C2-EP worker processes default to 1 when not specified; set `detection.m3c2.ep.workers` to override.

## Known Considerations

- Thread environment variables may be read by numerical libraries at import time. While we set them early in the script, some backends might require setting these before the Python process starts to take full effect.
- Large datasets can be memory intensive; tune `alignment.subsample_size`, `detection.c2c.max_points`, and `detection.m3c2.core_points` accordingly.

## Next Steps (Optional)

- Add environment variable overrides (e.g., `TCD_paths__base_dir`) for containerized deployments.
- Add small config parsing tests and a smoke test for config-driven parameters.
- Consider exposing config parameters through a future web UI.

