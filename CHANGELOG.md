# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Config toggles to enable/disable individual detection methods (`dod`, `c2c`, `m3c2`).
  - New `enabled` flags in `AppConfig.detection.*` (src/terrain_change_detection/utils/config.py).
  - `scripts/run_workflow.py` now respects these flags and skips disabled steps.
- C2C vertical-plane mode for terrain-aware distances.
  - `ChangeDetector.compute_c2c_vertical_plane(...)` fits a local plane on the target and measures signed vertical offsets.
  - Configurable via `detection.c2c.mode: euclidean | vertical_plane` with `radius | k_neighbors | min_neighbors`.
- In-memory C2C 3D visualization similar to M3C2.
  - `PointCloudVisualizer.visualize_c2c_points(...)` colors source points by C2C distances.

### Changed
- Standardized comments and structure across YAML configs (default, synthetic, large_scale):
  - Consistent helper comments for sections: paths, preprocessing, discovery, alignment, detection, visualization, logging, outofcore.
  - In detection blocks, `enabled` appears first; unified wording for DoD/C2C/M3C2 helper lines.
- Default profile detection flags set to disabled by default to allow selective runs via profiles.
- Fixed try/except structure around detection steps in `scripts/run_workflow.py` (resolved SyntaxError when gating was added).

### Removed
- Reverted streaming C2C 3D point sampling/visualization experiment; streaming C2C remains histogram-only. In-memory C2C 3D rendering retained.

### Notes
- Streaming C2C continues to return distances without full coordinates to preserve memory characteristics. Visualization in streaming mode falls back to histograms.

