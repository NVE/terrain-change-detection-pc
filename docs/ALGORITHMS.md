# Algorithms: DoD, C2C, M3C2 — Tiling and Streaming

This document explains how the project’s out-of-core tiling and streaming primitives relate to three core change-detection approaches:

- DEM of Difference (DoD): Gridded DEMs per epoch, then subtraction.
- Cloud-to-Cloud (C2C): Point-based nearest-neighbor or radius-bounded distances between epochs.
- Multiscale Model to Model Cloud Comparison (M3C2): Point-based, normal-oriented distances between epochs.

The goal is to clarify which components are algorithm-agnostic infrastructure (used by all methods) versus which are DoD‑specific.

## Shared Infrastructure (Used by All Methods)

- Tiling with halos: `Tiler`/`Tile` (inner/outer bounds) in `src/terrain_change_detection/acceleration/tiling.py`.
  - Inner bounds: the area where results are produced for a tile.
  - Outer bounds (halo): a buffer ensuring neighborhoods near tile edges have sufficient context.
- Out-of-core streaming: `LaspyStreamReader` for chunked LAZ/LAS reading with bbox + classification filters.
- Spatial extents: `Bounds2D` and `union_bounds()` to define and combine processing domains across epochs.

These primitives keep memory usage bounded by processing spatial subsets and streaming points in manageable chunks.

## DoD (DEM of Difference)

DoD produces gridded DEMs (e.g., mean Z per cell) for each epoch and computes their difference.

- DoD‑specific components:
  - `GridAccumulator`: Streams points into a fixed XY grid, maintaining per‑cell sums and counts to form a DEM via mean.
  - `MosaicAccumulator`: Mosaics per‑tile DEMs (with halos) into a seamless global DEM by averaging overlaps.
- Typical workflow:
  1. Determine global bounds (often the union of both epochs).
  2. For each epoch separately, tile the domain with a halo and stream points per tile.
  3. Within each tile, accumulate to grid (e.g., mean) → tile DEM → add to mosaic.
  4. Finalize both mosaics, then compute DoD = DEM2 − DEM1.
- Classification guidance: Terrain DoD typically uses ground‑only (class 2) filtering.
- Halo guidance: Choose halo ≥ the gridding kernel/interpolation radius so tile edges have consistent support.

Notes:
- The current `GridAccumulator` targets a mean DEM prototype (CPU‑only). Other reducers (median/percentiles) can be layered similarly.

## C2C (Cloud-to-Cloud)

C2C computes distances from each source point to the nearest target point, typically within a bounded radius to keep neighborhoods finite for streaming. It does not require DEM gridding or mosaicking.

- Reused infrastructure:
  - `Tiler`/`Tile` with halos: Tile over the source domain; use outer bounds to stream target points that can serve as neighbors.
  - `LaspyStreamReader`: Stream source and target points per tile, filtered to the tile’s outer bbox.
  - `Bounds2D`/`union_bounds()`: Define spatial extents when needed.
- Algorithm specifics (outside `tiling.py`):
  - Build a local spatial index per tile for target points (e.g., k-d tree) and query distances for source points in the inner tile.
  - Prefer a finite `max_distance` (search radius) to make streaming viable; unbounded KNN requires global indexing.
  - Optionally compute summary stats (RMSE/mean/median) and track indices if memory permits.
- Classification guidance: Choose classes relevant to your surface of interest; not necessarily ground-only.
- Halo guidance: Use halo ≥ `max_distance` so neighborhoods for inner-tile sources are complete near edges.

## M3C2 (Point-based, normal-oriented)

M3C2 computes signed distances along normals at a set of core (base) points.
It does not require DEM gridding or mosaicking.

- Reused infrastructure:
  - `Tiler`/`Tile` with halos: Tile the XY domain of core points. Use outer bounds to fetch neighbors from both epochs.
  - `LaspyStreamReader`: Stream points within each tile’s outer bbox for memory efficiency.
  - `Bounds2D`/`union_bounds()`: Define spatial extents when needed.
- Algorithm specifics (outside `tiling.py`):
  - Choose core points (e.g., a subset of one epoch or a uniform sampling).
  - Estimate local surface normals (projection scale / normal scale).
  - For each core point, gather neighbors from both epochs within a cylinder radius and depth along the normal.
  - Compute signed distances and (optionally) uncertainty/significance.
- Classification guidance: Depends on the surface of interest; often not ground‑only.
- Halo guidance: Use halo ≥ max(projection_scale, cylinder_radius) so neighborhoods near tile edges remain complete.

## Choosing Extents

- DoD: Grid the union extent so both DEMs align and differences can be computed everywhere; expect NaNs where one epoch lacks support.
- C2C: Tile the source domain (or union) and evaluate only source points within the inner tile; use a halo ≥ `max_distance` to fetch target neighbors.
- M3C2: Compute only for core points (e.g., sampled from a reference epoch); tile over XY as a processing convenience.

## Where in the Code

- Tiling and streaming primitives: `src/terrain_change_detection/acceleration/tiling.py`
  - `Tiler`, `Tile`, `LaspyStreamReader`, `Bounds2D`, `union_bounds()` → shared.
  - `GridAccumulator`, `MosaicAccumulator` → DoD‑specific (gridded DEM workflows).
- Change detection APIs and results: `src/terrain_change_detection/detection/change_detection.py`

## Quick Decision Guide

- You want a raster map of elevation changes over an area → Use DoD (grid each epoch, subtract).
- You want nearest-neighbor distances between clouds (radius-bounded) → Use C2C (point-based neighborhoods, no DEM mosaic).
- You want distances at points along locally estimated surface normals → Use M3C2 (point‑based neighborhoods, no DEM mosaic).
