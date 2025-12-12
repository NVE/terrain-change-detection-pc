"""
DEM of Difference (DoD) Change Detection

This module implements DoD algorithms for detecting vertical changes between
multi-temporal point cloud datasets using DEM differencing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Literal, List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..utils.coordinate_transform import LocalCoordinateTransform

from ..utils.logging import setup_logger
from ..acceleration import (
    GridAccumulator,
    LaspyStreamReader,
    Bounds2D,
    union_bounds,
    MosaicAccumulator,
    Tile,
    Tiler,
)
from ..utils.config import AppConfig

logger = setup_logger(__name__)


@dataclass
class DoDResult:
    """
    Result of DEM of Difference (DoD).

    Attributes:
        grid_x: 2D array of X coordinates (meshgrid)
        grid_y: 2D array of Y coordinates (meshgrid)
        dem1: 2D array of DEM at time T1
        dem2: 2D array of DEM at time T2 (aligned to T1)
        dod: 2D array of DoD = dem2 - dem1
        cell_size: Grid resolution used (meters)
        bounds: (min_x, min_y, max_x, max_y) used for gridding
        stats: Summary statistics (e.g., mean_change, positive_area, negative_area)
        metadata: Optional metadata dictionary
    """

    grid_x: np.ndarray
    grid_y: np.ndarray
    dem1: np.ndarray
    dem2: np.ndarray
    dod: np.ndarray
    cell_size: float
    bounds: Tuple[float, float, float, float]
    stats: Dict[str, float]
    metadata: Optional[Dict] = None


class DoDDetector:
    """DEM of Difference (DoD) change detection methods."""

    @staticmethod
    def compute_dod(
        points_t1: np.ndarray,
        points_t2: np.ndarray,
        cell_size: float = 1.0,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        aggregator: Literal["mean", "median", "p95", "p5"] = "mean",
        config: Optional[AppConfig] = None,
    ) -> DoDResult:
        """
        Compute DEM of Difference (DoD) by gridding the point clouds and differencing.

        Args:
            points_t1: Ground points at time T1 (N x 3)
            points_t2: Ground points at time T2 (aligned to T1) (M x 3)
            cell_size: Grid cell size in same units as coordinates (meters)
            bounds: Optional (min_x, min_y, max_x, max_y). If None, use union bounds.
            aggregator: Aggregation method for DEM: mean, median, p95, p5
            config: Optional configuration (for GPU settings)

        Returns:
            DoDResult with DEMs and difference grid.
        """
        if points_t1.size == 0 or points_t2.size == 0:
            raise ValueError("Input point arrays must be non-empty.")
        
        # Note: In-memory DoD uses bucket-based aggregation which only supports CPU
        # For GPU-accelerated DoD, use streaming methods (compute_dod_streaming_files*)
        logger.info(f"DoD (in-memory) using CPU-only backend (aggregator: {aggregator}, GPU not supported for in-memory DoD)")

        if bounds is None:
            min_x = float(min(points_t1[:, 0].min(), points_t2[:, 0].min()))
            max_x = float(max(points_t1[:, 0].max(), points_t2[:, 0].max()))
            min_y = float(min(points_t1[:, 1].min(), points_t2[:, 1].min()))
            max_y = float(max(points_t1[:, 1].max(), points_t2[:, 1].max()))
            bounds = (min_x, min_y, max_x, max_y)

        min_x, min_y, max_x, max_y = bounds
        nx = int(np.ceil((max_x - min_x) / cell_size)) + 1
        ny = int(np.ceil((max_y - min_y) / cell_size)) + 1

        # Create grid coordinates
        x_edges = np.linspace(min_x, min_x + nx * cell_size, nx + 1)
        y_edges = np.linspace(min_y, min_y + ny * cell_size, ny + 1)
        grid_x, grid_y = np.meshgrid(
            (x_edges[:-1] + x_edges[1:]) / 2.0,
            (y_edges[:-1] + y_edges[1:]) / 2.0,
        )

        def grid_dem(points: np.ndarray) -> np.ndarray:
            # Bin points into grid cells
            xi = np.clip(((points[:, 0] - min_x) / cell_size).astype(int), 0, nx - 1)
            yi = np.clip(((points[:, 1] - min_y) / cell_size).astype(int), 0, ny - 1)

            # Aggregate Z by cell using chosen reducer
            dem = np.full((ny, nx), np.nan, dtype=float)
            # Use lists of z-values per cell (sparse accumulation)
            buckets: Dict[Tuple[int, int], list] = {}
            for xidx, yidx, z in zip(xi, yi, points[:, 2]):
                buckets.setdefault((yidx, xidx), []).append(z)

            if aggregator == "mean":
                reducer = np.mean
            elif aggregator == "median":
                reducer = np.median
            elif aggregator == "p95":
                reducer = lambda zs: np.percentile(zs, 95)
            elif aggregator == "p5":
                reducer = lambda zs: np.percentile(zs, 5)
            else:
                raise ValueError(f"Unknown aggregator: {aggregator}")

            for (yidx, xidx), zs in buckets.items():
                dem[yidx, xidx] = reducer(zs)
            return dem

        dem1 = grid_dem(points_t1)
        dem2 = grid_dem(points_t2)

        # Difference (preserve NaNs where either DEM is NaN)
        dod = dem2 - dem1

        # Basic stats over valid cells
        valid = np.isfinite(dod)
        stats: Dict[str, float] = {
            "n_cells": int(valid.sum()),
            "mean_change": float(np.nanmean(dod)),
            "median_change": float(np.nanmedian(dod)),
            "rmse": float(np.sqrt(np.nanmean(np.square(dod)))),
            "min_change": float(np.nanmin(dod)),
            "max_change": float(np.nanmax(dod)),
        }

        # Log completion with statistics
        logger.info(
            "DoD completed: n_cells=%d, mean=%.4f m, median=%.4f m, rmse=%.4f m, range=[%.4f, %.4f] m",
            stats["n_cells"],
            stats["mean_change"],
            stats["median_change"],
            stats["rmse"],
            stats["min_change"],
            stats["max_change"],
        )

        return DoDResult(
            grid_x=grid_x,
            grid_y=grid_y,
            dem1=dem1,
            dem2=dem2,
            dod=dod,
            cell_size=cell_size,
            bounds=bounds,
            stats=stats,
            metadata={"aggregator": aggregator},
        )

    @staticmethod
    def compute_dod_streaming_files(
        files_t1: list[str],
        files_t2: list[str],
        cell_size: float = 1.0,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        *,
        ground_only: bool = True,
        classification_filter: Optional[list[int]] = None,
        chunk_points: int = 1_000_000,
        config: Optional[AppConfig] = None,
    ) -> DoDResult:
        """
        Streaming DoD that reads LAS/LAZ files in chunks and computes mean-based DEMs.

        Args:
            files_t1: LAZ/LAS files for epoch 1
            files_t2: LAZ/LAS files for epoch 2
            cell_size: Grid cell size in data units
            bounds: Optional (min_x, min_y, max_x, max_y). If None, use union bounds.
            ground_only: Apply ground/classification filter
            classification_filter: Optional classification codes
            chunk_points: Points per streaming chunk
            config: Optional configuration (for GPU settings)

        Returns:
            DoDResult with DEMs and difference grid

        Notes:
        - Aggregator is mean-only in this prototype (streaming-friendly).
        - bounds defaults to the union of LAS headers.
        """
        if not files_t1 or not files_t2:
            raise ValueError("compute_dod_streaming_files requires non-empty file lists")

        # Determine if we should use GPU (config-driven)
        use_gpu = False
        if config is not None and hasattr(config, "gpu"):
            use_gpu = config.gpu.enabled and config.gpu.use_for_dod
        
        # Log which backend is being used with clear explanation
        if use_gpu:
            logger.info("DoD streaming files using GPU-accelerated grid accumulation")
        else:
            logger.info(
                "DoD streaming files using CPU grid accumulation (GPU disabled or not available)"
            )

        if bounds is None:
            bounds2d = union_bounds(files_t1, files_t2)
            bounds_tuple = (bounds2d.min_x, bounds2d.min_y, bounds2d.max_x, bounds2d.max_y)
        else:
            min_x, min_y, max_x, max_y = bounds
            bounds2d = Bounds2D(min_x, min_y, max_x, max_y)
            bounds_tuple = bounds

        acc1 = GridAccumulator(bounds2d, cell_size, use_gpu=use_gpu)
        acc2 = GridAccumulator(bounds2d, cell_size, use_gpu=use_gpu)

        reader1 = LaspyStreamReader(files_t1, ground_only=ground_only, classification_filter=classification_filter, chunk_points=chunk_points)
        reader2 = LaspyStreamReader(files_t2, ground_only=ground_only, classification_filter=classification_filter, chunk_points=chunk_points)

        for pts in reader1.stream_points(bounds2d):
            acc1.accumulate(pts)
        for pts in reader2.stream_points(bounds2d):
            acc2.accumulate(pts)

        dem1 = acc1.finalize()
        dem2 = acc2.finalize()
        dod = dem2 - dem1
        valid = np.isfinite(dod)
        stats: Dict[str, float] = {
            "n_cells": int(valid.sum()),
            "mean_change": float(np.nanmean(dod)),
            "median_change": float(np.nanmedian(dod)),
            "rmse": float(np.sqrt(np.nanmean(np.square(dod)))),
            "min_change": float(np.nanmin(dod)),
            "max_change": float(np.nanmax(dod)),
        }

        # Log completion with statistics
        logger.info(
            "DoD completed: n_cells=%d, mean=%.4f m, median=%.4f m, rmse=%.4f m, range=[%.4f, %.4f] m",
            stats["n_cells"],
            stats["mean_change"],
            stats["median_change"],
            stats["rmse"],
            stats["min_change"],
            stats["max_change"],
        )

        return DoDResult(
            grid_x=acc1.grid_x,
            grid_y=acc1.grid_y,
            dem1=dem1,
            dem2=dem2,
            dod=dod,
            cell_size=float(cell_size),
            bounds=bounds_tuple,
            stats=stats,
            metadata={"aggregator": "mean", "streaming": True},
        )

    @staticmethod
    def compute_dod_streaming_files_tiled(
        files_t1: list[str],
        files_t2: list[str],
        cell_size: float,
        tile_size: float,
        halo: float,
        *,
        ground_only: bool = True,
        classification_filter: Optional[list[int]] = None,
        chunk_points: int = 1_000_000,
        memmap_dir: Optional[str] = None,
        transform_t2: Optional[np.ndarray] = None,
        config: Optional[AppConfig] = None,
    ) -> DoDResult:
        """
        Out-of-core tiled DoD by streaming LAS/LAZ files in spatial tiles.
        Out-of-core tiled DoD (mean). Tiles are grid-aligned; overlapping contributions are averaged.
        """
        from ..acceleration.tiling import LaspyStreamReader, scan_las_bounds, bounds_intersect

        if not files_t1 or not files_t2:
            raise ValueError("compute_dod_streaming_files_tiled requires non-empty file lists")
        
        # Determine if we should use GPU (config-driven)
        use_gpu = False
        if config is not None and hasattr(config, "gpu"):
            use_gpu = config.gpu.enabled and config.gpu.use_for_dod
        
        # Log which backend is being used with clear explanation
        if use_gpu:
            logger.info("DoD tiled streaming using GPU-accelerated grid accumulation")
        else:
            logger.info(
                "DoD tiled streaming using CPU grid accumulation (GPU disabled or not available)"
            )
        
        gb = union_bounds(files_t1, files_t2)

        # Prepare global tiling parameters (grid-aligned with inner tiles)
        tx = int(np.ceil((gb.max_x - gb.min_x) / tile_size))
        ty = int(np.ceil((gb.max_y - gb.min_y) / tile_size))

        # Practical extent info for engineers
        dx = float(gb.max_x - gb.min_x)
        dy = float(gb.max_y - gb.min_y)
        logger.info(
            "Global extent: dX=%.1fm (%.3f km), dY=%.1fm (%.3f km)",
            dx, dx / 1000.0, dy, dy / 1000.0,
        )

        def _make_tile(i: int, j: int) -> Tile:
            ox = gb.min_x + i * tile_size
            oy = gb.min_y + j * tile_size
            inner = Bounds2D(
                ox,
                oy,
                min(ox + tile_size, gb.max_x),
                min(oy + tile_size, gb.max_y),
            )
            outer = Bounds2D(
                max(gb.min_x, inner.min_x - halo),
                max(gb.min_y, inner.min_y - halo),
                min(gb.max_x, inner.max_x + halo),
                min(gb.max_y, inner.max_y + halo),
            )
            # Compute grid indices for this tile (must match GridAccumulator logic)
            nx_tile = int(np.ceil((inner.max_x - inner.min_x) / cell_size)) + 1
            ny_tile = int(np.ceil((inner.max_y - inner.min_y) / cell_size)) + 1
            x0_idx = int(np.floor((inner.min_x - gb.min_x) / cell_size))
            y0_idx = int(np.floor((inner.min_y - gb.min_y) / cell_size))
            
            return Tile(
                i=i, j=j,
                inner=inner, outer=outer,
                x0_idx=x0_idx, y0_idx=y0_idx,
                nx=nx_tile, ny=ny_tile
            )

        def _accumulate_files(
            files: list[str], *, transform: Optional[np.ndarray] = None
        ) -> tuple[
            dict[tuple[int, int], Tile],
            dict[tuple[int, int], GridAccumulator],
            int,
            dict[tuple[int, int], tuple[int, int]],
        ]:
            from ..acceleration.tile_workers import apply_transform

            tiles_dict = {}
            accs = {}
            total_pts = 0
            per_tile_stats: dict[tuple[int, int], tuple[int, int]] = {}
            reader = LaspyStreamReader(
                files,
                ground_only=ground_only,
                classification_filter=classification_filter,
                chunk_points=chunk_points,
            )
            for j in range(ty):
                for i in range(tx):
                    tile = _make_tile(i, j)
                    tiles_dict[(i, j)] = tile
                    acc = GridAccumulator(tile.inner, cell_size, use_gpu=use_gpu)
                    n_chunks = 0
                    n_pts = 0
                    for chunk in reader.stream_points(bbox=tile.outer):
                        if transform is not None:
                            chunk = apply_transform(chunk, transform)
                        acc.accumulate(chunk)
                        n_chunks += 1
                        n_pts += len(chunk)
                    if n_pts > 0:
                        accs[(i, j)] = acc
                        total_pts += n_pts
                        per_tile_stats[(i, j)] = (n_pts, n_chunks)
            return tiles_dict, accs, total_pts, per_tile_stats

        logger.info(
            "Streaming tiled DoD: tiles=%dx%d (tile=%.1fm, halo=%.1fm), chunk_points=%d",
            tx,
            ty,
            tile_size,
            halo,
            chunk_points,
        )

        t0 = time.time()
        tiles1, accs1, npts1, stats1 = _accumulate_files(files_t1)
        t1 = time.time()
        logger.info("T1 streamed: %d points into %d tiles in %.2fs", npts1, len(accs1), t1 - t0)
        for (i, j), (pts_total, chunks) in stats1.items():
            logger.info(
                "T1 tile (%d,%d): %d points in %d chunks", i, j, pts_total, chunks
            )

        tiles2, accs2, npts2, stats2 = _accumulate_files(files_t2, transform=transform_t2)
        t2 = time.time()
        logger.info("T2 streamed: %d points into %d tiles in %.2fs", npts2, len(accs2), t2 - t1)
        for (i, j), (pts_total, chunks) in stats2.items():
            logger.info(
                "T2 tile (%d,%d): %d points in %d chunks", i, j, pts_total, chunks
            )

        # Build mosaics (optionally memmap-backed)
        mosaic1 = MosaicAccumulator(gb, cell_size, memmap_dir=memmap_dir)
        mosaic2 = MosaicAccumulator(gb, cell_size, memmap_dir=memmap_dir)

        # Finalize and add tiles
        for k, acc in accs1.items():
            tile = tiles1[k]
            mosaic1.add_tile(tile, acc.finalize())
        for k, acc in accs2.items():
            tile = tiles2[k]
            mosaic2.add_tile(tile, acc.finalize())

        dem1_global = mosaic1.finalize()
        dem2_global = mosaic2.finalize()
        t3 = time.time()
        logger.info("Mosaics finalized in %.2fs", t3 - t2)
        dod = dem2_global - dem1_global
        valid = np.isfinite(dod)
        stats: Dict[str, float] = {
            "n_cells": int(valid.sum()),
            "mean_change": float(np.nanmean(dod)),
            "median_change": float(np.nanmedian(dod)),
            "rmse": float(np.sqrt(np.nanmean(np.square(dod)))),
            "min_change": float(np.nanmin(dod)),
            "max_change": float(np.nanmax(dod)),
        }

        # Log completion with statistics
        logger.info(
            "DoD completed: n_cells=%d, mean=%.4f m, median=%.4f m, rmse=%.4f m, range=[%.4f, %.4f] m",
            stats["n_cells"],
            stats["mean_change"],
            stats["median_change"],
            stats["rmse"],
            stats["min_change"],
            stats["max_change"],
        )

        return DoDResult(
            grid_x=mosaic1.grid_x,
            grid_y=mosaic1.grid_y,
            dem1=dem1_global,
            dem2=dem2_global,
            dod=dod,
            cell_size=float(cell_size),
            bounds=(gb.min_x, gb.min_y, gb.max_x, gb.max_y),
            stats=stats,
            metadata={"aggregator": "mean", "streaming": True, "tiled": True},
        )

    @staticmethod
    def compute_dod_streaming_files_tiled_parallel(
        files_t1: list[str],
        files_t2: list[str],
        cell_size: float,
        tile_size: float,
        halo: float,
        *,
        ground_only: bool = True,
        classification_filter: Optional[list[int]] = None,
        chunk_points: int = 1_000_000,
        memmap_dir: Optional[str] = None,
        transform_t2: Optional[np.ndarray] = None,
        n_workers: Optional[int] = None,
        threads_per_worker: Optional[int] = 1,
        config: Optional[AppConfig] = None,
        clip_bounds: Optional[tuple[float, float, float, float]] = None,
        local_transform: Optional["LocalCoordinateTransform"] = None,
    ) -> DoDResult:
        """
        Parallel version of out-of-core tiled DoD.
        
        Processes tiles in parallel using multiple CPU cores for significant speedup.
        Each tile is processed independently by a worker process.
        
        Args:
            files_t1: Epoch 1 file paths
            files_t2: Epoch 2 file paths
            cell_size: DEM grid cell size
            tile_size: Tile size in meters
            halo: Halo buffer (informational for DoD)
            ground_only: Filter ground points only
            classification_filter: Optional classification codes to include
            chunk_points: Points per streaming chunk
            memmap_dir: Optional directory for memory-mapped mosaicking
            transform_t2: Optional transformation matrix for epoch 2
            n_workers: Number of parallel workers (None = auto-detect)
            clip_bounds: Optional (minx, miny, maxx, maxy) to restrict processing
                         to tiles overlapping this region of interest
        
        Returns:
            DoDResult with DEMs, DoD grid, and statistics
        """
        from ..acceleration import TileParallelExecutor, process_dod_tile
        from pathlib import Path
        
        if not files_t1 or not files_t2:
            raise ValueError("compute_dod_streaming_files_tiled_parallel requires non-empty file lists")
        
        # Note: File header bounds are scanned below and filtered per tile.
        # Paths are passed per-tile to workers to avoid redundant I/O.
        
        # Get global bounds from file headers
        gb_global = union_bounds(files_t1, files_t2)
        
        # Transform bounds to local coordinates if transform is provided
        if local_transform is not None:
            gb = Bounds2D(
                min_x=gb_global.min_x - local_transform.offset_x,
                min_y=gb_global.min_y - local_transform.offset_y,
                max_x=gb_global.max_x - local_transform.offset_x,
                max_y=gb_global.max_y - local_transform.offset_y,
            )
        else:
            gb = gb_global
        
        # Calculate tile grid
        tx = int(np.ceil((gb.max_x - gb.min_x) / tile_size))
        ty = int(np.ceil((gb.max_y - gb.min_y) / tile_size))
        
        # Log extent info
        dx = float(gb.max_x - gb.min_x)
        dy = float(gb.max_y - gb.min_y)
        logger.info(
            "Global extent: dX=%.1fm (%.3f km), dY=%.1fm (%.3f km)",
            dx, dx / 1000.0, dy, dy / 1000.0,
        )
        
        # Create tiler
        tiler = Tiler(gb, cell_size, tile_size, halo)
        tiles = list(tiler.tiles())
        n_tiles_total = len(tiles)
        
        # Filter tiles by clip_bounds if provided
        if clip_bounds is not None:
            clip_minx, clip_miny, clip_maxx, clip_maxy = clip_bounds
            
            def tile_intersects_clip(tile) -> bool:
                """Check if tile overlaps the clip region."""
                t_minx = tile.inner.min_x
                t_miny = tile.inner.min_y
                t_maxx = tile.inner.max_x
                t_maxy = tile.inner.max_y
                # Check for intersection (not disjoint)
                return not (t_maxx < clip_minx or t_minx > clip_maxx or
                           t_maxy < clip_miny or t_miny > clip_maxy)
            
            tiles = [t for t in tiles if tile_intersects_clip(t)]
            n_tiles = len(tiles)
            logger.info(
                f"Clip bounds filter: {n_tiles}/{n_tiles_total} tiles overlap region of interest"
            )
            
            if n_tiles == 0:
                logger.warning("No tiles overlap the clip bounds - returning empty result")
                # Return empty result
                return DoDResult(
                    dem_t1=np.array([]),
                    dem_t2=np.array([]),
                    dod=np.array([]),
                    cell_size=cell_size,
                    origin=(gb.min_x, gb.min_y),
                    stats={},
                )
        else:
            n_tiles = n_tiles_total
        
        logger.info(
            "Parallel tiled DoD: tiles=%dx%d (%d total), tile=%.1fm, halo=%.1fm, chunk_points=%d",
            tx, ty, n_tiles, tile_size, halo, chunk_points,
        )
        
        # Determine whether to use GPU for grid accumulation (configuration-driven)
        use_gpu = False
        if config is not None and hasattr(config, "gpu"):
            use_gpu = config.gpu.enabled and config.gpu.use_for_dod

        # Log which backend is being used with clear explanation
        if use_gpu:
            logger.info("DoD parallel tiled using GPU-accelerated grid accumulation")
        else:
            logger.info(
                "DoD parallel tiled using CPU grid accumulation (GPU disabled or not available)"
            )

        # Create parallel executor
        executor = TileParallelExecutor(n_workers=n_workers, threads_per_worker=threads_per_worker)
        
        # Log worker info
        from ..acceleration import estimate_speedup_factor
        eff_workers = min(executor.n_workers, n_tiles)
        expected_speedup = estimate_speedup_factor(eff_workers, n_tiles)
        logger.info(
            f"Using {eff_workers} workers for {n_tiles} tiles "
            f"(expected speedup: {expected_speedup:.1f}x)"
        )
        
        # Pre-filter files per tile using LAS/LAZ header bounds to avoid
        # rescanning non-overlapping files in each worker
        from ..acceleration import scan_las_bounds, bounds_intersect
        t1_bounds = scan_las_bounds(files_t1)
        t2_bounds = scan_las_bounds(files_t2)

        per_tile_kwargs = []
        for tile in tiles:
            # Convert tile bounds back to global coords for file intersection check
            # (file header bounds are in global coords, tile bounds are in local coords)
            if local_transform is not None:
                tile_outer_global = Bounds2D(
                    min_x=tile.outer.min_x + local_transform.offset_x,
                    min_y=tile.outer.min_y + local_transform.offset_y,
                    max_x=tile.outer.max_x + local_transform.offset_x,
                    max_y=tile.outer.max_y + local_transform.offset_y,
                )
            else:
                tile_outer_global = tile.outer
            
            files_t1_tile = [str(f) for f, b in t1_bounds if bounds_intersect(tile_outer_global, b)]
            files_t2_tile = [str(f) for f, b in t2_bounds if bounds_intersect(tile_outer_global, b)]
            per_tile_kwargs.append({
                'files_t1': [Path(f) for f in files_t1_tile],
                'files_t2': [Path(f) for f in files_t2_tile],
            })

        # Process tiles in parallel (with per-tile file lists)
        worker_kwargs = {
            'cell_size': cell_size,
            'chunk_points': chunk_points,
            'classification_filter': classification_filter,
            'transform_matrix': transform_t2,
            'ground_only': ground_only,
            'use_gpu': use_gpu,
            'local_transform': local_transform,
        }

        t0 = time.time()
        results = executor.map_tiles(
            tiles=tiles,
            worker_fn=process_dod_tile,
            worker_kwargs=worker_kwargs,
            per_tile_kwargs=per_tile_kwargs,
        )
        t1 = time.time()
        
        logger.info(
            "Parallel tile processing complete: %d tiles in %.2fs (%.2f tiles/s)",
            n_tiles, t1 - t0, n_tiles / (t1 - t0)
        )
        
        # Build mosaics from results (optionally memmap-backed)
        mosaic1 = MosaicAccumulator(gb, cell_size, memmap_dir=memmap_dir)
        mosaic2 = MosaicAccumulator(gb, cell_size, memmap_dir=memmap_dir)
        
        for tile, dem1, dem2 in results:
            mosaic1.add_tile(tile, dem1)
            mosaic2.add_tile(tile, dem2)
        
        dem1_global = mosaic1.finalize()
        dem2_global = mosaic2.finalize()
        t2 = time.time()
        
        logger.info("Mosaics finalized in %.2fs", t2 - t1)
        
        # Compute DoD and statistics
        dod = dem2_global - dem1_global
        valid = np.isfinite(dod)
        stats: Dict[str, float] = {
            "n_cells": int(valid.sum()),
            "mean_change": float(np.nanmean(dod)),
            "median_change": float(np.nanmedian(dod)),
            "rmse": float(np.sqrt(np.nanmean(np.square(dod)))),
            "min_change": float(np.nanmin(dod)),
            "max_change": float(np.nanmax(dod)),
        }
        
        total_time = t2 - t0
        logger.info(
            "Total parallel DoD time: %.2fs (processing: %.2fs, mosaicking: %.2fs)",
            total_time, t1 - t0, t2 - t1
        )
        
        # Log completion with statistics
        logger.info(
            "DoD completed: n_cells=%d, mean=%.4f m, median=%.4f m, rmse=%.4f m, range=[%.4f, %.4f] m",
            stats["n_cells"],
            stats["mean_change"],
            stats["median_change"],
            stats["rmse"],
            stats["min_change"],
            stats["max_change"],
        )
        
        return DoDResult(
            grid_x=mosaic1.grid_x,
            grid_y=mosaic1.grid_y,
            dem1=dem1_global,
            dem2=dem2_global,
            dod=dod,
            cell_size=float(cell_size),
            bounds=(gb.min_x, gb.min_y, gb.max_x, gb.max_y),
            stats=stats,
            metadata={"aggregator": "mean", "streaming": True, "tiled": True, "parallel": True},
        )
