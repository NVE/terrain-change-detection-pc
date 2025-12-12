"""
Multiscale Model to Model Cloud Comparison (M3C2)

This module implements M3C2 algorithms for robust change detection between
multi-temporal point cloud datasets using cylindrical neighborhoods and
normal-based distance measurements.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..utils.coordinate_transform import LocalCoordinateTransform

from ..utils.logging import (
    setup_logger,
    redirect_stdout_stderr_to_logger,
    capture_c_streams_to_logger,
)
from ..acceleration import (
    LaspyStreamReader,
    Bounds2D,
    union_bounds,
    Tile,
    Tiler,
)
from ..utils.config import AppConfig

logger = setup_logger(__name__)


@dataclass
class M3C2Params:
    """
    Parameters for M3C2 computations (shared across variants).

    Attributes:
        projection_scale: Diameter or radius for normal estimation neighborhood
        cylinder_radius: Radius for measuring distances along core point normals
        max_depth: Maximum cylinder half-length along the normal direction
        min_neighbors: Minimum neighbors required to compute a metric
        normal_scale: Scale for computing surface normals (if different)
        confidence: Confidence level for significance tests (e.g., 0.95)
    """

    projection_scale: float
    cylinder_radius: float
    max_depth: float
    min_neighbors: int = 10
    normal_scale: Optional[float] = None
    confidence: float = 0.95


@dataclass
class M3C2Result:
    """
    Result of M3C2 computations.

    Attributes:
        core_points: Array of core point coordinates (N x 3)
        distances: Signed distances at core points (N,)
        uncertainty: Optional uncertainty estimates (N,), when available
        significant: Optional boolean mask of significant changes, when available
        metadata: Optional metadata dictionary (parameters, variants, etc.)
    """

    core_points: np.ndarray
    distances: np.ndarray
    uncertainty: Optional[np.ndarray] = None
    significant: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None


class M3C2Detector:
    """Multiscale Model to Model Cloud Comparison (M3C2) change detection methods."""

    @staticmethod
    def autotune_m3c2_params(
        cloud: np.ndarray,
        target_neighbors: int = 16,
        max_depth_factor: float = 0.6,
        min_radius: float = 1.0,
        max_radius: float = 20.0,
    ) -> M3C2Params:
        """
        Suggest M3C2 parameter scales based on point density.

        Heuristic:
        - Estimate density = N / area from XY bounding box.
        - Choose radius r so that expected neighbors   target_neighbors.
        - projection_scale and cylinder_radius both set to r.
        - max_depth set to max_depth_factor * r (covers signal while limiting search).

        Args:
            cloud: Nx3 array used to estimate density (e.g., T1 or combined core region)
            target_neighbors: desired neighbors within cylinder radius (~12 24 typical)
            max_depth_factor: fraction of radius for cylinder half-length (0.5 1.0)
            min_radius: lower bound on radius
            max_radius: upper bound on radius

        Returns:
            M3C2Params with tuned scales; min_neighbors set to min(10,target_neighbors//2).
        """
        if cloud.size == 0:
            raise ValueError("Cannot autotune M3C2 parameters from an empty cloud")

        x_min, y_min = cloud[:, 0].min(), cloud[:, 1].min()
        x_max, y_max = cloud[:, 0].max(), cloud[:, 1].max()
        area = max(1e-6, (x_max - x_min) * (y_max - y_min))
        density = len(cloud) / area  # pts per m^2

        # Expected neighbors ~ density *   r^2 => r = sqrt(target / (  * density))
        if density <= 0:
            r_est = min_radius
        else:
            r_est = np.sqrt(target_neighbors / (np.pi * density))
        r_est = float(np.clip(r_est, min_radius, max_radius))

        proj_scale = r_est
        cyl_radius = r_est
        max_depth = float(max(0.5, max_depth_factor * r_est))
        min_neigh = int(max(5, min(20, target_neighbors // 2)))

        params = M3C2Params(
            projection_scale=proj_scale,
            cylinder_radius=cyl_radius,
            max_depth=max_depth,
            min_neighbors=min_neigh,
            normal_scale=None,
            confidence=0.95,
        )
        
        logger.info(
            "M3C2 autotuned (sample-based): radius=%.2f m, max_depth=%.2f m, normal_scale=%.2f m",
            params.projection_scale,
            params.max_depth,
            params.normal_scale if params.normal_scale is not None else params.projection_scale,
        )
        
        return params

    @staticmethod
    def autotune_m3c2_params_from_headers(
        files_t1: list[str],
        files_t2: list[str],
        target_neighbors: int = 16,
        max_depth_factor: float = 0.6,
        min_radius: float = 1.0,
        max_radius: float = 20.0,
    ) -> M3C2Params:
        """
        Suggest M3C2 parameter scales using LAS header counts and union extent.

        Mode-agnostic alternative to sample-based autotune: uses only headers
        so results are consistent across streaming/in-memory paths.
        """
        # Compute union bounds for conservative area
        gb = union_bounds(files_t1, files_t2)
        area = max(1e-6, float((gb.max_x - gb.min_x) * (gb.max_y - gb.min_y)))

        # Sum header point counts for T1 epoch
        try:
            import laspy
        except Exception as e:
            logger.error("laspy not available for header-based autotune: %s", e)
            raise ValueError("laspy required for autotune_m3c2_params_from_headers") from e

        n_total = 0.0
        for f in files_t1:
            with laspy.open(f) as lf:
                n_total += lf.header.point_count

        density = n_total / area  # pts per m^2
        if density <= 0:
            r_est = min_radius
        else:
            r_est = np.sqrt(target_neighbors / (np.pi * density))
        r_est = float(np.clip(r_est, min_radius, max_radius))

        proj_scale = r_est
        cyl_radius = r_est
        max_depth = float(max(0.5, max_depth_factor * r_est))
        min_neigh = int(max(5, min(20, target_neighbors // 2)))

        params = M3C2Params(
            projection_scale=proj_scale,
            cylinder_radius=cyl_radius,
            max_depth=max_depth,
            min_neighbors=min_neigh,
            normal_scale=None,
            confidence=0.95,
        )
        
        logger.info(
            "M3C2 autotuned (header-based): radius=%.2f m, max_depth=%.2f m, normal_scale=%.2f m",
            params.projection_scale,
            params.max_depth,
            params.normal_scale if params.normal_scale is not None else params.projection_scale,
        )
        
        return params

    @staticmethod
    def compute_m3c2_original(
        core_points: np.ndarray,
        cloud_t1: np.ndarray,
        cloud_t2: np.ndarray,
        params: M3C2Params,
        *,
        _verbose: bool = True,
    ) -> M3C2Result:
        """
        Compute original M3C2 distances using py4dgeo.

        Args:
            core_points: Core points where distances are evaluated (N x 3)
            cloud_t1: Point cloud at time T1 (M1 x 3)
            cloud_t2: Point cloud at time T2 (M2 x 3)
            params: Parameters controlling neighborhood scales and cylinder

        Returns:
            M3C2Result with distances and optional uncertainties
        """
        if core_points.size == 0 or cloud_t1.size == 0 or cloud_t2.size == 0:
            raise ValueError("Input point arrays must be non-empty")

        try:
            from py4dgeo import Epoch, M3C2
        except Exception as e:
            logger.error("py4dgeo not available: %s", e)
            raise ImportError("py4dgeo is required for M3C2 computation") from e

        logger.debug(
            "Running M3C2: core_points=%d, cloud_t1=%d, cloud_t2=%d, cyl_radius=%.3f, max_depth=%.3f",
            len(core_points), len(cloud_t1), len(cloud_t2), params.cylinder_radius, params.max_depth,
        )

        # Construct Epochs (py4dgeo ensures DP and contiguous memory)
        epoch1 = Epoch(cloud_t1)
        epoch2 = Epoch(cloud_t2)

        # Prepare normal radii inline in call to avoid indentation issues

        # Initialize algorithm
        algo = M3C2(
            epochs=(epoch1, epoch2),
            corepoints=core_points,
            cyl_radius=float(params.cylinder_radius),
            max_distance=float(params.max_depth),
            registration_error=0.0,
            robust_aggr=False,
            normal_radii=[float(params.normal_scale if params.normal_scale is not None else params.projection_scale)],
        )

        # Run algorithm, capturing both Python and C-level prints (e.g., KDTree build)
        with capture_c_streams_to_logger(logger, level=logging.DEBUG, include_patterns=["Building KDTree"]), \
             redirect_stdout_stderr_to_logger(logger, level=logging.DEBUG, pattern="Building KDTree"):
            distances, uncertainties = algo.run()

        # distances is a float array; uncertainties may be structured
        distances = np.asarray(distances, dtype=float).reshape(-1)

        unc_vec: Optional[np.ndarray] = None
        meta_extra: Dict[str, float] = {}
        if uncertainties is not None:
            try:
                # py4dgeo uncertainties is a structured array with fields like lodetection, spread1, spread2
                if hasattr(uncertainties, 'dtype') and uncertainties.dtype.names is not None:
                    # Extract lodetection (level of detection) as primary uncertainty
                    if 'lodetection' in uncertainties.dtype.names:
                        unc_vec = uncertainties['lodetection'].astype(float)
                    elif 'spread1' in uncertainties.dtype.names:
                        unc_vec = uncertainties['spread1'].astype(float)
                    else:
                        unc_vec = None
                else:
                    unc_vec = np.asarray(uncertainties, dtype=float).reshape(-1)
            except Exception as e:
                logger.debug("Could not extract uncertainties: %s", e)
                unc_vec = None

        # Compute NaN-robust summary and valid count
        valid_mask = np.isfinite(distances)
        n_valid = int(valid_mask.sum())
        mean_v = float(np.nanmean(distances)) if n_valid > 0 else float("nan")
        median_v = float(np.nanmedian(distances)) if n_valid > 0 else float("nan")
        std_v = float(np.nanstd(distances)) if n_valid > 0 else float("nan")

        result = M3C2Result(
            core_points=core_points,
            distances=distances,
            uncertainty=unc_vec,
            significant=None,  # Original M3C2 does not produce significance mask without EP
            metadata={
                "variant": "original",
                "normal_radii": [float(params.normal_scale if params.normal_scale is not None else params.projection_scale)],
                "cylinder_radius": float(params.cylinder_radius),
                "max_depth": float(params.max_depth),
                "min_neighbors": int(params.min_neighbors),
                "confidence": float(params.confidence),
                "n_valid": n_valid,
                "mean": mean_v,
                "median": median_v,
                "std": std_v,
                **meta_extra,
            },
        )

        # Log completion with statistics (only if verbose)
        if _verbose:
            logger.info(
                "M3C2 completed: n=%d (valid=%d), mean=%.4f m, median=%.4f m, std=%.4f m",
                result.distances.size,
                n_valid,
                mean_v,
                median_v,
                std_v,
            )

        return result

    @staticmethod
    def compute_m3c2_streaming_files_tiled(
        core_points: np.ndarray,
        files_t1: list[str],
        files_t2: list[str],
        params: M3C2Params,
        *,
        tile_size: float,
        halo: Optional[float] = None,
        ground_only: bool = True,
        classification_filter: Optional[list[int]] = None,
        chunk_points: int = 1_000_000,
        transform_t2: Optional[np.ndarray] = None,
        local_transform: Optional["LocalCoordinateTransform"] = None,
    ) -> M3C2Result:
        """
        Out-of-core tiled M3C2 over streaming LAS/LAZ files using py4dgeo.

        Tile core points in the XY plane, stream each epoch per tile with a halo
        that safely covers the cylinder radius and normal-projection radius, run
        py4dgeo M3C2 over the tile core points, and stitch the results.

        Args:
            core_points: Core points where distances are evaluated (N x 3), in local coords if local_transform provided
            files_t1: LAS/LAZ file paths for epoch T1
            files_t2: LAS/LAZ file paths for epoch T2 (aligned to T1)
            params: M3C2 parameters (uses cylinder_radius, projection_scale, max_depth, ...)
            tile_size: Tile size in meters for core point partitioning
            halo: Optional XY halo; if None, uses max(cylinder_radius, projection_scale)
            ground_only: Apply ground/classification filter during streaming
            classification_filter: Optional classification filter list
            chunk_points: Points per streaming chunk
            transform_t2: Optional transformation for T2 points (ICP alignment)
            local_transform: Optional local coordinate transform (for numerical precision)

        Returns:
            M3C2Result with distances for all input core points
        """
        if core_points.size == 0:
            raise ValueError("core_points must be non-empty")
        if not files_t1 or not files_t2:
            raise ValueError("files_t1 and files_t2 must be non-empty")

        # Determine processing bounds from the union of LAS headers, not only core extents.
        # Using core-point bounds can clip halos at the dataset edge and alter neighborhoods.
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

        # Determine halo
        normal_r = float(params.normal_scale) if getattr(params, "normal_scale", None) is not None else float(params.projection_scale)
        default_halo = max(float(params.cylinder_radius), float(params.projection_scale), normal_r)
        used_halo = float(default_halo if halo is None else max(halo, default_halo))
        # Practical extent info for engineers (core points extent)
        dx = float(gb.max_x - gb.min_x)
        dy = float(gb.max_y - gb.min_y)
        logger.info(
            "Core extent: dX=%.1fm (%.3f km), dY=%.1fm (%.3f km)",
            dx, dx / 1000.0, dy, dy / 1000.0,
        )
        # Compute tile grid based on core extents
        tx = max(1, int(np.ceil((gb.max_x - gb.min_x) / tile_size)))
        ty = max(1, int(np.ceil((gb.max_y - gb.min_y) / tile_size)))
        logger.info(
            "Streaming M3C2 tiled: cores=%d, tiles=%dx%d (tile=%.1fm, halo=%.1fm), chunk_points=%d",
            len(core_points), tx, ty, tile_size, used_halo, chunk_points,
        )

        # Assign core points to tiles (core points are in local coords)
        ix = np.floor((core_points[:, 0] - gb.min_x) / tile_size).astype(int)
        iy = np.floor((core_points[:, 1] - gb.min_y) / tile_size).astype(int)
        ix = np.clip(ix, 0, max(0, tx - 1))
        iy = np.clip(iy, 0, max(0, ty - 1))
        keys = np.stack([ix, iy], axis=1)
        view = keys.view([('i', ix.dtype), ('j', iy.dtype)])
        uniq_tiles = np.unique(view)

        # Prepare output vectors
        n = len(core_points)
        distances_out = np.full(n, np.nan, dtype=float)
        unc_out: Optional[np.ndarray] = None

        from ..acceleration.tiling import LaspyStreamReader

        def _tile_bounds(i: int, j: int) -> tuple[Bounds2D, Bounds2D]:
            """Returns inner and outer bounds in LOCAL coordinates."""
            ox = gb.min_x + i * tile_size
            oy = gb.min_y + j * tile_size
            inner = Bounds2D(ox, oy, min(ox + tile_size, gb.max_x), min(oy + tile_size, gb.max_y))
            outer = Bounds2D(
                max(gb.min_x, inner.min_x - used_halo),
                max(gb.min_y, inner.min_y - used_halo),
                min(gb.max_x, inner.max_x + used_halo),
                min(gb.max_y, inner.max_y + used_halo),
            )
            return inner, outer

        # Process tiles that have core points
        t0 = __import__('time').time()
        for rec in uniq_tiles:
            i, j = int(rec['i']), int(rec['j'])
            inner, outer = _tile_bounds(i, j)  # Local coordinates
            tile_mask = (ix == i) & (iy == j)
            tile_cores = core_points[tile_mask]
            
            # Convert tile bounds back to global for file bbox filtering
            if local_transform is not None:
                outer_global = Bounds2D(
                    min_x=outer.min_x + local_transform.offset_x,
                    min_y=outer.min_y + local_transform.offset_y,
                    max_x=outer.max_x + local_transform.offset_x,
                    max_y=outer.max_y + local_transform.offset_y,
                )
            else:
                outer_global = outer
            
            # Stream epoch 1 points for this tile (outer bounds)
            reader1 = LaspyStreamReader(
                files_t1,
                ground_only=ground_only,
                classification_filter=classification_filter,
                chunk_points=chunk_points,
            )
            chunks_t1 = list(reader1.stream_points(bbox=outer_global, transform=local_transform))
            if not chunks_t1:
                logger.debug(f"Tile ({i},{j}) has no T1 points, skipping")
                continue
            pts_t1 = np.vstack(chunks_t1)
            
            # Stream epoch 2 points for this tile (outer bounds)
            reader2 = LaspyStreamReader(
                files_t2,
                ground_only=ground_only,
                classification_filter=classification_filter,
                chunk_points=chunk_points,
            )
            chunks_t2 = list(reader2.stream_points(bbox=outer_global, transform=local_transform))
            if not chunks_t2:
                logger.debug(f"Tile ({i},{j}) has no T2 points, skipping")
                continue
            pts_t2 = np.vstack(chunks_t2)
            
            if transform_t2 is not None:
                from ..acceleration.tile_workers import apply_transform
                pts_t2 = apply_transform(pts_t2, transform_t2)
            
            # Run M3C2 for this tile
            tile_result = M3C2Detector.compute_m3c2_original(
                core_points=tile_cores,
                cloud_t1=pts_t1,
                cloud_t2=pts_t2,
                params=params,
                _verbose=False,  # Suppress detailed logging (we'll log tile summary instead)
            )
            
            # Log tile progress
            valid = np.isfinite(tile_result.distances)
            n_valid = int(valid.sum())
            if n_valid > 0:
                mean_v = float(np.nanmean(tile_result.distances))
                median_v = float(np.nanmedian(tile_result.distances))
                std_v = float(np.nanstd(tile_result.distances))
                logger.info(
                    "Tile (%d,%d): n=%d (valid=%d), mean=%.4f m, median=%.4f m, std=%.4f m",
                    i, j, len(tile_cores), n_valid, mean_v, median_v, std_v
                )
            else:
                logger.info("Tile (%d,%d): n=%d (valid=0)", i, j, len(tile_cores))
            
            # Stitch results back into global arrays
            distances_out[tile_mask] = tile_result.distances
            if tile_result.uncertainty is not None:
                if unc_out is None:
                    unc_out = np.full(n, np.nan, dtype=float)
                unc_out[tile_mask] = tile_result.uncertainty

        t1 = __import__('time').time()
        logger.info("Streaming M3C2 tiled finished in %.2fs", t1 - t0)

        # Compute statistics
        dists = np.asarray(distances_out, dtype=float)
        valid_mask = np.isfinite(dists)
        n_all = int(dists.size)
        n_valid = int(valid_mask.sum())
        if n_valid > 0:
            mean_v = float(np.nanmean(dists))
            median_v = float(np.nanmedian(dists))
            std_v = float(np.nanstd(dists))
        else:
            mean_v = float("nan")
            median_v = float("nan")
            std_v = float("nan")

        # Log completion with statistics
        logger.info(
            "M3C2 completed: n=%d (valid=%d), mean=%.4f m, median=%.4f m, std=%.4f m",
            n_all, n_valid, mean_v, median_v, std_v
        )

        return M3C2Result(
            core_points=core_points,
            distances=distances_out,
            uncertainty=unc_out,
            significant=None,
            metadata={
                "variant": "streaming_tiled",
                "tile_size": float(tile_size),
                "halo": float(used_halo),
                "cylinder_radius": float(params.cylinder_radius),
                "max_depth": float(params.max_depth),
            },
        )

    @staticmethod
    def compute_m3c2_streaming_files_tiled_parallel(
        core_points: np.ndarray,
        files_t1: list[str],
        files_t2: list[str],
        params: M3C2Params,
        *,
        tile_size: float,
        halo: Optional[float] = None,
        ground_only: bool = True,
        classification_filter: Optional[list[int]] = None,
        chunk_points: int = 1_000_000,
        transform_t2: Optional[np.ndarray] = None,
        n_workers: Optional[int] = None,
        threads_per_worker: Optional[int] = 1,
    ) -> M3C2Result:
        """
        Parallel version of out-of-core tiled M3C2.
        
        Processes tiles in parallel using multiple CPU cores. Each tile
        runs M3C2 independently on its core points.
        
        Args:
            core_points: Core points where distances are evaluated (N x 3)
            files_t1: Epoch 1 file paths
            files_t2: Epoch 2 file paths
            params: M3C2 parameters
            tile_size: Tile size in meters
            halo: Optional XY halo (default: max(cylinder_radius, projection_scale))
            ground_only: Filter ground points only
            classification_filter: Optional classification codes
            chunk_points: Points per streaming chunk
            transform_t2: Optional transformation for epoch 2
            n_workers: Number of parallel workers (None = auto-detect)
        
        Returns:
            M3C2Result with distances for all core points
        """
        from ..acceleration import TileParallelExecutor, process_m3c2_tile
        from pathlib import Path
        
        if core_points.size == 0:
            raise ValueError("core_points must be non-empty")
        if not files_t1 or not files_t2:
            raise ValueError("files_t1 and files_t2 must be non-empty")
        
        # Note: File header bounds are scanned below and filtered per tile.
        # Paths are passed per-tile to workers to avoid redundant I/O.
        
        # Determine processing bounds from the union of LAS headers, not only core extents.
        # Using core-point bounds can clip tile halos at the dataset edge and change neighborhoods.
        gb = union_bounds(files_t1, files_t2)
        
        # Determine halo
        # Halo must cover all XY neighborhoods used by M3C2: cylinder_radius and normal radius.
        normal_r = float(params.normal_scale) if getattr(params, "normal_scale", None) is not None else float(params.projection_scale)
        default_halo = max(float(params.cylinder_radius), float(params.projection_scale), normal_r)
        used_halo = float(default_halo if halo is None else max(halo, default_halo))
        
        # Log extent info
        dx = float(gb.max_x - gb.min_x)
        dy = float(gb.max_y - gb.min_y)
        logger.info(
            "Core extent: dX=%.1fm (%.3f km), dY=%.1fm (%.3f km)",
            dx, dx / 1000.0, dy, dy / 1000.0,
        )
        
        # Compute tile grid
        tx = max(1, int(np.ceil((gb.max_x - gb.min_x) / tile_size)))
        ty = max(1, int(np.ceil((gb.max_y - gb.min_y) / tile_size)))
        
        # Assign core points to tiles
        ix = np.floor((core_points[:, 0] - gb.min_x) / tile_size).astype(int)
        iy = np.floor((core_points[:, 1] - gb.min_y) / tile_size).astype(int)
        ix = np.clip(ix, 0, max(0, tx - 1))
        iy = np.clip(iy, 0, max(0, ty - 1))
        
        # Group core points by tile
        tile_cores = {}
        tile_masks = {}
        for i in range(tx):
            for j in range(ty):
                tile_mask = (ix == i) & (iy == j)
                if np.any(tile_mask):
                    tile_cores[(i, j)] = core_points[tile_mask]
                    tile_masks[(i, j)] = tile_mask
        
        n_tiles = len(tile_cores)
        logger.info(
            "Parallel M3C2 tiled: cores=%d, tiles=%dx%d (%d non-empty), tile=%.1fm, halo=%.1fm, chunk_points=%d",
            len(core_points), tx, ty, n_tiles, tile_size, used_halo, chunk_points,
        )
        
        # Create tiler with dummy cell size
        tiler = Tiler(gb, cell_size=1.0, tile_size=tile_size, halo=used_halo)
        all_tiles = list(tiler.tiles())
        
        # Filter to only tiles with core points and add core point data
        tiles_with_data = []
        for tile in all_tiles:
            ij_key = (tile.i, tile.j)
            if ij_key in tile_cores:
                tiles_with_data.append(tile)
        
        logger.info(f"Processing {len(tiles_with_data)} tiles with core points (out of {len(all_tiles)} total)")
        
        # Create parallel executor
        executor = TileParallelExecutor(n_workers=n_workers, threads_per_worker=threads_per_worker)
        
        # Log worker info
        from ..acceleration import estimate_speedup_factor
        eff_workers = min(executor.n_workers, len(tiles_with_data))
        expected_speedup = estimate_speedup_factor(eff_workers, len(tiles_with_data))
        logger.info(
            f"Using {eff_workers} workers for {len(tiles_with_data)} tiles "
            f"(expected speedup: {expected_speedup:.1f}x)"
        )
        
        # Pre-filter files per tile using LAS/LAZ header bounds and pass only
        # the core points for that tile to minimize serialization overhead
        from ..acceleration import scan_las_bounds, bounds_intersect
        t1_bounds = scan_las_bounds(files_t1)
        t2_bounds = scan_las_bounds(files_t2)

        per_tile_kwargs = []
        for tile in tiles_with_data:
            ij_key = (tile.i, tile.j)
            files_t1_tile = [str(f) for f, b in t1_bounds if bounds_intersect(tile.outer, b)]
            files_t2_tile = [str(f) for f, b in t2_bounds if bounds_intersect(tile.outer, b)]
            per_tile_kwargs.append({
                'files_t1': [Path(f) for f in files_t1_tile],
                'files_t2': [Path(f) for f in files_t2_tile],
                'tile_cores_dict': {ij_key: tile_cores[ij_key]},
            })

        # Process tiles in parallel
        worker_kwargs = {
            'params': params,
            'chunk_points': chunk_points,
            'ground_only': ground_only,
            'classification_filter': classification_filter,
            'transform_matrix': transform_t2,
        }

        t0 = time.time()
        results = executor.map_tiles(
            tiles=tiles_with_data,
            worker_fn=process_m3c2_tile,
            worker_kwargs=worker_kwargs,
            per_tile_kwargs=per_tile_kwargs,
        )
        t1 = time.time()
        
        logger.info(
            "Parallel M3C2 processing complete: %d tiles in %.2fs (%.2f tiles/s)",
            len(tiles_with_data), t1 - t0, len(tiles_with_data) / (t1 - t0)
        )
        
        # Assemble results
        n = len(core_points)
        distances_out = np.full(n, np.nan, dtype=float)
        unc_out: Optional[np.ndarray] = None
        
        for tile, tile_result in results:
            ij_key = (tile.i, tile.j)
            tile_mask = tile_masks[ij_key]
            distances_out[tile_mask] = tile_result.distances
            if tile_result.uncertainty is not None:
                if unc_out is None:
                    unc_out = np.full(n, np.nan, dtype=float)
                unc_out[tile_mask] = tile_result.uncertainty
        
        # Compute statistics
        valid = np.isfinite(distances_out)
        if not np.any(valid):
            rmse = float("inf")
            mean = float("nan")
            median = float("nan")
            n_valid = 0
        else:
            rmse = float(np.sqrt(np.mean(np.square(distances_out[valid]))))
            mean = float(np.mean(distances_out[valid]))
            median = float(np.median(distances_out[valid]))
            n_valid = int(np.count_nonzero(valid))
        
        logger.info(
            "Parallel M3C2 complete: cores=%d, valid=%d, RMSE=%.4fm",
            len(core_points), n_valid, rmse
        )
        
        # Log completion with statistics
        std_v = float(np.nanstd(distances_out))
        logger.info(
            "M3C2 completed: n=%d (valid=%d), mean=%.4f m, median=%.4f m, std=%.4f m",
            len(core_points), n_valid, mean, median, std_v
        )
        
        return M3C2Result(
            core_points=core_points,
            distances=distances_out,
            uncertainty=unc_out,
            metadata={"rmse": rmse, "mean": mean, "median": median, "n_valid": n_valid, "streaming": True, "tiled": True, "parallel": True},
        )

    @staticmethod
    def compute_m3c2_streaming_pertile_parallel(
        files_t1: list[str],
        files_t2: list[str],
        params: M3C2Params,
        core_points_percent: float,
        *,
        tile_size: float,
        halo: Optional[float] = None,
        ground_only: bool = True,
        classification_filter: Optional[list[int]] = None,
        chunk_points: int = 1_000_000,
        transform_t2: Optional[np.ndarray] = None,
        n_workers: Optional[int] = None,
        threads_per_worker: Optional[int] = 1,
        local_transform: Optional["LocalCoordinateTransform"] = None,
    ) -> M3C2Result:
        """
        Parallel out-of-core tiled M3C2 with per-tile core point selection.
        
        This variant is truly out-of-core: each tile selects its own core points
        from the T1 data within that tile using reservoir sampling. No global
        core point array is needed, making it suitable for arbitrarily large
        datasets even with 100% core points.
        
        Args:
            files_t1: Epoch 1 file paths
            files_t2: Epoch 2 file paths
            params: M3C2 parameters
            core_points_percent: Percentage of T1 points to use as core points (per tile)
            tile_size: Tile size in meters
            halo: Optional XY halo (default: max(cylinder_radius, projection_scale))
            ground_only: Filter ground points only
            classification_filter: Optional classification codes
            chunk_points: Points per streaming chunk
            transform_t2: Optional transformation for epoch 2
            n_workers: Number of parallel workers (None = auto-detect)
            threads_per_worker: Threads per worker for nested parallelism
        
        Returns:
            M3C2Result with distances for all core points selected across tiles
        """
        from ..acceleration import TileParallelExecutor, process_m3c2_tile
        from pathlib import Path
        
        if not files_t1 or not files_t2:
            raise ValueError("files_t1 and files_t2 must be non-empty")
        
        # Determine processing bounds from the union of LAS headers
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
        
        # Determine halo (must cover M3C2 neighborhoods)
        normal_r = float(params.normal_scale) if getattr(params, "normal_scale", None) is not None else float(params.projection_scale)
        default_halo = max(float(params.cylinder_radius), float(params.projection_scale), normal_r)
        used_halo = float(default_halo if halo is None else max(halo, default_halo))
        
        # Log extent info
        dx = float(gb.max_x - gb.min_x)
        dy = float(gb.max_y - gb.min_y)
        logger.info(
            "Per-tile M3C2 extent: dX=%.1fm (%.3f km), dY=%.1fm (%.3f km)",
            dx, dx / 1000.0, dy, dy / 1000.0,
        )
        
        # Compute tile grid
        tx = max(1, int(np.ceil((gb.max_x - gb.min_x) / tile_size)))
        ty = max(1, int(np.ceil((gb.max_y - gb.min_y) / tile_size)))
        
        logger.info(
            "Per-tile M3C2: core_pct=%.1f%%, tiles=%dx%d (%d total), tile=%.1fm, halo=%.1fm",
            core_points_percent, tx, ty, tx * ty, tile_size, used_halo,
        )
        
        # Create tiler
        tiler = Tiler(gb, cell_size=1.0, tile_size=tile_size, halo=used_halo)
        all_tiles = list(tiler.tiles())
        
        # Create parallel executor
        executor = TileParallelExecutor(n_workers=n_workers, threads_per_worker=threads_per_worker)
        
        # Log worker info
        from ..acceleration import estimate_speedup_factor
        eff_workers = min(executor.n_workers, len(all_tiles))
        expected_speedup = estimate_speedup_factor(eff_workers, len(all_tiles))
        logger.info(
            f"Using {eff_workers} workers for {len(all_tiles)} tiles "
            f"(expected speedup: {expected_speedup:.1f}x)"
        )
        
        # Pre-filter files per tile using LAS/LAZ header bounds
        from ..acceleration import scan_las_bounds, bounds_intersect
        t1_bounds = scan_las_bounds(files_t1)
        t2_bounds = scan_las_bounds(files_t2)

        per_tile_kwargs = []
        for tile in all_tiles:
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

        # Process tiles in parallel (worker will select core points per-tile)
        worker_kwargs = {
            'params': params,
            'chunk_points': chunk_points,
            'ground_only': ground_only,
            'classification_filter': classification_filter,
            'transform_matrix': transform_t2,
            'core_points_percent': core_points_percent,  # Per-tile selection
            'local_transform': local_transform,
        }

        t0 = time.time()
        results = executor.map_tiles(
            tiles=all_tiles,
            worker_fn=process_m3c2_tile,
            worker_kwargs=worker_kwargs,
            per_tile_kwargs=per_tile_kwargs,
        )
        t1 = time.time()
        
        logger.info(
            "Per-tile M3C2 processing complete: %d tiles in %.2fs (%.2f tiles/s)",
            len(all_tiles), t1 - t0, len(all_tiles) / max(1e-6, t1 - t0)
        )
        
        # Assemble results - concatenate core points and distances from all tiles
        all_core_points = []
        all_distances = []
        all_uncertainties = []
        
        for tile, tile_result in results:
            if tile_result.core_points.size > 0:
                all_core_points.append(tile_result.core_points)
                all_distances.append(tile_result.distances)
                if tile_result.uncertainty is not None:
                    all_uncertainties.append(tile_result.uncertainty)
        
        if not all_core_points:
            logger.warning("No core points found across all tiles")
            return M3C2Result(
                core_points=np.empty((0, 3)),
                distances=np.array([]),
                uncertainty=None,
                metadata={"n_valid": 0, "streaming": True, "tiled": True, "parallel": True, "per_tile": True},
            )
        
        core_points_out = np.vstack(all_core_points)
        distances_out = np.concatenate(all_distances)
        unc_out = np.concatenate(all_uncertainties) if all_uncertainties else None
        
        # Compute statistics
        valid = np.isfinite(distances_out)
        n_valid = int(np.count_nonzero(valid))
        if n_valid > 0:
            rmse = float(np.sqrt(np.mean(np.square(distances_out[valid]))))
            mean = float(np.mean(distances_out[valid]))
            median = float(np.median(distances_out[valid]))
            std_v = float(np.std(distances_out[valid]))
        else:
            rmse = float("inf")
            mean = float("nan")
            median = float("nan")
            std_v = float("nan")
        
        logger.info(
            "Per-tile M3C2 complete: cores=%d, valid=%d, RMSE=%.4fm",
            len(core_points_out), n_valid, rmse
        )
        
        logger.info(
            "M3C2 completed: n=%d (valid=%d), mean=%.4f m, median=%.4f m, std=%.4f m",
            len(core_points_out), n_valid, mean, median, std_v
        )
        
        return M3C2Result(
            core_points=core_points_out,
            distances=distances_out,
            uncertainty=unc_out,
            metadata={"rmse": rmse, "mean": mean, "median": median, "n_valid": n_valid, 
                      "streaming": True, "tiled": True, "parallel": True, "per_tile": True,
                      "core_points_percent": core_points_percent},
        )

    @staticmethod
    def compute_m3c2_plane_based(
        core_points: np.ndarray,
        cloud_t1: np.ndarray,
        cloud_t2: np.ndarray,
        params: M3C2Params,
    ) -> M3C2Result:
        """
        Compute correspondence-driven plane-based M3C2 (CD-PB M3C2).

        Note:
            Implemented in a subsequent phase. This is a typed interface placeholder.
        """
        raise NotImplementedError("M3C2 (plane-based) will be implemented in the next phase.")

    @staticmethod
    def compute_m3c2_error_propagation(
        core_points: np.ndarray,
        cloud_t1: np.ndarray,
        cloud_t2: np.ndarray,
        params: M3C2Params,
        *,
        workers: int = 4,
    ) -> M3C2Result:
        """
        Compute M3C2 with error propagation (M3C2-EP) and significance assessment.

        Uses py4dgeo's M3C2 to obtain structured uncertainties and applies a
        level-of-detection (LoD) threshold to flag significant changes.
        """
        # Use dedicated M3C2EP implementation from py4dgeo
        try:
            from py4dgeo import Epoch, m3c2ep
        except Exception as e:
            logger.error("py4dgeo not available: %s", e)
            raise ImportError("py4dgeo is required for M3C2-EP computation") from e

        # Build Epochs and ensure scan position metadata is present and valid
        epoch1 = Epoch(cloud_t1)
        epoch2 = Epoch(cloud_t2)

        # Helper to enforce 1-based scanpos_id and complete scanpos_info
        def _ensure_scanpos(epoch, points: np.ndarray) -> None:
            # Scan positions are used by M3C2-EP for error propagation; set placeholder if missing
            # Note: For real scan data, scan positions should be extracted from LAS metadata
            if not hasattr(epoch, 'scanpos_id') or epoch.scanpos_id is None:
                # Placeholder: single synthetic scan position at dataset centroid
                centroid = points.mean(axis=0)
                epoch.scanpos_id = np.ones(len(points), dtype=np.int32)
                epoch.scanpos_info = np.array([centroid], dtype=np.float64)
                logger.debug(
                    "M3C2-EP: Synthetic single scan position at (%.2f, %.2f, %.2f) used for epoch",
                    centroid[0], centroid[1], centroid[2]
                )
            else:
                # Ensure 1-based scan positions (py4dgeo requirement)
                if epoch.scanpos_id.min() == 0:
                    epoch.scanpos_id = epoch.scanpos_id + 1
                    logger.debug("M3C2-EP: Adjusted scanpos_id to 1-based indexing")
                # Ensure scanpos_info is present
                if not hasattr(epoch, 'scanpos_info') or epoch.scanpos_info is None:
                    # Use placeholder scan position if missing
                    centroid = points.mean(axis=0)
                    epoch.scanpos_info = np.array([centroid], dtype=np.float64)
                    logger.debug(
                        "M3C2-EP: Missing scanpos_info, using centroid (%.2f, %.2f, %.2f)",
                        centroid[0], centroid[1], centroid[2]
                    )

        # Apply to both epochs
        _ensure_scanpos(epoch1, cloud_t1)
        _ensure_scanpos(epoch2, cloud_t2)

        # Defaults for transformation and alignment covariance
        tfM = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0]], dtype=float)
        Cxx = np.zeros((12, 12), dtype=float)
        refPointMov = np.array([0.0, 0.0, 0.0], dtype=float)

        normal_scale = params.normal_scale if params.normal_scale is not None else params.projection_scale
        # Optionally run in single-process mode (Windows spawn safety)
        serial_mode = max(1, int(workers)) == 1
        if serial_mode:
            # Run M3C2EP serially without multiprocessing to avoid spawn issues
            # Use internal _run_single_process method if available
            try:
                algo = m3c2ep.M3C2EP(
                    tfM=tfM,
                    Cxx=Cxx,
                    refPointMov=refPointMov,
                    perform_trans=True,
                    epochs=(epoch1, epoch2),
                    corepoints=core_points,
                    cyl_radius=float(params.cylinder_radius),
                    max_distance=float(params.max_depth),
                    registration_error=0.0,
                    robust_aggr=False,
                    normal_radii=[float(normal_scale)],
                )
                # Attempt to run single-process if available
                if hasattr(algo, '_run_single_process'):
                    logger.debug("M3C2-EP: Running in serial mode (single process)")
                    with capture_c_streams_to_logger(logger, level=logging.DEBUG, include_patterns=["Building KDTree"]), \
                         redirect_stdout_stderr_to_logger(logger, level=logging.DEBUG, pattern="Building KDTree"):
                        distances, uncertainties = algo._run_single_process()
                else:
                    # Fallback to normal run (may spawn workers)
                    logger.debug("M3C2-EP: Single-process method not available, using standard run")
                    with capture_c_streams_to_logger(logger, level=logging.DEBUG, include_patterns=["Building KDTree"]), \
                         redirect_stdout_stderr_to_logger(logger, level=logging.DEBUG, pattern="Building KDTree"):
                        distances, uncertainties = algo.run()
            except Exception as e:
                logger.error("M3C2-EP execution failed: %s", e)
                raise
        else:
            # Standard multiprocessing path
            algo = m3c2ep.M3C2EP(
                tfM=tfM,
                Cxx=Cxx,
                refPointMov=refPointMov,
                perform_trans=True,
                epochs=(epoch1, epoch2),
                corepoints=core_points,
                cyl_radius=float(params.cylinder_radius),
                max_distance=float(params.max_depth),
                registration_error=0.0,
                robust_aggr=False,
                normal_radii=[float(normal_scale)],
            )

            # Capture verbose stdout from py4dgeo internals (both Python and C streams)
            with capture_c_streams_to_logger(logger, level=logging.DEBUG, include_patterns=["Building KDTree"]), \
                 redirect_stdout_stderr_to_logger(logger, level=logging.DEBUG, pattern="Building KDTree"):
                distances, uncertainties = algo.run()

        d = np.asarray(distances, dtype=float).reshape(-1)
        sig_mask: Optional[np.ndarray] = None
        ep_meta: Dict[str, float] = {}
        if uncertainties is not None:
            # Extract lodetection as level of detection (LoD)
            lod = uncertainties['lodetection'].astype(float)
            # Mark significant changes where |distance| > LoD
            sig_mask = np.abs(d) > lod
            ep_meta["significant_count"] = int(np.sum(sig_mask))

        result = M3C2Result(
            core_points=core_points,
            distances=d,
            uncertainty=None,  # We use LoD for significance rather than storing uncertainties
            significant=sig_mask,
            metadata={
                "variant": "m3c2_ep",
                "normal_radii": [float(normal_scale)],
                "cylinder_radius": float(params.cylinder_radius),
                "max_depth": float(params.max_depth),
                **ep_meta,
            },
        )

        logger.info(
            "M3C2-EP finished: n=%d, mean=%.4f m, median=%.4f m, std=%.4f m, sig=%d",
            result.distances.size,
            float(np.nanmean(result.distances)),
            float(np.nanmedian(result.distances)),
            float(np.nanstd(result.distances)),
            int(np.sum(result.significant)) if result.significant is not None else 0,
        )

        return result
