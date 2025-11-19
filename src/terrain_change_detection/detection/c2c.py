"""
Cloud-to-Cloud (C2C) Change Detection

This module implements C2C algorithms for computing nearest-neighbor distances
between multi-temporal point cloud datasets.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np

from ..utils.logging import setup_logger
from ..acceleration import (
    LaspyStreamReader,
    Bounds2D,
    union_bounds,
    Tile,
    Tiler,
    create_gpu_neighbors,
    get_array_backend,
    ensure_cpu_array,
)
from ..utils.config import AppConfig

logger = setup_logger(__name__)


@dataclass
class C2CResult:
    """
    Result of Cloud-to-Cloud distance comparison.

    Attributes:
        distances: Per-point distances from source to nearest neighbor in target
        indices: Indices of matched target points for each source point
        rmse: Root mean square error (computed over valid pairs)
        mean: Mean of distances
        median: Median of distances
        n: Number of valid correspondences
        metadata: Optional metadata dictionary
    """

    distances: np.ndarray
    indices: np.ndarray
    rmse: float
    mean: float
    median: float
    n: int
    metadata: Optional[Dict] = None


class C2CDetector:
    """Cloud-to-Cloud (C2C) change detection methods."""

    @staticmethod
    def compute_c2c(
        source: np.ndarray,
        target: np.ndarray,
        max_distance: Optional[float] = None,
        config: Optional[AppConfig] = None,
    ) -> C2CResult:
        """
        Compute nearest-neighbor distances from source cloud to target cloud.

        Args:
            source: Source point cloud (N x 3)
            target: Target point cloud (M x 3)
            max_distance: Optional distance threshold to filter correspondences
            config: Optional configuration (for GPU settings)

        Returns:
            C2CResult with per-point distances and summary stats.
        """
        if source.size == 0 or target.size == 0:
            raise ValueError("Input point arrays must be non-empty.")

        # Determine if we should use GPU (config-driven; may still fall back)
        use_gpu = False
        gpu_backend: str = "none"
        if config is not None and hasattr(config, "gpu"):
            use_gpu = config.gpu.enabled and config.gpu.use_for_c2c

        dists: np.ndarray
        indices: np.ndarray
        
        # Optional heuristic: avoid GPU for very small problems (GPU overhead dominates)
        # Controlled via env var TCD_MIN_POINTS_FOR_GPU; default 0 (disabled) to honor tests/config
        if use_gpu:
            try:
                _min_pts_env = int(os.environ.get("TCD_MIN_POINTS_FOR_GPU", "0"))
            except Exception:
                _min_pts_env = 0
            if _min_pts_env > 0 and (len(source) < _min_pts_env or len(target) < _min_pts_env):
                logger.debug(
                    "Skipping GPU for small problem (src=%d, tgt=%d < %d threshold)",
                    len(source),
                    len(target),
                    _min_pts_env,
                )
                use_gpu = False

        # Try GPU-accelerated nearest neighbors first (if enabled by config and size)
        if use_gpu:
            try:
                nbrs = create_gpu_neighbors(n_neighbors=1, use_gpu=True)
                nbrs.fit(target)
                dists_raw, indices_raw = nbrs.kneighbors(source)
                dists = ensure_cpu_array(dists_raw).flatten()
                indices = ensure_cpu_array(indices_raw).flatten()
                try:
                    gpu_backend = getattr(nbrs, "backend_", "unknown")
                except Exception:
                    gpu_backend = "unknown"
                logger.debug(
                    "C2C (GPU[%s]) complete: src=%d, tgt=%d, rmse=%.4f m",
                    gpu_backend,
                    len(source),
                    len(target),
                    float(np.sqrt(np.mean(np.square(dists.astype(np.float64))))),
                )
            except Exception as e:
                logger.warning(
                    "GPU nearest neighbors failed (%s), falling back to CPU", str(e)
                )
                use_gpu = False
        
        # CPU path (original sklearn or numpy fallback)
        if not use_gpu:
            try:
                from sklearn.neighbors import KDTree
                tree = KDTree(target)
                dists, indices = tree.query(source, k=1)
                dists = dists.flatten()
                indices = indices.flatten()
            except Exception:
                logger.warning("KDTree failed, using naive nearest neighbor")
                dists = np.linalg.norm(source[:, None, :] - target[None, :, :], axis=2).min(axis=1)
                indices = np.argmin(np.linalg.norm(source[:, None, :] - target[None, :, :], axis=2), axis=1)

        if max_distance is not None:
            mask = dists <= max_distance
            valid_dists = dists[mask]
            valid_indices = indices[mask]
        else:
            valid_dists = dists
            valid_indices = indices

        if valid_dists.size == 0:
            rmse = float("inf")
            mean = float("nan")
            median = float("nan")
            n = 0
        else:
            # Convert to float64 for statistics to avoid overflow
            valid_dists_f64 = valid_dists.astype(np.float64) if valid_dists.dtype != np.float64 else valid_dists
            rmse = float(np.sqrt(np.mean(np.square(valid_dists_f64))))
            mean = float(np.mean(valid_dists_f64))
            median = float(np.median(valid_dists_f64))
            n = int(valid_dists.size)

        # Log completion with statistics
        backend = "GPU" if use_gpu else "CPU"
        logger.info(
            "C2C completed (%s): n=%d, mean=%.4f m, median=%.4f m, rmse=%.4f m",
            backend, n, mean, median, rmse
        )

        return C2CResult(
            distances=dists,
            indices=indices,
            rmse=rmse,
            mean=mean,
            median=median,
            n=n,
            metadata={
                "max_distance": max_distance,
                "gpu_used": use_gpu,
                "gpu_backend": gpu_backend,
            },
        )

    @staticmethod
    def compute_c2c_vertical_plane(
        source: np.ndarray,
        target: np.ndarray,
        *,
        radius: Optional[float] = None,
        k_neighbors: int = 20,
        min_neighbors: int = 6,
        config: Optional[AppConfig] = None,
    ) -> C2CResult:
        """
        Compute per-point signed vertical distances from the source cloud to a local plane
        fitted on the target cloud around each source point.

        This emulates CloudCompare's "C2C with local modeling" but measures the vertical
        offset (along Z) to the locally fitted plane, which is often more meaningful for
        terrain.

        Args:
            source: Source point cloud (N x 3)
            target: Target point cloud (M x 3)
            radius: If provided, use radius-neighborhoods; otherwise use k-NN
            k_neighbors: Number of neighbors when radius is None
            min_neighbors: Minimum neighbors required to fit a plane; otherwise fallback
            config: Optional configuration (for GPU settings)

        Returns:
            C2CResult with signed distances (positive up), indices of nearest NN for bookkeeping,
            and summary stats.
        """
        if source.size == 0 or target.size == 0:
            raise ValueError("Input point arrays must be non-empty.")

        # Determine if we should use GPU
        use_gpu = False
        gpu_backend: str = "none"
        if config is not None and hasattr(config, "gpu"):
            use_gpu = config.gpu.enabled and config.gpu.use_for_c2c

        # Try GPU-accelerated approach first
        if use_gpu:
            try:
                nbrs = create_gpu_neighbors(
                    n_neighbors=k_neighbors if radius is None else None,
                    radius=radius,
                    use_gpu=True,
                )
                nbrs.fit(target)
                try:
                    gpu_backend = getattr(nbrs, "backend_", "unknown")
                except Exception:
                    gpu_backend = "unknown"
            except Exception as e:
                logger.warning(
                    "GPU neighbors initialization failed (%s), falling back to CPU", str(e)
                )
                use_gpu = False

        # CPU fallback path
        if not use_gpu:
            try:
                from sklearn.neighbors import NearestNeighbors
                nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean")
            except Exception as e:
                logger.error("Failed to initialize CPU nearest neighbors: %s", e)
                raise ValueError("Cannot initialize neighbor search structure") from e

            # Build neighbor structure on target
            if radius is not None and radius > 0:
                nbrs = NearestNeighbors(radius=radius, metric="euclidean")
                nbrs.fit(target)
            else:
                nbrs.fit(target)

        N = len(source)
        out = np.full(N, np.nan, dtype=float)
        indices = np.full(N, -1, dtype=int)

        if radius is not None and radius > 0:
            ind_lists = nbrs.radius_neighbors(source, radius=radius, return_distance=False)
            # Ensure ind_lists is on CPU for iteration
            if use_gpu:
                ind_lists = [ensure_cpu_array(inds) for inds in ind_lists]
            # Use precomputed nearest NN for indices
            for i in range(N):
                inds = ind_lists[i]
                if len(inds) < min_neighbors:
                    continue
                nbs = target[inds]
                c = nbs.mean(axis=0)
                nbs_c = nbs - c
                try:
                    U, s, Vt = np.linalg.svd(nbs_c, full_matrices=False)
                    normal = Vt[-1, :]
                    if normal[2] < 0:
                        normal = -normal
                    plane_z = c[2]
                    if abs(normal[2]) < 1e-9:
                        continue
                    src_z = source[i, 2]
                    plane_z_at_src = plane_z + (
                        normal[0] * (source[i, 0] - c[0]) + normal[1] * (source[i, 1] - c[1])
                    ) / (-normal[2])
                    out[i] = src_z - plane_z_at_src
                    # Bookkeep index of nearest neighbor in this radius
                    dist_to_nbs = np.linalg.norm(nbs - source[i, :], axis=1)
                    closest_idx = inds[np.argmin(dist_to_nbs)]
                    indices[i] = int(closest_idx)
                except Exception:
                    continue
        else:
            # k-NN neighborhoods for all points
            idx_matrix = nbrs.kneighbors(source, return_distance=False)
            if use_gpu:
                idx_matrix = ensure_cpu_array(idx_matrix)
            for i in range(N):
                inds = idx_matrix[i, :]
                if len(inds) < min_neighbors:
                    continue
                nbs = target[inds]
                c = nbs.mean(axis=0)
                nbs_c = nbs - c
                try:
                    U, s, Vt = np.linalg.svd(nbs_c, full_matrices=False)
                    normal = Vt[-1, :]
                    if normal[2] < 0:
                        normal = -normal
                    plane_z = c[2]
                    if abs(normal[2]) < 1e-9:
                        continue
                    src_z = source[i, 2]
                    plane_z_at_src = plane_z + (
                        normal[0] * (source[i, 0] - c[0]) + normal[1] * (source[i, 1] - c[1])
                    ) / (-normal[2])
                    out[i] = src_z - plane_z_at_src
                    indices[i] = int(inds[0])
                except Exception:
                    continue

        # Stats (signed distances)
        valid = np.isfinite(out)
        if not np.any(valid):
            rmse = float("inf")
            mean = float("nan")
            median = float("nan")
            n = 0
        else:
            rmse = float(np.sqrt(np.mean(np.square(out[valid]))))
            mean = float(np.mean(out[valid]))
            median = float(np.median(out[valid]))
            n = int(np.count_nonzero(valid))

        # Log completion with statistics
        backend = "GPU" if use_gpu else "CPU"
        logger.info(
            "C2C vertical plane completed (%s): n=%d, mean=%.4f m, median=%.4f m, rmse=%.4f m",
            backend, n, mean, median, rmse
        )

        return C2CResult(
            distances=out,
            indices=indices,
            rmse=rmse,
            mean=mean,
            median=median,
            n=n,
            metadata={
                "mode": "vertical_plane",
                "radius": radius,
                "k_neighbors": k_neighbors,
                "min_neighbors": min_neighbors,
                "gpu_used": use_gpu,
                "gpu_backend": gpu_backend,
            },
        )

    @staticmethod
    def compute_c2c_streaming_files_tiled(
        files_src: list[str],
        files_tgt: list[str],
        *,
        tile_size: float,
        max_distance: float,
        ground_only: bool = True,
        classification_filter: Optional[list[int]] = None,
        chunk_points: int = 1_000_000,
        transform_src: Optional[np.ndarray] = None,
        config: Optional[AppConfig] = None,
    ) -> C2CResult:
        """
        Out-of-core tiled Cloud-to-Cloud distances (nearest neighbor) between two epochs.

        Requires a finite max_distance to bound the search radius and ensure tile-local
        neighborhoods are sufficient (halo >= max_distance).

        Returns concatenated per-source-point distances (order is by tile stream order),
        along with summary statistics. Target indices are not tracked in streaming mode
        (set to -1).

        Args:
            files_src: LAZ/LAS files for the source epoch (points to measure from)
            files_tgt: LAZ/LAS files for the target epoch (points to measure to)
            tile_size: XY tile size in data units (meters)
            max_distance: Maximum search radius; distances beyond remain > max_distance
            ground_only: Apply ground/classification filter during streaming
            classification_filter: Optional classification filter list
            chunk_points: Points per streaming chunk
            transform_src: Optional transformation matrix for source points
            config: Optional configuration (for GPU settings)
        """
        if not files_src or not files_tgt:
            raise ValueError("compute_c2c_streaming_files_tiled requires non-empty file lists")
        if not (isinstance(max_distance, (int, float)) and max_distance > 0):
            raise ValueError("max_distance must be provided and > 0 for streaming tiled C2C")

        # Determine if we should use GPU
        use_gpu = False
        if config is not None and hasattr(config, 'gpu'):
            use_gpu = config.gpu.enabled and config.gpu.use_for_c2c

        gb = union_bounds(files_src, files_tgt)

        # Tile grid for core (source) and neighborhood (target)
        tx = max(1, int(np.ceil((gb.max_x - gb.min_x) / tile_size)))
        ty = max(1, int(np.ceil((gb.max_y - gb.min_y) / tile_size)))
        halo = float(max_distance)

        # Engineer-friendly extent
        dx = float(gb.max_x - gb.min_x)
        dy = float(gb.max_y - gb.min_y)
        logger.info(
            "C2C extent: dX=%.1fm (%.3f km), dY=%.1fm (%.3f km)", dx, dx / 1000.0, dy, dy / 1000.0
        )
        logger.info(
            "Streaming C2C tiled: tiles=%dx%d (tile=%.1fm, halo=%.1fm), chunk_points=%d, r=%.2fm",
            tx, ty, tile_size, halo, chunk_points, max_distance,
        )

        from ..acceleration.tiling import LaspyStreamReader

        def _inner_outer(i: int, j: int) -> tuple[Bounds2D, Bounds2D]:
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
            return inner, outer

        # Prepare readers
        reader_src = LaspyStreamReader(
            files_src,
            ground_only=ground_only,
            classification_filter=classification_filter,
            chunk_points=chunk_points,
        )
        reader_tgt = LaspyStreamReader(
            files_tgt,
            ground_only=ground_only,
            classification_filter=classification_filter,
            chunk_points=chunk_points,
        )

        all_dists: list[np.ndarray] = []
        total_src = 0
        gpu_tiles = 0
        processed_tiles = 0
        t0 = __import__('time').time()
        for j in range(ty):
            for i in range(tx):
                inner, outer = _inner_outer(i, j)
                src_chunks = list(reader_src.stream_points(bbox=inner))
                if not src_chunks:
                    continue
                src = np.vstack(src_chunks)
                if transform_src is not None:
                    from ..acceleration.tile_workers import apply_transform
                    src = apply_transform(src, transform_src)
                tgt_chunks = list(reader_tgt.stream_points(bbox=outer))
                if not tgt_chunks:
                    logger.debug(f"Tile ({i},{j}) has no target points, skipping")
                    all_dists.append(np.full(len(src), np.inf, dtype=float))
                    total_src += len(src)
                    processed_tiles += 1
                    continue
                tgt = np.vstack(tgt_chunks)
                processed_tiles += 1
                
                # Compute distances for this tile
                tile_gpu = False
                if use_gpu:
                    try:
                        nbrs = create_gpu_neighbors(n_neighbors=1, use_gpu=True)
                        nbrs.fit(tgt)
                        dists_raw, _ = nbrs.kneighbors(src)
                        dists = ensure_cpu_array(dists_raw).flatten()
                        gpu_tiles += 1
                        tile_gpu = True
                    except Exception as e:
                        logger.debug(f"GPU C2C failed for tile ({i},{j}): {e}, using CPU")
                        from sklearn.neighbors import KDTree
                        tree = KDTree(tgt)
                        dists, _ = tree.query(src, k=1)
                        dists = dists.flatten()
                else:
                    from sklearn.neighbors import KDTree
                    tree = KDTree(tgt)
                    dists, _ = tree.query(src, k=1)
                    dists = dists.flatten()
                
                # Apply max_distance threshold
                dists = np.where(dists <= max_distance, dists, np.inf)
                all_dists.append(dists)
                total_src += len(src)
                
                # Per-tile progress logging
                n_valid = int(np.count_nonzero(np.isfinite(dists)))
                logger.info(
                    "Tile (%d,%d): src=%d, tgt=%d, valid=%d%s",
                    i, j, len(src), len(tgt), n_valid,
                    " (GPU)" if tile_gpu else ""
                )

        dists_concat = np.concatenate(all_dists) if all_dists else np.array([], dtype=float)
        valid = np.isfinite(dists_concat)
        if not np.any(valid):
            rmse = float("inf"); mean = float("nan"); median = float("nan"); n = 0
        else:
            valid_dists_f64 = dists_concat[valid].astype(np.float64)
            rmse = float(np.sqrt(np.mean(np.square(valid_dists_f64))))
            mean = float(np.mean(valid_dists_f64))
            median = float(np.median(valid_dists_f64))
            n = int(np.count_nonzero(valid))

        # Indices not tracked in streaming path
        idx = -np.ones_like(dists_concat, dtype=int)
        gpu_used_any = use_gpu and gpu_tiles > 0
        logger.info(
            "C2C streaming tiled finished: src_total=%d, valid=%d (<= %.2fm)%s (gpu_tiles=%d/%d)",
            total_src,
            n,
            max_distance,
            " (GPU used)" if gpu_used_any else "",
            gpu_tiles,
            processed_tiles,
        )
        
        # Log completion with statistics
        backend = "GPU" if gpu_used_any else "CPU"
        logger.info(
            "C2C completed (%s): n=%d, mean=%.4f m, median=%.4f m, rmse=%.4f m",
            backend, n, mean, median, rmse
        )
        
        return C2CResult(
            distances=dists_concat,
            indices=idx,
            rmse=rmse,
            mean=mean,
            median=median,
            n=n,
            metadata={
                "streaming": True,
                "tiled": True,
                "tile_size": float(tile_size),
                "halo": float(halo),
                "max_distance": float(max_distance),
                "gpu_used": gpu_used_any,
                "gpu_tiles": gpu_tiles,
                "tiles_with_src": processed_tiles,
            },
        )

    @staticmethod
    def compute_c2c_streaming_files_tiled_parallel(
        files_src: list[str],
        files_tgt: list[str],
        *,
        tile_size: float,
        max_distance: float,
        ground_only: bool = True,
        classification_filter: Optional[list[int]] = None,
        chunk_points: int = 1_000_000,
        transform_src: Optional[np.ndarray] = None,
        n_workers: Optional[int] = None,
        threads_per_worker: Optional[int] = 1,
        config: Optional[AppConfig] = None,
    ) -> C2CResult:
        """
        Parallel version of out-of-core tiled C2C.
        
        Processes tiles in parallel using multiple CPU cores. Each tile
        computes nearest neighbor distances independently.
        
        Args:
            files_src: Source epoch file paths
            files_tgt: Target epoch file paths
            tile_size: Tile size in meters
            max_distance: Maximum search radius
            ground_only: Filter ground points only
            classification_filter: Optional classification codes to include
            chunk_points: Points per streaming chunk
            transform_src: Optional transformation matrix for source
            n_workers: Number of parallel workers (None = auto-detect)
            threads_per_worker: Number of threads per worker
            config: Optional configuration (for GPU settings)
        
        Returns:
            C2CResult with concatenated distances and statistics
        """
        from ..acceleration import TileParallelExecutor, process_c2c_tile
        from pathlib import Path
        
        if not files_src or not files_tgt:
            raise ValueError("compute_c2c_streaming_files_tiled_parallel requires non-empty file lists")
        if not (isinstance(max_distance, (int, float)) and max_distance > 0):
            raise ValueError("max_distance must be provided and > 0 for parallel streaming tiled C2C")
        
        # Determine if we should use GPU
        use_gpu = False
        if config is not None and hasattr(config, 'gpu'):
            use_gpu = config.gpu.enabled and config.gpu.use_for_c2c
        
        # Note: File header bounds are scanned below and filtered per tile.
        # Paths are passed per-tile to workers to avoid redundant I/O.
        
        # Get global bounds
        gb = union_bounds(files_src, files_tgt)
        
        # Calculate tile grid (halo = max_distance for radius coverage)
        halo = float(max_distance)
        tx = int(np.ceil((gb.max_x - gb.min_x) / tile_size))
        ty = int(np.ceil((gb.max_y - gb.min_y) / tile_size))
        
        # Log extent info
        dx = float(gb.max_x - gb.min_x)
        dy = float(gb.max_y - gb.min_y)
        logger.info(
            "C2C extent: dX=%.1fm (%.3f km), dY=%.1fm (%.3f km)",
            dx, dx / 1000.0, dy, dy / 1000.0,
        )
        
        # Create tiler with dummy cell size (not used for C2C)
        tiler = Tiler(gb, cell_size=1.0, tile_size=tile_size, halo=halo)
        tiles = list(tiler.tiles())
        n_tiles = len(tiles)
        
        logger.info(
            "Parallel tiled C2C: tiles=%dx%d (%d total), tile=%.1fm, halo=%.1fm, chunk_points=%d, r=%.2fm",
            tx, ty, n_tiles, tile_size, halo, chunk_points, max_distance,
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
        
        # Pre-filter files per tile using LAS/LAZ header bounds
        from ..acceleration import scan_las_bounds, bounds_intersect
        src_bounds = scan_las_bounds(files_src)
        tgt_bounds = scan_las_bounds(files_tgt)

        per_tile_kwargs = []
        for tile in tiles:
            files_src_tile = [str(f) for f, b in src_bounds if bounds_intersect(tile.inner, b)]
            files_tgt_tile = [str(f) for f, b in tgt_bounds if bounds_intersect(tile.outer, b)]
            per_tile_kwargs.append({'files_source': [Path(f) for f in files_src_tile], 'files_target': [Path(f) for f in files_tgt_tile]})

        # Process tiles in parallel
        worker_kwargs = {
            'max_distance': max_distance,
            'chunk_points': chunk_points,
            'classification_filter': classification_filter,
            'transform_matrix': transform_src,
            'ground_only': ground_only,
            'use_gpu': use_gpu,
        }
        
        if use_gpu:
            logger.info("C2C parallel processing with GPU acceleration enabled")

        t0 = time.time()
        results = executor.map_tiles(
            tiles=tiles,
            worker_fn=process_c2c_tile,
            worker_kwargs=worker_kwargs,
            per_tile_kwargs=per_tile_kwargs,
        )
        t1 = time.time()
        
        logger.info(
            "Parallel C2C processing complete: %d tiles in %.2fs (%.2f tiles/s)",
            n_tiles, t1 - t0, n_tiles / (t1 - t0)
        )
        
        # Concatenate all distance arrays
        all_dists: list[np.ndarray] = []
        total_src = 0
        for tile, distances in results:
            if distances.size > 0:
                all_dists.append(distances)
                total_src += len(distances)
        
        dists_concat = np.concatenate(all_dists) if all_dists else np.array([], dtype=float)
        
        # Compute statistics
        valid = np.isfinite(dists_concat) & (dists_concat <= max_distance)
        if not np.any(valid):
            rmse = float("inf")
            mean = float("nan")
            median = float("nan")
            n = 0
        else:
            valid_dists_f64 = dists_concat[valid].astype(np.float64)
            rmse = float(np.sqrt(np.mean(np.square(valid_dists_f64))))
            mean = float(np.mean(valid_dists_f64))
            median = float(np.median(valid_dists_f64))
            n = int(np.count_nonzero(valid))
        
        # Indices not tracked in streaming path
        idx = -np.ones_like(dists_concat, dtype=int)
        
        logger.info(
            "Parallel C2C complete: src_total=%d, valid=%d (<= %.2fm), RMSE=%.4fm%s",
            total_src,
            n,
            max_distance,
            rmse,
            " (GPU requested)" if use_gpu else "",
        )
        
        # Log completion with statistics
        backend = "GPU" if use_gpu else "CPU"
        logger.info(
            "C2C completed (%s): n=%d, mean=%.4f m, median=%.4f m, rmse=%.4f m",
            backend, n, mean, median, rmse
        )
        
        return C2CResult(
            distances=dists_concat,
            indices=idx,
            rmse=rmse,
            mean=mean,
            median=median,
            n=n,
            metadata={
                "streaming": True,
                "tiled": True,
                "parallel": True,
                "max_distance": max_distance,
                "gpu_used": use_gpu,
            },
        )
