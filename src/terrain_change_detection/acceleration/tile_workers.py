"""
Worker functions for parallel tile processing.

Each worker function processes a single tile independently and returns results.
All functions are designed to be picklable for multiprocessing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..utils.coordinate_transform import LocalCoordinateTransform

from ..acceleration.tiling import (
    Bounds2D,
    GridAccumulator,
    LaspyStreamReader,
    Tile,
)
from ..acceleration.gpu_neighbors import create_gpu_neighbors
from ..acceleration.gpu_array_ops import ensure_cpu_array

logger = logging.getLogger(__name__)


def apply_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Apply 4x4 transformation matrix to points.
    
    Args:
        points: (N, 3) array of XYZ coordinates
        matrix: (4, 4) transformation matrix
    
    Returns:
        Transformed points (N, 3)
    """
    if points.size == 0:
        return points

    try:
        from ..acceleration import apply_transform_jit

        return apply_transform_jit(points, matrix)
    except Exception:
        # Fallback: vectorized NumPy when numba is unavailable or fails
        R = matrix[:3, :3]
        t = matrix[:3, 3]
        return points @ R.T + t


def process_dod_tile(
    tile: Tile,
    files_t1: List[Path],
    files_t2: List[Path],
    cell_size: float,
    chunk_points: int,
    classification_filter: Optional[List[int]] = None,
    transform_matrix: Optional[np.ndarray] = None,
    *,
    ground_only: bool = True,
    use_gpu: bool = False,
    local_transform: Optional["LocalCoordinateTransform"] = None,
) -> Tuple[Tile, np.ndarray, np.ndarray]:
    """
    Process single DoD tile in worker process.
    
    Streams points from both epochs, accumulates them into grid cells,
    and computes mean DEMs for the tile.
    
    Args:
        tile: Tile with inner and outer bounds
        files_t1: Epoch 1 LAZ/LAS file paths
        files_t2: Epoch 2 LAZ/LAS file paths
        cell_size: DEM grid cell size in data units
        chunk_points: Maximum points per streaming chunk
        classification_filter: Optional list of classification codes to include
        transform_matrix: Optional 4x4 transformation matrix for epoch 2
    
    Returns:
        Tuple of (tile, dem1, dem2) where DEMs are 2D arrays with shape (ny, nx)
    """
    # Create accumulators for this tile's inner bounds
    acc1 = GridAccumulator(tile.inner, cell_size, use_gpu=use_gpu)
    acc2 = GridAccumulator(tile.inner, cell_size, use_gpu=use_gpu)
    
    # Convert tile bounds back to global coords for bbox filtering if using local transform
    # (files have global coords, but tile bounds were converted to local)
    if local_transform is not None:
        outer_bbox = Bounds2D(
            min_x=tile.outer.min_x + local_transform.offset_x,
            min_y=tile.outer.min_y + local_transform.offset_y,
            max_x=tile.outer.max_x + local_transform.offset_x,
            max_y=tile.outer.max_y + local_transform.offset_y,
        )
    else:
        outer_bbox = tile.outer
    
    # Stream and accumulate epoch 1 points (using outer bounds for complete coverage)
    reader1 = LaspyStreamReader(
        files_t1,
        ground_only=ground_only,
        classification_filter=classification_filter,
        chunk_points=chunk_points
    )
    n_chunks_1 = 0
    n_points_1 = 0
    for chunk in reader1.stream_points(bbox=outer_bbox, transform=local_transform):
        acc1.accumulate(chunk)
        n_chunks_1 += 1
        n_points_1 += len(chunk)
    
    # Stream and accumulate epoch 2 points (with optional transformation)
    reader2 = LaspyStreamReader(
        files_t2,
        ground_only=ground_only,
        classification_filter=classification_filter,
        chunk_points=chunk_points
    )
    n_chunks_2 = 0
    n_points_2 = 0
    for chunk in reader2.stream_points(bbox=outer_bbox, transform=local_transform):
        if transform_matrix is not None:
            # Apply ICP transformation on-the-fly (now in local coordinate space)
            chunk = apply_transform(chunk, transform_matrix)
        acc2.accumulate(chunk)
        n_chunks_2 += 1
        n_points_2 += len(chunk)
    
    # Compute mean DEMs
    dem1 = acc1.finalize()
    dem2 = acc2.finalize()
    
    # Log tile completion with stats
    logger.debug(
        f"Tile complete: inner=({tile.inner.min_x:.1f}, {tile.inner.min_y:.1f}) "
        f"T1={n_points_1:,} pts ({n_chunks_1} chunks), "
        f"T2={n_points_2:,} pts ({n_chunks_2} chunks), "
        f"DEM shape={dem1.shape}"
    )
    
    return (tile, dem1, dem2)


def process_c2c_tile(
    tile: Tile,
    files_source: List[Path],
    files_target: List[Path],
    max_distance: float,
    chunk_points: int,
    classification_filter: Optional[List[int]] = None,
    transform_matrix: Optional[np.ndarray] = None,
    k_neighbors: int = 1,
    *,
    ground_only: bool = True,
    use_gpu: bool = False,
    local_transform: Optional["LocalCoordinateTransform"] = None,
) -> Tuple[Tile, np.ndarray]:
    """
    Process single C2C tile in worker process.
    
    Loads source points from inner tile and target points from outer tile
    (with halo for radius coverage), then computes nearest neighbor distances.
    
    Args:
        tile: Tile with inner and outer bounds
        files_source: Source point cloud file paths
        files_target: Target point cloud file paths
        max_distance: Maximum distance threshold
        chunk_points: Maximum points per streaming chunk
        classification_filter: Optional list of classification codes
        transform_matrix: Optional 4x4 transformation for source points
        k_neighbors: Number of nearest neighbors (typically 1)
        ground_only: Whether to filter ground points only
        use_gpu: Whether to use GPU acceleration for nearest neighbors
    
    Returns:
        Tuple of (tile, distances) where distances is 1D array
    """
    # Convert tile bounds back to global coords for bbox filtering if using local transform
    # (files have global coords, but tile bounds were converted to local)
    if local_transform is not None:
        inner_bbox = Bounds2D(
            min_x=tile.inner.min_x + local_transform.offset_x,
            min_y=tile.inner.min_y + local_transform.offset_y,
            max_x=tile.inner.max_x + local_transform.offset_x,
            max_y=tile.inner.max_y + local_transform.offset_y,
        )
        outer_bbox = Bounds2D(
            min_x=tile.outer.min_x + local_transform.offset_x,
            min_y=tile.outer.min_y + local_transform.offset_y,
            max_x=tile.outer.max_x + local_transform.offset_x,
            max_y=tile.outer.max_y + local_transform.offset_y,
        )
    else:
        inner_bbox = tile.inner
        outer_bbox = tile.outer
    
    # Load source points (inner tile only - these are the query points)
    source_points = []
    reader_src = LaspyStreamReader(
        files_source,
        ground_only=ground_only,
        classification_filter=classification_filter,
        chunk_points=chunk_points
    )
    for chunk in reader_src.stream_points(bbox=inner_bbox, transform=local_transform):
        if transform_matrix is not None:
            # Apply ICP transformation on-the-fly (now in local coordinate space)
            chunk = apply_transform(chunk, transform_matrix)
        source_points.append(chunk)
    
    source = np.vstack(source_points) if source_points else np.empty((0, 3))
    
    # Load target points (outer tile with halo for radius coverage)
    target_points = []
    reader_tgt = LaspyStreamReader(
        files_target,
        ground_only=ground_only,
        classification_filter=classification_filter,
        chunk_points=chunk_points
    )
    for chunk in reader_tgt.stream_points(bbox=outer_bbox, transform=local_transform):
        target_points.append(chunk)
    
    target = np.vstack(target_points) if target_points else np.empty((0, 3))
    
    # Handle empty point clouds
    if len(source) == 0:
        logger.debug(f"Tile has no source points: {tile.inner}")
        return (tile, np.array([]))
    
    if len(target) == 0:
        logger.warning(f"Tile has no target points: {tile.outer}")
        # Return inf distances for all source points
        return (tile, np.full(len(source), np.inf, dtype=float))
    
    # Build nearest neighbors structure and query
    try:
        # Try GPU first if requested
        if use_gpu:
            try:
                nbrs = create_gpu_neighbors(n_neighbors=k_neighbors, use_gpu=True)
                nbrs.fit(target)
                distances, indices = nbrs.kneighbors(source)
                # Ensure results are on CPU for downstream processing
                distances = ensure_cpu_array(distances).flatten()
                try:
                    backend = getattr(nbrs, "backend_", "unknown")
                except Exception:
                    backend = "unknown"
                gpu_used = True
            except Exception as e:
                logger.debug(f"GPU nearest neighbors failed, falling back to CPU: {e}")
                use_gpu = False
        
        # CPU fallback
        if not use_gpu:
            from sklearn.neighbors import KDTree
            tree = KDTree(target)
            distances, indices = tree.query(source, k=k_neighbors)
            distances = distances.flatten()
            gpu_used = False
            backend = "cpu"
        
        # Apply radius cutoff
        distances = np.where(distances <= max_distance, distances, np.inf)
        
        if gpu_used:
            backend_str = f"GPU[{backend}]"
        else:
            backend_str = "CPU"

        logger.debug(
            f"Tile C2C complete ({backend_str}): "
            f"{len(source):,} source, {len(target):,} target, "
            f"{np.isfinite(distances).sum():,} valid distances (max={max_distance:.2f})"
        )
        
        return (tile, distances)
        
    except Exception as e:
        logger.error(f"Nearest neighbors query failed for tile: {e}")
        raise


def process_m3c2_tile(
    tile: Tile,
    files_t1: List[Path],
    files_t2: List[Path],
    params,  # M3C2Params object
    chunk_points: int,
    ground_only: bool = True,
    classification_filter: Optional[List[int]] = None,
    transform_matrix: Optional[np.ndarray] = None,
    tile_cores_dict: Optional[Dict] = None,
    core_points_percent: Optional[float] = None,
    local_transform: Optional["LocalCoordinateTransform"] = None,
):
    """
    Process single M3C2 tile in worker process.
    
    Loads both epoch point clouds for the tile and runs M3C2 on core points.
    Core points can be either pre-computed (tile_cores_dict) or selected
    per-tile using reservoir sampling (core_points_percent).
    
    Args:
        tile: Tile with inner and outer bounds
        files_t1: Epoch 1 file paths
        files_t2: Epoch 2 file paths
        params: M3C2Params object with algorithm parameters
        chunk_points: Maximum points per streaming chunk
        ground_only: Whether to filter ground points only
        classification_filter: Optional classification filter
        transform_matrix: Optional transformation for epoch 2
        tile_cores_dict: Dictionary mapping (i,j) to pre-computed core points
        core_points_percent: If provided, select this percentage of T1 points as cores
    
    Returns:
        Tuple of (tile, M3C2Result) for assembly
    """
    from ..detection.m3c2 import M3C2Detector, M3C2Result
    
    ij_key = (tile.i, tile.j)
    
    # Convert tile bounds back to global coords for bbox filtering if using local transform
    # (files have global coords, but tile bounds were converted to local)
    if local_transform is not None:
        outer_bbox = Bounds2D(
            min_x=tile.outer.min_x + local_transform.offset_x,
            min_y=tile.outer.min_y + local_transform.offset_y,
            max_x=tile.outer.max_x + local_transform.offset_x,
            max_y=tile.outer.max_y + local_transform.offset_y,
        )
    else:
        outer_bbox = tile.outer
    
    # Load epoch 1 points (outer bounds for neighborhoods)
    points_t1 = []
    reader1 = LaspyStreamReader(
        files_t1,
        ground_only=ground_only,
        classification_filter=classification_filter,
        chunk_points=chunk_points
    )
    for chunk in reader1.stream_points(bbox=outer_bbox, transform=local_transform):
        points_t1.append(chunk)
    
    pc1 = np.vstack(points_t1) if points_t1 else np.empty((0, 3))
    
    # Handle empty T1
    if len(pc1) == 0:
        logger.warning(f"Tile ({tile.i},{tile.j}) has no T1 points")
        return (tile, M3C2Result(core_points=np.empty((0, 3)), distances=np.array([]), uncertainty=None, metadata={}))
    
    # Determine core points for this tile
    if tile_cores_dict is not None and ij_key in tile_cores_dict:
        # Use pre-computed core points (backward compatibility)
        core_points_tile = tile_cores_dict[ij_key]
    elif core_points_percent is not None:
        # Per-tile core point selection: sample from T1 points in inner tile
        # Filter pc1 to inner bounds for core selection
        inner_mask = (
            (pc1[:, 0] >= tile.inner.min_x) & (pc1[:, 0] <= tile.inner.max_x) &
            (pc1[:, 1] >= tile.inner.min_y) & (pc1[:, 1] <= tile.inner.max_y)
        )
        pc1_inner = pc1[inner_mask]
        
        if len(pc1_inner) == 0:
            logger.warning(f"Tile ({tile.i},{tile.j}) has no T1 points in inner bounds")
            return (tile, M3C2Result(core_points=np.empty((0, 3)), distances=np.array([]), uncertainty=None, metadata={}))
        
        # Calculate number of core points for this tile
        n_cores = max(1, int(len(pc1_inner) * core_points_percent / 100.0))
        
        # Subsample if needed
        if len(pc1_inner) > n_cores:
            idx = np.random.choice(len(pc1_inner), n_cores, replace=False)
            core_points_tile = pc1_inner[idx]
        else:
            core_points_tile = pc1_inner
        
        logger.debug(
            f"Tile ({tile.i},{tile.j}) selected {len(core_points_tile):,} core points "
            f"({core_points_percent:.1f}% of {len(pc1_inner):,} inner T1 points)"
        )
    else:
        logger.warning(f"Tile ({tile.i},{tile.j}) has no core points source (neither dict nor percent)")
        return (tile, M3C2Result(core_points=np.empty((0, 3)), distances=np.array([]), uncertainty=None, metadata={}))
    
    # Load epoch 2 points
    points_t2 = []
    reader2 = LaspyStreamReader(
        files_t2,
        ground_only=ground_only,
        classification_filter=classification_filter,
        chunk_points=chunk_points
    )
    for chunk in reader2.stream_points(bbox=outer_bbox, transform=local_transform):
        if transform_matrix is not None:
            # Apply ICP transformation on-the-fly (now in local coordinate space)
            chunk = apply_transform(chunk, transform_matrix)
        points_t2.append(chunk)
    
    pc2 = np.vstack(points_t2) if points_t2 else np.empty((0, 3))
    
    # Handle empty T2
    if len(pc2) == 0:
        logger.warning(
            f"Tile ({tile.i},{tile.j}) has no T2 points: "
            f"T1={len(pc1)}, core={len(core_points_tile)}"
        )
        empty_distances = np.full(len(core_points_tile), np.nan, dtype=float)
        return (tile, M3C2Result(core_points=core_points_tile, distances=empty_distances, uncertainty=None, metadata={}))
    
    # Run M3C2 computation
    try:
        result = M3C2Detector.compute_m3c2_original(core_points_tile, pc1, pc2, params, _verbose=False)
        
        logger.debug(
            f"Tile ({tile.i},{tile.j}) M3C2 complete: {len(core_points_tile):,} core points, "
            f"{len(pc1):,} T1 pts, {len(pc2):,} T2 pts, "
            f"{np.sum(np.isfinite(result.distances)):,} valid distances"
        )
        
        return (tile, result)
        
    except Exception as e:
        logger.error(f"M3C2 computation failed for tile ({tile.i},{tile.j}): {e}")
        empty_distances = np.full(len(core_points_tile), np.nan, dtype=float)
        return (tile, M3C2Result(core_points=core_points_tile, distances=empty_distances, uncertainty=None, metadata={}))

