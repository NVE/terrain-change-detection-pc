"""
Worker functions for parallel tile processing.

Each worker function processes a single tile independently and returns results.
All functions are designed to be picklable for multiprocessing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..acceleration.tiling import (
    Bounds2D,
    GridAccumulator,
    LaspyStreamReader,
    Tile,
)

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
    
    # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    points_h = np.hstack([points, ones])
    
    # Apply transformation
    transformed_h = points_h @ matrix.T
    
    # Convert back to 3D
    return transformed_h[:, :3]


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
    acc1 = GridAccumulator(tile.inner, cell_size)
    acc2 = GridAccumulator(tile.inner, cell_size)
    
    # Stream and accumulate epoch 1 points (using outer bounds for complete coverage)
    reader1 = LaspyStreamReader(
        files_t1,
        ground_only=ground_only,
        classification_filter=classification_filter,
        chunk_points=chunk_points
    )
    n_chunks_1 = 0
    n_points_1 = 0
    for chunk in reader1.stream_points(bbox=tile.outer):
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
    for chunk in reader2.stream_points(bbox=tile.outer):
        if transform_matrix is not None:
            # Apply transformation on-the-fly
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
    
    Returns:
        Tuple of (tile, distances) where distances is 1D array
    """
    from sklearn.neighbors import KDTree
    
    # Load source points (inner tile only - these are the query points)
    source_points = []
    reader_src = LaspyStreamReader(
        files_source,
        ground_only=ground_only,
        classification_filter=classification_filter,
        chunk_points=chunk_points
    )
    for chunk in reader_src.stream_points(bbox=tile.inner):
        if transform_matrix is not None:
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
    for chunk in reader_tgt.stream_points(bbox=tile.outer):
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
    
    # Build KD-tree and query
    try:
        tree = KDTree(target)
        distances, indices = tree.query(source, k=k_neighbors)
        distances = distances.flatten()
        
        # Apply radius cutoff
        distances = np.where(distances <= max_distance, distances, np.inf)
        
        logger.debug(
            f"Tile C2C complete: {len(source):,} source, {len(target):,} target, "
            f"{np.isfinite(distances).sum():,} valid distances (max={max_distance:.2f})"
        )
        
        return (tile, distances)
        
    except Exception as e:
        logger.error(f"KDTree query failed for tile: {e}")
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
):
    """
    Process single M3C2 tile in worker process.
    
    Loads both epoch point clouds for the tile and runs M3C2 on the
    pre-partitioned core points assigned to this tile.
    
    Args:
        tile: Tile with inner and outer bounds
        files_t1: Epoch 1 file paths
        files_t2: Epoch 2 file paths
        params: M3C2Params object with algorithm parameters
        chunk_points: Maximum points per streaming chunk
        classification_filter: Optional classification filter
        transform_matrix: Optional transformation for epoch 2
        tile_cores_dict: Dictionary mapping (i,j) to core points for that tile
    
    Returns:
        Tuple of (tile, M3C2Result) for assembly
    """
    from ..detection.change_detection import ChangeDetector, M3C2Result
    
    # Get core points for this tile
    ij_key = (tile.i, tile.j)
    if tile_cores_dict is None or ij_key not in tile_cores_dict:
        logger.warning(f"Tile ({tile.i},{tile.j}) has no core points")
        return (tile, M3C2Result(core_points=np.empty((0, 3)), distances=np.array([]), uncertainty=None, metadata={}))
    
    core_points_tile = tile_cores_dict[ij_key]
    
    # Load epoch 1 points (outer bounds for neighborhoods)
    points_t1 = []
    reader1 = LaspyStreamReader(
        files_t1,
        ground_only=ground_only,
        classification_filter=classification_filter,
        chunk_points=chunk_points
    )
    for chunk in reader1.stream_points(bbox=tile.outer):
        points_t1.append(chunk)
    
    pc1 = np.vstack(points_t1) if points_t1 else np.empty((0, 3))
    
    # Load epoch 2 points
    points_t2 = []
    reader2 = LaspyStreamReader(
        files_t2,
        ground_only=ground_only,
        classification_filter=classification_filter,
        chunk_points=chunk_points
    )
    for chunk in reader2.stream_points(bbox=tile.outer):
        if transform_matrix is not None:
            chunk = apply_transform(chunk, transform_matrix)
        points_t2.append(chunk)
    
    pc2 = np.vstack(points_t2) if points_t2 else np.empty((0, 3))
    
    # Handle empty tiles
    if len(pc1) == 0 or len(pc2) == 0:
        logger.warning(
            f"Tile ({tile.i},{tile.j}) has insufficient points: "
            f"T1={len(pc1)}, T2={len(pc2)}, core={len(core_points_tile)}"
        )
        empty_distances = np.full(len(core_points_tile), np.nan, dtype=float)
        return (tile, M3C2Result(core_points=core_points_tile, distances=empty_distances, uncertainty=None, metadata={}))
    
    # Run M3C2 computation
    try:
        result = ChangeDetector.compute_m3c2_original(core_points_tile, pc1, pc2, params)
        
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
