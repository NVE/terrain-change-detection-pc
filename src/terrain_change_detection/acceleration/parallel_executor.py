"""
Parallel execution infrastructure for tile-based processing.

Provides TileParallelExecutor for distributing tile processing across
multiple CPU cores using multiprocessing.
"""

from __future__ import annotations

import logging
import os
import time
from multiprocessing import Pool, cpu_count
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _worker_wrapper(args: Tuple[int, Any, Callable, Dict[str, Any]]) -> Tuple[int, Any, Optional[str]]:
    """
    Worker wrapper function for parallel tile processing.
    
    Must be at module level for pickling on Windows.
    
    Args:
        args: Tuple of (tile_index, tile, worker_fn, worker_kwargs)
    
    Returns:
        Tuple of (tile_index, result, error_message)
    """
    idx, tile, worker_fn, worker_kwargs = args
    try:
        result = worker_fn(tile, **worker_kwargs)
        return (idx, result, None)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Worker error on tile {idx}: {error_msg}")
        return (idx, None, error_msg)


class TileParallelExecutor:
    """
    Parallel executor for tile-based processing.
    
    Manages worker pool, distributes tiles to workers, and collects results
    while maintaining order. Handles memory constraints by limiting concurrent
    workers and provides progress tracking.
    
    Example:
        executor = TileParallelExecutor(n_workers=4)
        results = executor.map_tiles(
            tiles=tile_list,
            worker_fn=process_dod_tile,
            worker_kwargs={'cell_size': 1.0, 'chunk_points': 1_000_000}
        )
    """
    
    def __init__(
        self,
        n_workers: Optional[int] = None,
        memory_limit_gb: Optional[float] = None,
    ):
        """
        Initialize parallel executor.
        
        Args:
            n_workers: Number of worker processes. If None, uses cpu_count - 1
                to leave one core for system/coordination. Minimum is 1.
            memory_limit_gb: Soft memory limit to guide concurrency.
                Not strictly enforced but logged if exceeded.
        """
        if n_workers is None:
            n_workers = max(1, cpu_count() - 1)
        else:
            n_workers = max(1, int(n_workers))
        
        self.n_workers = n_workers
        self.memory_limit_gb = memory_limit_gb
        
        logger.info(
            f"Initialized TileParallelExecutor with {self.n_workers} workers "
            f"(total CPUs: {cpu_count()})"
        )
    
    def map_tiles(
        self,
        tiles: List[Any],
        worker_fn: Callable,
        worker_kwargs: Dict[str, Any],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Any]:
        """
        Map worker function over tiles in parallel.
        
        Distributes tiles across worker processes and collects results in
        the same order as input tiles. Handles errors gracefully and provides
        progress tracking.
        
        Args:
            tiles: List of tiles to process (typically Tile objects)
            worker_fn: Function to apply to each tile. Must be picklable and
                have signature: worker_fn(tile, **worker_kwargs) -> result
            worker_kwargs: Fixed keyword arguments passed to each worker call
            progress_callback: Optional callback function called after each tile
                completes. Signature: callback(completed_count, total_count)
        
        Returns:
            List of results in same order as input tiles
            
        Raises:
            RuntimeError: If all workers fail or critical error occurs
        """
        n_tiles = len(tiles)
        
        if n_tiles == 0:
            logger.warning("No tiles to process")
            return []
        
        logger.info(f"Processing {n_tiles} tiles with {self.n_workers} workers")
        start_time = time.time()
        
        # If only 1 worker or 1 tile, use sequential processing (no pool overhead)
        if self.n_workers == 1 or n_tiles == 1:
            logger.info("Using sequential processing (1 worker or 1 tile)")
            results = []
            for i, tile in enumerate(tiles):
                try:
                    result = worker_fn(tile, **worker_kwargs)
                    results.append(result)
                    
                    if progress_callback:
                        progress_callback(i + 1, n_tiles)
                    
                    if (i + 1) % 10 == 0 or i + 1 == n_tiles:
                        elapsed = time.time() - start_time
                        rate = (i + 1) / elapsed
                        eta = (n_tiles - i - 1) / rate if rate > 0 else 0
                        logger.info(
                            f"Progress: {i + 1}/{n_tiles} tiles "
                            f"({100 * (i + 1) / n_tiles:.1f}%) - "
                            f"Rate: {rate:.2f} tiles/s - ETA: {eta:.1f}s"
                        )
                except Exception as e:
                    logger.error(f"Error processing tile {i}: {e}", exc_info=True)
                    raise RuntimeError(f"Tile processing failed: {e}") from e
            
            total_time = time.time() - start_time
            logger.info(
                f"Sequential processing complete: {n_tiles} tiles in {total_time:.1f}s "
                f"({n_tiles / total_time:.2f} tiles/s)"
            )
            return results
        
        # Parallel processing with multiprocessing.Pool
        try:
            results = self._parallel_map(
                tiles, worker_fn, worker_kwargs, progress_callback, start_time
            )
            
            total_time = time.time() - start_time
            logger.info(
                f"Parallel processing complete: {n_tiles} tiles in {total_time:.1f}s "
                f"({n_tiles / total_time:.2f} tiles/s) - "
                f"Speedup: {n_tiles / total_time / (n_tiles / total_time / self.n_workers):.2f}x theoretical"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Parallel processing failed: {e}", exc_info=True)
            raise RuntimeError(f"Parallel tile processing failed: {e}") from e
    
    def _parallel_map(
        self,
        tiles: List[Any],
        worker_fn: Callable,
        worker_kwargs: Dict[str, Any],
        progress_callback: Optional[Callable],
        start_time: float,
    ) -> List[Any]:
        """
        Execute parallel mapping using multiprocessing.Pool.
        
        Uses imap_unordered for better memory efficiency and progress tracking,
        then reorders results to match input tile order.
        """
        n_tiles = len(tiles)
        
        # Create arguments for worker wrapper (need worker_fn and kwargs for each tile)
        worker_args = [(i, tile, worker_fn, worker_kwargs) for i, tile in enumerate(tiles)]
        
        # Process with pool
        with Pool(processes=self.n_workers) as pool:
            # Use imap_unordered for better responsiveness
            results_dict = {}
            errors = []
            
            for i, (idx, result, error) in enumerate(
                pool.imap_unordered(_worker_wrapper, worker_args)
            ):
                if error:
                    errors.append((idx, error))
                    logger.error(f"Tile {idx} failed: {error}")
                else:
                    results_dict[idx] = result
                
                # Progress reporting
                completed = i + 1
                if progress_callback:
                    progress_callback(completed, n_tiles)
                
                # Log progress at intervals
                if completed % 10 == 0 or completed == n_tiles:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (n_tiles - completed) / rate if rate > 0 else 0
                    success_rate = 100 * len(results_dict) / completed
                    
                    logger.info(
                        f"Progress: {completed}/{n_tiles} tiles "
                        f"({100 * completed / n_tiles:.1f}%) - "
                        f"Rate: {rate:.2f} tiles/s - ETA: {eta:.1f}s - "
                        f"Success: {success_rate:.1f}%"
                    )
        
        # Check for errors
        if errors:
            error_msg = f"{len(errors)} tiles failed out of {n_tiles}"
            logger.error(error_msg)
            for idx, error in errors[:5]:  # Log first 5 errors
                logger.error(f"  Tile {idx}: {error}")
            if len(errors) > 5:
                logger.error(f"  ... and {len(errors) - 5} more errors")
            
            raise RuntimeError(error_msg)
        
        # Reorder results to match input tile order
        results = [results_dict[i] for i in range(n_tiles)]
        
        return results
    
    def get_optimal_workers(
        self,
        dataset_size_gb: float,
        available_memory_gb: float,
        tile_count: int,
    ) -> int:
        """
        Estimate optimal worker count based on resources.
        
        This is a heuristic that considers CPU count, available memory,
        and number of tiles to process.
        
        Args:
            dataset_size_gb: Approximate size of dataset in GB
            available_memory_gb: Available system memory in GB
            tile_count: Number of tiles to process
        
        Returns:
            Recommended worker count
        """
        # Memory constraint: assume 3 GB per worker for working set + overhead
        memory_per_worker = 3.0
        max_workers_memory = int(available_memory_gb * 0.7 / memory_per_worker)
        
        # CPU constraint
        max_workers_cpu = max(1, cpu_count() - 1)
        
        # Work constraint: no point having more workers than tiles
        max_workers_work = tile_count
        
        # Take minimum of all constraints
        optimal = min(max_workers_memory, max_workers_cpu, max_workers_work)
        
        logger.info(
            f"Optimal workers: {optimal} "
            f"(constraints - memory: {max_workers_memory}, "
            f"cpu: {max_workers_cpu}, work: {max_workers_work})"
        )
        
        return max(1, optimal)


def estimate_speedup_factor(n_workers: int, n_tiles: int) -> float:
    """
    Estimate expected speedup factor for parallel processing.
    
    Accounts for parallel overhead and assumes near-linear scaling
    up to CPU count, with diminishing returns beyond that due to I/O.
    
    Args:
        n_workers: Number of worker processes
        n_tiles: Number of tiles to process
    
    Returns:
        Expected speedup factor compared to sequential processing
    """
    if n_workers <= 1 or n_tiles <= 1:
        return 1.0
    
    # Parallel overhead (process creation, communication)
    overhead_factor = 0.9  # 10% overhead
    
    # I/O saturation beyond ~16 workers on typical storage
    if n_workers > 16:
        io_factor = 16 / n_workers * 0.8  # Diminishing returns
    else:
        io_factor = 1.0
    
    # Can't speedup more than number of tiles
    work_factor = min(1.0, n_tiles / n_workers)
    
    theoretical_speedup = n_workers * overhead_factor * io_factor * work_factor
    
    # Real-world is usually 70-85% of theoretical
    realistic_speedup = theoretical_speedup * 0.8
    
    return max(1.0, realistic_speedup)
