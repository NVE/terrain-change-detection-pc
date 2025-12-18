"""
Unit tests for parallel tile processing infrastructure.

Tests TileParallelExecutor and worker functions for correctness,
error handling, and performance characteristics.
"""

import numpy as np
import pytest

from terrain_change_detection.acceleration import (
    TileParallelExecutor,
    estimate_speedup_factor,
)
from terrain_change_detection.acceleration.tiling import Bounds2D, Tile


# Module-level worker functions for pickling compatibility
def _simple_worker(tile):
    """Simple worker that returns tile index."""
    return tile.i


def _scaling_worker(tile, scale=1):
    """Worker that scales tile index."""
    import time
    import random
    time.sleep(random.uniform(0.001, 0.01))
    return tile.i * scale


def _error_worker(tile):
    """Worker that raises an error."""
    raise ValueError(f"Intentional error on tile {tile.i}")


class TestTileParallelExecutor:
    """Test suite for TileParallelExecutor."""

    def test_executor_initialization(self):
        """Test executor initializes with correct worker count."""
        # Default initialization
        executor = TileParallelExecutor()
        assert executor.n_workers >= 1
        assert executor.memory_limit_gb is None

        # Explicit worker count
        executor = TileParallelExecutor(n_workers=4)
        assert executor.n_workers == 4

        # Memory limit
        executor = TileParallelExecutor(n_workers=2, memory_limit_gb=16.0)
        assert executor.n_workers == 2
        assert executor.memory_limit_gb == 16.0

        # Minimum workers (should be at least 1)
        executor = TileParallelExecutor(n_workers=0)
        assert executor.n_workers == 1

    def test_sequential_fallback_one_tile(self):
        """Test executor uses sequential processing for single tile."""
        executor = TileParallelExecutor(n_workers=4)

        # Create single tile
        tile = Tile(
            i=0, j=0,
            inner=Bounds2D(0, 0, 10, 10),
            outer=Bounds2D(0, 0, 10, 10),
            x0_idx=0, y0_idx=0, nx=10, ny=10
        )

        # Simple worker function
        results = executor.map_tiles(
            tiles=[tile],
            worker_fn=_scaling_worker,
            worker_kwargs={'scale': 2}
        )

        assert len(results) == 1
        assert results[0] == 0  # 0 * 2

    def test_sequential_fallback_one_worker(self):
        """Test executor uses sequential processing with 1 worker."""
        executor = TileParallelExecutor(n_workers=1)

        # Create multiple tiles
        tiles = [
            Tile(i=i, j=0,
                 inner=Bounds2D(i*10, 0, (i+1)*10, 10),
                 outer=Bounds2D(i*10, 0, (i+1)*10, 10),
                 x0_idx=i*10, y0_idx=0, nx=10, ny=10)
            for i in range(5)
        ]

        results = executor.map_tiles(
            tiles=tiles,
            worker_fn=_scaling_worker,
            worker_kwargs={'scale': 1}
        )

        assert len(results) == 5
        assert results == [0, 1, 2, 3, 4]

    def test_parallel_processing_order_preserved(self):
        """Test parallel processing preserves tile order."""
        executor = TileParallelExecutor(n_workers=2)

        # Create tiles
        tiles = [
            Tile(i=i, j=0,
                 inner=Bounds2D(i*10, 0, (i+1)*10, 10),
                 outer=Bounds2D(i*10, 0, (i*1)*10, 10),
                 x0_idx=i*10, y0_idx=0, nx=10, ny=10)
            for i in range(10)
        ]

        results = executor.map_tiles(
            tiles=tiles,
            worker_fn=_scaling_worker,
            worker_kwargs={'scale': 3}
        )

        # Results should be in same order as input
        assert len(results) == 10
        assert results == [i * 3 for i in range(10)]

    def test_empty_tile_list(self):
        """Test executor handles empty tile list gracefully."""
        executor = TileParallelExecutor(n_workers=4)

        results = executor.map_tiles(
            tiles=[],
            worker_fn=_simple_worker,
            worker_kwargs={}
        )

        assert results == []

    def test_worker_error_handling(self):
        """Test executor handles worker errors."""
        executor = TileParallelExecutor(n_workers=2)

        tiles = [
            Tile(i=i, j=0,
                 inner=Bounds2D(i*10, 0, (i+1)*10, 10),
                 outer=Bounds2D(i*10, 0, (i+1)*10, 10),
                 x0_idx=i*10, y0_idx=0, nx=10, ny=10)
            for i in range(5)
        ]

        # Should raise RuntimeError due to worker failure
        with pytest.raises(RuntimeError, match="failed"):
            executor.map_tiles(
                tiles=tiles,
                worker_fn=_error_worker,
                worker_kwargs={}
            )

    def test_progress_callback(self):
        """Test progress callback is called correctly."""
        executor = TileParallelExecutor(n_workers=2)

        tiles = [
            Tile(i=i, j=0,
                 inner=Bounds2D(i*10, 0, (i+1)*10, 10),
                 outer=Bounds2D(i*10, 0, (i+1)*10, 10),
                 x0_idx=i*10, y0_idx=0, nx=10, ny=10)
            for i in range(5)
        ]

        progress_calls = []

        def progress_callback(completed, total):
            progress_calls.append((completed, total))

        executor.map_tiles(
            tiles=tiles,
            worker_fn=_simple_worker,
            worker_kwargs={},
            progress_callback=progress_callback
        )

        # Should have been called for each tile
        assert len(progress_calls) == 5
        # Last call should be (5, 5)
        assert progress_calls[-1] == (5, 5)

    def test_get_optimal_workers(self):
        """Test optimal worker count estimation."""
        executor = TileParallelExecutor()

        # Small dataset, plenty of memory
        optimal = executor.get_optimal_workers(
            dataset_size_gb=1.0,
            available_memory_gb=32.0,
            tile_count=4
        )
        # Should be limited by tile count
        assert optimal == 4

        # Many tiles, limited memory
        optimal = executor.get_optimal_workers(
            dataset_size_gb=50.0,
            available_memory_gb=16.0,
            tile_count=100
        )
        # Should be limited by memory (16 * 0.7 / 3 ≈ 3)
        assert optimal >= 1
        assert optimal <= 4

        # Many tiles, plenty of resources
        optimal = executor.get_optimal_workers(
            dataset_size_gb=10.0,
            available_memory_gb=64.0,
            tile_count=100
        )
        # Should be limited by CPU count
        from multiprocessing import cpu_count
        assert optimal <= cpu_count()


class TestSpeedupEstimation:
    """Test suite for speedup estimation."""

    def test_estimate_speedup_single_worker(self):
        """Test speedup estimation with single worker."""
        speedup = estimate_speedup_factor(n_workers=1, n_tiles=10)
        assert speedup == 1.0

    def test_estimate_speedup_single_tile(self):
        """Test speedup estimation with single tile."""
        speedup = estimate_speedup_factor(n_workers=8, n_tiles=1)
        assert speedup == 1.0

    def test_estimate_speedup_typical(self):
        """Test speedup estimation for typical scenarios."""
        # 4 workers, 20 tiles - should get close to 4x
        speedup = estimate_speedup_factor(n_workers=4, n_tiles=20)
        assert 2.5 < speedup < 4.0

        # 8 workers, 50 tiles - should get close to 8x
        speedup = estimate_speedup_factor(n_workers=8, n_tiles=50)
        assert 5.0 < speedup < 8.0

    def test_estimate_speedup_io_saturation(self):
        """Test speedup estimation accounts for I/O saturation."""
        # Many workers should show diminishing returns
        speedup_16 = estimate_speedup_factor(n_workers=16, n_tiles=100)
        speedup_32 = estimate_speedup_factor(n_workers=32, n_tiles=100)

        # 32 workers shouldn't be 2x as fast as 16 workers
        assert speedup_32 < speedup_16 * 2.0


class TestWorkerFunctions:
    """Test suite for worker functions (integration-style tests)."""

    def test_worker_is_picklable(self):
        """Test that worker functions can be pickled for multiprocessing."""
        import pickle
        from terrain_change_detection.acceleration.tile_workers import (
            process_dod_tile,
            process_c2c_tile,
            process_m3c2_tile,
        )

        # All worker functions should be picklable
        assert pickle.loads(pickle.dumps(process_dod_tile))
        assert pickle.loads(pickle.dumps(process_c2c_tile))
        assert pickle.loads(pickle.dumps(process_m3c2_tile))

    def test_apply_transform(self):
        """Test point transformation helper."""
        from terrain_change_detection.acceleration.tile_workers import apply_transform

        # Create test points
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=np.float64)

        # Identity transform should return same points
        identity = np.eye(4)
        transformed = apply_transform(points, identity)
        np.testing.assert_array_almost_equal(transformed, points)

        # Translation
        translation = np.eye(4)
        translation[:3, 3] = [10, 20, 30]
        transformed = apply_transform(points, translation)
        expected = points + np.array([10, 20, 30])
        np.testing.assert_array_almost_equal(transformed, expected)

        # Empty points
        empty = np.empty((0, 3))
        transformed = apply_transform(empty, identity)
        assert transformed.shape == (0, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
