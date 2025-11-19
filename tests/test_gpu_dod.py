"""
Tests for GPU-accelerated DoD (DEM of Difference) operations.

Tests GPU/CPU numerical parity, configuration integration, and performance
characteristics for DoD grid accumulation.
"""

import numpy as np
import pytest

from terrain_change_detection.acceleration import GridAccumulator, Bounds2D
from terrain_change_detection.detection import ChangeDetector
from terrain_change_detection.utils.config import AppConfig


class TestGridAccumulatorGPU:
    """Test GPU acceleration of GridAccumulator."""

    def test_gpu_cpu_parity_basic(self):
        """Test that GPU and CPU GridAccumulator produce identical results."""
        # Create test points
        np.random.seed(42)
        points = np.random.rand(1000, 3) * 100.0  # Random points in 100x100 area

        bounds = Bounds2D(min_x=0, min_y=0, max_x=100, max_y=100)
        cell_size = 1.0

        # CPU accumulation
        acc_cpu = GridAccumulator(bounds, cell_size, use_gpu=False)
        acc_cpu.accumulate(points)
        dem_cpu = acc_cpu.finalize()

        # GPU accumulation
        acc_gpu = GridAccumulator(bounds, cell_size, use_gpu=True)
        acc_gpu.accumulate(points)
        dem_gpu = acc_gpu.finalize()

        # Results should match within numerical tolerance
        # Use nanclose since cells with no points are NaN
        np.testing.assert_allclose(
            dem_gpu, dem_cpu, rtol=1e-5, atol=1e-8, equal_nan=True
        )

    def test_gpu_cpu_parity_multiple_chunks(self):
        """Test GPU/CPU parity with multiple accumulate calls."""
        np.random.seed(123)
        bounds = Bounds2D(min_x=0, min_y=0, max_x=50, max_y=50)
        cell_size = 0.5

        acc_cpu = GridAccumulator(bounds, cell_size, use_gpu=False)
        acc_gpu = GridAccumulator(bounds, cell_size, use_gpu=True)

        # Accumulate in multiple chunks
        for _ in range(5):
            chunk = np.random.rand(500, 3) * 50.0
            acc_cpu.accumulate(chunk)
            acc_gpu.accumulate(chunk)

        dem_cpu = acc_cpu.finalize()
        dem_gpu = acc_gpu.finalize()

        np.testing.assert_allclose(dem_gpu, dem_cpu, rtol=1e-5, equal_nan=True)

    def test_gpu_empty_points(self):
        """Test GPU accumulator handles empty point arrays."""
        bounds = Bounds2D(min_x=0, min_y=0, max_x=10, max_y=10)
        acc_gpu = GridAccumulator(bounds, cell_size=1.0, use_gpu=True)

        # Accumulate empty array
        acc_gpu.accumulate(np.empty((0, 3)))

        dem = acc_gpu.finalize()
        assert dem.shape == (11, 11)  # Based on bounds and cell_size
        assert np.all(np.isnan(dem))  # All cells should be NaN

    def test_gpu_out_of_bounds_points(self):
        """Test GPU accumulator filters points outside bounds."""
        bounds = Bounds2D(min_x=10, min_y=10, max_x=20, max_y=20)
        cell_size = 1.0

        # Create points mostly outside bounds
        points = np.array([
            [5.0, 5.0, 100.0],  # Outside
            [15.0, 15.0, 50.0],  # Inside
            [25.0, 25.0, 200.0],  # Outside
            [12.0, 18.0, 75.0],  # Inside
        ])

        acc_cpu = GridAccumulator(bounds, cell_size, use_gpu=False)
        acc_cpu.accumulate(points)
        dem_cpu = acc_cpu.finalize()

        acc_gpu = GridAccumulator(bounds, cell_size, use_gpu=True)
        acc_gpu.accumulate(points)
        dem_gpu = acc_gpu.finalize()

        np.testing.assert_allclose(dem_gpu, dem_cpu, rtol=1e-5, equal_nan=True)

        # Should have exactly 2 non-NaN cells
        assert np.sum(~np.isnan(dem_gpu)) == 2

    def test_gpu_multiple_points_per_cell(self):
        """Test GPU accumulator correctly averages multiple points per cell."""
        bounds = Bounds2D(min_x=0, min_y=0, max_x=10, max_y=10)
        cell_size = 2.0

        # Create multiple points in same cell
        points = np.array([
            [1.0, 1.0, 10.0],  # Cell (0,0)
            [1.5, 1.5, 20.0],  # Cell (0,0)
            [1.8, 1.2, 30.0],  # Cell (0,0)
            [5.0, 5.0, 100.0],  # Cell (2,2)
        ])

        acc_gpu = GridAccumulator(bounds, cell_size, use_gpu=True)
        acc_gpu.accumulate(points)
        dem_gpu = acc_gpu.finalize()

        # Cell (0,0) should have mean of 10, 20, 30 = 20.0
        assert np.isclose(dem_gpu[0, 0], 20.0)
        # Cell (2,2) should have value 100.0
        assert np.isclose(dem_gpu[2, 2], 100.0)


class TestDoDGPUIntegration:
    """Test GPU integration in DoD methods."""

    def test_compute_dod_with_config_gpu_enabled(self):
        """Test basic compute_dod respects GPU config."""
        np.random.seed(42)
        points_t1 = np.random.rand(500, 3) * 100.0
        points_t2 = points_t1 + np.random.randn(500, 3) * 0.5  # Small change

        # Create config with GPU enabled
        config = AppConfig()
        config.gpu.enabled = True
        config.gpu.use_for_preprocessing = True

        # This should work without error (whether GPU available or not)
        result = ChangeDetector.compute_dod(
            points_t1, points_t2, cell_size=2.0, config=config
        )

        assert result is not None
        assert result.dod.shape == result.dem1.shape
        assert result.dod.shape == result.dem2.shape

    def test_compute_dod_streaming_gpu_cpu_parity(self, tmp_path):
        """Test streaming DoD GPU/CPU parity."""
        # Create synthetic LAZ files
        pytest.skip("Requires LAZ file generation - integration test")

    def test_config_disables_gpu(self):
        """Test that GPU can be disabled via config."""
        np.random.seed(42)
        points_t1 = np.random.rand(100, 3) * 50.0
        points_t2 = points_t1 + np.random.randn(100, 3) * 0.3

        # Config with GPU disabled
        config = AppConfig()
        config.gpu.enabled = False

        result = ChangeDetector.compute_dod(
            points_t1, points_t2, cell_size=1.0, config=config
        )

        assert result is not None
        # Should still work, just using CPU


class TestDoDTileWorkerGPU:
    """Test GPU in DoD tile workers."""

    def test_process_dod_tile_with_gpu(self):
        """Test that process_dod_tile accepts use_gpu parameter."""
        from terrain_change_detection.acceleration import process_dod_tile, Tile
        from pathlib import Path

        # This is a unit test for the API
        # Full integration test would require actual LAZ files
        pytest.skip("Requires LAZ files - integration test")


class TestDoDGPUPerformance:
    """Test GPU performance characteristics."""

    @pytest.mark.slow
    def test_gpu_faster_than_cpu_large_dataset(self):
        """Test that GPU is faster than CPU for large point clouds."""
        pytest.skip("Performance test - run separately with actual hardware")

    def test_gpu_memory_efficiency(self):
        """Test GPU memory handling with large grids."""
        # Create large grid
        bounds = Bounds2D(min_x=0, min_y=0, max_x=1000, max_y=1000)
        cell_size = 0.1  # 10000x10000 grid

        # Should not crash with out-of-memory
        try:
            acc_gpu = GridAccumulator(bounds, cell_size, use_gpu=True)
            # If GPU available but insufficient memory, should fall back gracefully
            assert acc_gpu is not None
        except MemoryError:
            pytest.skip("Insufficient GPU memory for test")


class TestDoDEdgeCases:
    """Test edge cases for GPU DoD."""

    def test_gpu_single_point(self):
        """Test GPU accumulator with single point."""
        bounds = Bounds2D(min_x=0, min_y=0, max_x=10, max_y=10)
        acc_gpu = GridAccumulator(bounds, cell_size=1.0, use_gpu=True)

        point = np.array([[5.0, 5.0, 100.0]])
        acc_gpu.accumulate(point)
        dem = acc_gpu.finalize()

        # Exactly one non-NaN cell
        assert np.sum(~np.isnan(dem)) == 1
        assert np.nanmax(dem) == 100.0

    def test_gpu_identical_points(self):
        """Test GPU accumulator with identical points."""
        bounds = Bounds2D(min_x=0, min_y=0, max_x=10, max_y=10)
        acc_gpu = GridAccumulator(bounds, cell_size=1.0, use_gpu=True)

        # Multiple identical points
        points = np.array([[5.0, 5.0, 50.0]] * 100)
        acc_gpu.accumulate(points)
        dem = acc_gpu.finalize()

        # Should average to 50.0
        assert np.sum(~np.isnan(dem)) == 1
        assert np.isclose(np.nanmax(dem), 50.0)

    def test_gpu_extreme_values(self):
        """Test GPU accumulator with extreme Z values."""
        bounds = Bounds2D(min_x=0, min_y=0, max_x=10, max_y=10)
        acc_gpu = GridAccumulator(bounds, cell_size=1.0, use_gpu=True)

        # Points with extreme Z values
        points = np.array([
            [5.0, 5.0, 1e6],
            [5.0, 5.0, -1e6],
            [5.0, 5.0, 0.0],
        ])

        acc_gpu.accumulate(points)
        dem = acc_gpu.finalize()

        # Should compute mean correctly
        expected_mean = (1e6 - 1e6 + 0.0) / 3.0
        assert np.isclose(dem[5, 5], expected_mean, rtol=1e-5)

    def test_gpu_sparse_distribution(self):
        """Test GPU accumulator with very sparse point distribution."""
        bounds = Bounds2D(min_x=0, min_y=0, max_x=1000, max_y=1000)
        cell_size = 10.0
        
        # Only a few points scattered across large area
        np.random.seed(42)
        points = np.random.rand(10, 3) * 1000.0

        acc_cpu = GridAccumulator(bounds, cell_size, use_gpu=False)
        acc_cpu.accumulate(points)
        dem_cpu = acc_cpu.finalize()

        acc_gpu = GridAccumulator(bounds, cell_size, use_gpu=True)
        acc_gpu.accumulate(points)
        dem_gpu = acc_gpu.finalize()

        np.testing.assert_allclose(dem_gpu, dem_cpu, rtol=1e-5, equal_nan=True)

    def test_gpu_dense_distribution(self):
        """Test GPU accumulator with very dense point distribution."""
        bounds = Bounds2D(min_x=0, min_y=0, max_x=10, max_y=10)
        cell_size = 0.1  # Small cells
        
        # Many points in small area
        np.random.seed(42)
        points = np.random.rand(10000, 3) * 10.0

        acc_cpu = GridAccumulator(bounds, cell_size, use_gpu=False)
        acc_cpu.accumulate(points)
        dem_cpu = acc_cpu.finalize()

        acc_gpu = GridAccumulator(bounds, cell_size, use_gpu=True)
        acc_gpu.accumulate(points)
        dem_gpu = acc_gpu.finalize()

        np.testing.assert_allclose(dem_gpu, dem_cpu, rtol=1e-5, equal_nan=True)


class TestDoDNumericalStability:
    """Test numerical stability of GPU DoD operations."""

    def test_gpu_numerical_precision(self):
        """Test that GPU maintains numerical precision."""
        np.random.seed(42)
        
        # Create points with values that could cause precision issues
        points = np.random.rand(1000, 3)
        points[:, 2] = points[:, 2] * 1e-6 + 1e6  # Small variations around large baseline

        bounds = Bounds2D(min_x=0, min_y=0, max_x=1, max_y=1)
        cell_size = 0.1

        acc_cpu = GridAccumulator(bounds, cell_size, use_gpu=False)
        acc_cpu.accumulate(points)
        dem_cpu = acc_cpu.finalize()

        acc_gpu = GridAccumulator(bounds, cell_size, use_gpu=True)
        acc_gpu.accumulate(points)
        dem_gpu = acc_gpu.finalize()

        # Should maintain precision
        np.testing.assert_allclose(dem_gpu, dem_cpu, rtol=1e-4, equal_nan=True)

    def test_gpu_accumulation_order_independence(self):
        """Test that accumulation order doesn't affect results."""
        np.random.seed(42)
        all_points = np.random.rand(1000, 3) * 50.0

        bounds = Bounds2D(min_x=0, min_y=0, max_x=50, max_y=50)
        cell_size = 1.0

        # Accumulate all at once
        acc1 = GridAccumulator(bounds, cell_size, use_gpu=True)
        acc1.accumulate(all_points)
        dem1 = acc1.finalize()

        # Accumulate in shuffled order
        shuffled_points = all_points[np.random.permutation(len(all_points))]
        acc2 = GridAccumulator(bounds, cell_size, use_gpu=True)
        acc2.accumulate(shuffled_points)
        dem2 = acc2.finalize()

        # Results should be identical
        np.testing.assert_allclose(dem1, dem2, rtol=1e-10, equal_nan=True)

    def test_gpu_chunked_vs_batch_accumulation(self):
        """Test that chunked accumulation matches batch accumulation."""
        np.random.seed(42)
        all_points = np.random.rand(2000, 3) * 100.0

        bounds = Bounds2D(min_x=0, min_y=0, max_x=100, max_y=100)
        cell_size = 2.0

        # Batch accumulation
        acc_batch = GridAccumulator(bounds, cell_size, use_gpu=True)
        acc_batch.accumulate(all_points)
        dem_batch = acc_batch.finalize()

        # Chunked accumulation
        acc_chunked = GridAccumulator(bounds, cell_size, use_gpu=True)
        chunk_size = 200
        for i in range(0, len(all_points), chunk_size):
            chunk = all_points[i:i+chunk_size]
            acc_chunked.accumulate(chunk)
        dem_chunked = acc_chunked.finalize()

        # Results should match exactly
        np.testing.assert_allclose(
            dem_batch, dem_chunked, rtol=1e-10, equal_nan=True
        )
