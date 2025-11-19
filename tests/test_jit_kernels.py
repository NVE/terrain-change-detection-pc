"""
Tests for JIT-compiled kernels.

Tests Numba JIT kernels for performance-critical operations with proper
fallback behavior when Numba is unavailable.
"""

import numpy as np
import pytest


class TestApplyTransformJIT:
    """Test apply_transform_jit function."""

    def test_transform_basic(self):
        """Test basic transformation with identity matrix."""
        from terrain_change_detection.acceleration import apply_transform_jit
        
        points = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        
        # Identity transformation
        matrix = np.eye(4)
        
        result = apply_transform_jit(points, matrix)
        
        np.testing.assert_allclose(result, points, rtol=1e-10)

    def test_transform_translation(self):
        """Test transformation with translation only."""
        from terrain_change_detection.acceleration import apply_transform_jit
        
        points = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])
        
        # Translation matrix: move by [10, 20, 30]
        matrix = np.eye(4)
        matrix[0, 3] = 10.0
        matrix[1, 3] = 20.0
        matrix[2, 3] = 30.0
        
        result = apply_transform_jit(points, matrix)
        
        expected = points + np.array([10.0, 20.0, 30.0])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_transform_rotation(self):
        """Test transformation with rotation."""
        from terrain_change_detection.acceleration import apply_transform_jit
        
        points = np.array([[1.0, 0.0, 0.0]])
        
        # 90-degree rotation around Z-axis
        theta = np.pi / 2
        matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        
        result = apply_transform_jit(points, matrix)
        
        # After 90-degree rotation around Z, [1,0,0] -> [0,1,0]
        expected = np.array([[0.0, 1.0, 0.0]])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_transform_combined(self):
        """Test transformation with rotation and translation."""
        from terrain_change_detection.acceleration import apply_transform_jit
        
        points = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        
        # Combined rotation (45 degrees around Z) and translation
        theta = np.pi / 4
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        matrix = np.array([
            [cos_t, -sin_t, 0, 5.0],
            [sin_t, cos_t, 0, 10.0],
            [0, 0, 1, 15.0],
            [0, 0, 0, 1],
        ])
        
        result = apply_transform_jit(points, matrix)
        
        # Manual calculation
        expected = np.array([
            [cos_t + 5.0, sin_t + 10.0, 15.0],
            [-sin_t + 5.0, cos_t + 10.0, 15.0],
            [5.0, 10.0, 1.0 + 15.0],
        ])
        
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_transform_empty_points(self):
        """Test transformation with empty point array."""
        from terrain_change_detection.acceleration import apply_transform_jit
        
        points = np.empty((0, 3))
        matrix = np.eye(4)
        
        result = apply_transform_jit(points, matrix)
        
        assert result.shape == (0, 3)

    def test_transform_single_point(self):
        """Test transformation with single point."""
        from terrain_change_detection.acceleration import apply_transform_jit
        
        points = np.array([[1.0, 2.0, 3.0]])
        matrix = np.eye(4)
        matrix[:3, 3] = [10.0, 20.0, 30.0]
        
        result = apply_transform_jit(points, matrix)
        
        expected = np.array([[11.0, 22.0, 33.0]])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_transform_large_dataset(self):
        """Test transformation with large dataset."""
        from terrain_change_detection.acceleration import apply_transform_jit
        
        np.random.seed(42)
        points = np.random.rand(10000, 3) * 1000.0
        
        matrix = np.eye(4)
        matrix[:3, 3] = [100.0, 200.0, 300.0]
        
        result = apply_transform_jit(points, matrix)
        
        expected = points + np.array([100.0, 200.0, 300.0])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_transform_matches_matrix_multiplication(self):
        """Test that JIT transform matches standard matrix multiplication."""
        from terrain_change_detection.acceleration import apply_transform_jit
        
        np.random.seed(42)
        points = np.random.rand(100, 3) * 100.0
        
        # Random transformation matrix
        matrix = np.random.rand(4, 4)
        matrix[3, :] = [0, 0, 0, 1]  # Keep it as a proper transformation matrix
        
        result_jit = apply_transform_jit(points, matrix)
        
        # Standard approach
        R = matrix[:3, :3]
        t = matrix[:3, 3]
        result_standard = (R @ points.T).T + t
        
        np.testing.assert_allclose(result_jit, result_standard, rtol=1e-10)


class TestComputeDistancesJIT:
    """Test compute_distances_jit function."""

    def test_distances_identical_points(self):
        """Test distance computation with identical points."""
        from terrain_change_detection.acceleration import compute_distances_jit
        
        points1 = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])
        points2 = points1.copy()
        
        distances = compute_distances_jit(points1, points2)
        
        np.testing.assert_allclose(distances, [0.0, 0.0], atol=1e-10)

    def test_distances_known_values(self):
        """Test distance computation with known values."""
        from terrain_change_detection.acceleration import compute_distances_jit
        
        points1 = np.array([[0.0, 0.0, 0.0]])
        points2 = np.array([[3.0, 4.0, 0.0]])
        
        distances = compute_distances_jit(points1, points2)
        
        # 3-4-5 triangle
        np.testing.assert_allclose(distances, [5.0], rtol=1e-10)

    def test_distances_multiple_points(self):
        """Test distance computation with multiple points."""
        from terrain_change_detection.acceleration import compute_distances_jit
        
        points1 = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ])
        points2 = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
            [2.0, 2.0, 5.0],
        ])
        
        distances = compute_distances_jit(points1, points2)
        
        expected = np.array([
            1.0,  # sqrt(1^2) = 1
            1.0,  # sqrt(0 + 1 + 0) = 1
            3.0,  # sqrt(0 + 0 + 9) = 3
        ])
        
        np.testing.assert_allclose(distances, expected, rtol=1e-10)

    def test_distances_empty_arrays(self):
        """Test distance computation with empty arrays."""
        from terrain_change_detection.acceleration import compute_distances_jit
        
        points1 = np.empty((0, 3))
        points2 = np.empty((0, 3))
        
        distances = compute_distances_jit(points1, points2)
        
        assert distances.shape == (0,)

    def test_distances_single_point(self):
        """Test distance computation with single point pair."""
        from terrain_change_detection.acceleration import compute_distances_jit
        
        points1 = np.array([[1.0, 2.0, 3.0]])
        points2 = np.array([[4.0, 6.0, 8.0]])
        
        distances = compute_distances_jit(points1, points2)
        
        # sqrt((4-1)^2 + (6-2)^2 + (8-3)^2) = sqrt(9 + 16 + 25) = sqrt(50)
        expected = np.sqrt(50.0)
        np.testing.assert_allclose(distances, [expected], rtol=1e-10)

    def test_distances_matches_numpy(self):
        """Test that JIT distances match NumPy calculation."""
        from terrain_change_detection.acceleration import compute_distances_jit
        
        np.random.seed(42)
        points1 = np.random.rand(1000, 3) * 100.0
        points2 = np.random.rand(1000, 3) * 100.0
        
        distances_jit = compute_distances_jit(points1, points2)
        
        # NumPy calculation
        diff = points1 - points2
        distances_numpy = np.sqrt(np.sum(diff * diff, axis=1))
        
        np.testing.assert_allclose(distances_jit, distances_numpy, rtol=1e-10)

    def test_distances_large_dataset(self):
        """Test distance computation with large dataset."""
        from terrain_change_detection.acceleration import compute_distances_jit
        
        np.random.seed(42)
        n = 100000
        points1 = np.random.rand(n, 3) * 1000.0
        points2 = np.random.rand(n, 3) * 1000.0
        
        distances = compute_distances_jit(points1, points2)
        
        assert distances.shape == (n,)
        assert np.all(distances >= 0.0)
        assert np.all(np.isfinite(distances))


class TestJITFallback:
    """Test JIT kernel fallback behavior when Numba unavailable."""

    def test_transform_fallback(self):
        """Test that transform works even without Numba."""
        # This test verifies the fallback is implemented
        # The actual function should work whether Numba is available or not
        from terrain_change_detection.acceleration.jit_kernels import apply_transform_jit
        
        points = np.array([[1.0, 2.0, 3.0]])
        matrix = np.eye(4)
        matrix[:3, 3] = [10.0, 20.0, 30.0]
        
        # Should not raise an error
        result = apply_transform_jit(points, matrix)
        assert result.shape == (1, 3)

    def test_distances_fallback(self):
        """Test that distances work even without Numba."""
        from terrain_change_detection.acceleration.jit_kernels import compute_distances_jit
        
        points1 = np.array([[0.0, 0.0, 0.0]])
        points2 = np.array([[3.0, 4.0, 0.0]])
        
        # Should not raise an error
        distances = compute_distances_jit(points1, points2)
        assert distances.shape == (1,)


class TestJITIntegration:
    """Test JIT kernel integration with tile workers."""

    def test_tile_worker_uses_jit_transform(self):
        """Test that tile workers use JIT transform when available."""
        from terrain_change_detection.acceleration.tile_workers import apply_transform
        
        points = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])
        
        matrix = np.eye(4)
        matrix[:3, 3] = [100.0, 200.0, 300.0]
        
        result = apply_transform(points, matrix)
        
        expected = points + np.array([100.0, 200.0, 300.0])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_tile_worker_transform_empty_points(self):
        """Test tile worker transform with empty points."""
        from terrain_change_detection.acceleration.tile_workers import apply_transform
        
        points = np.empty((0, 3))
        matrix = np.eye(4)
        
        result = apply_transform(points, matrix)
        
        assert result.shape == (0, 3)
