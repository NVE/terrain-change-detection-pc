"""
Tests for GPU-accelerated C2C change detection integration.

Validates that GPU C2C produces numerically identical results to CPU C2C
across all detection modes (basic, vertical plane, streaming/tiled).
"""

import numpy as np
import pytest

from terrain_change_detection.detection.change_detection import ChangeDetector
from terrain_change_detection.utils.config import AppConfig
from terrain_change_detection.acceleration import get_gpu_info


@pytest.fixture
def gpu_available():
    """Check if GPU is available for testing."""
    info = get_gpu_info()
    return info.available


@pytest.fixture
def sample_clouds():
    """Generate sample point clouds for testing."""
    np.random.seed(42)
    
    # Source cloud (100 points)
    source = np.random.rand(100, 3) * 10.0
    
    # Target cloud (120 points, slightly offset)
    target = np.random.rand(120, 3) * 10.0 + np.array([0.1, 0.1, 0.05])
    
    return source, target


@pytest.fixture
def large_sample_clouds():
    """Generate larger point clouds for performance testing."""
    np.random.seed(42)
    
    # Source cloud (10K points)
    source = np.random.rand(10_000, 3) * 100.0
    
    # Target cloud (12K points, slightly offset)
    target = np.random.rand(12_000, 3) * 100.0 + np.array([0.5, 0.5, 0.2])
    
    return source, target


@pytest.fixture
def gpu_config():
    """Configuration with GPU enabled."""
    return AppConfig(gpu=AppConfig.GPUConfig(enabled=True, use_for_c2c=True))


@pytest.fixture
def cpu_config():
    """Configuration with GPU disabled."""
    return AppConfig(gpu=AppConfig.GPUConfig(enabled=False, use_for_c2c=False))


class TestBasicC2CGPU:
    """Test basic C2C GPU integration."""
    
    def test_gpu_cpu_parity_basic_c2c(self, sample_clouds, gpu_config, cpu_config, gpu_available):
        """Verify GPU and CPU produce identical results for basic C2C."""
        if not gpu_available:
            pytest.skip("GPU not available")
        
        source, target = sample_clouds
        
        # Compute with CPU
        cpu_result = ChangeDetector.compute_c2c(source, target, config=cpu_config)
        
        # Compute with GPU
        gpu_result = ChangeDetector.compute_c2c(source, target, config=gpu_config)
        
        # Check metadata
        assert cpu_result.metadata.get("gpu_used") is False
        assert gpu_result.metadata.get("gpu_used") is True
        
        # Check numerical parity (distances should match within floating point tolerance)
        np.testing.assert_allclose(gpu_result.distances, cpu_result.distances, rtol=1e-5)
        
        # Check statistics match
        assert abs(gpu_result.rmse - cpu_result.rmse) < 1e-5
        assert abs(gpu_result.mean - cpu_result.mean) < 1e-5
        assert abs(gpu_result.median - cpu_result.median) < 1e-5
        assert gpu_result.n == cpu_result.n
    
    def test_gpu_c2c_with_max_distance(self, sample_clouds, gpu_config, gpu_available):
        """Test GPU C2C with distance filtering."""
        if not gpu_available:
            pytest.skip("GPU not available")
        
        source, target = sample_clouds
        max_distance = 1.0
        
        result = ChangeDetector.compute_c2c(
            source, target, max_distance=max_distance, config=gpu_config
        )
        
        assert result.metadata.get("gpu_used") is True
        assert result.metadata.get("max_distance") == max_distance
        
        # The function returns all distances, but statistics only count those within threshold
        # Verify that result.n counts only distances within max_distance
        distances_within = np.sum(result.distances <= max_distance)
        assert result.n == distances_within
    
    def test_gpu_c2c_fallback_to_cpu(self, sample_clouds, cpu_config):
        """Test that C2C works with GPU disabled (CPU fallback)."""
        source, target = sample_clouds
        
        result = ChangeDetector.compute_c2c(source, target, config=cpu_config)
        
        assert result.metadata.get("gpu_used") is False
        assert result.distances.shape[0] == source.shape[0]
        assert np.all(np.isfinite(result.distances))
    
    def test_gpu_c2c_empty_clouds(self, gpu_config, gpu_available):
        """Test GPU C2C handles empty clouds gracefully."""
        if not gpu_available:
            pytest.skip("GPU not available")
        
        source = np.random.rand(100, 3)
        empty_target = np.empty((0, 3))
        
        with pytest.raises(ValueError, match="Input point arrays must be non-empty"):
            ChangeDetector.compute_c2c(source, empty_target, config=gpu_config)


class TestVerticalPlaneC2CGPU:
    """Test vertical plane C2C GPU integration."""
    
    def test_gpu_cpu_parity_vertical_plane(self, sample_clouds, gpu_config, cpu_config, gpu_available):
        """Verify GPU and CPU produce identical results for vertical plane C2C."""
        if not gpu_available:
            pytest.skip("GPU not available")
        
        source, target = sample_clouds
        
        # Compute with CPU
        cpu_result = ChangeDetector.compute_c2c_vertical_plane(
            source, target, k_neighbors=20, config=cpu_config
        )
        
        # Compute with GPU
        gpu_result = ChangeDetector.compute_c2c_vertical_plane(
            source, target, k_neighbors=20, config=gpu_config
        )
        
        # Check metadata
        assert cpu_result.metadata.get("gpu_used") is False
        assert gpu_result.metadata.get("gpu_used") is True
        assert cpu_result.metadata.get("mode") == "vertical_plane"
        assert gpu_result.metadata.get("mode") == "vertical_plane"
        
        # Check numerical parity (vertical distances should match)
        # Note: Small differences may occur due to neighbor selection in edge cases
        valid_cpu = np.isfinite(cpu_result.distances)
        valid_gpu = np.isfinite(gpu_result.distances)
        assert np.sum(valid_cpu) == np.sum(valid_gpu), "Different number of valid results"
        
        # Check valid distances match (where both are finite)
        both_valid = valid_cpu & valid_gpu
        if np.any(both_valid):
            np.testing.assert_allclose(
                gpu_result.distances[both_valid],
                cpu_result.distances[both_valid],
                rtol=1e-4,  # Slightly larger tolerance for plane fitting
            )
    
    def test_gpu_vertical_plane_with_radius(self, sample_clouds, gpu_config, gpu_available):
        """Test GPU vertical plane C2C with radius neighborhoods."""
        if not gpu_available:
            pytest.skip("GPU not available")
        
        source, target = sample_clouds
        radius = 2.0
        
        result = ChangeDetector.compute_c2c_vertical_plane(
            source, target, radius=radius, min_neighbors=6, config=gpu_config
        )
        
        assert result.metadata.get("gpu_used") is True
        assert result.metadata.get("radius") == radius
        assert result.distances.shape[0] == source.shape[0]
    
    def test_gpu_vertical_plane_knn(self, sample_clouds, gpu_config, gpu_available):
        """Test GPU vertical plane C2C with k-NN neighborhoods."""
        if not gpu_available:
            pytest.skip("GPU not available")
        
        source, target = sample_clouds
        k_neighbors = 15
        
        result = ChangeDetector.compute_c2c_vertical_plane(
            source, target, k_neighbors=k_neighbors, config=gpu_config
        )
        
        assert result.metadata.get("gpu_used") is True
        assert result.metadata.get("k_neighbors") == k_neighbors
        assert result.distances.shape[0] == source.shape[0]


class TestPerformance:
    """Performance benchmarks for GPU vs CPU C2C."""
    
    def test_gpu_speedup_basic_c2c(self, large_sample_clouds, gpu_config, cpu_config, gpu_available):
        """Benchmark GPU speedup for basic C2C (informational, not a strict test)."""
        if not gpu_available:
            pytest.skip("GPU not available")
        
        source, target = large_sample_clouds
        
        import time
        
        # Warm up GPU
        _ = ChangeDetector.compute_c2c(source[:100], target[:100], config=gpu_config)
        
        # CPU timing
        t0 = time.time()
        cpu_result = ChangeDetector.compute_c2c(source, target, config=cpu_config)
        cpu_time = time.time() - t0
        
        # GPU timing
        t0 = time.time()
        gpu_result = ChangeDetector.compute_c2c(source, target, config=gpu_config)
        gpu_time = time.time() - t0
        
        # Just verify results match within reasonable floating-point tolerance
        np.testing.assert_allclose(gpu_result.distances, cpu_result.distances, rtol=1e-4)
        
        # Log performance info (informational only, no strict assertions)
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"\nC2C Performance (10K points):")
        print(f"  CPU time: {cpu_time:.4f}s")
        print(f"  GPU time: {gpu_time:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Note: GPU may be slower on small datasets due to overhead
        # This is expected and acceptable for datasets < 100K points
    
    def test_gpu_memory_efficiency(self, gpu_available):
        """Test that GPU handles various point cloud sizes."""
        if not gpu_available:
            pytest.skip("GPU not available")
        
        from terrain_change_detection.acceleration import get_gpu_info
        
        gpu_info = get_gpu_info()
        config = AppConfig(gpu=AppConfig.GPUConfig(enabled=True, use_for_c2c=True))
        
        # Test different point cloud sizes
        sizes = [100, 1_000, 10_000, 50_000]
        
        for size in sizes:
            source = np.random.rand(size, 3) * 100.0
            target = np.random.rand(size, 3) * 100.0
            
            result = ChangeDetector.compute_c2c(source, target, config=config)
            
            assert result.metadata.get("gpu_used") is True
            assert result.distances.shape[0] == size
            print(f"  ✓ Processed {size:,} points successfully on GPU")


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_gpu_c2c_single_point(self, gpu_config, gpu_available):
        """Test GPU C2C with single-point clouds."""
        if not gpu_available:
            pytest.skip("GPU not available")
        
        source = np.array([[1.0, 2.0, 3.0]])
        target = np.array([[1.1, 2.1, 3.1]])
        
        result = ChangeDetector.compute_c2c(source, target, config=gpu_config)
        
        assert result.distances.shape[0] == 1
        assert np.isfinite(result.distances[0])
    
    def test_gpu_c2c_identical_clouds(self, sample_clouds, gpu_config, gpu_available):
        """Test GPU C2C with identical source and target."""
        if not gpu_available:
            pytest.skip("GPU not available")
        
        source, _ = sample_clouds
        target = source.copy()
        
        result = ChangeDetector.compute_c2c(source, target, config=gpu_config)
        
        # Distances should be very close to zero
        assert np.all(result.distances < 1e-6)
    
    def test_gpu_c2c_no_valid_neighbors(self, gpu_config, gpu_available):
        """Test GPU C2C when no neighbors within max_distance."""
        if not gpu_available:
            pytest.skip("GPU not available")
        
        # Clouds far apart
        source = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        target = np.array([[100.0, 100.0, 100.0], [101.0, 101.0, 101.0]])
        
        result = ChangeDetector.compute_c2c(
            source, target, max_distance=1.0, config=gpu_config
        )
        
        # Should have no valid neighbors
        assert result.n == 0
        assert np.isinf(result.rmse) or np.isnan(result.mean)


@pytest.mark.skipif(not get_gpu_info().available, reason="GPU not available")
class TestConfigIntegration:
    """Test configuration integration."""
    
    def test_config_enables_gpu(self, sample_clouds):
        """Test that GPU config properly enables GPU."""
        source, target = sample_clouds
        
        config = AppConfig(gpu=AppConfig.GPUConfig(enabled=True, use_for_c2c=True))
        result = ChangeDetector.compute_c2c(source, target, config=config)
        
        assert result.metadata.get("gpu_used") is True
    
    def test_config_disables_gpu(self, sample_clouds):
        """Test that disabling GPU config forces CPU."""
        source, target = sample_clouds
        
        config = AppConfig(gpu=AppConfig.GPUConfig(enabled=False, use_for_c2c=False))
        result = ChangeDetector.compute_c2c(source, target, config=config)
        
        assert result.metadata.get("gpu_used") is False
    
    def test_config_use_for_c2c_flag(self, sample_clouds):
        """Test that use_for_c2c flag is respected."""
        source, target = sample_clouds
        
        # GPU enabled but not for C2C
        config = AppConfig(gpu=AppConfig.GPUConfig(enabled=True, use_for_c2c=False))
        result = ChangeDetector.compute_c2c(source, target, config=config)
        
        assert result.metadata.get("gpu_used") is False
    
    def test_no_config_defaults_to_cpu(self, sample_clouds):
        """Test that missing config defaults to CPU."""
        source, target = sample_clouds
        
        result = ChangeDetector.compute_c2c(source, target, config=None)
        
        assert result.metadata.get("gpu_used") is False


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
