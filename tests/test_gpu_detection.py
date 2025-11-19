"""
Tests for GPU hardware detection.
"""

import pytest
from terrain_change_detection.acceleration.hardware_detection import (
    GPUInfo,
    detect_gpu,
    get_gpu_info,
    check_gpu_memory,
    get_optimal_batch_size,
    clear_gpu_cache,
)


def test_detect_gpu():
    """Test GPU detection returns valid GPUInfo."""
    info = detect_gpu()
    
    assert isinstance(info, GPUInfo)
    assert isinstance(info.available, bool)
    assert isinstance(info.device_count, int)
    assert info.device_count >= 0
    
    if info.available:
        # GPU is available
        assert info.device_count > 0
        assert info.device_name is not None
        assert info.memory_gb is not None
        assert info.memory_gb > 0
        assert info.cuda_version is not None
        assert info.compute_capability is not None
        assert len(info.compute_capability) == 2
        assert info.error_message is None
        print(f"GPU detected: {info.device_name}, {info.memory_gb:.1f} GB")
    else:
        # GPU is not available
        assert info.device_count == 0
        assert info.error_message is not None
        print(f"GPU not available: {info.error_message}")


def test_get_gpu_info_caching():
    """Test that get_gpu_info() returns cached result."""
    clear_gpu_cache()
    
    info1 = get_gpu_info()
    info2 = get_gpu_info()
    
    # Should return same object (cached)
    assert info1 is info2
    
    clear_gpu_cache()


def test_check_gpu_memory():
    """Test GPU memory checking."""
    has_memory, available_gb = check_gpu_memory(1.0)  # Require 1 GB
    
    assert isinstance(has_memory, bool)
    
    if has_memory:
        assert available_gb is not None
        assert available_gb >= 1.0
        print(f"GPU has sufficient memory: {available_gb:.1f} GB available")
    else:
        print("GPU memory check: insufficient or unavailable")


def test_get_optimal_batch_size():
    """Test optimal batch size calculation."""
    point_count = 1_000_000
    
    batch_size = get_optimal_batch_size(point_count)
    
    assert isinstance(batch_size, int)
    assert batch_size > 0
    assert batch_size <= point_count
    
    # Minimum batch size should be respected
    small_batch = get_optimal_batch_size(100)
    assert small_batch >= 100
    
    print(f"Optimal batch size for {point_count:,} points: {batch_size:,}")


def test_gpu_info_graceful_degradation():
    """Test that GPU detection gracefully handles errors."""
    # This should not raise exceptions even if GPU is unavailable
    info = detect_gpu()
    assert isinstance(info, GPUInfo)
    
    # Memory check should not raise
    has_mem, avail = check_gpu_memory(1.0)
    assert isinstance(has_mem, bool)
    
    # Batch size should not raise
    batch = get_optimal_batch_size(1000)
    assert batch > 0


@pytest.mark.skipif(
    not get_gpu_info().available,
    reason="GPU not available"
)
def test_gpu_compute_capability():
    """Test GPU compute capability is valid (requires GPU)."""
    info = get_gpu_info()
    
    assert info.compute_capability is not None
    major, minor = info.compute_capability
    
    # CUDA compute capability should be reasonable (>= 3.5)
    assert major >= 3
    assert 0 <= minor <= 9
    
    print(f"Compute capability: {major}.{minor}")


@pytest.mark.skipif(
    not get_gpu_info().available,
    reason="GPU not available"
)
def test_gpu_memory_info():
    """Test GPU memory information is valid (requires GPU)."""
    info = get_gpu_info()
    
    assert info.memory_gb is not None
    assert info.memory_gb > 0
    # Typical GPUs have at least 1GB, most have 4GB+
    assert info.memory_gb >= 1.0
    
    print(f"GPU memory: {info.memory_gb:.1f} GB")


if __name__ == "__main__":
    # Run basic tests
    print("Testing GPU detection...")
    test_detect_gpu()
    print("\nTesting GPU info caching...")
    test_get_gpu_info_caching()
    print("\nTesting GPU memory check...")
    test_check_gpu_memory()
    print("\nTesting batch size calculation...")
    test_get_optimal_batch_size()
    print("\n✓ All basic tests passed")
