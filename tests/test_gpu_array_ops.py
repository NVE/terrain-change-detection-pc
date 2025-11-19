"""
Tests for GPU array operations abstraction layer.
"""

import numpy as np
import pytest

from terrain_change_detection.acceleration.gpu_array_ops import (
    ArrayBackend,
    get_array_backend,
    reset_array_backend,
    ensure_cpu_array,
    ensure_gpu_array,
    is_gpu_array,
)
from terrain_change_detection.acceleration.hardware_detection import get_gpu_info


@pytest.fixture
def backend():
    """Create fresh backend for each test."""
    reset_array_backend()
    return get_array_backend(use_gpu=True)


def test_backend_initialization(backend):
    """Test backend initializes correctly."""
    assert backend is not None
    assert backend.xp is not None
    
    # Should have numpy at minimum
    assert hasattr(backend.xp, 'array')
    assert hasattr(backend.xp, 'zeros')
    
    gpu_info = get_gpu_info()
    if gpu_info.available:
        assert backend.is_gpu
        print(f"GPU backend active: {gpu_info.device_name}")
    else:
        assert not backend.is_gpu
        print("CPU backend active")


def test_array_creation(backend):
    """Test array creation functions."""
    # Zeros
    arr = backend.zeros((10, 3))
    assert arr.shape == (10, 3)
    
    # Ones
    arr = backend.ones((5, 5))
    assert arr.shape == (5, 5)
    cpu_arr = backend.to_cpu(arr)
    assert np.all(cpu_arr == 1.0)
    
    # Empty
    arr = backend.empty((100,))
    assert arr.shape == (100,)
    
    # Arange
    arr = backend.arange(0, 10, 1)
    cpu_arr = backend.to_cpu(arr)
    assert len(cpu_arr) == 10
    assert cpu_arr[0] == 0
    assert cpu_arr[-1] == 9
    
    # Linspace
    arr = backend.linspace(0, 1, 11)
    cpu_arr = backend.to_cpu(arr)
    assert len(cpu_arr) == 11
    assert cpu_arr[0] == 0.0
    assert cpu_arr[-1] == 1.0


def test_array_conversion(backend):
    """Test CPU/GPU array conversion."""
    # NumPy to backend
    np_arr = np.array([1.0, 2.0, 3.0])
    arr = backend.asarray(np_arr)
    assert arr.shape == (3,)
    
    # Backend to CPU
    cpu_arr = backend.to_cpu(arr)
    assert isinstance(cpu_arr, np.ndarray)
    np.testing.assert_array_equal(cpu_arr, np_arr)
    
    # CPU to GPU (if available)
    gpu_arr = backend.to_gpu(np_arr)
    assert gpu_arr.shape == (3,)
    
    # GPU to CPU roundtrip
    cpu_arr2 = backend.to_cpu(gpu_arr)
    assert isinstance(cpu_arr2, np.ndarray)
    np.testing.assert_array_equal(cpu_arr2, np_arr)


def test_mathematical_operations(backend):
    """Test mathematical operations."""
    arr = backend.asarray(np.array([1.0, 4.0, 9.0, 16.0]))
    
    # Sqrt
    sqrt_arr = backend.sqrt(arr)
    result = backend.to_cpu(sqrt_arr)
    np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0, 4.0])
    
    # Sum
    total = backend.sum(arr)
    total_cpu = backend.to_cpu(total)
    assert float(total_cpu) == 30.0
    
    # Mean
    avg = backend.mean(arr)
    avg_cpu = backend.to_cpu(avg)
    assert float(avg_cpu) == 7.5
    
    # Std
    std = backend.std(arr)
    std_cpu = backend.to_cpu(std)
    assert std_cpu > 0
    
    # Min/Max
    min_val = backend.min(arr)
    max_val = backend.max(arr)
    assert float(backend.to_cpu(min_val)) == 1.0
    assert float(backend.to_cpu(max_val)) == 16.0
    
    # Abs
    neg_arr = backend.asarray(np.array([-1.0, -2.0, 3.0]))
    abs_arr = backend.abs(neg_arr)
    result = backend.to_cpu(abs_arr)
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


def test_array_manipulation(backend):
    """Test array manipulation functions."""
    arr1 = backend.asarray(np.array([1.0, 2.0]))
    arr2 = backend.asarray(np.array([3.0, 4.0]))
    
    # Concatenate
    concat = backend.concatenate([arr1, arr2])
    result = backend.to_cpu(concat)
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0])
    
    # Stack
    stacked = backend.stack([arr1, arr2])
    result = backend.to_cpu(stacked)
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result, [[1.0, 2.0], [3.0, 4.0]])
    
    # Clip
    arr = backend.asarray(np.array([0.0, 5.0, 10.0]))
    clipped = backend.clip(arr, 2.0, 8.0)
    result = backend.to_cpu(clipped)
    np.testing.assert_array_equal(result, [2.0, 5.0, 8.0])


def test_logical_operations(backend):
    """Test logical operations."""
    arr = backend.asarray(np.array([1.0, np.nan, 3.0, np.inf, 5.0]))
    
    # isnan
    nan_mask = backend.isnan(arr)
    result = backend.to_cpu(nan_mask)
    assert result[1] == True
    assert result[0] == False
    
    # isfinite
    finite_mask = backend.isfinite(arr)
    result = backend.to_cpu(finite_mask)
    assert result[0] == True
    assert result[1] == False
    assert result[3] == False
    
    # where
    arr = backend.asarray(np.array([1.0, 2.0, 3.0, 4.0]))
    result = backend.where(arr > 2.0, arr, 0.0)
    result_cpu = backend.to_cpu(result)
    np.testing.assert_array_equal(result_cpu, [0.0, 0.0, 3.0, 4.0])
    
    # any/all
    mask = backend.asarray(np.array([True, False, True]))
    assert backend.to_cpu(backend.any(mask))
    assert not backend.to_cpu(backend.all(mask))


def test_ensure_cpu_array():
    """Test ensure_cpu_array convenience function."""
    np_arr = np.array([1.0, 2.0, 3.0])
    
    # From NumPy
    cpu_arr = ensure_cpu_array(np_arr)
    assert isinstance(cpu_arr, np.ndarray)
    np.testing.assert_array_equal(cpu_arr, np_arr)
    
    # From GPU (if available)
    backend = get_array_backend()
    gpu_arr = backend.to_gpu(np_arr)
    cpu_arr = ensure_cpu_array(gpu_arr)
    assert isinstance(cpu_arr, np.ndarray)
    np.testing.assert_array_equal(cpu_arr, np_arr)


def test_ensure_gpu_array():
    """Test ensure_gpu_array convenience function."""
    np_arr = np.array([1.0, 2.0, 3.0])
    
    gpu_arr = ensure_gpu_array(np_arr)
    assert gpu_arr.shape == (3,)
    
    # Should work with GPU arrays too
    gpu_arr2 = ensure_gpu_array(gpu_arr)
    assert gpu_arr2.shape == (3,)


def test_is_gpu_array():
    """Test is_gpu_array detection."""
    backend = get_array_backend()
    np_arr = np.array([1.0, 2.0, 3.0])
    
    # NumPy array is not GPU
    assert not is_gpu_array(np_arr)
    
    # Backend array depends on GPU availability
    backend_arr = backend.asarray(np_arr)
    if backend.is_gpu:
        assert is_gpu_array(backend_arr)
    else:
        assert not is_gpu_array(backend_arr)


def test_backend_singleton():
    """Test backend singleton behavior."""
    reset_array_backend()
    
    backend1 = get_array_backend()
    backend2 = get_array_backend()
    
    # Should return same instance
    assert backend1 is backend2
    
    reset_array_backend()
    backend3 = get_array_backend()
    
    # Should be new instance after reset
    assert backend3 is not backend1


def test_cpu_only_backend():
    """Test backend with GPU disabled."""
    reset_array_backend()
    backend = get_array_backend(use_gpu=False)
    
    assert not backend.is_gpu
    assert backend.xp is np
    
    # Operations should work on CPU
    arr = backend.zeros((10, 3))
    assert isinstance(arr, np.ndarray)


@pytest.mark.skipif(
    not get_gpu_info().available,
    reason="GPU not available"
)
def test_gpu_operations():
    """Test GPU-specific operations (requires GPU)."""
    reset_array_backend()
    backend = get_array_backend(use_gpu=True)
    
    assert backend.is_gpu
    
    # Create array on GPU
    gpu_arr = backend.zeros((1000, 3))
    assert is_gpu_array(gpu_arr)
    
    # Operations should stay on GPU
    gpu_sum = backend.sum(gpu_arr)
    assert is_gpu_array(gpu_sum)
    
    # Transfer to CPU
    cpu_arr = backend.to_cpu(gpu_arr)
    assert isinstance(cpu_arr, np.ndarray)
    assert cpu_arr.shape == (1000, 3)


@pytest.mark.skipif(
    not get_gpu_info().available,
    reason="GPU not available"
)
def test_large_array_operations():
    """Test operations on larger arrays (requires GPU)."""
    reset_array_backend()
    backend = get_array_backend(use_gpu=True)
    
    # Create large array
    n = 1_000_000
    arr = backend.arange(n, dtype=np.float32)
    
    # Mathematical operations
    sqrt_arr = backend.sqrt(arr)
    mean_val = backend.mean(sqrt_arr)
    
    # Should be on GPU
    assert is_gpu_array(arr)
    assert is_gpu_array(sqrt_arr)
    
    # Verify result
    mean_cpu = float(backend.to_cpu(mean_val))
    assert mean_cpu > 0


def test_dtype_preservation(backend):
    """Test that dtypes are preserved correctly."""
    # Float32
    arr32 = backend.zeros((10,), dtype=np.float32)
    cpu_arr32 = backend.to_cpu(arr32)
    assert cpu_arr32.dtype == np.float32
    
    # Float64
    arr64 = backend.zeros((10,), dtype=np.float64)
    cpu_arr64 = backend.to_cpu(arr64)
    assert cpu_arr64.dtype == np.float64
    
    # Int32
    arr_int = backend.zeros((10,), dtype=np.int32)
    cpu_arr_int = backend.to_cpu(arr_int)
    assert cpu_arr_int.dtype == np.int32


if __name__ == "__main__":
    # Run basic tests
    print("Testing GPU array operations...")
    reset_array_backend()
    backend = get_array_backend()
    
    print(f"Backend initialized: {'GPU' if backend.is_gpu else 'CPU'}")
    
    test_backend_initialization(backend)
    print("✓ Backend initialization")
    
    test_array_creation(backend)
    print("✓ Array creation")
    
    test_array_conversion(backend)
    print("✓ Array conversion")
    
    test_mathematical_operations(backend)
    print("✓ Mathematical operations")
    
    test_array_manipulation(backend)
    print("✓ Array manipulation")
    
    test_logical_operations(backend)
    print("✓ Logical operations")
    
    print("\n✓ All basic tests passed!")
