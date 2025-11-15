"""
Tests for GPU-accelerated nearest neighbors.
"""

import numpy as np
import pytest

from terrain_change_detection.acceleration.gpu_neighbors import (
    GPUNearestNeighbors,
    create_gpu_neighbors,
)
from terrain_change_detection.acceleration.hardware_detection import get_gpu_info


@pytest.fixture
def sample_data():
    """Create sample point cloud data for testing."""
    np.random.seed(42)
    X_train = np.random.randn(1000, 3).astype(np.float32)
    X_query = np.random.randn(100, 3).astype(np.float32)
    return X_train, X_query


def test_cpu_backend(sample_data):
    """Test nearest neighbors with CPU backend."""
    X_train, X_query = sample_data
    
    nn = create_gpu_neighbors(n_neighbors=5, use_gpu=False)
    assert nn.backend_ == 'sklearn-cpu'
    assert not nn.gpu_available_
    
    nn.fit(X_train)
    assert nn.n_samples_fit_ == 1000
    
    distances, indices = nn.kneighbors(X_query)
    assert distances.shape == (100, 5)
    assert indices.shape == (100, 5)
    assert distances.min() >= 0
    assert indices.max() < 1000


def test_gpu_backend_initialization(sample_data):
    """Test GPU backend initialization."""
    X_train, X_query = sample_data
    
    nn = create_gpu_neighbors(n_neighbors=5, use_gpu=True)
    nn.fit(X_train)  # Initialize backend by fitting
    
    gpu_info = get_gpu_info()
    if gpu_info.available:
        # Should use sklearn-gpu on Windows (cuML not available)
        assert nn.gpu_available_ and nn.backend_ == 'sklearn-gpu'
    else:
        # Should fall back to CPU
        assert nn.backend_ == 'sklearn-cpu'
        assert not nn.gpu_available_


def test_kneighbors_basic(sample_data):
    """Test basic k-neighbors query."""
    X_train, X_query = sample_data
    
    nn = GPUNearestNeighbors(n_neighbors=10, use_gpu=True)
    nn.fit(X_train)
    
    distances, indices = nn.kneighbors(X_query, n_neighbors=5)
    
    assert distances.shape == (100, 5)
    assert indices.shape == (100, 5)
    
    # Distances should be sorted ascending
    assert np.all(np.diff(distances, axis=1) >= 0)
    
    # Indices should be valid
    assert indices.min() >= 0
    assert indices.max() < 1000


def test_kneighbors_self_query(sample_data):
    """Test k-neighbors on training data itself."""
    X_train, _ = sample_data
    
    nn = GPUNearestNeighbors(n_neighbors=5, use_gpu=True)
    nn.fit(X_train)
    
    # Query training data (first neighbor should be self with distance ~0)
    distances, indices = nn.kneighbors(X_train[:10])
    
    assert distances.shape == (10, 5)
    
    # First neighbor should be self
    assert np.all(indices[:, 0] == np.arange(10))
    assert np.allclose(distances[:, 0], 0.0, atol=1e-6)


def test_kneighbors_return_distance_false(sample_data):
    """Test k-neighbors without returning distances."""
    X_train, X_query = sample_data
    
    nn = GPUNearestNeighbors(n_neighbors=5)
    nn.fit(X_train)
    
    indices = nn.kneighbors(X_query, return_distance=False)
    
    assert isinstance(indices, np.ndarray)
    assert indices.shape == (100, 5)


def test_kneighbors_custom_k(sample_data):
    """Test k-neighbors with custom k parameter."""
    X_train, X_query = sample_data
    
    nn = GPUNearestNeighbors(n_neighbors=5)
    nn.fit(X_train)
    
    # Query with different k
    distances, indices = nn.kneighbors(X_query, n_neighbors=15)
    
    assert distances.shape == (100, 15)
    assert indices.shape == (100, 15)


def test_radius_neighbors_cpu(sample_data):
    """Test radius neighbors (CPU only for now)."""
    X_train, X_query = sample_data
    
    nn = GPUNearestNeighbors(n_neighbors=5, radius=1.0, use_gpu=False)
    nn.fit(X_train)
    
    distances, indices = nn.radius_neighbors(X_query[:10])
    
    # radius_neighbors returns lists of arrays (variable length results)
    assert isinstance(distances, (np.ndarray, list))
    assert isinstance(indices, (np.ndarray, list))
    
    # Check that all distances are within radius
    if isinstance(distances, list):
        for dist_array in distances:
            assert np.all(dist_array <= 1.0)
    else:
        # If ndarray, check all values
        for dist_array in distances:
            assert np.all(dist_array <= 1.0)


def test_different_algorithms(sample_data):
    """Test different tree algorithms."""
    X_train, X_query = sample_data
    
    algorithms = ['auto', 'ball_tree', 'kd_tree']
    results = []
    
    for algo in algorithms:
        nn = GPUNearestNeighbors(n_neighbors=5, algorithm=algo, use_gpu=False)
        nn.fit(X_train)
        distances, indices = nn.kneighbors(X_query[:10])
        results.append((distances, indices))
    
    # All algorithms should give same results
    for i in range(1, len(results)):
        np.testing.assert_array_almost_equal(results[0][0], results[i][0], decimal=5)
        np.testing.assert_array_equal(results[0][1], results[i][1])


def test_fit_before_query_error():
    """Test that querying before fitting raises error."""
    nn = GPUNearestNeighbors(n_neighbors=5)
    
    with pytest.raises(ValueError, match="must be fitted"):
        nn.kneighbors(np.random.randn(10, 3))


def test_factory_function():
    """Test create_gpu_neighbors factory function."""
    nn = create_gpu_neighbors(n_neighbors=10, algorithm='kd_tree', use_gpu=False)
    
    assert isinstance(nn, GPUNearestNeighbors)
    assert nn.n_neighbors == 10
    assert nn.algorithm == 'kd_tree'
    assert not nn.use_gpu


@pytest.mark.skipif(
    not get_gpu_info().available,
    reason="GPU not available"
)
def test_gpu_vs_cpu_consistency(sample_data):
    """Test that GPU and CPU backends give consistent results."""
    X_train, X_query = sample_data
    
    # CPU version
    nn_cpu = GPUNearestNeighbors(n_neighbors=5, use_gpu=False)
    nn_cpu.fit(X_train)
    dist_cpu, idx_cpu = nn_cpu.kneighbors(X_query[:10])
    
    # GPU version
    nn_gpu = GPUNearestNeighbors(n_neighbors=5, use_gpu=True)
    nn_gpu.fit(X_train)
    dist_gpu, idx_gpu = nn_gpu.kneighbors(X_query[:10])
    
    # Results should be very similar (allowing for floating point differences)
    np.testing.assert_array_almost_equal(dist_cpu, dist_gpu, decimal=4)
    # Indices might differ slightly if there are ties, but should be mostly same
    agreement = (idx_cpu == idx_gpu).mean()
    assert agreement > 0.95, f"Only {agreement:.2%} agreement between CPU and GPU"


@pytest.mark.skipif(
    not get_gpu_info().available,
    reason="GPU not available"
)
def test_large_scale_performance(sample_data):
    """Test performance with larger dataset."""
    # Create larger dataset
    np.random.seed(42)
    X_large = np.random.randn(50000, 3).astype(np.float32)
    X_query_large = np.random.randn(5000, 3).astype(np.float32)
    
    nn = GPUNearestNeighbors(n_neighbors=10, use_gpu=True)
    nn.fit(X_large)
    
    distances, indices = nn.kneighbors(X_query_large)
    
    assert distances.shape == (5000, 10)
    assert indices.shape == (5000, 10)
    assert distances.min() >= 0
    print(f"\nGPU backend: {nn.backend_}")
    print(f"Mean distance: {distances.mean():.4f}")


def test_edge_cases():
    """Test edge cases."""
    # Small dataset
    X_small = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
    
    nn = GPUNearestNeighbors(n_neighbors=2)
    nn.fit(X_small)
    
    distances, indices = nn.kneighbors(X_small)
    assert distances.shape == (3, 2)
    
    # Single query point
    X_single = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    distances, indices = nn.kneighbors(X_single)
    assert distances.shape == (1, 2)


if __name__ == "__main__":
    # Run basic tests
    print("Testing GPU Nearest Neighbors...")
    
    np.random.seed(42)
    X_train = np.random.randn(1000, 3).astype(np.float32)
    X_query = np.random.randn(100, 3).astype(np.float32)
    data = (X_train, X_query)
    
    test_cpu_backend(data)
    print("✓ CPU backend")
    
    test_gpu_backend_initialization(data)
    print("✓ GPU backend initialization")
    
    test_kneighbors_basic(data)
    print("✓ K-neighbors basic")
    
    test_kneighbors_self_query(data)
    print("✓ K-neighbors self query")
    
    test_different_algorithms(data)
    print("✓ Different algorithms")
    
    print("\n✓ All basic tests passed!")
