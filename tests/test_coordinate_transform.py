"""
Unit tests for LocalCoordinateTransform utility.

These tests verify:
- Transform creation from different origin methods
- Round-trip coordinate preservation (to_local -> to_global)
- Handling of large UTM coordinates (realistic Norwegian data)
- Serialization/deserialization
"""

import numpy as np
import pytest
from terrain_change_detection.utils.coordinate_transform import LocalCoordinateTransform


class TestLocalCoordinateTransformCreation:
    """Tests for transform creation methods."""
    
    def test_from_bounds(self):
        """Test transform creation from min bounds."""
        transform = LocalCoordinateTransform.from_bounds(466000.0, 6650000.0)
        
        assert transform.offset_x == 466000.0
        assert transform.offset_y == 6650000.0
        assert transform.offset_z == 0.0
        assert transform.origin_method == "min_bounds"
    
    def test_from_bounds_with_z(self):
        """Test transform creation from bounds including Z."""
        transform = LocalCoordinateTransform.from_bounds(
            466000.0, 6650000.0, 100.0, include_z=True
        )
        
        assert transform.offset_z == 100.0
    
    def test_from_centroid(self):
        """Test transform creation from point cloud centroid."""
        points = np.array([
            [466000.0, 6650000.0, 100.0],
            [466200.0, 6650200.0, 150.0],
            [466100.0, 6650100.0, 125.0],
        ])
        
        transform = LocalCoordinateTransform.from_centroid(points)
        
        assert transform.offset_x == pytest.approx(466100.0)
        assert transform.offset_y == pytest.approx(6650100.0)
        assert transform.offset_z == 0.0  # Z not included by default
        assert transform.origin_method == "centroid"
    
    def test_from_centroid_with_z(self):
        """Test centroid transform with Z offset included."""
        points = np.array([
            [466000.0, 6650000.0, 100.0],
            [466200.0, 6650200.0, 200.0],
        ])
        
        transform = LocalCoordinateTransform.from_centroid(points, include_z=True)
        
        assert transform.offset_z == pytest.approx(150.0)
    
    def test_from_first_point(self):
        """Test transform creation from first point."""
        points = np.array([
            [466123.45, 6651234.56, 178.9],
            [466200.0, 6650200.0, 150.0],
        ])
        
        transform = LocalCoordinateTransform.from_first_point(points)
        
        assert transform.offset_x == 466123.45
        assert transform.offset_y == 6651234.56
        assert transform.offset_z == 0.0
        assert transform.origin_method == "first_point"
    
    def test_from_points_method_selection(self):
        """Test from_points with different method arguments."""
        points = np.array([
            [466100.0, 6650100.0, 100.0],
            [466200.0, 6650200.0, 200.0],
        ])
        
        # min_bounds (default)
        t1 = LocalCoordinateTransform.from_points(points, method="min_bounds")
        assert t1.offset_x == 466100.0
        assert t1.origin_method == "min_bounds"
        
        # centroid
        t2 = LocalCoordinateTransform.from_points(points, method="centroid")
        assert t2.offset_x == pytest.approx(466150.0)
        assert t2.origin_method == "centroid"
        
        # first_point
        t3 = LocalCoordinateTransform.from_points(points, method="first_point")
        assert t3.offset_x == 466100.0
        assert t3.origin_method == "first_point"
    
    def test_identity(self):
        """Test identity transform (no offset)."""
        transform = LocalCoordinateTransform.identity()
        
        assert transform.offset_x == 0.0
        assert transform.offset_y == 0.0
        assert transform.offset_z == 0.0
        assert transform.origin_method == "identity"
    
    def test_empty_points_raises_error(self):
        """Test that empty point arrays raise ValueError."""
        empty = np.empty((0, 3))
        
        with pytest.raises(ValueError, match="empty"):
            LocalCoordinateTransform.from_centroid(empty)
        
        with pytest.raises(ValueError, match="empty"):
            LocalCoordinateTransform.from_first_point(empty)
        
        with pytest.raises(ValueError, match="empty"):
            LocalCoordinateTransform.from_points(empty)
    
    def test_wrong_shape_raises_error(self):
        """Test that wrong point array shapes raise ValueError."""
        wrong_shape = np.array([[1, 2]])  # Only 2 columns
        
        with pytest.raises(ValueError, match="shape"):
            LocalCoordinateTransform.from_centroid(wrong_shape)


class TestLocalCoordinateTransformTransformation:
    """Tests for coordinate transformation methods."""
    
    @pytest.fixture
    def transform(self):
        """Standard transform for testing."""
        return LocalCoordinateTransform.from_bounds(466000.0, 6650000.0)
    
    def test_to_local(self, transform):
        """Test conversion from global to local coordinates."""
        global_pts = np.array([
            [466100.0, 6650200.0, 150.0],
            [466050.0, 6650050.0, 100.0],
        ])
        
        local_pts = transform.to_local(global_pts)
        
        expected = np.array([
            [100.0, 200.0, 150.0],
            [50.0, 50.0, 100.0],
        ])
        np.testing.assert_array_almost_equal(local_pts, expected)
    
    def test_to_global(self, transform):
        """Test conversion from local to global coordinates."""
        local_pts = np.array([
            [100.0, 200.0, 150.0],
            [50.0, 50.0, 100.0],
        ])
        
        global_pts = transform.to_global(local_pts)
        
        expected = np.array([
            [466100.0, 6650200.0, 150.0],
            [466050.0, 6650050.0, 100.0],
        ])
        np.testing.assert_array_almost_equal(global_pts, expected)
    
    def test_round_trip_precision(self, transform):
        """Verify to_local -> to_global preserves coordinates exactly."""
        # Use typical Norwegian UTM coordinates with high precision
        original = np.array([
            [466123.456789, 6651234.567890, 178.901234],
            [466789.012345, 6659876.543210, 99.999999],
        ])
        
        # Round-trip transformation
        local = transform.to_local(original)
        restored = transform.to_global(local)
        
        # Should match exactly (within float64 precision)
        np.testing.assert_array_almost_equal(restored, original, decimal=9)
    
    def test_large_coordinates_precision(self):
        """Test with realistic large UTM coordinates to verify precision."""
        # Typical Norwegian LiDAR coordinates (EPSG:25833)
        points = np.array([
            [466000.123, 6650000.456, 150.789],
            [467000.987, 6651000.654, 160.321],
            [465000.555, 6649000.999, 140.111],
        ], dtype=np.float64)
        
        # Create transform from bounds
        transform = LocalCoordinateTransform.from_bounds(
            float(points[:, 0].min()),
            float(points[:, 1].min()),
        )
        
        # Transform to local
        local = transform.to_local(points)
        
        # Local coordinates should be small (near zero)
        assert local[:, 0].min() >= 0
        assert local[:, 0].max() < 3000  # ~2km range
        assert local[:, 1].min() >= 0
        assert local[:, 1].max() < 3000
        
        # Z should be unchanged
        np.testing.assert_array_almost_equal(local[:, 2], points[:, 2])
        
        # Round-trip should be exact
        restored = transform.to_global(local)
        np.testing.assert_array_almost_equal(restored, points, decimal=9)
    
    def test_empty_array_handling(self, transform):
        """Test that empty arrays are handled gracefully."""
        empty = np.empty((0, 3), dtype=np.float64)
        
        local = transform.to_local(empty)
        assert local.shape == (0, 3)
        
        global_pts = transform.to_global(empty)
        assert global_pts.shape == (0, 3)
    
    def test_identity_transform(self):
        """Test that identity transform doesn't modify coordinates."""
        transform = LocalCoordinateTransform.identity()
        
        points = np.array([
            [466000.0, 6650000.0, 150.0],
            [467000.0, 6651000.0, 160.0],
        ])
        
        local = transform.to_local(points)
        np.testing.assert_array_equal(local, points)
        
        restored = transform.to_global(local)
        np.testing.assert_array_equal(restored, points)


class TestLocalCoordinateTransformBounds:
    """Tests for bounds transformation."""
    
    def test_transform_bounds_to_local(self):
        """Test transforming 2D bounds to local coordinates."""
        transform = LocalCoordinateTransform.from_bounds(466000.0, 6650000.0)
        
        local_bounds = transform.transform_bounds(
            466100.0, 6650100.0, 467000.0, 6651000.0,
            to_local=True
        )
        
        assert local_bounds == (100.0, 100.0, 1000.0, 1000.0)
    
    def test_transform_bounds_to_global(self):
        """Test transforming 2D bounds to global coordinates."""
        transform = LocalCoordinateTransform.from_bounds(466000.0, 6650000.0)
        
        global_bounds = transform.transform_bounds(
            100.0, 100.0, 1000.0, 1000.0,
            to_local=False
        )
        
        assert global_bounds == (466100.0, 6650100.0, 467000.0, 6651000.0)


class TestLocalCoordinateTransformSerialization:
    """Tests for serialization and deserialization."""
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        transform = LocalCoordinateTransform.from_bounds(466000.0, 6650000.0, 100.0)
        
        data = transform.to_dict()
        
        assert data["offset_x"] == 466000.0
        assert data["offset_y"] == 6650000.0
        assert data["offset_z"] == 0.0  # include_z was False
        assert data["origin_method"] == "min_bounds"
    
    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "offset_x": 466000.0,
            "offset_y": 6650000.0,
            "offset_z": 50.0,
            "origin_method": "centroid",
        }
        
        transform = LocalCoordinateTransform.from_dict(data)
        
        assert transform.offset_x == 466000.0
        assert transform.offset_y == 6650000.0
        assert transform.offset_z == 50.0
        assert transform.origin_method == "centroid"
    
    def test_serialization_round_trip(self):
        """Test that to_dict -> from_dict preserves all values."""
        original = LocalCoordinateTransform(
            offset_x=466123.456,
            offset_y=6651234.567,
            offset_z=178.9,
            origin_method="min_bounds",
        )
        
        restored = LocalCoordinateTransform.from_dict(original.to_dict())
        
        assert restored.offset_x == original.offset_x
        assert restored.offset_y == original.offset_y
        assert restored.offset_z == original.offset_z
        assert restored.origin_method == original.origin_method
    
    def test_from_dict_missing_optional_fields(self):
        """Test deserialization with missing optional fields."""
        data = {
            "offset_x": 466000.0,
            "offset_y": 6650000.0,
        }
        
        transform = LocalCoordinateTransform.from_dict(data)
        
        assert transform.offset_z == 0.0
        assert transform.origin_method == "unknown"


class TestLocalCoordinateTransformStringRepresentation:
    """Tests for string representation."""
    
    def test_str_without_z(self):
        """Test string representation without Z offset."""
        transform = LocalCoordinateTransform.from_bounds(466000.0, 6650000.0)
        
        s = str(transform)
        
        assert "466000" in s
        assert "6650000" in s
        assert "min_bounds" in s
    
    def test_str_with_z(self):
        """Test string representation with Z offset."""
        transform = LocalCoordinateTransform(
            offset_x=466000.0,
            offset_y=6650000.0,
            offset_z=100.0,
            origin_method="centroid",
        )
        
        s = str(transform)
        
        assert "100" in s
        assert "centroid" in s
