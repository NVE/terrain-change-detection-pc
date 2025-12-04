"""
Unit tests for the area clipping module.

Tests cover:
- AreaClipper creation from various sources (GeoJSON, polygon coords, bounds)
- Point cloud clipping with various geometries
- Edge cases and error handling
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Check if shapely is available
try:
    from shapely.geometry import Polygon, MultiPolygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

# Import clipping module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from terrain_change_detection.preprocessing.clipping import (
    AreaClipper,
    check_shapely_available,
    check_fiona_available,
)


# Skip all tests if shapely is not available
pytestmark = pytest.mark.skipif(
    not SHAPELY_AVAILABLE,
    reason="shapely is required for clipping tests"
)


class TestAreaClipperCreation:
    """Tests for creating AreaClipper instances from various sources."""
    
    def test_from_polygon_simple_rectangle(self):
        """Test creating clipper from a simple rectangular polygon."""
        coords = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]
        clipper = AreaClipper.from_polygon(coords)
        
        assert clipper is not None
        assert clipper.bounds == (0, 0, 100, 100)
        assert abs(clipper.area - 10000) < 1e-6  # 100 * 100 = 10000
    
    def test_from_polygon_with_hole(self):
        """Test creating clipper from a polygon with a hole."""
        exterior = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]
        hole = [(25, 25), (75, 25), (75, 75), (25, 75), (25, 25)]
        clipper = AreaClipper.from_polygon(exterior, holes=[hole])
        
        assert clipper is not None
        expected_area = 10000 - 2500  # Outer - hole
        assert abs(clipper.area - expected_area) < 1e-6
    
    def test_from_bounds(self):
        """Test creating clipper from bounding box coordinates."""
        clipper = AreaClipper.from_bounds(10, 20, 110, 120)
        
        assert clipper is not None
        assert clipper.bounds == (10, 20, 110, 120)
        assert abs(clipper.area - 10000) < 1e-6  # 100 * 100 = 10000
    
    def test_from_geojson_file_polygon(self):
        """Test loading clipper from a GeoJSON file with a single polygon."""
        geojson = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[(0, 0), (50, 0), (50, 50), (0, 50), (0, 0)]]
            },
            "properties": {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as f:
            json.dump(geojson, f)
            temp_path = f.name
        
        try:
            clipper = AreaClipper.from_file(temp_path)
            assert clipper is not None
            assert abs(clipper.area - 2500) < 1e-6  # 50 * 50
        finally:
            Path(temp_path).unlink()
    
    def test_from_geojson_file_featurecollection(self):
        """Test loading clipper from a GeoJSON FeatureCollection."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]]
                    },
                    "properties": {"name": "area1"}
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[(20, 20), (30, 20), (30, 30), (20, 30), (20, 20)]]
                    },
                    "properties": {"name": "area2"}
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as f:
            json.dump(geojson, f)
            temp_path = f.name
        
        try:
            clipper = AreaClipper.from_file(temp_path)
            assert clipper is not None
            # Combined area of two 10x10 squares
            assert abs(clipper.area - 200) < 1e-6
        finally:
            Path(temp_path).unlink()
    
    def test_from_geojson_file_not_found(self):
        """Test error handling when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            AreaClipper.from_file("nonexistent_file.geojson")
    
    def test_from_geojson_unsupported_format(self):
        """Test error handling for unsupported file formats."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("not a geojson")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                AreaClipper.from_file(temp_path)
        finally:
            Path(temp_path).unlink()


class TestPointClipping:
    """Tests for clipping point clouds."""
    
    @pytest.fixture
    def simple_clipper(self):
        """Create a simple rectangular clipper for testing."""
        return AreaClipper.from_bounds(10, 10, 50, 50)
    
    @pytest.fixture
    def sample_points(self):
        """Create sample point cloud with known positions."""
        # Create a grid of points from 0 to 60 in both X and Y
        x = np.linspace(0, 60, 13)  # 0, 5, 10, ..., 60
        y = np.linspace(0, 60, 13)
        xx, yy = np.meshgrid(x, y)
        z = np.zeros_like(xx)  # Flat surface
        
        return np.column_stack([xx.ravel(), yy.ravel(), z.ravel()])
    
    def test_clip_basic(self, simple_clipper, sample_points):
        """Test basic clipping of points."""
        clipped = simple_clipper.clip(sample_points)
        
        # All clipped points should be inside the bounds
        assert np.all(clipped[:, 0] >= 10)
        assert np.all(clipped[:, 0] <= 50)
        assert np.all(clipped[:, 1] >= 10)
        assert np.all(clipped[:, 1] <= 50)
    
    def test_clip_return_mask(self, simple_clipper, sample_points):
        """Test clipping with return_mask=True."""
        clipped, mask = simple_clipper.clip(sample_points, return_mask=True)
        
        assert len(mask) == len(sample_points)
        assert mask.dtype == bool
        assert np.sum(mask) == len(clipped)
        
        # Verify mask is correct - note: points on boundary may be excluded
        # due to computational geometry conventions (strict interior test)
        for i, m in enumerate(mask):
            x, y = sample_points[i, 0], sample_points[i, 1]
            strictly_inside = (10 < x < 50) and (10 < y < 50)
            # If strictly inside, must be in mask
            if strictly_inside:
                assert m, f"Point ({x}, {y}) should be inside"
            # If clearly outside, must not be in mask
            if x < 10 or x > 50 or y < 10 or y > 50:
                assert not m, f"Point ({x}, {y}) should be outside"
    
    def test_clip_empty_points(self, simple_clipper):
        """Test clipping an empty point array."""
        empty_points = np.zeros((0, 3))
        clipped = simple_clipper.clip(empty_points)
        
        assert clipped.shape == (0, 3)
    
    def test_clip_all_outside(self, simple_clipper):
        """Test clipping when all points are outside the boundary."""
        # All points at (0, 0) - outside the clipper bounds
        outside_points = np.zeros((100, 3))
        clipped = simple_clipper.clip(outside_points)
        
        assert len(clipped) == 0
    
    def test_clip_all_inside(self, simple_clipper):
        """Test clipping when all points are inside the boundary."""
        # All points at (30, 30) - inside the clipper bounds
        inside_points = np.full((100, 3), 30.0)
        clipped = simple_clipper.clip(inside_points)
        
        assert len(clipped) == 100
    
    def test_clip_with_irregular_polygon(self):
        """Test clipping with a non-rectangular polygon."""
        # Triangle
        triangle = [(0, 0), (100, 0), (50, 100), (0, 0)]
        clipper = AreaClipper.from_polygon(triangle)
        
        # Create points
        points = np.array([
            [50, 10, 0],   # Inside
            [10, 50, 0],   # Outside (left of triangle)
            [90, 50, 0],   # Outside (right of triangle)
            [50, 50, 0],   # Inside
        ])
        
        clipped = clipper.clip(points)
        
        # Should keep 2 points that are inside the triangle
        assert len(clipped) == 2
    
    def test_clip_preserves_z_coordinate(self, simple_clipper):
        """Test that Z coordinates are preserved during clipping."""
        points = np.array([
            [30, 30, 100],
            [30, 30, 200],
            [30, 30, 300],
        ])
        
        clipped = simple_clipper.clip(points)
        
        assert len(clipped) == 3
        np.testing.assert_array_equal(clipped[:, 2], [100, 200, 300])
    
    def test_clip_invalid_points_shape(self, simple_clipper):
        """Test error handling for invalid point array shapes."""
        with pytest.raises(ValueError, match="at least 2 columns"):
            simple_clipper.clip(np.array([[1]]))
        
        with pytest.raises(ValueError):
            simple_clipper.clip(np.array([1, 2, 3]))  # 1D array


class TestClipWithAttributes:
    """Tests for clipping points with associated attributes."""
    
    def test_clip_with_attributes(self):
        """Test that attributes are properly filtered along with points."""
        clipper = AreaClipper.from_bounds(10, 10, 50, 50)
        
        points = np.array([
            [5, 5, 0],    # Outside
            [30, 30, 0],  # Inside
            [60, 60, 0],  # Outside
            [20, 20, 0],  # Inside
        ])
        
        attributes = {
            'intensity': np.array([100, 200, 300, 400]),
            'classification': np.array([1, 2, 1, 2]),
        }
        
        clipped_pts, clipped_attrs = clipper.clip_with_attributes(points, attributes)
        
        assert len(clipped_pts) == 2
        np.testing.assert_array_equal(clipped_attrs['intensity'], [200, 400])
        np.testing.assert_array_equal(clipped_attrs['classification'], [2, 2])


class TestStatistics:
    """Tests for clipping statistics."""
    
    def test_get_statistics(self):
        """Test getting clipping statistics."""
        clipper = AreaClipper.from_bounds(0, 0, 50, 50)
        
        # 100 points spread across 0-100, y fixed at 25
        points = np.column_stack([
            np.linspace(0, 100, 101),
            np.full(101, 25),
            np.zeros(101)
        ])
        
        stats = clipper.get_statistics(points)
        
        assert stats['total_points'] == 101
        # Points strictly inside (1-49) should be included
        # Boundary points (0, 50) may or may not be included
        assert 49 <= stats['points_inside'] <= 51
        assert 50 <= stats['points_outside'] <= 52
        assert 48 <= stats['percentage_inside'] <= 51
        assert stats['clipping_bounds'] == (0, 0, 50, 50)


class TestGeoJSONExport:
    """Tests for exporting clipping boundaries to GeoJSON."""
    
    def test_to_geojson(self):
        """Test exporting clipper to GeoJSON dictionary."""
        coords = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        clipper = AreaClipper.from_polygon(coords)
        
        geojson = clipper.to_geojson()
        
        assert geojson['type'] == 'Feature'
        assert geojson['geometry']['type'] == 'Polygon'
        assert 'area' in geojson['properties']
    
    def test_save_geojson(self):
        """Test saving clipper to GeoJSON file."""
        coords = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        clipper = AreaClipper.from_polygon(coords)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_boundary.geojson"
            clipper.save_geojson(output_path)
            
            assert output_path.exists()
            
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            assert data['type'] == 'FeatureCollection'
            assert len(data['features']) == 1


class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_check_shapely_available(self):
        """Test shapely availability check."""
        result = check_shapely_available()
        assert result == SHAPELY_AVAILABLE
    
    def test_check_fiona_available(self):
        """Test fiona availability check."""
        result = check_fiona_available()
        # This may or may not be available, just check it returns a bool
        assert isinstance(result, bool)


class TestLargeScale:
    """Tests for large-scale clipping performance."""
    
    @pytest.mark.slow
    def test_clip_large_point_cloud(self):
        """Test clipping a large point cloud (performance test)."""
        clipper = AreaClipper.from_bounds(100, 100, 900, 900)
        
        # Generate 1 million points randomly distributed in 0-1000 range
        np.random.seed(42)
        points = np.random.uniform(0, 1000, size=(1_000_000, 3))
        
        clipped = clipper.clip(points)
        
        # Approximately 64% should be inside (800/1000)^2 = 0.64
        expected_fraction = 0.64
        actual_fraction = len(clipped) / len(points)
        
        assert abs(actual_fraction - expected_fraction) < 0.05  # 5% tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
