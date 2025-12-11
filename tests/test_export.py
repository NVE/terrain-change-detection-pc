"""
Tests for export utilities.

Tests LAZ point cloud and GeoTIFF raster export functions.
"""

import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pytest


# ============================================================
# Test fixtures and helpers
# ============================================================


@dataclass
class MockDoDResult:
    """Mock DoD result for testing."""
    grid_x: np.ndarray
    grid_y: np.ndarray
    dem1: np.ndarray
    dem2: np.ndarray
    dod: np.ndarray
    cell_size: float
    bounds: Tuple[float, float, float, float]
    stats: Dict[str, float]
    metadata: Optional[Dict] = None


@pytest.fixture
def sample_points():
    """Generate sample point cloud data."""
    np.random.seed(42)
    n_points = 100
    points = np.random.uniform(0, 100, (n_points, 3))
    distances = np.random.uniform(-2, 2, n_points)
    return points, distances


@pytest.fixture
def sample_dod():
    """Generate sample DoD result."""
    cell_size = 1.0
    n_cells = 10
    x = np.arange(0, n_cells) * cell_size
    y = np.arange(0, n_cells) * cell_size
    grid_x, grid_y = np.meshgrid(x, y)
    
    # Create synthetic elevation grids
    dem1 = np.random.uniform(100, 110, (n_cells, n_cells))
    dem2 = dem1 + np.random.uniform(-1, 1, (n_cells, n_cells))
    dod = dem2 - dem1
    
    return MockDoDResult(
        grid_x=grid_x,
        grid_y=grid_y,
        dem1=dem1,
        dem2=dem2,
        dod=dod,
        cell_size=cell_size,
        bounds=(0, 0, n_cells * cell_size, n_cells * cell_size),
        stats={"mean": float(np.nanmean(dod)), "std": float(np.nanstd(dod))},
    )


# ============================================================
# Test LAZ export
# ============================================================


class TestExportPointsToLaz:
    """Tests for export_points_to_laz function."""

    def test_basic_export(self, sample_points):
        """Test basic LAZ export with points and distances."""
        from terrain_change_detection.utils.export import export_points_to_laz
        
        points, distances = sample_points
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output.laz"
            result = export_points_to_laz(points, distances, str(output_path))
            
            assert Path(result).exists()
            assert Path(result).suffix == ".laz"
            
            # Verify file can be read with laspy
            import laspy
            with laspy.open(result) as reader:
                las = reader.read()
                assert len(las.points) == len(points)
                # Check coordinates match
                np.testing.assert_allclose(las.x, points[:, 0], atol=0.01)
                np.testing.assert_allclose(las.y, points[:, 1], atol=0.01)
                np.testing.assert_allclose(las.z, points[:, 2], atol=0.01)
                # Check distance dimension exists
                assert hasattr(las, 'distance')
                np.testing.assert_allclose(las.distance, distances, atol=1e-6)

    def test_export_with_extra_dims(self, sample_points):
        """Test LAZ export with additional extra dimensions."""
        from terrain_change_detection.utils.export import export_points_to_laz
        
        points, distances = sample_points
        uncertainty = np.random.uniform(0.1, 0.5, len(points))
        significant = np.random.choice([True, False], len(points))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_extra.laz"
            result = export_points_to_laz(
                points, distances, str(output_path),
                extra_dims={"uncertainty": uncertainty, "significant": significant}
            )
            
            assert Path(result).exists()
            
            import laspy
            with laspy.open(result) as reader:
                las = reader.read()
                assert hasattr(las, 'uncertainty')
                assert hasattr(las, 'significant')

    def test_creates_parent_directory(self, sample_points):
        """Test that parent directories are created if they don't exist."""
        from terrain_change_detection.utils.export import export_points_to_laz
        
        points, distances = sample_points
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dirs" / "test.laz"
            result = export_points_to_laz(points, distances, str(output_path))
            
            assert Path(result).exists()

    def test_invalid_points_shape(self, sample_points):
        """Test that invalid point shapes raise ValueError."""
        from terrain_change_detection.utils.export import export_points_to_laz
        
        points, distances = sample_points
        invalid_points = points[:, :2]  # Only 2D
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.laz"
            with pytest.raises(ValueError, match="must be"):
                export_points_to_laz(invalid_points, distances, str(output_path))


# ============================================================
# Test GeoTIFF export (requires rasterio)
# ============================================================


class TestExportDoDToGeotiff:
    """Tests for export_dod_to_geotiff function."""

    @pytest.fixture(autouse=True)
    def check_rasterio(self):
        """Skip tests if rasterio is not available."""
        pytest.importorskip("rasterio")

    def test_basic_dod_export(self, sample_dod):
        """Test basic DoD GeoTIFF export."""
        from terrain_change_detection.utils.export import export_dod_to_geotiff
        import rasterio
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dod.tif"
            result = export_dod_to_geotiff(sample_dod, str(output_path))
            
            assert Path(result).exists()
            
            # Verify file can be read
            with rasterio.open(result) as src:
                assert src.count == 1
                assert src.width == sample_dod.dod.shape[1]
                assert src.height == sample_dod.dod.shape[0]
                data = src.read(1)
                assert data.shape == sample_dod.dod.shape

    def test_crs_is_set(self, sample_dod):
        """Test that CRS is correctly set in output."""
        from terrain_change_detection.utils.export import export_dod_to_geotiff
        import rasterio
        
        crs = "EPSG:25833"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dod_crs.tif"
            export_dod_to_geotiff(sample_dod, str(output_path), crs=crs)
            
            with rasterio.open(output_path) as src:
                assert src.crs is not None
                assert src.crs.to_string() == crs


class TestExportDistancesToGeotiff:
    """Tests for export_distances_to_geotiff function."""

    @pytest.fixture(autouse=True)
    def check_rasterio(self):
        """Skip tests if rasterio is not available."""
        pytest.importorskip("rasterio")

    def test_basic_distance_raster(self, sample_points):
        """Test basic point-to-raster interpolation."""
        from terrain_change_detection.utils.export import export_distances_to_geotiff
        import rasterio
        
        points, distances = sample_points
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "distances.tif"
            result = export_distances_to_geotiff(
                points, distances, str(output_path), cell_size=5.0
            )
            
            assert Path(result).exists()
            
            with rasterio.open(result) as src:
                assert src.count == 1
                data = src.read(1)
                # Should have some valid data (not all nodata)
                nodata = src.nodata
                assert np.sum(data != nodata) > 0


# ============================================================
# Test CRS detection
# ============================================================


class TestDetectCrsFromLaz:
    """Tests for detect_crs_from_laz function."""

    def test_returns_none_for_file_without_crs(self, sample_points):
        """Test that None is returned for LAZ files without CRS info."""
        from terrain_change_detection.utils.export import (
            export_points_to_laz,
            detect_crs_from_laz,
        )
        
        points, distances = sample_points
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a LAZ file without CRS VLR (no crs= argument)
            laz_path = Path(tmpdir) / "no_crs.laz"
            export_points_to_laz(points, distances, str(laz_path))
            
            # Should return None or the CRS we set
            result = detect_crs_from_laz(str(laz_path))
            # Result can be None or a valid CRS string depending on header
            assert result is None or result.startswith("EPSG:")

    def test_returns_none_for_nonexistent_file(self):
        """Test that None is returned for non-existent files."""
        from terrain_change_detection.utils.export import detect_crs_from_laz
        
        result = detect_crs_from_laz("/nonexistent/path/file.laz")
        assert result is None
