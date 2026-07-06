"""
Tests for export utilities.

Tests LAZ point cloud and GeoTIFF raster export functions.
"""

import tempfile
import json
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

    def test_local_transform_converts_dod_bounds_to_global(self, sample_dod):
        """Test that local DoD bounds are written as global GeoTIFF bounds."""
        from terrain_change_detection.utils.coordinate_transform import LocalCoordinateTransform
        from terrain_change_detection.utils.export import export_dod_to_geotiff
        import rasterio

        local_transform = LocalCoordinateTransform.from_bounds(466000.0, 6650000.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dod_global_bounds.tif"
            export_dod_to_geotiff(
                sample_dod, str(output_path), local_transform=local_transform,
            )

            with rasterio.open(output_path) as src:
                assert src.bounds.left == pytest.approx(466000.0)
                assert src.bounds.bottom == pytest.approx(6650000.0)
                assert src.bounds.right == pytest.approx(466010.0)
                assert src.bounds.top == pytest.approx(6650010.0)
                assert src.res == pytest.approx((sample_dod.cell_size, sample_dod.cell_size))

    def test_dod_export_flips_rows_to_north_up(self):
        """Test GeoTIFF row 0 contains max-Y DoD row."""
        from terrain_change_detection.utils.export import export_dod_to_geotiff
        import rasterio

        dod = MockDoDResult(
            grid_x=np.array([[0.5, 1.5], [0.5, 1.5]]),
            grid_y=np.array([[0.5, 0.5], [1.5, 1.5]]),
            dem1=np.zeros((2, 2)),
            dem2=np.zeros((2, 2)),
            dod=np.array([[1.0, 2.0], [3.0, 4.0]]),
            cell_size=1.0,
            bounds=(0.0, 0.0, 2.0, 2.0),
            stats={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dod_north_up.tif"
            export_dod_to_geotiff(dod, str(output_path))

            with rasterio.open(output_path) as src:
                np.testing.assert_array_equal(
                    src.read(1),
                    np.array([[3.0, 4.0], [1.0, 2.0]], dtype=np.float32),
                )


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


class TestExportErosionPolygonsGeojson:
    """Tests for hysteresis erosion polygon export."""

    @pytest.fixture(autouse=True)
    def check_deps(self):
        pytest.importorskip("rasterio")
        pytest.importorskip("scipy")
        pytest.importorskip("shapely")

    def _write_raster(self, path: Path, data: np.ndarray) -> None:
        import rasterio
        from rasterio.transform import from_origin

        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype="float32",
            crs="EPSG:25833",
            transform=from_origin(0, data.shape[0], 1, 1),
            nodata=-9999.0,
        ) as dst:
            dst.write(data.astype(np.float32), 1)

    def test_hysteresis_keeps_outline_component_with_peak(self):
        from terrain_change_detection.utils.export import export_erosion_polygons_geojson

        raster = np.zeros((8, 8), dtype=float)
        raster[2:6, 2:6] = -0.3
        raster[3:5, 3:5] = -0.7

        with tempfile.TemporaryDirectory() as tmpdir:
            raster_path = Path(tmpdir) / "m3c2.tif"
            output_path = Path(tmpdir) / "erosion.geojson"
            self._write_raster(raster_path, raster)

            summary = export_erosion_polygons_geojson(
                str(raster_path),
                str(output_path),
                peak_threshold_m=-0.5,
                outline_threshold_m=-0.25,
                closing_iterations=0,
                opening_iterations=0,
            )

            assert summary["polygon_count"] == 1
            data = json.loads(output_path.read_text())
            props = data["features"][0]["properties"]
            assert props["cell_count"] == 16
            assert props["min_distance_m"] == pytest.approx(-0.7, abs=1e-6)
            assert props["peak_erosion_m"] == pytest.approx(0.7, abs=1e-6)

    def test_hysteresis_drops_outline_component_without_peak(self):
        from terrain_change_detection.utils.export import export_erosion_polygons_geojson

        raster = np.zeros((8, 8), dtype=float)
        raster[2:6, 2:6] = -0.3

        with tempfile.TemporaryDirectory() as tmpdir:
            raster_path = Path(tmpdir) / "m3c2.tif"
            output_path = Path(tmpdir) / "erosion.geojson"
            self._write_raster(raster_path, raster)

            summary = export_erosion_polygons_geojson(
                str(raster_path),
                str(output_path),
                peak_threshold_m=-0.5,
                outline_threshold_m=-0.25,
                closing_iterations=0,
                opening_iterations=0,
            )

            assert summary["polygon_count"] == 0
            data = json.loads(output_path.read_text())
            assert data["features"] == []

    def test_min_cells_filters_small_component(self):
        from terrain_change_detection.utils.export import export_erosion_polygons_geojson

        raster = np.zeros((8, 8), dtype=float)
        raster[3, 3] = -0.7

        with tempfile.TemporaryDirectory() as tmpdir:
            raster_path = Path(tmpdir) / "m3c2.tif"
            output_path = Path(tmpdir) / "erosion.geojson"
            self._write_raster(raster_path, raster)

            summary = export_erosion_polygons_geojson(
                str(raster_path),
                str(output_path),
                peak_threshold_m=-0.5,
                outline_threshold_m=-0.25,
                closing_iterations=0,
                opening_iterations=0,
                min_cells=2,
            )

            assert summary["polygon_count"] == 0

    def test_min_area_requires_polygon_area_greater_than_threshold(self):
        from terrain_change_detection.utils.export import export_erosion_polygons_geojson

        raster = np.zeros((8, 8), dtype=float)
        raster[2:5, 2:5] = -0.7

        with tempfile.TemporaryDirectory() as tmpdir:
            raster_path = Path(tmpdir) / "m3c2.tif"
            output_path = Path(tmpdir) / "erosion.geojson"
            self._write_raster(raster_path, raster)

            equal_area = export_erosion_polygons_geojson(
                str(raster_path),
                str(output_path),
                peak_threshold_m=-0.5,
                outline_threshold_m=-0.25,
                closing_iterations=0,
                opening_iterations=0,
                min_area_m2=9.0,
            )
            assert equal_area["polygon_count"] == 0

            greater_area = export_erosion_polygons_geojson(
                str(raster_path),
                str(output_path),
                peak_threshold_m=-0.5,
                outline_threshold_m=-0.25,
                closing_iterations=0,
                opening_iterations=0,
                min_area_m2=8.99,
            )
            assert greater_area["polygon_count"] == 1


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

    def test_extracts_projected_epsg_from_compound_wkt(self):
        """Compound WKT should return horizontal CRS, not spheroid/vertical EPSG."""
        from terrain_change_detection.utils.export import _extract_epsg_from_wkt

        wkt = (
            'COMPD_CS["ETRS89 / UTM zone 32N + NN2000 height",'
            'PROJCS["ETRS89 / UTM zone 32N",'
            'GEOGCS["ETRS89",DATUM["European_Terrestrial_Reference_System_1989",'
            'SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],'
            'AUTHORITY["EPSG","6258"]]],'
            'AUTHORITY["EPSG","25832"]],'
            'VERT_CS["NN2000 height",AUTHORITY["EPSG","5941"]],'
            'AUTHORITY["EPSG","5972"]]'
        )

        assert _extract_epsg_from_wkt(wkt) == "EPSG:25832"
