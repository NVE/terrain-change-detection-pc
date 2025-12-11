"""
Export utilities for terrain change detection results.

Provides functions to export detection results to:
- Point cloud formats (LAZ/LAS) with distances as extra dimensions
- Raster formats (GeoTIFF) for grid-based outputs

These outputs are compatible with QGIS and similar GIS software.
"""

from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING
import numpy as np

from .logging import setup_logger

if TYPE_CHECKING:
    from ..detection.dod import DoDResult

logger = setup_logger(__name__)


def detect_crs_from_laz(laz_path: str) -> Optional[str]:
    """
    Attempt to detect CRS from a LAZ/LAS file.

    Reads the file header and looks for WKT or GeoTIFF VLRs to extract
    the coordinate reference system.

    Args:
        laz_path: Path to LAZ/LAS file

    Returns:
        EPSG string (e.g., "EPSG:25833") if found, None otherwise
    """
    try:
        import laspy

        with laspy.open(laz_path) as reader:
            header = reader.header

            # Check for WKT VLR (common in newer LAS files)
            for vlr in header.vlrs:
                # WKT VLR has record_id 2112 or user_id "LASF_Projection"
                if vlr.user_id == "LASF_Projection" and vlr.record_id == 2112:
                    wkt = vlr.record_data.decode("utf-8", errors="ignore").strip("\x00")
                    # Try to extract EPSG from WKT
                    epsg = _extract_epsg_from_wkt(wkt)
                    if epsg:
                        return epsg

            # Check for GeoTIFF VLR (older format)
            for vlr in header.vlrs:
                if vlr.user_id == "LASF_Projection" and vlr.record_id == 34735:
                    # GeoKeyDirectoryTag - more complex to parse
                    # For now, just log that we found it
                    logger.debug(f"Found GeoTIFF VLR in {laz_path}, but parsing not implemented")

    except Exception as e:
        logger.debug(f"Could not detect CRS from {laz_path}: {e}")

    return None


def _extract_epsg_from_wkt(wkt: str) -> Optional[str]:
    """Extract EPSG code from WKT string."""
    import re

    # Look for AUTHORITY["EPSG","25833"] or similar patterns
    match = re.search(r'AUTHORITY\s*\[\s*"EPSG"\s*,\s*"(\d+)"\s*\]', wkt, re.IGNORECASE)
    if match:
        return f"EPSG:{match.group(1)}"

    # Look for ID["EPSG",25833] (WKT2 format)
    match = re.search(r'ID\s*\[\s*"EPSG"\s*,\s*(\d+)\s*\]', wkt, re.IGNORECASE)
    if match:
        return f"EPSG:{match.group(1)}"

    return None


def export_points_to_laz(
    points: np.ndarray,
    distances: np.ndarray,
    output_path: str,
    *,
    crs: Optional[str] = None,
    extra_dims: Optional[Dict[str, np.ndarray]] = None,
    source_laz_path: Optional[str] = None,
) -> str:
    """
    Export points with distance values to a LAZ/LAS file.

    The distance values are stored as an extra dimension named "distance".
    Additional extra dimensions can be provided (e.g., uncertainty, significance).

    Args:
        points: (N, 3) array of point coordinates
        distances: (N,) array of distance values
        output_path: Path for output file (extension determines format)
        crs: CRS string (e.g., "EPSG:25833"). If None, attempts auto-detection.
        extra_dims: Optional dict of additional arrays to store as extra dimensions
        source_laz_path: Optional path to source LAZ for CRS auto-detection

    Returns:
        Path to created file
    """
    import laspy

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    points = np.asarray(points, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}")
    if distances.ndim != 1 or len(distances) != len(points):
        raise ValueError(f"distances must be (N,), got {distances.shape}")

    # Try to auto-detect CRS if not provided
    detected_crs = None
    if crs is None and source_laz_path:
        detected_crs = detect_crs_from_laz(source_laz_path)
        if detected_crs:
            logger.info(f"Auto-detected CRS from source: {detected_crs}")
            crs = detected_crs

    # Create LAS file with extra dimensions
    # Use LAS 1.4 point format 6 which supports extra bytes
    header = laspy.LasHeader(point_format=6, version="1.4")

    # Add distance as extra dimension
    header.add_extra_dim(laspy.ExtraBytesParams(name="distance", type=np.float64))

    # Add any additional extra dimensions
    if extra_dims:
        for dim_name, dim_data in extra_dims.items():
            if dim_name != "distance":  # Already added
                dtype = dim_data.dtype
                if dtype == np.bool_:
                    dtype = np.uint8  # Store bools as uint8
                header.add_extra_dim(laspy.ExtraBytesParams(name=dim_name, type=dtype))

    # Add CRS as WKT VLR if provided
    if crs:
        try:
            wkt = _epsg_to_wkt(crs)
            if wkt:
                vlr = laspy.VLR(
                    user_id="LASF_Projection",
                    record_id=2112,
                    description="WKT Coordinate System",
                    record_data=wkt.encode("utf-8"),
                )
                header.vlrs.append(vlr)
                logger.debug(f"Added CRS VLR: {crs}")
        except Exception as e:
            logger.warning(f"Could not add CRS VLR: {e}")

    # Create the LAS data
    las = laspy.LasData(header)

    # Set coordinates
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    # Set distance dimension
    las.distance = distances

    # Set additional dimensions
    if extra_dims:
        for dim_name, dim_data in extra_dims.items():
            if dim_name != "distance":
                data = np.asarray(dim_data)
                if data.dtype == np.bool_:
                    data = data.astype(np.uint8)
                setattr(las, dim_name, data)

    # Write file
    las.write(str(output_path))
    logger.info(f"Exported {len(points):,} points to {output_path}")

    return str(output_path)


def _epsg_to_wkt(epsg_str: str) -> Optional[str]:
    """Convert EPSG string to WKT using pyproj if available."""
    try:
        from pyproj import CRS

        crs = CRS.from_string(epsg_str)
        return crs.to_wkt()
    except ImportError:
        # pyproj not available, use simple WKT template for common CRS
        if epsg_str == "EPSG:25833":
            return 'PROJCS["ETRS89 / UTM zone 33N",GEOGCS["ETRS89",DATUM["European_Terrestrial_Reference_System_1989",SPHEROID["GRS 1980",6378137,298.257222101]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",15],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1],AUTHORITY["EPSG","25833"]]'
        logger.debug(f"pyproj not available, no WKT for {epsg_str}")
        return None
    except Exception as e:
        logger.debug(f"Could not convert {epsg_str} to WKT: {e}")
        return None


def export_dod_to_geotiff(
    dod_result: "DoDResult",
    output_path: str,
    *,
    crs: str = "EPSG:25833",
    nodata: float = -9999.0,
) -> str:
    """
    Export DEM of Difference result to a GeoTIFF file.

    Args:
        dod_result: DoDResult from ChangeDetector.compute_dod
        output_path: Path for output GeoTIFF file
        crs: Coordinate reference system (default: EPSG:25833)
        nodata: NoData value for missing cells

    Returns:
        Path to created file
    """
    import rasterio
    from rasterio.transform import from_bounds

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get DoD array and dimensions
    dod = np.asarray(dod_result.dod, dtype=np.float32)
    min_x, min_y, max_x, max_y = dod_result.bounds
    cell_size = dod_result.cell_size

    # Handle NaN values
    dod = np.where(np.isnan(dod), nodata, dod)

    # Compute raster dimensions
    height, width = dod.shape

    # Create affine transform (top-left origin for raster)
    transform = from_bounds(min_x, min_y, max_x, max_y, width, height)

    # Write GeoTIFF
    with rasterio.open(
        str(output_path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=dod.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="lzw",
    ) as dst:
        dst.write(dod, 1)

    logger.info(f"Exported DoD raster ({width}x{height}) to {output_path}")
    return str(output_path)


def export_distances_to_geotiff(
    points: np.ndarray,
    distances: np.ndarray,
    output_path: str,
    *,
    cell_size: float = 1.0,
    crs: str = "EPSG:25833",
    nodata: float = -9999.0,
    bounds: Optional[tuple] = None,
) -> str:
    """
    Export point distances to a GeoTIFF raster using nearest-neighbor interpolation.

    Uses KDTree for efficient nearest-neighbor lookup to assign each raster cell
    the distance value of the closest point.

    Args:
        points: (N, 3) array of point coordinates
        distances: (N,) array of distance values
        output_path: Path for output GeoTIFF file
        cell_size: Raster cell size in meters
        crs: Coordinate reference system
        nodata: NoData value for cells with no nearby points
        bounds: Optional (min_x, min_y, max_x, max_y); computed from points if None

    Returns:
        Path to created file
    """
    import rasterio
    from rasterio.transform import from_bounds
    from scipy.spatial import cKDTree

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    points = np.asarray(points, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"points must be (N, 2+), got {points.shape}")

    # Use XY coordinates only for gridding
    xy = points[:, :2]

    # Compute bounds if not provided
    if bounds is None:
        min_x, min_y = xy.min(axis=0)
        max_x, max_y = xy.max(axis=0)
    else:
        min_x, min_y, max_x, max_y = bounds

    # Pad bounds slightly to ensure all points are covered
    min_x -= cell_size / 2
    min_y -= cell_size / 2
    max_x += cell_size / 2
    max_y += cell_size / 2

    # Compute grid dimensions
    width = int(np.ceil((max_x - min_x) / cell_size))
    height = int(np.ceil((max_y - min_y) / cell_size))

    # Create grid of cell centers
    x_centers = np.linspace(min_x + cell_size / 2, max_x - cell_size / 2, width)
    y_centers = np.linspace(max_y - cell_size / 2, min_y + cell_size / 2, height)  # Top to bottom
    xx, yy = np.meshgrid(x_centers, y_centers)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    # Build KDTree for efficient nearest neighbor lookup
    tree = cKDTree(xy)

    # Find nearest point for each grid cell
    # Use max search distance of 2 * cell_size to avoid assigning points too far away
    max_dist = cell_size * 2
    dists, indices = tree.query(grid_points, k=1, distance_upper_bound=max_dist)

    # Create raster array
    raster = np.full((height, width), nodata, dtype=np.float32)

    # Assign distance values where points were found
    valid = indices < len(distances)  # Valid indices (within max_dist)
    raster_flat = raster.ravel()
    raster_flat[valid] = distances[indices[valid]]
    raster = raster_flat.reshape((height, width))

    # Create affine transform
    transform = from_bounds(min_x, min_y, max_x, max_y, width, height)

    # Write GeoTIFF
    with rasterio.open(
        str(output_path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=raster.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="lzw",
    ) as dst:
        dst.write(raster, 1)

    n_valid = np.sum(valid)
    logger.info(
        f"Exported distance raster ({width}x{height}, {n_valid:,} cells with data) to {output_path}"
    )
    return str(output_path)
