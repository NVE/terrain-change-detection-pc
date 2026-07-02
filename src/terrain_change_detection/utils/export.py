"""
Export utilities for terrain change detection results.

Provides functions to export detection results to:
- Point cloud formats (LAZ/LAS) with distances as extra dimensions
- Raster formats (GeoTIFF) for grid-based outputs

These outputs are compatible with QGIS and similar GIS software.
"""

from pathlib import Path
from typing import Optional, Dict, TYPE_CHECKING, Any
import json
import numpy as np

from .logging import setup_logger

if TYPE_CHECKING:
    from ..detection.dod import DoDResult
    from .coordinate_transform import LocalCoordinateTransform

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
    distances: Optional[np.ndarray],
    output_path: str,
    *,
    crs: Optional[str] = None,
    extra_dims: Optional[Dict[str, np.ndarray]] = None,
    source_laz_path: Optional[str] = None,
    local_transform: Optional["LocalCoordinateTransform"] = None,
    classification: Optional[int] = None,
) -> str:
    """
    Export points to a LAZ/LAS file, optionally with distance values.

    When *distances* is provided they are stored as an extra dimension named
    ``"distance"``.  Additional extra dimensions can be provided (e.g.,
    uncertainty, significance).

    Args:
        points: (N, 3) array of point coordinates (in local or global system)
        distances: (N,) array of distance values, or None to export only XYZ
        output_path: Path for output file (extension determines format)
        crs: CRS string (e.g., "EPSG:25833"). If None, attempts auto-detection.
        extra_dims: Optional dict of additional arrays to store as extra dimensions
        source_laz_path: Optional path to source LAZ for CRS auto-detection
        local_transform: If provided, reverts points from local to global coordinates
        classification: Optional LAS classification code to assign to every point

    Returns:
        Path to created file
    """
    import laspy

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}")

    has_distances = distances is not None
    if has_distances:
        distances = np.asarray(distances, dtype=np.float64)
        if distances.ndim != 1 or len(distances) != len(points):
            raise ValueError(f"distances must be (N,), got {distances.shape}")

    # Revert to global coordinates if transform was used
    if local_transform is not None:
        points = local_transform.to_global(points)
        logger.info(f"Reverting {len(points):,} points from local to global coordinates for export")

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

    # Add distance as extra dimension (only if distances provided)
    if has_distances:
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

    if classification is not None:
        las.classification = np.full(len(points), classification, dtype=np.uint8)

    # Set distance dimension
    if has_distances:
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
    local_transform: Optional["LocalCoordinateTransform"] = None,
) -> str:
    """
    Export point distances to a GeoTIFF raster using nearest-neighbor interpolation.

    Uses KDTree for efficient nearest-neighbor lookup to assign each raster cell
    the distance value of the closest point.

    Args:
        points: (N, 3) array of point coordinates (in local or global system)
        distances: (N,) array of distance values
        output_path: Path for output GeoTIFF file
        cell_size: Raster cell size in meters
        crs: Coordinate reference system
        nodata: NoData value for cells with no nearby points
        bounds: Optional (min_x, min_y, max_x, max_y); computed from points if None
        local_transform: If provided, reverts points from local to global coordinates

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

    # Revert to global coordinates if transform was used
    if local_transform is not None:
        points = local_transform.to_global(points)
        logger.info(f"Reverting {len(points):,} points from local to global coordinates for raster export")

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


def export_erosion_polygons_geojson(
    raster_path: str,
    output_path: str,
    *,
    peak_threshold_m: float,
    outline_threshold_m: float,
    use_significance: bool = False,
    significant_points: Optional[np.ndarray] = None,
    significant_values: Optional[np.ndarray] = None,
    closing_iterations: int = 1,
    opening_iterations: int = 1,
    structure_radius_cells: int = 1,
    min_area_m2: float = 0.0,
    min_cells: int = 1,
    simplify_tolerance_m: float = 0.0,
    local_transform: Optional["LocalCoordinateTransform"] = None,
) -> dict[str, Any]:
    """Export erosion polygons from a signed M3C2 distance raster.

    Uses hysteresis thresholding: components are grown from the broader
    ``outline_threshold_m`` mask, then kept only if they contain at least one
    cell below ``peak_threshold_m``.
    """
    import rasterio
    from rasterio.features import shapes
    from scipy import ndimage
    from shapely.geometry import shape, mapping

    if peak_threshold_m >= 0 or outline_threshold_m >= 0:
        raise ValueError("erosion thresholds must be negative")
    if peak_threshold_m > outline_threshold_m:
        raise ValueError("peak_threshold_m must be <= outline_threshold_m")

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(raster_path) as src:
        raster = src.read(1).astype(np.float64)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs.to_string() if src.crs is not None else None

    valid = np.isfinite(raster)
    if nodata is not None and np.isfinite(nodata):
        valid &= raster != nodata

    candidate_mask = valid & (raster <= outline_threshold_m)
    seed_mask = valid & (raster <= peak_threshold_m)

    if use_significance and significant_points is not None and significant_values is not None:
        significant_grid = _rasterize_nearest_to_existing_grid(
            significant_points,
            significant_values,
            raster.shape,
            transform,
            local_transform=local_transform,
        )
        significant_mask = significant_grid > 0.5
        candidate_mask &= significant_mask
        seed_mask &= significant_mask

    structure = _binary_structure(structure_radius_cells)
    if closing_iterations > 0:
        candidate_mask = ndimage.binary_closing(candidate_mask, structure=structure, iterations=closing_iterations)
    if opening_iterations > 0:
        candidate_mask = ndimage.binary_opening(candidate_mask, structure=structure, iterations=opening_iterations)
    candidate_mask &= valid

    labels, label_count = ndimage.label(candidate_mask, structure=structure)
    if label_count == 0:
        _write_geojson(output_path_obj, [], crs)
        return {"path": str(output_path_obj), "polygon_count": 0, "total_area_m2": 0.0, "total_volume_loss_m3": 0.0}

    seed_labels = np.unique(labels[seed_mask & (labels > 0)])
    keep_labels = set(int(label) for label in seed_labels)
    if min_cells > 1:
        counts = np.bincount(labels.ravel())
        keep_labels = {label for label in keep_labels if counts[label] >= min_cells}

    kept_mask = np.isin(labels, list(keep_labels)) if keep_labels else np.zeros_like(candidate_mask, dtype=bool)
    pixel_area = abs(transform.a * transform.e)
    features = []

    for geom_mapping, value in shapes(labels.astype(np.int32), mask=kept_mask, transform=transform):
        label = int(value)
        if label <= 0 or label not in keep_labels:
            continue
        geom = shape(geom_mapping)
        if simplify_tolerance_m > 0:
            geom = geom.simplify(simplify_tolerance_m, preserve_topology=True)
        if geom.is_empty:
            continue
        if min_area_m2 > 0 and geom.area <= min_area_m2:
            continue

        component_values = raster[labels == label]
        component_values = component_values[np.isfinite(component_values)]
        if component_values.size == 0:
            continue
        erosion_values = component_values[component_values < 0]
        if erosion_values.size == 0:
            continue

        volume_loss = float(np.sum(np.abs(erosion_values)) * pixel_area)
        min_distance = float(np.min(component_values))
        props = {
            "label": label,
            "cell_count": int(np.sum(labels == label)),
            "area_m2": float(geom.area),
            "min_distance_m": min_distance,
            "peak_erosion_m": float(abs(min_distance)),
            "mean_erosion_m": float(np.mean(np.abs(erosion_values))),
            "max_erosion_m": float(np.max(np.abs(erosion_values))),
            "p05_erosion_m": float(np.percentile(np.abs(erosion_values), 5)),
            "p50_erosion_m": float(np.percentile(np.abs(erosion_values), 50)),
            "p95_erosion_m": float(np.percentile(np.abs(erosion_values), 95)),
            "volume_loss_m3": volume_loss,
            "peak_threshold_m": float(peak_threshold_m),
            "outline_threshold_m": float(outline_threshold_m),
            "closing_iterations": int(closing_iterations),
            "opening_iterations": int(opening_iterations),
            "structure_radius_cells": int(structure_radius_cells),
        }
        features.append({"type": "Feature", "geometry": mapping(geom), "properties": props})

    _write_geojson(output_path_obj, features, crs)
    total_area = float(sum(feature["properties"]["area_m2"] for feature in features))
    total_volume = float(sum(feature["properties"]["volume_loss_m3"] for feature in features))
    logger.info(
        "Exported %d erosion polygons (area %.2f m2, volume %.2f m3) to %s",
        len(features), total_area, total_volume, output_path_obj,
    )
    return {
        "path": str(output_path_obj),
        "polygon_count": len(features),
        "total_area_m2": total_area,
        "total_volume_loss_m3": total_volume,
    }


def _binary_structure(radius_cells: int) -> np.ndarray:
    if radius_cells <= 0:
        return np.ones((3, 3), dtype=bool)
    y, x = np.ogrid[-radius_cells:radius_cells + 1, -radius_cells:radius_cells + 1]
    return (x * x + y * y) <= radius_cells * radius_cells


def _rasterize_nearest_to_existing_grid(
    points: np.ndarray,
    values: np.ndarray,
    shape: tuple[int, int],
    transform,
    *,
    local_transform: Optional["LocalCoordinateTransform"] = None,
) -> np.ndarray:
    import rasterio.transform
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if local_transform is not None:
        points = local_transform.to_global(points)

    rows, cols = shape
    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
    grid_x, grid_y = rasterio.transform.xy(transform, ys, xs, offset="center")
    grid_points = np.column_stack([np.asarray(grid_x).ravel(), np.asarray(grid_y).ravel()])
    tree = cKDTree(points[:, :2])
    max_dist = max(abs(transform.a), abs(transform.e)) * 2
    dists, indices = tree.query(grid_points, k=1, distance_upper_bound=max_dist)
    out = np.zeros(rows * cols, dtype=np.float64)
    valid = np.isfinite(dists) & (indices < len(values))
    out[valid] = values[indices[valid]]
    return out.reshape(shape)


def _write_geojson(output_path: Path, features: list[dict[str, Any]], crs: Optional[str]) -> None:
    data: dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if crs:
        data["crs"] = {"type": "name", "properties": {"name": crs}}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
