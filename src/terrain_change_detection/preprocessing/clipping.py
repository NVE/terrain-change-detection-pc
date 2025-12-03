"""
Area Clipping Module for Point Clouds

This module provides functionality to clip point clouds to a specific region
of interest defined by a polygon boundary. It supports both GeoJSON and 
Shapefile formats for defining the clipping boundary.

Use cases:
- Focus analysis on specific areas (e.g., rivers, erosion zones)
- Reduce data volume before ICP registration and change detection
- Exclude irrelevant areas from computations

The clipping is performed before ICP registration to ensure only relevant
areas are processed in subsequent calculations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    from shapely.geometry import shape, Point, Polygon, MultiPolygon
    from shapely.prepared import prep
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

try:
    import fiona
    FIONA_AVAILABLE = True
except ImportError:
    FIONA_AVAILABLE = False

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class AreaClipper:
    """
    Clips point cloud data to a region of interest defined by polygon boundaries.
    
    Supports:
    - GeoJSON files (.geojson, .json)
    - Shapefiles (.shp) - requires fiona library
    - Direct polygon coordinate input
    
    The clipping uses an efficient point-in-polygon test from shapely.
    
    Example usage:
        # From GeoJSON file
        clipper = AreaClipper.from_file("boundary.geojson")
        clipped_points = clipper.clip(points)
        
        # From polygon coordinates
        polygon_coords = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]
        clipper = AreaClipper.from_polygon(polygon_coords)
        clipped_points = clipper.clip(points)
    """
    
    def __init__(self, geometry: Union["Polygon", "MultiPolygon"]):
        """
        Initialize the clipper with a shapely geometry.
        
        Args:
            geometry: A shapely Polygon or MultiPolygon defining the clipping boundary.
        
        Raises:
            ImportError: If shapely is not installed.
            ValueError: If geometry is invalid or not a polygon type.
        """
        if not SHAPELY_AVAILABLE:
            raise ImportError(
                "shapely is required for area clipping. "
                "Install it with: uv add shapely"
            )
        
        if not isinstance(geometry, (Polygon, MultiPolygon)):
            raise ValueError(
                f"Geometry must be a Polygon or MultiPolygon, got {type(geometry).__name__}"
            )
        
        if not geometry.is_valid:
            logger.warning("Input geometry is invalid, attempting to fix with buffer(0)")
            geometry = geometry.buffer(0)
            if not geometry.is_valid:
                raise ValueError("Could not fix invalid geometry")
        
        self.geometry = geometry
        # Prepare geometry for faster repeated point-in-polygon tests
        self._prepared = prep(geometry)
        
        # Cache bounds for quick rejection
        self._bounds = geometry.bounds  # (minx, miny, maxx, maxy)
        
        logger.info(
            f"AreaClipper initialized with {type(geometry).__name__}, "
            f"bounds: ({self._bounds[0]:.2f}, {self._bounds[1]:.2f}) to "
            f"({self._bounds[2]:.2f}, {self._bounds[3]:.2f})"
        )
    
    @classmethod
    def from_file(
        cls, 
        file_path: Union[str, Path],
        feature_name: Optional[str] = None
    ) -> "AreaClipper":
        """
        Create an AreaClipper from a GeoJSON or Shapefile.
        
        Args:
            file_path: Path to the vector file (.geojson, .json, or .shp)
            feature_name: Optional name of specific feature to use. If the file 
                         contains multiple features, this filters by the 'name' 
                         property. If None, all polygons are combined.
        
        Returns:
            AreaClipper instance
        
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is unsupported or contains no valid polygons.
            ImportError: If required libraries are not installed.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Clipping boundary file not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix in ['.geojson', '.json']:
            return cls._from_geojson(file_path, feature_name=feature_name)
        elif suffix == '.shp':
            return cls._from_shapefile(file_path, feature_name=feature_name)
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                "Supported formats: .geojson, .json, .shp"
            )
    
    @classmethod
    def _from_geojson(cls, file_path: Path, feature_name: Optional[str] = None) -> "AreaClipper":
        """Load clipping boundary from a GeoJSON file."""
        if not SHAPELY_AVAILABLE:
            raise ImportError(
                "shapely is required to read GeoJSON files. "
                "Install it with: uv add shapely"
            )
        
        logger.info(f"Loading clipping boundary from: {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
        
        return cls._from_geojson_dict(geojson_data, str(file_path), feature_name=feature_name)
    
    @classmethod
    def _from_geojson_dict(
        cls, 
        geojson_data: dict, 
        source_name: str = "input",
        feature_name: Optional[str] = None
    ) -> "AreaClipper":
        """Create an AreaClipper from a GeoJSON dictionary."""
        polygons = []
        matched_feature = False
        
        # Handle different GeoJSON structures
        if geojson_data.get('type') == 'FeatureCollection':
            features = geojson_data.get('features', [])
        elif geojson_data.get('type') == 'Feature':
            features = [geojson_data]
        elif geojson_data.get('type') in ['Polygon', 'MultiPolygon']:
            # Direct geometry object
            geom = shape(geojson_data)
            if isinstance(geom, (Polygon, MultiPolygon)):
                polygons.append(geom)
            features = []
        else:
            raise ValueError(f"Unsupported GeoJSON type: {geojson_data.get('type')}")
        
        # Extract polygons from features
        for feature in features:
            # Check if we should filter by feature name
            if feature_name is not None:
                props = feature.get('properties', {})
                fname = props.get('name') or props.get('id') or props.get('NAME') or props.get('Id')
                if fname != feature_name:
                    continue
                matched_feature = True
            
            geom_dict = feature.get('geometry')
            if geom_dict is None:
                continue
            
            geom = shape(geom_dict)
            if isinstance(geom, Polygon):
                polygons.append(geom)
            elif isinstance(geom, MultiPolygon):
                polygons.extend(geom.geoms)
        
        if not polygons:
            if feature_name is not None and not matched_feature:
                raise ValueError(
                    f"Feature '{feature_name}' not found in {source_name}. "
                    "Check the feature name property in your GeoJSON file."
                )
            raise ValueError(
                f"No valid polygons found in {source_name}. "
                "The file must contain Polygon or MultiPolygon geometries."
            )
        
        # Combine all polygons into a single geometry
        if len(polygons) == 1:
            combined = polygons[0]
        else:
            combined = MultiPolygon(polygons)
        
        if feature_name:
            logger.info(f"Using feature '{feature_name}' with {len(polygons)} polygon(s)")
        else:
            logger.info(f"Loaded {len(polygons)} polygon(s), combined into clipping boundary")
        
        return cls(combined)
    
    @classmethod
    def _from_shapefile(cls, file_path: Path, feature_name: Optional[str] = None) -> "AreaClipper":
        """Load clipping boundary from a Shapefile."""
        if not FIONA_AVAILABLE:
            raise ImportError(
                "fiona is required to read Shapefiles. "
                "Install it with: uv add fiona\n"
                "Alternatively, convert your Shapefile to GeoJSON using QGIS or ogr2ogr."
            )
        
        if not SHAPELY_AVAILABLE:
            raise ImportError(
                "shapely is required for area clipping. "
                "Install it with: uv add shapely"
            )
        
        logger.info(f"Loading clipping boundary from Shapefile: {file_path}")
        
        polygons = []
        
        with fiona.open(file_path, 'r') as src:
            for feature in src:
                geom = shape(feature['geometry'])
                if isinstance(geom, Polygon):
                    polygons.append(geom)
                elif isinstance(geom, MultiPolygon):
                    polygons.extend(geom.geoms)
        
        if not polygons:
            raise ValueError(
                f"No valid polygons found in {file_path}. "
                "The Shapefile must contain polygon geometries."
            )
        
        # Combine all polygons
        if len(polygons) == 1:
            combined = polygons[0]
        else:
            combined = MultiPolygon(polygons)
        
        logger.info(f"Loaded {len(polygons)} polygon(s) from Shapefile")
        
        return cls(combined)
    
    @classmethod
    def from_polygon(
        cls,
        exterior_coords: list[tuple[float, float]],
        holes: Optional[list[list[tuple[float, float]]]] = None
    ) -> "AreaClipper":
        """
        Create an AreaClipper from polygon coordinates.
        
        Args:
            exterior_coords: List of (x, y) tuples defining the exterior ring.
                            The ring should be closed (first == last point).
            holes: Optional list of interior rings (holes) in the polygon.
        
        Returns:
            AreaClipper instance
        
        Example:
            # Simple rectangle
            coords = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]
            clipper = AreaClipper.from_polygon(coords)
        """
        if not SHAPELY_AVAILABLE:
            raise ImportError(
                "shapely is required for area clipping. "
                "Install it with: uv add shapely"
            )
        
        polygon = Polygon(exterior_coords, holes)
        
        return cls(polygon)
    
    @classmethod
    def from_bounds(
        cls,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float
    ) -> "AreaClipper":
        """
        Create an AreaClipper from bounding box coordinates.
        
        Args:
            min_x: Minimum X coordinate
            min_y: Minimum Y coordinate
            max_x: Maximum X coordinate
            max_y: Maximum Y coordinate
        
        Returns:
            AreaClipper instance
        
        Example:
            clipper = AreaClipper.from_bounds(500, 500, 1500, 1500)
        """
        coords = [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
            (min_x, min_y)  # Close the ring
        ]
        return cls.from_polygon(coords)
    
    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return the bounding box (minx, miny, maxx, maxy) of the clipping geometry."""
        return self._bounds
    
    @property
    def area(self) -> float:
        """Return the area of the clipping geometry in square units."""
        return self.geometry.area
    
    def clip(
        self,
        points: np.ndarray,
        *,
        return_mask: bool = False
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Clip points to the region of interest.
        
        Uses an efficient two-stage approach:
        1. Bounding box pre-filtering (fast rejection)
        2. Vectorized point-in-polygon test using shapely's contains_xy
        
        Args:
            points: (N, 3) array of XYZ coordinates. Only X and Y are used for clipping.
            return_mask: If True, also return the boolean mask of kept points.
        
        Returns:
            If return_mask=False: (M, 3) array of points inside the clipping boundary.
            If return_mask=True: Tuple of (clipped_points, mask) where mask is boolean (N,).
        """
        if points.ndim != 2 or points.shape[1] < 2:
            raise ValueError(
                f"Points must be a 2D array with at least 2 columns (X, Y), "
                f"got shape {points.shape}"
            )
        
        n_points = len(points)
        
        if n_points == 0:
            mask = np.zeros(0, dtype=bool)
            if return_mask:
                return points, mask
            return points
        
        # Stage 1: Bounding box pre-filter (very fast)
        minx, miny, maxx, maxy = self._bounds
        x = points[:, 0]
        y = points[:, 1]
        
        bbox_mask = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
        n_in_bbox = int(np.sum(bbox_mask))
        
        if n_in_bbox == 0:
            logger.debug(
                f"Clipping: 0/{n_points} points - all outside bounding box"
            )
            mask = np.zeros(n_points, dtype=bool)
            if return_mask:
                return points[mask], mask
            return points[mask]
        
        # Stage 2: Vectorized point-in-polygon test
        # Use shapely's vectorized contains_xy for much better performance
        inside_mask = np.zeros(n_points, dtype=bool)
        candidate_indices = np.where(bbox_mask)[0]
        
        # Extract candidate coordinates
        candidate_x = x[candidate_indices]
        candidate_y = y[candidate_indices]
        
        # Use vectorized contains_xy (available in shapely >= 2.0)
        # This is MUCH faster than checking points one by one
        try:
            from shapely import contains_xy
            candidate_inside = contains_xy(self.geometry, candidate_x, candidate_y)
            inside_mask[candidate_indices] = candidate_inside
        except ImportError:
            # Fallback for older shapely versions - use batched approach
            logger.debug("Using fallback point-in-polygon (shapely < 2.0)")
            batch_size = 100000
            for start in range(0, len(candidate_indices), batch_size):
                end = min(start + batch_size, len(candidate_indices))
                batch_indices = candidate_indices[start:end]
                
                for idx in batch_indices:
                    pt = Point(points[idx, 0], points[idx, 1])
                    if self._prepared.covers(pt):
                        inside_mask[idx] = True
        
        n_inside = int(np.sum(inside_mask))
        percentage = 100.0 * n_inside / n_points if n_points > 0 else 0.0
        
        logger.info(
            f"Clipped to {n_inside:,}/{n_points:,} points ({percentage:.1f}%)"
        )
        
        if return_mask:
            return points[inside_mask], inside_mask
        return points[inside_mask]
    
    def clip_with_attributes(
        self,
        points: np.ndarray,
        attributes: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Clip points and their associated attributes.
        
        Args:
            points: (N, 3) array of XYZ coordinates.
            attributes: Dictionary mapping attribute names to (N,) or (N, K) arrays.
        
        Returns:
            Tuple of (clipped_points, clipped_attributes).
        """
        clipped_points, mask = self.clip(points, return_mask=True)
        
        clipped_attrs = {}
        for name, arr in attributes.items():
            clipped_attrs[name] = arr[mask]
        
        return clipped_points, clipped_attrs
    
    def get_statistics(self, points: np.ndarray) -> dict:
        """
        Get statistics about the clipping operation without actually clipping.
        
        Args:
            points: (N, 3) array of XYZ coordinates.
        
        Returns:
            Dictionary with statistics about points inside/outside the boundary.
        """
        _, mask = self.clip(points, return_mask=True)
        
        n_total = len(points)
        n_inside = np.sum(mask)
        n_outside = n_total - n_inside
        
        return {
            'total_points': n_total,
            'points_inside': int(n_inside),
            'points_outside': int(n_outside),
            'percentage_inside': float(100.0 * n_inside / n_total) if n_total > 0 else 0.0,
            'percentage_outside': float(100.0 * n_outside / n_total) if n_total > 0 else 0.0,
            'clipping_bounds': self._bounds,
            'clipping_area': float(self.area),
        }
    
    def to_geojson(self) -> dict:
        """
        Export the clipping boundary as a GeoJSON dictionary.
        
        Returns:
            GeoJSON Feature dictionary.
        """
        from shapely.geometry import mapping
        
        return {
            'type': 'Feature',
            'properties': {
                'description': 'Clipping boundary',
                'area': self.area,
            },
            'geometry': mapping(self.geometry)
        }
    
    def save_geojson(self, file_path: Union[str, Path]) -> None:
        """
        Save the clipping boundary to a GeoJSON file.
        
        Args:
            file_path: Output file path.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        geojson = {
            'type': 'FeatureCollection',
            'features': [self.to_geojson()]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2)
        
        logger.info(f"Saved clipping boundary to {file_path}")


def clip_point_cloud_files(
    input_files: list[Union[str, Path]],
    clipper: AreaClipper,
    output_dir: Union[str, Path],
    *,
    ground_only: bool = True,
    classification_filter: Optional[list[int]] = None,
) -> list[Path]:
    """
    Clip multiple LAZ files to a region of interest and save the results.
    
    This function streams through input files, clips points to the specified
    boundary, and writes the clipped results to new LAZ files.
    
    Args:
        input_files: List of input LAZ file paths.
        clipper: AreaClipper instance defining the clipping boundary.
        output_dir: Directory to save clipped files.
        ground_only: If True, only keep ground points (class 2).
        classification_filter: Optional list of classification codes to keep.
    
    Returns:
        List of output file paths.
    """
    import laspy
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_files = []
    total_input = 0
    total_output = 0
    
    for input_file in input_files:
        input_file = Path(input_file)
        output_file = output_dir / input_file.name
        
        logger.info(f"Clipping {input_file.name}...")
        
        # Read input file
        las = laspy.read(input_file)
        
        # Extract points
        points = np.column_stack([
            np.array(las.x, dtype=np.float64),
            np.array(las.y, dtype=np.float64),
            np.array(las.z, dtype=np.float64)
        ])
        
        # Apply classification filter if needed
        if ground_only or classification_filter:
            if hasattr(las, 'classification'):
                classes = np.array(las.classification)
                if classification_filter:
                    class_mask = np.isin(classes, classification_filter)
                elif ground_only:
                    class_mask = classes == 2
                else:
                    class_mask = np.ones(len(points), dtype=bool)
            else:
                class_mask = np.ones(len(points), dtype=bool)
        else:
            class_mask = np.ones(len(points), dtype=bool)
        
        # Clip to region of interest
        _, clip_mask = clipper.clip(points, return_mask=True)
        
        # Combine masks
        combined_mask = class_mask & clip_mask
        n_output = np.sum(combined_mask)
        
        total_input += len(points)
        total_output += n_output
        
        if n_output == 0:
            logger.warning(f"No points remain after clipping {input_file.name}, skipping output")
            continue
        
        # Create output LAS with clipped points
        header = laspy.LasHeader(
            point_format=las.header.point_format,
            version=las.header.version
        )
        header.offsets = las.header.offsets
        header.scales = las.header.scales
        
        out_las = laspy.LasData(header)
        out_las.x = np.array(las.x)[combined_mask]
        out_las.y = np.array(las.y)[combined_mask]
        out_las.z = np.array(las.z)[combined_mask]
        
        # Copy other attributes
        if hasattr(las, 'classification'):
            out_las.classification = np.array(las.classification)[combined_mask]
        if hasattr(las, 'intensity'):
            out_las.intensity = np.array(las.intensity)[combined_mask]
        if hasattr(las, 'return_number'):
            out_las.return_number = np.array(las.return_number)[combined_mask]
        if hasattr(las, 'number_of_returns'):
            out_las.number_of_returns = np.array(las.number_of_returns)[combined_mask]
        if hasattr(las, 'gps_time'):
            out_las.gps_time = np.array(las.gps_time)[combined_mask]
        
        out_las.write(str(output_file))
        output_files.append(output_file)
        
        logger.info(
            f"  {input_file.name}: {n_output:,}/{len(points):,} points "
            f"({100.0 * n_output / len(points):.1f}%)"
        )
    
    percentage = 100.0 * total_output / total_input if total_input > 0 else 0.0
    logger.info(
        f"Clipping complete: {total_output:,}/{total_input:,} points "
        f"({percentage:.1f}%) saved to {len(output_files)} files"
    )
    
    return output_files


def check_shapely_available() -> bool:
    """Check if shapely is available for clipping operations."""
    return SHAPELY_AVAILABLE


def check_fiona_available() -> bool:
    """Check if fiona is available for Shapefile reading."""
    return FIONA_AVAILABLE
