"""
Local Coordinate Transformation for Point Cloud Processing.

This module provides utilities for transforming point cloud coordinates between
global (e.g., UTM) and local coordinate systems. Large coordinate values in
geospatial data can cause floating-point precision issues, especially on GPUs.

The solution is to:
1. Shift coordinates to a local origin near (0, 0) during loading
2. Process all computations in local coordinates
3. Revert to original coordinates when writing outputs

This approach is standard in point cloud software like CloudCompare and PDAL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class LocalCoordinateTransform:
    """Transform between global and local coordinate systems.
    
    Stores the offset used to shift coordinates from global (UTM/projected)
    to local (near-origin) coordinates for numerical stability.
    
    The transform is defined as:
        local = global - offset
        global = local + offset
    
    Attributes:
        offset_x: X offset subtracted from global coordinates
        offset_y: Y offset subtracted from global coordinates
        offset_z: Z offset subtracted from global coordinates (default 0.0)
        origin_method: Method used to compute the offset ('min_bounds', 'centroid', 'first_point')
    
    Example:
        >>> transform = LocalCoordinateTransform.from_bounds(466000.0, 6650000.0)
        >>> global_pts = np.array([[466100, 6650200, 150.0]])
        >>> local_pts = transform.to_local(global_pts)  # -> [[100, 200, 150]]
        >>> restored = transform.to_global(local_pts)   # -> [[466100, 6650200, 150]]
    """
    
    offset_x: float
    offset_y: float
    offset_z: float = 0.0
    origin_method: str = field(default="unknown", repr=False)
    
    @classmethod
    def from_centroid(
        cls,
        points: "NDArray[np.floating]",
        *,
        include_z: bool = False,
    ) -> "LocalCoordinateTransform":
        """Create transform from point cloud centroid.
        
        Args:
            points: Nx3 array of point coordinates [X, Y, Z]
            include_z: If True, also offset Z coordinate
            
        Returns:
            LocalCoordinateTransform with offset set to centroid
            
        Raises:
            ValueError: If points array is empty or has wrong shape
        """
        if points.size == 0:
            raise ValueError("Cannot compute centroid from empty point cloud")
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"Expected Nx3 array, got shape {points.shape}")
            
        centroid = np.mean(points[:, :3], axis=0)
        return cls(
            offset_x=float(centroid[0]),
            offset_y=float(centroid[1]),
            offset_z=float(centroid[2]) if include_z else 0.0,
            origin_method="centroid",
        )
    
    @classmethod
    def from_bounds(
        cls,
        min_x: float,
        min_y: float,
        min_z: float = 0.0,
        *,
        include_z: bool = False,
    ) -> "LocalCoordinateTransform":
        """Create transform from bounding box minimum.
        
        This is the recommended method as it guarantees all local
        coordinates will be positive.
        
        Args:
            min_x: Minimum X coordinate (bounding box)
            min_y: Minimum Y coordinate (bounding box)
            min_z: Minimum Z coordinate (bounding box)
            include_z: If True, also offset Z coordinate
            
        Returns:
            LocalCoordinateTransform with offset set to min bounds
        """
        return cls(
            offset_x=float(min_x),
            offset_y=float(min_y),
            offset_z=float(min_z) if include_z else 0.0,
            origin_method="min_bounds",
        )
    
    @classmethod
    def from_first_point(
        cls,
        points: "NDArray[np.floating]",
        *,
        include_z: bool = False,
    ) -> "LocalCoordinateTransform":
        """Create transform from the first point in the cloud.
        
        Simple but arbitrary - use only when other methods aren't suitable.
        
        Args:
            points: Nx3 array of point coordinates [X, Y, Z]
            include_z: If True, also offset Z coordinate
            
        Returns:
            LocalCoordinateTransform with offset set to first point
            
        Raises:
            ValueError: If points array is empty
        """
        if points.size == 0:
            raise ValueError("Cannot use first point from empty point cloud")
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"Expected Nx3 array, got shape {points.shape}")
            
        first = points[0, :3]
        return cls(
            offset_x=float(first[0]),
            offset_y=float(first[1]),
            offset_z=float(first[2]) if include_z else 0.0,
            origin_method="first_point",
        )
    
    @classmethod
    def from_points(
        cls,
        points: "NDArray[np.floating]",
        *,
        method: Literal["min_bounds", "centroid", "first_point"] = "min_bounds",
        include_z: bool = False,
    ) -> "LocalCoordinateTransform":
        """Create transform from points using the specified method.
        
        Args:
            points: Nx3 array of point coordinates [X, Y, Z]
            method: Method for computing the offset
            include_z: If True, also offset Z coordinate
            
        Returns:
            LocalCoordinateTransform instance
        """
        if method == "centroid":
            return cls.from_centroid(points, include_z=include_z)
        elif method == "first_point":
            return cls.from_first_point(points, include_z=include_z)
        else:  # min_bounds (default)
            if points.size == 0:
                raise ValueError("Cannot compute bounds from empty point cloud")
            min_x = float(points[:, 0].min())
            min_y = float(points[:, 1].min())
            min_z = float(points[:, 2].min()) if include_z else 0.0
            return cls.from_bounds(min_x, min_y, min_z, include_z=include_z)
    
    @classmethod
    def identity(cls) -> "LocalCoordinateTransform":
        """Create an identity transform (no offset).
        
        Useful as a fallback when no transformation is needed.
        """
        return cls(offset_x=0.0, offset_y=0.0, offset_z=0.0, origin_method="identity")
    
    def to_local(self, points: "NDArray[np.floating]") -> "NDArray[np.floating]":
        """Transform global coordinates to local.
        
        Args:
            points: Nx3 array of global coordinates [X, Y, Z]
            
        Returns:
            Nx3 array of local coordinates (points - offset)
        """
        if points.size == 0:
            return points.copy()
            
        result = points.copy()
        result[:, 0] -= self.offset_x
        result[:, 1] -= self.offset_y
        if self.offset_z != 0.0:
            result[:, 2] -= self.offset_z
        return result
    
    def to_global(self, points: "NDArray[np.floating]") -> "NDArray[np.floating]":
        """Transform local coordinates back to global.
        
        Args:
            points: Nx3 array of local coordinates [X, Y, Z]
            
        Returns:
            Nx3 array of global coordinates (points + offset)
        """
        if points.size == 0:
            return points.copy()
            
        result = points.copy()
        result[:, 0] += self.offset_x
        result[:, 1] += self.offset_y
        if self.offset_z != 0.0:
            result[:, 2] += self.offset_z
        return result
    
    def transform_bounds(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        *,
        to_local: bool = True,
    ) -> tuple[float, float, float, float]:
        """Transform 2D bounds between coordinate systems.
        
        Args:
            min_x, min_y, max_x, max_y: Bounding box coordinates
            to_local: If True, convert global->local; else local->global
            
        Returns:
            Tuple of (min_x, min_y, max_x, max_y) in target coordinate system
        """
        if to_local:
            return (
                min_x - self.offset_x,
                min_y - self.offset_y,
                max_x - self.offset_x,
                max_y - self.offset_y,
            )
        else:
            return (
                min_x + self.offset_x,
                min_y + self.offset_y,
                max_x + self.offset_x,
                max_y + self.offset_y,
            )
    
    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON/YAML storage.
        
        Returns:
            Dictionary representation of the transform
        """
        return {
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
            "offset_z": self.offset_z,
            "origin_method": self.origin_method,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "LocalCoordinateTransform":
        """Deserialize from dictionary.
        
        Args:
            data: Dictionary with offset_x, offset_y, and optionally offset_z
            
        Returns:
            LocalCoordinateTransform instance
        """
        return cls(
            offset_x=float(data["offset_x"]),
            offset_y=float(data["offset_y"]),
            offset_z=float(data.get("offset_z", 0.0)),
            origin_method=data.get("origin_method", "unknown"),
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.offset_z != 0.0:
            return f"LocalTransform(offset=[{self.offset_x:.2f}, {self.offset_y:.2f}, {self.offset_z:.2f}], method={self.origin_method})"
        return f"LocalTransform(offset=[{self.offset_x:.2f}, {self.offset_y:.2f}], method={self.origin_method})"
