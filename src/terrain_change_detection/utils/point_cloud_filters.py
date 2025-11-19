"""
Point Cloud Filtering Utilities

Shared utilities for filtering point cloud data based on various criteria.
These functions are used across preprocessing, streaming, and acceleration modules.
"""

from typing import List, Optional

import numpy as np


def create_classification_mask(
    classification: np.ndarray,
    ground_only: bool = True,
    classification_filter: Optional[List[int]] = None,
) -> np.ndarray:
    """Create a boolean mask for point classification filtering.
    
    This is a shared utility used by both in-memory loaders and streaming readers
    to ensure consistent filtering behavior across the workflow.
    
    Args:
        classification: Array of classification codes for each point
        ground_only: If True, only accept ground points (class 2).
            Ignored if classification_filter is provided.
        classification_filter: List of classification codes to accept.
            If provided, overrides ground_only behavior.
            
    Returns:
        Boolean array indicating which points pass the filter (True = accept)
        
    Examples:
        >>> classes = np.array([1, 2, 2, 3, 2, 1])
        >>> mask = create_classification_mask(classes, ground_only=True)
        >>> mask
        array([False,  True,  True, False,  True, False])
        
        >>> mask = create_classification_mask(classes, classification_filter=[1, 2])
        >>> mask
        array([ True,  True,  True, False,  True,  True])
    """
    n = len(classification)
    
    # If custom filter provided, use it (takes precedence)
    if classification_filter is not None:
        return np.isin(classification, np.array(classification_filter))
    
    # If ground_only requested, filter for class 2
    if ground_only:
        return classification == 2
    
    # No filtering - accept all points
    return np.ones(n, dtype=bool)


def apply_classification_filter(
    points: np.ndarray,
    classification: np.ndarray,
    ground_only: bool = True,
    classification_filter: Optional[List[int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter points based on classification codes.
    
    Convenience function that creates a mask and applies it to both
    points and classification arrays.
    
    Args:
        points: Nx3 array of point coordinates [X, Y, Z]
        classification: Array of N classification codes
        ground_only: If True, only keep ground points (class 2)
        classification_filter: Optional list of classification codes to keep
        
    Returns:
        Tuple of (filtered_points, filtered_classifications)
        Both arrays will have M rows where M <= N
        
    Examples:
        >>> pts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> classes = np.array([1, 2, 2])
        >>> filtered_pts, filtered_classes = apply_classification_filter(
        ...     pts, classes, ground_only=True
        ... )
        >>> filtered_pts.shape
        (2, 3)
    """
    mask = create_classification_mask(classification, ground_only, classification_filter)
    return points[mask], classification[mask]


def get_filter_statistics(
    total_points: int,
    filtered_points: int,
    ground_only: bool = True,
    classification_filter: Optional[List[int]] = None,
) -> dict:
    """Generate statistics about point filtering results.
    
    Useful for logging and validation of filtering operations.
    
    Args:
        total_points: Total number of points before filtering
        filtered_points: Number of points after filtering
        ground_only: Whether ground-only filtering was applied
        classification_filter: Classification filter that was used, if any
        
    Returns:
        Dictionary with statistics including counts, percentage, and filter description
    """
    percentage = (filtered_points / total_points * 100.0) if total_points > 0 else 0.0
    
    # Describe the filter
    if classification_filter is not None:
        filter_desc = f"classification filter: {classification_filter}"
    elif ground_only:
        filter_desc = "ground only (class 2)"
    else:
        filter_desc = "no filter"
    
    return {
        "total_points": total_points,
        "filtered_points": filtered_points,
        "percentage": percentage,
        "filter_description": filter_desc,
        "ground_only": ground_only,
        "classification_filter": classification_filter,
    }
