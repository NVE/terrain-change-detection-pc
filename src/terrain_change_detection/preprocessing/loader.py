"""
Point Cloud Data Loader

This module handles loading and initial validation of point cloud datasets.
"""

import laspy
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class PointCloudLoader:
    """
    A class for loading point cloud data from LAS/LAZ files.

    Features:
    - Support for LAS/LAZ formats using laspy
    - Metadata extraction
    - Coordinate system information handling
    - Data validation and quality checks
    """

    def __init__(self):
        """
        Initialize the point cloud loader.
        """
        pass

    def load(self, file_path: str) -> dict:
        """
        Load a point cloud file and return its data and metadata.

        Args:
            file_path: Path to the LAS/LAZ file

        Returns:
            dict: A dictionary containing point cloud data and metadata

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file format is unsupported or invalid
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in ['.las', '.laz']:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Loading point cloud data from {file_path}")

        try:
            # Load LAS/LAZ file using laspy
            laz_file = laspy.read(file_path)
            # Extract coordinates
            points = np.column_stack([
                np.array(laz_file.x, dtype=np.float64),
                np.array(laz_file.y, dtype=np.float64),
                np.array(laz_file.z, dtype=np.float64)
            ])
            # Keep only the 'Ground' points
            
            # Extract additional attributes if available
            attributes = {}
            if hasattr(laz_file, 'intensity'):
                attributes['intensity'] = np.array(laz_file.intensity)
            if hasattr(laz_file, 'classification'):
                attributes['classification'] = np.array(laz_file.classification)
            if hasattr(laz_file, 'return_number'):
                attributes['return_number'] = np.array(laz_file.return_number)
            if hasattr(laz_file, 'red') and hasattr(laz_file, 'green') and hasattr(laz_file, 'blue'):
                attributes['colors'] = np.column_stack([
                    np.array(laz_file.red),
                    np.array(laz_file.green),
                    np.array(laz_file.blue)
                ])

            # Extract metadata
            metadata = {
                'num_points': len(points),
                'bounds': {
                    'min_x': float(laz_file.header.x_min),
                    'max_x': float(laz_file.header.x_max),
                    'min_y': float(laz_file.header.y_min),
                    'max_y': float(laz_file.header.y_max),
                    'min_z': float(laz_file.header.z_min),
                    'max_z': float(laz_file.header.z_max)
                },
                'scales': [
                    float(laz_file.header.x_scale),
                    float(laz_file.header.y_scale),
                    float(laz_file.header.z_scale)
                ],
                'offsets': [
                    float(laz_file.header.x_offset),
                    float(laz_file.header.y_offset),
                    float(laz_file.header.z_offset)
                ],
                'point_format': getattr(laz_file.header, 'point_data_record_format', getattr(laz_file.header, 'point_format', 0)),
                'file_path': str(file_path)
            }

            return {
                'points': points,
                'attributes': attributes,
                'metadata': metadata
            }
        
        except Exception as e:
            logger.error(f"Error loading point cloud data from {file_path}: {e}")
            raise

    def validate_file(self, file_path: str) -> bool:
        """
        Validatea a point cloud file.

        Args:
            file_path: Path to the LAS/LAZ file

        Returns:
            True if the file is valid, False otherwise
        """




        
        