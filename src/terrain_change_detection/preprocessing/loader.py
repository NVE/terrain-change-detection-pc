"""
Point Cloud Data Loader

This module handles loading and initial validation of point cloud datasets.
"""

import laspy
import numpy as np
from pathlib import Path
from typing import Optional, List
from ..utils.logging import setup_logger
from ..utils.point_cloud_filters import create_classification_mask, get_filter_statistics

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

    def __init__(self, *, ground_only: bool = True, classification_filter: Optional[List[int]] = None):
        """
        Initialize the point cloud loader.

        Args:
            ground_only: If True, filter points by ground classification (default True)
            classification_filter: List of classification codes to keep (default [2] when ground_only)
        """
        self.ground_only = ground_only
        self.classification_filter = classification_filter

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

            # Get total point count before filtering
            total_points = len(laz_file.points)

            # Build mask using shared utility function
            if hasattr(laz_file, 'classification'):
                classes = np.array(laz_file.classification)
                ground_mask = create_classification_mask(classes, self.ground_only, self.classification_filter)
            else:
                # If classification is not available, keep all points (warn if ground_only requested)
                if self.ground_only:
                    logger.warning("Classification not available; proceeding without ground filtering.")
                ground_mask = np.ones(total_points, dtype=bool)

            ground_point_count = int(np.sum(ground_mask))

            if ground_point_count == 0:
                logger.warning(f"No ground points found in file: {file_path}")

            # Log filtering statistics using shared utility
            if total_points > 0:
                stats = get_filter_statistics(total_points, ground_point_count, self.ground_only, self.classification_filter)
                logger.info(
                    f"Found {stats['filtered_points']} points ({stats['filter_description']}) "
                    f"out of {stats['total_points']} total ({stats['percentage']:.1f}%)"
                )

            # Extract coordinates for selected points
            points = np.column_stack([
                np.array(laz_file.x, dtype=np.float64)[ground_mask],
                np.array(laz_file.y, dtype=np.float64)[ground_mask],
                np.array(laz_file.z, dtype=np.float64)[ground_mask]
            ])

            # Extract useful attributes for selected points based on sample exploration
            attributes = {}

            # Core attributes that are typically available
            if hasattr(laz_file, 'intensity'):
                attributes['intensity'] = np.array(laz_file.intensity)[ground_mask]
            if hasattr(laz_file, 'return_number'):
                attributes['return_number'] = np.array(laz_file.return_number)[ground_mask]
            if hasattr(laz_file, 'number_of_returns'):
                attributes['number_of_returns'] = np.array(laz_file.number_of_returns)[ground_mask]
            if hasattr(laz_file, 'scan_angle_rank'):
                attributes['scan_angle_rank'] = np.array(laz_file.scan_angle_rank)[ground_mask]
            if hasattr(laz_file, 'point_source_id'):
                attributes['point_source_id'] = np.array(laz_file.point_source_id)[ground_mask]
            if hasattr(laz_file, 'gps_time'):
                attributes['gps_time'] = np.array(laz_file.gps_time)[ground_mask]

            # Quality flags
            if hasattr(laz_file, 'scan_direction_flag'):
                attributes['scan_direction_flag'] = np.array(laz_file.scan_direction_flag)[ground_mask]
            if hasattr(laz_file, 'edge_of_flight_line'):
                attributes['edge_of_flight_line'] = np.array(laz_file.edge_of_flight_line)[ground_mask]
            if hasattr(laz_file, 'synthetic'):
                attributes['synthetic'] = np.array(laz_file.synthetic)[ground_mask]
            if hasattr(laz_file, 'key_point'):
                attributes['key_point'] = np.array(laz_file.key_point)[ground_mask]
            if hasattr(laz_file, 'withheld'):
                attributes['withheld'] = np.array(laz_file.withheld)[ground_mask]
            if hasattr(laz_file, 'user_data'):
                attributes['user_data'] = np.array(laz_file.user_data)[ground_mask]

            # RGB colors if available
            if hasattr(laz_file, 'red') and hasattr(laz_file, 'green') and hasattr(laz_file, 'blue'):
                attributes['colors'] = np.column_stack([
                    np.array(laz_file.red)[ground_mask],
                    np.array(laz_file.green)[ground_mask],
                    np.array(laz_file.blue)[ground_mask]
                ])

            # Extract metadata using the dedicated method
            metadata = self.get_metadata(str(file_path))
            # Add file_path to metadata for consistency with original implementation
            metadata['file_path'] = str(file_path)

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
        try:
            file_path = Path(file_path)

            # Check if file exists
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                return False

            # Check file extension
            if file_path.suffix.lower() not in ['.las', '.laz']:
                logger.warning(f"Unsupported file format: {file_path.suffix}")
                return False

            # Try to read the header
            laz_file = laspy.read(file_path)
            # Basic validation checks
            if len(laz_file.points) == 0:
                logger.warning(f"No points found in file: {file_path}")
                return False

            # Check if the file has valid coordinates
            if not (np.isfinite(laz_file.x).all() and
                    np.isfinite(laz_file.y).all() and
                    np.isfinite(laz_file.z).all()):
                logger.warning(f"Invalid coordinates in file: {file_path}")
                return False

            logger.info(f"File validated successfully: {file_path}")
            return True

        except Exception as e:
            logger.error(f"File validation failed for {file_path}: {e}")
            return False

    def get_metadata(self, file_path: str, *, header_only: bool = False, chunk_points: int = 1_000_000) -> dict:
        """
        Extract metadata from a point cloud file.

        Args:
            file_path: Path to the LAS/LAZ file

        Returns:
            Dictionary containing metadata information
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            if header_only:
                # Lightweight path: use header and stream-only attributes
                with laspy.open(file_path) as reader:
                    header = reader.header
                    num_points = int(header.point_count)
                    bounds = {
                        'min_x': float(header.x_min),
                        'max_x': float(header.x_max),
                        'min_y': float(header.y_min),
                        'max_y': float(header.y_max),
                        'min_z': float(header.z_min),
                        'max_z': float(header.z_max),
                    }
                    # Classification stats via streaming (no full load)
                    class_counts: dict[int, int] = {}
                    try:
                        for chunk in reader.chunk_iterator(chunk_points):
                            if hasattr(chunk, 'classification'):
                                cls = np.asarray(chunk.classification)
                                # bincount over observed classes in this chunk
                                bc = np.bincount(cls, minlength=256)
                                for code, cnt in enumerate(bc):
                                    if cnt:
                                        class_counts[code] = class_counts.get(code, 0) + int(cnt)
                            else:
                                # No classification; skip stats
                                pass
                    except Exception:
                        # If streaming classification fails, leave stats minimal
                        class_counts = {}

                    ground_points = int(class_counts.get(2, 0)) if class_counts else 0
                    ground_percentage = (100.0 * ground_points / num_points) if num_points > 0 and ground_points > 0 else 0.0

                    classification_stats = {
                        'unique_classes': sorted([int(k) for k, v in class_counts.items() if v > 0]),
                        'class_counts': class_counts,
                        'ground_points': ground_points,
                        'ground_percentage': float(ground_percentage),
                    }

                    metadata = {
                        'filename': file_path.name,
                        'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                        'num_points': num_points,
                        'point_format': getattr(header, 'point_data_record_format', getattr(header, 'point_format', 0)),
                        'version': f"{header.version}",
                        'creation_date': getattr(header, 'creation_date', None),
                        'bounds': bounds,
                        'scales': [float(header.x_scale), float(header.y_scale), float(header.z_scale)],
                        'offsets': [float(header.x_offset), float(header.y_offset), float(header.z_offset)],
                        'classification_stats': classification_stats,
                        # No heavy per-point statistics in header_only mode
                    }
                    return metadata

            # Full metadata path (may load arrays)
            laz_file = laspy.read(file_path)

            # Get classification statistics
            classification_stats = {}
            if hasattr(laz_file, 'classification'):
                classifications = np.array(laz_file.classification)
                unique_classes, counts = np.unique(classifications, return_counts=True)
                total_points = len(classifications)

                classification_stats = {
                    'unique_classes': unique_classes.tolist(),
                    'class_counts': dict(zip(unique_classes.tolist(), counts.tolist())),
                    'ground_points': int(counts[unique_classes == 2][0]) if 2 in unique_classes else 0,
                    'ground_percentage': float(counts[unique_classes == 2][0] / total_points * 100) if 2 in unique_classes else 0.0
                }

            metadata = {
                'filename': file_path.name,
                'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                'num_points': len(laz_file.points),
                'point_format': getattr(laz_file.header, 'point_data_record_format', getattr(laz_file.header, 'point_format', 0)),
                'version': f"{laz_file.header.version}",
                'creation_date': getattr(laz_file.header, 'creation_date', None),
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
                'classification_stats': classification_stats
            }

            # Available dimensions based on point format
            available_dimensions = ['X', 'Y', 'Z']
            if hasattr(laz_file, 'intensity'):
                available_dimensions.append('intensity')
            if hasattr(laz_file, 'return_number'):
                available_dimensions.append('return_number')
            if hasattr(laz_file, 'number_of_returns'):
                available_dimensions.append('number_of_returns')
            if hasattr(laz_file, 'scan_direction_flag'):
                available_dimensions.append('scan_direction_flag')
            if hasattr(laz_file, 'edge_of_flight_line'):
                available_dimensions.append('edge_of_flight_line')
            if hasattr(laz_file, 'classification'):
                available_dimensions.append('classification')
            if hasattr(laz_file, 'synthetic'):
                available_dimensions.append('synthetic')
            if hasattr(laz_file, 'key_point'):
                available_dimensions.append('key_point')
            if hasattr(laz_file, 'withheld'):
                available_dimensions.append('withheld')
            if hasattr(laz_file, 'scan_angle_rank'):
                available_dimensions.append('scan_angle_rank')
            if hasattr(laz_file, 'user_data'):
                available_dimensions.append('user_data')
            if hasattr(laz_file, 'point_source_id'):
                available_dimensions.append('point_source_id')
            if hasattr(laz_file, 'gps_time'):
                available_dimensions.append('gps_time')
            if hasattr(laz_file, 'red') and hasattr(laz_file, 'green') and hasattr(laz_file, 'blue'):
                available_dimensions.extend(['red', 'green', 'blue'])

            metadata['available_dimensions'] = available_dimensions

            # Add statistics for ground points only
            if classification_stats.get('ground_points', 0) > 0:
                ground_mask = np.array(laz_file.classification) == 2
                x_array = np.array(laz_file.x)[ground_mask]
                y_array = np.array(laz_file.y)[ground_mask]
                z_array = np.array(laz_file.z)[ground_mask]
            else:
                # Fallback to all points if no ground points
                x_array = np.array(laz_file.x)
                y_array = np.array(laz_file.y)
                z_array = np.array(laz_file.z)

            metadata['statistics'] = {
                'mean_x': float(np.mean(x_array)),
                'mean_y': float(np.mean(y_array)),
                'mean_z': float(np.mean(z_array)),
                'std_x': float(np.std(x_array)),
                'std_y': float(np.std(y_array)),
                'std_z': float(np.std(z_array)),
                'centroid': [
                    float(np.mean(x_array)),
                    float(np.mean(y_array)),
                    float(np.mean(z_array))
                ],
                'points_used_for_stats': 'ground_points_only' if classification_stats.get('ground_points', 0) > 0 else 'all_points'
            }

            return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata from {file_path}: {e}")
            raise
