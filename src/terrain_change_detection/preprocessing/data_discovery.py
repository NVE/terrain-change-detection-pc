"""
Data Discovery and Batch Loading

This module handles discovery and batch loading of point cloud datasets
organized in nested directory structures like:

data/raw/
├── area1/
│   ├── timeperiod1/
│   │   ├── data/
│   │   │   ├── file1.laz
│   │   │   ├── file2.laz
│   │   │   └── ...
│   │   └── metadata/
│   └── timeperiod2/
│       ├── data/
│       │   ├── file1.laz
│       │   ├── file2.laz
│       │   └── ...
│       └── metadata/
"""

import laspy
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from ..utils.logging import setup_logger
from .loader import PointCloudLoader

logger = setup_logger(__name__)

@dataclass
class DatasetInfo:
    """Information about a dataset."""
    area_name: str
    time_period: str
    laz_files: List[Path]
    metadata_dir: Optional[Path]
    total_points: int = 0  # This will store ground points count for terrain analysis
    bounds: Optional[Dict] = None
    per_file_stats: List[Dict] = field(default_factory=list)  # Add per-file stats

@dataclass
class AreaInfo:
    """Information about an area with multiple time periods."""
    area_name: str
    datasets: Dict[str, DatasetInfo] # time_period -> DatasetInfo

    @property
    def time_periods(self) -> List[str]:
        """Get sorted list of time periods."""
        return sorted(self.datasets.keys())

class DataDiscovery:
    """
    Discovers and catalogs point cloud datasets in nested directory structures.

    Expected structure:
    data/raw/
    ├── area1/
    │   ├── timeperiod1/
    │   │   ├── data/
    │   │   │   ├── file1.laz
    │   │   │   ├── file2.laz
    │   │   │   └── ...
    │   │   └── metadata/
    │   └── timeperiod2/
    │       ├── data/
    │       │   ├── file1.laz
    │       │   ├── file2.laz
    │       │   └── ...
    │       └── metadata/
    """

    def __init__(self, base_dir: str):
        """
        Initialize data discovery.

        Args:
            base_dir: Base directory containing area subdirectories.
        """
        self.base_dir = Path(base_dir)
        self.loader = PointCloudLoader()


    def scan_areas(self) -> Dict[str, AreaInfo]:
        """
        Scan the base directory for area subdirectories and their datasets.

        Returns:
            Dictionary mapping area names to AreaInfo objects.
        """
        areas = {}

        if not self.base_dir.exists():
            logger.warning(f"Base directory {self.base_dir} does not exist.")
            return areas

        # FInd area directories
        for area_path in self.base_dir.iterdir():
            if area_path.is_dir() and not area_path.name.startswith('.'):
                area_name = area_path.name
                logger.info(f"Found area directory: {area_name}")
                logger.info(f"Scanning area: {area_name}")

                datasets = self._scan_area_datasets(area_path)
                if datasets:
                    areas[area_name] = AreaInfo(area_name=area_name, datasets=datasets)
                    logger.info(f"Found {len(datasets)} time periods for area: {area_name}: {list(datasets.keys())}")

        return areas

    def _scan_area_datasets(self, area_path: Path) -> Dict[str, DatasetInfo]:
        """
        Scan an area directory for time periods and their datasets.

        Args:
            area_path: Path to the area directory.

        Returns:
            Dictionary mapping time period names to DatasetInfo objects.
        """
        datasets = {}

        for time_path in area_path.iterdir():
            if time_path.is_dir() and not time_path.name.startswith('.'):
                time_period = time_path.name
                logger.info(f"Found time period directory: {time_period}")

                data_dir = time_path / 'data'
                if not data_dir.is_dir():
                    continue

                # Find LAZ files in this time period's data directory
                laz_files = list(data_dir.glob('*.laz')) + list(data_dir.glob('*.las'))

                if laz_files:
                    # Check for metadata directory
                    metadata_dir = time_path / 'metadata'
                    if not metadata_dir.exists():
                        metadata_dir = None

                    # Create DatasetInfo object
                    dataset_info = DatasetInfo(
                        area_name=area_path.name,
                        time_period=time_period,
                        laz_files=laz_files,
                        metadata_dir=metadata_dir
                    )

                    # Get basic dataset statistics (can be slow for large datasets or many laz files)
                    try:
                        dataset_info.total_points, dataset_info.bounds, dataset_info.per_file_stats = self._get_dataset_stats(laz_files)
                    except Exception as e:
                        logger.error(f"Error getting stats for dataset {dataset_info.area_name}/{dataset_info.time_period}: {e}")

                    datasets[time_period] = dataset_info

        return datasets

    def _get_dataset_stats(self, laz_files: List[Path]) -> Tuple[int, Dict, List[Dict]]:
        """
        Get basic statistics for a dataset.

        Args:
            laz_files: List of LAZ file paths.

        Returns:
            Tuple of total ground points, bounds dictionary, and per-file statistics.
        """
        total_ground_points = 0
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')
        per_file_stats = []

        for laz_file in laz_files:
            try:
                metadata = self.loader.get_metadata(str(laz_file))
                
                # Get ground points count from classification stats
                classification_stats = metadata.get('classification_stats', {})
                ground_points = classification_stats.get('ground_points', 0)
                total_ground_points += ground_points

                bounds = metadata['bounds']
                min_x = min(min_x, bounds['min_x'])
                min_y = min(min_y, bounds['min_y'])
                min_z = min(min_z, bounds['min_z'])
                max_x = max(max_x, bounds['max_x'])
                max_y = max(max_y, bounds['max_y'])
                max_z = max(max_z, bounds['max_z'])

                # Store per-file stats
                per_file_stats.append({
                    'file': laz_file.name,
                    'num_points': metadata['num_points'],
                    'ground_points': ground_points,
                    'bounds': bounds
                })

            except Exception as e:
                logger.error(f"Error reading file {laz_file}: {e}")

        combined_bounds = {
            'min_x': min_x,
            'min_y': min_y,
            'min_z': min_z,
            'max_x': max_x,
            'max_y': max_y,
            'max_z': max_z
        }

        return total_ground_points, combined_bounds, per_file_stats


class BatchLoader:
    """
    Loads and comibines multiple point cloud files into unified datasets
    """

    def __init__(self):
        """
        Initialize the batch loader.
        """
        self.loader = PointCloudLoader()

    def load_dataset(self, dataset_info: DatasetInfo,
                     max_points_per_file: Optional[int] = None) -> Dict:
        """
        Load all files and combine into a single point cloud.

        Args:
            dataset_info: DatasetInfo object containing dataset details.
            max_points_per_file: Optional maximum points to load from each file.

        Returns:
            Combined point cloud data
        """
        all_points = []
        all_attributes = {}
        file_metadata = []

        for laz_file in dataset_info.laz_files:
            try:
                logger.info(f"Loading file: {laz_file}")
                pc_data = self.loader.load(str(laz_file))

                points = pc_data['points']

                # Subsample if max_points_per_file is set
                if max_points_per_file and len(points) > max_points_per_file:
                    indices = np.random.choice(len(points), max_points_per_file, replace=False)
                    points = points[indices]

                    # Also subsample attributes
                    for attr_name, attr_data in pc_data['attributes'].items():
                        if attr_name not in all_attributes:
                            all_attributes[attr_name] = []
                        all_attributes[attr_name].append(attr_data[indices])
                else:
                    # Add full attributes
                    for attr_name, attr_data in pc_data['attributes'].items():
                        if attr_name not in all_attributes:
                            all_attributes[attr_name] = []
                        all_attributes[attr_name].append(attr_data)

                all_points.append(points)
                file_metadata.append(pc_data['metadata'])

            except Exception as e:
                logger.warning(f"Failed to load {laz_file}: {e}")

        if not all_points:
            logger.warning(f"No valid point cloud data found in dataset {dataset_info.area_name}/{dataset_info.time_period}")
            return {
                'points': np.array([]),
                'attributes': {},
                'metadata': {
                    'area_name': dataset_info.area_name,
                    'time_period': dataset_info.time_period,
                    'num_files': 0,
                    'total_points': 0,
                    'file_metadata': [],
                    'bounds': {},
                    'dataset_info': dataset_info
                }
            }

        # Combine all points
        combined_points = np.vstack(all_points)

        # Combine all attributes
        combined_attributes = {}
        for attr_name, attr_list in all_attributes.items():
            if attr_list:
                combined_attributes[attr_name] = np.concatenate(attr_list)

        # Create combined metadata
        combined_metadata = {
            'area_name': dataset_info.area_name,
            'time_period': dataset_info.time_period,
            'num_files': len(all_points),
            'total_points': len(combined_points),
            'file_metadata': file_metadata,
            'bounds': self._compute_bounds(combined_points),
            'dataset_info': dataset_info
        }

        logger.info(f"Combined dataset: {len(combined_points)} points from {len(all_points)} files")

        return {
            'points': combined_points,
            'attributes': combined_attributes,
            'metadata': combined_metadata
        }

    def _load_separate_files(self, dataset_info: DatasetInfo,
                             max_points_per_file: Optional[int] = None) -> Dict:
        """
        Load files separately (useful for processing individual tiles).

        Args:
            dataset_info: DatasetInfo object containing dataset details.
            max_points_per_file: Optional maximum points to load from each file.

        Returns:
            Dictionary with separate point cloud data
        """
        file_data = []

        for laz_file in dataset_info.laz_files:
            try:
                pc_data = self.loader.load(str(laz_file))

                # Subsample if max_points_per_file is set
                if max_points_per_file and len(pc_data['points']) > max_points_per_file:
                    indices = np.random.choice(len(pc_data['points']), max_points_per_file, replace=False)
                    pc_data['points'] = pc_data['points'][indices]

                    # Also subsample attributes
                    for attr_name, attr_data in pc_data['attributes'].items():
                        pc_data['attributes'][attr_name] = attr_data[indices]

                file_data.append(pc_data)

            except Exception as e:
                logger.warning(f"Failed to load {laz_file}: {e}")

        return {
            'file_data': file_data,
            'metadata': {
                'area_name': dataset_info.area_name,
                'time_period': dataset_info.time_period,
                'num_files': len(file_data),
                'dataset_info': dataset_info
            }
        }

    def _compute_bounds(self, points: np.ndarray) -> Dict:
        """
        Compute bounding box for point cloud.
        """
        return {
            'min_x': float(np.min(points[:, 0])),
            'min_y': float(np.min(points[:, 1])),
            'min_z': float(np.min(points[:, 2])),
            'max_x': float(np.max(points[:, 0])),
            'max_y': float(np.max(points[:, 1])),
            'max_z': float(np.max(points[:, 2]))
        }