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

    def __init__(self, base_dir: str, *, data_dir_name: str = 'data', metadata_dir_name: str = 'metadata', 
                 source_type: str = 'hoydedata', loader: Optional[PointCloudLoader] = None):
        """
        Initialize data discovery.

        Args:
            base_dir: Base directory containing area subdirectories.
            data_dir_name: Name of data subdirectory (ignored if source_type='drone')
            metadata_dir_name: Name of metadata subdirectory (ignored if source_type='drone')
            source_type: Data source type - 'hoydedata' or 'drone'
                        - 'hoydedata': area/time_period/data/*.laz structure
                        - 'drone': area/time_period/*.laz structure (no data/ subdir)
            loader: Optional PointCloudLoader instance
        """
        self.base_dir = Path(base_dir)
        self.data_dir_name = data_dir_name if source_type == 'hoydedata' else None
        self.metadata_dir_name = metadata_dir_name if source_type == 'hoydedata' else None
        self.source_type = source_type
        self.loader = loader if loader is not None else PointCloudLoader()


    def scan_areas(self) -> Dict[str, AreaInfo]:
        """
        Scan the base directory for area subdirectories and their datasets.
        
        Expected structure:
        - base_dir/area/time_period/data/*.laz (hoydedata)
        - base_dir/area/time_period/*.laz (drone)
        
        Note: base_dir should point to the parent of area folders, not an area folder itself.

        Returns:
            Dictionary mapping area names to AreaInfo objects.
        """
        areas = {}

        if not self.base_dir.exists():
            logger.warning(f"Base directory {self.base_dir} does not exist.")
            return areas

        # Find area directories
        subdirs = [d for d in self.base_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        # Check if user might have pointed to an area folder instead of parent
        if len(subdirs) > 0:
            # Check if subdirs look like time periods (e.g., "2024", "2024-12-03", etc.)
            time_period_like = sum(1 for d in subdirs if d.name.replace('-', '').isdigit())
            if time_period_like == len(subdirs) and len(subdirs) > 0:
                logger.warning("=" * 80)
                logger.warning("CONFIGURATION WARNING:")
                logger.warning(f"All subdirectories in {self.base_dir} look like time periods.")
                logger.warning(f"Found: {[d.name for d in subdirs]}")
                logger.warning("")
                logger.warning("It appears you may have set base_dir to an area folder instead of its parent.")
                logger.warning(f"Expected structure: base_dir/area/time_period/{'data/' if self.source_type == 'hoydedata' else ''}*.laz")
                logger.warning(f"Current structure appears to be: base_dir/time_period/{'data/' if self.source_type == 'hoydedata' else ''}*.laz")
                logger.warning("")
                logger.warning("If this is intentional, the subdirectories will be treated as 'area' names.")
                logger.warning("=" * 80)
        
        for area_path in subdirs:
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

                # For drone data, look directly in time_path
                # For hoydedata, look in time_path/data_dir_name
                if self.data_dir_name:
                    data_dir = time_path / self.data_dir_name
                    if not data_dir.is_dir():
                        continue
                else:
                    # Drone source: files directly in time period directory
                    data_dir = time_path

                # Find LAZ files in this time period's data directory
                laz_files = list(data_dir.glob('*.laz')) + list(data_dir.glob('*.las'))

                if laz_files:
                    # Check for metadata directory (only for hoydedata)
                    metadata_dir = None
                    if self.metadata_dir_name:
                        metadata_dir_path = time_path / self.metadata_dir_name
                        if metadata_dir_path.exists():
                            metadata_dir = metadata_dir_path

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
                # Use header-only + streaming classification stats to avoid full loads
                metadata = self.loader.get_metadata(str(laz_file), header_only=True)

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
    Loads and combines multiple point cloud files into unified datasets.
    
    Supports both in-memory loading and streaming-aware mode that returns
    file paths and metadata for out-of-core processing.
    """

    def __init__(self, loader: Optional[PointCloudLoader] = None, streaming_mode: bool = False):
        """
        Initialize the batch loader.
        
        Args:
            loader: PointCloudLoader instance for in-memory loading
            streaming_mode: If True, prefer returning file paths over loading data
        """
        self.loader = loader if loader is not None else PointCloudLoader()
        self.streaming_mode = streaming_mode

    def load_dataset(self, dataset_info: DatasetInfo,
                     max_points_per_file: Optional[int] = None,
                     streaming: Optional[bool] = None) -> Dict:
        """
        Load all files and combine into a single point cloud, or return metadata for streaming.

        Args:
            dataset_info: DatasetInfo object containing dataset details.
            max_points_per_file: Optional maximum points to load from each file (in-memory mode only).
            streaming: If True, return file paths and metadata without loading data.
                      If None, uses self.streaming_mode.

        Returns:
            Combined point cloud data (in-memory mode) or file metadata (streaming mode)
        """
        use_streaming = streaming if streaming is not None else self.streaming_mode
        
        if use_streaming:
            return self._prepare_streaming_dataset(dataset_info)
        
        # In-memory loading (original behavior)
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

    def _prepare_streaming_dataset(self, dataset_info: DatasetInfo) -> Dict:
        """
        Prepare dataset metadata for streaming/out-of-core processing.
        
        Instead of loading all data into memory, returns file paths and
        metadata that can be used with LaspyStreamReader and tiling.
        
        Args:
            dataset_info: DatasetInfo object containing dataset details
            
        Returns:
            Dictionary with file paths, bounds, and metadata for streaming
        """
        logger.info(f"Preparing streaming dataset for {dataset_info.area_name}/{dataset_info.time_period}")
        
        # Return file paths as strings for compatibility with LaspyStreamReader
        file_paths = [str(f) for f in dataset_info.laz_files]
        
        # Aggregate totals for clearer logging
        total_ground = int(dataset_info.total_points or 0)
        total_all = 0
        if dataset_info.per_file_stats:
            for s in dataset_info.per_file_stats:
                total_all += int(s.get('num_points', 0))

        return {
            'mode': 'streaming',
            'file_paths': file_paths,
            'num_files': len(file_paths),
            'metadata': {
                'area_name': dataset_info.area_name,
                'time_period': dataset_info.time_period,
                'num_files': len(file_paths),
                'total_points_ground': total_ground,
                'total_points_all': total_all,
                'ground_percentage': (100.0 * total_ground / total_all) if total_all > 0 else None,
                'bounds': dataset_info.bounds,
                'dataset_info': dataset_info,
            }
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
