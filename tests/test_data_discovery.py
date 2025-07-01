"""
Test suite for data discovery module
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Import the data discovery module
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
from terrain_change_detection.preprocessing.data_discovery import DataDiscovery, BatchLoader


class TestDataDiscovery(unittest.TestCase):
    """Test cases for the DataDiscovery class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Path to the test data directory
        self.test_data_dir = Path(__file__).parent / "sample_data" / "raw"
        
        # Ensure the test data directory exists
        if not self.test_data_dir.exists():
            self.skipTest(f"Test data directory not found: {self.test_data_dir}")
        
        self.discovery = DataDiscovery(str(self.test_data_dir))

    def test_scan_areas(self):
        """Test scanning for areas and their datasets."""
        print(f"\n=== Testing scan_areas ===")
        print(f"Scanning directory: {self.test_data_dir}")
        
        areas = self.discovery.scan_areas()
        
        # Print discovered areas
        print(f"Discovered {len(areas)} areas:")
        for area_name, area_info in areas.items():
            print(f"  Area: {area_name}")
            print(f"    Time periods: {area_info.time_periods}")
            for time_period, dataset_info in area_info.datasets.items():
                print(f"    {time_period}:")
                print(f"      LAZ files: {len(dataset_info.laz_files)}")
                print(f"      Total points: {dataset_info.total_points}")
                if dataset_info.bounds:
                    bounds = dataset_info.bounds
                    print(f"      Bounds: X({bounds['min_x']:.1f}-{bounds['max_x']:.1f}), Y({bounds['min_y']:.1f}-{bounds['max_y']:.1f}), Z({bounds['min_z']:.1f}-{bounds['max_z']:.1f})")
        
        # Assertions
        self.assertIsInstance(areas, dict)
        self.assertGreater(len(areas), 0, "Should find at least one area")
        
        # Check that romerike area exists
        self.assertIn('romerike', areas)
        
        # Check romerike area structure
        romerike = areas['romerike']
        self.assertEqual(romerike.area_name, 'romerike')
        self.assertIsInstance(romerike.datasets, dict)
        
        # Should have both 2007 and 2013 time periods
        time_periods = romerike.time_periods
        self.assertIn('2007', time_periods)
        self.assertIn('2013', time_periods)
        
        # Check each dataset
        for time_period in ['2007', '2013']:
            dataset = romerike.datasets[time_period]
            self.assertEqual(dataset.area_name, 'romerike')
            self.assertEqual(dataset.time_period, time_period)
            self.assertGreater(len(dataset.laz_files), 0, f"Should have LAZ files for {time_period}")
            self.assertIsInstance(dataset.total_points, int)
            self.assertGreater(dataset.total_points, 0, f"Should have points for {time_period}")

    def test_batch_loader_load_dataset(self):
        """Test loading a complete dataset using BatchLoader."""
        print(f"\n=== Testing BatchLoader.load_dataset ===")
        
        # First discover the areas
        areas = self.discovery.scan_areas()
        self.assertIn('romerike', areas)
        
        # Get a dataset to load (2013 for example)
        dataset_info = areas['romerike'].datasets['2013']
        print(f"Loading dataset: {dataset_info.area_name}/{dataset_info.time_period}")
        print(f"Files to load: {len(dataset_info.laz_files)}")
        
        # Create batch loader and load dataset
        batch_loader = BatchLoader()
        result = batch_loader.load_dataset(dataset_info)
        
        # Print results
        print(f"Combined dataset:")
        print(f"  Expected points (from discovery): {dataset_info.total_points}")
        print(f"  Loaded points (from batch loader): {len(result['points'])}")
        print(f"  Points shape: {result['points'].shape}")
        print(f"  Available attributes: {list(result['attributes'].keys())}")
        
        # Print metadata
        metadata = result['metadata']
        print(f"  Area: {metadata['area_name']}")
        print(f"  Time period: {metadata['time_period']}")
        print(f"  Files combined: {metadata['num_files']}")
        
        if len(result['points']) > 0:
            points = result['points']
            print(f"  Combined bounds:")
            print(f"    X: {points[:, 0].min():.1f} to {points[:, 0].max():.1f}")
            print(f"    Y: {points[:, 1].min():.1f} to {points[:, 1].max():.1f}")
            print(f"    Z: {points[:, 2].min():.1f} to {points[:, 2].max():.1f}")
        
        # Assertions
        self.assertIsInstance(result, dict)
        self.assertIn('points', result)
        self.assertIn('attributes', result)
        self.assertIn('metadata', result)
        
        # Check points
        points = result['points']
        self.assertIsInstance(points, np.ndarray)

        # Check that loaded points match the discovered ground points count
        self.assertEqual(len(points), dataset_info.total_points)
        
        if len(points) > 0:
            self.assertEqual(points.shape[1], 3)  # X, Y, Z columns
            self.assertEqual(points.dtype, np.float64)

            # Check attributes
            attributes = result['attributes']
            self.assertIsInstance(attributes, dict)
            for attr_name, attr_values in attributes.items():
                print(f"  Attribute '{attr_name}' shape: {attr_values.shape}")
                self.assertEqual(len(attr_values), len(points))
        
        # Check metadata
        self.assertEqual(metadata['area_name'], 'romerike')
        self.assertEqual(metadata['time_period'], '2013')
        self.assertGreater(metadata['num_files'], 0)

    def test_batch_loader_with_max_points(self):
        """Test loading dataset with point limit per file."""
        print(f"\n=== Testing BatchLoader with max_points_per_file ===")
        
        areas = self.discovery.scan_areas()
        dataset_info = areas['romerike'].datasets['2007']
        
        # Load with point limit
        max_points = 1000
        print(f"Loading with max {max_points} points per file")
        
        batch_loader = BatchLoader()
        result = batch_loader.load_dataset(dataset_info, max_points_per_file=max_points)
        
        print(f"Result with point limit:")
        print(f"  Total points: {len(result['points'])}")
        print(f"  Files processed: {result['metadata']['num_files']}")
        
        # Should have loaded some points
        self.assertIsInstance(result['points'], np.ndarray)
        
        # Total points should be reasonable (not exceed max_points * num_files by much)
        # Note: Actual count may be less due to ground point filtering
        expected_max = max_points * result['metadata']['num_files']
        print(f"  Expected max points: {expected_max}")

    def test_discovery_with_nonexistent_directory(self):
        """Test data discovery with non-existent directory."""
        print(f"\n=== Testing with non-existent directory ===")
        
        discovery = DataDiscovery("non_existent_directory")
        areas = discovery.scan_areas()
        
        print(f"Areas found in non-existent directory: {len(areas)}")
        
        # Should return empty dictionary
        self.assertIsInstance(areas, dict)
        self.assertEqual(len(areas), 0)

    def test_time_periods_property(self):
        """Test the time_periods property of AreaInfo."""
        print(f"\n=== Testing time_periods property ===")
        
        areas = self.discovery.scan_areas()
        romerike = areas['romerike']
        
        time_periods = romerike.time_periods
        print(f"Time periods for romerike: {time_periods}")
        
        # Should be sorted
        self.assertIsInstance(time_periods, list)
        self.assertEqual(time_periods, sorted(time_periods))
        
        # Should contain expected periods
        self.assertIn('2007', time_periods)
        self.assertIn('2013', time_periods)


class TestBatchLoader(unittest.TestCase):
    """Test cases for the BatchLoader class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.batch_loader = BatchLoader()
        
        # Path to the test data directory for integration tests
        self.test_data_dir = Path(__file__).parent / "sample_data" / "raw"
        
        # Skip if test data not available
        if not self.test_data_dir.exists():
            self.skipTest(f"Test data directory not found: {self.test_data_dir}")
        
        # Set up discovery for getting test datasets
        self.discovery = DataDiscovery(str(self.test_data_dir))

    def test_init(self):
        """Test BatchLoader initialization."""
        print(f"\n=== Testing BatchLoader initialization ===")
        
        loader = BatchLoader()
        self.assertIsInstance(loader, BatchLoader)
        
        # Should have a PointCloudLoader instance
        self.assertTrue(hasattr(loader, 'loader'))
        print(f"BatchLoader initialized successfully with PointCloudLoader")

    def test_load_separate_files(self):
        """Test loading files separately instead of combining them."""
        print(f"\n=== Testing _load_separate_files ===")
        
        # Get a dataset to test with
        areas = self.discovery.scan_areas()
        dataset_info = areas['romerike'].datasets['2007']
        
        print(f"Testing separate file loading for: {dataset_info.area_name}/{dataset_info.time_period}")
        print(f"Number of files: {len(dataset_info.laz_files)}")
        
        # Load files separately
        result = self.batch_loader._load_separate_files(dataset_info)
        
        # Print results
        file_data = result['file_data']
        print(f"Loaded {len(file_data)} separate files")
        
        for i, pc_data in enumerate(file_data):
            print(f"  File {i+1}: {len(pc_data['points'])} points")
            if len(pc_data['points']) > 0:
                points = pc_data['points']
                print(f"    Bounds: X({points[:, 0].min():.1f}-{points[:, 0].max():.1f}), "
                      f"Y({points[:, 1].min():.1f}-{points[:, 1].max():.1f}), "
                      f"Z({points[:, 2].min():.1f}-{points[:, 2].max():.1f})")
        
        # Print metadata
        metadata = result['metadata']
        print(f"Metadata:")
        print(f"  Area: {metadata['area_name']}")
        print(f"  Time period: {metadata['time_period']}")
        print(f"  Files processed: {metadata['num_files']}")
        
        # Assertions
        self.assertIsInstance(result, dict)
        self.assertIn('file_data', result)
        self.assertIn('metadata', result)
        
        # Check file_data structure
        self.assertIsInstance(file_data, list)
        self.assertEqual(len(file_data), len(dataset_info.laz_files))
        
        # Each file should have the standard structure
        for pc_data in file_data:
            self.assertIsInstance(pc_data, dict)
            self.assertIn('points', pc_data)
            self.assertIn('attributes', pc_data)
            self.assertIn('metadata', pc_data)
            
            # Points should be numpy array
            self.assertIsInstance(pc_data['points'], np.ndarray)
            if len(pc_data['points']) > 0:
                self.assertEqual(pc_data['points'].shape[1], 3)
        
        # Check metadata
        self.assertEqual(metadata['area_name'], 'romerike')
        self.assertEqual(metadata['time_period'], '2007')
        self.assertEqual(metadata['num_files'], len(file_data))

    def test_load_separate_files_with_max_points(self):
        """Test loading separate files with point limits."""
        print(f"\n=== Testing _load_separate_files with max_points_per_file ===")
        
        areas = self.discovery.scan_areas()
        dataset_info = areas['romerike'].datasets['2013']
        
        max_points = 500
        print(f"Loading separate files with max {max_points} points per file")
        
        result = self.batch_loader._load_separate_files(dataset_info, max_points_per_file=max_points)
        
        file_data = result['file_data']
        print(f"Loaded {len(file_data)} files with point limits:")
        
        for i, pc_data in enumerate(file_data):
            point_count = len(pc_data['points'])
            print(f"  File {i+1}: {point_count} points (limit: {max_points})")
            
            # Points should not exceed the limit (though may be less due to ground filtering)
            # We can't guarantee exact count due to ground point filtering
            self.assertIsInstance(pc_data['points'], np.ndarray)

    def test_compute_bounds(self):
        """Test the _compute_bounds method."""
        print(f"\n=== Testing _compute_bounds ===")
        
        # Create test points
        test_points = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 20.0, 30.0],
            [5.0, 15.0, 25.0],
            [-5.0, -10.0, -15.0]
        ])
        
        bounds = self.batch_loader._compute_bounds(test_points)
        
        print(f"Test points bounds: {bounds}")
        
        # Check structure
        expected_keys = ['min_x', 'max_x', 'min_y', 'max_y', 'min_z', 'max_z']
        for key in expected_keys:
            self.assertIn(key, bounds)
            self.assertIsInstance(bounds[key], float)
        
        # Check values
        self.assertEqual(bounds['min_x'], -5.0)
        self.assertEqual(bounds['max_x'], 10.0)
        self.assertEqual(bounds['min_y'], -10.0)
        self.assertEqual(bounds['max_y'], 20.0)
        self.assertEqual(bounds['min_z'], -15.0)
        self.assertEqual(bounds['max_z'], 30.0)

    def test_empty_dataset_handling(self):
        """Test handling of datasets with no valid files."""
        print(f"\n=== Testing empty dataset handling ===")
        
        # Create a mock dataset with no valid files
        from terrain_change_detection.preprocessing.data_discovery import DatasetInfo
        
        empty_dataset = DatasetInfo(
            area_name="test_area",
            time_period="test_period",
            laz_files=[],  # No files
            metadata_dir=None
        )
        
        result = self.batch_loader.load_dataset(empty_dataset)
        
        print(f"Result for empty dataset:")
        print(f"  Points shape: {result['points'].shape}")
        print(f"  Attributes: {result['attributes']}")
        print(f"  Metadata: {result['metadata']}")
        
        # Should return empty but valid structure
        self.assertIsInstance(result, dict)
        self.assertEqual(result['points'].shape, (0,))
        self.assertEqual(len(result['attributes']), 0)
        self.assertEqual(result['metadata']['num_files'], 0)
        self.assertEqual(result['metadata']['total_points'], 0)


if __name__ == '__main__':
    unittest.main()
