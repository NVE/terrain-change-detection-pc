"""
Test suite for point cloud preprocessing module
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Import the loader module
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
from terrain_change_detection.preprocessing.loader import PointCloudLoader


class TestPointCloudLoader(unittest.TestCase):
    """Test cases for the PointCloudLoader class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.loader = PointCloudLoader()
        
        # Path to the test LAZ file
        self.test_data_dir = Path(__file__).parent / "sample_data" / "single_laz_file"
        self.test_laz_file = self.test_data_dir / "33-1-466-136-14.laz"
        
        # Ensure the test file exists
        if not self.test_laz_file.exists():
            self.skipTest(f"Test LAZ file not found: {self.test_laz_file}")

    def test_load_valid_file(self):
        """Test loading a valid LAZ file."""
        print(f"\n=== Testing load_valid_file ===")
        print(f"Loading file: {self.test_laz_file}")
        
        result = self.loader.load(str(self.test_laz_file))
        
        # Check that result has expected structure
        self.assertIsInstance(result, dict)
        self.assertIn('points', result)
        self.assertIn('attributes', result)
        self.assertIn('metadata', result)
        
        # Check points array (ground points only)
        points = result['points']
        print(f"Loaded {len(points)} ground points")
        print(f"Points shape: {points.shape}")
        print(f"Points data type: {points.dtype}")
        
        # Show some basic statistics
        if len(points) > 0:
            print(f"X range: {points[:, 0].min():.2f} to {points[:, 0].max():.2f}")
            print(f"Y range: {points[:, 1].min():.2f} to {points[:, 1].max():.2f}")
            print(f"Z range: {points[:, 2].min():.2f} to {points[:, 2].max():.2f}")
        
        # Show available attributes
        attributes = result['attributes']
        print(f"Available attributes: {list(attributes.keys())}")
        for attr_name, attr_data in attributes.items():
            if hasattr(attr_data, 'shape'):
                print(f"  {attr_name}: shape {attr_data.shape}")
            else:
                print(f"  {attr_name}: length {len(attr_data)}")
        
        self.assertIsInstance(points, np.ndarray)
        self.assertEqual(points.shape[1], 3)  # X, Y, Z columns
        self.assertEqual(points.dtype, np.float64)

    def test_validate_file(self):
        """Test file validation."""
        print(f"\n=== Testing validate_file ===")
        
        # Valid file
        print(f"Validating valid file: {self.test_laz_file.name}")
        result_valid = self.loader.validate_file(str(self.test_laz_file))
        print(f"Validation result: {result_valid}")
        self.assertTrue(result_valid)
        
        # Non-existent file
        print(f"Validating non-existent file: non_existent_file.laz")
        result_invalid = self.loader.validate_file("non_existent_file.laz")
        print(f"Validation result: {result_invalid}")
        self.assertFalse(result_invalid)

    def test_get_metadata(self):
        """Test metadata extraction."""
        print(f"\n=== Testing get_metadata ===")
        metadata = self.loader.get_metadata(str(self.test_laz_file))
        
        # Print key metadata information
        print(f"Filename: {metadata['filename']}")
        print(f"File size: {metadata['file_size_mb']:.2f} MB")
        print(f"Total points in file: {metadata['num_points']}")
        print(f"Point format: {metadata['point_format']}")
        print(f"Version: {metadata['version']}")
        
        # Print bounds
        bounds = metadata['bounds']
        print(f"Bounds:")
        print(f"  X: {bounds['min_x']:.2f} to {bounds['max_x']:.2f}")
        print(f"  Y: {bounds['min_y']:.2f} to {bounds['max_y']:.2f}")
        print(f"  Z: {bounds['min_z']:.2f} to {bounds['max_z']:.2f}")
        
        # Print classification stats
        classification_stats = metadata['classification_stats']
        print(f"Classification statistics:")
        print(f"  Unique classes: {classification_stats['unique_classes']}")
        print(f"  Class counts: {classification_stats['class_counts']}")
        print(f"  Ground points: {classification_stats['ground_points']}")
        print(f"  Ground percentage: {classification_stats['ground_percentage']:.1f}%")
        
        # Print available dimensions
        print(f"Available dimensions: {metadata['available_dimensions']}")
        
        # Print statistics
        stats = metadata['statistics']
        print(f"Statistics (for {stats['points_used_for_stats']}):")
        print(f"  Centroid: ({stats['centroid'][0]:.2f}, {stats['centroid'][1]:.2f}, {stats['centroid'][2]:.2f})")
        print(f"  Mean Z: {stats['mean_z']:.2f}")
        print(f"  Std Z: {stats['std_z']:.2f}")
        
        # Check required metadata fields
        self.assertIsInstance(metadata, dict)
        self.assertIn('filename', metadata)
        self.assertIn('num_points', metadata)
        self.assertIn('bounds', metadata)
        self.assertIn('classification_stats', metadata)
        
        # Check classification stats for ground points
        self.assertIn('ground_points', classification_stats)
        self.assertIn('ground_percentage', classification_stats)

    def test_load_file_not_found(self):
        """Test loading a non-existent file raises error."""
        with self.assertRaises(FileNotFoundError):
            self.loader.load("non_existent_file.laz")


if __name__ == '__main__':
    unittest.main()