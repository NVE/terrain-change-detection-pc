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
        result = self.loader.load(str(self.test_laz_file))
        
        # Check that result has expected structure
        self.assertIsInstance(result, dict)
        self.assertIn('points', result)
        self.assertIn('attributes', result)
        self.assertIn('metadata', result)
        
        # Check points array (ground points only)
        points = result['points']
        self.assertIsInstance(points, np.ndarray)
        self.assertEqual(points.shape[1], 3)  # X, Y, Z columns
        self.assertEqual(points.dtype, np.float64)

    def test_validate_file(self):
        """Test file validation."""
        # Valid file
        self.assertTrue(self.loader.validate_file(str(self.test_laz_file)))
        
        # Non-existent file
        self.assertFalse(self.loader.validate_file("non_existent_file.laz"))

    def test_get_metadata(self):
        """Test metadata extraction."""
        metadata = self.loader.get_metadata(str(self.test_laz_file))
        
        # Check required metadata fields
        self.assertIsInstance(metadata, dict)
        self.assertIn('filename', metadata)
        self.assertIn('num_points', metadata)
        self.assertIn('bounds', metadata)
        self.assertIn('classification_stats', metadata)
        
        # Check classification stats for ground points
        classification_stats = metadata['classification_stats']
        self.assertIn('ground_points', classification_stats)
        self.assertIn('ground_percentage', classification_stats)

    def test_load_file_not_found(self):
        """Test loading a non-existent file raises error."""
        with self.assertRaises(FileNotFoundError):
            self.loader.load("non_existent_file.laz")


if __name__ == '__main__':
    unittest.main()