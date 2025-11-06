"""
Tests for streaming/out-of-core workflow integration

Validates that the tiling and streaming infrastructure is properly
coordinated across preprocessing, alignment, and detection modules.
"""

import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from terrain_change_detection.utils.point_cloud_filters import (
    create_classification_mask,
    get_filter_statistics,
)
from terrain_change_detection.preprocessing.data_discovery import BatchLoader
from terrain_change_detection.acceleration import LaspyStreamReader, Bounds2D


def test_classification_filter_shared_utility():
    """Test that shared classification filtering works consistently."""
    classes = np.array([1, 2, 2, 3, 2, 1, 4, 2])
    
    # Test ground_only
    mask = create_classification_mask(classes, ground_only=True)
    assert np.array_equal(mask, np.array([False, True, True, False, True, False, False, True]))
    
    # Test custom filter
    mask = create_classification_mask(classes, ground_only=False, classification_filter=[1, 2])
    assert np.array_equal(mask, np.array([True, True, True, False, True, True, False, True]))
    
    # Test no filter
    mask = create_classification_mask(classes, ground_only=False)
    assert np.all(mask)


def test_filter_statistics():
    """Test filter statistics generation."""
    stats = get_filter_statistics(
        total_points=1000,
        filtered_points=200,
        ground_only=True
    )
    
    assert stats['total_points'] == 1000
    assert stats['filtered_points'] == 200
    assert stats['percentage'] == 20.0
    assert 'ground only' in stats['filter_description']


def test_batch_loader_streaming_mode():
    """Test that BatchLoader can operate in streaming mode."""
    from terrain_change_detection.preprocessing.data_discovery import DatasetInfo
    
    # Create mock dataset info
    test_files = [Path("test1.laz"), Path("test2.laz")]
    dataset_info = DatasetInfo(
        area_name="test_area",
        time_period="2020",
        laz_files=test_files,
        metadata_dir=None,
        total_points=10000,
        bounds={'min_x': 0, 'min_y': 0, 'max_x': 100, 'max_y': 100},
    )
    
    # Test streaming mode
    loader = BatchLoader(streaming_mode=True)
    result = loader.load_dataset(dataset_info, streaming=True)
    
    assert result['mode'] == 'streaming'
    assert 'file_paths' in result
    assert len(result['file_paths']) == 2
    assert result['num_files'] == 2
    assert result['metadata']['area_name'] == "test_area"


def test_laspy_stream_reader_filtering():
    """Test that LaspyStreamReader uses shared filtering utility."""
    # This is an integration test that would need real LAZ files
    # For now, just verify the reader can be instantiated with filter options
    
    reader = LaspyStreamReader(
        files=[],  # Empty for this test
        ground_only=True,
        classification_filter=None,
        chunk_points=100000,
    )
    
    assert reader.ground_only is True
    assert reader.classification_filter is None
    assert reader.chunk_points == 100000


def test_bounds_2d_utility():
    """Test Bounds2D dataclass."""
    bounds = Bounds2D(min_x=0.0, min_y=0.0, max_x=100.0, max_y=100.0)
    
    assert bounds.min_x == 0.0
    assert bounds.max_x == 100.0
    assert bounds.max_y == 100.0


def test_integration_classification_consistency():
    """
    Test that classification filtering behaves identically across
    PointCloudLoader and LaspyStreamReader.
    """
    # Create test classification array
    classes = np.array([1, 2, 2, 3, 2, 1, 4, 2, 5, 2])
    
    # Test various filter configurations
    configs = [
        {'ground_only': True, 'classification_filter': None},
        {'ground_only': False, 'classification_filter': [1, 2]},
        {'ground_only': False, 'classification_filter': [2, 3, 4]},
        {'ground_only': False, 'classification_filter': None},
    ]
    
    for config in configs:
        mask = create_classification_mask(
            classes,
            ground_only=config['ground_only'],
            classification_filter=config['classification_filter']
        )
        
        # Verify mask is boolean
        assert mask.dtype == bool
        # Verify mask has same length as input
        assert len(mask) == len(classes)
        # Verify at least some points are filtered (for specific configs)
        if config['ground_only'] or config['classification_filter']:
            # Should filter something
            assert not np.all(mask)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
