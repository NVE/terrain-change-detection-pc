"""Tests for shared point cloud filtering utilities."""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from terrain_change_detection.utils.point_cloud_filters import (
    create_classification_mask,
    apply_classification_filter,
    get_filter_statistics,
)


def test_create_classification_mask_ground_only():
    """Test ground-only filtering (class 2)."""
    classes = np.array([1, 2, 2, 3, 2, 1, 5, 2])
    mask = create_classification_mask(classes, ground_only=True)
    
    expected = np.array([False, True, True, False, True, False, False, True])
    assert np.array_equal(mask, expected)
    assert mask.sum() == 4  # 4 ground points


def test_create_classification_mask_custom_filter():
    """Test custom classification filter."""
    classes = np.array([1, 2, 2, 3, 2, 1, 5, 2])
    mask = create_classification_mask(classes, classification_filter=[1, 2, 3])
    
    # Should include classes 1, 2, and 3 (not 5)
    expected = np.array([True, True, True, True, True, True, False, True])
    assert np.array_equal(mask, expected)
    assert mask.sum() == 7


def test_create_classification_mask_no_filter():
    """Test no filtering - accept all points."""
    classes = np.array([1, 2, 2, 3, 2, 1, 5, 2])
    mask = create_classification_mask(classes, ground_only=False, classification_filter=None)
    
    # Should accept all points
    assert mask.all()
    assert mask.sum() == len(classes)


def test_apply_classification_filter():
    """Test applying filter to points and classifications."""
    points = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ])
    classes = np.array([1, 2, 2, 3])
    
    filtered_pts, filtered_classes = apply_classification_filter(
        points, classes, ground_only=True
    )
    
    # Should only keep rows with class 2
    expected_pts = np.array([
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    expected_classes = np.array([2, 2])
    
    assert np.array_equal(filtered_pts, expected_pts)
    assert np.array_equal(filtered_classes, expected_classes)


def test_get_filter_statistics_ground_only():
    """Test filter statistics for ground-only filtering."""
    stats = get_filter_statistics(
        total_points=1000,
        filtered_points=250,
        ground_only=True,
        classification_filter=None
    )
    
    assert stats['total_points'] == 1000
    assert stats['filtered_points'] == 250
    assert stats['percentage'] == 25.0
    assert 'ground only' in stats['filter_description']
    assert stats['ground_only'] == True


def test_get_filter_statistics_custom_filter():
    """Test filter statistics for custom classification filter."""
    stats = get_filter_statistics(
        total_points=1000,
        filtered_points=600,
        ground_only=False,
        classification_filter=[2, 9]
    )
    
    assert stats['percentage'] == 60.0
    assert '[2, 9]' in stats['filter_description']
    assert stats['classification_filter'] == [2, 9]


def test_classification_filter_overrides_ground_only():
    """Test that classification_filter takes precedence over ground_only."""
    classes = np.array([1, 2, 2, 3, 2, 1, 5, 2])
    
    # Both ground_only=True and classification_filter set
    # classification_filter should take precedence
    mask = create_classification_mask(
        classes, 
        ground_only=True,  # This should be ignored
        classification_filter=[1, 5]
    )
    
    # Should only get classes 1 and 5
    expected = np.array([True, False, False, False, False, True, True, False])
    assert np.array_equal(mask, expected)
    assert mask.sum() == 3
