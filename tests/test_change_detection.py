"""
Unit tests for change detection interfaces.

Covers DoD and C2C basic behavior.
"""

import numpy as np
from terrain_change_detection.detection import ChangeDetector


def test_dod_basic_shapes():
    # Create two small synthetic point sets
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 1000)
    y = rng.uniform(0, 10, 1000)
    z1 = 0.1 * x + 0.2 * y
    z2 = z1 + 1.0  # uniform uplift of +1 m
    pts1 = np.column_stack([x, y, z1])
    pts2 = np.column_stack([x, y, z2])

    res = ChangeDetector.compute_dod(pts1, pts2, cell_size=1.0, aggregator="mean")

    assert res.dod.shape == res.dem1.shape == res.dem2.shape
    assert res.grid_x.shape == res.grid_y.shape == res.dod.shape
    # Mean change should be close to +1 where valid
    assert np.isfinite(res.dod).any()
    mean_change = np.nanmean(res.dod)
    assert 0.9 < mean_change < 1.1


def test_c2c_identical_clouds_zero_distance():
    # Two identical point clouds -> nearest neighbor distance is zero
    rng = np.random.default_rng(1)
    source = rng.normal(size=(500, 3))
    target = source.copy()

    res = ChangeDetector.compute_c2c(source, target)

    # Distances should be exactly zero in this case
    assert res.n == len(source)
    assert float(res.mean) == 0.0
    assert float(res.median) == 0.0
    assert float(res.rmse) == 0.0
