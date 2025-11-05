import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from terrain_change_detection.pipeline.tiling import GridAccumulator, Bounds2D
from terrain_change_detection.detection import ChangeDetector


def test_grid_accumulator_mean_parity_small():
    rng = np.random.default_rng(0)
    # Create two clouds over 0..10 x 0..10
    n = 20000
    x = rng.uniform(0, 10, n)
    y = rng.uniform(0, 10, n)
    z1 = 0.1 * x + 0.2 * y
    z2 = z1 + 1.0
    pts1 = np.column_stack([x, y, z1])
    pts2 = np.column_stack([x, y, z2])

    b = Bounds2D(0.0, 0.0, 10.0, 10.0)
    cell = 1.0
    acc1 = GridAccumulator(b, cell)
    acc2 = GridAccumulator(b, cell)
    acc1.accumulate(pts1)
    acc2.accumulate(pts2)
    dem1 = acc1.finalize()
    dem2 = acc2.finalize()
    dod = dem2 - dem1

    # Reference via in-memory DoD
    res = ChangeDetector.compute_dod(pts1, pts2, cell_size=cell, aggregator="mean")
    # Shapes equal and mean close to +1
    assert dod.shape == res.dod.shape
    m = float(np.nanmean(dod))
    assert 0.98 < m < 1.02

