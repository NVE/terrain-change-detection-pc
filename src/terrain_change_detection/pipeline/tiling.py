"""
Out-of-core streaming primitives for gridded computations (prototype).

Provides:
- GridAccumulator: streaming mean aggregator on a fixed XY grid
- LaspyStreamReader: chunked LAS/LAZ reader with bbox + classification filter

Note: This is an initial CPU-only implementation targeting 'mean' DoD.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

import numpy as np


@dataclass
class Bounds2D:
    min_x: float
    min_y: float
    max_x: float
    max_y: float


class GridAccumulator:
    """Streaming mean aggregator over a regular XY grid.

    Keeps running sums and counts per cell for later mean computation.
    """

    def __init__(self, bounds: Bounds2D, cell_size: float):
        self.bounds = bounds
        self.cell = float(cell_size)
        # Grid sizes
        self.nx = int(np.ceil((bounds.max_x - bounds.min_x) / self.cell)) + 1
        self.ny = int(np.ceil((bounds.max_y - bounds.min_y) / self.cell)) + 1
        # Accumulators
        self.sum = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.cnt = np.zeros((self.ny, self.nx), dtype=np.int64)

        # Precompute grid centers
        x_edges = np.linspace(bounds.min_x, bounds.min_x + self.nx * self.cell, self.nx + 1)
        y_edges = np.linspace(bounds.min_y, bounds.min_y + self.ny * self.cell, self.ny + 1)
        self.grid_x, self.grid_y = np.meshgrid(
            (x_edges[:-1] + x_edges[1:]) / 2.0,
            (y_edges[:-1] + y_edges[1:]) / 2.0,
        )

    def accumulate(self, points: np.ndarray) -> None:
        """Accumulate a chunk of points (N x 3)."""
        if points.size == 0:
            return
        # Compute fractional indices to filter strictly to tile bounds
        fx = (points[:, 0] - self.bounds.min_x) / self.cell
        fy = (points[:, 1] - self.bounds.min_y) / self.cell
        mask = (fx >= 0.0) & (fy >= 0.0) & (fx < self.nx) & (fy < self.ny)
        if not np.any(mask):
            return
        xi = fx[mask].astype(int)
        yi = fy[mask].astype(int)

        # Aggregate sums and counts per unique cell (vectorized)
        lin = yi.astype(np.int64) * np.int64(self.nx) + xi.astype(np.int64)
        # unique linear indices
        uniq, idx, inv, counts = np.unique(lin, return_index=True, return_inverse=True, return_counts=True)
        # sum z per unique
        z = points[mask, 2]
        sums = np.zeros_like(uniq, dtype=np.float64)
        np.add.at(sums, inv, z)
        # Scatter to grid arrays
        ux = uniq % self.nx
        uy = uniq // self.nx
        self.sum[uy, ux] += sums
        self.cnt[uy, ux] += counts

    def finalize(self) -> np.ndarray:
        """Return mean grid with NaNs for empty cells."""
        dem = np.full_like(self.sum, np.nan, dtype=np.float64)
        mask = self.cnt > 0
        dem[mask] = self.sum[mask] / self.cnt[mask]
        return dem


@dataclass
class Tile:
    i: int
    j: int
    inner: Bounds2D
    outer: Bounds2D
    x0_idx: int  # starting column in global grid
    y0_idx: int  # starting row in global grid
    nx: int      # columns in tile grid
    ny: int      # rows in tile grid


class Tiler:
    """Generate grid-aligned tiles with optional halo."""

    def __init__(self, global_bounds: Bounds2D, cell_size: float, tile_size: float, halo: float) -> None:
        self.gb = global_bounds
        self.cell = float(cell_size)
        self.tile = float(tile_size)
        self.halo = float(halo)
        # Global grid origin (centers start at min + cell/2)
        self.nx = int(np.ceil((self.gb.max_x - self.gb.min_x) / self.cell)) + 1
        self.ny = int(np.ceil((self.gb.max_y - self.gb.min_y) / self.cell)) + 1

    def tiles(self) -> Iterator[Tile]:
        # Number of tiles in each direction (ceil)
        tx = int(np.ceil((self.gb.max_x - self.gb.min_x) / self.tile))
        ty = int(np.ceil((self.gb.max_y - self.gb.min_y) / self.tile))
        for j in range(ty):
            y0 = self.gb.min_y + j * self.tile
            y1 = min(self.gb.max_y, y0 + self.tile)
            for i in range(tx):
                x0 = self.gb.min_x + i * self.tile
                x1 = min(self.gb.max_x, x0 + self.tile)
                inner = Bounds2D(min_x=x0, min_y=y0, max_x=x1, max_y=y1)
                # Outer bounds with halo, clipped to global
                outer = Bounds2D(
                    min_x=max(self.gb.min_x, x0 - self.halo),
                    min_y=max(self.gb.min_y, y0 - self.halo),
                    max_x=min(self.gb.max_x, x1 + self.halo),
                    max_y=min(self.gb.max_y, y1 + self.halo),
                )
                # Tile grid size and starting indices (aligned to global origin)
                nx = int(np.ceil((inner.max_x - inner.min_x) / self.cell)) + 1
                ny = int(np.ceil((inner.max_y - inner.min_y) / self.cell)) + 1
                x0_idx = int(round((inner.min_x - self.gb.min_x) / self.cell))
                y0_idx = int(round((inner.min_y - self.gb.min_y) / self.cell))
                yield Tile(i=i, j=j, inner=inner, outer=outer, x0_idx=x0_idx, y0_idx=y0_idx, nx=nx, ny=ny)


class MosaicAccumulator:
    """Assemble tile DEMs into a global grid by averaging overlaps."""

    def __init__(self, global_bounds: Bounds2D, cell_size: float) -> None:
        self.gb = global_bounds
        self.cell = float(cell_size)
        self.nx = int(np.ceil((self.gb.max_x - self.gb.min_x) / self.cell)) + 1
        self.ny = int(np.ceil((self.gb.max_y - self.gb.min_y) / self.cell)) + 1
        self.sum = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.cnt = np.zeros((self.ny, self.nx), dtype=np.int64)
        # Precompute centers for output
        x_edges = np.linspace(self.gb.min_x, self.gb.min_x + self.nx * self.cell, self.nx + 1)
        y_edges = np.linspace(self.gb.min_y, self.gb.min_y + self.ny * self.cell, self.ny + 1)
        self.grid_x, self.grid_y = np.meshgrid(
            (x_edges[:-1] + x_edges[1:]) / 2.0,
            (y_edges[:-1] + y_edges[1:]) / 2.0,
        )

    def add_tile(self, tile: Tile, dem_tile: np.ndarray) -> None:
        h, w = dem_tile.shape
        assert h == tile.ny and w == tile.nx
        y_slice = slice(tile.y0_idx, tile.y0_idx + tile.ny)
        x_slice = slice(tile.x0_idx, tile.x0_idx + tile.nx)
        valid = np.isfinite(dem_tile)
        # Accumulate sums and counts only where valid
        self.sum[y_slice, x_slice][valid] += dem_tile[valid]
        self.cnt[y_slice, x_slice][valid] += 1

    def finalize(self) -> np.ndarray:
        out = np.full((self.ny, self.nx), np.nan, dtype=np.float64)
        mask = self.cnt > 0
        out[mask] = self.sum[mask] / self.cnt[mask]
        return out


class LaspyStreamReader:
    """Chunked LAS/LAZ iterator with bbox and classification filtering."""

    def __init__(
        self,
        files: Iterable[str | Path],
        *,
        ground_only: bool = True,
        classification_filter: Optional[List[int]] = None,
        chunk_points: int = 1_000_000,
    ) -> None:
        self.files = [str(Path(f)) for f in files]
        self.ground_only = ground_only
        self.classification_filter = classification_filter
        self.chunk_points = int(chunk_points)

    def _mask_classes(self, las) -> np.ndarray:
        n = len(las.points)
        if hasattr(las, "classification"):
            classes = np.asarray(las.classification)
            if self.ground_only and self.classification_filter is not None:
                return np.isin(classes, np.array(self.classification_filter))
            elif self.ground_only:
                return classes == 2
            elif self.classification_filter is not None:
                return np.isin(classes, np.array(self.classification_filter))
        return np.ones(n, dtype=bool)

    def stream_points(self, bbox: Optional[Bounds2D] = None) -> Iterator[np.ndarray]:
        import laspy

        for fp in self.files:
            with laspy.open(fp) as reader:
                for chunk in reader.chunk_iterator(self.chunk_points):
                    mask = self._mask_classes(chunk)
                    x = np.asarray(chunk.x, dtype=np.float64)
                    y = np.asarray(chunk.y, dtype=np.float64)
                    z = np.asarray(chunk.z, dtype=np.float64)
                    if bbox is not None:
                        mask &= (
                            (x >= bbox.min_x)
                            & (x <= bbox.max_x)
                            & (y >= bbox.min_y)
                            & (y <= bbox.max_y)
                        )
                    if not np.any(mask):
                        continue
                    pts = np.column_stack([x[mask], y[mask], z[mask]])
                    yield pts


def union_bounds(files_a: Iterable[str | Path], files_b: Iterable[str | Path]) -> Bounds2D:
    import laspy

    def scan(files: Iterable[str | Path]) -> Tuple[float, float, float, float]:
        min_x = min_y = float("inf")
        max_x = max_y = float("-inf")
        for fp in files:
            with laspy.open(str(fp)) as r:
                h = r.header
                min_x = min(min_x, float(h.x_min))
                min_y = min(min_y, float(h.y_min))
                max_x = max(max_x, float(h.x_max))
                max_y = max(max_y, float(h.y_max))
        return min_x, min_y, max_x, max_y

    ax0, ay0, ax1, ay1 = scan(files_a)
    bx0, by0, bx1, by1 = scan(files_b)
    return Bounds2D(
        min_x=min(ax0, bx0),
        min_y=min(ay0, by0),
        max_x=max(ax1, bx1),
        max_y=max(ay1, by1),
    )
