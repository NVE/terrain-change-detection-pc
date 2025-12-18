"""
Out-of-core streaming primitives for gridded computations (prototype).

Provides:
- GridAccumulator: streaming mean aggregator on a fixed XY grid
- LaspyStreamReader: chunked LAS/LAZ reader with bbox + classification filter

Note: This is an initial CPU-only implementation targeting 'mean' DoD.

Applicability:
- DoD (DEM of Difference): Uses GridAccumulator to build per-tile DEMs and MosaicAccumulator to blend overlaps.
- C2C (point-based): Reuses Tiler/Tile (inner+halo) and LaspyStreamReader for radius-bounded nearest-neighbor queries; does not use DEM gridding/mosaicking.
- M3C2 (point-based): Reuses Tiler/Tile (inner+halo) and LaspyStreamReader for out-of-core neighborhoods; does not use DEM gridding/mosaicking.

See docs/ALGORITHMS.md for a full mapping of DoD, C2C, and M3C2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple, Dict, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..utils.coordinate_transform import LocalCoordinateTransform

from ..utils.point_cloud_filters import create_classification_mask
from .gpu_array_ops import ArrayBackend, get_array_backend, ensure_cpu_array, is_gpu_array

# Simple in-process cache for LAS/LAZ header bounds to avoid rescanning
_BOUNDS_CACHE: Dict[str, Bounds2D] = {}


@dataclass
class Bounds2D:
    """2D bounding box defined by min/max coordinates.
    
    Attributes:
        min_x: Minimum X coordinate
        min_y: Minimum Y coordinate
        max_x: Maximum X coordinate
        max_y: Maximum Y coordinate
    """
    min_x: float
    min_y: float
    max_x: float
    max_y: float


class GridAccumulator:
    """Streaming mean aggregator over a regular XY grid.

    Keeps running sums and counts per cell for later mean computation.
    This enables memory-efficient processing of large point clouds by
    accumulating statistics incrementally without loading all data at once.

    The grid is aligned to the provided bounds with cells of a fixed size.
    Grid cells are indexed by (row, col) in the internal arrays.

    Attributes:
        bounds: 2D bounding box defining the grid extent
        cell: Cell size (spacing) in data units
        nx: Number of grid columns
        ny: Number of grid rows
        sum: Running sum of Z values per cell (ny x nx)
        cnt: Count of points per cell (ny x nx)
        grid_x: X coordinates of cell centers (ny x nx)
        grid_y: Y coordinates of cell centers (ny x nx)
        use_gpu: Whether this accumulator attempts to use GPU-backed arrays
    """

    def __init__(self, bounds: Bounds2D, cell_size: float, use_gpu: bool = False):
        """Initialize the grid accumulator.

        Args:
            bounds: 2D bounding box for the grid
            cell_size: Size of each grid cell in data units
            use_gpu: If True, use GPU-backed arrays when available
        """
        self.bounds = bounds
        self.cell = float(cell_size)
        self.use_gpu = use_gpu

        # Grid sizes
        self.nx = int(np.ceil((bounds.max_x - bounds.min_x) / self.cell)) + 1
        self.ny = int(np.ceil((bounds.max_y - bounds.min_y) / self.cell)) + 1

        # Backend for accumulators
        if use_gpu:
            try:
                backend = get_array_backend(use_gpu=True)
                xp = backend.xp
                self._backend: ArrayBackend | None = backend
            except Exception:
                # GPU initialization failed, fall back to CPU
                xp = np
                self._backend = None
                self.use_gpu = False
        else:
            xp = np
            self._backend = None

        # Accumulators (GPU or CPU arrays)
        self.sum = xp.zeros((self.ny, self.nx), dtype=xp.float64)
        self.cnt = xp.zeros((self.ny, self.nx), dtype=xp.int64)

        # Precompute grid centers (always on CPU, small arrays)
        x_edges = np.linspace(bounds.min_x, bounds.min_x + self.nx * self.cell, self.nx + 1)
        y_edges = np.linspace(bounds.min_y, bounds.min_y + self.ny * self.cell, self.ny + 1)
        self.grid_x, self.grid_y = np.meshgrid(
            (x_edges[:-1] + x_edges[1:]) / 2.0,
            (y_edges[:-1] + y_edges[1:]) / 2.0,
        )

    def _accumulate_cpu(self, row: np.ndarray, col: np.ndarray, z: np.ndarray) -> None:
        """CPU accumulation using bincount on linearized indices."""
        if row.size == 0:
            return

        lin = row.astype(np.int64) * np.int64(self.nx) + col.astype(np.int64)
        uniq, inv, counts = np.unique(lin, return_inverse=True, return_counts=True)
        sums = np.zeros_like(uniq, dtype=np.float64)
        np.add.at(sums, inv, ensure_cpu_array(z).astype(np.float64))
        ux = uniq % self.nx
        uy = uniq // self.nx
        self.sum[uy, ux] += sums
        self.cnt[uy, ux] += counts

    def _accumulate_gpu(self, row, col, z) -> None:
        """GPU accumulation using CuPy; falls back to CPU if GPU arrays unavailable."""
        # This method assumes self.sum/self.cnt live on the same backend (NumPy or CuPy).
        if not self.use_gpu or self._backend is None or not self._backend.is_gpu:
            # Treat as CPU arrays
            self._accumulate_cpu(ensure_cpu_array(row), ensure_cpu_array(col), ensure_cpu_array(z))
            return

        try:
            xp = self._backend.xp
            row = xp.asarray(row, dtype=xp.int64)
            col = xp.asarray(col, dtype=xp.int64)
            z = xp.asarray(z, dtype=xp.float64)

            lin = row * xp.int64(self.nx) + col
            uniq, inv, counts = xp.unique(lin, return_inverse=True, return_counts=True)
            sums = xp.zeros_like(uniq, dtype=xp.float64)
            xp.add.at(sums, inv, z)
            ux = uniq % self.nx
            uy = uniq // self.nx
            self.sum[uy, ux] += sums
            self.cnt[uy, ux] += counts
        except Exception:
            # GPU operation failed, fall back to CPU
            self._accumulate_cpu(ensure_cpu_array(row), ensure_cpu_array(col), ensure_cpu_array(z))

    def accumulate(self, points: np.ndarray) -> None:
        """Accumulate a chunk of points into the grid.

        Points are assigned to grid cells based on their XY coordinates.
        The Z values are summed per cell for later averaging. Points
        outside the grid bounds are automatically filtered out.

        Args:
            points: Nx3 array of point coordinates [X, Y, Z]
        """
        if points.size == 0:
            return

        # Compute fractional indices to filter strictly to tile bounds
        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]

        fx = (xs - self.bounds.min_x) / self.cell
        fy = (ys - self.bounds.min_y) / self.cell

        mask = (fx >= 0.0) & (fy >= 0.0) & (fx < self.nx) & (fy < self.ny)
        if not np.any(mask):
            return

        col = fx[mask].astype(np.int64)
        row = fy[mask].astype(np.int64)
        z = zs[mask]

        if self.use_gpu and self._backend is not None and self._backend.is_gpu:
            self._accumulate_gpu(row, col, z)
        else:
            self._accumulate_cpu(row, col, z)

    def finalize(self) -> np.ndarray:
        """Compute the final mean grid from accumulated statistics.

        Cells with no points are assigned NaN values.

        Returns:
            2D array (ny x nx) containing mean Z values per cell.
            Empty cells contain np.nan.
        """
        if self.use_gpu and self._backend is not None and self._backend.is_gpu and is_gpu_array(self.sum):
            xp = self._backend.xp
            dem = xp.where(self.cnt > 0, self.sum / self.cnt, xp.nan)
            dem_cpu = ensure_cpu_array(dem).astype(np.float64)
            return dem_cpu

        dem = np.full_like(ensure_cpu_array(self.sum), np.nan, dtype=np.float64)
        cnt_cpu = ensure_cpu_array(self.cnt)
        mask = cnt_cpu > 0
        dem[mask] = ensure_cpu_array(self.sum)[mask] / cnt_cpu[mask]
        return dem


@dataclass
class Tile:
    """Represents a single tile in a tiled processing scheme.
    
    Each tile has:
    - 'inner' bounds: the actual tile area to be processed
    - 'outer' bounds: inner + halo region for edge handling
    
    Grid alignment ensures tiles fit perfectly into a global grid
    with no gaps or misalignments at tile boundaries.
    
    Attributes:
        i: Tile column index in the tile grid
        j: Tile row index in the tile grid
        inner: Bounding box of the tile's core processing area
        outer: Bounding box including halo region for edge handling
        x0_idx: Starting column index in the global grid (0-based)
        y0_idx: Starting row index in the global grid (0-based)
        nx: Number of columns in this tile's grid
        ny: Number of rows in this tile's grid
    """
    i: int
    j: int
    inner: Bounds2D
    outer: Bounds2D
    x0_idx: int  # starting column in global grid
    y0_idx: int  # starting row in global grid
    nx: int      # columns in tile grid
    ny: int      # rows in tile grid


class Tiler:
    """Generate grid-aligned tiles with optional halo for out-of-core processing.
    
    Divides a global spatial extent into smaller tiles that can be processed
    independently. Each tile includes:
    - An 'inner' region: the actual area to compute results for
    - An 'outer' region: inner + halo buffer to handle edge effects
    
    The halo allows algorithms to access neighboring data without artifacts
    at tile boundaries. Tiles are aligned to a global grid to ensure seamless
    mosaicking of results.
    
    Attributes:
        gb: Global bounding box covering the entire processing area
        cell: Grid cell size in data units
        tile: Tile size (width/height) in data units
        halo: Halo/buffer width around each tile in data units
        nx: Total columns in the global grid
        ny: Total rows in the global grid
    """

    def __init__(self, global_bounds: Bounds2D, cell_size: float, tile_size: float, halo: float) -> None:
        """Initialize the tiler.
        
        Args:
            global_bounds: Bounding box covering the entire processing area
            cell_size: Size of grid cells in data units
            tile_size: Target size of tiles (inner bounds) in data units
            halo: Buffer width around tiles for edge handling in data units
        """
        self.gb = global_bounds
        self.cell = float(cell_size)
        self.tile = float(tile_size)
        self.halo = float(halo)
        # Compute global grid dimensions
        self.nx = int(np.ceil((self.gb.max_x - self.gb.min_x) / self.cell)) + 1
        self.ny = int(np.ceil((self.gb.max_y - self.gb.min_y) / self.cell)) + 1

    def tiles(self) -> Iterator[Tile]:
        """Generate all tiles covering the global bounds.
        
        Tiles are generated row-by-row (y first, then x) to enable
        sequential processing. Each tile is grid-aligned and includes
        both inner and outer bounds.
        
        Yields:
            Tile objects with all necessary spatial information and
            grid alignment parameters for seamless mosaicking.
        """
        # Compute number of tiles needed in each direction
        tx = int(np.ceil((self.gb.max_x - self.gb.min_x) / self.tile))
        ty = int(np.ceil((self.gb.max_y - self.gb.min_y) / self.tile))
        
        for j in range(ty):
            # Compute Y bounds for this tile row
            y0 = self.gb.min_y + j * self.tile
            y1 = min(self.gb.max_y, y0 + self.tile)
            
            for i in range(tx):
                # Compute X bounds for this tile column
                x0 = self.gb.min_x + i * self.tile
                x1 = min(self.gb.max_x, x0 + self.tile)
                
                # Define inner processing bounds
                inner = Bounds2D(min_x=x0, min_y=y0, max_x=x1, max_y=y1)
                
                # Define outer bounds with halo, clipped to global extent
                outer = Bounds2D(
                    min_x=max(self.gb.min_x, x0 - self.halo),
                    min_y=max(self.gb.min_y, y0 - self.halo),
                    max_x=min(self.gb.max_x, x1 + self.halo),
                    max_y=min(self.gb.max_y, y1 + self.halo),
                )
                
                # Compute tile grid dimensions aligned to global grid
                nx = int(np.ceil((inner.max_x - inner.min_x) / self.cell)) + 1
                ny = int(np.ceil((inner.max_y - inner.min_y) / self.cell)) + 1
                
                # Compute starting indices in the global grid
                x0_idx = int(round((inner.min_x - self.gb.min_x) / self.cell))
                y0_idx = int(round((inner.min_y - self.gb.min_y) / self.cell))
                
                yield Tile(i=i, j=j, inner=inner, outer=outer, x0_idx=x0_idx, y0_idx=y0_idx, nx=nx, ny=ny)


class MosaicAccumulator:
    """Assemble tile DEMs into a global grid by averaging overlaps.
    
    When tiles are processed with halos, they produce overlapping results.
    This class accumulates multiple tile DEMs into a single global grid,
    averaging values in overlapping regions to produce seamless output.
    
    The accumulator maintains running sums and counts to handle arbitrary
    overlap patterns and variable numbers of contributing tiles per cell.
    
    Attributes:
        gb: Global bounding box for the output grid
        cell: Grid cell size in data units
        nx: Total columns in the global grid
        ny: Total rows in the global grid
        sum: Running sum of DEM values per cell (ny x nx)
        cnt: Count of contributing tiles per cell (ny x nx)
        grid_x: X coordinates of cell centers (ny x nx)
        grid_y: Y coordinates of cell centers (ny x nx)
    """

    def __init__(self, global_bounds: Bounds2D, cell_size: float, *, memmap_dir: Optional[str | Path] = None) -> None:
        """Initialize the mosaic accumulator.
        
        Args:
            global_bounds: Bounding box for the output global grid
            cell_size: Size of grid cells in data units
        """
        from pathlib import Path as _Path
        self.gb = global_bounds
        self.cell = float(cell_size)
        self.nx = int(np.ceil((self.gb.max_x - self.gb.min_x) / self.cell)) + 1
        self.ny = int(np.ceil((self.gb.max_y - self.gb.min_y) / self.cell)) + 1

        # Optional on-disk backing for large outputs
        self._memmap_dir: Optional[_Path] = _Path(memmap_dir) if memmap_dir is not None else None
        if self._memmap_dir is not None:
            self._memmap_dir.mkdir(parents=True, exist_ok=True)
            sum_path = self._memmap_dir / "mosaic_sum.float64.memmap"
            cnt_path = self._memmap_dir / "mosaic_cnt.int64.memmap"
            # Create/overwrite memmap files
            self.sum = np.memmap(sum_path, mode="w+", dtype=np.float64, shape=(self.ny, self.nx))
            self.cnt = np.memmap(cnt_path, mode="w+", dtype=np.int64, shape=(self.ny, self.nx))
            # Zero-initialize
            self.sum[...] = 0.0
            self.cnt[...] = 0
            self._sum_path = str(sum_path)
            self._cnt_path = str(cnt_path)
        else:
            # In-memory accumulators
            self.sum = np.zeros((self.ny, self.nx), dtype=np.float64)
            self.cnt = np.zeros((self.ny, self.nx), dtype=np.int64)
        
        # Precompute grid cell centers for output
        x_edges = np.linspace(self.gb.min_x, self.gb.min_x + self.nx * self.cell, self.nx + 1)
        y_edges = np.linspace(self.gb.min_y, self.gb.min_y + self.ny * self.cell, self.ny + 1)
        self.grid_x, self.grid_y = np.meshgrid(
            (x_edges[:-1] + x_edges[1:]) / 2.0,
            (y_edges[:-1] + y_edges[1:]) / 2.0,
        )

    def add_tile(self, tile: Tile, dem_tile: np.ndarray) -> None:
        """Add a tile DEM to the global accumulator.
        
        The tile DEM is placed into the global grid at the appropriate location
        based on the tile's grid alignment parameters. Valid (non-NaN) values
        are accumulated for later averaging.
        
        Args:
            tile: Tile object containing grid alignment information
            dem_tile: 2D array (ny x nx) of DEM values for this tile
        
        Raises:
            AssertionError: If dem_tile shape doesn't match tile dimensions
        """
        h, w = dem_tile.shape
        assert h == tile.ny and w == tile.nx
        
        # Determine slice in global grid for this tile
        y_slice = slice(tile.y0_idx, tile.y0_idx + tile.ny)
        x_slice = slice(tile.x0_idx, tile.x0_idx + tile.nx)
        
        # Only accumulate valid (finite) values
        valid = np.isfinite(dem_tile)
        if not np.any(valid):
            return
        # Compute global indices for valid cells to avoid temporary-view writes
        ys, xs = np.nonzero(valid)
        self.sum[tile.y0_idx + ys, tile.x0_idx + xs] += dem_tile[ys, xs]
        self.cnt[tile.y0_idx + ys, tile.x0_idx + xs] += 1

    def finalize(self) -> np.ndarray:
        """Compute the final global DEM by averaging accumulated tiles.
        
        Returns:
            2D array (ny x nx) containing the mosaicked DEM.
            Cells with no contributing tiles contain np.nan.
        """
        out = np.full((self.ny, self.nx), np.nan, dtype=np.float64)
        mask = self.cnt > 0
        out[mask] = self.sum[mask] / self.cnt[mask]
        return out


class LaspyStreamReader:
    """Chunked LAS/LAZ iterator with bbox and classification filtering.
    
    Enables memory-efficient streaming of point cloud data from one or more
    LAZ/LAS files. Supports filtering by spatial extent and point classification
    to reduce memory usage and processing time.
    
    Points are yielded in chunks to enable incremental processing without
    loading entire files into memory.
    
    Attributes:
        files: List of file paths to stream from
        ground_only: If True, only return ground points (class 2)
        classification_filter: List of classification codes to include (overrides ground_only)
        chunk_points: Maximum points per yielded chunk
    """

    def __init__(
        self,
        files: Iterable[str | Path],
        *,
        ground_only: bool = True,
        classification_filter: Optional[List[int]] = None,
        chunk_points: int = 1_000_000,
    ) -> None:
        """Initialize the streaming reader.
        
        Args:
            files: Iterable of file paths to LAS/LAZ files
            ground_only: If True, only stream ground points (class 2).
                Ignored if classification_filter is provided.
            classification_filter: Optional list of classification codes to include.
                If provided, overrides ground_only behavior.
            chunk_points: Target number of points per chunk
        """
        self.files = [str(Path(f)) for f in files]
        self.ground_only = ground_only
        self.classification_filter = classification_filter
        self.chunk_points = int(chunk_points)

    def _mask_classes(self, las) -> np.ndarray:
        """Create a boolean mask for point classification filtering.
        
        Args:
            las: Laspy LAS/LAZ data object
            
        Returns:
            Boolean array indicating which points pass the classification filter
        """
        n = len(las)  # laspy 2.x: len() works directly on point record
        if hasattr(las, "classification"):
            classes = np.asarray(las.classification)
            return create_classification_mask(classes, self.ground_only, self.classification_filter)
        
        # No classification attribute: accept all points
        return np.ones(n, dtype=bool)

    def stream_points(
        self,
        bbox: Optional[Bounds2D] = None,
        *,
        transform: Optional["LocalCoordinateTransform"] = None,
    ) -> Iterator[np.ndarray]:
        """Stream point coordinates from all files with optional spatial filtering.
        
        Points are read in chunks, filtered by classification and optionally by
        bounding box, then yielded as Nx3 numpy arrays [X, Y, Z].
        
        Args:
            bbox: Optional bounding box to filter points spatially.
                Only points within bbox are yielded.
            transform: Optional LocalCoordinateTransform to apply to points.
                If provided, points are transformed to local coordinates.
                
        Yields:
            Nx3 arrays of filtered point coordinates [X, Y, Z]
        """
        import laspy

        for fp in self.files:
            with laspy.open(fp) as reader:
                for chunk in reader.chunk_iterator(self.chunk_points):
                    # Apply classification filter
                    mask = self._mask_classes(chunk)
                    
                    # Extract coordinates
                    x = np.asarray(chunk.x, dtype=np.float64)
                    y = np.asarray(chunk.y, dtype=np.float64)
                    z = np.asarray(chunk.z, dtype=np.float64)
                    
                    # Apply spatial filter if bbox is provided
                    if bbox is not None:
                        mask &= (
                            (x >= bbox.min_x)
                            & (x <= bbox.max_x)
                            & (y >= bbox.min_y)
                            & (y <= bbox.max_y)
                        )
                    
                    # Skip empty chunks
                    if not np.any(mask):
                        continue
                    
                    # Yield filtered points as Nx3 array
                    pts = np.column_stack([x[mask], y[mask], z[mask]])
                    
                    # Apply local coordinate transform if provided
                    if transform is not None:
                        pts = transform.to_local(pts)
                    
                    yield pts

    def reservoir_sample(
        self,
        n: int,
        transform: Optional["LocalCoordinateTransform"] = None,
        bbox: Optional[Bounds2D] = None,
    ) -> np.ndarray:
        """Reservoir sample n points uniformly from all files.
        
        Uses Algorithm R (reservoir sampling) to select n points uniformly at
        random from the streaming data without loading all points into memory.
        
        Args:
            n: Number of points to sample.
            transform: Optional local coordinate transform to apply.
            bbox: Optional spatial bounding box filter.
            
        Returns:
            np.ndarray: (n, 3) array of sampled points, or fewer if total
                points available is less than n.
        """
        reservoir = None
        filled = 0
        seen = 0
        
        for chunk in self.stream_points(bbox=bbox, transform=transform):
            if chunk.size == 0:
                continue
            m = len(chunk)
            
            if reservoir is None:
                reservoir = np.empty((n, 3), dtype=np.float64)
            
            # Fill reservoir first
            take = min(n - filled, m)
            if take > 0:
                reservoir[filled:filled + take] = chunk[:take]
                filled += take
                seen += take
                start = take
            else:
                start = 0
            
            # Replacement phase (Algorithm R)
            for k in range(start, m):
                j = seen + (k - start)
                r = np.random.randint(0, j + 1)
                if r < n:
                    reservoir[r] = chunk[k]
            seen += (m - start)
        
        if reservoir is None:
            return np.empty((0, 3), dtype=np.float64)
        
        # Return only filled portion if we didn't get n points
        return reservoir[:filled] if filled < n else reservoir


def union_bounds(files_a: Iterable[str | Path], files_b: Iterable[str | Path]) -> Bounds2D:
    """Compute the union of bounding boxes from two sets of LAS/LAZ files.
    
    This is useful for determining the global extent when processing multiple
    point cloud datasets that need to be aligned or compared.
    
    Args:
        files_a: First set of LAS/LAZ file paths
        files_b: Second set of LAS/LAZ file paths
        
    Returns:
        Bounds2D covering the union of all input file extents
    """
    import laspy

    def scan(files: Iterable[str | Path]) -> Tuple[float, float, float, float]:
        """Scan file headers to find the overall bounding box.
        
        Args:
            files: Iterable of LAS/LAZ file paths
            
        Returns:
            Tuple of (min_x, min_y, max_x, max_y) covering all files
        """
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

    # Scan both file sets
    ax0, ay0, ax1, ay1 = scan(files_a)
    bx0, by0, bx1, by1 = scan(files_b)
    
    # Return union bounds
    return Bounds2D(
        min_x=min(ax0, bx0),
        min_y=min(ay0, by0),
        max_x=max(ax1, bx1),
        max_y=max(ay1, by1),
    )


def scan_las_bounds(files: Iterable[str | Path]) -> List[Tuple[Path, Bounds2D]]:
    """Scan LAS/LAZ file headers to get 2D bounds per file.

    Args:
        files: Iterable of LAS/LAZ file paths

    Returns:
        List of (Path, Bounds2D) tuples with header extents.
    """
    import laspy

    out: List[Tuple[Path, Bounds2D]] = []
    for f in files:
        fp = Path(f)
        key = str(fp)
        b = _BOUNDS_CACHE.get(key)
        if b is None:
            with laspy.open(str(fp)) as r:
                h = r.header
                b = Bounds2D(
                    float(h.x_min),
                    float(h.y_min),
                    float(h.x_max),
                    float(h.y_max),
                )
            _BOUNDS_CACHE[key] = b
        out.append((fp, b))
    return out


def bounds_intersect(a: Bounds2D, b: Bounds2D) -> bool:
    """Check if two 2D bounding boxes intersect (inclusive edges)."""
    return not (a.max_x < b.min_x or a.min_x > b.max_x or a.max_y < b.min_y or a.min_y > b.max_y)
