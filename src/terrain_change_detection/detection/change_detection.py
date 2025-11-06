"""
Change Detection Algorithms

This module provides algorithms for detecting changes between multi-temporal
point cloud datasets. It defines stable interfaces and result data structures
for:

- DEM of Difference (DoD)
- Cloud-to-Cloud (C2C) comparison
- Multiscale Model to Model Cloud Comparison (M3C2) variants

Implementations are designed to integrate with the project workflow and
logging conventions. Heavy computations (e.g., M3C2 using py4dgeo) are added
step by step in subsequent phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Literal

import numpy as np

from ..utils.logging import setup_logger
from ..acceleration import (
    GridAccumulator,
    LaspyStreamReader,
    Bounds2D,
    union_bounds,
    Tiler,
    MosaicAccumulator,
)

logger = setup_logger(__name__)


# -------------------------------
# Result data structures (API)
# -------------------------------

@dataclass
class DoDResult:
	"""
	Result of DEM of Difference (DoD).

	Attributes:
		grid_x: 2D array of X coordinates (meshgrid)
		grid_y: 2D array of Y coordinates (meshgrid)
		dem1: 2D array of DEM at time T1
		dem2: 2D array of DEM at time T2 (aligned to T1)
		dod: 2D array of DoD = dem2 - dem1
		cell_size: Grid resolution used (meters)
		bounds: (min_x, min_y, max_x, max_y) used for gridding
		stats: Summary statistics (e.g., mean_change, positive_area, negative_area)
		metadata: Optional metadata dictionary
	"""

	grid_x: np.ndarray
	grid_y: np.ndarray
	dem1: np.ndarray
	dem2: np.ndarray
	dod: np.ndarray
	cell_size: float
	bounds: Tuple[float, float, float, float]
	stats: Dict[str, float]
	metadata: Optional[Dict] = None


@dataclass
class C2CResult:
	"""
	Result of Cloud-to-Cloud distance comparison.

	Attributes:
		distances: Per-point distances from source to nearest neighbor in target
		indices: Indices of matched target points for each source point
		rmse: Root mean square error (computed over valid pairs)
		mean: Mean of distances
		median: Median of distances
		n: Number of valid correspondences
		metadata: Optional metadata dictionary
	"""

	distances: np.ndarray
	indices: np.ndarray
	rmse: float
	mean: float
	median: float
	n: int
	metadata: Optional[Dict] = None


@dataclass
class M3C2Result:
	"""
	Result of M3C2 computations.

	Attributes:
		core_points: Array of core point coordinates (N x 3)
		distances: Signed distances at core points (N,)
		uncertainty: Optional uncertainty estimates (N,), when available
		significant: Optional boolean mask of significant changes, when available
		metadata: Optional metadata dictionary (parameters, variants, etc.)
	"""

	core_points: np.ndarray
	distances: np.ndarray
	uncertainty: Optional[np.ndarray] = None
	significant: Optional[np.ndarray] = None
	metadata: Optional[Dict] = None


@dataclass
class M3C2Params:
	"""
	Parameters for M3C2 computations (shared across variants).

	Attributes:
		projection_scale: Diameter or radius for normal estimation neighborhood
		cylinder_radius: Radius for measuring distances along core point normals
		max_depth: Maximum cylinder half-length along the normal direction
		min_neighbors: Minimum neighbors required to compute a metric
		normal_scale: Scale for computing surface normals (if different)
		confidence: Confidence level for significance tests (e.g., 0.95)
	"""

	projection_scale: float
	cylinder_radius: float
	max_depth: float
	min_neighbors: int = 10
	normal_scale: Optional[float] = None
	confidence: float = 0.95


# ---------------------------------
# Change detection main interfaces
# ---------------------------------

class ChangeDetector:
	"""
	Collection of change detection methods.

	Notes:
		- All inputs are expected as numpy arrays with shape (N, 3) for point clouds.
		- Prior spatial alignment should be performed before invoking these methods.
	"""

	# --- DEM of Difference (DoD) ---
	@staticmethod
	def compute_dod(
		points_t1: np.ndarray,
		points_t2: np.ndarray,
		cell_size: float = 1.0,
		bounds: Optional[Tuple[float, float, float, float]] = None,
		aggregator: Literal["mean", "median", "p95", "p5"] = "mean",
	) -> DoDResult:
		"""
		Compute DEM of Difference (DoD) by gridding the point clouds and differencing.

		Args:
			points_t1: Ground points at time T1 (N x 3)
			points_t2: Ground points at time T2 (aligned to T1) (M x 3)
			cell_size: Grid cell size in same units as coordinates (meters)
			bounds: Optional (min_x, min_y, max_x, max_y). If None, use union bounds.
			aggregator: Aggregation method for DEM: mean, median, p95, p5

		Returns:
			DoDResult with DEMs and difference grid.
		"""
		if points_t1.size == 0 or points_t2.size == 0:
			raise ValueError("Input point arrays must be non-empty.")

		if bounds is None:
			min_x = float(min(points_t1[:, 0].min(), points_t2[:, 0].min()))
			max_x = float(max(points_t1[:, 0].max(), points_t2[:, 0].max()))
			min_y = float(min(points_t1[:, 1].min(), points_t2[:, 1].min()))
			max_y = float(max(points_t1[:, 1].max(), points_t2[:, 1].max()))
			bounds = (min_x, min_y, max_x, max_y)

		min_x, min_y, max_x, max_y = bounds
		nx = int(np.ceil((max_x - min_x) / cell_size)) + 1
		ny = int(np.ceil((max_y - min_y) / cell_size)) + 1

		# Create grid coordinates
		x_edges = np.linspace(min_x, min_x + nx * cell_size, nx + 1)
		y_edges = np.linspace(min_y, min_y + ny * cell_size, ny + 1)
		grid_x, grid_y = np.meshgrid(
			(x_edges[:-1] + x_edges[1:]) / 2.0,
			(y_edges[:-1] + y_edges[1:]) / 2.0,
		)

		def grid_dem(points: np.ndarray) -> np.ndarray:
			# Bin points into grid cells
			xi = np.clip(((points[:, 0] - min_x) / cell_size).astype(int), 0, nx - 1)
			yi = np.clip(((points[:, 1] - min_y) / cell_size).astype(int), 0, ny - 1)

			# Aggregate Z by cell using chosen reducer
			dem = np.full((ny, nx), np.nan, dtype=float)
			# Use lists of z-values per cell (sparse accumulation)
			buckets: Dict[Tuple[int, int], list] = {}
			for xidx, yidx, z in zip(xi, yi, points[:, 2]):
				buckets.setdefault((yidx, xidx), []).append(float(z))

			if aggregator == "mean":
				reducer = lambda arr: float(np.mean(arr))
			elif aggregator == "median":
				reducer = lambda arr: float(np.median(arr))
			elif aggregator == "p95":
				reducer = lambda arr: float(np.percentile(arr, 95))
			elif aggregator == "p5":
				reducer = lambda arr: float(np.percentile(arr, 5))
			else:
				raise ValueError(f"Unsupported aggregator: {aggregator}")

			for (yidx, xidx), zs in buckets.items():
				dem[yidx, xidx] = reducer(zs)
			return dem

		dem1 = grid_dem(points_t1)
		dem2 = grid_dem(points_t2)

		# Difference (preserve NaNs where either DEM is NaN)
		dod = dem2 - dem1

		# Basic stats over valid cells
		valid = np.isfinite(dod)
		stats: Dict[str, float] = {
			"n_cells": int(valid.sum()),
			"mean_change": float(np.nanmean(dod)),
			"median_change": float(np.nanmedian(dod)),
			"rmse": float(np.sqrt(np.nanmean(np.square(dod)))),
			"min_change": float(np.nanmin(dod)),
			"max_change": float(np.nanmax(dod)),
		}

		return DoDResult(
			grid_x=grid_x,
			grid_y=grid_y,
			dem1=dem1,
			dem2=dem2,
			dod=dod,
			cell_size=cell_size,
			bounds=bounds,
			stats=stats,
			metadata={"aggregator": aggregator},
		)

	# --- Cloud-to-Cloud (C2C) ---
	@staticmethod
	def compute_c2c(
		source: np.ndarray,
		target: np.ndarray,
		max_distance: Optional[float] = None,
	) -> C2CResult:
		"""
		Compute nearest-neighbor distances from source cloud to target cloud.

		Args:
			source: Source point cloud (N x 3)
			target: Target point cloud (M x 3)
			max_distance: Optional distance threshold to filter correspondences

		Returns:
			C2CResult with per-point distances and summary stats.
		"""
		if source.size == 0 or target.size == 0:
			raise ValueError("Input point arrays must be non-empty.")

		dists: np.ndarray
		indices: np.ndarray
		try:
			from sklearn.neighbors import NearestNeighbors  # type: ignore
			nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
			nbrs.fit(target)
			d, i = nbrs.kneighbors(source)
			dists = d.flatten()
			indices = i.flatten()
		except Exception:
			# Fallback: pure NumPy brute-force for small arrays
			n, m = len(source), len(target)
			# Heuristic cap to avoid excessive memory/time
			if n * m > 2_000_000:
				raise ImportError(
					"C2C requires scikit-learn for large inputs. Please add 'scikit-learn' to dependencies."
				)
			# Compute squared distances in chunks to limit memory
			dmin = np.empty(n, dtype=float)
			imin = np.empty(n, dtype=int)
			chunk = max(1, 2000)
			for start in range(0, n, chunk):
				stop = min(n, start + chunk)
				a = source[start:stop]
				# (stop-start, m, 3)
				diff = a[:, None, :] - target[None, :, :]
				dsq = np.einsum("ijk,ijk->ij", diff, diff)
				local_min = np.argmin(dsq, axis=1)
				dmin[start:stop] = np.sqrt(dsq[np.arange(stop - start), local_min])
				imin[start:stop] = local_min
			dists = dmin
			indices = imin

		if max_distance is not None:
			mask = dists <= max_distance
			valid_dists = dists[mask]
			valid_indices = indices[mask]
		else:
			valid_dists = dists
			valid_indices = indices

		if valid_dists.size == 0:
			rmse = float("inf")
			mean = float("nan")
			median = float("nan")
			n = 0
		else:
			rmse = float(np.sqrt(np.mean(np.square(valid_dists))))
			mean = float(np.mean(valid_dists))
			median = float(np.median(valid_dists))
			n = int(valid_dists.size)

		return C2CResult(
			distances=dists,
			indices=indices,
			rmse=rmse,
		mean=mean,
		median=median,
		n=n,
		metadata={"max_distance": max_distance},
	)

	# --- Streaming DoD from files (mean only) ---
	@staticmethod
	def compute_dod_streaming_files(
		files_t1: list[str],
		files_t2: list[str],
		cell_size: float = 1.0,
		bounds: Optional[Tuple[float, float, float, float]] = None,
		*,
		ground_only: bool = True,
		classification_filter: Optional[list[int]] = None,
		chunk_points: int = 1_000_000,
	) -> DoDResult:
		"""
		Streaming DoD that reads LAS/LAZ files in chunks and computes mean-based DEMs.		Notes:
		- Aggregator is mean-only in this prototype (streaming-friendly).
		- bounds defaults to the union of LAS headers.
		"""
		if not files_t1 or not files_t2:
			raise ValueError("compute_dod_streaming_files requires non-empty file lists for T1 and T2")

		if bounds is None:
			b = union_bounds(files_t1, files_t2)
			bounds2d = b
			bounds_tuple = (b.min_x, b.min_y, b.max_x, b.max_y)
		else:
			bounds2d = Bounds2D(min_x=bounds[0], min_y=bounds[1], max_x=bounds[2], max_y=bounds[3])
			bounds_tuple = bounds

		acc1 = GridAccumulator(bounds2d, cell_size)
		acc2 = GridAccumulator(bounds2d, cell_size)

		reader1 = LaspyStreamReader(files_t1, ground_only=ground_only, classification_filter=classification_filter, chunk_points=chunk_points)
		reader2 = LaspyStreamReader(files_t2, ground_only=ground_only, classification_filter=classification_filter, chunk_points=chunk_points)

		for pts in reader1.stream_points(bounds2d):
			acc1.accumulate(pts)
		for pts in reader2.stream_points(bounds2d):
			acc2.accumulate(pts)

		dem1 = acc1.finalize()
		dem2 = acc2.finalize()
		dod = dem2 - dem1
		valid = np.isfinite(dod)
		stats: Dict[str, float] = {
			"n_cells": int(valid.sum()),
			"mean_change": float(np.nanmean(dod)),
			"median_change": float(np.nanmedian(dod)),
			"rmse": float(np.sqrt(np.nanmean(np.square(dod)))),
			"min_change": float(np.nanmin(dod)),
			"max_change": float(np.nanmax(dod)),
		}

		return DoDResult(
			grid_x=acc1.grid_x,
			grid_y=acc1.grid_y,
			dem1=dem1,
			dem2=dem2,
			dod=dod,
			cell_size=float(cell_size),
			bounds=bounds_tuple,
			stats=stats,
			metadata={"aggregator": "mean", "streaming": True},
		)

	@staticmethod
	def compute_dod_streaming_files_tiled(
		files_t1: list[str],
		files_t2: list[str],
		cell_size: float,
		tile_size: float,
		halo: float,
		*,
		ground_only: bool = True,
		classification_filter: Optional[list[int]] = None,
		chunk_points: int = 1_000_000,
	) -> DoDResult:
		"""
		Out-of-core tiled DoD (mean). Tiles are grid-aligned; overlapping contributions are averaged.
		"""
		if not files_t1 or not files_t2:
			raise ValueError("compute_dod_streaming_files_tiled requires file lists for T1/T2")
		gb = union_bounds(files_t1, files_t2)
		tiler = Tiler(gb, cell_size=cell_size, tile_size=tile_size, halo=halo)
		mosaic1 = MosaicAccumulator(gb, cell_size)
		mosaic2 = MosaicAccumulator(gb, cell_size)
		reader1 = LaspyStreamReader(files_t1, ground_only=ground_only, classification_filter=classification_filter, chunk_points=chunk_points)
		reader2 = LaspyStreamReader(files_t2, ground_only=ground_only, classification_filter=classification_filter, chunk_points=chunk_points)

		# For each tile: stream-filter chunks using tile.outer, accumulate into tile grid (inner only), add to mosaic
		for tile in tiler.tiles():
			acc1 = GridAccumulator(tile.inner, cell_size)
			acc2 = GridAccumulator(tile.inner, cell_size)
			for pts in reader1.stream_points(tile.outer):
				acc1.accumulate(pts)
			for pts in reader2.stream_points(tile.outer):
				acc2.accumulate(pts)
			dem1 = acc1.finalize()
			dem2 = acc2.finalize()
			mosaic1.add_tile(tile, dem1)
			mosaic2.add_tile(tile, dem2)

		dem1_global = mosaic1.finalize()
		dem2_global = mosaic2.finalize()
		dod = dem2_global - dem1_global
		valid = np.isfinite(dod)
		stats: Dict[str, float] = {
			"n_cells": int(valid.sum()),
			"mean_change": float(np.nanmean(dod)),
			"median_change": float(np.nanmedian(dod)),
			"rmse": float(np.sqrt(np.nanmean(np.square(dod)))),
			"min_change": float(np.nanmin(dod)),
			"max_change": float(np.nanmax(dod)),
		}

		return DoDResult(
			grid_x=mosaic1.grid_x,
			grid_y=mosaic1.grid_y,
			dem1=dem1_global,
			dem2=dem2_global,
			dod=dod,
			cell_size=float(cell_size),
			bounds=(gb.min_x, gb.min_y, gb.max_x, gb.max_y),
			stats=stats,
			metadata={"aggregator": "mean", "streaming": True, "tiled": True},
		)

	# --- M3C2 variants (using py4dgeo) ---

	@staticmethod
	def autotune_m3c2_params(
		cloud: np.ndarray,
		target_neighbors: int = 16,
		max_depth_factor: float = 0.6,
		min_radius: float = 1.0,
		max_radius: float = 20.0,
	) -> M3C2Params:
		"""
		Suggest M3C2 parameter scales based on point density.

		Heuristic:
		- Estimate density = N / area from XY bounding box.
		- Choose radius r so that expected neighbors ≈ target_neighbors.
		- projection_scale and cylinder_radius both set to r.
		- max_depth set to max_depth_factor * r (covers signal while limiting search).

		Args:
			cloud: Nx3 array used to estimate density (e.g., T1 or combined core region)
			target_neighbors: desired neighbors within cylinder radius (~12–24 typical)
			max_depth_factor: fraction of radius for cylinder half-length (0.5–1.0)
			min_radius: lower bound on radius
			max_radius: upper bound on radius

		Returns:
			M3C2Params with tuned scales; min_neighbors set to min(10,target_neighbors//2).
		"""
		if cloud.size == 0:
			raise ValueError("autotune_m3c2_params: cloud is empty")

		x_min, y_min = float(np.min(cloud[:, 0])), float(np.min(cloud[:, 1]))
		x_max, y_max = float(np.max(cloud[:, 0])), float(np.max(cloud[:, 1]))
		area = max(1e-6, (x_max - x_min) * (y_max - y_min))
		density = len(cloud) / area  # pts per m^2

		# Expected neighbors ~ density * π r^2 => r = sqrt(target / (π * density))
		if density <= 0:
			r_est = max_radius
		else:
			r_est = float(np.sqrt(max(1e-9, target_neighbors) / (np.pi * density)))
		r_est = float(np.clip(r_est, min_radius, max_radius))

		proj_scale = r_est
		cyl_radius = r_est
		max_depth = float(max(0.5, max_depth_factor * r_est))
		min_neigh = int(max(5, min(20, target_neighbors // 2)))

		return M3C2Params(
			projection_scale=proj_scale,
			cylinder_radius=cyl_radius,
			max_depth=max_depth,
			min_neighbors=min_neigh,
			normal_scale=None,
			confidence=0.95,
		)
	@staticmethod
	def compute_m3c2_original(
		core_points: np.ndarray,
		cloud_t1: np.ndarray,
		cloud_t2: np.ndarray,
		params: M3C2Params,
	) -> M3C2Result:
		"""
		Compute original M3C2 distances using py4dgeo.

		Args:
			core_points: Core points where distances are evaluated (N x 3)
			cloud_t1: Point cloud at time T1 (M1 x 3)
			cloud_t2: Point cloud at time T2 (M2 x 3)
			params: Parameters controlling neighborhood scales and cylinder

		Returns:
			M3C2Result with distances and optional uncertainties
		"""
		if core_points.size == 0 or cloud_t1.size == 0 or cloud_t2.size == 0:
			raise ValueError("Inputs must be non-empty arrays.")

		try:
			from py4dgeo.m3c2 import M3C2, Epoch  # type: ignore
		except Exception as e:
			raise ImportError(
				"py4dgeo is required for M3C2. Please ensure it is installed."
			) from e

		logger.info(
			"Running M3C2: core_points=%d, cloud_t1=%d, cloud_t2=%d, cyl_radius=%.3f, max_depth=%.3f",
			len(core_points), len(cloud_t1), len(cloud_t2), params.cylinder_radius, params.max_depth,
		)

		# Construct Epochs (py4dgeo ensures DP and contiguous memory)
		epoch1 = Epoch(cloud_t1)
		epoch2 = Epoch(cloud_t2)

		# Prepare normal radii list
		normal_scale = params.normal_scale if params.normal_scale is not None else params.projection_scale
		normal_radii = [float(normal_scale)]

		# Initialize algorithm
		algo = M3C2(
			epochs=(epoch1, epoch2),
			corepoints=core_points,
			cyl_radius=float(params.cylinder_radius),
			max_distance=float(params.max_depth),
			registration_error=0.0,
			robust_aggr=False,
			normal_radii=normal_radii,
		)

		# Run algorithm
		distances, uncertainties = algo.run()

		# distances is a float array; uncertainties may be structured
		distances = np.asarray(distances, dtype=float).reshape(-1)

		unc_vec: Optional[np.ndarray] = None
		meta_extra: Dict[str, float] = {}
		if uncertainties is not None:
			un = np.asarray(uncertainties)
			if un.dtype.fields:  # structured
				# Capture some summary fields in metadata if available
				for field in ("lodetection", "spread1", "spread2"):
					if field in un.dtype.fields:
						try:
							arr = un[field]
							meta_extra[f"unc_{field}_mean"] = float(np.mean(arr))
							meta_extra[f"unc_{field}_median"] = float(np.median(arr))
						except Exception:
							pass
				# Do not set numeric uncertainty to avoid misinterpretation
				unc_vec = None
			else:
				try:
					unc_vec = un.astype(float).reshape(-1)
				except Exception:
					unc_vec = None

		result = M3C2Result(
			core_points=core_points,
			distances=distances,
			uncertainty=unc_vec,
			significant=None,  # Filled by EP variant
			metadata={
				"variant": "original",
				"normal_radii": normal_radii,
				"cylinder_radius": float(params.cylinder_radius),
				"max_depth": float(params.max_depth),
				"min_neighbors": int(params.min_neighbors),
				"confidence": float(params.confidence),
				**meta_extra,
			},
		)

		logger.info(
			"M3C2 finished: n=%d, mean=%.4f m, median=%.4f m, std=%.4f m",
			result.distances.size,
			float(np.mean(result.distances)),
			float(np.median(result.distances)),
			float(np.std(result.distances)),
		)

		return result

	@staticmethod
	def compute_m3c2_plane_based(
		core_points: np.ndarray,
		cloud_t1: np.ndarray,
		cloud_t2: np.ndarray,
		params: M3C2Params,
	) -> M3C2Result:
		"""
		Compute correspondence-driven plane-based M3C2 (CD-PB M3C2).

		Note:
			Implemented in a subsequent phase. This is a typed interface placeholder.
		"""
		raise NotImplementedError("M3C2 (plane-based) will be implemented in the next phase.")

	@staticmethod
	def compute_m3c2_error_propagation(
		core_points: np.ndarray,
		cloud_t1: np.ndarray,
		cloud_t2: np.ndarray,
		params: M3C2Params,
		*,
		workers: int = 4,
	) -> M3C2Result:
		"""
		Compute M3C2 with error propagation (M3C2-EP) and significance assessment.

		Uses py4dgeo's M3C2 to obtain structured uncertainties and applies a
		level-of-detection (LoD) threshold to flag significant changes.
		"""
		# Use dedicated M3C2EP implementation from py4dgeo
		try:
			from py4dgeo.m3c2 import Epoch  # type: ignore
			import py4dgeo.m3c2ep as m3c2ep  # type: ignore
		except Exception as e:
			raise ImportError("py4dgeo with m3c2ep is required for M3C2-EP.") from e

		# Build Epochs and ensure scan position metadata is present and valid
		epoch1 = Epoch(cloud_t1)
		epoch2 = Epoch(cloud_t2)

		# Helper to enforce 1-based scanpos_id and complete scanpos_info
		def _ensure_scanpos(epoch, points: np.ndarray) -> None:
			# IDs: if missing, set to ones; if 0-based, shift to 1-based
			try:
				existing = epoch.scanpos_id  # may raise if not set yet
			except Exception:
				existing = None
			if existing is None or len(existing) != len(points):
				ids = np.ones(len(points), dtype=np.int32)
			else:
				ids = np.asarray(existing, dtype=np.int32).copy()
				# Ensure minimum is 1 (py4dgeo.m3c2ep expects 1..N)
				min_id = int(ids.min()) if ids.size else 1
				if min_id < 1:
					ids = ids - min_id + 1
				# Replace any zeros or negatives with 1
				ids[ids <= 0] = 1
			# Assign via property to ensure structured array is created
			epoch.scanpos_id = ids

			# Info: ensure a list with length max_id having required keys
			max_id = int(ids.max(initial=1)) if ids.size else 1
			infos = getattr(epoch, "scanpos_info", None)
			# Reasonable defaults (units: meters, radians)
			default_origin = np.mean(points, axis=0)
			default_info = {
				"origin": np.asarray(default_origin, dtype=float),
				"sigma_range": float(0.02),  # 2 cm range noise
				"sigma_scan": float(0.001),   # ~0.057 degrees
				"sigma_yaw": float(0.001),    # ~0.057 degrees
			}
			if not isinstance(infos, list) or len(infos) < max_id:
				infos = [default_info.copy() for _ in range(max_id)]
			else:
				# Patch missing keys and ensure length covers max_id
				for i in range(len(infos)):
					if not isinstance(infos[i], dict):
						infos[i] = default_info.copy()
					else:
						infos[i].setdefault("origin", np.asarray(default_origin, dtype=float))
						infos[i].setdefault("sigma_range", float(0.02))
						infos[i].setdefault("sigma_scan", float(0.001))
						infos[i].setdefault("sigma_yaw", float(0.001))
				# Extend if needed
				if len(infos) < max_id:
					infos.extend([default_info.copy() for _ in range(max_id - len(infos))])
			epoch.scanpos_info = infos

		# Apply to both epochs
		_ensure_scanpos(epoch1, cloud_t1)
		_ensure_scanpos(epoch2, cloud_t2)

		# Defaults for transformation and alignment covariance
		tfM = np.array([[1, 0, 0, 0],
						[0, 1, 0, 0],
						[0, 0, 1, 0]], dtype=float)
		Cxx = np.zeros((12, 12), dtype=float)
		refPointMov = np.array([0.0, 0.0, 0.0], dtype=float)

		normal_scale = params.normal_scale if params.normal_scale is not None else params.projection_scale
		# Optionally run in single-process mode (Windows spawn safety)
		serial_mode = max(1, int(workers)) == 1
		if serial_mode:
			# Monkey-patch m3c2ep to use NUM_THREADS=1 and NUM_BLOCKS=1
			# by wrapping the calculate_distances method
			orig_calc = m3c2ep.M3C2EP.calculate_distances
			def _serial_calc(self, epoch1, epoch2):
				import numpy as _np, math as _math, queue as _queue
				# Copy of original up to block/thread config
				print(self.name + " running (serial)")
				if not isinstance(self.cyl_radius, float):
					raise Exception("M3C2EP requires a single cylinder radius")
				epoch1.build_kdtree(); epoch2.build_kdtree()
				p1_coords = epoch1.cloud; p1_positions = epoch1.scanpos_id
				p2_coords = epoch2.cloud; p2_positions = epoch2.scanpos_id
				M3C2Meta = {"searchrad": self.cyl_radius, "maxdist": self.max_distance, "minneigh":5, "maxneigh":100000,
						   "spInfos":[epoch1.scanpos_info, epoch2.scanpos_info], "tfM": self.tfM, "Cxx": self.Cxx, "redPoint": self.refPointMov}
				refPointMov = self.refPointMov; tfM = self.tfM
				if self.perform_trans:
					p2_coords = p2_coords - refPointMov
					p2_coords = (tfM[:3, :3] @ p2_coords.T).T + tfM[:, 3] + refPointMov
				query_coords = self.corepoints
				query_norms = self.directions()
				if query_norms.shape[0] == 1:
					query_norms = _np.repeat(query_norms, query_coords.shape[0], axis=0)
				# Single block execution using existing helpers
				from py4dgeo.m3c2ep import radius_search, process_corepoint_list
				max_dist = M3C2Meta["maxdist"]; search_radius = M3C2Meta["searchrad"]
				effective_search_radius = _math.hypot(max_dist, search_radius)
				p1_idx = radius_search(epoch1, query_coords, effective_search_radius)
				p2_idx = radius_search(epoch2, query_coords, effective_search_radius)
				# Shared memory buffers
				import multiprocessing as _mp
				import multiprocessing.shared_memory as _shm
				# Ensure submodule is visible to py4dgeo's mp alias
				import py4dgeo.m3c2ep as _m
				_m.mp.shared_memory = _shm  # type: ignore
				p1_coords_shm = _shm.SharedMemory(create=True, size=p1_coords.nbytes)
				p1_coords_sha = _np.ndarray(p1_coords.shape, dtype=p1_coords.dtype, buffer=p1_coords_shm.buf); p1_coords_sha[:] = p1_coords[:]
				p2_coords_shm = _shm.SharedMemory(create=True, size=p2_coords.nbytes)
				p2_coords_sha = _np.ndarray(p2_coords.shape, dtype=p2_coords.dtype, buffer=p2_coords_shm.buf); p2_coords_sha[:] = p2_coords[:]
				return_dict = {}
				pbarQueue = _queue.Queue()
				# Directly invoke worker function without spawning
				process_corepoint_list(
					query_coords, query_norms, p1_idx, p1_coords_shm.name, p1_coords.shape, p1_positions,
					p2_idx, p2_coords_shm.name, p2_coords.shape, p2_positions, M3C2Meta, 0, return_dict, pbarQueue
				)
				# Collect results
				out = return_dict[0]
				p1_coords_shm.close(); p1_coords_shm.unlink(); p2_coords_shm.close(); p2_coords_shm.unlink()
				distances = out["val"]; cov1 = out["m3c2_cov1"]; cov2 = out["m3c2_cov2"]
				unc = {
					"lodetection": out["lod_new"],
					"spread1": out["m3c2_spread1"],
					"num_samples1": out["m3c2_n1"],
					"spread2": out["m3c2_spread2"],
					"num_samples2": out["m3c2_n2"],
				}
				uncertainties = _np.array(list(zip(unc["lodetection"], unc["spread1"], unc["num_samples1"], unc["spread2"], unc["num_samples2"])), dtype=[
					("lodetection","f8"),("spread1","f8"),("num_samples1","i8"),("spread2","f8"),("num_samples2","i8")
				])
				covariance = _np.array(list(zip(cov1, cov2)), dtype=[("cov1","f8",(3,3)), ("cov2","f8",(3,3))])
				print(self.name + " end (serial)")
				return distances, uncertainties, covariance
			m3c2ep.M3C2EP.calculate_distances = _serial_calc  # type: ignore

		algo = m3c2ep.M3C2EP(
			tfM=tfM,
			Cxx=Cxx,
			refPointMov=refPointMov,
			perform_trans=True,
			epochs=(epoch1, epoch2),
			corepoints=core_points,
			cyl_radius=float(params.cylinder_radius),
			max_distance=float(params.max_depth),
			registration_error=0.0,
			robust_aggr=False,
			normal_radii=[float(normal_scale)],
		)

		distances, uncertainties, covariance = algo.run()

		d = np.asarray(distances, dtype=float).reshape(-1)
		sig_mask: Optional[np.ndarray] = None
		ep_meta: Dict[str, float] = {}
		if uncertainties is not None:
			un = np.asarray(uncertainties)
			if un.dtype.fields and 'lodetection' in un.dtype.fields:
				lod = un['lodetection'].astype(float).reshape(-1)
				sig_mask = np.isfinite(d) & np.isfinite(lod) & (np.abs(d) > lod)
				ep_meta["ep_lod_mean"] = float(np.nanmean(lod))
				ep_meta["ep_lod_median"] = float(np.nanmedian(lod))

		result = M3C2Result(
			core_points=core_points,
			distances=d,
			uncertainty=None,  # EP provides structured uncertainties; stored in metadata summaries
			significant=sig_mask,
			metadata={
				"variant": "m3c2_ep",
				"normal_radii": [float(normal_scale)],
				"cylinder_radius": float(params.cylinder_radius),
				"max_depth": float(params.max_depth),
				**ep_meta,
			},
		)

		logger.info(
			"M3C2-EP finished: n=%d, mean=%.4f m, median=%.4f m, std=%.4f m, sig=%d",
			result.distances.size,
			float(np.nanmean(result.distances)),
			float(np.nanmedian(result.distances)),
			float(np.nanstd(result.distances)),
			int(np.sum(result.significant)) if result.significant is not None else 0,
		)

		return result
