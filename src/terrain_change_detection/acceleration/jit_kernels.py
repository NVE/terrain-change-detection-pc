"""JIT-compiled kernels for performance-critical operations.

These kernels provide optimized CPU implementations for common operations
that are executed many times in the pipeline. They are used as an optional
acceleration path; the codebase falls back to NumPy when numba is unavailable.
"""

from __future__ import annotations


import numpy as np

try:
    import numba
except Exception:  # pragma: no cover - numba is optional
    numba = None  # type: ignore


def _jit(nopython: bool = True, parallel: bool = False):
    """Small helper to guard numba usage when not installed."""
    if numba is None:
        # Identity decorator when numba is unavailable
        def decorator(func):
            return func

        return decorator
    return numba.jit(nopython=nopython, parallel=parallel)


@_jit(nopython=True, parallel=False)
def apply_transform_jit(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply 4x4 transformation matrix to points (JIT-compiled).

    Args:
        points: (N, 3) array of XYZ coordinates.
        matrix: (4, 4) transformation matrix.

    Returns:
        Transformed points (N, 3).
    """
    n = points.shape[0]
    result = np.empty_like(points)

    for i in range(n):
        x = points[i, 0]
        y = points[i, 1]
        z = points[i, 2]
        result[i, 0] = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2] * z + matrix[0, 3]
        result[i, 1] = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2] * z + matrix[1, 3]
        result[i, 2] = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2] * z + matrix[2, 3]

    return result


@_jit(nopython=True, parallel=False)
def compute_distances_jit(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """Compute Euclidean distances between corresponding points (JIT-compiled).

    Args:
        points1: (N, 3) array of XYZ coordinates.
        points2: (N, 3) array of XYZ coordinates.

    Returns:
        1D array of distances of length N.
    """
    n = points1.shape[0]
    distances = np.empty(n, dtype=np.float64)

    for i in range(n):
        dx = points1[i, 0] - points2[i, 0]
        dy = points1[i, 1] - points2[i, 1]
        dz = points1[i, 2] - points2[i, 2]
        distances[i] = np.sqrt(dx * dx + dy * dy + dz * dz)

    return distances
