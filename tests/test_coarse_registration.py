"""
Tests for coarse registration methods (centroid, PCA).
"""

import numpy as np
from pathlib import Path
import sys

# Ensure src is importable
sys.path.append(str(Path(__file__).parent.parent / "src"))

from terrain_change_detection.alignment.coarse_registration import CoarseRegistration


def _nn_rmse(A: np.ndarray, B: np.ndarray) -> float:
    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(B)
        d, _ = nbrs.kneighbors(A)
        return float(np.sqrt(np.mean(d ** 2)))
    except Exception:
        # Fallback: brute-force for small arrays
        n, m = len(A), len(B)
        dmin = []
        for i in range(n):
            dsq = np.sum((B - A[i]) ** 2, axis=1)
            dmin.append(np.sqrt(float(dsq.min())))
        return float(np.sqrt(np.mean(np.square(dmin))))


def test_centroid_translation_alignment():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(2000, 3)) * np.array([10.0, 5.0, 2.0])
    t = np.array([2.3, -1.1, 0.4])
    B = A + t

    before = _nn_rmse(A, B)
    T = CoarseRegistration(method="centroid").compute_initial_transform(A, B)
    A2 = CoarseRegistration.apply_transformation(A, T)
    after = _nn_rmse(A2, B)

    assert after < before
    # Should be nearly perfect (numerical noise only)
    assert after < 1e-6


def test_pca_coarse_alignment_reduces_error():
    rng = np.random.default_rng(1)
    # Anisotropic cloud to avoid PCA degeneracy
    A = rng.normal(size=(4000, 3)) * np.array([20.0, 5.0, 1.0]) + np.array([100.0, -50.0, 10.0])
    # Apply a known rotation around Z and translation
    th = np.deg2rad(15.0)
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    t = np.array([5.0, -3.0, 0.7])
    B = (A @ Rz.T) + t

    before = _nn_rmse(A, B)
    T = CoarseRegistration(method="pca").compute_initial_transform(A, B)
    A2 = CoarseRegistration.apply_transformation(A, T)
    after = _nn_rmse(A2, B)

    assert after < before
    # PCA is coarse; expect residual but significantly reduced
    assert after < before * 0.5

