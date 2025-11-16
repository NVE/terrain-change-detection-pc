"""
Tests for fine registration (ICP) implementation.

These tests focus on correctness of the recovered transform and
basic convergence behavior on synthetic data. They exercise the
CPU code path only; GPU acceleration for ICP will be handled
separately.
"""

from pathlib import Path
import sys

import numpy as np

# Ensure src is importable
sys.path.append(str(Path(__file__).parent.parent / "src"))

from terrain_change_detection.alignment.fine_registration import ICPRegistration


def _make_random_cloud(n: int = 5000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Anisotropic spread to avoid degenerate covariance
    base = rng.normal(size=(n, 3)) * np.array([10.0, 5.0, 2.0])
    base += np.array([100.0, -50.0, 20.0])
    return base.astype(float)


def _apply_rigid_transform(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (points @ R.T) + t


def test_icp_recovers_known_transform():
    """ICP should approximately recover a known rigid transform."""
    src = _make_random_cloud(n=4000, seed=1)

    # Construct a small rotation and translation
    th = np.deg2rad(5.0)
    Rz = np.array(
        [
            [np.cos(th), -np.sin(th), 0.0],
            [np.sin(th), np.cos(th), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    t = np.array([1.5, -0.7, 0.3])

    tgt = _apply_rigid_transform(src, Rz, t)

    icp = ICPRegistration(
        max_iterations=50,
        tolerance=1e-8,
        max_correspondence_distance=5.0,
        convergence_translation_epsilon=1e-6,
        convergence_rotation_epsilon_deg=0.01,
    )

    aligned, T_est, final_err = icp.align_point_clouds(source=src, target=tgt)

    # Compare against a naive identity-transform baseline using NN RMSE
    from sklearn.neighbors import NearestNeighbors  # type: ignore

    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(tgt)
    d0, _ = nn.kneighbors(src)
    baseline_rmse = float(np.sqrt(np.mean(d0 ** 2)))

    # ICP should significantly reduce RMSE vs baseline
    assert final_err < baseline_rmse * 0.8


def test_icp_handles_empty_inputs_gracefully():
    """ICP should not crash on empty point sets."""
    icp = ICPRegistration()
    src = np.empty((0, 3), dtype=float)
    tgt = np.empty((0, 3), dtype=float)

    # align_point_clouds should still return arrays/matrix, though
    # compute_registration_error will be inf when there are no correspondences.
    aligned, T, err = icp.align_point_clouds(source=src, target=tgt)

    assert aligned.shape[0] == 0
    assert T.shape == (4, 4)
    assert np.isfinite(T).all()
    # With no points, error is expected to be inf (as per implementation)
    assert err == float("inf")
