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
import pytest

# Ensure src is importable
sys.path.append(str(Path(__file__).parent.parent / "src"))

from terrain_change_detection.alignment.fine_registration import (
    ICPRegistration,
    compute_overlap_mask,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_random_cloud(n: int = 5000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Anisotropic spread to avoid degenerate covariance
    base = rng.normal(size=(n, 3)) * np.array([10.0, 5.0, 2.0])
    base += np.array([100.0, -50.0, 20.0])
    return base.astype(float)


def _apply_rigid_transform(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (points @ R.T) + t


def _small_rotation_z(deg: float) -> np.ndarray:
    th = np.deg2rad(deg)
    return np.array([
        [np.cos(th), -np.sin(th), 0.0],
        [np.sin(th),  np.cos(th), 0.0],
        [0.0,         0.0,        1.0],
    ])


# ---------------------------------------------------------------------------
# Existing tests
# ---------------------------------------------------------------------------

def test_icp_recovers_known_transform():
    """ICP should approximately recover a known rigid transform."""
    src = _make_random_cloud(n=4000, seed=1)

    Rz = _small_rotation_z(5.0)
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

    aligned, T, err = icp.align_point_clouds(source=src, target=tgt)

    assert aligned.shape[0] == 0
    assert T.shape == (4, 4)
    assert np.isfinite(T).all()
    assert err == float("inf")


# ---------------------------------------------------------------------------
# Issue 3 – Determinism
# ---------------------------------------------------------------------------

def test_icp_deterministic_with_same_input():
    """Two ICP runs on the same data must produce identical transforms."""
    src = _make_random_cloud(n=3000, seed=10)
    Rz = _small_rotation_z(3.0)
    t = np.array([0.5, -0.3, 0.1])
    tgt = _apply_rigid_transform(src, Rz, t)

    icp = ICPRegistration(
        max_iterations=50,
        tolerance=1e-8,
        max_correspondence_distance=5.0,
    )

    _, T1, err1 = icp.align_point_clouds(source=src, target=tgt)
    _, T2, err2 = icp.align_point_clouds(source=src, target=tgt)

    np.testing.assert_array_equal(T1, T2)
    assert err1 == err2


def test_subsampling_deterministic_with_rng():
    """np.random.default_rng with fixed seed produces identical subsamples."""
    cloud = _make_random_cloud(n=10000, seed=5)

    rng1 = np.random.default_rng(42)
    idx1 = rng1.choice(len(cloud), 500, replace=False)

    rng2 = np.random.default_rng(42)
    idx2 = rng2.choice(len(cloud), 500, replace=False)

    np.testing.assert_array_equal(idx1, idx2)


def test_subsampling_different_seeds_differ():
    """Different seeds should produce different subsamples."""
    cloud = _make_random_cloud(n=10000, seed=5)

    rng1 = np.random.default_rng(42)
    idx1 = rng1.choice(len(cloud), 500, replace=False)

    rng2 = np.random.default_rng(99)
    idx2 = rng2.choice(len(cloud), 500, replace=False)

    assert not np.array_equal(idx1, idx2)


# ---------------------------------------------------------------------------
# Issue 4 – Percentage-based subsampling
# ---------------------------------------------------------------------------

def test_resolve_subsample_count_count_mode():
    """Count mode returns the configured subsample_size."""
    from types import SimpleNamespace
    cfg = SimpleNamespace(
        subsample_mode="count",
        subsample_size=5000,
        subsample_percent=10.0,
        max_subsample_size=500_000,
    )
    # Import helper from run_workflow (it's a top-level function)
    sys.path.append(str(Path(__file__).parent.parent / "scripts"))
    from run_workflow import resolve_subsample_count

    assert resolve_subsample_count(100_000, cfg) == 5000


def test_resolve_subsample_count_percent_mode():
    """Percent mode computes the right count."""
    from types import SimpleNamespace
    cfg = SimpleNamespace(
        subsample_mode="percent",
        subsample_size=5000,
        subsample_percent=10.0,
        max_subsample_size=500_000,
    )
    sys.path.append(str(Path(__file__).parent.parent / "scripts"))
    from run_workflow import resolve_subsample_count

    assert resolve_subsample_count(100_000, cfg) == 10_000  # 10% of 100k


def test_resolve_subsample_count_respects_cap():
    """The safety cap should limit both modes."""
    from types import SimpleNamespace
    cfg = SimpleNamespace(
        subsample_mode="percent",
        subsample_size=5000,
        subsample_percent=50.0,
        max_subsample_size=20_000,
    )
    sys.path.append(str(Path(__file__).parent.parent / "scripts"))
    from run_workflow import resolve_subsample_count

    # 50% of 100k = 50000, but cap is 20000
    assert resolve_subsample_count(100_000, cfg) == 20_000


# ---------------------------------------------------------------------------
# Issue 5 – Overlap filtering
# ---------------------------------------------------------------------------

def test_overlap_mask_full_overlap():
    """Two clouds with same extent should have all points pass."""
    rng = np.random.default_rng(0)
    cloud = rng.uniform(0, 100, size=(1000, 3))

    mask_a, mask_b = compute_overlap_mask(cloud, cloud, margin=0.0)

    assert mask_a.all()
    assert mask_b.all()


def test_overlap_mask_partial_overlap():
    """One cloud shifted by 50m in X should produce partial overlap."""
    rng = np.random.default_rng(0)
    cloud_a = rng.uniform(0, 100, size=(1000, 3))
    cloud_b = cloud_a.copy()
    cloud_b[:, 0] += 50  # Shift X by 50m

    mask_a, mask_b = compute_overlap_mask(cloud_a, cloud_b, margin=0.0)

    # Only points in [50, 100] for A and [50, 100] for B (which is [0, 50] original)
    assert 0 < mask_a.sum() < len(cloud_a)
    assert 0 < mask_b.sum() < len(cloud_b)


def test_overlap_mask_no_overlap():
    """Two disjoint clouds should produce empty masks."""
    rng = np.random.default_rng(0)
    cloud_a = rng.uniform(0, 10, size=(500, 3))
    cloud_b = rng.uniform(100, 110, size=(500, 3))

    mask_a, mask_b = compute_overlap_mask(cloud_a, cloud_b, margin=0.0)

    assert mask_a.sum() == 0
    assert mask_b.sum() == 0


def test_overlap_mask_with_margin():
    """Margin should expand the overlap region to include near-boundary points."""
    rng = np.random.default_rng(0)
    cloud_a = rng.uniform(0, 10, size=(500, 3))
    cloud_b = rng.uniform(10, 20, size=(500, 3))  # Just touching at boundary

    mask_no_margin_a, mask_no_margin_b = compute_overlap_mask(cloud_a, cloud_b, margin=0.0)
    mask_margin_a, mask_margin_b = compute_overlap_mask(cloud_a, cloud_b, margin=2.0)

    # With 2m margin, more points should pass
    assert mask_margin_a.sum() >= mask_no_margin_a.sum()
    assert mask_margin_b.sum() >= mask_no_margin_b.sum()


# ---------------------------------------------------------------------------
# Issue 2 – Reference swap
# ---------------------------------------------------------------------------

def test_reference_swap_produces_approximate_inverse():
    """Swapping source/target should produce approximately inverse transforms."""
    src = _make_random_cloud(n=3000, seed=20)
    Rz = _small_rotation_z(2.0)
    t = np.array([0.3, -0.2, 0.1])
    tgt = _apply_rigid_transform(src, Rz, t)

    icp = ICPRegistration(
        max_iterations=50,
        tolerance=1e-8,
        max_correspondence_distance=5.0,
    )

    # src -> tgt
    _, T_fwd, _ = icp.align_point_clouds(source=src, target=tgt)
    # tgt -> src
    _, T_rev, _ = icp.align_point_clouds(source=tgt, target=src)

    # T_fwd @ T_rev should be close to identity
    product = T_fwd @ T_rev
    np.testing.assert_allclose(product, np.eye(4), atol=0.05)


# ---------------------------------------------------------------------------
# Issue 6 – Open3D backend
# ---------------------------------------------------------------------------

_has_open3d = True
try:
    import open3d as _o3d  # noqa: F401
except ImportError:
    _has_open3d = False


@pytest.mark.skipif(not _has_open3d, reason="Open3D not installed")
def test_open3d_icp_recovers_known_transform():
    """Open3D backend should recover a known rigid transform."""
    from terrain_change_detection.alignment.open3d_icp import Open3DICP

    src = _make_random_cloud(n=4000, seed=1)
    Rz = _small_rotation_z(5.0)
    t = np.array([1.5, -0.7, 0.3])
    tgt = _apply_rigid_transform(src, Rz, t)

    icp = Open3DICP(
        max_iterations=50,
        tolerance=1e-8,
        max_correspondence_distance=5.0,
    )

    aligned, T_est, rmse = icp.align_point_clouds(source=src, target=tgt)

    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(tgt)
    d0, _ = nn.kneighbors(src)
    baseline_rmse = float(np.sqrt(np.mean(d0 ** 2)))

    assert rmse < baseline_rmse * 0.8


@pytest.mark.skipif(not _has_open3d, reason="Open3D not installed")
def test_backends_produce_similar_results():
    """Custom and Open3D backends should produce similar alignment quality."""
    from terrain_change_detection.alignment.open3d_icp import Open3DICP

    src = _make_random_cloud(n=3000, seed=30)
    Rz = _small_rotation_z(3.0)
    t = np.array([1.0, -0.5, 0.2])
    tgt = _apply_rigid_transform(src, Rz, t)

    params = dict(
        max_iterations=50,
        tolerance=1e-8,
        max_correspondence_distance=5.0,
    )

    _, _, err_custom = ICPRegistration(**params).align_point_clouds(source=src, target=tgt)
    _, _, err_open3d = Open3DICP(**params).align_point_clouds(source=src, target=tgt)

    # Both should be low; allow up to 50% relative difference
    assert err_custom < 0.5, f"Custom ICP error too high: {err_custom}"
    assert err_open3d < 0.5, f"Open3D ICP error too high: {err_open3d}"
    rel_diff = abs(err_custom - err_open3d) / max(err_custom, err_open3d, 1e-12)
    assert rel_diff < 0.5, f"Backends differ too much: custom={err_custom:.6f}, open3d={err_open3d:.6f}"
