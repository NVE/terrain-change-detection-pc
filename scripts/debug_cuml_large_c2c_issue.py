#!/usr/bin/env python
"""
Debug script to isolate cuML nearest-neighbor issues on large real-world point clouds.

This focuses on the basic k-NN distance computation used by compute_c2c(), but calls
cuML.NearestNeighbors directly (bypassing the project wrapper) so we can see exactly
how cuML behaves at different sizes.

Usage (from repo root, with GPU env active):

    source activate_gpu.sh
    uv run scripts/debug_cuml_large_c2c_issue.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from terrain_change_detection.utils.config import load_config  # noqa: E402
from terrain_change_detection.preprocessing.loader import PointCloudLoader  # noqa: E402


def load_points(config_path: Path, max_points: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load reference (2015) and comparison (2020) ground points, optionally subsampled."""
    cfg = load_config(config_path)
    base_dir = Path(cfg.paths.base_dir)

    f_ref = base_dir / "eksport_1225654_20250602" / "2015" / "data" / "eksport_1225654_416_1.laz"
    f_cmp = base_dir / "eksport_1225654_20250602" / "2020" / "data" / "eksport_1225654_4241_1.laz"

    if not f_ref.exists() or not f_cmp.exists():
        raise FileNotFoundError(
            "Expected real-data files not found.\n"
            f"  {f_ref}\n"
            f"  {f_cmp}\n"
            "Adjust paths in scripts/debug_cuml_large_c2c_issue.py if your layout differs."
        )

    loader = PointCloudLoader(
        ground_only=cfg.preprocessing.ground_only,
        classification_filter=cfg.preprocessing.classification_filter,
    )

    print(f"Loading reference:   {f_ref}")
    ref_data = loader.load(f_ref)
    print(f"Loading comparison:  {f_cmp}")
    cmp_data = loader.load(f_cmp)

    ref_pts = ref_data["points"]
    cmp_pts = cmp_data["points"]
    print(f"  Loaded reference:  {len(ref_pts):,} ground points")
    print(f"  Loaded comparison: {len(cmp_pts):,} ground points")

    if max_points is not None:
        rng = np.random.default_rng(42)
        if len(ref_pts) > max_points:
            idx = rng.choice(len(ref_pts), max_points, replace=False)
            ref_pts = ref_pts[idx]
        if len(cmp_pts) > max_points:
            idx = rng.choice(len(cmp_pts), max_points, replace=False)
            cmp_pts = cmp_pts[idx]
        print(f"  Subsampled to at most {max_points:,} points per cloud")
        print(f"    reference:  {len(ref_pts):,}")
        print(f"    comparison: {len(cmp_pts):,}")

    return ref_pts, cmp_pts


def run_cuml_knn(ref: np.ndarray, cmp: np.ndarray, label: str, do_cpu_baseline: bool = False) -> None:
    """Run cuML NearestNeighbors on centered float32 coordinates and report statistics.

    Optionally also compute a CPU baseline with sklearn for direct comparison.
    """
    import cupy as cp
    from cuml.neighbors import NearestNeighbors as CuMLNN

    n_ref = ref.shape[0]
    n_cmp = cmp.shape[0]

    print(f"\n=== {label} ===")
    print(f"Reference (target) points:   {n_ref:,}")
    print(f"Comparison (query) points:   {n_cmp:,}")

    # Center and cast to float32 exactly as in compute_c2c()
    ref_f32 = np.asarray(ref, dtype=np.float32)
    cmp_f32 = np.asarray(cmp, dtype=np.float32)
    center = np.mean(ref_f32, axis=0)
    ref_centered = ref_f32 - center
    cmp_centered = cmp_f32 - center

    ref_gpu = cp.asarray(ref_centered)
    cmp_gpu = cp.asarray(cmp_centered)

    nn = CuMLNN(
        n_neighbors=1,
        algorithm="auto",  # mirrors project usage
        metric="euclidean",
    )

    # Fit
    t0 = time.time()
    nn.fit(ref_gpu)
    cp.cuda.Stream.null.synchronize()
    t1 = time.time()

    # Query
    t2 = time.time()
    dist_gpu, idx_gpu = nn.kneighbors(cmp_gpu)
    cp.cuda.Stream.null.synchronize()
    t3 = time.time()

    d_gpu = cp.asnumpy(dist_gpu).ravel()

    finite_mask_gpu = np.isfinite(d_gpu)
    n_finite_gpu = int(finite_mask_gpu.sum())
    n_total = d_gpu.size

    print(f"  Fit time (GPU):    {t1 - t0:.3f}s")
    print(f"  Query time (GPU):  {t3 - t2:.3f}s")
    print(f"  Total time (GPU):  {t3 - t0:.3f}s")
    print(f"  Distances (GPU):   {n_total:,} values, {n_finite_gpu:,} finite")

    if n_finite_gpu == 0:
        print("  All GPU distances are non-finite (NaN/inf)!")
        return

    d_gpu_finite = d_gpu[finite_mask_gpu]
    print(f"  GPU min:    {float(np.min(d_gpu_finite)):.6f} m")
    print(f"  GPU median: {float(np.median(d_gpu_finite)):.6f} m")
    print(f"  GPU mean:   {float(np.mean(d_gpu_finite)):.6f} m")
    print(f"  GPU max:    {float(np.max(d_gpu_finite)):.6e} m")

    # Optional CPU baseline for comparison on the same centered data
    if do_cpu_baseline:
        from sklearn.neighbors import NearestNeighbors

        cpu_nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
        t0c = time.time()
        cpu_nn.fit(ref_centered)
        t1c = time.time()
        d_cpu, _ = cpu_nn.kneighbors(cmp_centered)
        t2c = time.time()

        d_cpu = d_cpu.ravel()
        finite_mask_cpu = np.isfinite(d_cpu)
        n_finite_cpu = int(finite_mask_cpu.sum())

        print(f"  Fit time (CPU):    {t1c - t0c:.3f}s")
        print(f"  Query time (CPU):  {t2c - t1c:.3f}s")
        print(f"  Total time (CPU):  {t2c - t0c:.3f}s")
        print(f"  Distances (CPU):   {d_cpu.size:,} values, {n_finite_cpu:,} finite")

        d_cpu_finite = d_cpu[finite_mask_cpu]
        print(f"  CPU min:    {float(np.min(d_cpu_finite)):.6f} m")
        print(f"  CPU median: {float(np.median(d_cpu_finite)):.6f} m")
        print(f"  CPU mean:   {float(np.mean(d_cpu_finite)):.6f} m")
        print(f"  CPU max:    {float(np.max(d_cpu_finite)):.6e} m")

        # Compare summary statistics
        diff_mean = float(np.abs(np.mean(d_cpu_finite) - np.mean(d_gpu_finite)))
        diff_med = float(np.abs(np.median(d_cpu_finite) - np.median(d_gpu_finite)))
        print(f"  |CPU-GPU mean diff|   = {diff_mean:.6f} m")
        print(f"  |CPU-GPU median diff| = {diff_med:.6f} m")

    # Quick sanity check for obviously broken GPU results
    if np.max(d_gpu_finite) > 1e5:
        print("  [WARN] GPU max distance > 1e5 m (implausible for terrain).")


def main() -> None:
    cfg_path = Path("config/default.yaml")
    ref_full, cmp_full = load_points(cfg_path, max_points=None)

    total_ref = ref_full.shape[0]
    total_cmp = cmp_full.shape[0]

    print("\n=== Global availability ===")
    print(f"Reference total:   {total_ref:,}")
    print(f"Comparison total:  {total_cmp:,}")

    # Sizes to probe (ensure we don't exceed available points)
    sizes = [
        100_000,
        200_000,
        500_000,
        1_000_000,
        min(2_000_000, total_cmp),
        min(total_cmp, total_cmp),
    ]

    # Deduplicate and ensure strictly increasing
    sizes = sorted(set(s for s in sizes if s > 0))

    rng = np.random.default_rng(123)

    for size in sizes:
        if size > total_cmp or size > total_ref:
            print(f"\nSkipping size {size:,}: not enough points in one of the clouds.")
            continue

        # Random subsets to keep behavior realistic but bounded
        idx_ref = rng.choice(total_ref, size, replace=False)
        idx_cmp = rng.choice(total_cmp, size, replace=False)

        ref_subset = ref_full[idx_ref]
        cmp_subset = cmp_full[idx_cmp]

        # Only run full CPU baseline on the largest size we probe, to keep runtime reasonable
        do_cpu = size == sizes[-1]
        run_cuml_knn(ref_subset, cmp_subset, label=f"cuML kNN on {size:,} points", do_cpu_baseline=do_cpu)


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    main()
