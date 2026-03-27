"""Diagnostic script: compare original M3C2 vs M3C2-EP on a controlled test case.

Creates a flat plane (T1) and a shifted plane (T2) with known +0.5m Z offset,
then runs both variants and prints results side-by-side.
"""
import sys
import os
import io
from pathlib import Path

import numpy as np

# Suppress py4dgeo's multiprocessing prints
import logging
logging.getLogger("py4dgeo").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from terrain_change_detection.detection.m3c2 import (
    M3C2Detector,
    M3C2Params,
    _build_epoch,
)


def make_grid(nx=30, ny=30, spacing=1.0, z_offset=0.0, noise=0.01, seed=42):
    rng = np.random.default_rng(seed)
    x = np.arange(nx) * spacing
    y = np.arange(ny) * spacing
    X, Y = np.meshgrid(x, y)
    Z = z_offset + noise * rng.standard_normal(X.shape)
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


def main():
    known_dz = 0.5  # T2 is 0.5 m above T1
    cloud_t1 = make_grid(z_offset=0.0, seed=1)
    cloud_t2 = make_grid(z_offset=known_dz, seed=2)

    # Use a subset as core points (every 4th point from T1)
    core_points = cloud_t1[::4].copy()

    params = M3C2Params(
        projection_scale=3.0,
        cylinder_radius=3.0,
        max_depth=2.0,
        normal_scale=3.0,
    )

    # --- Original M3C2 (both clouds already aligned) ---
    print("=" * 60)
    print("ORIGINAL M3C2 (identity transform, pre-aligned clouds)")
    print("=" * 60)
    res_orig = M3C2Detector.compute_m3c2_original(
        core_points=core_points,
        cloud_t1=cloud_t1,
        cloud_t2=cloud_t2,
        params=params,
    )
    valid_orig = np.isfinite(res_orig.distances)
    print(f"  Valid:  {valid_orig.sum()} / {len(res_orig.distances)}")
    print(f"  Mean:   {np.nanmean(res_orig.distances):.6f}  (expected ~{known_dz:+.1f})")
    print(f"  Median: {np.nanmedian(res_orig.distances):.6f}")
    print(f"  Std:    {np.nanstd(res_orig.distances):.6f}")
    print(f"  Min:    {np.nanmin(res_orig.distances):.6f}")
    print(f"  Max:    {np.nanmax(res_orig.distances):.6f}")

    # --- M3C2-EP (identity transform, same clouds) ---
    print()
    print("=" * 60)
    print("M3C2-EP (identity transform, same pre-aligned clouds)")
    print("=" * 60)

    raw_scan_ids_t1 = np.ones(len(cloud_t1), dtype=np.int32)
    raw_scan_ids_t2 = np.ones(len(cloud_t2), dtype=np.int32)

    scan_meta_t1 = M3C2Detector.resolve_scan_metadata(
        points=cloud_t1,
        raw_scan_ids=raw_scan_ids_t1,
        scan_metadata_source="synthetic_from_point_source_id",
        synthetic_origin_height=100.0,
        epoch_label="T1",
    )
    scan_meta_t2 = M3C2Detector.resolve_scan_metadata(
        points=cloud_t2,
        raw_scan_ids=raw_scan_ids_t2,
        scan_metadata_source="synthetic_from_point_source_id",
        synthetic_origin_height=100.0,
        epoch_label="T2",
    )

    try:
        # Suppress py4dgeo's multiprocessing stdout noise
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        res_ep = M3C2Detector.compute_m3c2_ep(
            core_points=core_points,
            cloud_t1=cloud_t1,
            cloud_t2=cloud_t2,
            params=params,
            transform_matrix=np.eye(4),
            cxx=np.zeros((12, 12), dtype=np.float64),
            scan_metadata_t1=scan_meta_t1,
            scan_metadata_t2=scan_meta_t2,
            reduction_point=np.zeros(3, dtype=np.float64),
            perform_transform=True,
        )
        sys.stdout = old_stdout
    except (OSError, EOFError, BrokenPipeError) as exc:
        sys.stdout = old_stdout
        print(f"  SKIPPED: py4dgeo M3C2EP multiprocessing not supported ({exc})")
        return

    valid_ep = np.isfinite(res_ep.distances)
    print(f"  Valid:  {valid_ep.sum()} / {len(res_ep.distances)}")
    print(f"  Mean:   {np.nanmean(res_ep.distances):.6f}  (expected ~{known_dz:+.1f})")
    print(f"  Median: {np.nanmedian(res_ep.distances):.6f}")
    print(f"  Std:    {np.nanstd(res_ep.distances):.6f}")
    print(f"  Min:    {np.nanmin(res_ep.distances):.6f}")
    print(f"  Max:    {np.nanmax(res_ep.distances):.6f}")

    # --- Per-point comparison ---
    print()
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    both_valid = valid_orig & valid_ep
    if both_valid.sum() > 0:
        d_orig = res_orig.distances[both_valid]
        d_ep = res_ep.distances[both_valid]
        corr = np.corrcoef(d_orig, d_ep)[0, 1]
        corr_flip = np.corrcoef(d_orig, -d_ep)[0, 1]
        print(f"  Both valid: {both_valid.sum()}")
        print(f"  corr(orig, ep):   {corr:.6f}")
        print(f"  corr(orig, -ep):  {corr_flip:.6f}")
        print(f"  mean(orig - ep):  {np.mean(d_orig - d_ep):.6f}")
        print(f"  max |orig - ep|:  {np.max(np.abs(d_orig - d_ep)):.6f}")

        # Check sign: does EP have the right sign relative to known change?
        orig_sign_ok = np.mean(d_orig) > 0  # should be positive for +0.5 dz
        ep_sign_ok = np.mean(d_ep) > 0
        print(f"  Original sign correct (mean > 0): {orig_sign_ok}")
        print(f"  EP sign correct (mean > 0):       {ep_sign_ok}")
    else:
        print("  No overlapping valid points!")

    # --- Now test with a small misalignment (simulating ICP scenario) ---
    print()
    print("=" * 60)
    print("M3C2-EP WITH SIMULATED MISALIGNMENT")
    print("=" * 60)
    # Shift T2 by a known amount, then provide the inverse transform
    shift = np.array([0.5, -0.3, 0.1])
    cloud_t2_shifted = cloud_t2 + shift  # misaligned T2

    # Transform that reverses the shift: T = [I | -shift]
    transform = np.eye(4)
    transform[:3, 3] = -shift  # inverse of the shift

    scan_meta_t2_shifted = M3C2Detector.resolve_scan_metadata(
        points=cloud_t2_shifted,
        raw_scan_ids=raw_scan_ids_t2,
        scan_metadata_source="synthetic_from_point_source_id",
        synthetic_origin_height=100.0,
        epoch_label="T2-shifted",
    )

    try:
        sys.stdout = io.StringIO()
        res_ep_shift = M3C2Detector.compute_m3c2_ep(
            core_points=core_points,
            cloud_t1=cloud_t1,
            cloud_t2=cloud_t2_shifted,  # misaligned T2
            params=params,
            transform_matrix=transform,  # ICP transform to align T2 to T1
            cxx=np.zeros((12, 12), dtype=np.float64),
            scan_metadata_t1=scan_meta_t1,
            scan_metadata_t2=scan_meta_t2_shifted,
            reduction_point=np.zeros(3, dtype=np.float64),
            perform_transform=True,
        )
        sys.stdout = old_stdout
    except (OSError, EOFError, BrokenPipeError) as exc:
        sys.stdout = old_stdout
        print(f"  SKIPPED: py4dgeo M3C2EP multiprocessing not supported ({exc})")
        return

    valid_shift = np.isfinite(res_ep_shift.distances)
    print(f"  Valid:  {valid_shift.sum()} / {len(res_ep_shift.distances)}")
    print(f"  Mean:   {np.nanmean(res_ep_shift.distances):.6f}  (expected ~{known_dz:+.1f})")
    print(f"  Median: {np.nanmedian(res_ep_shift.distances):.6f}")
    print(f"  Std:    {np.nanstd(res_ep_shift.distances):.6f}")

    both_valid2 = valid_orig & valid_shift
    if both_valid2.sum() > 0:
        d_orig2 = res_orig.distances[both_valid2]
        d_shift = res_ep_shift.distances[both_valid2]
        corr2 = np.corrcoef(d_orig2, d_shift)[0, 1]
        print(f"  corr(orig, ep_shift): {corr2:.6f}")
        print(f"  mean(orig - ep_shift): {np.mean(d_orig2 - d_shift):.6f}")


if __name__ == "__main__":
    main()
