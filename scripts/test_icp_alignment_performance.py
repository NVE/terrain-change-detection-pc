"""
Quick ICP alignment performance test (CPU vs GPU).

This script loads a subset of your real data and benchmarks ICP alignment
with GPU acceleration enabled vs disabled (for the ICP nearest-neighbor
search), using the same subsampled points and coarse initialization.

Usage (from repo root):
    uv run scripts/test_icp_alignment_performance.py

Optional flags:
    --config PATH       YAML config path (default: config/default.yaml)
    --points N          Points per cloud for alignment (default: alignment.subsample_size)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from terrain_change_detection.utils.config import AppConfig, load_config
from terrain_change_detection.preprocessing.loader import PointCloudLoader
from terrain_change_detection.alignment.coarse_registration import CoarseRegistration
from terrain_change_detection.alignment.fine_registration import ICPRegistration
from terrain_change_detection.acceleration.hardware_detection import detect_gpu


def load_data(config: AppConfig, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a subset of the real data for ICP benchmarking.

    Uses the same Norwegian dataset files as other GPU benchmarks.
    """
    base_dir = Path(config.paths.base_dir)

    file_2015 = base_dir / "eksport_1225654_20250602" / "2015" / "data" / "eksport_1225654_416_1.laz"
    file_2020 = base_dir / "eksport_1225654_20250602" / "2020" / "data" / "eksport_1225654_4241_1.laz"

    if not file_2015.exists() or not file_2020.exists():
        raise FileNotFoundError(f"Data files not found. Expected:\n  {file_2015}\n  {file_2020}")

    loader = PointCloudLoader(
        ground_only=config.preprocessing.ground_only,
        classification_filter=config.preprocessing.classification_filter,
    )

    print(f"Loading data for ICP benchmark (up to {max_points:,} points per cloud)...")
    data_2015 = loader.load(file_2015)
    data_2020 = loader.load(file_2020)

    pts1 = data_2015["points"]
    pts2 = data_2020["points"]

    # Subsample to max_points if needed
    if len(pts1) > max_points:
        idx = np.random.choice(len(pts1), max_points, replace=False)
        pts1 = pts1[idx]
    if len(pts2) > max_points:
        idx = np.random.choice(len(pts2), max_points, replace=False)
        pts2 = pts2[idx]

    print(f"  2015: {len(pts1):,} points")
    print(f"  2020: {len(pts2):,} points")

    return pts1, pts2


def run_icp_once(
    src: np.ndarray,
    tgt: np.ndarray,
    config: AppConfig,
    initial_transform: np.ndarray | None,
    use_gpu: bool,
    label: str,
) -> tuple[float, float, str]:
    """
    Run a single ICP alignment and report duration, RMSE, and backend.
    """
    print(f"\n{label}")
    print("-" * 60)

    icp = ICPRegistration(
        max_iterations=config.alignment.max_iterations,
        tolerance=config.alignment.tolerance,
        max_correspondence_distance=config.alignment.max_correspondence_distance,
        use_gpu=use_gpu,
        convergence_translation_epsilon=config.alignment.convergence_translation_epsilon,
        convergence_rotation_epsilon_deg=config.alignment.convergence_rotation_epsilon_deg,
    )

    t0 = time.time()
    _, _, final_err = icp.align_point_clouds(
        source=src,
        target=tgt,
        initial_transform=initial_transform,
    )
    t1 = time.time()

    backend = getattr(icp, "_last_nn_backend", "unknown")

    print(f"Duration: {t1 - t0:.3f} seconds")
    print(f"Final RMSE: {final_err:.6f} m")
    print(f"NN backend: {backend}")

    return t1 - t0, float(final_err), backend


def main() -> None:
    parser = argparse.ArgumentParser(description="ICP alignment CPU vs GPU benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML configuration file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=None,
        help="Number of points per cloud for alignment (default: alignment.subsample_size)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("ICP Alignment CPU vs GPU Performance Test")
    print("=" * 80)

    # Check GPU availability
    gpu_info = detect_gpu()
    if gpu_info.available:
        print(f"\n[OK] GPU Available: {gpu_info.device_name}")
        print(f"  Memory: {gpu_info.memory_gb:.2f} GB")
        print(f"  CUDA: {gpu_info.cuda_version}")
    else:
        print(f"\n[X] GPU Not Available - {gpu_info.error_message}")
        print("  GPU runs will fall back to CPU backend")

    # Load configuration
    cfg_path = Path(args.config)
    config = load_config(cfg_path)
    print(f"\nLoaded config from: {cfg_path}")

    # Determine alignment sample size
    n_points = args.points or config.alignment.subsample_size
    print(f"\nUsing {n_points:,} points per cloud for ICP alignment.")

    # Reproducible subsampling
    np.random.seed(42)

    # Load data and subsample
    try:
        pts_ref, pts_cmp = load_data(config, max_points=n_points)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Optional coarse registration to mimic workflow
    initial_T = np.eye(4)
    if getattr(config.alignment, "coarse", None) and config.alignment.coarse.enabled:
        coarse_cfg = config.alignment.coarse
        coarse = CoarseRegistration(
            method=coarse_cfg.method,
            voxel_size=coarse_cfg.voxel_size,
            phase_grid_cell=coarse_cfg.phase_grid_cell,
        )
        print("\nComputing coarse initial transform (method=%s)..." % coarse_cfg.method)
        t0 = time.time()
        initial_T = coarse.compute_initial_transform(pts_cmp, pts_ref)
        t1 = time.time()
        print(f"Coarse registration time: {t1 - t0:.3f} seconds")

    # Benchmark CPU ICP
    cpu_time, cpu_rmse, cpu_backend = run_icp_once(
        src=pts_cmp,
        tgt=pts_ref,
        config=config,
        initial_transform=initial_T,
        use_gpu=False,
        label="[1/2] ICP with GPU DISABLED",
    )

    # Benchmark GPU ICP (may still fall back to CPU backend internally)
    gpu_time, gpu_rmse, gpu_backend = run_icp_once(
        src=pts_cmp,
        tgt=pts_ref,
        config=config,
        initial_transform=initial_T,
        use_gpu=True,
        label="[2/2] ICP with GPU ENABLED",
    )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"CPU   : {cpu_time:.3f} s, RMSE={cpu_rmse:.6f} m, backend={cpu_backend}")
    print(f"GPU   : {gpu_time:.3f} s, RMSE={gpu_rmse:.6f} m, backend={gpu_backend}")

    if gpu_time > 0:
        speedup = cpu_time / gpu_time
        print(f"\nSpeedup (CPU / GPU): {speedup:.2f}x")
    else:
        print("\nSpeedup: N/A (GPU time is zero)")

    print("\nNote: If the NN backend for the GPU run is 'sklearn-cpu' or 'cpu-fallback',")
    print("      no true GPU acceleration is being used (only the wrapper).")


if __name__ == "__main__":
    main()

