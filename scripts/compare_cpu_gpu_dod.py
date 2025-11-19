"""
Compare CPU vs GPU DoD Performance

Benchmarks DEM of Difference (DoD) computation using CPU vs GPU
based on the default configuration. Disables C2C and M3C2 to focus
on DoD performance only.

This script loads data in-memory and runs DoD with CPU vs GPU to measure
the performance difference. Useful for validating GPU acceleration benefits.
"""

import time
import sys
import argparse
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from terrain_change_detection.utils.config import load_config, AppConfig
from terrain_change_detection.detection import ChangeDetector
from terrain_change_detection.preprocessing.data_discovery import DataDiscovery, BatchLoader
from terrain_change_detection.preprocessing.loader import PointCloudLoader
from terrain_change_detection.acceleration import get_gpu_info


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_real_data(config: AppConfig, max_points_per_file: int = None):
    """
    Load real point cloud data from the configured data directory.
    
    Args:
        config: Configuration object
        max_points_per_file: Maximum points to load per file (for testing)
    
    Returns:
        Tuple of (points_t1, points_t2, dataset_info)
    """
    print("\n📂 Loading point cloud data...")
    
    # Setup loader with preprocessing config
    loader = PointCloudLoader(
        ground_only=config.preprocessing.ground_only,
        classification_filter=config.preprocessing.classification_filter,
    )
    
    # Discover available datasets
    discovery = DataDiscovery(
        base_dir=str(config.paths.base_dir),
        data_dir_name=config.discovery.data_dir_name,
        metadata_dir_name=config.discovery.metadata_dir_name,
        loader=loader,
    )
    areas = discovery.scan_areas()
    
    if not areas:
        raise ValueError(
            f"No datasets found in {config.paths.base_dir}. "
            f"Please check the data directory."
        )
    
    # Find first area with at least 2 time periods
    selected_area = None
    for area_name, area_info in areas.items():
        if len(area_info.time_periods) >= 2:
            selected_area = area_info
            break
    
    if not selected_area:
        raise ValueError("No area found with at least 2 time periods")
    
    # Get first two time periods
    t1, t2 = selected_area.time_periods[:2]
    ds1 = selected_area.datasets[t1]
    ds2 = selected_area.datasets[t2]
    
    print(f"   Area: {selected_area.area_name}")
    print(f"   Time periods: {t1}, {t2}")
    
    # Load datasets
    batch_loader = BatchLoader(loader=loader)
    
    print(f"\n   Loading T1 ({t1})...")
    if len(ds1.laz_files) > 1:
        pc1_data = batch_loader.load_dataset(ds1, max_points_per_file=max_points_per_file)
    else:
        pc1_data = loader.load(str(ds1.laz_files[0]))
    
    print(f"   Loading T2 ({t2})...")
    if len(ds2.laz_files) > 1:
        pc2_data = batch_loader.load_dataset(ds2, max_points_per_file=max_points_per_file)
    else:
        pc2_data = loader.load(str(ds2.laz_files[0]))
    
    points_t1 = pc1_data['points']
    points_t2 = pc2_data['points']
    
    print(f"\n   ✓ T1: {len(points_t1):,} points")
    print(f"   ✓ T2: {len(points_t2):,} points")
    
    dataset_info = {
        'area': selected_area.area_name,
        'period_1': t1,
        'period_2': t2,
        'n_files_t1': len(ds1.laz_files),
        'n_files_t2': len(ds2.laz_files),
    }
    
    return points_t1, points_t2, dataset_info


def benchmark_dod(points_t1, points_t2, config: AppConfig, use_gpu: bool, n_runs: int = 3):
    """
    Benchmark DoD computation with given configuration.
    
    Args:
        points_t1: Time 1 point cloud
        points_t2: Time 2 point cloud
        config: Configuration object
        use_gpu: Whether to use GPU acceleration
        n_runs: Number of runs for timing
    
    Returns:
        Dictionary with timing and result information
    """
    cell_size = config.detection.dod.cell_size
    
    times = []
    results = []
    
    for i in range(n_runs):
        t0 = time.perf_counter()
        
        result = ChangeDetector.compute_dod(
            points_t1=points_t1,
            points_t2=points_t2,
            cell_size=cell_size,
            aggregator=config.detection.dod.aggregator,
            config=config,
        )
        
        t1 = time.perf_counter()
        elapsed = t1 - t0
        times.append(elapsed)
        results.append(result)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    # Get statistics from last result
    last_result = results[-1]
    
    return {
        'times': times,
        'avg_time': avg_time,
        'std_time': std_time,
        'min_time': np.min(times),
        'max_time': np.max(times),
        'result': last_result,
        'use_gpu': use_gpu,
    }


def verify_parity(cpu_result, gpu_result, rtol=1e-5):
    """
    Verify that CPU and GPU results match numerically.
    
    Args:
        cpu_result: Result from CPU computation
        gpu_result: Result from GPU computation
        rtol: Relative tolerance for comparison
    
    Returns:
        Tuple of (matches, max_difference)
    """
    # Compare DoD grids
    dod_cpu = cpu_result.dod
    dod_gpu = gpu_result.dod
    
    # Check for NaN parity
    nan_cpu = np.isnan(dod_cpu)
    nan_gpu = np.isnan(dod_gpu)
    
    if not np.array_equal(nan_cpu, nan_gpu):
        return False, np.inf
    
    # Compare non-NaN values
    valid_mask = ~nan_cpu
    if not np.any(valid_mask):
        return True, 0.0
    
    cpu_valid = dod_cpu[valid_mask]
    gpu_valid = dod_gpu[valid_mask]
    
    try:
        np.testing.assert_allclose(gpu_valid, cpu_valid, rtol=rtol)
        max_diff = np.max(np.abs(cpu_valid - gpu_valid))
        return True, max_diff
    except AssertionError:
        max_diff = np.max(np.abs(cpu_valid - gpu_valid))
        return False, max_diff


def main():
    """Main comparison script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Compare CPU vs GPU DoD Performance")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (defaults to config/default.yaml)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Override base directory containing area folders",
    )
    parser.add_argument(
        "--max-points-per-file",
        type=int,
        default=None,
        help="Maximum points to load per file (for testing with smaller datasets)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=3,
        help="Number of benchmark runs for each method (default: 3)",
    )
    args = parser.parse_args()
    
    print_section("DoD CPU vs GPU Performance Comparison")
    
    # Check GPU availability
    gpu_info = get_gpu_info()
    if gpu_info.available:
        print(f"\n✓ GPU Detected: {gpu_info.device_name}")
        print(f"  Memory: {gpu_info.memory_gb:.2f} GB")
        print(f"  CUDA Version: {gpu_info.cuda_version}")
    else:
        print(f"\n✗ No GPU detected: {gpu_info.error_message}")
        print("  GPU benchmarks will fall back to CPU")
    
    # Load configuration
    print("\n📋 Loading configuration from config/default.yaml...")
    config = load_config(args.config)
    
    # Override base_dir if provided
    if args.base_dir:
        config.paths.base_dir = args.base_dir
    
    # Verify DoD is enabled
    if not config.detection.dod.enabled:
        print("\n⚠ DoD is disabled in config. Enabling for this test...")
        config.detection.dod.enabled = True
    
    # Disable other methods
    config.detection.c2c.enabled = False
    config.detection.m3c2.enabled = False
    
    print(f"\n   DoD Configuration:")
    print(f"   - Cell Size: {config.detection.dod.cell_size}m")
    print(f"   - Aggregator: {config.detection.dod.aggregator}")
    print(f"   - GPU Enabled: {config.gpu.enabled}")
    print(f"   - GPU for Preprocessing: {config.gpu.use_for_preprocessing}")
    
    # Load data
    try:
        # Load data using configured max_points_per_file
        points_t1, points_t2, dataset_info = load_real_data(
            config,
            max_points_per_file=args.max_points_per_file
        )
    except Exception as e:
        print(f"\n✗ Error loading data: {e}")
        print("\nℹ If no real data is available, you can run the synthetic data generator:")
        print("  uv run python scripts/generate_synthetic_laz.py")
        return 1
    
    print(f"\n   Dataset Info:")
    print(f"   - Area: {dataset_info['area']}")
    print(f"   - T1 ({dataset_info['period_1']}): {dataset_info['n_files_t1']} files")
    print(f"   - T2 ({dataset_info['period_2']}): {dataset_info['n_files_t2']} files")
    
    # Calculate grid dimensions
    bounds = (
        min(points_t1[:, 0].min(), points_t2[:, 0].min()),
        min(points_t1[:, 1].min(), points_t2[:, 1].min()),
        max(points_t1[:, 0].max(), points_t2[:, 0].max()),
        max(points_t1[:, 1].max(), points_t2[:, 1].max()),
    )
    
    nx = int(np.ceil((bounds[2] - bounds[0]) / config.detection.dod.cell_size)) + 1
    ny = int(np.ceil((bounds[3] - bounds[1]) / config.detection.dod.cell_size)) + 1
    n_cells = nx * ny
    
    print(f"\n   Grid Dimensions:")
    print(f"   - Bounds: ({bounds[0]:.1f}, {bounds[1]:.1f}) to ({bounds[2]:.1f}, {bounds[3]:.1f})")
    print(f"   - Grid: {nx} × {ny} = {n_cells:,} cells")
    
    # Benchmark CPU
    print_section("1. CPU DoD Benchmark")
    config_cpu = load_config(args.config)
    if args.base_dir:
        config_cpu.paths.base_dir = args.base_dir
    config_cpu.detection.dod.enabled = True
    config_cpu.detection.c2c.enabled = False
    config_cpu.detection.m3c2.enabled = False
    config_cpu.gpu.enabled = False  # Disable GPU
    
    print(f"\n🔄 Running CPU benchmark ({args.n_runs} runs)...")
    cpu_benchmark = benchmark_dod(points_t1, points_t2, config_cpu, use_gpu=False, n_runs=args.n_runs)
    
    print(f"\n   Results:")
    print(f"   - Average Time: {cpu_benchmark['avg_time']:.3f}s (±{cpu_benchmark['std_time']:.3f}s)")
    print(f"   - Min Time: {cpu_benchmark['min_time']:.3f}s")
    print(f"   - Max Time: {cpu_benchmark['max_time']:.3f}s")
    print(f"   - Valid Cells: {cpu_benchmark['result'].stats['n_cells']:,} / {n_cells:,}")
    print(f"   - Mean Change: {cpu_benchmark['result'].stats['mean_change']:.3f}m")
    print(f"   - RMSE: {cpu_benchmark['result'].stats['rmse']:.3f}m")
    
    # Benchmark GPU
    print_section("2. GPU DoD Benchmark")
    config_gpu = load_config(args.config)
    if args.base_dir:
        config_gpu.paths.base_dir = args.base_dir
    config_gpu.detection.dod.enabled = True
    config_gpu.detection.c2c.enabled = False
    config_gpu.detection.m3c2.enabled = False
    config_gpu.gpu.enabled = True  # Enable GPU
    config_gpu.gpu.use_for_preprocessing = True
    
    print(f"\n🔄 Running GPU benchmark ({args.n_runs} runs)...")
    gpu_benchmark = benchmark_dod(points_t1, points_t2, config_gpu, use_gpu=True, n_runs=args.n_runs)
    
    print(f"\n   Results:")
    print(f"   - Average Time: {gpu_benchmark['avg_time']:.3f}s (±{gpu_benchmark['std_time']:.3f}s)")
    print(f"   - Min Time: {gpu_benchmark['min_time']:.3f}s")
    print(f"   - Max Time: {gpu_benchmark['max_time']:.3f}s")
    print(f"   - Valid Cells: {gpu_benchmark['result'].stats['n_cells']:,} / {n_cells:,}")
    print(f"   - Mean Change: {gpu_benchmark['result'].stats['mean_change']:.3f}m")
    print(f"   - RMSE: {gpu_benchmark['result'].stats['rmse']:.3f}m")
    
    # Verify numerical parity
    print_section("3. Numerical Parity Verification")
    print("\n🔍 Comparing CPU and GPU results...")
    
    matches, max_diff = verify_parity(cpu_benchmark['result'], gpu_benchmark['result'])
    
    if matches:
        print(f"\n   ✓ Results Match!")
        print(f"   - Maximum Difference: {max_diff:.2e}")
        print(f"   - Relative Tolerance: 1e-5")
    else:
        print(f"\n   ✗ Results Do Not Match!")
        print(f"   - Maximum Difference: {max_diff:.2e}")
        print(f"   - This may indicate a GPU implementation issue")
    
    # Performance comparison
    print_section("4. Performance Summary")
    
    speedup = cpu_benchmark['avg_time'] / gpu_benchmark['avg_time']
    
    print(f"\n   Dataset Characteristics:")
    print(f"   - T1 Points: {len(points_t1):,}")
    print(f"   - T2 Points: {len(points_t2):,}")
    print(f"   - Total Points: {len(points_t1) + len(points_t2):,}")
    print(f"   - Grid Cells: {n_cells:,}")
    print(f"   - Cell Size: {config.detection.dod.cell_size}m")
    
    print(f"\n   Timing Comparison:")
    print(f"   - CPU Time: {cpu_benchmark['avg_time']:.3f}s")
    print(f"   - GPU Time: {gpu_benchmark['avg_time']:.3f}s")
    print(f"   - Speedup: {speedup:.2f}x")
    
    print(f"\n   Performance Assessment:")
    if speedup > 2.0:
        print(f"   ✓ Excellent GPU acceleration ({speedup:.1f}x speedup)")
    elif speedup > 1.3:
        print(f"   ○ Good GPU acceleration ({speedup:.1f}x speedup)")
    elif speedup > 0.8:
        print(f"   ~ Marginal benefit ({speedup:.1f}x)")
    else:
        print(f"   ⚠ CPU faster than GPU ({speedup:.1f}x)")
        print(f"      This is expected for small datasets due to GPU overhead")
    
    # Detailed statistics comparison
    print(f"\n   Statistics Comparison:")
    print(f"   {'Metric':<20} {'CPU':>15} {'GPU':>15} {'Difference':>15}")
    print(f"   {'-'*20} {'-'*15} {'-'*15} {'-'*15}")
    
    stats_keys = ['mean_change', 'median_change', 'rmse', 'min_change', 'max_change']
    for key in stats_keys:
        cpu_val = cpu_benchmark['result'].stats[key]
        gpu_val = gpu_benchmark['result'].stats[key]
        diff = gpu_val - cpu_val
        print(f"   {key:<20} {cpu_val:>15.6f} {gpu_val:>15.6f} {diff:>15.6e}")
    
    # Recommendations
    print_section("5. Recommendations")
    
    if speedup > 1.5:
        print("\n   ✓ GPU acceleration is beneficial for this dataset size")
        print("   - Recommend enabling GPU in production configuration")
        print("   - Expected speedup will increase with larger datasets")
    elif speedup > 0.8:
        print("\n   ○ GPU provides marginal benefit at this scale")
        print("   - Consider GPU for larger datasets (10M+ points)")
        print("   - CPU processing is acceptable for this data size")
    else:
        print("\n   ⚠ GPU overhead dominates at this scale")
        print("   - Keep GPU disabled for datasets of this size")
        print("   - GPU benefits emerge at 100K+ points per epoch")
    
    print(f"\n   Configuration Settings:")
    print(f"   - For this data size, set gpu.use_for_preprocessing = {speedup > 1.2}")
    print(f"   - Consider out-of-core tiling for datasets > 10M points")
    print(f"   - Use CPU parallelization (parallel.enabled) for multi-core systems")
    
    print("\n" + "=" * 80)
    print("Comparison Complete")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
