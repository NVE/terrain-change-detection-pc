"""
Quick GPU vs CPU C2C performance test on real data.

This script loads a subset of your real data and benchmarks C2C with GPU enabled vs disabled.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from terrain_change_detection.utils.config import AppConfig, load_config
from terrain_change_detection.preprocessing.loader import PointCloudLoader
from terrain_change_detection.detection import ChangeDetector
from terrain_change_detection.acceleration.hardware_detection import detect_gpu, get_gpu_info


def load_data(config: AppConfig, max_points: int = None):
    """Load a subset of the real data for benchmarking."""
    base_dir = Path(config.paths.base_dir)
    
    # Load 2015 data
    file_2015 = base_dir / "eksport_1225654_20250602" / "2015" / "data" / "eksport_1225654_416_1.laz"
    file_2020 = base_dir / "eksport_1225654_20250602" / "2020" / "data" / "eksport_1225654_4241_1.laz"
    
    if not file_2015.exists() or not file_2020.exists():
        raise FileNotFoundError(f"Data files not found. Expected:\n  {file_2015}\n  {file_2020}")
    
    loader = PointCloudLoader(
        ground_only=config.preprocessing.ground_only,
        classification_filter=config.preprocessing.classification_filter
    )
    
    if max_points:
        print(f"Loading data (up to {max_points:,} points)...")
    else:
        print(f"Loading full dataset...")
    
    data_2015 = loader.load(file_2015)
    data_2020 = loader.load(file_2020)
    
    points_2015 = data_2015['points']
    points_2020 = data_2020['points']
    
    # Subsample to max_points if specified
    if max_points:
        if len(points_2015) > max_points:
            idx = np.random.choice(len(points_2015), max_points, replace=False)
            points_2015 = points_2015[idx]
        
        if len(points_2020) > max_points:
            idx = np.random.choice(len(points_2020), max_points, replace=False)
            points_2020 = points_2020[idx]
    
    if max_points:
        print(f"  2015: {len(points_2015):,} points")
        print(f"  2020: {len(points_2020):,} points")
    
    return points_2015, points_2020


def benchmark_c2c(ref_points, comp_points, config, label):
    """Benchmark C2C computation."""
    print(f"\n{label}")
    print("-" * 60)
    
    # Run benchmark (source=comp_points, target=ref_points for C2C from comparison to reference)
    start_time = time.time()
    result = ChangeDetector.compute_c2c(
        source=comp_points,
        target=ref_points,
        max_distance=None,
        config=config
    )
    duration = time.time() - start_time
    
    # Check if GPU was used
    gpu_used = result.metadata.get('gpu_used', False) if hasattr(result, 'metadata') else False
    
    print(f"Duration: {duration:.3f} seconds")
    print(f"GPU Used: {gpu_used}")
    print(f"Mean distance: {result.mean:.3f} m")
    print(f"Median distance: {result.median:.3f} m")
    print(f"RMSE: {result.rmse:.3f} m")
    
    return duration, gpu_used


def main():
    """Run GPU vs CPU comparison."""
    print("=" * 80)
    print("GPU vs CPU C2C Performance Test")
    print("=" * 80)
    
    # Check GPU availability
    gpu_info = detect_gpu()
    if gpu_info.available:
        print(f"\n[OK] GPU Available: {gpu_info.device_name}")
        print(f"  Memory: {gpu_info.memory_gb:.2f} GB")
        print(f"  CUDA: {gpu_info.cuda_version}")
    else:
        print(f"\n[X] GPU Not Available - {gpu_info.error_message}")
        print("  Will test CPU fallback only")
    
    # Load configuration
    config_path = Path("config/default.yaml")
    config = load_config(config_path)
    print(f"\nLoaded config from: {config_path}")
    
    # First, load full dataset to check available points
    print("\nChecking available data...")
    try:
        ref_full, comp_full = load_data(config, max_points=None)
        total_ref = len(ref_full)
        total_comp = len(comp_full)
        print(f"Available points:")
        print(f"  Reference (2015): {total_ref:,} points")
        print(f"  Comparison (2020): {total_comp:,} points")
        
        # Define test sizes based on available data
        # Test progressively larger sizes up to the maximum available
        test_sizes = [
            1_000,
            5_000,
            10_000,
            20_000,
            50_000,
            100_000,
            min(200_000, total_comp),
            min(500_000, total_comp),
            min(1_000_000, total_comp),
            total_comp  # Full dataset
        ]
        # Remove duplicates and filter out sizes that are too close
        test_sizes = sorted(set(test_sizes))
        # Only keep sizes that differ by at least 10%
        filtered_sizes = [test_sizes[0]]
        for size in test_sizes[1:]:
            if size > filtered_sizes[-1] * 1.1:  # At least 10% larger
                filtered_sizes.append(size)
        test_sizes = filtered_sizes
        
        print(f"\nWill test with sizes: {[f'{s:,}' for s in test_sizes]}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Run benchmarks
    results = []
    
    for size in test_sizes:
        print(f"\n{'=' * 80}")
        print(f"Testing with {size:,} points per cloud")
        print(f"{'=' * 80}")
        
        # Load data
        try:
            ref_points, comp_points = load_data(config, max_points=size)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
        
        # Test 1: GPU enabled
        config.gpu.enabled = True
        config.gpu.use_for_c2c = True
        duration_gpu, gpu_used = benchmark_c2c(
            ref_points, comp_points, config,
            f"[1/2] C2C with GPU ENABLED ({size:,} points)"
        )
        
        # Test 2: GPU disabled (CPU baseline)
        config.gpu.enabled = False
        duration_cpu, _ = benchmark_c2c(
            ref_points, comp_points, config,
            f"[2/2] C2C with GPU DISABLED ({size:,} points)"
        )
        
        # Calculate speedup
        if gpu_used and duration_cpu > 0:
            speedup = duration_cpu / duration_gpu
            results.append({
                'size': size,
                'gpu_time': duration_gpu,
                'cpu_time': duration_cpu,
                'speedup': speedup
            })
            
            print(f"\n{'=' * 60}")
            print(f"SPEEDUP: {speedup:.2f}x")
            if speedup > 5:
                print(f"*** Excellent GPU performance!")
            elif speedup > 2:
                print(f"** Good GPU speedup")
            elif speedup > 1.2:
                print(f"* Moderate GPU benefit")
            else:
                print(f"! Limited GPU benefit for this size")
            print(f"{'=' * 60}")
        elif not gpu_used:
            print(f"\n! GPU was not used (fell back to CPU)")
    
    # Print comprehensive summary
    print(f"\n{'=' * 80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'=' * 80}")
    
    if results:
        print(f"\n{'Size':<15} {'GPU Time':<15} {'CPU Time':<15} {'Speedup':<15}")
        print("-" * 60)
        for r in results:
            print(f"{r['size']:>13,}  {r['gpu_time']:>13.3f}s  {r['cpu_time']:>13.3f}s  {r['speedup']:>13.2f}x")
        
        # Find best speedup
        best = max(results, key=lambda x: x['speedup'])
        print(f"\nBest speedup: {best['speedup']:.2f}x at {best['size']:,} points")
        
        # Calculate average speedup (excluding small sizes with overhead)
        large_results = [r for r in results if r['size'] >= 10000]
        if large_results:
            avg_speedup = sum(r['speedup'] for r in large_results) / len(large_results)
            print(f"Average speedup (10K+ points): {avg_speedup:.2f}x")
    
    print(f"\n{'=' * 80}")
    print("[OK] Comprehensive performance test complete!")
    print("\nKey findings:")
    if results:
        if any(r['speedup'] > 5 for r in results):
            print("- *** GPU shows excellent performance on larger datasets")
        print(f"- GPU benefits scale with dataset size")
        print(f"- Maximum speedup: {max(r['speedup'] for r in results):.2f}x")
        
        # Find optimal size (if any show speedup > 2)
        good_results = [r for r in results if r['speedup'] > 2]
        if good_results:
            print(f"- GPU optimal for {min(r['size'] for r in good_results):,}+ points")
        else:
            print("- Note: GPU showing no significant speedup - may need investigation")
    print("- Full results saved above for analysis")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main()
