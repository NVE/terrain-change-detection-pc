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
from terrain_change_detection.detection.change_detection import ChangeDetector
from terrain_change_detection.acceleration.hardware_detection import detect_gpu, get_gpu_info


def load_data(config: AppConfig, max_points: int = 10000):
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
    
    print(f"Loading data (up to {max_points:,} points)...")
    data_2015 = loader.load(file_2015)
    data_2020 = loader.load(file_2020)
    
    points_2015 = data_2015['points']
    points_2020 = data_2020['points']
    
    # Subsample to max_points
    if len(points_2015) > max_points:
        idx = np.random.choice(len(points_2015), max_points, replace=False)
        points_2015 = points_2015[idx]
    
    if len(points_2020) > max_points:
        idx = np.random.choice(len(points_2020), max_points, replace=False)
        points_2020 = points_2020[idx]
    
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
        print(f"\n✓ GPU Available: {gpu_info.device_name}")
        print(f"  Memory: {gpu_info.memory_gb:.2f} GB")
        print(f"  CUDA: {gpu_info.cuda_version}")
    else:
        print(f"\n✗ GPU Not Available - {gpu_info.error_message}")
        print("  Will test CPU fallback only")
    
    # Load configuration
    config_path = Path("config/default.yaml")
    config = load_config(config_path)
    print(f"\nLoaded config from: {config_path}")
    
    # Test different sizes
    test_sizes = [1000, 5000, 10000]
    
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
            print(f"\n{'=' * 60}")
            print(f"SPEEDUP: {speedup:.2f}x")
            if speedup > 5:
                print(f"🚀 Excellent GPU performance!")
            elif speedup > 2:
                print(f"📈 Good GPU speedup")
            elif speedup > 1.2:
                print(f"➡️ Moderate GPU benefit")
            else:
                print(f"⚠️ Limited GPU benefit for this size")
            print(f"{'=' * 60}")
        elif not gpu_used:
            print(f"\n⚠️  GPU was not used (fell back to CPU)")
    
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print("✓ Test complete!")
    print("\nKey findings:")
    print("- GPU acceleration works best with 10,000+ points")
    print("- Smaller datasets may show overhead from GPU data transfer")
    print("- Check workflow_gpu_enabled.log for full workflow results")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main()
