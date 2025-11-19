"""
GPU vs CPU Performance Benchmark for DoD (DEM of Difference)

Tests grid accumulation performance at various scales to validate
GPU acceleration benefits.
"""

import time
import numpy as np
from pathlib import Path

from terrain_change_detection.acceleration import GridAccumulator, Bounds2D
from terrain_change_detection.acceleration.hardware_detection import detect_gpu


def benchmark_grid_accumulation(n_points: int, grid_size: float, cell_size: float, n_runs: int = 3):
    """
    Benchmark grid accumulation with GPU vs CPU.
    
    Args:
        n_points: Number of points to accumulate
        grid_size: Size of the grid (square)
        cell_size: Cell size for gridding
        n_runs: Number of runs for timing
    
    Returns:
        Dict with timing results
    """
    # Generate random points
    np.random.seed(42)
    points = np.random.rand(n_points, 3) * grid_size
    
    bounds = Bounds2D(min_x=0, min_y=0, max_x=grid_size, max_y=grid_size)
    
    # CPU benchmark
    cpu_times = []
    for _ in range(n_runs):
        acc_cpu = GridAccumulator(bounds, cell_size, use_gpu=False)
        t0 = time.perf_counter()
        acc_cpu.accumulate(points)
        dem_cpu = acc_cpu.finalize()
        t1 = time.perf_counter()
        cpu_times.append(t1 - t0)
    
    cpu_time = np.mean(cpu_times)
    cpu_std = np.std(cpu_times)
    
    # GPU benchmark
    gpu_times = []
    gpu_available = False
    for _ in range(n_runs):
        acc_gpu = GridAccumulator(bounds, cell_size, use_gpu=True)
        t0 = time.perf_counter()
        acc_gpu.accumulate(points)
        dem_gpu = acc_gpu.finalize()
        t1 = time.perf_counter()
        gpu_times.append(t1 - t0)
        
        # Check if GPU was actually used
        if hasattr(acc_gpu, '_backend') and acc_gpu._backend is not None:
            gpu_available = acc_gpu._backend.is_gpu
    
    gpu_time = np.mean(gpu_times)
    gpu_std = np.std(gpu_times)
    
    speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0
    
    # Verify numerical parity
    acc_cpu_verify = GridAccumulator(bounds, cell_size, use_gpu=False)
    acc_cpu_verify.accumulate(points)
    dem_cpu_verify = acc_cpu_verify.finalize()
    
    acc_gpu_verify = GridAccumulator(bounds, cell_size, use_gpu=True)
    acc_gpu_verify.accumulate(points)
    dem_gpu_verify = acc_gpu_verify.finalize()
    
    # Check if results match
    parity_ok = np.allclose(dem_cpu_verify, dem_gpu_verify, rtol=1e-5, equal_nan=True)
    
    return {
        'n_points': n_points,
        'grid_size': grid_size,
        'cell_size': cell_size,
        'grid_cells': int((grid_size / cell_size) ** 2),
        'cpu_time': cpu_time,
        'cpu_std': cpu_std,
        'gpu_time': gpu_time,
        'gpu_std': gpu_std,
        'speedup': speedup,
        'gpu_available': gpu_available,
        'parity_ok': parity_ok,
    }


def benchmark_chunked_accumulation(
    total_points: int, chunk_size: int, grid_size: float, cell_size: float
):
    """
    Benchmark chunked accumulation (simulating streaming).
    
    Args:
        total_points: Total number of points
        chunk_size: Points per chunk
        grid_size: Size of the grid
        cell_size: Cell size for gridding
    
    Returns:
        Dict with timing results
    """
    # Generate all points
    np.random.seed(42)
    all_points = np.random.rand(total_points, 3) * grid_size
    
    bounds = Bounds2D(min_x=0, min_y=0, max_x=grid_size, max_y=grid_size)
    
    # CPU chunked accumulation
    acc_cpu = GridAccumulator(bounds, cell_size, use_gpu=False)
    t0_cpu = time.perf_counter()
    for i in range(0, total_points, chunk_size):
        chunk = all_points[i:i+chunk_size]
        acc_cpu.accumulate(chunk)
    dem_cpu = acc_cpu.finalize()
    t1_cpu = time.perf_counter()
    cpu_time = t1_cpu - t0_cpu
    
    # GPU chunked accumulation
    acc_gpu = GridAccumulator(bounds, cell_size, use_gpu=True)
    t0_gpu = time.perf_counter()
    for i in range(0, total_points, chunk_size):
        chunk = all_points[i:i+chunk_size]
        acc_gpu.accumulate(chunk)
    dem_gpu = acc_gpu.finalize()
    t1_gpu = time.perf_counter()
    gpu_time = t1_gpu - t0_gpu
    
    speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0
    parity_ok = np.allclose(dem_cpu, dem_gpu, rtol=1e-5, equal_nan=True)
    
    n_chunks = (total_points + chunk_size - 1) // chunk_size
    
    return {
        'total_points': total_points,
        'chunk_size': chunk_size,
        'n_chunks': n_chunks,
        'cpu_time': cpu_time,
        'gpu_time': gpu_time,
        'speedup': speedup,
        'parity_ok': parity_ok,
    }


def main():
    """Run comprehensive DoD GPU benchmarks."""
    print("=" * 80)
    print("DoD GPU Acceleration Performance Benchmark")
    print("=" * 80)
    print()
    
    # Check GPU availability
    gpu_info = detect_gpu()
    if gpu_info.available:
        print(f"✓ GPU Detected: {gpu_info.device_name}")
        print(f"  Memory: {gpu_info.memory_gb:.2f} GB")
        print(f"  CUDA Version: {gpu_info.cuda_version}")
        print()
    else:
        print("✗ No GPU detected - results will show CPU fallback behavior")
        if gpu_info.error:
            print(f"  Error: {gpu_info.error}")
        print()
    
    # Test configurations: (n_points, grid_size, cell_size, description)
    test_configs = [
        (1_000, 100, 1.0, "Small (1K points, 100x100 grid)"),
        (10_000, 100, 1.0, "Medium (10K points, 100x100 grid)"),
        (100_000, 100, 1.0, "Large (100K points, 100x100 grid)"),
        (1_000_000, 100, 1.0, "Very Large (1M points, 100x100 grid)"),
        (100_000, 1000, 2.0, "Large Area (100K points, 1000x1000 grid)"),
        (1_000_000, 1000, 2.0, "Massive (1M points, 1000x1000 grid)"),
    ]
    
    print("=" * 80)
    print("1. Basic Grid Accumulation Benchmark")
    print("=" * 80)
    print()
    
    results = []
    for n_points, grid_size, cell_size, description in test_configs:
        print(f"Testing: {description}")
        print(f"  Points: {n_points:,}, Grid: {grid_size}x{grid_size}m, Cell: {cell_size}m")
        
        try:
            result = benchmark_grid_accumulation(n_points, grid_size, cell_size, n_runs=3)
            results.append(result)
            
            print(f"  CPU: {result['cpu_time']:.3f}s (±{result['cpu_std']:.3f}s)")
            print(f"  GPU: {result['gpu_time']:.3f}s (±{result['gpu_std']:.3f}s)")
            print(f"  Speedup: {result['speedup']:.2f}x")
            print(f"  GPU Used: {'✓' if result['gpu_available'] else '✗'}")
            print(f"  Parity: {'✓' if result['parity_ok'] else '✗ MISMATCH!'}")
            
            if not result['parity_ok']:
                print("  ⚠ WARNING: GPU and CPU results do not match!")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        print()
    
    # Summary table
    if results:
        print("=" * 80)
        print("Summary Table")
        print("=" * 80)
        print()
        print(f"{'Points':<12} {'Grid Cells':<12} {'CPU Time':<12} {'GPU Time':<12} {'Speedup':<10} {'Status'}")
        print("-" * 80)
        
        for r in results:
            status = "✓" if r['parity_ok'] and r['gpu_available'] else "○"
            if not r['parity_ok']:
                status = "✗"
            
            print(
                f"{r['n_points']:<12,} {r['grid_cells']:<12,} "
                f"{r['cpu_time']:<12.3f} {r['gpu_time']:<12.3f} "
                f"{r['speedup']:<10.2f} {status}"
            )
        print()
    
    # Chunked accumulation tests
    print("=" * 80)
    print("2. Chunked Accumulation Benchmark (Streaming Simulation)")
    print("=" * 80)
    print()
    
    chunked_configs = [
        (100_000, 10_000, 100, 1.0, "100K points, 10K chunks"),
        (1_000_000, 50_000, 100, 1.0, "1M points, 50K chunks"),
        (1_000_000, 100_000, 1000, 2.0, "1M points, 100K chunks, large grid"),
    ]
    
    for total_points, chunk_size, grid_size, cell_size, description in chunked_configs:
        print(f"Testing: {description}")
        
        try:
            result = benchmark_chunked_accumulation(
                total_points, chunk_size, grid_size, cell_size
            )
            
            print(f"  Total Points: {result['total_points']:,}")
            print(f"  Chunks: {result['n_chunks']}")
            print(f"  CPU: {result['cpu_time']:.3f}s")
            print(f"  GPU: {result['gpu_time']:.3f}s")
            print(f"  Speedup: {result['speedup']:.2f}x")
            print(f"  Parity: {'✓' if result['parity_ok'] else '✗'}")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        print()
    
    # Performance insights
    print("=" * 80)
    print("Performance Insights")
    print("=" * 80)
    print()
    
    if results:
        avg_speedup = np.mean([r['speedup'] for r in results if r['parity_ok']])
        max_speedup = max([r['speedup'] for r in results if r['parity_ok']])
        
        if gpu_info.available:
            print(f"Average Speedup: {avg_speedup:.2f}x")
            print(f"Maximum Speedup: {max_speedup:.2f}x")
            print()
            
            if avg_speedup < 1.5:
                print("⚠ GPU acceleration appears limited on this hardware/dataset size.")
                print("  Possible reasons:")
                print("  - Dataset too small (GPU overhead dominates)")
                print("  - Platform limitation (Windows sklearn-gpu uses CPU KDTree)")
                print("  - Insufficient GPU memory (falling back to CPU)")
            elif avg_speedup < 3.0:
                print("○ Moderate GPU acceleration observed.")
                print("  - Benefits increase with larger datasets")
                print("  - Consider testing with production-scale data")
            else:
                print("✓ Strong GPU acceleration observed!")
                print(f"  - {avg_speedup:.1f}x average speedup demonstrates clear benefit")
                print("  - GPU acceleration is working as expected")
        else:
            print("No GPU available - all operations ran on CPU.")
            print("Install GPU dependencies to enable acceleration:")
            print("  uv pip install -e '.[gpu]'")
    
    print()
    print("=" * 80)
    print("Benchmark Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
