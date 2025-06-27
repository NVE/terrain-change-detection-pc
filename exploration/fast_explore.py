#!/usr/bin/env python3
"""
High-performance script runner for LAZ file analysis
Usage: uv run fast_explore.py [file_path]
"""

import sys
import os
from pathlib import Path

def main():
    # Set numpy to use all available cores
    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
    os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
    os.environ['NUMEXPR_NUM_THREADS'] = str(os.cpu_count())
    
    # Import after setting environment variables
    import laspy
    import numpy as np
    import time
    
    # Get file path from command line or use default
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    else:
        file_path = Path("33-1-466-136-13.laz")
    
    # Start timing
    start_time = time.time()
    
    print(f"Loading file: {file_path}")
    if not file_path.exists():
        print(f"Error: File {file_path} not found!")
        return 1
    
    # Load with optimized settings
    las_file = laspy.read(file_path)
    
    print(f"✓ File loaded successfully!")
    print(f"📊 Point format: {las_file.header.point_format}")
    print(f"📈 Number of points: {len(las_file.points):,}")
    
    # Quick analysis with vectorized operations
    print(f"\n📍 Coordinate ranges:")
    print(f"   X: {las_file.x.min():.2f} to {las_file.x.max():.2f}")
    print(f"   Y: {las_file.y.min():.2f} to {las_file.y.max():.2f}")
    print(f"   Z: {las_file.z.min():.2f} to {las_file.z.max():.2f}")
    
    # Fast classification analysis
    if hasattr(las_file, 'classification'):
        unique_classes, counts = np.unique(las_file.classification, return_counts=True)
        print(f"\n🏷️  Classifications found: {len(unique_classes)} types")
        
        # Show top 5 most common classes
        sorted_indices = np.argsort(counts)[::-1]
        print("   Top 5 most common:")
        for i in sorted_indices[:5]:
            class_code = unique_classes[i]
            count = counts[i]
            percentage = (count / len(las_file.classification)) * 100
            print(f"     Class {class_code}: {count:,} points ({percentage:.1f}%)")
    
    # Performance summary
    end_time = time.time()
    execution_time = end_time - start_time
    points_per_second = len(las_file.points) / execution_time
    
    print(f"\n⚡ Performance:")
    print(f"   Execution time: {execution_time:.3f} seconds")
    print(f"   Processing rate: {points_per_second:,.0f} points/second")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
