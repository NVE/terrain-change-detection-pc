"""
Data exploration script.

THis script helps explore and validate the structure of your point cloud data.
Use this to check what data is available and verify it can be loaded correctly.
"""

import sys
from pathlib import Path

# Add the src to the path to import modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from terrain_change_detection.preprocessing import DataDiscovery, BatchLoader
from terrain_change_detection.preprocessing import PointCloudLoader 
from terrain_change_detection.utils.logging import setup_logger

def main():
    """
    Explores a data structure and validates the point cloud data.
    """
    logger = setup_logger(__name__)

    print("Point Cloud Data Explorer")
    print("=========================")

    # Set base directory
    base_dir = Path(__file__).parent.parent / "data" / "raw"
    print(f"Scanning directory: {base_dir}")

    if not base_dir.exists():
        print(f"Base directory {base_dir} does not exist.")
        return
    
    # Discover data
    discovery = DataDiscovery(base_dir)
    areas = discovery.scan_areas()

    if not areas:
        print("No data found!")
        print(
            "Expected data structure:\n"
            "data/raw/\n"
            "├── area1/\n"
            "│   ├── timeperiod1/\n"
            "│   │   ├── data/\n"
            "│   │   │   ├── file1.laz\n"
            "│   │   │   ├── file2.laz\n"
            "│   │   │   └── ...\n"
            "│   │   └── metadata/\n"
            "│   └── timeperiod2/\n"
            "│       ├── data/\n"
            "│       │   ├── file1.laz\n"
            "│       │   ├── file2.laz\n"
            "│       │   └── ...\n"
            "│       └── metadata/"
        )
        return
    
    # Display discovered areas/area directories
    print(f"\n{'='*60}")
    print(f"Data Discovery Summary")
    print(f"{'='*60}")
    print(f"Found {len(areas)} area(s):\n")

    total_files = 0
    total_ground_points = 0

    for area_name, area_info in areas.items():
        print(f"Area: {area_name}")
        print(f"  Time periods: {len(area_info.time_periods)}")
        for time_period in area_info.time_periods:
            dataset = area_info.datasets[time_period]
            print(f"    ├─ Time period: {time_period}")
            print(f"    │   Number of files: {len(dataset.laz_files)}")
            print(f"    │   Total ground points: {dataset.total_points}")
            print(f"    │   Bounds: {dataset.bounds}")
            print(f"    │   Metadata dir: {dataset.metadata_dir}")
            if dataset.per_file_stats:
                print(f"    │   Example file stats:")
                for stat in dataset.per_file_stats[:1]:
                    print(f"    │     - File: {stat['file']}")
                    print(f"    │       Num points: {stat['num_points']}")
                    print(f"    │       Ground points: {stat['ground_points']}")
                    print(f"    │       Bounds: {stat['bounds']}")
            else:
                print(f"    │   No per-file stats available.")
            print(f"    │   Example files:")
            for i, laz_file in enumerate(dataset.laz_files[:5]):
                file_size_mb = laz_file.stat().st_size / (1024 * 1024)
                print(f"    │     - {laz_file.name} ({file_size_mb:.1f} MB)")
            total_files += len(dataset.laz_files)
            total_ground_points += dataset.total_points
        print()

    print(f"{'='*60}")    
    print(f"Total files: {total_files}")
    print(f"Total ground points (sum across all datasets): {total_ground_points}")
    print(f"{'='*60}\n")

    # test loading a sample point cloud
    print("Sample Point Cloud Loading Test")
    print("-"*40)
    test_area_name = list(areas.keys())[0]
    test_time_period = list(areas[test_area_name].time_periods)[0]
    test_dataset = areas[test_area_name].datasets[test_time_period]

    print(f"Testing: {test_area_name}/{test_time_period}")

    try:
        batch_loader = BatchLoader()
        first_file = test_dataset.laz_files[0]
        print(f"  Loading file: {first_file.name}")
        loader = PointCloudLoader()
        if loader.validate_file(str(first_file)):
            print(f"  File validation successful: {first_file.name}")
            metadata = loader.get_metadata(str(first_file))
            print(f"  Metadata for {first_file.name}:")
            for key, value in metadata.items():
                print(f"    {key}: {value}")
        else:
            print(f"  File validation failed: {first_file.name}")
    except Exception as e:
        print(f"  Error loading point cloud: {e}")

    # Show potential change detection analyses
    print("\nChange Detection Opportunities")
    print("-"*40)
    for area_name, area_info in areas.items():
        if len(area_info.time_periods) >= 2:
            periods = area_info.time_periods
            for i in range(len(periods) - 1):
                for j in range(i + 1, len(periods)):
                    dataset1 = area_info.datasets[periods[i]]
                    dataset2 = area_info.datasets[periods[j]]
                    print(f"  {area_name}: {periods[i]} vs {periods[j]} "
                          f"({dataset1.total_points} vs {dataset2.total_points} ground points)")
        else:
            print(f"  {area_name} has {len(area_info.time_periods)} time period(s). ")
            print("  Two or more time periods are required for change detection analysis.")
    print(f"\nTo run change detection, use:")
    print(f"  uv run scripts/run_workflow.py")


if __name__ == "__main__":
    main()



