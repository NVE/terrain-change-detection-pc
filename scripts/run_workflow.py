"""
Example script for complete terrain change detection workflow

This script demonstrates the full workflow from data discovery to change detection.
"""

import sys
import numpy as np
from pathlib import Path

# Add the src to the path to import modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from terrain_change_detection.preprocessing.loader import PointCloudLoader
from terrain_change_detection.preprocessing.data_discovery import DataDiscovery, BatchLoader
from terrain_change_detection.alignment.fine_registration import ICPRegistration
from terrain_change_detection.utils.logging import setup_logger

# Hardware optimizations
# TO DO: Implement hardware optimizations for large datasets

def main():
    """
    Main function to run the terrain change detection workflow.
    """
    logger = setup_logger(__name__)

    print("Terrain Change Detection Workflow")
    print("=================================")

    # Load configuration
    # TO DO: Load configuration from a file or command line arguments

    # Example data paths
    base_dir = Path(__file__).parent.parent / "data" / "raw"
    # base_dir = Path(__file__).parent.parent / "tests" / "test_preprocessing" / "sample_data" / "raw"
    if not base_dir.exists():
        logger.error(f"Base directory {base_dir} does not exist.")
        return
    
    # Discover data
    data_discovery = DataDiscovery(base_dir)
    areas = data_discovery.scan_areas()

    if not areas:
        logger.error("No area/area directory found in the base directory.")
        return
    
    # Find the first area with at least two time periods
    selected_area = None
    for area_name, area_info in areas.items():
        if len(area_info.time_periods) >= 2:
            selected_area = area_info
            break
    
    if not selected_area:
        logger.error("Could not find an area with at least two time periods for change detection.")
        return

    # Select the first two time periods
    t1, t2 = selected_area.time_periods[:2]
    ds1 = selected_area.datasets[t1]
    ds2 = selected_area.datasets[t2]

    logger.info(f"Selected area: {selected_area.area_name}")
    logger.info(f"Time period 1: {t1} ({len(ds1.laz_files)} files)")
    logger.info(f"Time period 2: {t2} ({len(ds2.laz_files)} files)")

    try:
        # Step 1: Load Data
        logger.info("--- Step 1: Loading point cloud data ---")
        batch_loader = BatchLoader()
        if len(ds1.laz_files) > 1:
            logger.info(f"Batch loading {len(ds1.laz_files)} files for time period {t1}...")
            pc1_data = batch_loader.load_dataset(ds1)
        else:
            logger.info(f"Loading single file for time period {t1}...")
            pc1_data = batch_loader.loader.load(str(ds1.laz_files[0]))

        if len(ds2.laz_files) > 1:
            logger.info(f"Batch loading {len(ds2.laz_files)} files for time period {t2}...")
            pc2_data = batch_loader.load_dataset(ds2)
        else:
            logger.info(f"Loading single file for time period {t2}...")
            pc2_data = batch_loader.loader.load(str(ds2.laz_files[0]))

        logger.info(f"Dataset 1 ({t1}): {pc1_data['points'].shape[0]} points")
        logger.info(f"Dataset 2 ({t2}): {pc2_data['points'].shape[0]} points")

        points1 = pc1_data['points']
        points2 = pc2_data['points']

        # Step 2: ICP Registration
        logger.info("--- Step 2: Performing spatial alignment... ---")
        icp = ICPRegistration(
            max_iterations=100,
            tolerance=1e-6,
            max_correspondence_distance=1.0
        )

        # Subsample for alignment if datasets are large
        if len(points1) > 50000:
            indices1 = np.random.choice(len(points1), 20000, replace=False)
            points1_subsampled = points1[indices1]
        else:
            points1_subsampled = points1

        if len(points2) > 50000:
            indices2 = np.random.choice(len(points2), 20000, replace=False)
            points2_subsampled = points2[indices2]
        else:
            points2_subsampled = points2

        # Perform ICP alignment
        points2_subsampled_aligned, transform_matrix, final_error = icp.align_point_clouds(
            source=points2_subsampled,
            target=points1_subsampled
        )        

        # Apply the transformation to the original points2
        if len(points2) > 50000:
            points2_full_aligned = icp.apply_transformation(points2, transform_matrix)
        else:
            points2_full_aligned = points2_subsampled_aligned

        # Compute the registration error
        alignment_error = icp.compute_registration_error(
            source=points2_full_aligned,
            target=points1
        )

        print(f"ICP Alignment completed with final error: {alignment_error:.6f}")

        # Step 3: Change Detection
        logger.info("--- Step 3: Detecting terrain changes... ---")
        # TO DO: Implement change detection

        # Step 4: Visualization
        logger.info("--- Step 4: Visualizing results... ---")
        # TO DO: Implement visualization of results

    except Exception as e:
        logger.error(f"Change detection workflow failed: {e}")
        return
    
if __name__ == "__main__":
    main()



