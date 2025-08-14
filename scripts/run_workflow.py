"""
Example script for complete terrain change detection workflow

This script demonstrates the full workflow from data discovery to change detection.
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add the src to the path to import modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from terrain_change_detection.preprocessing.loader import PointCloudLoader
from terrain_change_detection.preprocessing.data_discovery import DataDiscovery, BatchLoader
from terrain_change_detection.alignment.fine_registration import ICPRegistration
from terrain_change_detection.utils.logging import setup_logger
from terrain_change_detection.detection import ChangeDetector, M3C2Params, autotune_m3c2_params
from terrain_change_detection.visualization.point_cloud import PointCloudVisualizer

# Hardware optimizations
# TO DO: Implement hardware optimizations for large datasets

# Tuning knobs (independent):
# - ICP_SAMPLE_SIZE: subsample size for ICP alignment
# - M3C2_CORE_POINTS: number of core points for M3C2
# - C2C_MAX_POINTS: maximum points per cloud for C2C distances
# - VIS_SAMPLE_SIZE: sample size for visualization
ICP_SAMPLE_SIZE = 50000
M3C2_CORE_POINTS = 10000
C2C_MAX_POINTS = 10000
VIS_SAMPLE_SIZE = 50000

def main():
    """
    Main function to run the terrain change detection workflow.
    """
    logger = setup_logger(__name__)

    print("Terrain Change Detection Workflow")
    print("=================================")

    # Load configuration
    # TO DO: Load configuration from a file or command line arguments

    # CLI: allow overriding the data root (default to data/raw)
    parser = argparse.ArgumentParser(description="Terrain Change Detection Workflow")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "raw"),
        help="Base directory containing area folders (e.g., data/raw or data/synthetic)",
    )
    # No explicit CLI args for M3C2 tuning; parameters are auto-tuned from data density
    args, unknown = parser.parse_known_args()

    base_dir = Path(args["base_dir"] if isinstance(args, dict) else args.base_dir)
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

        # Instantiate the visualizer (choose backend)
        VIS_BACKEND = 'plotly'   # 'plotly', 'pyvista', or 'pyvistaqt'
        visualizer = PointCloudVisualizer(backend=VIS_BACKEND)

        # Visualize the original point clouds
        logger.info("--- Visualizing original point clouds ---")
        visualizer.visualize_clouds(
            point_clouds=[points1, points2],
            names=[f"PC from {t1}", f"PC from {t2}"],
            sample_size=VIS_SAMPLE_SIZE  # Downsample for visualization
        )

        # Step 2: ICP Registration
        logger.info("--- Step 2: Performing spatial alignment... ---")
        icp = ICPRegistration(
            max_iterations=100,
            tolerance=1e-6,
            max_correspondence_distance=1.0
        )

        # Subsample for alignment if datasets are large
        if len(points1) > ICP_SAMPLE_SIZE:
            n1 = min(len(points1), ICP_SAMPLE_SIZE)
            indices1 = np.random.choice(len(points1), n1, replace=False)
            points1_subsampled = points1[indices1]
        else:
            points1_subsampled = points1

        if len(points2) > ICP_SAMPLE_SIZE:
            n2 = min(len(points2), ICP_SAMPLE_SIZE)
            indices2 = np.random.choice(len(points2), n2, replace=False)
            points2_subsampled = points2[indices2]
        else:
            points2_subsampled = points2

        # Perform ICP alignment
        points2_subsampled_aligned, transform_matrix, final_error = icp.align_point_clouds(
            source=points2_subsampled,
            target=points1_subsampled
        )        

        # Apply the transformation to the original points2
        if len(points2) > ICP_SAMPLE_SIZE:
            points2_full_aligned = icp.apply_transformation(points2, transform_matrix)
        else:
            points2_full_aligned = points2_subsampled_aligned

        # Compute the registration error
        alignment_error = icp.compute_registration_error(
            source=points2_full_aligned,
            target=points1
        )

        print(f"ICP Alignment completed with final error: {alignment_error:.6f}")

        # Visualize the aligned point clouds
        logger.info("--- Visualizing aligned point clouds ---")
        visualizer.visualize_clouds(
            point_clouds=[points1, points2_full_aligned],
            names=[f"PC from {t1} (Target)", f"PC from {t2} (Aligned)"],
            sample_size=VIS_SAMPLE_SIZE  # Downsample for visualization
        )

        # Step 3: Change Detection + on-the-spot visualization
        logger.info("--- Step 3: Detecting terrain changes... ---")

        # 3a) DEM of Difference (DoD)
        try:
            logger.info("Computing DEM of Difference (DoD)...")
            dod_res = ChangeDetector.compute_dod(
                points_t1=points1,
                points_t2=points2_full_aligned,
                cell_size=1.0,
                aggregator="mean",
            )
            logger.info(
                "DoD stats: n_cells=%d, mean=%.4f m, median=%.4f m, rmse=%.4f m, min=%.4f m, max=%.4f m",
                dod_res.stats.get("n_cells", 0),
                dod_res.stats.get("mean_change", float('nan')),
                dod_res.stats.get("median_change", float('nan')),
                dod_res.stats.get("rmse", float('nan')),
                dod_res.stats.get("min_change", float('nan')),
                dod_res.stats.get("max_change", float('nan')),
            )
            # Visualize DoD immediately
            visualizer.visualize_dod_heatmap(dod_res, title="DEM of Difference (m)")
        except Exception as e:
            logger.error(f"DoD computation failed: {e}")

        # 3b) Cloud-to-Cloud (C2C)
        try:
            logger.info("Computing Cloud-to-Cloud (C2C) distances (downsampled for speed)...")
            # Downsample to keep pairwise search manageable if sklearn is unavailable
            max_points = C2C_MAX_POINTS
            src = points2_full_aligned
            tgt = points1
            if len(src) > max_points:
                idx = np.random.choice(len(src), max_points, replace=False)
                src = src[idx]
            if len(tgt) > max_points:
                idx = np.random.choice(len(tgt), max_points, replace=False)
                tgt = tgt[idx]

            c2c_res = ChangeDetector.compute_c2c(src, tgt)
            logger.info(
                "C2C stats: n=%d, mean=%.4f m, median=%.4f m, rmse=%.4f m",
                c2c_res.n,
                c2c_res.mean,
                c2c_res.median,
                c2c_res.rmse,
            )
            # Visualize C2C histogram immediately
            visualizer.visualize_distance_histogram(c2c_res.distances, title="C2C distances (m)", bins=60)
        except Exception as e:
            logger.error(f"C2C computation failed: {e}")

        # 3c) M3C2 (Original)
        try:
            logger.info("Computing M3C2 (Original) distances on core points (downsampled)...")
            # Generate core points by uniform subsampling of target (T1)
            max_core = M3C2_CORE_POINTS
            core_src = points1
            if len(core_src) > max_core:
                idx = np.random.choice(len(core_src), max_core, replace=False)
                core_src = core_src[idx]

            # Auto-tune M3C2 parameters based on point density
            m3c2_params = autotune_m3c2_params(points1)
            logger.info(
                "M3C2 auto-tuned params: proj_scale=%.2f, cyl_radius=%.2f, max_depth=%.2f, min_neighbors=%d",
                m3c2_params.projection_scale,
                m3c2_params.cylinder_radius,
                m3c2_params.max_depth,
                m3c2_params.min_neighbors,
            )

            m3c2_res = ChangeDetector.compute_m3c2_original(
                core_points=core_src,
                cloud_t1=points1,
                cloud_t2=points2_full_aligned,
                params=m3c2_params,
            )
            logger.info(
                "M3C2 stats: n=%d, mean=%.4f m, median=%.4f m, std=%.4f m",
                m3c2_res.distances.size,
                float(np.mean(m3c2_res.distances)),
                float(np.median(m3c2_res.distances)),
                float(np.std(m3c2_res.distances)),
            )
            # Visualize M3C2 core points immediately
            visualizer.visualize_m3c2_corepoints(m3c2_res.core_points, m3c2_res.distances, sample_size=VIS_SAMPLE_SIZE, title="M3C2 distances (m)")
        except Exception as e:
            logger.error(f"M3C2 computation failed: {e}")

        # 3d) M3C2 with Error Propagation (EP)
        try:
            logger.info("Computing M3C2 with Error Propagation (EP) and significance flags...")
            import platform
            workers = 1 if platform.system().lower().startswith('win') else 4
            m3c2_ep = ChangeDetector.compute_m3c2_error_propagation(
                core_points=core_src,
                cloud_t1=points1,
                cloud_t2=points2_full_aligned,
                params=m3c2_params,
                workers=workers,
            )
            sig_count = int(np.sum(m3c2_ep.significant)) if m3c2_ep.significant is not None else 0
            logger.info(
                "M3C2-EP: significant=%d of %d (%.1f%%)",
                sig_count,
                m3c2_ep.distances.size,
                100.0 * sig_count / max(1, m3c2_ep.distances.size),
            )
            # Optional: visualize EP distributions
            visualizer.visualize_distance_histogram(m3c2_ep.distances, title="M3C2-EP distances (m)", bins=60)
            if m3c2_ep.significant is not None:
                visualizer.visualize_distance_histogram(m3c2_ep.distances[m3c2_ep.significant], title="M3C2-EP distances (significant)", bins=60)
        except Exception as e:
            logger.error(f"M3C2-EP computation failed: {e}")

    except Exception as e:
        logger.error(f"Change detection workflow failed: {e}")
        return
    
if __name__ == "__main__":
    main()



