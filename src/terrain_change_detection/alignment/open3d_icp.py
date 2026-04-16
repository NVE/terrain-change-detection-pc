"""
Open3D ICP Backend

Wraps Open3D's ``registration_icp`` (point-to-point) to provide the same
interface as :class:`ICPRegistration`.  Open3D is an optional dependency;
this module raises ``ImportError`` at class instantiation if it is absent.
"""

from typing import Optional, Tuple

import numpy as np

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class Open3DICP:
    """Open3D-based ICP registration with the same interface as ICPRegistration."""

    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        max_correspondence_distance: float = 1.0,
        convergence_translation_epsilon: float = 1e-4,
        convergence_rotation_epsilon_deg: float = 0.1,
    ):
        import open3d  # noqa: F401 – fail fast if missing

        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.max_correspondence_distance = max_correspondence_distance
        self.convergence_translation_epsilon = convergence_translation_epsilon
        self.convergence_rotation_epsilon_deg = convergence_rotation_epsilon_deg

    def align_point_clouds(
        self,
        source: np.ndarray,
        target: np.ndarray,
        initial_transform: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Align *source* to *target* using Open3D point-to-point ICP.

        Returns:
            Tuple of (aligned_source_points, transformation_matrix, final_rmse).
        """
        import open3d as o3d

        n_src = len(source)
        n_tgt = len(target)
        logger.info(
            "Starting Open3D ICP alignment with %d source points and %d target points.",
            n_src,
            n_tgt,
        )

        if n_src == 0 or n_tgt == 0:
            logger.warning(
                "Open3D ICP called with empty input (source=%d, target=%d); "
                "returning identity/initial transform.",
                n_src,
                n_tgt,
            )
            T = np.eye(4) if initial_transform is None else initial_transform.copy()
            return source.copy(), T, float("inf")

        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(
            np.ascontiguousarray(source, dtype=np.float64)
        )

        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(
            np.ascontiguousarray(target, dtype=np.float64)
        )

        init = initial_transform if initial_transform is not None else np.eye(4)

        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=self.max_iterations,
            relative_fitness=self.tolerance,
            relative_rmse=self.tolerance,
        )

        result = o3d.pipelines.registration.registration_icp(
            src_pcd,
            tgt_pcd,
            max_correspondence_distance=self.max_correspondence_distance,
            init=init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=criteria,
        )

        transform = np.asarray(result.transformation)
        aligned_source = self.apply_transformation(source, transform)
        rmse = float(result.inlier_rmse)

        logger.info(
            "Open3D ICP finished: %d iterations, RMSE=%.6f, fitness=%.4f",
            result.fitness,  # fraction of inlier correspondences
            rmse,
            result.fitness,
        )

        return aligned_source, transform, rmse

    # ------------------------------------------------------------------
    # Helpers matching ICPRegistration interface
    # ------------------------------------------------------------------

    @staticmethod
    def apply_transformation(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        R = transform[:3, :3]
        t = transform[:3, 3]
        return points @ R.T + t

    def compute_registration_error(
        self,
        source: np.ndarray,
        target: np.ndarray,
        nbrs=None,
    ) -> float:
        if source.size == 0 or target.size == 0:
            return float("inf")
        from sklearn.neighbors import NearestNeighbors

        if nbrs is None:
            nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(target)
        distances, _ = nbrs.kneighbors(source)
        distances = distances.ravel()
        valid = distances < self.max_correspondence_distance
        if not np.any(valid):
            return float("inf")
        return float(np.sqrt(np.mean(distances[valid] ** 2)))
