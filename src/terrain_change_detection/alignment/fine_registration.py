"""
ICP Registration Implementation

This module implements the Iterative Closest Point (ICP) algorithm for
spatial alignment of multi-temporal point cloud datasets.
"""

from typing import Optional, Tuple
import time

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class ICPRegistration:
    """
    Implementation of ICP algorithm for point cloud registration.

    The ICP algorithm iteratively:
    1. Finds closest point correspondences
    2. Estimates optimal transformation (rotation + translation)
    3. Applies transformation to source points
    4. Repeats until convergence
    """

    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        max_correspondence_distance: float = 1.0,
        use_gpu: bool = False,
        convergence_translation_epsilon: float = 1e-4,
        convergence_rotation_epsilon_deg: float = 0.1,
    ):
        """
        Initialize ICP parameters.

        Args:
            max_iterations: Maximum number of ICP iterations.
            tolerance: Convergence tolerance on change in mean squared error.
            max_correspondence_distance: Maximum distance for point correspondences.
            use_gpu: If True, attempt to use GPU-accelerated nearest neighbors
                (via GPUNearestNeighbors) with automatic CPU fallback.
            convergence_translation_epsilon: Minimum translation step (meters) below
                which the algorithm is considered converged.
            convergence_rotation_epsilon_deg: Minimum rotation step (degrees) below
                which the algorithm is considered converged.
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.max_correspondence_distance = max_correspondence_distance
        self.use_gpu = use_gpu
        self.convergence_translation_epsilon = convergence_translation_epsilon
        # Store rotation epsilon in radians for internal use
        self.convergence_rotation_epsilon_rad = np.deg2rad(convergence_rotation_epsilon_deg)
        # Runtime metadata (e.g., which NN backend was used)
        self._last_nn_backend: str = "none"

    def align_point_clouds(
        self,
        source: np.ndarray,
        target: np.ndarray,
        initial_transform: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Align source point cloud to target using ICP.

        Args:
            source: Source point cloud (N x 3).
            target: Target point cloud (M x 3).
            initial_transform: Initial transformation matrix (4 x 4) or None.

        Returns:
            Tuple of (aligned_source_points, transformation_matrix, final_error).
        """
        n_src = len(source)
        n_tgt = len(target)
        logger.info(
            "Starting ICP alignment with %d source points and %d target points.",
            n_src,
            n_tgt,
        )

        if n_src == 0 or n_tgt == 0:
            logger.warning(
                "ICP called with empty source or target (source=%d, target=%d); "
                "returning identity (or initial) transform and infinite error.",
                n_src,
                n_tgt,
            )
            if initial_transform is None:
                transform = np.eye(4)
            else:
                transform = initial_transform.copy()
            # No alignment possible; preserve input source array
            return source.copy(), transform, float("inf")

        # Initialize transformation
        if initial_transform is None:
            transform = np.eye(4)
        else:
            transform = initial_transform.copy()

        current_source = source.copy()
        previous_error = float("inf")

        # Build the nearest-neighbor search structure for the target point cloud ONCE.
        # Optionally use GPU-accelerated neighbors with automatic CPU fallback.
        build_start = time.time()
        nbrs = None
        nn_backend = "cpu"

        if self.use_gpu:
            try:
                from ..acceleration.gpu_neighbors import create_gpu_neighbors  # type: ignore

                nbrs_gpu = create_gpu_neighbors(
                    n_neighbors=1,
                    algorithm="auto",
                    metric="euclidean",
                    use_gpu=True,
                )
                # Fit on a float32 copy for better numerical behavior on GPU
                target_f32 = np.asarray(target, dtype=np.float32)
                nbrs_gpu.fit(target_f32)

                nn_backend = getattr(nbrs_gpu, "backend_", "unknown")
                logger.info("ICP using GPUNearestNeighbors backend: %s", nn_backend)

                # If the GPU wrapper ended up on a pure CPU backend, treat it as CPU
                # but still use the same interface.
                nbrs = nbrs_gpu
            except Exception as e:
                logger.warning(
                    "ICP GPU neighbor initialization failed (%s); falling back to CPU.",
                    e,
                )
                nbrs = None
                nn_backend = "cpu-fallback"

        if nbrs is None:
            logger.debug("Building CPU KD-Tree for target point cloud...")
            nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(target)
            nn_backend = "cpu"

        build_end = time.time()
        # Record backend for observability / benchmarking
        self._last_nn_backend = nn_backend
        logger.debug(
            "Nearest-neighbor structure built in %.4f s (backend=%s).",
            build_end - build_start,
            nn_backend,
        )

        iter_durations = []
        icp_start = time.time()
        n_iterations = 0

        for iteration in range(self.max_iterations):
            iter_start = time.time()

            # Use the pre-built neighbor structure to find correspondences.
            correspondences, distances = self.find_correspondences(
                source=current_source,
                target=target,
                nbrs=nbrs,
            )

            # Basic sanity checks for GPU distances; fall back to CPU on obvious issues.
            if self.use_gpu:
                if not np.all(np.isfinite(distances)):
                    logger.warning(
                        "ICP GPU neighbors produced non-finite distances; "
                        "restarting ICP with CPU KD-Tree."
                    )
                    # Restart using pure CPU KD-Tree
                    self.use_gpu = False
                    return self.align_point_clouds(
                        source=source,
                        target=target,
                        initial_transform=initial_transform,
                    )
                try:
                    max_dist = float(np.max(distances))
                except Exception:
                    max_dist = float("inf")
                if max_dist > 1e5:
                    logger.warning(
                        "ICP GPU neighbors produced implausibly large distances (max=%.3e m); "
                        "restarting ICP with CPU KD-Tree.",
                        max_dist,
                    )
                    self.use_gpu = False
                    return self.align_point_clouds(
                        source=source,
                        target=target,
                        initial_transform=initial_transform,
                    )

            # Filter out correspondences that exceed the max distance
            valid_mask = distances < self.max_correspondence_distance
            if np.sum(valid_mask) < 3:  # Need at least 3 points to define a plane
                logger.warning("Not enough valid correspondences found. Stopping ICP.")
                break

            valid_source = current_source[valid_mask]
            valid_target = target[correspondences[valid_mask]]

            # Estimate incremental transformation
            delta_transform = self.estimate_transformation(valid_source, valid_target)

            # Update transformation: new_transform = delta_transform * current_transform
            transform = delta_transform @ transform

            # Apply the cumulative transformation to the ORIGINAL source cloud
            # This prevents compounding floating point errors from repeated transforms
            current_source = self.apply_transformation(source, transform)

            # Compute the error (mean squared error)
            current_error = float(np.mean(distances[valid_mask] ** 2))

            # Compute incremental motion metrics for convergence check
            delta_t = delta_transform[:3, 3]
            delta_R = delta_transform[:3, :3]
            trans_step = float(np.linalg.norm(delta_t))
            # Clamp argument to arccos to valid range to avoid NaNs
            trace_delta_R = float(np.trace(delta_R))
            cos_theta = max(min((trace_delta_R - 1.0) * 0.5, 1.0), -1.0)
            rot_step = float(np.arccos(cos_theta))

            logger.debug(
                "Iteration %d: MSE=%.6f, |Δt|=%.6e m, Δθ=%.6e rad",
                iteration + 1,
                current_error,
                trans_step,
                rot_step,
            )

            iter_end = time.time()
            iter_durations.append(iter_end - iter_start)
            n_iterations = iteration + 1

            # Check for convergence (both error change and motion magnitude)
            if abs(previous_error - current_error) < self.tolerance:
                logger.info(
                    "ICP converged after %d iterations (MSE change < %.3e).",
                    n_iterations,
                    self.tolerance,
                )
                break

            if (
                trans_step < self.convergence_translation_epsilon
                and rot_step < self.convergence_rotation_epsilon_rad
            ):
                logger.info(
                    "ICP converged after %d iterations (motion below thresholds: "
                    "|Δt|=%.3e m, Δθ=%.3e rad).",
                    n_iterations,
                    trans_step,
                    rot_step,
                )
                break

            previous_error = current_error
        else:
            logger.info("ICP did not converge after %d iterations.", self.max_iterations)

        icp_end = time.time()

        # Pass the pre-built tree to the final error calculation
        final_error = self.compute_registration_error(current_source, target, nbrs)

        total_time = icp_end - icp_start
        mean_iter_time = (sum(iter_durations) / len(iter_durations)) if iter_durations else 0.0
        logger.info(
            "ICP finished in %.4f s (%d iterations, mean iter %.4f s). Final RMSE: %.6f",
            total_time,
            n_iterations,
            mean_iter_time,
            final_error,
        )

        return current_source, transform, final_error

    def find_correspondences(
        self,
        source: np.ndarray,
        target: Optional[np.ndarray] = None,
        nbrs: Optional[NearestNeighbors] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find closest point correspondences between source and target.

        Args:
            source: Source point cloud (N x 3).
            target: Target point cloud (M x 3). Only used if `nbrs` is None,
                in which case a KD-tree is built on this array.
            nbrs: Optional pre-built NearestNeighbors instance for the target.

        Returns:
            Tuple of (correspondence_indices, distances).
        """
        if nbrs is None:
            if target is None:
                raise ValueError("Either 'target' or a pre-built 'nbrs' must be provided.")
            nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(target)

        # Some GPU backends may be fitted on float32 targets; convert queries
        # to float32 when appropriate to avoid unnecessary casting inside.
        try:
            backend = getattr(nbrs, "backend_", "cpu")
        except Exception:
            backend = "cpu"

        query = (
            np.asarray(source, dtype=np.float32)
            if backend in ("cuml", "sklearn-gpu")
            else source
        )

        distances, indices = nbrs.kneighbors(query)

        correspondence_indices = indices.ravel()
        distances = distances.ravel()

        return correspondence_indices, distances

    def estimate_transformation(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate optimal rigid transformation between source and target point clouds.

        Args:
            source_points: Source point cloud points (N x 3).
            target_points: Corresponding target point cloud points (N x 3).

        Returns:
            Transformation matrix (4 x 4).
        """
        # Center the point sets
        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)

        source_centered = source_points - source_centroid
        target_centered = target_points - target_centroid

        # Compute cross-covariance matrix
        H = source_centered.T @ target_centered

        # Singular Value Decomposition
        U, _, Vt = np.linalg.svd(H)

        # Compute rotation matrix
        R = Vt.T @ U.T

        # Ensure proper rotation (det(R) should be 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = target_centroid - R @ source_centroid

        # Construct the transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t

        return transform

    def apply_transformation(
        self,
        points: np.ndarray,
        transform: np.ndarray,
    ) -> np.ndarray:
        """
        Apply a transformation matrix to a set of points.

        Args:
            points: Point cloud (N x 3).
            transform: Transformation matrix (4 x 4).

        Returns:
            Transformed point cloud (N x 3).
        """
        if points.size == 0:
            return points

        # Use direct affine transform in 3D (faster and less memory than homogeneous coords).
        R = transform[:3, :3]
        t = transform[:3, 3]
        return points @ R.T + t

    def compute_registration_error(
        self,
        source: np.ndarray,
        target: np.ndarray,
        nbrs: Optional[NearestNeighbors] = None,
    ) -> float:
        """
        Compute the registration error (RMSE) between aligned source and target point clouds.

        Args:
            source: Aligned source point cloud.
            target: Target point cloud.
            nbrs: Optional pre-built NearestNeighbors instance for target point cloud.

        Returns:
            Registration error as RMSE.
        """
        if source.size == 0 or target.size == 0:
            logger.warning(
                "compute_registration_error called with empty source or target "
                "(source=%d, target=%d); returning infinite error.",
                len(source),
                len(target),
            )
            return float("inf")
        # Find correspondences
        correspondences, distances = self.find_correspondences(source, target, nbrs)

        # Filter valid correspondences
        valid_mask = distances < self.max_correspondence_distance

        if np.sum(valid_mask) == 0:
            logger.warning("No valid correspondences found for error computation.")
            return float("inf")

        # Compute RMSE
        rmse = float(np.sqrt(np.mean(distances[valid_mask] ** 2)))

        return rmse
