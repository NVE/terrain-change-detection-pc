"""
ICP Registration Implementation

This module implements the Iterative Closest Point (ICP) algorithm for
spatial alignment of multi-temporal point cloud datasets.
"""

import numpy as np
from typing import Tuple, Optional, Dict
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

    def __init__(self,
                 max_iterations: int = 100,
                 tolerance: float = 1e-6,
                 max_correspondence_distance: float = 1.0):
        """
        Initialize ICP parameters.

        Args:
            max_iterations: Maximum number of ICP iterations
            tolerance: Convergence tolerance
            max_correspondence_distance: Maximum distance for point correspondences
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.max_correspondence_distance = max_correspondence_distance

    def align_point_clouds(self,
                          source: np.ndarray,
                          target: np.ndarray,
                          initial_transform: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Align source point cloud to target using ICP.

        Args:
            source: Source point cloud (N x 3)
            target: Target point cloud (M x 3)
            initial_transform: Initial transformation matrix (4 x 4) or None

        Returns:
            Tuple of (aligned_source_points, transformation_matrix, final_error)
        """
        logger.info(f"Starting ICP alignment with {len(source)} source points and {len(target)} target points.")
        
        # Initialize transformation
        if initial_transform is None:
            transform = np.eye(4)
        else:
            transform = initial_transform.copy()

        current_source = source.copy()
        previous_error = float('inf')

        # Build the KD-Tree for the target point cloud ONCE (!important for performance)
        logger.debug("Building KD-Tree for target point cloud...")
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target)
        logger.debug("KD-Tree built successfully.")

        for iteration in range(self.max_iterations):
            # Use the pre-built KD-Tree to find correspondences.
            correspondences, distances = self.find_correspondences(current_source, nbrs)

            # Filter out correspondences that exceed the max distance
            valid_mask = distances < self.max_correspondence_distance
            if np.sum(valid_mask) < 3: # Need at least 3 points to define a plane
                logger.warning("Not enough valid correspondences found. Stopping ICP.")
                break

            valid_source = current_source[valid_mask]
            valid_target = target[correspondences[valid_mask]]

            # Estimate transformation
            delta_transform = self.estimate_transformation(valid_source, valid_target)

            # Update transformation: new_transform = delta_transform * current_transform
            transform = delta_transform @ transform

            # Apply the CUMULATIVE transformation to the ORIGINAL source cloud
            # This prevents compounding floating point errors from repeated transformations
            current_source = self.apply_transformation(source, transform)

            # Compute the error (mean squared error)
            current_error = np.mean(distances[valid_mask] ** 2)

            logger.debug(f"Iteration {iteration + 1}: MSE = {current_error:.6f}")
            # Check for convergence
            if abs(previous_error - current_error) < self.tolerance:
                logger.info(f"ICP converged after {iteration + 1} iterations.")
                break

            previous_error = current_error
        else:
            logger.info(f"ICP did not converge after {self.max_iterations} iterations.")

        # Pass the pre-built tree to the final error calculation
        final_error = self.compute_registration_error(current_source, target, nbrs)
        logger.info(f"Final registration RMSE: {final_error:.6f}")

        return current_source, transform, final_error
    
    def find_correspondences(self,
                             source: np.ndarray,
                             target: np.ndarray,
                             nbrs: Optional[NearestNeighbors] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find closest point correspondences between source and target.

        Args:
            source: Source point cloud (N x 3)
            target: Target point cloud (M x 3) (only used if nbrs is None)
            nbrs: Pre-built NearestNeighbors instance for target point cloud

        Returns:
            Tuple of (correspondence_indices, distances)        
        """
        if nbrs is None:
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target)

        distances, indices = nbrs.kneighbors(source)

        correspondence_indices = indices.flatten()
        distances = distances.flatten()

        return correspondence_indices, distances

    def estimate_transformation(self,
                                source_points: np.ndarray,
                                target_points: np.ndarray) -> np.ndarray:
        """
        Estimate optimal rigid transformation between source and target point clouds.

        Args:
            source_points: Source point cloud points (N x 3)
            target_points: Corresponding target point cloud points (N x 3)

        Returns:
            Transformation matrix (4 x 4)
        """
        # Center the point sets
        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)

        source_cenetered = source_points - source_centroid
        target_cenetered = target_points - target_centroid

        # Compute cross-covariance matrix
        H = source_cenetered.T @ target_cenetered

        # Singular Value Decomposition
        U, S, Vt = np.linalg.svd(H)

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

    def apply_transformation(self,
                             points: np.ndarray,
                             transform: np.ndarray) -> np.ndarray:
        """
        Apply a transformation matrix to a set of points.

        Args:
            points: Point cloud (N x 3)
            transform: Transformation matrix (4 x 4)

        Returns:
            Transformed point cloud (N x 3)
        """
        # Convert points to homogeneous coordinates
        homogeneous_points = np.column_stack([points, np.ones(len(points))])

        # Apply the transformation
        transformed_points = (transform @ homogeneous_points.T).T

        # Convert back to 3D coordinates
        return transformed_points[:, :3]
    
    def compute_registration_error(self,
                                   source: np.ndarray,
                                   target: np.ndarray,
                                   nbrs: Optional[NearestNeighbors] = None) -> float:
        """
        Compute the registration error (RMSE) between aligned source and target point clouds.

        Args:
            source: Aligned source point cloud
            target: Target point cloud
            nbrs: Pre-built NearestNeighbors instance for target point cloud

        Returns:
            Registration error as RMSE
        """
        # Find correspondences
        correspondences, distances = self.find_correspondences(source, target, nbrs)

        # Filter valid correspondences
        valid_mask = distances < self.max_correspondence_distance

        if np.sum(valid_mask) == 0:
            logger.warning("No valid correspondences found for error computation.")
            return float('inf')
        
        # Compute RMSE
        rmse = np.sqrt(np.mean(distances[valid_mask] ** 2))

        return rmse