"""
Coarse Registration Methods

Provides coarse alignment strategies to initialize ICP on multi-temporal point clouds.

Methods implemented:
- centroid: translation-only alignment by centroids
- pca: rigid alignment by principal axes (3D), then centroid translation
- phase: 2D phase correlation on XY occupancy grid to estimate translation (no rotation)
- open3d_fpfh: optional global feature-based RANSAC via Open3D (if installed)

All methods return a 4x4 transform suitable for initializing ICP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


@dataclass
class CoarseRegistration:
    method: str = "pca"  # centroid | pca | phase | open3d_fpfh | none
    voxel_size: float = 2.0
    phase_grid_cell: float = 2.0

    def compute_initial_transform(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Compute a coarse initial transform aligning source -> target.

        Args:
            source: Nx3 array
            target: Mx3 array

        Returns:
            4x4 transform matrix
        """
        if self.method == "none":
            return np.eye(4)

        if source.size == 0 or target.size == 0:
            logger.warning("CoarseRegistration: empty inputs; returning identity transform.")
            return np.eye(4)

        method = self.method.lower()
        if method == "centroid":
            return self._centroid_transform(source, target)
        if method == "pca":
            T = self._pca_transform(source, target)
            return self._validate_or_fallback(source, target, T)
        if method == "phase":
            try:
                T = self._phase_correlation_xy(source, target, cell=self.phase_grid_cell)
            except Exception as e:
                logger.warning(f"Phase correlation failed: {e}; falling back to centroid.")
                return self._centroid_transform(source, target)
            return self._validate_or_fallback(source, target, T)
        if method == "open3d_fpfh":
            try:
                T = self._open3d_fpfh_transform(source, target, voxel=self.voxel_size)
            except Exception as e:
                logger.warning(f"Open3D FPFH coarse registration failed: {e}; falling back to PCA.")
                T = self._pca_transform(source, target)
            return self._validate_or_fallback(source, target, T)

        logger.warning(f"Unknown coarse registration method '{self.method}', using identity.")
        return np.eye(4)

    # ------------------------ Methods ------------------------
    def _centroid_transform(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        c_src = np.mean(src, axis=0)
        c_dst = np.mean(dst, axis=0)
        t = c_dst - c_src
        T = np.eye(4)
        T[:3, 3] = t
        return T

    def _pca_transform(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        # Center
        c_src = np.mean(src, axis=0)
        c_dst = np.mean(dst, axis=0)
        A = src - c_src
        B = dst - c_dst

        # Covariance and eigenvectors (principal axes)
        # Add small epsilon regularization to avoid singularities on degenerate clouds
        C_A = (A.T @ A) / max(1, len(A)) + 1e-12 * np.eye(3)
        C_B = (B.T @ B) / max(1, len(B)) + 1e-12 * np.eye(3)

        wA, VA = np.linalg.eigh(C_A)
        wB, VB = np.linalg.eigh(C_B)
        # Sort by descending eigenvalues
        idxA = np.argsort(wA)[::-1]
        idxB = np.argsort(wB)[::-1]
        VA = VA[:, idxA]
        VB = VB[:, idxB]

        # Construct rotation mapping src axes to dst axes
        R = VB @ VA.T
        # Fix reflection if needed
        if np.linalg.det(R) < 0:
            VB[:, -1] *= -1
            R = VB @ VA.T

        t = c_dst - (R @ c_src)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def _phase_correlation_xy(self, src: np.ndarray, dst: np.ndarray, *, cell: float = 2.0) -> np.ndarray:
        """Estimate XY translation using phase correlation of occupancy grids.

        Returns 4x4 with identity rotation and estimated XY translation.
        """
        if cell <= 0:
            cell = 2.0

        # Determine union bounds (XY)
        x_min = float(min(src[:, 0].min(), dst[:, 0].min()))
        y_min = float(min(src[:, 1].min(), dst[:, 1].min()))
        x_max = float(max(src[:, 0].max(), dst[:, 0].max()))
        y_max = float(max(src[:, 1].max(), dst[:, 1].max()))

        # Construct grid sizes (power of two for FFT efficiency)
        nx = int(np.ceil((x_max - x_min) / cell)) + 1
        ny = int(np.ceil((y_max - y_min) / cell)) + 1
        # Cap sizes to avoid very large arrays
        max_side = 4096
        if nx > max_side or ny > max_side:
            factor = max(nx / max_side, ny / max_side)
            cell *= factor
            nx = int(np.ceil((x_max - x_min) / cell)) + 1
            ny = int(np.ceil((y_max - y_min) / cell)) + 1

        def to_grid(points: np.ndarray) -> np.ndarray:
            gx = np.clip(((points[:, 0] - x_min) / cell).astype(int), 0, nx - 1)
            gy = np.clip(((points[:, 1] - y_min) / cell).astype(int), 0, ny - 1)
            img = np.zeros((ny, nx), dtype=np.float32)
            # Occupancy count
            np.add.at(img, (gy, gx), 1.0)
            # Normalize
            if img.max() > 0:
                img /= img.max()
            return img

        A = to_grid(src)
        B = to_grid(dst)

        # Phase correlation: cross power spectrum
        FA = np.fft.rfftn(A)
        FB = np.fft.rfftn(B)
        # Cross power spectrum (A vs B)
        R_ab = FA * np.conj(FB)
        denom = np.abs(R_ab)
        denom[denom == 0] = 1.0
        R_ab /= denom
        r_ab = np.fft.irfftn(R_ab, s=A.shape)

        # Peak location => shift (wrap-aware)
        peak = np.unravel_index(np.argmax(r_ab), r_ab.shape)
        shift_y = peak[0]
        shift_x = peak[1]
        if shift_x > nx // 2:
            shift_x -= nx
        if shift_y > ny // 2:
            shift_y -= ny

        # Ambiguity in correlation direction: test both signs quickly and pick best
        # Roll A to align with B and compare L2 error
        def roll_err(sign: int) -> float:
            ry = -sign * shift_y
            rx = -sign * shift_x
            A_shift = np.roll(A, shift=(ry, rx), axis=(0, 1))
            diff = A_shift - B
            return float(np.sum(diff * diff))

        err_pos = roll_err(+1)
        err_neg = roll_err(-1)
        if err_neg < err_pos:
            shift_x = -shift_x
            shift_y = -shift_y

        t_xy = np.array([shift_x * cell, shift_y * cell, 0.0], dtype=float)
        T = np.eye(4)
        T[:3, 3] = t_xy
        return T

    def _open3d_fpfh_transform(self, src: np.ndarray, dst: np.ndarray, *, voxel: float = 2.0) -> np.ndarray:
        try:
            import open3d as o3d  # type: ignore
        except Exception as e:
            raise ImportError("Open3D is required for open3d_fpfh coarse registration") from e

        def to_pcd(points: np.ndarray) -> "o3d.geometry.PointCloud":
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            return pcd

        src_pcd = to_pcd(src)
        dst_pcd = to_pcd(dst)
        if voxel and voxel > 0:
            src_pcd = src_pcd.voxel_down_sample(voxel)
            dst_pcd = dst_pcd.voxel_down_sample(voxel)

        src_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))
        dst_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))

        src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            src_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100)
        )
        dst_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            dst_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100)
        )

        distance_threshold = voxel * 1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_pcd,
            dst_pcd,
            src_fpfh,
            dst_fpfh,
            mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000),
        )

        T = np.eye(4)
        try:
            T = np.asarray(result.transformation, dtype=float)
        except Exception:
            pass
        return T

    # ------------------------ Helpers ------------------------
    def _validate_or_fallback(self, src: np.ndarray, dst: np.ndarray, T: np.ndarray, *, threshold: float = 1.1) -> np.ndarray:
        """Quickly evaluate coarse transform; fallback to centroid if clearly worse.

        Uses a small NN-based RMSE on random subsamples to score the candidate vs. centroid.
        """
        try:
            rmse_T = self._score_rmse(src, dst, T)
            T_cent = self._centroid_transform(src, dst)
            rmse_C = self._score_rmse(src, dst, T_cent)
            if not np.isfinite(rmse_T) or rmse_T > threshold * rmse_C:
                logger.warning(
                    "CoarseRegistration: candidate transform worse than centroid (rmse %.3f vs %.3f). Using centroid.",
                    rmse_T, rmse_C,
                )
                return T_cent
            return T
        except Exception:
            return T

    def _score_rmse(self, src: np.ndarray, dst: np.ndarray, T: np.ndarray, *, max_pairs: int = 3000) -> float:
        if src.size == 0 or dst.size == 0:
            return float("inf")
        rng = np.random.default_rng(0)
        n_src = min(max_pairs, len(src))
        # sample targets proportional so that brute force remains bounded
        # cap product to ~2e6 distance evaluations
        max_prod = 2_000_000
        n_tgt = min(len(dst), max(1000, int(max_prod / max(1, n_src))))
        idx_s = rng.choice(len(src), n_src, replace=False) if len(src) > n_src else np.arange(len(src))
        idx_t = rng.choice(len(dst), n_tgt, replace=False) if len(dst) > n_tgt else np.arange(len(dst))
        A = src[idx_s]
        B = dst[idx_t]
        # apply transform to sampled source
        A1 = self.apply_transformation(A, T)
        try:
            from sklearn.neighbors import NearestNeighbors  # type: ignore
            nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
            nn.fit(B)
            d, _ = nn.kneighbors(A1)
            d = d.reshape(-1)
        except Exception:
            diff = A1[:, None, :] - B[None, :, :]
            dsq = np.einsum("ijk,ijk->ij", diff, diff)
            d = np.sqrt(np.min(dsq, axis=1))
        return float(np.sqrt(np.mean(d ** 2)))
    @staticmethod
    def apply_transformation(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        homog = np.column_stack([points, np.ones(len(points))])
        out = (transform @ homog.T).T
        return out[:, :3]
