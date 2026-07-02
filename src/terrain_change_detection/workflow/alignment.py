"""
Spatial alignment pipeline for the terrain change detection workflow.

Handles coarse registration, multi-scale ICP, fine ICP, overlap filtering,
subsampling, reference/target selection, aligned-point export, and streaming
transformed-file saving.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from terrain_change_detection.alignment import (
    apply_transform_to_files,
    compute_overlap_mask,
    save_transform_matrix,
)
from terrain_change_detection.alignment.coarse_registration import CoarseRegistration
from terrain_change_detection.alignment.fine_registration import ICPRegistration
from terrain_change_detection.utils.config import AppConfig
from terrain_change_detection.utils.export import export_points_to_laz

from .artifacts import get_forced_classification, load_alignment_artifact, write_alignment_artifact
from .data_loading import resolve_subsample_count
from .export_helpers import detect_output_crs, resolve_output_dir
from .types import AlignmentResult, PreparedData

logger = logging.getLogger(__name__)


def run_alignment(
    cfg: AppConfig,
    data: PreparedData,
    rng: np.random.Generator,
) -> AlignmentResult:
    """Execute the full alignment pipeline (Step 2).

    If alignment is disabled in config, the original point clouds are returned
    unchanged with an identity transform.

    Args:
        cfg: Application configuration.
        data: Prepared dataset from :func:`data_loading.discover_and_load`.
        rng: Seeded NumPy RNG for deterministic subsampling.

    Returns:
        An :class:`AlignmentResult` with the aligned point clouds and transform.
    """
    alignment_enabled = getattr(cfg.alignment, 'enabled', True)

    if not alignment_enabled:
        logger.info("=== STEP 2: Spatial Alignment (SKIPPED) ===")
        logger.info("ICP alignment disabled in config; using original point clouds.")
        return AlignmentResult(
            points1_aligned=data.points1,
            points2_aligned=data.points2,
            transform_matrix=np.eye(4),
            aligned_epoch=None,
            alignment_error=None,
        )

    cached_alignment = load_alignment_artifact(cfg, data)
    if cached_alignment is not None:
        return cached_alignment

    logger.info("=== STEP 2: Spatial Alignment ===")
    step2_start = time.time()

    points1 = data.points1
    points2 = data.points2

    # ----------------------------------------------------------------
    # Coarse registration
    # ----------------------------------------------------------------
    initial_transform = None
    try:
        if getattr(cfg.alignment, "coarse", None) and cfg.alignment.coarse.enabled:
            coarse = CoarseRegistration(
                method=cfg.alignment.coarse.method,
                voxel_size=cfg.alignment.coarse.voxel_size,
                phase_grid_cell=cfg.alignment.coarse.phase_grid_cell,
            )
            initial_transform = coarse.compute_initial_transform(points2, points1)
            # Optional pre-ICP error report
            try:
                points2_init = coarse.apply_transformation(points2, initial_transform)
                tmp_icp = ICPRegistration(
                    max_iterations=1,
                    tolerance=cfg.alignment.tolerance,
                    max_correspondence_distance=cfg.alignment.max_correspondence_distance,
                    use_gpu=(cfg.gpu.enabled and cfg.gpu.use_for_alignment),
                    convergence_translation_epsilon=cfg.alignment.convergence_translation_epsilon,
                    convergence_rotation_epsilon_deg=cfg.alignment.convergence_rotation_epsilon_deg,
                )
                pre_err = tmp_icp.compute_registration_error(points2_init, points1)
                logger.info("Alignment validation (pre-ICP): RMSE=%.6f m", pre_err)
            except Exception:
                pass
    except Exception as e:
        logger.warning("Coarse registration failed: %s", e)

    # ----------------------------------------------------------------
    # Multi-scale ICP
    # ----------------------------------------------------------------
    transform_matrix = initial_transform if initial_transform is not None else np.eye(4)

    if getattr(cfg.alignment, "multiscale", None) and cfg.alignment.multiscale.enabled:
        transform_matrix = _run_multiscale_icp(
            cfg, points1, points2, transform_matrix, rng,
        )

    # ----------------------------------------------------------------
    # ICP backend selection
    # ----------------------------------------------------------------
    icp = _create_icp_backend(cfg)

    # ----------------------------------------------------------------
    # Overlap filtering (in-memory only)
    # ----------------------------------------------------------------
    points1_for_icp = points1
    points2_for_icp = points2
    if cfg.alignment.overlap_filter and not data.use_streaming:
        mask1, mask2 = compute_overlap_mask(
            points1, points2, margin=cfg.alignment.overlap_margin_m,
        )
        n1_overlap = int(mask1.sum())
        n2_overlap = int(mask2.sum())
        if n1_overlap >= 100 and n2_overlap >= 100:
            points1_for_icp = points1[mask1]
            points2_for_icp = points2[mask2]
            logger.info(
                "Overlap filter: T1 %d/%d, T2 %d/%d points in overlap region",
                n1_overlap, len(points1), n2_overlap, len(points2),
            )
        else:
            logger.warning(
                "Overlap filter: too few points in overlap (%d, %d); using full clouds",
                n1_overlap, n2_overlap,
            )

    # ----------------------------------------------------------------
    # Subsample for fine alignment
    # ----------------------------------------------------------------
    n1_target = resolve_subsample_count(len(points1_for_icp), cfg.alignment)
    if len(points1_for_icp) > n1_target:
        indices1 = rng.choice(len(points1_for_icp), n1_target, replace=False)
        points1_subsampled = points1_for_icp[indices1]
    else:
        points1_subsampled = points1_for_icp

    n2_target = resolve_subsample_count(len(points2_for_icp), cfg.alignment)
    if len(points2_for_icp) > n2_target:
        indices2 = rng.choice(len(points2_for_icp), n2_target, replace=False)
        points2_subsampled = points2_for_icp[indices2]
    else:
        points2_subsampled = points2_for_icp

    # ----------------------------------------------------------------
    # Reference / target selection
    # ----------------------------------------------------------------
    if cfg.alignment.reference == "t2":
        icp_source = points1_subsampled
        icp_target = points2_subsampled
        icp_source_full = points1
        logger.info("ICP direction: aligning T1 (%s) to T2 (%s) reference", data.t1, data.t2)
    else:
        icp_source = points2_subsampled
        icp_target = points1_subsampled
        icp_source_full = points2
        logger.info("ICP direction: aligning T2 (%s) to T1 (%s) reference", data.t2, data.t1)

    # ----------------------------------------------------------------
    # Perform ICP
    # ----------------------------------------------------------------
    _, transform_matrix, final_error = icp.align_point_clouds(
        source=icp_source,
        target=icp_target,
        initial_transform=transform_matrix,
    )

    source_full_aligned = icp.apply_transformation(icp_source_full, transform_matrix)

    if cfg.alignment.reference == "t2":
        points1_aligned = source_full_aligned
        points2_aligned = points2
        aligned_epoch = data.t1
    else:
        points1_aligned = points1
        points2_aligned = source_full_aligned
        aligned_epoch = data.t2

    # ----------------------------------------------------------------
    # Validation RMSE
    # ----------------------------------------------------------------
    alignment_error = _compute_validation_error(
        icp, source_full_aligned, icp_target, rng,
    )
    logger.info("Alignment validation (post-ICP): RMSE=%.6f m", alignment_error)

    # ----------------------------------------------------------------
    # Export aligned point cloud
    # ----------------------------------------------------------------
    if cfg.alignment.export_aligned_pc:
        _export_aligned_pc(
            cfg, data, source_full_aligned, aligned_epoch, transform_matrix,
        )

    # ----------------------------------------------------------------
    # Streaming: apply transform to original files
    # ----------------------------------------------------------------
    if data.use_streaming and cfg.outofcore.save_transformed_files:
        _apply_streaming_transform(cfg, data, transform_matrix)

    step2_end = time.time()
    logger.info("Spatial alignment completed in %.2f seconds", step2_end - step2_start)

    result = AlignmentResult(
        points1_aligned=points1_aligned,
        points2_aligned=points2_aligned,
        transform_matrix=transform_matrix,
        aligned_epoch=aligned_epoch,
        alignment_error=alignment_error,
    )
    write_alignment_artifact(cfg, data, result)
    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _run_multiscale_icp(
    cfg: AppConfig,
    points1: np.ndarray,
    points2: np.ndarray,
    transform_matrix: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run multi-scale ICP refinement and return the (possibly updated) transform."""
    logger.info("Running multi-scale ICP refinement...")

    n_coarse = cfg.alignment.multiscale.coarse_subsample_size
    n1c = min(len(points1), n_coarse)
    n2c = min(len(points2), n_coarse)
    idx1c = rng.choice(len(points1), n1c, replace=False) if len(points1) > n1c else np.arange(len(points1))
    idx2c = rng.choice(len(points2), n2c, replace=False) if len(points2) > n2c else np.arange(len(points2))
    points1_coarse = points1[idx1c]
    points2_coarse = points2[idx2c]

    coarse_max_corr = (
        cfg.alignment.multiscale.coarse_max_correspondence_distance
        if cfg.alignment.multiscale.coarse_max_correspondence_distance is not None
        else cfg.alignment.max_correspondence_distance
    )

    icp_coarse = ICPRegistration(
        max_iterations=cfg.alignment.multiscale.coarse_max_iterations,
        tolerance=cfg.alignment.tolerance,
        max_correspondence_distance=coarse_max_corr,
        convergence_translation_epsilon=cfg.alignment.convergence_translation_epsilon,
        convergence_rotation_epsilon_deg=cfg.alignment.convergence_rotation_epsilon_deg,
    )

    # Pre-refinement RMSE
    try:
        points2_coarse_init = icp_coarse.apply_transformation(points2_coarse, transform_matrix)
        pre_coarse_err = icp_coarse.compute_registration_error(
            source=points2_coarse_init, target=points1_coarse,
        )
    except Exception:
        pre_coarse_err = None

    _, T_coarse, coarse_err = icp_coarse.align_point_clouds(
        source=points2_coarse,
        target=points1_coarse,
        initial_transform=transform_matrix,
    )

    if pre_coarse_err is not None and coarse_err > pre_coarse_err:
        logger.info(
            "Multi-scale refinement unchanged (no improvement): RMSE %.6f m → %.6f m",
            pre_coarse_err, coarse_err,
        )
    else:
        transform_matrix = T_coarse
        if pre_coarse_err is not None:
            logger.info("Multi-scale refinement improved: RMSE %.6f m → %.6f m", pre_coarse_err, coarse_err)
        else:
            logger.info("Multi-scale refinement completed: RMSE=%.6f m", coarse_err)

    return transform_matrix


def _create_icp_backend(cfg: AppConfig):
    """Create the ICP registration instance based on config backend."""
    if cfg.alignment.icp_backend == "open3d":
        try:
            from terrain_change_detection.alignment.open3d_icp import Open3DICP
            icp = Open3DICP(
                max_iterations=cfg.alignment.max_iterations,
                tolerance=cfg.alignment.tolerance,
                max_correspondence_distance=cfg.alignment.max_correspondence_distance,
                convergence_translation_epsilon=cfg.alignment.convergence_translation_epsilon,
                convergence_rotation_epsilon_deg=cfg.alignment.convergence_rotation_epsilon_deg,
            )
            logger.info("Using Open3D ICP backend")
            return icp
        except ImportError:
            logger.warning("Open3D not available; falling back to custom ICP backend")

    return ICPRegistration(
        max_iterations=cfg.alignment.max_iterations,
        tolerance=cfg.alignment.tolerance,
        max_correspondence_distance=cfg.alignment.max_correspondence_distance,
        use_gpu=(cfg.gpu.enabled and cfg.gpu.use_for_alignment),
        convergence_translation_epsilon=cfg.alignment.convergence_translation_epsilon,
        convergence_rotation_epsilon_deg=cfg.alignment.convergence_rotation_epsilon_deg,
    )


def _compute_validation_error(
    icp, source_aligned: np.ndarray, icp_target: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Compute post-ICP RMSE on a downsampled subset."""
    max_err_points = 200_000
    src_err = source_aligned
    tgt_err = icp_target
    if len(src_err) > max_err_points:
        src_err = src_err[rng.choice(len(src_err), max_err_points, replace=False)]
    if len(tgt_err) > max_err_points:
        tgt_err = tgt_err[rng.choice(len(tgt_err), max_err_points, replace=False)]

    return icp.compute_registration_error(source=src_err, target=tgt_err)


def _export_aligned_pc(
    cfg: AppConfig,
    data: PreparedData,
    source_full_aligned: np.ndarray,
    aligned_epoch: str,
    transform_matrix: np.ndarray,
) -> None:
    """Export the aligned point cloud as a LAZ file."""
    crs = detect_output_crs(cfg, str(data.ds1.laz_files[0]))
    export_dir = resolve_output_dir(cfg, data.selected_area.area_name)

    aligned_pc_path = export_dir / f"aligned_{aligned_epoch}.laz"
    export_points = (
        data.local_transform.to_global(source_full_aligned)
        if data.local_transform else source_full_aligned
    )
    source_laz = data.ds1.laz_files[0] if cfg.alignment.reference == "t2" else data.ds2.laz_files[0]
    export_points_to_laz(
        export_points, None, str(aligned_pc_path),
        crs=crs, source_laz_path=str(source_laz),
        classification=get_forced_classification(cfg),
    )
    logger.info("Aligned point cloud exported to: %s", aligned_pc_path)


def _apply_streaming_transform(
    cfg: AppConfig,
    data: PreparedData,
    transform_matrix: np.ndarray,
) -> None:
    """Apply the alignment transform to original LAZ files (streaming mode)."""
    logger.info("--- Applying transformation to full datasets (streaming) ---")

    if cfg.alignment.reference == "t2":
        files_to_transform = data.pc1_data['file_paths']
        aligned_label = f"{data.t1}_aligned"
    else:
        files_to_transform = data.pc2_data['file_paths']
        aligned_label = f"{data.t2}_aligned"

    if cfg.outofcore.output_dir:
        output_dir = Path(cfg.outofcore.output_dir) / data.selected_area.area_name / aligned_label
    else:
        output_dir = Path(cfg.paths.base_dir).parent / "processed" / data.selected_area.area_name / aligned_label

    try:
        aligned_files = apply_transform_to_files(
            input_files=files_to_transform,
            output_dir=str(output_dir),
            transform=transform_matrix,
            ground_only=cfg.preprocessing.ground_only,
            classification_filter=cfg.preprocessing.classification_filter,
            chunk_points=cfg.outofcore.chunk_points,
        )
        if cfg.alignment.reference == "t2":
            data.pc1_data['aligned_file_paths'] = aligned_files
        else:
            data.pc2_data['aligned_file_paths'] = aligned_files

        transform_file = output_dir / "transformation_matrix.txt"
        save_transform_matrix(transform_matrix, str(transform_file))

        logger.info("Transformed %d files saved to %s", len(aligned_files), output_dir)
    except Exception as e:
        logger.error("Failed to apply transformation to files: %s", e)
        logger.info("Falling back to in-memory aligned points for DoD")
