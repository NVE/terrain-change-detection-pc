"""Reusable workflow artifacts for expensive intermediate results."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from terrain_change_detection.preprocessing.loader import PointCloudLoader
from terrain_change_detection.utils.config import AppConfig
from terrain_change_detection.utils.export import export_points_to_laz

from .export_helpers import detect_output_crs, resolve_output_dir
from .types import AlignmentResult, PreparedData

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def load_alignment_artifact(cfg: AppConfig, data: PreparedData) -> AlignmentResult | None:
    """Load reusable post-ICP alignment artifact when metadata still matches."""
    if not _artifacts_enabled(cfg) or not cfg.artifacts.read_existing:
        return None
    if data.use_streaming:
        logger.info("Alignment artifact skip: streaming reuse not implemented yet")
        return None

    metadata_path = _alignment_metadata_path(cfg, data)
    if not metadata_path.exists():
        logger.info("Alignment artifact miss: metadata not found at %s", metadata_path)
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.info("Alignment artifact miss: metadata unreadable (%s)", e)
        return None

    valid, reason = _validate_alignment_metadata(cfg, data, metadata)
    if not valid:
        logger.info("Alignment artifact miss: %s", reason)
        return None

    transform_matrix = np.asarray(metadata["transform_matrix"], dtype=np.float64)
    aligned_epoch = metadata["aligned_epoch"]

    aligned_laz_path = metadata.get("aligned_laz_path")
    if not aligned_laz_path:
        logger.info("Alignment artifact miss: aligned_laz_path missing")
        return None
    loader = PointCloudLoader(
        ground_only=cfg.preprocessing.ground_only,
        classification_filter=cfg.preprocessing.classification_filter,
    )
    aligned_points = loader.load(aligned_laz_path, transform=data.local_transform)["points"]
    if cfg.alignment.reference == "t2":
        points1_aligned = aligned_points
        points2_aligned = data.points2
    else:
        points1_aligned = data.points1
        points2_aligned = aligned_points

    logger.info("Alignment artifact hit: %s", metadata_path)
    return AlignmentResult(
        points1_aligned=points1_aligned,
        points2_aligned=points2_aligned,
        transform_matrix=transform_matrix,
        aligned_epoch=aligned_epoch,
        alignment_error=metadata.get("alignment_error"),
    )


def write_alignment_artifact(
    cfg: AppConfig,
    data: PreparedData,
    alignment: AlignmentResult,
) -> None:
    """Write post-ICP alignment artifact files and metadata."""
    if not _artifacts_enabled(cfg) or not cfg.artifacts.write_outputs:
        return
    if data.use_streaming:
        logger.info("Alignment artifact write skipped: streaming artifacts not implemented yet")
        return

    artifacts_dir = _artifacts_dir(cfg, data.selected_area.area_name)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    forced_class = get_forced_classification(cfg)

    if cfg.alignment.reference == "t2":
        aligned_points = alignment.points1_aligned
        source_laz = data.ds1.laz_files[0]
    else:
        aligned_points = alignment.points2_aligned
        source_laz = data.ds2.laz_files[0]

    aligned_laz_path = artifacts_dir / f"aligned_{alignment.aligned_epoch}.laz"
    crs = detect_output_crs(cfg, str(data.ds1.laz_files[0]))
    export_points_to_laz(
        aligned_points,
        None,
        str(aligned_laz_path),
        crs=crs,
        source_laz_path=str(source_laz),
        local_transform=data.local_transform,
        classification=forced_class,
    )

    transform_path = artifacts_dir / "transformation_matrix.txt"
    np.savetxt(transform_path, alignment.transform_matrix, fmt="%.18e", header="4x4 transformation matrix")

    metadata = _build_alignment_metadata(
        cfg,
        data,
        alignment,
        aligned_laz_path=aligned_laz_path,
        transform_path=transform_path,
        forced_classification=forced_class,
    )
    metadata_path = _alignment_metadata_path(cfg, data)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Alignment artifact written: %s", metadata_path)


def get_forced_classification(cfg: AppConfig) -> int | None:
    if not cfg.artifacts.force_classification:
        return None
    class_filter = cfg.preprocessing.classification_filter
    if class_filter:
        return int(class_filter[0])
    if cfg.preprocessing.ground_only:
        return 2
    return None


def _artifacts_enabled(cfg: AppConfig) -> bool:
    return bool(getattr(cfg, "artifacts", None) and cfg.artifacts.enabled)


def _artifacts_dir(cfg: AppConfig, area_name: str) -> Path:
    if cfg.artifacts.dir:
        return Path(cfg.artifacts.dir) / area_name / "artifacts"
    return resolve_output_dir(cfg, area_name) / "artifacts"


def _alignment_metadata_path(cfg: AppConfig, data: PreparedData) -> Path:
    name = f"alignment_{data.t1}_{data.t2}_{cfg.alignment.reference}.json"
    return _artifacts_dir(cfg, data.selected_area.area_name) / name


def _build_alignment_metadata(
    cfg: AppConfig,
    data: PreparedData,
    alignment: AlignmentResult,
    *,
    aligned_laz_path: Path,
    transform_path: Path,
    forced_classification: int | None,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "alignment",
        "area_name": data.selected_area.area_name,
        "t1": data.t1,
        "t2": data.t2,
        "reference": cfg.alignment.reference,
        "aligned_epoch": alignment.aligned_epoch,
        "alignment_error": alignment.alignment_error,
        "transform_matrix": alignment.transform_matrix.tolist(),
        "transform_path": str(transform_path),
        "aligned_laz_path": str(aligned_laz_path),
        "forced_classification": forced_classification,
        "inputs": _input_fingerprint(data),
        "compat": _compatibility_payload(cfg, data),
    }


def _validate_alignment_metadata(cfg: AppConfig, data: PreparedData, metadata: dict) -> tuple[bool, str]:
    if metadata.get("schema_version") != SCHEMA_VERSION:
        return False, "schema_version mismatch"
    expected = {
        "artifact_type": "alignment",
        "area_name": data.selected_area.area_name,
        "t1": data.t1,
        "t2": data.t2,
        "reference": cfg.alignment.reference,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            return False, f"{key} mismatch"
    if metadata.get("inputs") != _input_fingerprint(data):
        return False, "input fingerprint mismatch"
    if metadata.get("compat") != _compatibility_payload(cfg, data):
        return False, "config compatibility mismatch"
    for path_key in ("aligned_laz_path", "transform_path"):
        path = metadata.get(path_key)
        if not path or not Path(path).exists():
            return False, f"{path_key} missing"
    return True, "ok"


def _input_fingerprint(data: PreparedData) -> list[dict]:
    paths = [*data.ds1.laz_files, *data.ds2.laz_files]
    result = []
    for path_like in paths:
        path = Path(path_like)
        stat = path.stat()
        result.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return result


def _compatibility_payload(cfg: AppConfig, data: PreparedData) -> dict:
    local = data.local_transform
    return {
        "preprocessing": {
            "ground_only": cfg.preprocessing.ground_only,
            "classification_filter": cfg.preprocessing.classification_filter,
        },
        "clipping": {
            "enabled": cfg.clipping.enabled,
            "clip_bounds": list(data.clip_bounds) if data.clip_bounds is not None else None,
        },
        "coordinates": cfg.coordinates.model_dump(),
        "local_transform": None if local is None else {
            "offset_x": local.offset_x,
            "offset_y": local.offset_y,
            "offset_z": local.offset_z,
            "origin_method": local.origin_method,
        },
        "alignment": {
            "enabled": cfg.alignment.enabled,
            "reference": cfg.alignment.reference,
            "icp_backend": cfg.alignment.icp_backend,
            "max_iterations": cfg.alignment.max_iterations,
            "tolerance": cfg.alignment.tolerance,
            "max_correspondence_distance": cfg.alignment.max_correspondence_distance,
            "subsample_size": cfg.alignment.subsample_size,
            "subsample_mode": cfg.alignment.subsample_mode,
            "subsample_percent": cfg.alignment.subsample_percent,
            "random_seed": cfg.alignment.random_seed,
            "overlap_filter": cfg.alignment.overlap_filter,
            "overlap_margin_m": cfg.alignment.overlap_margin_m,
            "coarse": cfg.alignment.coarse.model_dump(),
            "multiscale": cfg.alignment.multiscale.model_dump(),
        },
        "forced_classification": get_forced_classification(cfg),
    }

