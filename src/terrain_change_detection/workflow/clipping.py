"""
Area clipping logic for the terrain change detection workflow.

Validates clipping configuration, loads boundary files, and clips point clouds
using :class:`~terrain_change_detection.preprocessing.clipping.AreaClipper`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from dataclasses import dataclass

import numpy as np

from terrain_change_detection.preprocessing.clipping import (
    AreaClipper,
    check_shapely_available,
)
from terrain_change_detection.utils.config import AppConfig
from terrain_change_detection.utils.coordinate_transform import LocalCoordinateTransform

from .types import WorkflowAbort

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClipFeature:
    """Single clipping feature prepared for an isolated workflow run."""

    index: int
    label: str
    boundary_file: Path


def _resolve_boundary_path(cfg: AppConfig, project_root: Path | None = None) -> Path | None:
    """Resolve configured clipping boundary path without validating geometry."""
    clipping_cfg = getattr(cfg, 'clipping', None)
    if clipping_cfg is None or not clipping_cfg.enabled or not clipping_cfg.boundary_file:
        return None

    boundary_path = Path(clipping_cfg.boundary_file)
    if not boundary_path.is_absolute():
        if project_root is None:
            project_root = Path(__file__).resolve().parents[3]
        boundary_path = project_root / boundary_path

    return boundary_path


def clipping_export_suffix(cfg: AppConfig, *, project_root: Path | None = None) -> str:
    """Return filename suffix for clipped outputs, or empty string if clipping is off."""
    clipping_cfg = getattr(cfg, 'clipping', None)
    boundary_path = _resolve_boundary_path(cfg, project_root)
    if clipping_cfg is None or boundary_path is None:
        return ""

    label = clipping_cfg.feature_name or _single_geojson_feature_name(boundary_path) or boundary_path.stem
    safe_label = _safe_filename_part(label)
    return f"_clipped_{safe_label}" if safe_label else "_clipped"


def _single_geojson_feature_name(boundary_path: Path) -> str | None:
    """Return properties.name when boundary is a single named GeoJSON feature."""
    if boundary_path.suffix.lower() not in {'.geojson', '.json'} or not boundary_path.exists():
        return None

    try:
        with open(boundary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return None

    if data.get('type') == 'FeatureCollection':
        features = data.get('features') or []
        if len(features) != 1:
            return None
        feature = features[0]
    elif data.get('type') == 'Feature':
        feature = data
    else:
        return None

    name = (feature.get('properties') or {}).get('name')
    return str(name) if name else None


def _safe_filename_part(value: str) -> str:
    """Sanitize a human label for filenames."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    safe = re.sub(r"_+", "_", safe).strip("._-")
    return safe.lower()


def split_clipping_features(cfg: AppConfig, output_dir: Path) -> list[ClipFeature]:
    """Write one temporary GeoJSON per boundary feature for split processing."""
    boundary_path = _resolve_boundary_path(cfg)
    if boundary_path is None or not boundary_path.exists():
        raise WorkflowAbort(f"Clipping boundary file not found: {boundary_path}")

    if boundary_path.suffix.lower() not in {'.geojson', '.json'}:
        raise WorkflowAbort("clipping.split_features currently supports GeoJSON boundary files only.")

    try:
        with open(boundary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        raise WorkflowAbort(f"Could not read clipping GeoJSON for split_features: {e}") from e

    if data.get('type') == 'FeatureCollection':
        features = data.get('features') or []
    elif data.get('type') == 'Feature':
        features = [data]
    else:
        raise WorkflowAbort("clipping.split_features requires a GeoJSON Feature or FeatureCollection.")

    if not features:
        raise WorkflowAbort("clipping.split_features found no GeoJSON features.")

    requested_name = getattr(cfg.clipping, 'feature_name', None)
    if requested_name:
        features = [f for f in features if (f.get('properties') or {}).get('name') == requested_name]
        if not features:
            raise WorkflowAbort(f"clipping.split_features found no feature with properties.name='{requested_name}'.")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_features = []
    for i, feature in enumerate(features, start=1):
        if feature.get('type') != 'Feature' or feature.get('geometry') is None:
            logger.warning("Skipping clipping feature %03d without geometry", i)
            continue

        label = _feature_label(feature, i)
        feature_path = output_dir / f"{i:03d}_{label}.geojson"
        properties = feature.get("properties") or {}
        if not properties.get("name"):
            properties = {**properties, "name": label}
        feature_path.write_text(
            json.dumps({"type": "Feature", "geometry": feature["geometry"], "properties": properties}),
            encoding='utf-8',
        )
        split_features.append(ClipFeature(index=i, label=label, boundary_file=feature_path))

    if not split_features:
        raise WorkflowAbort("clipping.split_features found no valid GeoJSON geometries.")

    return split_features


def _feature_label(feature: dict, index: int) -> str:
    """Return safe feature label from properties.name, falling back to index."""
    name = (feature.get('properties') or {}).get('name')
    label = _safe_filename_part(str(name)) if name else ""
    return label or f"{index:03d}"


def apply_clipping(
    cfg: AppConfig,
    points1: np.ndarray,
    points2: np.ndarray,
    local_transform: LocalCoordinateTransform | None,
    *,
    project_root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple | None]:
    """Apply optional area clipping to both point clouds.

    Args:
        cfg: Application configuration.
        points1: T1 point cloud (N×3).
        points2: T2 point cloud (N×3).
        local_transform: Active local transform (or ``None``).
        project_root: Repository root for resolving relative boundary paths.
            Defaults to the parent of the ``scripts/`` directory.

    Returns:
        ``(points1_clipped, points2_clipped, clip_bounds)`` — the clipped
        point clouds and the clipper bounds tuple (or ``None`` if clipping
        was not enabled).

    Raises:
        WorkflowAbort: If clipping is enabled but prerequisites are missing
            or clipping results in empty point clouds.
    """
    clipping_cfg = getattr(cfg, 'clipping', None)

    if clipping_cfg is None or not clipping_cfg.enabled:
        return points1, points2, None

    if not check_shapely_available():
        raise WorkflowAbort(
            "Clipping is enabled but shapely is not installed. Install with: uv add shapely"
        )

    if not clipping_cfg.boundary_file:
        raise WorkflowAbort(
            "Clipping is enabled but no boundary_file is specified in config."
        )

    boundary_path = _resolve_boundary_path(cfg, project_root)

    if boundary_path is None or not boundary_path.exists():
        raise WorkflowAbort(f"Clipping boundary file not found: {boundary_path}")

    logger.info("--- Applying area clipping ---")

    try:
        clipper = AreaClipper.from_file(
            str(boundary_path),
            feature_name=clipping_cfg.feature_name,
        )

        # Transform clipper to local coordinates if needed
        if local_transform is not None:
            clipper = clipper.transform_to_local(local_transform)

        clip_bounds = clipper.bounds

        original_count_1 = len(points1)
        original_count_2 = len(points2)

        points1 = clipper.clip(points1)
        points2 = clipper.clip(points2)

        pct1 = 100.0 * len(points1) / original_count_1 if original_count_1 > 0 else 0
        pct2 = 100.0 * len(points2) / original_count_2 if original_count_2 > 0 else 0
        logger.info(
            "Clipping complete: T1 %s pts (%.1f%%), T2 %s pts (%.1f%%)",
            f"{len(points1):,}", pct1,
            f"{len(points2):,}", pct2,
        )

        if len(points1) == 0 or len(points2) == 0:
            raise WorkflowAbort(
                "Clipping resulted in empty point clouds. Check your boundary file."
            )

        return points1, points2, clip_bounds

    except WorkflowAbort:
        raise
    except Exception as e:
        raise WorkflowAbort(f"Clipping failed: {e}") from e


def resolve_clipping_bounds(
    cfg: AppConfig,
    *,
    project_root: Path | None = None,
) -> tuple[float, float, float, float] | None:
    """Resolve global clipping bounds for coarse tile prefiltering."""
    clipping_cfg = getattr(cfg, 'clipping', None)

    if clipping_cfg is None or not clipping_cfg.enabled:
        return None

    if not clipping_cfg.boundary_file:
        return None

    if not check_shapely_available():
        return None

    boundary_path = _resolve_boundary_path(cfg, project_root)

    if boundary_path is None or not boundary_path.exists():
        return None

    try:
        clipper = AreaClipper.from_file(
            str(boundary_path),
            feature_name=clipping_cfg.feature_name,
        )
    except Exception as e:
        logger.warning("Could not resolve clipping bounds for tile prefiltering: %s", e)
        return None

    return clipper.bounds


def resolve_clipper(
    cfg: AppConfig,
    local_transform: LocalCoordinateTransform | None = None,
    *,
    project_root: Path | None = None,
) -> AreaClipper | None:
    """Resolve clipping geometry, optionally transformed to local coordinates."""
    clipping_cfg = getattr(cfg, 'clipping', None)

    if clipping_cfg is None or not clipping_cfg.enabled or not clipping_cfg.boundary_file:
        return None

    if not check_shapely_available():
        return None

    boundary_path = _resolve_boundary_path(cfg, project_root)

    if boundary_path is None or not boundary_path.exists():
        return None

    try:
        clipper = AreaClipper.from_file(
            str(boundary_path),
            feature_name=clipping_cfg.feature_name,
        )
        if local_transform is not None:
            clipper = clipper.transform_to_local(local_transform)
        return clipper
    except Exception as e:
        logger.warning("Could not resolve clipping geometry: %s", e)
        return None
