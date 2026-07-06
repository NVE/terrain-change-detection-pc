"""
Shared export helpers for the terrain change detection workflow.

Consolidates repeated patterns: CRS auto-detection (cached), output directory
resolution, DEM raster export, and run-input manifest writing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from pprint import pformat

import numpy as np

from terrain_change_detection.utils.config import AppConfig
from terrain_change_detection.utils.export import (
    detect_crs_from_laz,
    export_distances_to_geotiff,
)

from .types import WorkflowAbort

logger = logging.getLogger(__name__)


def make_run_id() -> str:
    """Create the timestamp identifier shared by outputs and run manifest."""
    from datetime import datetime

    return datetime.now().strftime('%Y%m%d_%H%M%S')

# ---------------------------------------------------------------------------
# CRS detection (cached per workflow run)
# ---------------------------------------------------------------------------

_cached_crs: str | None = None
_cached_crs_key: tuple[str, str | None] | None = None


def _detect_laz_crs(label: str, laz_file: str | Path) -> str | None:
    laz_file = Path(laz_file)
    detected = detect_crs_from_laz(str(laz_file))
    if detected:
        logger.info("CRS detected for %s from %s: %s", label, laz_file, detected)
    else:
        logger.warning("No CRS detected for %s from %s", label, laz_file)
    return detected


def resolve_workflow_crs(
    cfg: AppConfig,
    epoch1_laz: str | Path,
    epoch2_laz: str | Path | None = None,
) -> str:
    """Resolve CRS used by workflow outputs from both input epochs.

    The configured ``paths.output_crs`` remains the fallback for files without
    CRS metadata. Conflicting detected CRS values abort because this code does
    not reproject input coordinates.
    """
    global _cached_crs, _cached_crs_key

    cache_key = (str(Path(epoch1_laz)), str(Path(epoch2_laz)) if epoch2_laz is not None else None)
    if _cached_crs is not None and _cached_crs_key == cache_key:
        logger.debug("Using cached workflow CRS: %s", _cached_crs)
        return _cached_crs

    fallback = cfg.paths.output_crs
    logger.info("Configured CRS fallback: %s", fallback)

    crs1 = _detect_laz_crs("epoch 1", epoch1_laz)
    crs2 = _detect_laz_crs("epoch 2", epoch2_laz) if epoch2_laz is not None else None

    if crs1 and crs2:
        if crs1 != crs2:
            raise WorkflowAbort(
                "Conflicting CRS metadata detected between input epochs: "
                f"epoch 1 {epoch1_laz} declares {crs1}, "
                f"epoch 2 {epoch2_laz} declares {crs2}. "
                "Workflow does not reproject inputs; reproject data to a common CRS "
                "or fix LAZ metadata before running."
            )
        selected = crs1
        logger.info("Selected workflow CRS from matching input metadata: %s", selected)
    elif crs1 or crs2:
        selected = crs1 or crs2
        detected_label = "epoch 1" if crs1 else "epoch 2"
        logger.warning(
            "CRS detected only for %s (%s); assuming both epochs use this CRS.",
            detected_label,
            selected,
        )
    else:
        selected = fallback
        logger.warning("No CRS detected from input LAZ files; using configured fallback: %s", selected)

    if selected != fallback:
        logger.info("Detected input CRS overrides configured fallback/output_crs: %s -> %s", fallback, selected)

    _cached_crs = selected
    _cached_crs_key = cache_key
    return selected


def detect_output_crs(cfg: AppConfig, laz_file: str | Path) -> str:
    """Return the output CRS, auto-detecting from *laz_file* on first call.

    The detected CRS is cached so later calls skip the (potentially slow)
    header read.  Falls back to ``cfg.paths.output_crs`` when detection fails.
    """
    return resolve_workflow_crs(cfg, laz_file)


def reset_crs_cache() -> None:
    """Reset the cached CRS (useful between test runs)."""
    global _cached_crs, _cached_crs_key
    _cached_crs = None
    _cached_crs_key = None


# ---------------------------------------------------------------------------
# Output directory resolution
# ---------------------------------------------------------------------------


def resolve_output_dir(cfg: AppConfig, area_name: str, *, area_scoped: bool = True) -> Path:
    """Resolve the output directory, preserving current per-method conventions.

    Args:
        cfg: Application configuration.
        area_name: Name of the selected area.
        area_scoped: If ``True``, append *area_name* to the resolved directory
            (used by M3C2, alignment, DEMs).  If ``False``, return the flat
            output root (used by DoD / C2C when ``output_dir`` is not set).

    Returns:
        A :class:`Path` that has been ``mkdir``-ed.
    """
    if cfg.paths.output_dir:
        export_dir = Path(cfg.paths.output_dir)
    else:
        if area_scoped:
            export_dir = Path(cfg.paths.base_dir) / "output" / area_name
        else:
            export_dir = Path(cfg.paths.base_dir) / "output"

    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


# ---------------------------------------------------------------------------
# DEM raster snapshot export
# ---------------------------------------------------------------------------


def export_dem_rasters(
    *,
    cfg: AppConfig,
    time_labels: list[str],
    point_clouds: list[np.ndarray],
    area_name: str,
    laz_file: str | Path,
    laz_file_epoch2: str | Path | None = None,
    suffix: str,
) -> None:
    """Export DEM GeoTIFFs for one or more point clouds.

    This replaces the near-identical ``export_dem_rasters`` blocks that
    appeared before and after alignment in the original script.

    Args:
        cfg: Application configuration.
        time_labels: Time-period labels (e.g. ``["2015", "2020"]``).
        point_clouds: Corresponding point arrays (global coordinates).
        area_name: Selected area name.
        laz_file: A representative LAZ file from epoch 1 for CRS detection.
        laz_file_epoch2: Optional representative LAZ file from epoch 2.
        suffix: Filename suffix (``"raw"`` or ``"icp"``).
    """
    try:
        crs = resolve_workflow_crs(cfg, laz_file, laz_file_epoch2)
        export_dir = resolve_output_dir(cfg, area_name)

        for label, pc in zip(time_labels, point_clouds):
            dem_tif = export_dir / f"dem_{label}_{suffix}.tif"
            try:
                export_distances_to_geotiff(
                    pc[:, :2], pc[:, 2], str(dem_tif),
                    cell_size=cfg.detection.dod.cell_size, crs=crs,
                    local_transform=None,
                )
                logger.info("Exported DEM raster: %s", dem_tif)
            except Exception as e:
                logger.error("Failed to export DEM raster for %s: %s", label, e)
    except Exception as e:
        logger.error("Failed to export DEM rasters (%s): %s", suffix, e)


# ---------------------------------------------------------------------------
# Run-input manifest
# ---------------------------------------------------------------------------


def write_run_inputs(
    base_dir: Path,
    args,
    cfg: AppConfig,
    *,
    run_id: str | None = None,
    evaluation_summary: dict | None = None,
    config_files: list[str] | None = None,
    cli_overrides: list[str] | None = None,
) -> None:
    """Write the run inputs (CLI args and config) to a text file for record-keeping."""
    import yaml

    if run_id is None:
        run_id = make_run_id()
    filename = f"{run_id}.txt"

    output_path = base_dir / "logs" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("=== Run ===\n")
        f.write(f"run_id: {run_id}\n\n")

        if evaluation_summary:
            f.write("=== Run Evaluation Summary ===\n")
            f.write(pformat(evaluation_summary, sort_dicts=False))
            f.write("\n\n")

        f.write("=== Command Line Arguments ===\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

        f.write("\n=== Configuration Sources ===\n")
        f.write("base_config: config/default.yaml\n")
        if config_files:
            for idx, config_file in enumerate(config_files, start=1):
                f.write(f"override_file_{idx}: {config_file}\n")
        else:
            f.write("override_files: none\n")
        if cli_overrides:
            for idx, override in enumerate(cli_overrides, start=1):
                f.write(f"cli_override_{idx}: {override}\n")
        else:
            f.write("cli_overrides: none\n")

        f.write("\n=== Configuration ===\n")

        cfg_yaml = yaml.safe_dump(cfg.model_dump(), sort_keys=False)
        f.write(cfg_yaml)
