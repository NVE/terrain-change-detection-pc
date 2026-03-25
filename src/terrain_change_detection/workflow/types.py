"""
Typed contracts for inter-phase communication in the workflow.

These dataclasses define the explicit boundaries between workflow phases,
making each step's inputs and outputs clear and testable.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field

import numpy as np

from terrain_change_detection.utils.config import AppConfig
from terrain_change_detection.utils.coordinate_transform import LocalCoordinateTransform


# ---------------------------------------------------------------------------
# Phase-boundary result objects
# ---------------------------------------------------------------------------


@dataclass
class WorkflowRequest:
    """Parsed CLI request plus config source metadata."""

    args: argparse.Namespace
    cfg: AppConfig
    cli_overrides: list[str]
    config_files: list[str] = field(default_factory=list)


@dataclass
class DiscoveryResult:
    """Output of the discovery phase — metadata only, no loaded points.

    This lightweight object carries just enough information to set up the
    local coordinate transform before the (expensive) loading step.
    """

    selected_area: object  # AreaInfo from data_discovery
    t1: str
    t2: str
    ds1: object  # DatasetInfo
    ds2: object  # DatasetInfo
    use_streaming: bool = False


@dataclass
class PreparedData:
    """Output of data-loading and clipping phases."""

    # Selected area metadata
    selected_area: object  # AreaInfo from data_discovery
    t1: str
    t2: str
    ds1: object  # DatasetInfo
    ds2: object  # DatasetInfo

    # Point arrays (subsampled for streaming, full for in-memory)
    points1: np.ndarray
    points2: np.ndarray

    # Streaming metadata (None for in-memory mode)
    pc1_data: dict | None = None
    pc2_data: dict | None = None
    use_streaming: bool = False

    # Coordinate transform and clipping
    local_transform: LocalCoordinateTransform | None = None
    clip_bounds: tuple | None = None


@dataclass
class AlignmentResult:
    """Output of the alignment phase."""

    points1_aligned: np.ndarray
    points2_aligned: np.ndarray
    transform_matrix: np.ndarray  # 4×4
    aligned_epoch: str | None = None
    alignment_error: float | None = None


@dataclass
class WorkflowResult:
    """Summary object for tests and future automation (internal use only)."""

    selected_area: str
    epochs: tuple[str, str]
    streaming_used: bool = False
    alignment_enabled: bool = True
    alignment_error: float | None = None
    export_paths: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class WorkflowAbort(Exception):
    """Raised when the workflow encounters an expected fatal condition.

    Replaces the scattered ``return`` statements in the original monolithic
    ``main()`` function.  Caught once in :func:`runner.run` to preserve
    current "log error and stop" behavior without changing exit semantics.
    """

    def __init__(self, message: str, *, level: int = logging.ERROR):
        super().__init__(message)
        self.level = level
