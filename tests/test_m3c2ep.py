"""Tests for the M3C2-EP integration layer."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent / "src"))

from terrain_change_detection.alignment.fine_registration import estimate_alignment_covariance
from terrain_change_detection.detection.m3c2 import (
    M3C2Detector,
    M3C2EPScanMetadata,
    M3C2Params,
    _build_epoch,
    _transform_scan_metadata,
)


def test_resolve_scan_metadata_synthesizes_and_normalizes_ids():
    points = np.array(
        [
            [0.0, 0.0, 10.0],
            [1.0, 0.0, 12.0],
            [10.0, 5.0, 20.0],
            [12.0, 5.0, 22.0],
        ],
        dtype=float,
    )
    raw_scan_ids = np.array([444, 444, 10, 10], dtype=np.uint16)

    metadata = M3C2Detector.resolve_scan_metadata(
        points=points,
        raw_scan_ids=raw_scan_ids,
        scan_metadata_source="synthetic_from_point_source_id",
        synthetic_sigma_range=0.02,
        synthetic_sigma_scan=0.001,
        synthetic_sigma_yaw=0.001,
        synthetic_origin_height=100.0,
        epoch_label="T1",
    )

    assert metadata.source == "synthetic_from_point_source_id"
    assert metadata.raw_to_normalized == {10: 1, 444: 2}
    np.testing.assert_array_equal(metadata.normalized_scan_ids, [2, 2, 1, 1])
    assert set(metadata.scanpos_info) == {1, 2}
    assert metadata.scanpos_info[1]["origin"][2] == pytest.approx(122.0)
    assert metadata.scanpos_info[2]["origin"][2] == pytest.approx(112.0)


def test_resolve_scan_metadata_sidecar(tmp_path):
    points = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]], dtype=float)
    raw_scan_ids = np.array([444, 10], dtype=np.uint16)

    sidecar = tmp_path / "scan_positions.yaml"
    sidecar.write_text(
        "\n".join(
            [
                "10:",
                "  origin: [10.0, 20.0, 30.0]",
                "  sigma_range: 0.02",
                "  sigma_scan: 0.001",
                "  sigma_yaw: 0.001",
                "444:",
                "  origin: [40.0, 50.0, 60.0]",
                "  sigma_range: 0.03",
                "  sigma_scan: 0.002",
                "  sigma_yaw: 0.002",
            ]
        ),
        encoding="utf-8",
    )

    metadata = M3C2Detector.resolve_scan_metadata(
        points=points,
        raw_scan_ids=raw_scan_ids,
        scan_metadata_source="sidecar",
        explicit_path=sidecar,
        auto_discover_from_metadata_dir=False,
        epoch_label="T2",
    )

    assert metadata.source.startswith("sidecar:")
    assert metadata.raw_to_normalized == {10: 1, 444: 2}
    np.testing.assert_array_equal(metadata.normalized_scan_ids, [2, 1])
    assert metadata.scanpos_info[1]["origin"] == [10.0, 20.0, 30.0]
    assert metadata.scanpos_info[2]["sigma_range"] == pytest.approx(0.03)


def test_estimate_alignment_covariance_identity_is_zero():
    rng = np.random.default_rng(42)
    cloud = rng.normal(size=(64, 3))
    transform = np.eye(4)

    cxx = estimate_alignment_covariance(
        source=cloud,
        target=cloud,
        transform=transform,
        max_correspondence_distance=0.5,
    )

    assert cxx.shape == (12, 12)
    np.testing.assert_allclose(cxx, np.zeros((12, 12)), atol=1e-12)


def test_compute_m3c2_ep_converts_backend_outputs(monkeypatch):
    py4dgeo = pytest.importorskip("py4dgeo")

    class FakeM3C2EP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self):
            uncertainties = np.array(
                [
                    (0.2, 0.01, 5, 0.02, 6),
                    (0.3, 0.03, 7, 0.04, 8),
                ],
                dtype=[
                    ("lodetection", "f8"),
                    ("spread1", "f8"),
                    ("num_samples1", "i8"),
                    ("spread2", "f8"),
                    ("num_samples2", "i8"),
                ],
            )
            covariance = np.array(
                [
                    (np.eye(3), np.eye(3) * 2.0),
                    (np.eye(3) * 3.0, np.eye(3) * 4.0),
                ],
                dtype=[("cov1", "f8", (3, 3)), ("cov2", "f8", (3, 3))],
            )
            return np.array([0.25, -0.1], dtype=float), uncertainties, covariance

    monkeypatch.setattr(py4dgeo, "M3C2EP", FakeM3C2EP)
    monkeypatch.setattr(
        "terrain_change_detection.detection.m3c2._build_epoch",
        lambda points, scan_metadata=None: {"points": points, "scan_metadata": scan_metadata},
    )

    params = M3C2Params(
        projection_scale=1.0,
        cylinder_radius=1.0,
        max_depth=2.0,
        normal_scale=1.0,
    )
    scan_metadata = M3C2EPScanMetadata(
        raw_scan_ids=np.array([10, 10], dtype=np.int32),
        normalized_scan_ids=np.array([1, 1], dtype=np.int32),
        raw_to_normalized={10: 1},
        scanpos_info={
            1: {
                "origin": [0.0, 0.0, 100.0],
                "sigma_range": 0.02,
                "sigma_scan": 0.001,
                "sigma_yaw": 0.001,
            }
        },
        source="synthetic_from_point_source_id",
    )

    result = M3C2Detector.compute_m3c2_ep(
        core_points=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float),
        cloud_t1=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float),
        cloud_t2=np.array([[0.0, 0.0, 0.1], [1.0, 1.0, 1.1]], dtype=float),
        params=params,
        transform_matrix=np.eye(4),
        cxx=np.zeros((12, 12), dtype=float),
        scan_metadata_t1=scan_metadata,
        scan_metadata_t2=scan_metadata,
    )

    np.testing.assert_allclose(result.distances, [-0.25, 0.1])
    np.testing.assert_allclose(result.uncertainty, [0.2, 0.3])
    np.testing.assert_array_equal(result.significant, [True, False])
    assert result.metadata["variant"] == "ep"
    assert result.ep_details is not None
    np.testing.assert_allclose(result.ep_details.spread1, [0.01, 0.03])
    np.testing.assert_array_equal(result.ep_details.num_samples2, [6, 8])
    np.testing.assert_allclose(result.ep_details.covariance1[0], np.eye(3))
    np.testing.assert_allclose(result.ep_details.covariance2[1], np.eye(3) * 4.0)


def test_transform_scan_metadata_moves_origins_without_mutating_input():
    scan_metadata = M3C2EPScanMetadata(
        raw_scan_ids=np.array([10, 10], dtype=np.int32),
        normalized_scan_ids=np.array([1, 1], dtype=np.int32),
        raw_to_normalized={10: 1},
        scanpos_info={
            1: {
                "origin": [1.0, 2.0, 3.0],
                "sigma_range": 0.02,
                "sigma_scan": 0.001,
                "sigma_yaw": 0.001,
            }
        },
        source="synthetic_from_point_source_id",
    )
    transform_3x4 = np.array(
        [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, -5.0],
            [0.0, 0.0, 1.0, 2.0],
        ],
        dtype=float,
    )

    transformed = _transform_scan_metadata(
        scan_metadata,
        transform_3x4,
        np.zeros(3, dtype=float),
    )

    assert scan_metadata.scanpos_info[1]["origin"] == [1.0, 2.0, 3.0]
    assert transformed.scanpos_info[1]["origin"] == [11.0, -3.0, 5.0]


def test_build_epoch_with_scan_metadata_attaches_scanpos_ids():
    pytest.importorskip("py4dgeo")

    scan_metadata = M3C2EPScanMetadata(
        raw_scan_ids=np.array([10, 20], dtype=np.int32),
        normalized_scan_ids=np.array([1, 2], dtype=np.int32),
        raw_to_normalized={10: 1, 20: 2},
        scanpos_info={
            1: {
                "origin": [0.0, 0.0, 100.0],
                "sigma_range": 0.02,
                "sigma_scan": 0.001,
                "sigma_yaw": 0.001,
            },
            2: {
                "origin": [1.0, 1.0, 120.0],
                "sigma_range": 0.02,
                "sigma_scan": 0.001,
                "sigma_yaw": 0.001,
            },
        },
        source="synthetic_from_point_source_id",
    )

    epoch = _build_epoch(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float), scan_metadata)

    np.testing.assert_array_equal(epoch.scanpos_id, [1, 2])
    assert epoch.scanpos_info[0]["origin"] == [0.0, 0.0, 100.0]


def test_compute_m3c2_ep_smoke_real_backend():
    pytest.importorskip("py4dgeo")

    params = M3C2Params(
        projection_scale=1.0,
        cylinder_radius=1.0,
        max_depth=2.0,
        normal_scale=1.0,
    )
    cloud_t1 = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=float,
    )
    cloud_t2 = cloud_t1 + np.array([0.0, 0.0, 0.1])
    raw_scan_ids = np.ones(len(cloud_t1), dtype=np.int32)
    scan_metadata = M3C2Detector.resolve_scan_metadata(
        points=cloud_t1,
        raw_scan_ids=raw_scan_ids,
        scan_metadata_source="synthetic_from_point_source_id",
        synthetic_origin_height=100.0,
        epoch_label="T1",
    )

    try:
        result = M3C2Detector.compute_m3c2_ep(
            core_points=cloud_t1[:2],
            cloud_t1=cloud_t1,
            cloud_t2=cloud_t2,
            params=params,
            transform_matrix=np.eye(4),
            cxx=np.zeros((12, 12), dtype=float),
            scan_metadata_t1=scan_metadata,
            scan_metadata_t2=scan_metadata,
        )
    except (OSError, EOFError, BrokenPipeError) as exc:
        pytest.skip(f"py4dgeo M3C2EP multiprocessing is not supported in this environment: {exc}")

    assert result.distances.shape == (2,)
    assert result.uncertainty is not None
