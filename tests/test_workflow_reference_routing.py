"""Tests for reference-direction routing in the extracted workflow modules."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from terrain_change_detection.utils.config import AppConfig
from terrain_change_detection.workflow.detection_c2c import run_c2c
from terrain_change_detection.workflow.detection_dod import run_dod
from terrain_change_detection.workflow.detection_m3c2 import run_m3c2
from terrain_change_detection.workflow.export_helpers import resolve_output_dir
from terrain_change_detection.workflow.runner import _split_features_dir
from terrain_change_detection.workflow.types import AlignmentResult, PreparedData


def _make_cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.alignment.reference = "t2"
    cfg.detection.dod.enabled = True
    cfg.detection.dod.export_raster = False
    cfg.detection.c2c.enabled = True
    cfg.detection.c2c.export_pc = False
    cfg.detection.c2c.export_raster = False
    cfg.detection.m3c2.enabled = True
    cfg.detection.m3c2.export_pc = False
    cfg.detection.m3c2.export_raster = False
    cfg.detection.m3c2.use_autotune = False
    cfg.detection.m3c2.core_points_percent = 100.0
    cfg.outofcore.enabled = False
    return cfg


def test_split_features_dir_is_area_scoped(tmp_path):
    cfg = AppConfig()
    cfg.paths.output_dir = str(tmp_path / "outputs")

    assert _split_features_dir(cfg, "Romerike") == tmp_path / "outputs" / "Romerike" / "_split_features"


def test_method_exports_default_to_area_scoped_output(tmp_path):
    cfg = AppConfig()
    cfg.paths.base_dir = str(tmp_path)
    cfg.paths.output_dir = None

    assert resolve_output_dir(cfg, "Romerike", area_scoped=True) == tmp_path / "output" / "Romerike"


def _make_prepared_data(*, use_streaming: bool = False) -> PreparedData:
    area = SimpleNamespace(area_name="test_area")
    ds = SimpleNamespace(laz_files=[Path("dummy.laz")])
    return PreparedData(
        selected_area=area,
        t1="2015",
        t2="2020",
        ds1=ds,
        ds2=ds,
        points1=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float),
        points2=np.array([[10.0, 10.0, 10.0], [11.0, 11.0, 11.0]], dtype=float),
        pc1_data={"file_paths": ["t1_a.laz"]} if use_streaming else None,
        pc2_data={"file_paths": ["t2_a.laz"]} if use_streaming else None,
        use_streaming=use_streaming,
        local_transform=None,
        clip_bounds=None,
    )


def _make_alignment() -> AlignmentResult:
    return AlignmentResult(
        points1_aligned=np.array([[100.0, 100.0, 100.0], [101.0, 101.0, 101.0]], dtype=float),
        points2_aligned=np.array([[200.0, 200.0, 200.0], [201.0, 201.0, 201.0]], dtype=float),
        transform_matrix=np.eye(4),
        aligned_epoch="2015",
        alignment_error=0.0,
    )


def test_dod_uses_aligned_t1_for_reference_t2(monkeypatch):
    cfg = _make_cfg()
    data = _make_prepared_data(use_streaming=False)
    alignment = _make_alignment()
    captured = {}

    def fake_compute_dod(*, points_t1, points_t2, cell_size, aggregator, config):
        captured["points_t1"] = points_t1.copy()
        captured["points_t2"] = points_t2.copy()
        return SimpleNamespace(
            grid_x=np.zeros((1, 1)),
            grid_y=np.zeros((1, 1)),
            dod=np.zeros((1, 1)),
            bounds=(0.0, 0.0, 1.0, 1.0),
            cell_size=cell_size,
        )

    monkeypatch.setattr(
        "terrain_change_detection.workflow.detection_dod.ChangeDetector.compute_dod",
        fake_compute_dod,
    )

    run_dod(cfg, data, alignment, show_plots=False)

    np.testing.assert_array_equal(captured["points_t1"], alignment.points1_aligned)
    np.testing.assert_array_equal(captured["points_t2"], alignment.points2_aligned)


def test_dod_streaming_reference_t2_falls_back_without_aligned_t1(monkeypatch):
    cfg = _make_cfg()
    cfg.outofcore.enabled = True
    data = _make_prepared_data(use_streaming=True)
    alignment = _make_alignment()
    captured = {"streaming_called": False, "fallback_called": False}

    def fail_if_streaming_called(**kwargs):
        captured["streaming_called"] = True
        raise AssertionError("streaming path should not be used without aligned T1 files")

    def fake_compute_dod(*, points_t1, points_t2, cell_size, aggregator, config):
        captured["fallback_called"] = True
        return SimpleNamespace(
            grid_x=np.zeros((1, 1)),
            grid_y=np.zeros((1, 1)),
            dod=np.zeros((1, 1)),
            bounds=(0.0, 0.0, 1.0, 1.0),
            cell_size=cell_size,
        )

    monkeypatch.setattr(
        "terrain_change_detection.workflow.detection_dod.ChangeDetector.compute_dod_streaming_files_tiled",
        fail_if_streaming_called,
    )
    monkeypatch.setattr(
        "terrain_change_detection.workflow.detection_dod.ChangeDetector.compute_dod_streaming_files_tiled_parallel",
        fail_if_streaming_called,
    )
    monkeypatch.setattr(
        "terrain_change_detection.workflow.detection_dod.ChangeDetector.compute_dod",
        fake_compute_dod,
    )

    run_dod(cfg, data, alignment, show_plots=False)

    assert captured["fallback_called"] is True
    assert captured["streaming_called"] is False


def test_dod_export_filename_includes_run_id(tmp_path, monkeypatch):
    cfg = _make_cfg()
    cfg.paths.output_dir = str(tmp_path)
    data = _make_prepared_data(use_streaming=False)
    alignment = _make_alignment()
    cfg.detection.dod.export_raster = True
    captured = {}

    def fake_compute_dod(*, points_t1, points_t2, cell_size, aggregator, config):
        return SimpleNamespace(
            grid_x=np.zeros((1, 1)),
            grid_y=np.zeros((1, 1)),
            dem1=np.zeros((1, 1)),
            dem2=np.zeros((1, 1)),
            dod=np.zeros((1, 1)),
            bounds=(0.0, 0.0, 1.0, 1.0),
            cell_size=cell_size,
        )

    def fake_export(dod_res, output_path, **kwargs):
        captured["output_path"] = output_path
        return output_path

    monkeypatch.setattr(
        "terrain_change_detection.workflow.detection_dod.ChangeDetector.compute_dod",
        fake_compute_dod,
    )
    monkeypatch.setattr(
        "terrain_change_detection.workflow.detection_dod.resolve_workflow_crs",
        lambda *args, **kwargs: "EPSG:25833",
    )
    monkeypatch.setattr(
        "terrain_change_detection.workflow.detection_dod.export_dod_to_geotiff",
        fake_export,
    )

    run_dod(cfg, data, alignment, run_id="RUN123", show_plots=False)

    assert Path(captured["output_path"]).name == "dod_test_area_2015_2020_RUN123.tif"


def test_c2c_uses_aligned_t1_as_target_for_reference_t2(monkeypatch):
    cfg = _make_cfg()
    data = _make_prepared_data(use_streaming=False)
    alignment = _make_alignment()
    captured = {}

    def fake_compute_c2c(src, tgt, max_distance, config):
        captured["src"] = src.copy()
        captured["tgt"] = tgt.copy()
        return SimpleNamespace(distances=np.array([0.1, 0.2]))

    monkeypatch.setattr(
        "terrain_change_detection.workflow.detection_c2c.ChangeDetector.compute_c2c",
        fake_compute_c2c,
    )

    run_c2c(cfg, data, alignment, show_plots=False)

    np.testing.assert_array_equal(captured["src"], alignment.points2_aligned)
    np.testing.assert_array_equal(captured["tgt"], alignment.points1_aligned)


def test_c2c_export_filename_includes_run_id(tmp_path, monkeypatch):
    cfg = _make_cfg()
    cfg.paths.output_dir = str(tmp_path)
    data = _make_prepared_data(use_streaming=False)
    alignment = _make_alignment()
    cfg.detection.c2c.export_pc = True
    cfg.detection.c2c.export_raster = True
    captured = {}

    monkeypatch.setattr(
        "terrain_change_detection.workflow.detection_c2c.ChangeDetector.compute_c2c",
        lambda src, tgt, max_distance, config: SimpleNamespace(distances=np.array([0.1, 0.2])),
    )
    monkeypatch.setattr(
        "terrain_change_detection.workflow.detection_c2c.resolve_workflow_crs",
        lambda *args, **kwargs: "EPSG:25833",
    )
    monkeypatch.setattr(
        "terrain_change_detection.workflow.detection_c2c.export_points_to_laz",
        lambda points, distances, output_path, **kwargs: captured.setdefault("laz", output_path),
    )
    monkeypatch.setattr(
        "terrain_change_detection.workflow.detection_c2c.export_distances_to_geotiff",
        lambda points, distances, output_path, **kwargs: captured.setdefault("tif", output_path),
    )

    run_c2c(cfg, data, alignment, run_id="RUN123", show_plots=False)

    assert Path(captured["laz"]).name == "c2c_test_area_2015_2020_RUN123.laz"
    assert Path(captured["tif"]).name == "c2c_test_area_2015_2020_RUN123.tif"


def test_m3c2_uses_aligned_t1_for_reference_t2(monkeypatch):
    cfg = _make_cfg()
    data = _make_prepared_data(use_streaming=False)
    alignment = _make_alignment()
    captured = {}

    def fake_compute_m3c2_original(*, core_points, cloud_t1, cloud_t2, params):
        captured["core_points"] = core_points.copy()
        captured["cloud_t1"] = cloud_t1.copy()
        captured["cloud_t2"] = cloud_t2.copy()
        return SimpleNamespace(
            core_points=core_points,
            distances=np.zeros(len(core_points)),
            uncertainty=None,
            significant=None,
        )

    monkeypatch.setattr(
        "terrain_change_detection.workflow.detection_m3c2.ChangeDetector.compute_m3c2_original",
        fake_compute_m3c2_original,
    )

    run_m3c2(cfg, data, alignment, SimpleNamespace(cores_file=None, debug_m3c2_compare=False), show_plots=False)

    np.testing.assert_array_equal(captured["cloud_t1"], alignment.points1_aligned)
    np.testing.assert_array_equal(captured["cloud_t2"], alignment.points2_aligned)
    np.testing.assert_array_equal(captured["core_points"], alignment.points1_aligned)


def test_m3c2_evaluation_source_t2_uses_t2_cores_without_flipping_epochs(monkeypatch):
    cfg = _make_cfg()
    cfg.detection.m3c2.evaluation_source = "t2"
    data = _make_prepared_data(use_streaming=False)
    alignment = _make_alignment()
    captured = {}

    def fake_compute_m3c2_original(*, core_points, cloud_t1, cloud_t2, params):
        captured["core_points"] = core_points.copy()
        captured["cloud_t1"] = cloud_t1.copy()
        captured["cloud_t2"] = cloud_t2.copy()
        return SimpleNamespace(
            core_points=core_points,
            distances=np.zeros(len(core_points)),
            uncertainty=None,
            significant=None,
        )

    monkeypatch.setattr(
        "terrain_change_detection.workflow.detection_m3c2.ChangeDetector.compute_m3c2_original",
        fake_compute_m3c2_original,
    )

    run_m3c2(cfg, data, alignment, SimpleNamespace(cores_file=None, debug_m3c2_compare=False), show_plots=False)

    np.testing.assert_array_equal(captured["cloud_t1"], alignment.points1_aligned)
    np.testing.assert_array_equal(captured["cloud_t2"], alignment.points2_aligned)
    np.testing.assert_array_equal(captured["core_points"], alignment.points2_aligned)
