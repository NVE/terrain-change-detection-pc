from pathlib import Path

import pytest

from terrain_change_detection.utils.config import AppConfig
from terrain_change_detection.workflow.export_helpers import reset_crs_cache, resolve_workflow_crs
from terrain_change_detection.workflow.types import WorkflowAbort


@pytest.fixture(autouse=True)
def _reset_crs_cache():
    reset_crs_cache()
    yield
    reset_crs_cache()


def _cfg(fallback: str = "EPSG:25833") -> AppConfig:
    cfg = AppConfig()
    cfg.paths.output_crs = fallback
    return cfg


def _patch_detection(monkeypatch, values):
    def fake_detect(path):
        return values.get(Path(path).name)

    monkeypatch.setattr(
        "terrain_change_detection.workflow.export_helpers.detect_crs_from_laz",
        fake_detect,
    )


def test_resolve_workflow_crs_uses_matching_detected_crs(monkeypatch):
    _patch_detection(monkeypatch, {"t1.laz": "EPSG:25832", "t2.laz": "EPSG:25832"})

    crs = resolve_workflow_crs(_cfg(), "t1.laz", "t2.laz")

    assert crs == "EPSG:25832"


def test_resolve_workflow_crs_uses_single_detected_crs(monkeypatch, caplog):
    _patch_detection(monkeypatch, {"t1.laz": "EPSG:25832", "t2.laz": None})

    crs = resolve_workflow_crs(_cfg(), "t1.laz", "t2.laz")

    assert crs == "EPSG:25832"
    assert "CRS detected only for epoch 1" in caplog.text


def test_resolve_workflow_crs_uses_fallback_when_none_detected(monkeypatch, caplog):
    _patch_detection(monkeypatch, {"t1.laz": None, "t2.laz": None})


    crs = resolve_workflow_crs(_cfg("EPSG:25833"), "t1.laz", "t2.laz")

    assert crs == "EPSG:25833"
    assert "No CRS detected from input LAZ files" in caplog.text


def test_resolve_workflow_crs_aborts_on_conflict(monkeypatch):
    _patch_detection(monkeypatch, {"t1.laz": "EPSG:25832", "t2.laz": "EPSG:25833"})

    with pytest.raises(WorkflowAbort, match="Conflicting CRS metadata"):
        resolve_workflow_crs(_cfg(), "t1.laz", "t2.laz")
