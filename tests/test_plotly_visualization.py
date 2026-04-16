"""Tests for Plotly visualization fallback behavior in headless environments."""

from __future__ import annotations

import plotly.graph_objects as go

from terrain_change_detection.visualization.point_cloud import PointCloudVisualizer


def test_plotly_visualizer_falls_back_to_html_when_browser_is_unavailable(monkeypatch):
    vis = PointCloudVisualizer(backend="plotly")
    fig = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))
    calls = {"show": False, "write_html": None}

    monkeypatch.delenv("BROWSER", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.delenv("WSL_INTEROP", raising=False)

    def fake_show(self, renderer=None):
        calls["show"] = True
        raise AssertionError("browser renderer should not be called in headless mode")

    def fake_write_html(self, file, auto_open=False, include_plotlyjs="cdn"):
        calls["write_html"] = file

    monkeypatch.setattr(go.Figure, "show", fake_show)
    monkeypatch.setattr(go.Figure, "write_html", fake_write_html)

    vis._show_plotly_figure(fig, title="test")

    assert calls["show"] is False
    assert calls["write_html"] is not None


def test_plotly_visualizer_uses_browser_when_available(monkeypatch):
    vis = PointCloudVisualizer(backend="plotly")
    fig = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))
    calls = {"show": False, "write_html": False}

    monkeypatch.setenv("BROWSER", "dummy-browser")

    def fake_show(self, renderer=None):
        calls["show"] = True

    def fake_write_html(self, file, auto_open=False, include_plotlyjs="cdn"):
        calls["write_html"] = True
        raise AssertionError(
            "HTML fallback should not be used when a browser is available"
        )

    monkeypatch.setattr(go.Figure, "show", fake_show)
    monkeypatch.setattr(go.Figure, "write_html", fake_write_html)

    vis._show_plotly_figure(fig, title="test")

    assert calls["show"] is True
    assert calls["write_html"] is False


def test_plotly_visualizer_falls_back_on_wsl_without_browser_integration(monkeypatch):
    vis = PointCloudVisualizer(backend="plotly")
    fig = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))
    calls = {"show": False, "write_html": None}

    monkeypatch.delenv("BROWSER", raising=False)
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setenv("WSL_INTEROP", "/run/WSL/test_interop")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")

    def fake_show(self, renderer=None):
        calls["show"] = True
        raise AssertionError(
            "browser renderer should not be called on WSL without wslview"
        )

    def fake_write_html(self, file, auto_open=False, include_plotlyjs="cdn"):
        calls["write_html"] = file

    monkeypatch.setattr(go.Figure, "show", fake_show)
    monkeypatch.setattr(go.Figure, "write_html", fake_write_html)
    monkeypatch.setattr(
        "terrain_change_detection.visualization.point_cloud.shutil.which",
        lambda name: None,
    )

    vis._show_plotly_figure(fig, title="test")

    assert calls["show"] is False
    assert calls["write_html"] is not None
