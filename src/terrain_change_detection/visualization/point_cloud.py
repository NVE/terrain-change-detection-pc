"""
Point Cloud Visualization Tools

This module provides visualization tools for point cloud data and analysis results.
"""

from typing import Optional, Any
import numpy as np
import plotly.graph_objects as go
import pyvista as pv

# Optional Qt-based interactive plotter
try:
    from pyvistaqt import BackgroundPlotter  # type: ignore
except Exception:  # pragma: no cover
    BackgroundPlotter = None  # type: ignore


class PointCloudVisualizer:
    """A class for visualizing point cloud data using different backends."""

    def __init__(self, backend: str = 'plotly'):
        """
        Args:
            backend: 'plotly', 'pyvista', or 'pyvistaqt'
        """
        if backend not in ['plotly', 'pyvista', 'pyvistaqt']:
            raise ValueError(
                f"Unsupported backend: '{backend}'. Choose 'plotly', 'pyvista', or 'pyvistaqt'."
            )
        self.backend = backend

    # ----------------- Public API -----------------
    def visualize_clouds(self, point_clouds: list[np.ndarray], names: list[str], sample_size: int | None = None):
        if len(point_clouds) != len(names):
            raise ValueError("The number of point clouds must match the number of names.")
        if sample_size:
            point_clouds = [self._downsample(pc, sample_size) for pc in point_clouds]
        if self.backend == 'plotly':
            self._visualize_plotly(point_clouds, names)
        else:
            self._visualize_pyvista(point_clouds, names)

    def visualize_dod_heatmap(self, dod: Any, title: str = "DoD (m)"):
        if self.backend == 'plotly':
            fig = go.Figure(data=go.Heatmap(
                x=dod.grid_y[:, 0],
                y=dod.grid_x[0, :],
                z=dod.dod.T,
                colorscale='RdBu',
                zmid=0.0,
                colorbar=dict(title='m')
            ))
            # Enforce equal scaling for X and Y so distances are to scale
            fig.update_layout(title=title, xaxis_title='Y', yaxis_title='X')
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            fig.update_xaxes(constrain='domain')
            fig.show(renderer="browser")
            return
        # PyVista / PyVistaQt
        X = dod.grid_x
        Y = dod.grid_y
        Z = np.zeros_like(dod.dod)
        grid = pv.StructuredGrid()
        pts = np.c_[X.ravel(order='C'), Y.ravel(order='C'), Z.ravel(order='C')]
        grid.points = pts
        grid.dimensions = (X.shape[1], X.shape[0], 1)
        scalars = dod.dod.astype(float).ravel(order='C')
        vmax = float(np.nanmax(np.abs(scalars))) if np.isfinite(scalars).any() else 1.0
        plotter = self._get_plotter()
        plotter.add_mesh(
            grid,
            scalars=scalars,
            cmap='RdBu',
            clim=[-vmax, vmax],
            show_edges=False,
            show_scalar_bar=True,
            lighting=False,
        )
        # Top-down view and parallel projection to keep XY distances to scale
        plotter.view_xy()
        try:
            plotter.enable_parallel_projection()
        except Exception:
            pass
        plotter.show()

    def visualize_distance_histogram(self, distances: np.ndarray, title: str, bins: int = 60):
        if self.backend != 'plotly':
            raise ValueError("Histogram is only implemented for 'plotly' backend")
        distances = np.asarray(distances)
        hist, edges = np.histogram(distances[np.isfinite(distances)], bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        fig = go.Figure(data=go.Bar(x=centers, y=hist))
        fig.update_layout(title=title, xaxis_title='Distance (m)', yaxis_title='Count')
        fig.show(renderer="browser")

    def visualize_m3c2_corepoints(self, core_points: np.ndarray, distances: np.ndarray, sample_size: Optional[int] = None, title: str = "M3C2 distances"):
        pts = core_points
        d = np.asarray(distances, dtype=float)
        
        # Filter out points with invalid (NaN/inf) distances to avoid undefined colors
        valid_mask = np.isfinite(d)
        if not np.any(valid_mask):
            raise ValueError("No valid (finite) distances to visualize")
        pts = pts[valid_mask]
        d = d[valid_mask]
        
        if sample_size and len(pts) > sample_size:
            idx = np.random.choice(len(pts), sample_size, replace=False)
            pts = pts[idx]
            d = d[idx]
        vmax = float(np.nanmax(np.abs(d))) if np.isfinite(d).any() else 1.0
        if self.backend == 'plotly':
            fig = go.Figure(data=go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers',
                marker=dict(size=2, color=d, colorscale='RdBu', cmin=-vmax, cmax=vmax, colorbar=dict(title='m'))
            ))
            fig.update_layout(
                title=title,
                scene=dict(
                    aspectmode='data',
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                ),
                margin=dict(l=0, r=0, t=40, b=0)
            )
            fig.show(renderer="browser")
            return
        # PyVista / PyVistaQt
        poly = pv.PolyData(pts)
        poly["distance"] = d
        plotter = self._get_plotter()
        plotter.add_mesh(
            poly,
            scalars="distance",
            cmap='RdBu',
            clim=[-vmax, vmax],
            render_points_as_spheres=False,
            point_size=5,
            show_scalar_bar=True,
            lighting=False,
        )
        plotter.show()

    def visualize_c2c_points(self, source_points: np.ndarray, distances: np.ndarray, sample_size: Optional[int] = None, title: str = "C2C distances"):
        """
        Visualize C2C per-point distances by coloring the source points.

        Reuses the same rendering as M3C2 core points visualization.
        """
        return self.visualize_m3c2_corepoints(source_points, distances, sample_size=sample_size, title=title)

    # ----------------- Internal helpers -----------------
    def _downsample(self, point_cloud: np.ndarray, sample_size: int) -> np.ndarray:
        if sample_size >= len(point_cloud):
            return point_cloud
        indices = np.random.choice(len(point_cloud), sample_size, replace=False)
        return point_cloud[indices]

    def _visualize_plotly(self, point_clouds: list[np.ndarray], names: list[str]):
        fig = go.Figure()
        for pc, name in zip(point_clouds, names):
            fig.add_trace(go.Scatter3d(
                x=pc[:, 0], y=pc[:, 1], z=pc[:, 2],
                mode='markers',
                marker=dict(size=1),
                name=name,
            ))
        fig.update_layout(
            title="Point Cloud Visualization",
            scene=dict(
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data'
            ),
        )
        fig.show(renderer="browser")

    def _get_plotter(self):
        if self.backend == 'pyvistaqt':
            if BackgroundPlotter is None:
                raise ImportError("pyvistaqt is not installed. Install with 'uv add pyvistaqt PySide6'.")
            return BackgroundPlotter()
        return pv.Plotter()

    def _visualize_pyvista(self, point_clouds: list[np.ndarray], names: list[str]):
        plotter = self._get_plotter()
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        for i, (pc, name) in enumerate(zip(point_clouds, names)):
            poly_data = pv.PolyData(pc)
            plotter.add_mesh(
                poly_data,
                label=name,
                color=colors[i % len(colors)],
                render_points_as_spheres=False,
                point_size=3,
                lighting=False,
            )
        plotter.add_legend()
        plotter.show()
