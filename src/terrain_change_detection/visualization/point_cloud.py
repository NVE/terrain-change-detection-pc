"""
Point Cloud Visualization Tools

This module provides visualization tools for point cloud data and analysis results. 
"""

import numpy as np
import plotly.graph_objects as go
import pyvista as pv
from typing import Optional

try:
    # Optional import for type hints
    from ..detection.change_detection import DoDResult, M3C2Result
except Exception:
    DoDResult = M3C2Result = object  # type: ignore

class PointCloudVisualizer:
    """A class for visualizing point cloud data using different backends."""

    def __init__(self, backend: str = 'plotly'):
        """
        Initializes the visualizer with a specific backend.

        Args:
            backend: The visualization backend to use ('plotly' or 'pyvista').
                           Defaults to 'plotly'.

        Raises:
            ValueError: If the specified backend is not supported.
        """
        if backend not in ['plotly', 'pyvista']:
            raise ValueError(f"Unsupported backend: '{backend}'. Please choose 'plotly' or 'pyvista'.")
        self.backend = backend

    def visualize_clouds(self, point_clouds: list[np.ndarray], names: list[str], sample_size: int = None):
        """
        Visualizes a list of point clouds.

        Args:
            point_clouds: A list of point clouds to visualize.
            names: A list of names for each point cloud for the legend.
            sample_size: The number of points to downsample to. Defaults to None.
        """
        if len(point_clouds) != len(names):
            raise ValueError("The number of point clouds must match the number of names.")

        if sample_size:
            point_clouds = [self._downsample(pc, sample_size) for pc in point_clouds]

        if self.backend == 'plotly':
            self._visualize_plotly(point_clouds, names)
        elif self.backend == 'pyvista':
            self._visualize_pyvista(point_clouds, names)

    def _downsample(self, point_cloud: np.ndarray, sample_size: int) -> np.ndarray:
        """
        Downsamples a point cloud to a specified size.

        Args:
            point_cloud: The point cloud to downsample.
            sample_size: The number of points to downsample to.

        Returns:
            The downsampled point cloud.
        """
        if sample_size >= len(point_cloud):
            return point_cloud
        indices = np.random.choice(len(point_cloud), sample_size, replace=False)
        return point_cloud[indices]

    def _visualize_plotly(self, point_clouds: list[np.ndarray], names: list[str]):
        """
        Visualizes point clouds using Plotly.

        Args:
            point_clouds: List of point clouds to visualize.
            names: List of names for the legend.
        """
        fig = go.Figure()

        for pc, name in zip(point_clouds, names):
            fig.add_trace(go.Scatter3d(
                x=pc[:, 0],
                y=pc[:, 1],
                z=pc[:, 2],
                mode='markers',
                marker=dict(
                    size=1,
                ),
                name=name
            ))

        fig.update_layout(
            title="Point Cloud Visualization",
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data'
            )
        )
        fig.show(renderer="browser")

    def _visualize_pyvista(self, point_clouds: list[np.ndarray], names: list[str]):
        """
        Visualizes point clouds using PyVista.

        Args:
            point_clouds: List of point clouds to visualize.
            names: List of names for the legend.
        """
        plotter = pv.Plotter()
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        for i, (pc, name) in enumerate(zip(point_clouds, names)):
            poly_data = pv.PolyData(pc)
            plotter.add_mesh(poly_data, label=name, color=colors[i % len(colors)], render_points_as_spheres=True, point_size=3)
        plotter.add_legend()
        plotter.show(interactive=False)

    def visualize_dod_heatmap(self, dod: "DoDResult", title: str = "DoD (m)"):
        """
        Visualize a DoD grid as a heatmap using Plotly.

        Args:
            dod: DoDResult containing grid and values
            title: Plot title
        """
        if self.backend == 'plotly':
            # Swap x/y axes and transpose Z so that axes align as requested
            fig = go.Figure(data=go.Heatmap(
                x=dod.grid_y[:, 0],
                y=dod.grid_x[0, :],
                z=dod.dod.T,
                colorscale='RdBu',
                zmid=0.0,
                colorbar=dict(title='m')
            ))
            fig.update_layout(
                title=title,
                xaxis_title='Y',
                yaxis_title='X',
            )
            fig.show(renderer="browser")
        elif self.backend == 'pyvista':
            # Render as a colored surface in XY with scalar = DoD
            # Build structured grid coordinates from centers
            X = dod.grid_x
            Y = dod.grid_y
            Z = np.zeros_like(dod.dod)
            # Create a structured grid
            grid = pv.StructuredGrid()
            pts = np.c_[X.ravel(order='C'), Y.ravel(order='C'), Z.ravel(order='C')]
            grid.points = pts
            grid.dimensions = (X.shape[1], X.shape[0], 1)  # (nx, ny, 1)
            scalars = dod.dod.ravel(order='C')
            plotter = pv.Plotter()
            plotter.add_mesh(grid, scalars=scalars, cmap='RdBu', clim=[-np.nanmax(np.abs(scalars)), np.nanmax(np.abs(scalars))], show_edges=False)
            plotter.view_xy()
            plotter.show(interactive=False)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def visualize_distance_histogram(self, distances: np.ndarray, title: str, bins: int = 60):
        """
        Visualize a histogram of distances (C2C or M3C2).

        Args:
            distances: 1D array of distances
            title: Plot title
            bins: Number of bins
        """
        if self.backend != 'plotly':
            raise ValueError("Histogram is only implemented for 'plotly' backend")
        distances = np.asarray(distances)
        hist, edges = np.histogram(distances[np.isfinite(distances)], bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        fig = go.Figure(data=go.Bar(x=centers, y=hist))
        fig.update_layout(title=title, xaxis_title='Distance (m)', yaxis_title='Count')
        fig.show(renderer="browser")

    def visualize_m3c2_corepoints(self, core_points: np.ndarray, distances: np.ndarray, sample_size: Optional[int] = None, title: str = "M3C2 distances"):
        """
        Visualize core points colored by M3C2 distances.

        Args:
            core_points: (N,3) core points
            distances: (N,) distances
            sample_size: Optional downsample for speed
            title: Plot title
        """
        pts = core_points
        d = distances
        if sample_size and len(pts) > sample_size:
            idx = np.random.choice(len(pts), sample_size, replace=False)
            pts = pts[idx]
            d = d[idx]

        vmax = float(np.nanmax(np.abs(d))) if np.isfinite(d).any() else 1.0

        if self.backend == 'plotly':
            fig = go.Figure(data=go.Scatter3d(
                x=pts[:,0], y=pts[:,1], z=pts[:,2],
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
        elif self.backend == 'pyvista':
            poly = pv.PolyData(pts)
            poly["distance"] = d
            plotter = pv.Plotter()
            plotter.add_mesh(poly, scalars="distance", cmap='RdBu', clim=[-vmax, vmax], render_points_as_spheres=True, point_size=5, show_scalar_bar=True)
            plotter.show(interactive=False)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")