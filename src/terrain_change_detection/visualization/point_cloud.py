"""
Point Cloud Visualization Tools

This module provides visualization tools for point cloud data and analysis results. 
"""

import numpy as np
import plotly.graph_objects as go
import pyvista as pv

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