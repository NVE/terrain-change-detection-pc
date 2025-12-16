"""
Compare M3C2 histograms: CloudCompare vs. Python Toolkit

This script loads the CloudCompare histogram export and the toolkit's M3C2 output,
then plots both distributions side-by-side for visual comparison.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import laspy
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_cloudcompare_histogram(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load CloudCompare histogram CSV (semicolon-delimited)."""
    df = pd.read_csv(csv_path, sep=";", skipinitialspace=True)
    # Columns: Class; Value; Class start; Class end;
    counts = df["Value"].values
    bin_starts = df["Class start"].values
    bin_ends = df["Class end"].values
    bin_centers = 0.5 * (bin_starts + bin_ends)
    return bin_centers, counts


def load_toolkit_m3c2(laz_path: str) -> np.ndarray:
    """Load M3C2 distances from toolkit LAZ output."""
    with laspy.open(laz_path) as f:
        las = f.read()
    # M3C2 distances are stored in a scalar field, typically 'm3c2_distance' or 'distance'
    if hasattr(las, "m3c2_distance"):
        return np.array(las.m3c2_distance)
    elif hasattr(las, "distance"):
        return np.array(las.distance)
    else:
        # Try extra dimensions
        for dim in las.point_format.extra_dimensions:
            if "m3c2" in dim.name.lower() or "distance" in dim.name.lower():
                return np.array(las[dim.name])
        raise ValueError(f"Could not find M3C2 distance field in {laz_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare CloudCompare vs Toolkit M3C2 histograms")
    parser.add_argument(
        "--cc-csv",
        type=str,
        default="data/raw/CloudCompare_M3C2_Histogram.csv",
        help="Path to CloudCompare histogram CSV",
    )
    parser.add_argument(
        "--toolkit-laz",
        type=str,
        default="data/raw/output/m3c2_eksport_1225654_20250602_2015_2020.laz",
        help="Path to toolkit M3C2 LAZ output",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=256,
        help="Number of bins for toolkit histogram (default: 256 to match CloudCompare)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output HTML file path",
    )
    args = parser.parse_args()

    print(f"Loading CloudCompare histogram from: {args.cc_csv}")
    cc_centers, cc_counts = load_cloudcompare_histogram(args.cc_csv)

    print(f"Loading Toolkit M3C2 from: {args.toolkit_laz}")
    toolkit_distances = load_toolkit_m3c2(args.toolkit_laz)
    valid_distances = toolkit_distances[np.isfinite(toolkit_distances)]

    # Compute histogram for toolkit with same range as CloudCompare
    range_min, range_max = cc_centers.min(), cc_centers.max()
    toolkit_counts, toolkit_edges = np.histogram(
        valid_distances, bins=args.bins, range=(range_min, range_max)
    )
    toolkit_centers = 0.5 * (toolkit_edges[:-1] + toolkit_edges[1:])

    # Statistics
    cc_total = cc_counts.sum()
    tk_total = toolkit_counts.sum()
    tk_mean = float(np.mean(valid_distances))
    tk_std = float(np.std(valid_distances))

    print(f"\nCloudCompare: {cc_total:,} points")
    print(f"Toolkit:      {tk_total:,} valid points")
    print(f"Toolkit Mean: {tk_mean:.4f} m, Std: {tk_std:.4f} m")

    # Create comparison plot with publication-quality styling
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("CloudCompare M3C2", "Python Toolkit M3C2"),
        shared_xaxes=True,
        vertical_spacing=0.12,
    )

    # CloudCompare histogram
    fig.add_trace(
        go.Bar(
            x=cc_centers, y=cc_counts, name="CloudCompare",
            marker_color="#2E7D32",  # Dark green
            marker_line_width=0,
        ),
        row=1, col=1,
    )

    # Toolkit histogram
    fig.add_trace(
        go.Bar(
            x=toolkit_centers, y=toolkit_counts, name="Toolkit",
            marker_color="#1565C0",  # Dark blue
            marker_line_width=0,
        ),
        row=2, col=1,
    )

    # Publication-quality layout
    fig.update_layout(
        title=dict(
            text="M3C2 Distance Histogram Comparison",
            font=dict(size=16, family="Arial, sans-serif", color="black"),
            x=0.5,
            xanchor="center",
        ),
        showlegend=False,  # Remove legend, titles are clear enough
        height=600,
        width=800,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=12, color="black"),
        margin=dict(l=70, r=30, t=80, b=60),
    )

    # Style axes for both subplots
    for row in [1, 2]:
        fig.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor="lightgray",
            showline=True, linewidth=1, linecolor="black",
            mirror=True,
            zeroline=True, zerolinewidth=1, zerolinecolor="gray",
            row=row, col=1,
        )
        fig.update_yaxes(
            showgrid=True, gridwidth=1, gridcolor="lightgray",
            showline=True, linewidth=1, linecolor="black",
            mirror=True,
            title_text="Count",
            row=row, col=1,
        )

    fig.update_xaxes(title_text="M3C2 Distance (m)", row=2, col=1)

    # Update subplot title styling
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=13, family="Arial, sans-serif", color="black")

    # Save or show
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        print(f"\nSaved comparison plot to: {output_path}")
    else:
        fig.show(renderer="browser")


if __name__ == "__main__":
    main()
