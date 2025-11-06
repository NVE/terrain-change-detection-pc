"""
Generate two synthetic point clouds (LAZ) with controlled changes and misalignment.

- Creates a simple terrain surface with hills and noise.
- T1 is the baseline.
- T2 introduces:
    * A positive mound (deposition) in one area.
    * A negative pit (erosion) in another area.
    * A rigid transform (translation + small rotation) to simulate misalignment prior to ICP.
- Writes to data/synthetic/synthetic_area/{2015,2020}/data/*.laz with matching metadata folders.

Requires: laspy (already in pyproject). For LAZ writing, laspy needs lazrs or laszip.
This script tries lazrs first. If neither is installed, it will raise a helpful error.
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import laspy

# Attempt to enable LAZ writing
try:
    laspy.LazBackend.detect_available()  # type: ignore[attr-defined]
    LAZ_OK = True
except Exception:
    # Older laspy versions expose via laspy.LazBackend or via creating a writer; we try a fallback
    LAZ_OK = hasattr(laspy, "LazBackend")


def ensure_laz_writing_possible():
    # laspy requires either lazrs (pure python) or laszip (dll) to write .laz
    try:
        backends = laspy.LazBackend.detect_available()  # returns list
        if not backends:
            raise RuntimeError
    except Exception:
        raise RuntimeError(
            "LAZ compression backend not found. Install one of: 'lazrs' (recommended) or 'laszip'.\n"
            "For example: uv add lazrs"
        )


def make_surface(nx=400, ny=400, spacing=1.0, seed=42):
    rng = np.random.default_rng(seed)
    x = (np.arange(nx) - nx/2) * spacing
    y = (np.arange(ny) - ny/2) * spacing
    X, Y = np.meshgrid(x, y)
    # Base surface: gentle hills
    Z = 1.5 * np.sin(0.02*X) * np.cos(0.02*Y) + 0.3 * np.sin(0.05*X + 0.3) + 0.2 * np.cos(0.04*Y - 0.7)
    # Add low-amplitude noise
    Z += 0.05 * rng.standard_normal(size=Z.shape)
    return X, Y, Z


def apply_change(X, Y, Z, centers_radii_heights):
    """Add gaussian bumps (positive or negative)."""
    Z2 = Z.copy()
    for cx, cy, r, h in centers_radii_heights:
        R2 = (X - cx)**2 + (Y - cy)**2
        bump = h * np.exp(-R2 / (2 * r**2))
        Z2 += bump
    return Z2


def to_points(X, Y, Z, keep_ratio=0.3, seed=123):
    rng = np.random.default_rng(seed)
    H, W = Z.shape
    idx = rng.choice(H*W, size=int(keep_ratio*H*W), replace=False)
    xi = idx % W
    yi = idx // W
    pts = np.column_stack([X[yi, xi], Y[yi, xi], Z[yi, xi]])
    return pts


def transform_points(pts, translation=(0.5, -0.3, 0.2), rot_deg=(0.5, -0.3, 0.0)):
    rx, ry, rz = [math.radians(a) for a in rot_deg]
    # Rotation matrices (Rx, Ry, Rz)
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    t = np.array(translation)
    return (pts @ R.T) + t


def write_laz(path: Path, points: np.ndarray, classification_value=2):
    ensure_laz_writing_possible()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use LAS 1.4 point format 6 (has gps time, classification, etc.)
    hdr = laspy.LasHeader(point_format=6, version="1.4")
    las = laspy.LasData(hdr)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    # Mark all as ground (2) to match loader expectations
    las.classification = np.full(points.shape[0], classification_value, dtype=np.uint8)
    # Simple intensity
    las.intensity = np.full(points.shape[0], 100, dtype=np.uint16)
    # Encode as LAZ
    las.write(str(path))


def main():
    base = Path(__file__).parent.parent / "data" / "synthetic" / "synthetic_area"
    t1_dir = base / "2015" / "data"
    t2_dir = base / "2020" / "data"
    (base / "2015" / "metadata").mkdir(parents=True, exist_ok=True)
    (base / "2020" / "metadata").mkdir(parents=True, exist_ok=True)

    # Generate baseline surface and points (T1)
    X, Y, Z = make_surface(nx=500, ny=500, spacing=1.0, seed=1)
    pts_t1 = to_points(X, Y, Z, keep_ratio=0.25, seed=2)

    # T2 changes: deposition near (-50, 60), erosion near (40, -30)
    Z2 = apply_change(
        X, Y, Z,
        centers_radii_heights=[
            (-50.0, 60.0, 25.0, +0.6),  # mound
            (40.0, -30.0, 20.0, -0.5),  # pit
        ],
    )
    pts_t2 = to_points(X, Y, Z2, keep_ratio=0.25, seed=3)

    # Apply a small rigid transform to T2 to simulate misalignment
    pts_t2_misaligned = transform_points(pts_t2, translation=(0.7, -0.4, 0.25), rot_deg=(0.6, -0.4, 0.2))

    # Write LAZ files
    write_laz(t1_dir / "synthetic_tile_01.laz", pts_t1)
    write_laz(t2_dir / "synthetic_tile_01.laz", pts_t2_misaligned)

    print(f"Wrote: {t1_dir / 'synthetic_tile_01.laz'}")
    print(f"Wrote: {t2_dir / 'synthetic_tile_01.laz'}")


if __name__ == "__main__":
    main()
