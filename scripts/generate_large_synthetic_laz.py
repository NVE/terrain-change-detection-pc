"""
Generate large synthetic point clouds (LAZ) with controlled changes and misalignment.

This script creates a realistic large-scale dataset (50M+ points) to test
parallelization performance. It generates:
- Multiple tiles covering a large area
- Controlled terrain changes (deposition, erosion, landslides)
- Realistic point density variation
- Rigid transformation for alignment testing

Output: data/large_synthetic/large_area/{2015,2020}/data/*.laz

Requires: laspy with LAZ support (lazrs or laszip)
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
    LAZ_OK = hasattr(laspy, "LazBackend")


def ensure_laz_writing_possible():
    """Check that LAZ compression backend is available."""
    try:
        backends = laspy.LazBackend.detect_available()
        if not backends:
            raise RuntimeError
    except Exception:
        raise RuntimeError(
            "LAZ compression backend not found. Install one of: 'lazrs' (recommended) or 'laszip'.\n"
            "For example: uv add lazrs"
        )


def make_terrain_tile(
    x_offset: float,
    y_offset: float,
    nx: int = 1000,
    ny: int = 1000,
    spacing: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate realistic terrain surface for a tile.
    
    Args:
        x_offset: X coordinate offset for tile position
        y_offset: Y coordinate offset for tile position
        nx: Number of grid points in X
        ny: Number of grid points in Y
        spacing: Grid spacing in meters
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (X, Y, Z) meshgrid arrays
    """
    rng = np.random.default_rng(seed)
    
    # Create coordinate grids
    x = np.arange(nx) * spacing + x_offset
    y = np.arange(ny) * spacing + y_offset
    X, Y = np.meshgrid(x, y)
    
    # Base terrain: multiple scales of variation
    # Large-scale topography (hills and valleys)
    Z = (
        5.0 * np.sin(0.005 * X) * np.cos(0.005 * Y)
        + 3.0 * np.sin(0.01 * X + 0.5) * np.cos(0.008 * Y - 0.3)
        + 2.0 * np.cos(0.015 * X - 0.2) * np.sin(0.012 * Y + 0.7)
    )
    
    # Medium-scale features (ridges, slopes)
    Z += (
        1.0 * np.sin(0.03 * X) * np.cos(0.025 * Y)
        + 0.8 * np.cos(0.04 * (X + Y))
    )
    
    # Fine-scale roughness (natural variation)
    Z += 0.15 * rng.standard_normal(size=Z.shape)
    
    return X, Y, Z


def apply_terrain_changes(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    changes: list[tuple[float, float, float, float, str]],
) -> np.ndarray:
    """
    Apply various terrain changes to the surface.
    
    Args:
        X, Y, Z: Meshgrid coordinates and elevations
        changes: List of (center_x, center_y, radius, magnitude, type) tuples
                 type can be: 'mound', 'pit', 'landslide', 'ridge'
    
    Returns:
        Modified Z array
    """
    Z2 = Z.copy()
    
    for cx, cy, radius, magnitude, change_type in changes:
        R2 = (X - cx)**2 + (Y - cy)**2
        
        if change_type == 'mound':
            # Gaussian mound (deposition)
            change = magnitude * np.exp(-R2 / (2 * radius**2))
        
        elif change_type == 'pit':
            # Gaussian pit (erosion)
            change = -magnitude * np.exp(-R2 / (2 * radius**2))
        
        elif change_type == 'landslide':
            # Asymmetric change with directional flow
            angle = np.arctan2(Y - cy, X - cx)
            flow_direction = 0.0  # radians, 0 = east
            directional_weight = 0.5 + 0.5 * np.cos(angle - flow_direction)
            change = -magnitude * np.exp(-R2 / (2 * radius**2)) * directional_weight
        
        elif change_type == 'ridge':
            # Linear ridge feature
            dist_from_line = np.abs((Y - cy) - 0.5 * (X - cx))
            change = magnitude * np.exp(-dist_from_line**2 / (2 * (radius/3)**2))
        
        else:
            change = 0.0
        
        Z2 += change
    
    return Z2


def sample_points_from_surface(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    density: float = 0.3,
    seed: int = 123,
) -> np.ndarray:
    """
    Sample points from surface with controlled density.
    
    Args:
        X, Y, Z: Meshgrid surface
        density: Fraction of grid points to keep (0-1)
        seed: Random seed
    
    Returns:
        (N, 3) array of sampled points
    """
    rng = np.random.default_rng(seed)
    H, W = Z.shape
    n_total = H * W
    n_keep = int(density * n_total)
    
    # Random sampling with controlled density
    idx = rng.choice(n_total, size=n_keep, replace=False)
    xi = idx % W
    yi = idx // W
    
    points = np.column_stack([
        X[yi, xi],
        Y[yi, xi],
        Z[yi, xi]
    ])
    
    return points


def apply_rigid_transform(
    points: np.ndarray,
    translation: tuple[float, float, float] = (0.5, -0.3, 0.1),
    rotation_deg: tuple[float, float, float] = (0.2, -0.15, 0.1),
) -> np.ndarray:
    """
    Apply rigid transformation (rotation + translation) to points.
    
    Small, realistic misalignment for ICP testing:
    - Translation: ~0.5m in XY (typical GPS/positioning error)
    - Rotation: ~0.2 degrees (typical sensor orientation error)
    
    Args:
        points: (N, 3) array
        translation: (tx, ty, tz) translation vector in meters
        rotation_deg: (rx, ry, rz) rotation angles in degrees
    
    Returns:
        Transformed (N, 3) array
    """
    rx, ry, rz = [math.radians(a) for a in rotation_deg]
    
    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(rx), -math.sin(rx)],
        [0, math.sin(rx), math.cos(rx)]
    ])
    
    Ry = np.array([
        [math.cos(ry), 0, math.sin(ry)],
        [0, 1, 0],
        [-math.sin(ry), 0, math.cos(ry)]
    ])
    
    Rz = np.array([
        [math.cos(rz), -math.sin(rz), 0],
        [math.sin(rz), math.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation
    R = Rz @ Ry @ Rx
    
    # Apply rotation and translation
    t = np.array(translation)
    return (points @ R.T) + t


def write_laz(
    path: Path,
    points: np.ndarray,
    classification_value: int = 2,
) -> None:
    """
    Write points to LAZ file.
    
    Args:
        path: Output file path
        points: (N, 3) array of XYZ coordinates
        classification_value: LAS classification code (2 = ground)
    """
    ensure_laz_writing_possible()
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use LAS 1.4 point format 6
    hdr = laspy.LasHeader(point_format=6, version="1.4")
    hdr.offsets = np.min(points, axis=0)
    hdr.scales = np.array([0.001, 0.001, 0.001])  # 1mm precision
    
    las = laspy.LasData(hdr)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    
    # Mark all as ground (class 2)
    las.classification = np.full(points.shape[0], classification_value, dtype=np.uint8)
    
    # Add realistic intensity variation
    rng = np.random.default_rng(42)
    las.intensity = (100 + 50 * rng.standard_normal(points.shape[0])).clip(0, 255).astype(np.uint16)
    
    # Write as LAZ
    las.write(str(path))


def main():
    """Generate large synthetic dataset with multiple tiles."""
    print("=" * 70)
    print("Generating Large Synthetic Dataset (50M+ points)")
    print("=" * 70)
    
    # Output directories
    base = Path(__file__).parent.parent / "data" / "large_synthetic" / "large_area"
    t1_dir = base / "2015" / "data"
    t2_dir = base / "2020" / "data"
    (base / "2015" / "metadata").mkdir(parents=True, exist_ok=True)
    (base / "2020" / "metadata").mkdir(parents=True, exist_ok=True)
    
    # Tile configuration
    # Create 4x3 grid of tiles (12 tiles total)
    # Each tile: 1000x1000 grid at 0.5m spacing = 500m x 500m area
    # With 30% sampling density: ~300k points per tile = 3.6M points total per epoch
    # To reach 50M+, we'll use 2000x2000 grids with 40% density = 1.6M per tile
    # 4x3 grid = 12 tiles × 1.6M = ~19M points per epoch, but we'll make 4x4 = 16 tiles
    
    tiles_x = 4
    tiles_y = 4
    tile_grid_nx = 2000
    tile_grid_ny = 2000
    tile_spacing = 0.5  # meters
    sample_density = 0.4  # 40% of grid points
    
    tile_size_m = tile_grid_nx * tile_spacing  # 1000m per tile
    
    print(f"\nConfiguration:")
    print(f"  Grid: {tiles_x}x{tiles_y} tiles ({tiles_x * tiles_y} total)")
    print(f"  Tile size: {tile_size_m}m x {tile_size_m}m")
    print(f"  Grid resolution: {tile_grid_nx}x{tile_grid_ny} @ {tile_spacing}m spacing")
    print(f"  Sampling density: {sample_density * 100:.0f}%")
    print(f"  Expected points per tile: ~{int(tile_grid_nx * tile_grid_ny * sample_density):,}")
    print(f"  Expected total points per epoch: ~{int(tiles_x * tiles_y * tile_grid_nx * tile_grid_ny * sample_density):,}")
    
    # Define terrain changes for epoch 2
    # These are in world coordinates
    terrain_changes = [
        # (center_x, center_y, radius, magnitude, type)
        (800.0, 1200.0, 80.0, 2.5, 'mound'),      # Large deposition
        (2400.0, 2800.0, 100.0, 3.0, 'pit'),       # Large erosion
        (1600.0, 800.0, 120.0, 1.8, 'landslide'),  # Landslide
        (3200.0, 1600.0, 60.0, 1.2, 'ridge'),      # Ridge formation
        (400.0, 3200.0, 50.0, 1.5, 'mound'),       # Small deposition
        (2800.0, 400.0, 70.0, 2.0, 'pit'),         # Medium erosion
    ]
    
    print(f"\nTerrain changes: {len(terrain_changes)} features")
    for i, (cx, cy, r, mag, ctype) in enumerate(terrain_changes, 1):
        print(f"  {i}. {ctype:10s} at ({cx:6.0f}, {cy:6.0f}), r={r:.0f}m, Δz={mag:.1f}m")
    
    # Generate tiles
    print(f"\nGenerating tiles...")
    total_points_t1 = 0
    total_points_t2 = 0
    
    for i in range(tiles_x):
        for j in range(tiles_y):
            tile_idx = i * tiles_y + j
            x_offset = i * tile_size_m
            y_offset = j * tile_size_m
            
            print(f"\n  Tile {tile_idx + 1:2d}/{tiles_x * tiles_y} (i={i}, j={j}) at ({x_offset:.0f}, {y_offset:.0f})...", end=" ")
            
            # Generate terrain for this tile (T1)
            seed_base = 1000 + tile_idx
            X, Y, Z = make_terrain_tile(
                x_offset=x_offset,
                y_offset=y_offset,
                nx=tile_grid_nx,
                ny=tile_grid_ny,
                spacing=tile_spacing,
                seed=seed_base
            )
            
            # Sample points for T1 (baseline, no changes)
            pts_t1 = sample_points_from_surface(X, Y, Z, density=sample_density, seed=seed_base + 100)
            
            # For T2: apply terrain changes to the surface, then sample
            # NO misalignment - just pure terrain changes for clean testing
            Z2 = apply_terrain_changes(X, Y, Z, terrain_changes)
            pts_t2 = sample_points_from_surface(X, Y, Z2, density=sample_density, seed=seed_base + 200)
            
            # Write LAZ files
            tile_name = f"tile_{i:02d}_{j:02d}.laz"
            write_laz(t1_dir / tile_name, pts_t1)
            write_laz(t2_dir / tile_name, pts_t2)
            
            total_points_t1 += len(pts_t1)
            total_points_t2 += len(pts_t2)
            
            # Print Z-range for first tile to verify overlap
            if tile_idx == 0:
                print(f"T1={len(pts_t1):,} pts (Z: {pts_t1[:, 2].min():.2f} to {pts_t1[:, 2].max():.2f}), "
                      f"T2={len(pts_t2):,} pts (Z: {pts_t2[:, 2].min():.2f} to {pts_t2[:, 2].max():.2f})")
            else:
                print(f"T1={len(pts_t1):,} pts, T2={len(pts_t2):,} pts")
    
    print("\n" + "=" * 70)
    print("Generation Complete!")
    print("=" * 70)
    print(f"\nOutput directories:")
    print(f"  T1 (2015): {t1_dir}")
    print(f"  T2 (2020): {t2_dir}")
    print(f"\nTotal points:")
    print(f"  T1: {total_points_t1:,} points")
    print(f"  T2: {total_points_t2:,} points")
    print(f"  Combined: {total_points_t1 + total_points_t2:,} points")
    print(f"\nArea covered: {tiles_x * tile_size_m / 1000:.1f} km × {tiles_y * tile_size_m / 1000:.1f} km")
    print(f"\nTo run workflow:")
    print(f"  uv run scripts/run_workflow.py --config config/profiles/large_synthetic.yaml")


if __name__ == "__main__":
    main()
