# Area Clipping for Focused Analysis

This document describes the area clipping functionality that allows you to focus terrain change detection analysis on specific regions of interest.

## Overview

The area clipping feature enables you to:
- **Clip point clouds** to a specific region of interest before processing
- **Reduce data volume** for faster ICP registration and change detection
- **Focus analysis** on relevant areas (e.g., rivers, erosion zones, construction sites)
- **Exclude irrelevant regions** from computations

The clipping is performed **before ICP registration** to ensure only relevant areas are processed in subsequent calculations.

## Supported Input Formats

### GeoJSON (Recommended)
- File extensions: `.geojson`, `.json`
- Easy to create and edit
- Human-readable format
- Can be created with any text editor or GIS tool

### Shapefile
- File extension: `.shp`
- Traditional GIS format
- Requires the optional `fiona` dependency
- Install with: `uv add fiona` or `uv sync --extra shapefile`

## Quick Start

### 1. Create a Clipping Boundary

Create a GeoJSON file defining your area of interest:

```json
{
  "type": "Feature",
  "geometry": {
    "type": "Polygon",
    "coordinates": [
      [[100, 100], [500, 100], [500, 500], [100, 500], [100, 100]]
    ]
  },
  "properties": {
    "name": "my_study_area"
  }
}
```

### 2. Enable Clipping in Configuration

Update your config YAML:

```yaml
clipping:
  enabled: true
  boundary_file: path/to/your/boundary.geojson
  feature_name: null  # Use all features, or specify a name
  save_clipped_files: false
```

### 3. Run the Workflow

```bash
uv run scripts/run_workflow.py --config config/profiles/your_config.yaml
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `false` | Enable/disable clipping |
| `boundary_file` | string | `null` | Path to GeoJSON or Shapefile |
| `feature_name` | string | `null` | Specific feature name to use (if file has multiple features) |
| `save_clipped_files` | bool | `false` | Save clipped LAZ files to disk |
| `output_dir` | string | `null` | Directory for clipped files (auto if null) |

## Programmatic Usage

### Basic Clipping

```python
from terrain_change_detection.preprocessing.clipping import AreaClipper
import numpy as np

# From GeoJSON file
clipper = AreaClipper.from_file("boundary.geojson")

# From bounding box
clipper = AreaClipper.from_bounds(min_x=100, min_y=100, max_x=500, max_y=500)

# From polygon coordinates
coords = [(100, 100), (500, 100), (500, 500), (100, 500), (100, 100)]
clipper = AreaClipper.from_polygon(coords)

# Clip points
points = np.random.uniform(0, 600, size=(1000000, 3))
clipped_points = clipper.clip(points)
```

### With Return Mask

```python
clipped_points, mask = clipper.clip(points, return_mask=True)
# mask is a boolean array indicating which points are inside
```

### Clip with Attributes

```python
attributes = {
    'intensity': np.array([...]),
    'classification': np.array([...])
}

clipped_points, clipped_attrs = clipper.clip_with_attributes(points, attributes)
```

### Get Statistics

```python
stats = clipper.get_statistics(points)
print(f"Points inside: {stats['points_inside']} ({stats['percentage_inside']:.1f}%)")
```

### Clip LAZ Files

```python
from terrain_change_detection.preprocessing.clipping import clip_point_cloud_files

output_files = clip_point_cloud_files(
    input_files=["file1.laz", "file2.laz"],
    clipper=clipper,
    output_dir="clipped_data/",
    ground_only=True
)
```

## Creating Boundaries

### Using QGIS
1. Create a new vector layer (Polygon)
2. Draw your area(s) of interest
3. Export as GeoJSON: Right-click layer → Export → Save Features As → GeoJSON

### Using Python
```python
clipper = AreaClipper.from_bounds(min_x, min_y, max_x, max_y)
clipper.save_geojson("my_boundary.geojson")
```

### Using Online Tools
- [geojson.io](https://geojson.io) - Draw polygons on a map and export GeoJSON
- [mapshaper.org](https://mapshaper.org) - Convert and edit geographic data

## Example Configurations

### Focus on Central Area
```yaml
clipping:
  enabled: true
  boundary_file: data/boundaries/central_area.geojson
```

### Focus on River Corridor
```yaml
clipping:
  enabled: true
  boundary_file: data/boundaries/study_areas.geojson
  feature_name: river_corridor
```

### Save Clipped Files for Reuse
```yaml
clipping:
  enabled: true
  boundary_file: data/boundaries/roi.geojson
  save_clipped_files: true
  output_dir: data/clipped/
```

## Test Data

The repository includes test clipping boundaries for the large synthetic dataset:

**File:** `data/large_synthetic/clipping_boundaries.geojson`

**Available polygons:**
- `central_area` - Central 2000m × 2000m rectangle
- `change_zone_1` - Area around mound deposition feature
- `change_zone_2` - Area around erosion pit feature  
- `river_corridor` - Diagonal corridor simulating a river

## Demo Script

Run the demo to see clipping in action:

```bash
uv run scripts/demo_area_clipping.py
```

## Performance Considerations

- **Bounding box pre-filtering**: Points outside the bounding box are quickly rejected
- **Prepared geometry**: Shapely's prepared geometry speeds up repeated point-in-polygon tests
- **Batch processing**: Large point clouds are processed in batches to manage memory

For very large datasets (>10M points), consider:
1. Using rectangular clipping boundaries (faster)
2. Enabling streaming mode in the workflow
3. Saving clipped files for reuse

## Dependencies

**Required:**
- `shapely>=2.0.0` (included in base dependencies)

**Optional (for Shapefile support):**
- `fiona>=1.9.0`
- Install with: `uv add fiona` or `uv sync --extra shapefile`
