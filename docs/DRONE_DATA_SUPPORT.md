# Drone Scanning Data Support

## Overview

The terrain change detection pipeline supports two types of point cloud data structures:

1. **Hoydedata.no format**: `area/time_period/data/*.laz`
2. **Drone scanning format**: `area/time_period/*.laz` (no `data/` subdirectory)

Both use the same `DataDiscovery` class - just specify the `source_type` in configuration.

## Directory Structure

### Hoydedata.no Structure
```
data/raw/
├── area1/
│   ├── time_period1/
│   │   ├── data/
│   │   │   ├── file1.laz
│   │   │   └── file2.laz
│   │   └── metadata/
│   └── time_period2/
│       ├── data/
│       │   └── files.laz
│       └── metadata/
```

### Drone Scanning Structure
```
data/drone_scanning_data/
├── jeksla/
│   ├── 2024/
│   │   └── 2024-12-03 Jeksla_ground.las
│   └── 2025/
│       └── 2025-06-15 Jeksla_ground.las
├── veset_nes/
│   └── 2024/
│       └── 2024-12-03 Veset Nes_ground.las
└── vormsund/
    └── 2024/
        └── 2025-05-09 Vormsund_GROUND.las
```

**Key difference**: Drone data files are directly in the time period directory, without a `data/` subdirectory.

## Configuration

### For Drone Data

Use the provided drone profile or add `source_type: drone` to your config:

```yaml
# config/profiles/drone.yaml
paths:
  base_dir: data/drone_scanning_data

discovery:
  source_type: drone  # No 'data' subdirectory required
  
preprocessing:
  ground_only: true
  classification_filter: [2]
```

### For Hoydedata.no

```yaml
# config/default.yaml
paths:
  base_dir: data/raw

discovery:
  source_type: hoydedata  # Requires 'data' subdirectory
  data_dir_name: data
  metadata_dir_name: metadata
```

## Usage

### Python API

```python
from terrain_change_detection.preprocessing import DataDiscovery, BatchLoader

# For drone data
discovery = DataDiscovery("data/drone_scanning_data", source_type='drone')
areas = discovery.scan_areas()

# For hoydedata
discovery = DataDiscovery("data/raw", source_type='hoydedata')
areas = discovery.scan_areas()

# Explore discovered data
for area_name, area_info in areas.items():
    print(f"Area: {area_name}")
    print(f"Time periods: {area_info.time_periods}")
    
    for time_period, dataset in area_info.datasets.items():
        print(f"  {time_period}: {dataset.total_points:,} points in {len(dataset.laz_files)} file(s)")
```

### Command Line

```bash
# Using drone profile
python main.py --config config/profiles/drone.yaml

# Or with default config by specifying source type
python main.py --config config/default.yaml --source-type drone --data-dir data/drone_scanning_data
```

## Loading Data

The same `BatchLoader` works for both data sources:

```python
from terrain_change_detection.preprocessing import DataDiscovery, BatchLoader

# Discover datasets
discovery = DataDiscovery("data/drone_scanning_data", source_type='drone')
areas = discovery.scan_areas()

# Select dataset
area = areas['jeksla']
dataset = area.datasets['2024']

# Load into memory
batch_loader = BatchLoader(streaming_mode=False)
data = batch_loader.load_dataset(dataset)

print(f"Loaded {len(data['points']):,} points")
print(f"Bounds: {data['metadata']['bounds']}")

# Or prepare for streaming/out-of-core processing
batch_loader_streaming = BatchLoader(streaming_mode=True)
streaming_data = batch_loader_streaming.load_dataset(dataset)
print(f"Files: {streaming_data['file_paths']}")
```

## Data Organization Tips

### Drone Data Best Practices

1. **Consistent structure**: Organize as `area/time_period/*.laz`
2. **Clear naming**: Use descriptive area names (e.g., `jeksla`, `veset_nes`)
3. **Time periods**: Use consistent naming for time periods (e.g., year: `2024`, `2025`)
4. **Multiple files**: If an area has multiple files for the same time period, place them all in the same directory
5. **Ground classification**: Ensure files are pre-classified (class 2 for ground points)

### Example Organization

```bash
# Good structure
data/drone_scanning_data/
├── jeksla/
│   ├── 2024/
│   │   ├── scan_part1.las
│   │   └── scan_part2.las
│   └── 2025/
│       └── scan.las
```

## Comparison

| Feature | Hoydedata | Drone |
|---------|-----------|-------|
| Structure | `area/time_period/data/*.laz` | `area/time_period/*.laz` |
| `source_type` | `hoydedata` | `drone` |
| Metadata dir | Supported | Not required |
| File size | Large (GB) | Moderate (MB-GB) |
| Coverage | Regional | Targeted areas |

## Migration from Old Format

If you have drone data in a flat directory with date-prefixed names like:
- `2024-12-03 Jeksla_ground.las`
- `2024-12-03 Veset Nes_ground.las`

Reorganize to:
```bash
data/drone_scanning_data/
├── jeksla/
│   └── 2024/
│       └── 2024-12-03 Jeksla_ground.las
└── veset_nes/
    └── 2024/
        └── 2024-12-03 Veset Nes_ground.las
```

Then use `source_type: drone` in your configuration.

## See Also

- [Configuration Guide](CONFIGURATION_GUIDE.md) - Complete configuration reference
- [Algorithms Documentation](ALGORITHMS.md) - Change detection methods
