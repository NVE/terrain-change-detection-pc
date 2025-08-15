import laspy
import numpy as np
from pathlib import Path
import time
import sys

# Start timing
start_time = time.time()

# Load the LAZ file
print(f"Loading file...")
file_path = Path("33-1-466-136-13.laz")

# Check file exists first
if not file_path.exists():
    print(f"Error: File {file_path} not found!")
    sys.exit(1)

# Use memory-mapped loading for better performance with large files
las_file = laspy.read(file_path)
print(f"File loaded successfully!")
print(f"Point format: {las_file.header.point_format}")
print(f"Number of points: {len(las_file.points):,}")

# Display available point dimensions/attributes
print("Available point dimensions:")
for dim in las_file.point_format.dimension_names:
    print(f"  - {dim}")

# Basic coordinate statistics - use vectorized operations
print("Coordinate ranges:")
print(f"X: {las_file.x.min():.2f} to {las_file.x.max():.2f}")
print(f"Y: {las_file.y.min():.2f} to {las_file.y.max():.2f}")
print(f"Z: {las_file.z.min():.2f} to {las_file.z.max():.2f}")

# Header information
print("Header information:")
print(f"Version: {las_file.header.version}")
print(f"Creation date: {las_file.header.creation_date}")
print(f"Scale factors: X={las_file.header.x_scale}, Y={las_file.header.y_scale}, Z={las_file.header.z_scale}")
print(f"Offsets: X={las_file.header.x_offset}, Y={las_file.header.y_offset}, Z={las_file.header.z_offset}")

# Point classification analysis
print("\n" + "="*50)
print("POINT CLASSIFICATION ANALYSIS")
print("="*50)

# Check if classification data exists
if hasattr(las_file, 'classification'):
    classifications = las_file.classification
    # Use numpy's efficient unique function with return_counts for better performance
    unique_classes, counts = np.unique(classifications, return_counts=True)

    print(f"Classification field found!")
    print(f"Unique classification codes: {unique_classes}")
    print(f"Total unique classes: {len(unique_classes)}")

    # Standard LAS classification meanings (ASPRS standard)
    class_meanings = {
        0: "Created, never classified",
        1: "Unclassified",
        2: "Ground",
        3: "Low Vegetation",
        4: "Medium Vegetation",
        5: "High Vegetation",
        6: "Building",
        7: "Low Point (noise)",
        8: "Reserved",
        9: "Water",
        10: "Rail",
        11: "Road Surface",
        12: "Reserved",
        13: "Wire - Guard (Shield)",
        14: "Wire - Conductor (Phase)",
        15: "Transmission Tower",
        16: "Wire-structure Connector (e.g. Insulator)",
        17: "Bridge Deck",
        18: "High Noise"
    }

    print("\nClassification breakdown:")
    for class_code, count in zip(unique_classes, counts):
        percentage = (count / len(classifications)) * 100
        meaning = class_meanings.get(class_code, "User defined or reserved")
        print(f"  Class {class_code:2d}: {count:8,} points ({percentage:5.1f}%) - {meaning}")
else:
    print("No classification field found in this file.")

# End timing and display results
end_time = time.time()
execution_time = end_time - start_time
print(f"\nExecution completed in {execution_time:.2f} seconds")