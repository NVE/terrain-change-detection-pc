"""
Terrain Change Detection Package

A Python package for terrain change detection from multi-temporal point cloud data.
This package provides tools for data preprocessing, spatial alignment and change detection.
Data preprocessing is optimized for LiDAR point cloud data from hoydedata.no.
Spatial alignment is implemented using the Iterative Closest Point (ICP) algorithm. For a fine-grained
control of the spatial alignment, the ICP algorithm is implemented from scratch.
Change detection is a WIP...(will most likely use py4dgeo)
"""

__version__ = "0.1.0"
__author__ = "Yared Bekele"
__email__ = "yared.bekele@sintef.no"

from .preprocessing import *
from .alignment import *
from .detection import *
from .utils import *
from .visualization import *

__all__ = [
    "preprocessing",
    "alignment",
    "detection",
    "utils",
    "visualization",
]