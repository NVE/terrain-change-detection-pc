"""
Terrain Change Detection Package

A Python package for terrain change detection from multi-temporal point cloud data.
This package provides tools for data preprocessing, spatial alignment and change detection.
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