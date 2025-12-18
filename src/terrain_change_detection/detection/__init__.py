"""
Terrain Change Detection Module

Exports user-facing change detection interfaces and results.

This module has been refactored into separate submodules for better organization:
- dod.py: DEM of Difference algorithms
- c2c.py: Cloud-to-Cloud comparison algorithms  
- m3c2.py: Multiscale Model to Model Cloud Comparison algorithms

The original unified ChangeDetector interface is maintained for backward compatibility.
"""

# Import from new submodules
from .dod import DoDResult, DoDDetector
from .c2c import C2CResult, C2CDetector
from .m3c2 import M3C2Result, M3C2Params, M3C2Detector


class ChangeDetector:
    """
    Unified interface for change detection algorithms.
    
    This class provides backward compatibility by delegating to the specialized
    detector classes (DoDDetector, C2CDetector, M3C2Detector).
    """
    
    # DoD methods
    compute_dod = staticmethod(DoDDetector.compute_dod)
    compute_dod_streaming_files = staticmethod(DoDDetector.compute_dod_streaming_files)
    compute_dod_streaming_files_tiled = staticmethod(DoDDetector.compute_dod_streaming_files_tiled)
    compute_dod_streaming_files_tiled_parallel = staticmethod(DoDDetector.compute_dod_streaming_files_tiled_parallel)
    
    # C2C methods
    compute_c2c = staticmethod(C2CDetector.compute_c2c)
    compute_c2c_vertical_plane = staticmethod(C2CDetector.compute_c2c_vertical_plane)
    compute_c2c_streaming_files_tiled = staticmethod(C2CDetector.compute_c2c_streaming_files_tiled)
    compute_c2c_streaming_files_tiled_parallel = staticmethod(C2CDetector.compute_c2c_streaming_files_tiled_parallel)
    
    # M3C2 methods
    autotune_m3c2_params = staticmethod(M3C2Detector.autotune_m3c2_params)
    autotune_m3c2_params_from_headers = staticmethod(M3C2Detector.autotune_m3c2_params_from_headers)
    compute_m3c2_original = staticmethod(M3C2Detector.compute_m3c2_original)
    compute_m3c2_streaming_files_tiled = staticmethod(M3C2Detector.compute_m3c2_streaming_files_tiled)
    compute_m3c2_streaming_files_tiled_parallel = staticmethod(M3C2Detector.compute_m3c2_streaming_files_tiled_parallel)
    compute_m3c2_streaming_pertile_parallel = staticmethod(M3C2Detector.compute_m3c2_streaming_pertile_parallel)
    compute_m3c2_plane_based = staticmethod(M3C2Detector.compute_m3c2_plane_based)


# Re-export convenience functions
autotune_m3c2_params = ChangeDetector.autotune_m3c2_params
autotune_m3c2_params_from_headers = ChangeDetector.autotune_m3c2_params_from_headers

__all__ = [
    # Unified interface (backward compatibility)
    "ChangeDetector",
    
    # Result classes
    "DoDResult",
    "C2CResult",
    "M3C2Result",
    "M3C2Params",
    
    # Specialized detector classes (new)
    "DoDDetector",
    "C2CDetector",
    "M3C2Detector",
    
    # Convenience functions
    "autotune_m3c2_params",
    "autotune_m3c2_params_from_headers",
]
