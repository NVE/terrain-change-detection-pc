"""
Terrain Change Detection Module

Exports user-facing change detection interfaces and results.
"""

from .change_detection import (
	ChangeDetector,
	DoDResult,
	C2CResult,
	M3C2Result,
	M3C2Params,
)

# Re-export function for convenience
autotune_m3c2_params = ChangeDetector.autotune_m3c2_params
autotune_m3c2_params_from_headers = ChangeDetector.autotune_m3c2_params_from_headers

__all__ = [
	"ChangeDetector",
	"DoDResult",
	"C2CResult",
	"M3C2Result",
	"M3C2Params",
	"autotune_m3c2_params",
	"autotune_m3c2_params_from_headers",
]
