"""
GPU hardware detection and capability assessment.

Provides GPU availability detection with graceful CPU fallback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """GPU device information."""
    
    available: bool
    device_count: int
    device_name: Optional[str] = None
    memory_gb: Optional[float] = None
    cuda_version: Optional[str] = None
    compute_capability: Optional[tuple] = None
    error_message: Optional[str] = None


def detect_gpu() -> GPUInfo:
    """
    Detect GPU availability and capabilities.
    
    Attempts to import CuPy and query CUDA devices. Falls back gracefully
    to CPU-only mode if GPU is unavailable or CuPy is not installed.
    
    Returns:
        GPUInfo with device details, or unavailable marker with error message
        
    Example:
        >>> gpu_info = detect_gpu()
        >>> if gpu_info.available:
        ...     print(f"GPU: {gpu_info.device_name}, {gpu_info.memory_gb:.1f} GB")
        ... else:
        ...     print(f"GPU unavailable: {gpu_info.error_message}")
    """
    try:
        import cupy as cp
        
        if not cp.cuda.is_available():
            msg = "CUDA not available - GPU acceleration disabled"
            logger.info(msg)
            return GPUInfo(
                available=False,
                device_count=0,
                error_message="CUDA runtime not available"
            )
        
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count == 0:
            msg = "No CUDA devices found - GPU acceleration disabled"
            logger.info(msg)
            return GPUInfo(
                available=False,
                device_count=0,
                error_message="No CUDA devices detected"
            )
        
        device = cp.cuda.Device(0)
        mem_info = device.mem_info
        
        # Parse compute capability from string format "86" -> (8, 6)
        cc_str = str(device.compute_capability)
        if len(cc_str) >= 2:
            compute_capability = (int(cc_str[0]), int(cc_str[1:]))
        else:
            compute_capability = (int(cc_str), 0)
        
        # Get device name from properties
        device_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode("utf-8")
        
        info = GPUInfo(
            available=True,
            device_count=device_count,
            device_name=device_name,
            memory_gb=mem_info[1] / 1024**3,  # Total memory in GB
            cuda_version=str(cp.cuda.runtime.runtimeGetVersion()),
            compute_capability=compute_capability,
        )
        
        logger.info(
            f"GPU detected: {info.device_name} "
            f"({info.memory_gb:.1f} GB, "
            f"compute {info.compute_capability[0]}.{info.compute_capability[1]})"
        )
        
        return info
        
    except ImportError as e:
        msg = "CuPy not installed - GPU acceleration disabled"
        logger.info(f"{msg}: {e}")
        return GPUInfo(
            available=False,
            device_count=0,
            error_message="CuPy not installed"
        )
        
    except Exception as e:
        msg = f"GPU detection failed - GPU acceleration disabled: {e}"
        logger.warning(msg)
        return GPUInfo(
            available=False,
            device_count=0,
            error_message=str(e)
        )


def check_gpu_memory(required_gb: float) -> tuple[bool, Optional[float]]:
    """
    Check if GPU has sufficient free memory.
    
    Args:
        required_gb: Required memory in gigabytes
        
    Returns:
        Tuple of (has_sufficient_memory, available_gb)
        
    Example:
        >>> has_memory, available = check_gpu_memory(2.0)
        >>> if not has_memory:
        ...     print(f"Insufficient GPU memory: {available:.1f} GB available")
    """
    try:
        import cupy as cp
        
        if not cp.cuda.is_available():
            return False, None
        
        device = cp.cuda.Device(0)
        mem_info = device.mem_info
        free_gb = mem_info[0] / 1024**3
        
        has_sufficient = free_gb >= required_gb
        
        if not has_sufficient:
            logger.warning(
                f"Insufficient GPU memory: {free_gb:.1f} GB available, "
                f"{required_gb:.1f} GB required"
            )
        
        return has_sufficient, free_gb
        
    except ImportError:
        return False, None
    except Exception as e:
        logger.warning(f"Failed to check GPU memory: {e}")
        return False, None


def get_optimal_batch_size(
    point_count: int,
    bytes_per_point: int = 32,
    max_memory_fraction: float = 0.8
) -> int:
    """
    Calculate optimal batch size based on available GPU memory.
    
    Args:
        point_count: Total number of points to process
        bytes_per_point: Estimated memory per point (default: 32 bytes for xyz + features)
        max_memory_fraction: Maximum fraction of GPU memory to use (default: 0.8)
        
    Returns:
        Optimal batch size, or full point_count if GPU unavailable
        
    Example:
        >>> batch_size = get_optimal_batch_size(1_000_000)
        >>> for i in range(0, 1_000_000, batch_size):
        ...     batch = points[i:i+batch_size]
        ...     # Process batch on GPU
    """
    try:
        import cupy as cp
        
        if not cp.cuda.is_available():
            return point_count
        
        device = cp.cuda.Device(0)
        mem_info = device.mem_info
        available_bytes = mem_info[0] * max_memory_fraction
        
        batch_size = int(available_bytes / bytes_per_point)
        batch_size = min(batch_size, point_count)
        batch_size = max(batch_size, 1000)  # Minimum batch size
        
        logger.debug(
            f"Optimal batch size: {batch_size:,} points "
            f"({batch_size * bytes_per_point / 1024**3:.2f} GB)"
        )
        
        return batch_size
        
    except ImportError:
        return point_count
    except Exception as e:
        logger.warning(f"Failed to calculate batch size: {e}")
        return point_count


# Global GPU info cache
_gpu_info_cache: Optional[GPUInfo] = None


def get_gpu_info() -> GPUInfo:
    """
    Get cached GPU information.
    
    Detects GPU on first call and caches result for subsequent calls.
    
    Returns:
        GPUInfo object with current GPU state
    """
    global _gpu_info_cache
    
    if _gpu_info_cache is None:
        _gpu_info_cache = detect_gpu()
    
    return _gpu_info_cache


def clear_gpu_cache():
    """Clear the GPU info cache, forcing re-detection on next get_gpu_info() call."""
    global _gpu_info_cache
    _gpu_info_cache = None
