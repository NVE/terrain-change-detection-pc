"""
GPU array operations abstraction layer.

Provides transparent NumPy/CuPy switching for seamless CPU/GPU array operations.
Arrays can be on CPU (NumPy) or GPU (CuPy) with automatic detection and conversion.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

import numpy as np

from .hardware_detection import get_gpu_info

logger = logging.getLogger(__name__)

# Type aliases
ArrayType = Union[np.ndarray, Any]  # Any to support cupy.ndarray without import


class ArrayBackend:
    """
    Array backend manager for CPU/GPU operations.
    
    Provides a unified interface for NumPy (CPU) and CuPy (GPU) operations
    with automatic backend selection and graceful fallback.
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize array backend.
        
        Args:
            use_gpu: Whether to use GPU if available (default: True)
        """
        self.use_gpu = use_gpu
        self._gpu_available = False
        self._cp = None
        
        if use_gpu:
            self._initialize_gpu()
    
    def _initialize_gpu(self):
        """Initialize GPU backend if available."""
        gpu_info = get_gpu_info()
        
        if not gpu_info.available:
            logger.info(
                f"GPU not available ({gpu_info.error_message}), using CPU backend"
            )
            return
        
        try:
            import cupy as cp
            self._cp = cp
            self._gpu_available = True
            logger.info(
                f"GPU backend initialized: {gpu_info.device_name} "
                f"({gpu_info.memory_gb:.1f} GB)"
            )
        except ImportError as e:
            logger.warning(f"Failed to import CuPy: {e}, using CPU backend")
    
    @property
    def xp(self):
        """Get array module (numpy or cupy)."""
        if self._gpu_available and self._cp is not None:
            return self._cp
        return np
    
    @property
    def is_gpu(self) -> bool:
        """Check if GPU backend is active."""
        return self._gpu_available
    
    def asarray(self, arr: ArrayType, dtype=None) -> ArrayType:
        """
        Convert array to backend-appropriate array.
        
        Args:
            arr: Input array (numpy or cupy)
            dtype: Optional dtype for conversion
            
        Returns:
            Array on appropriate backend (CPU or GPU)
        """
        if dtype is not None:
            return self.xp.asarray(arr, dtype=dtype)
        return self.xp.asarray(arr)
    
    def to_cpu(self, arr: ArrayType) -> np.ndarray:
        """
        Transfer array to CPU (NumPy).
        
        Args:
            arr: Input array (numpy or cupy)
            
        Returns:
            NumPy array on CPU
        """
        if self._gpu_available and self._cp is not None:
            if isinstance(arr, self._cp.ndarray):
                return self._cp.asnumpy(arr)
        return np.asarray(arr)
    
    def to_gpu(self, arr: np.ndarray) -> ArrayType:
        """
        Transfer array to GPU (CuPy).
        
        Args:
            arr: NumPy array on CPU
            
        Returns:
            CuPy array on GPU, or NumPy array if GPU unavailable
        """
        if not self._gpu_available:
            logger.debug("GPU not available, keeping array on CPU")
            return arr
        
        return self._cp.asarray(arr)
    
    def zeros(self, shape, dtype=np.float64) -> ArrayType:
        """Create array of zeros."""
        return self.xp.zeros(shape, dtype=dtype)
    
    def ones(self, shape, dtype=np.float64) -> ArrayType:
        """Create array of ones."""
        return self.xp.ones(shape, dtype=dtype)
    
    def empty(self, shape, dtype=np.float64) -> ArrayType:
        """Create uninitialized array."""
        return self.xp.empty(shape, dtype=dtype)
    
    def arange(self, *args, **kwargs) -> ArrayType:
        """Create array with evenly spaced values."""
        return self.xp.arange(*args, **kwargs)
    
    def linspace(self, *args, **kwargs) -> ArrayType:
        """Create array with linearly spaced values."""
        return self.xp.linspace(*args, **kwargs)
    
    def concatenate(self, arrays, axis=0) -> ArrayType:
        """Concatenate arrays along axis."""
        return self.xp.concatenate(arrays, axis=axis)
    
    def stack(self, arrays, axis=0) -> ArrayType:
        """Stack arrays along new axis."""
        return self.xp.stack(arrays, axis=axis)
    
    def sqrt(self, arr: ArrayType) -> ArrayType:
        """Element-wise square root."""
        return self.xp.sqrt(arr)
    
    def sum(self, arr: ArrayType, axis=None, keepdims=False) -> ArrayType:
        """Sum array elements."""
        return self.xp.sum(arr, axis=axis, keepdims=keepdims)
    
    def mean(self, arr: ArrayType, axis=None, keepdims=False) -> ArrayType:
        """Compute mean of array elements."""
        return self.xp.mean(arr, axis=axis, keepdims=keepdims)
    
    def std(self, arr: ArrayType, axis=None, keepdims=False) -> ArrayType:
        """Compute standard deviation."""
        return self.xp.std(arr, axis=axis, keepdims=keepdims)
    
    def min(self, arr: ArrayType, axis=None, keepdims=False) -> ArrayType:
        """Find minimum value."""
        return self.xp.min(arr, axis=axis, keepdims=keepdims)
    
    def max(self, arr: ArrayType, axis=None, keepdims=False) -> ArrayType:
        """Find maximum value."""
        return self.xp.max(arr, axis=axis, keepdims=keepdims)
    
    def abs(self, arr: ArrayType) -> ArrayType:
        """Element-wise absolute value."""
        return self.xp.abs(arr)
    
    def clip(self, arr: ArrayType, a_min, a_max) -> ArrayType:
        """Clip array values to range."""
        return self.xp.clip(arr, a_min, a_max)
    
    def where(self, condition, x=None, y=None) -> ArrayType:
        """Return elements chosen from x or y depending on condition."""
        if x is None and y is None:
            return self.xp.where(condition)
        return self.xp.where(condition, x, y)
    
    def isnan(self, arr: ArrayType) -> ArrayType:
        """Test element-wise for NaN."""
        return self.xp.isnan(arr)
    
    def isfinite(self, arr: ArrayType) -> ArrayType:
        """Test element-wise for finiteness."""
        return self.xp.isfinite(arr)
    
    def any(self, arr: ArrayType, axis=None, keepdims=False) -> ArrayType:
        """Test whether any array element is true."""
        return self.xp.any(arr, axis=axis, keepdims=keepdims)
    
    def all(self, arr: ArrayType, axis=None, keepdims=False) -> ArrayType:
        """Test whether all array elements are true."""
        return self.xp.all(arr, axis=axis, keepdims=keepdims)


# Global backend instance
_backend: Optional[ArrayBackend] = None


def get_array_backend(use_gpu: bool = True) -> ArrayBackend:
    """
    Get or create global array backend.
    
    Args:
        use_gpu: Whether to use GPU if available
        
    Returns:
        ArrayBackend instance
        
    Example:
        >>> backend = get_array_backend()
        >>> arr = backend.zeros((1000, 3))  # GPU if available, else CPU
        >>> cpu_arr = backend.to_cpu(arr)
    """
    global _backend
    
    if _backend is None:
        _backend = ArrayBackend(use_gpu=use_gpu)
    
    return _backend


def reset_array_backend():
    """Reset global array backend, forcing reinitialization."""
    global _backend
    _backend = None


def ensure_cpu_array(arr: ArrayType) -> np.ndarray:
    """
    Ensure array is on CPU (NumPy).
    
    Convenience function for getting CPU arrays regardless of backend.
    
    Args:
        arr: Input array (numpy or cupy)
        
    Returns:
        NumPy array on CPU
        
    Example:
        >>> gpu_arr = backend.to_gpu(np.array([1, 2, 3]))
        >>> cpu_arr = ensure_cpu_array(gpu_arr)  # Always returns numpy
    """
    backend = get_array_backend()
    return backend.to_cpu(arr)


def ensure_gpu_array(arr: np.ndarray) -> ArrayType:
    """
    Ensure array is on GPU if available.
    
    Convenience function for getting GPU arrays when possible.
    
    Args:
        arr: NumPy array
        
    Returns:
        GPU array if available, else CPU array
        
    Example:
        >>> cpu_arr = np.array([1, 2, 3])
        >>> gpu_arr = ensure_gpu_array(cpu_arr)  # GPU if available
    """
    backend = get_array_backend()
    return backend.to_gpu(arr)


def is_gpu_array(arr: ArrayType) -> bool:
    """
    Check if array is on GPU.
    
    Args:
        arr: Array to check
        
    Returns:
        True if array is CuPy array on GPU
        
    Example:
        >>> arr = backend.zeros((10, 3))
        >>> if is_gpu_array(arr):
        ...     print("Array is on GPU")
    """
    backend = get_array_backend()
    if backend.is_gpu and backend._cp is not None:
        return isinstance(arr, backend._cp.ndarray)
    return False
