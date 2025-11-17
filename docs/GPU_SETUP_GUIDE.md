# GPU Acceleration Setup Guide

This guide covers setting up GPU acceleration for the terrain change detection pipeline (Phase 2).

## Prerequisites

### Hardware Requirements
- **NVIDIA GPU** with CUDA support
- **Compute Capability**: 6.0 or higher (Pascal architecture or newer)
  - GTX 10-series or newer
  - RTX 20/30/40-series
  - Tesla P100 or newer
  - A100, H100, etc.
- **GPU Memory**: Minimum 4 GB recommended, 8+ GB optimal
- **System RAM**: 16 GB minimum, 32 GB recommended

### Software Requirements
- **CUDA Toolkit**: Version 12.x (required for Windows, see installation below)
  - Download from: https://developer.nvidia.com/cuda-12-0-0-download-archive
  - **Critical**: Install the full toolkit, not just drivers
  - Required DLLs: `nvrtc64_120_0.dll`, `cudart64_12.dll`, etc.
- **NVIDIA Driver**: Latest stable version (581.x+ recommended)
- **Python**: 3.13+
- **Operating System**: Windows 10/11, Linux (Ubuntu 20.04+)

**Note for Windows Users**: Even if `nvidia-smi` shows CUDA support, you may still need to install the CUDA Toolkit to get runtime compilation libraries (NVRTC). See troubleshooting below.

## Installation

### Step 1: Install CUDA Toolkit (Windows)

**Why this is needed**: CuPy requires CUDA runtime libraries for JIT compilation. While GPU drivers provide basic CUDA support, the full toolkit includes essential libraries like NVRTC.

1. Download CUDA 12.0 Toolkit:
   - Visit: https://developer.nvidia.com/cuda-12-0-0-download-archive
   - Select: Windows → x86_64 → Version 10/11 → exe (local)
   
2. Run installer and select:
   - ✅ CUDA Toolkit
   - ✅ CUDA Runtime
   - ✅ Developer libraries
   - ⬜ Visual Studio Integration (optional)
   - ⬜ NSight tools (optional)

3. Verify installation:
```powershell
# Check CUDA_PATH environment variable
$env:CUDA_PATH  # Should show: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0

# Check for NVRTC library
Test-Path "$env:CUDA_PATH\bin\nvrtc64_120_0.dll"  # Should return: True

# Add to PATH if needed (restart terminal after)
$env:PATH += ";$env:CUDA_PATH\bin"
```

### Step 2: Verify CUDA Installation

Check if CUDA is properly installed:

```powershell
# Windows - check driver and toolkit
nvidia-smi  # Should show CUDA version
nvcc --version  # Should show CUDA compiler version

# Linux
nvidia-smi
nvcc --version
```

### Step 3: Install GPU Dependencies

Install the GPU acceleration packages:

```powershell
# From project root
uv sync --extra gpu
```

This installs:
- **CuPy** (≥13.0.0): NumPy-compatible GPU arrays
- **Numba** (≥0.59.0): JIT compilation with CUDA support
- **cuML** (≥24.0.0): GPU-accelerated ML algorithms (Linux only)

**Note on Windows**: cuML is currently Linux-only. On Windows, we use CuPy for GPU arrays with sklearn for nearest neighbors, which still provides significant acceleration for array operations.

### Step 3: Verify Installation

Test GPU detection:

```powershell
# Run GPU detection test
uv run pytest tests/test_gpu_detection.py -v

# Or run directly
uv run python -c "from terrain_change_detection.acceleration import get_gpu_info; print(get_gpu_info())"
```

Expected output (if GPU available):
```
GPUInfo(available=True, device_count=1, device_name='NVIDIA GeForce RTX 3080', 
        memory_gb=10.0, cuda_version='12000', compute_capability=(8, 6))
```

Expected output (if GPU unavailable):
```
GPUInfo(available=False, device_count=0, error_message='CuPy not installed')
```

## Configuration

GPU acceleration is controlled via YAML configuration:

```yaml
# config/default.yaml
acceleration:
  use_gpu: true              # Enable GPU acceleration
  gpu_memory_limit_gb: 8.0   # Max GPU memory to use
  gpu_batch_size: null       # Auto-calculate if null
  fallback_to_cpu: true      # Fallback to CPU if GPU fails
```

## Usage

### Automatic GPU Usage

GPU acceleration is automatically applied when available:

```python
from terrain_change_detection.detection import ChangeDetector

# GPU will be used automatically if available
detector = ChangeDetector(config)
c2c_result = detector.compute_c2c_streaming_tiled(...)  # Uses GPU for NN searches
```

### Manual GPU Control

Explicitly control GPU usage:

```python
from terrain_change_detection.acceleration import get_gpu_info

# Check GPU availability
gpu_info = get_gpu_info()
if gpu_info.available:
    print(f"Using GPU: {gpu_info.device_name}")
else:
    print(f"GPU unavailable: {gpu_info.error_message}")

# Check memory before processing
from terrain_change_detection.acceleration import check_gpu_memory
has_memory, available_gb = check_gpu_memory(required_gb=4.0)
```

## Performance Tips

### 1. Optimal Batch Sizes
Let the system auto-calculate batch sizes based on GPU memory:

```python
from terrain_change_detection.acceleration import get_optimal_batch_size

batch_size = get_optimal_batch_size(
    point_count=1_000_000,
    bytes_per_point=32,
    max_memory_fraction=0.8
)
```

### 2. Memory Management
- Keep `max_memory_fraction` at 0.8 to leave headroom for CUDA operations
- Monitor GPU memory with `nvidia-smi` during processing
- Use smaller tiles if encountering OOM errors

### 3. Multi-GPU (Future)
Currently supports single GPU only. Multi-GPU support planned for Phase 3.

## Troubleshooting

### "CUDA not available"
**Issue**: GPU detected but CUDA runtime not available

**Solutions**:
1. Verify CUDA Toolkit is installed: `nvcc --version`
2. Check NVIDIA driver is up to date: `nvidia-smi`
3. Ensure CuPy matches your CUDA version:
   ```powershell
   # For CUDA 11.x
   uv pip install cupy-cuda11x
   
   # For CUDA 12.x
   uv pip install cupy-cuda12x
   ```

### "CuPy not installed"
**Issue**: GPU packages not installed

**Solution**:
```powershell
uv pip install -e ".[gpu]"
```

### "Could not find module 'nvrtc64_120_0.dll'" (Windows)
**Issue**: CuPy cannot find CUDA runtime compilation library

**Root cause**: CUDA Toolkit not installed or not in PATH. GPU drivers alone don't include NVRTC.

**Solutions**:
1. **Install CUDA Toolkit 12.0** (recommended):
   - Download: https://developer.nvidia.com/cuda-12-0-0-download-archive
   - Install with "CUDA Runtime" and "Developer libraries" components
   - Restart terminal and verify: `Test-Path "$env:CUDA_PATH\bin\nvrtc64_120_0.dll"`

2. **Add CUDA bin to PATH**:
   ```powershell
   $env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin"
   # Add permanently via System Environment Variables
   ```

3. **Verify installation**:
   ```powershell
   # Should show: True
   Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\nvrtc64_120_0.dll"
   
   # Re-run detection
   uv run python -m terrain_change_detection.acceleration.hardware_detection
   ```

**Note**: Even if `nvidia-smi` shows CUDA 13.0, you still need Toolkit 12.0 for CuPy compatibility. The versions can coexist.

### "Insufficient GPU memory"
**Issue**: GPU runs out of memory during processing

**Solutions**:
1. Reduce tile size in configuration
2. Lower `gpu_memory_limit_gb` in config
3. Enable `fallback_to_cpu` for large operations
4. Close other GPU applications

### "No CUDA devices detected"
**Issue**: GPU not visible to CUDA

**Solutions**:
1. Check GPU is properly seated
2. Update NVIDIA driver
3. Verify GPU supports CUDA: https://developer.nvidia.com/cuda-gpus
4. Check device manager (Windows) or `lspci | grep -i nvidia` (Linux)

### Performance Slower Than CPU
**Possible causes**:
1. **Small datasets**: GPU overhead dominates for < 100K points
2. **Data transfer bottleneck**: Processing on CPU might be faster for small tiles
3. **Old GPU**: Compute capability < 6.0 has limited performance

**Solutions**:
- Use larger tile sizes
- Disable GPU for small datasets
- Upgrade GPU hardware

## Expected Performance Gains

With GPU acceleration enabled:

| Operation | Dataset Size | CPU Time | GPU Time | Speedup | Status |
|-----------|-------------|----------|----------|---------|--------|
| C2C (NN search) | 1M points | 100s | 5-10s | 10-20x | ✅ Complete |
| M3C2 (cylindrical NN) | 1M points | 200s | 10-30s | 10-20x | ⚠️ Limited (py4dgeo) |
| DoD (grid accumulation) | 1M points | 0.07s | 0.02s | 1.5-4x | ✅ Complete |
| ICP Alignment | 50K points | 0.78s | 3.5s | 0.28x | ⚠️ CPU faster (small data) |

### GPU Acceleration Status by Method

#### ✅ Cloud-to-Cloud (C2C) - Fully GPU Accelerated
- **Operations**: Nearest neighbor searches (k-NN and radius)
- **Speedup**: 10-20x on large datasets (100K+ points)
- **Platform**: Linux (cuML), Windows (limited, sklearn with GPU arrays)
- **Recommended**: Production workloads with large point clouds

#### ✅ DEM of Difference (DoD) - GPU Accelerated
- **Operations**: Grid accumulation and aggregation
- **Speedup**: 1.5-4x on large datasets (100K-1M points)
- **Best performance**: Large grids (250K+ cells) with many points
- **Configuration**: Set `gpu.use_for_preprocessing=true`
- **Benchmarks**: 
  - 100K points: 1.65x speedup
  - 1M points, large grid: 3.82x speedup
- **Notes**: GPU overhead significant for small datasets (< 10K points)

#### ⚠️ M3C2 - CPU Parallelization Only
- **Limitation**: py4dgeo uses C++ KDTree (no Python hooks for GPU)
- **Alternative**: CPU parallelization provides 2-3x speedup
- **Performance**: py4dgeo C++ KDTree already optimized (within 2-5x of GPU)
- **Future**: Custom GPU M3C2 possible but requires 2-3 weeks implementation

#### ⚠️ ICP Alignment - CPU Recommended for Small Data
- **GPU available**: cuML neighbors for alignment
- **Performance**: CPU faster for typical alignment datasets (< 100K points)
- **GPU overhead**: Data transfer + kernel launch dominates at small scale
- **Recommendation**: Use CPU unless aligning very large clouds

### Combined Speedup with CPU Parallelization

GPU + CPU parallelization provides multiplicative speedup:

| Method | CPU Parallel | GPU Accel | Combined | Total Speedup |
|--------|--------------|-----------|----------|---------------|
| C2C | 2-3x | 10-20x | 2-3x × 10-20x | **20-60x** |
| DoD | 2-3x | 1.5-4x | 2-3x × 1.5-4x | **3-12x** |
| M3C2 | 2-3x | 1x | 2-3x | **2-3x** |

**Overall Pipeline**: **20-60x faster** than sequential CPU on production workloads.

## Development

### Testing GPU Code

Run GPU-specific tests:
```powershell
# Run all GPU tests
uv run pytest tests/test_gpu_*.py -v

# Run with GPU marker
uv run pytest -m gpu -v

# Skip GPU tests
uv run pytest -m "not gpu" -v
```

### Debugging

Enable verbose GPU logging:
```python
import logging
logging.getLogger('terrain_change_detection.acceleration').setLevel(logging.DEBUG)
```

Monitor GPU utilization:
```powershell
# Watch GPU usage in real-time
watch -n 1 nvidia-smi  # Linux
nvidia-smi -l 1        # Windows
```

## References

- **CuPy Documentation**: https://docs.cupy.dev/
- **cuML Documentation**: https://docs.rapids.ai/api/cuml/stable/
- **Numba CUDA**: https://numba.readthedocs.io/en/stable/cuda/
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-toolkit

## Support

For GPU-related issues:
1. Check `docs/GPU_ACCELERATION_PLAN.md` for implementation details
2. Review `docs/ROADMAP.md` for current status
3. Open an issue on GitHub with GPU info from `get_gpu_info()`
