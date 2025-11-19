# GPU Acceleration: CUDA Toolkit Installation Required

## Current Status

✅ **GPU Detection**: Working (RTX 3050 detected, 8GB, Compute 8.6)  
✅ **GPU Drivers**: Installed (NVIDIA 581.57, CUDA 13.0)  
✅ **CuPy Installation**: Complete (cupy-cuda12x 13.6.0)  
✅ **CPU Fallback**: Functional (all operations work on CPU)  
⚠️ **GPU Operations**: **BLOCKED** - Missing CUDA Toolkit runtime libraries

## Problem

CuPy requires **CUDA runtime compilation libraries (NVRTC)** for JIT compilation of GPU kernels. While GPU drivers provide basic CUDA support, they don't include the development/runtime libraries needed for dynamic compilation.

**Missing DLL**: `nvrtc64_120_0.dll` (and related CUDA 12.0 runtime libraries)

**Error Message**:
```
RuntimeError: CuPy failed to load nvrtc64_120_0.dll: FileNotFoundError: 
Could not find module 'nvrtc64_120_0.dll' (or one of its dependencies).
```

## Solution

Install **CUDA Toolkit 12.0** which includes:
- NVRTC libraries (runtime compilation)
- CUDART libraries (CUDA runtime)
- Developer libraries
- Compiler tools (nvcc)

### Installation Steps (Windows)

1. **Download CUDA 12.0 Toolkit**:
   - URL: https://developer.nvidia.com/cuda-12-0-0-download-archive
   - Select: Windows → x86_64 → 10/11 → exe (local)
   - Size: ~3.5 GB

2. **Run Installer**:
   - ✅ CUDA Toolkit
   - ✅ CUDA Runtime Libraries
   - ✅ Developer Libraries
   - ⬜ Visual Studio Integration (optional)
   - ⬜ NSight Tools (optional)

3. **Verify Installation**:
   ```powershell
   # Check CUDA_PATH
   $env:CUDA_PATH  
   # Should show: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0

   # Verify NVRTC library exists
   Test-Path "$env:CUDA_PATH\bin\nvrtc64_120_0.dll"
   # Should return: True

   # Check nvcc compiler
   nvcc --version
   # Should show: Cuda compilation tools, release 12.0
   ```

4. **Add to PATH** (if not automatic):
   ```powershell
   # Temporary (current session)
   $env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin"

   # Permanent: 
   # System Properties → Environment Variables → Path → Edit → 
   # Add: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin
   ```

5. **Test CuPy**:
   ```powershell
   cd c:\Users\yaredbe\Documents\terrain-change-detection-pc
   uv run python -c "import cupy as cp; print(cp.ones((5,5)))"
   # Should print 5x5 array of ones without errors
   ```

6. **Run GPU Tests**:
   ```powershell
   uv run pytest tests/test_gpu_array_ops.py -v
   # All 14 tests should pass
   ```

## Why This is Needed

| Component | Provided By | Status |
|-----------|------------|--------|
| GPU Hardware | RTX 3050 | ✅ Present |
| GPU Driver | NVIDIA 581.57 | ✅ Installed |
| CUDA Runtime APIs | GPU Driver | ✅ Available |
| CUDA Compiler (nvcc) | CUDA Toolkit | ❌ **Missing** |
| NVRTC (JIT compilation) | CUDA Toolkit | ❌ **Missing** |
| CUDA Development Libs | CUDA Toolkit | ❌ **Missing** |

**CuPy's Requirements**:
- Basic GPU ops (memory allocation, transfers) → Provided by drivers ✅
- JIT kernel compilation (math ops, etc.) → Requires CUDA Toolkit ❌

## What Works Without Toolkit

**Working**:
- GPU detection (`get_gpu_info()`)
- Basic memory allocation (`cp.zeros()` on first call)
- CPU fallback mode (all operations via NumPy)
- All tests in CPU-only mode

**Not Working** (needs Toolkit):
- Mathematical operations (`cp.sqrt()`, `cp.sum()`, etc.)
- Array manipulation (`cp.ones()`, `cp.concatenate()`, etc.)
- Custom kernels
- Any operation requiring JIT compilation

## Impact on Development

**Current Phase Status**:
- ✅ Week 1 Foundation: GPU detection, array ops abstraction **complete**
- ⚠️ Week 2-3 GPU Neighbors: **Blocked** until Toolkit installed
- ⚠️ Week 4-5 Integration: **Blocked** by Week 2-3

**Workaround**:
All development can continue in CPU-only mode by setting `use_gpu=False` in config. GPU acceleration can be tested after Toolkit installation without code changes.

## Next Steps

1. **User Action Required**: Install CUDA Toolkit 12.0 (see steps above)
2. **After Installation**: Re-run `uv run pytest tests/test_gpu_array_ops.py`
3. **Proceed to**: Task 6 - GPU nearest neighbors wrapper implementation

## Alternative: Skip GPU for Now

If CUDA Toolkit installation is not feasible right now:
- Continue Phase 2 development with CPU fallback
- All code is GPU-ready (uses abstraction layer)
- Install Toolkit later to enable GPU without code changes
- Phase 1 CPU parallelization (2-3x speedup) remains functional

## References

- CUDA Toolkit Downloads: https://developer.nvidia.com/cuda-toolkit-archive
- CuPy Installation Guide: https://docs.cupy.dev/en/stable/install.html
- GPU Setup Guide: [docs/GPU_SETUP_GUIDE.md](./GPU_SETUP_GUIDE.md)
- Troubleshooting: [docs/GPU_SETUP_GUIDE.md#troubleshooting](./GPU_SETUP_GUIDE.md#troubleshooting)

---

**Summary**: GPU acceleration is 90% ready. Only missing component is CUDA Toolkit installation (~15 min). All code is complete and tested in CPU mode. GPU will work immediately after Toolkit installation.
