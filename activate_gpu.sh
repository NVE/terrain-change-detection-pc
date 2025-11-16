#!/bin/bash
# Activate virtual environment and set up GPU libraries

# Activate venv
source .venv/bin/activate

# Add CUDA libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${PWD}/.venv/lib/python3.13/site-packages/nvidia/cuda_nvrtc/lib:${PWD}/.venv/lib/python3.13/site-packages/nvidia/cuda_runtime/lib:${PWD}/.venv/lib/python3.13/site-packages/nvidia/cublas/lib:${PWD}/.venv/lib/python3.13/site-packages/nvidia/cufft/lib:${PWD}/.venv/lib/python3.13/site-packages/nvidia/curand/lib:${PWD}/.venv/lib/python3.13/site-packages/nvidia/cusolver/lib:${PWD}/.venv/lib/python3.13/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH"

echo "✓ Virtual environment activated"
echo "✓ CUDA libraries added to LD_LIBRARY_PATH"
echo ""
echo "Test GPU with: python -c 'from terrain_change_detection.acceleration import get_gpu_info; print(get_gpu_info())'"
