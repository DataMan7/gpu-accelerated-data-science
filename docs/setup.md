# Environment Setup Guide

## Hardware Requirements

### Minimum Requirements
- NVIDIA GPU with CUDA compute capability 3.5 or higher
- At least 4GB GPU memory (8GB recommended)
- Compatible NVIDIA drivers

### Recommended Hardware
- NVIDIA RTX 30-series or A-series GPUs
- 16GB+ GPU memory
- High-bandwidth memory (GDDR6X preferred)

## Software Installation

### Step 1: Install NVIDIA Drivers
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-driver-XXX  # Replace XXX with latest version

# Verify installation
nvidia-smi
```

### Step 2: Install CUDA Toolkit
Download from NVIDIA Developer website or use package manager:

```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get install cuda
```

### Step 3: Install cuDNN
1. Download cuDNN from NVIDIA Developer website
2. Extract and copy to CUDA installation directory
3. Update environment variables

### Step 4: Python Environment Setup
```bash
# Create virtual environment
python3 -m venv gpu-env
source gpu-env/bin/activate

# Install GPU libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install cupy-cuda11x
pip install cudf dask-cudf  # RAPIDS
pip install tensorflow-gpu
pip install numba
```

## Environment Variables

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda
```

## Verification

### CUDA Installation
```bash
nvcc --version
nvidia-smi
```

### Python Libraries
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

import cupy as cp
print(cp.cuda.runtime.getDeviceCount())
```

## Development Tools

### NVIDIA Nsight
- Nsight Systems: System-wide performance analysis
- Nsight Compute: Kernel-level optimization

### CUDA-GDB
GPU debugging capabilities

### Profiling
```bash
# Profile CUDA applications
nvprof ./your_cuda_program

# Modern profiling (CUDA 10.0+)
nsys profile ./your_program
```

## Troubleshooting

### Common Issues

1. **CUDA version compatibility**: Ensure GPU drivers match CUDA toolkit version
2. **Memory errors**: Check GPU memory usage with `nvidia-smi`
3. **Library conflicts**: Use virtual environments to isolate installations
4. **Permission issues**: Ensure user has access to GPU devices

### Performance Tuning

- Monitor GPU utilization with `nvidia-smi -l 1`
- Use `nvtop` for real-time GPU monitoring
- Profile applications to identify bottlenecks

## Docker Setup (Alternative)

For containerized development:

```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Install Python GPU libraries
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install cupy-cuda11x cudf

# Set working directory
WORKDIR /workspace
```

Run with GPU support:
```bash
docker run --gpus all -it your-gpu-image
```

This setup provides a solid foundation for GPU-accelerated data science development.