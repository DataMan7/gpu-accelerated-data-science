# Chapter 2: CUDA Fundamentals

## Understanding GPU Architecture

### CPU vs GPU Design Philosophy

Traditional CPUs are designed for sequential processing with complex control logic, branch prediction, and large caches to minimize latency. GPUs, however, prioritize throughput over latency, featuring:

- Thousands of simpler cores
- Massive parallel processing capabilities
- Optimized for data-parallel workloads

### NVIDIA GPU Hierarchy

#### Streaming Multiprocessors (SMs)
Each SM contains:
- Multiple CUDA cores (typically 64-128)
- Shared memory
- Registers
- Texture and constant caches

#### Memory Hierarchy
- **Global Memory**: Largest, slowest (off-chip GDDR)
- **Shared Memory**: Fast, on-chip, programmable
- **Registers**: Fastest, per-thread storage
- **Constant Memory**: Read-only, cached
- **Texture Memory**: Optimized for 2D/3D access patterns

## CUDA Programming Model

### Kernels
Functions that execute on the GPU, launched with `<<<>>>` syntax:

```cuda
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

### Thread Hierarchy
- **Thread**: Basic execution unit
- **Block**: Group of threads sharing shared memory
- **Grid**: Collection of blocks

### Memory Management
Proper memory allocation and transfer is crucial:

```cuda
// Allocate device memory
float* d_a;
cudaMalloc(&d_a, size);

// Copy data from host to device
cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

// Launch kernel
vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);

// Copy results back
cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

// Free device memory
cudaFree(d_a);
```

## CUDA-X Libraries Overview

### cuBLAS: Basic Linear Algebra
High-performance matrix operations:

```c
// Matrix multiplication: C = alpha*A*B + beta*C
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k, &alpha, d_A, lda, d_B, ldb,
            &beta, d_C, ldc);
```

### cuDNN: Deep Learning Primitives
Optimized neural network operations:

```c
// Convolution forward pass
cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input,
                       filterDesc, d_filter, convDesc,
                       algo, workspace, workspaceSize,
                       &beta, outputDesc, d_output);
```

### cuFFT: Fast Fourier Transforms
Signal processing capabilities:

```c
// Create plan
cufftPlan1d(&plan, NX, CUFFT_C2C, 1);

// Execute FFT
cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
```

## Performance Optimization Strategies

### Memory Coalescing
Ensure threads access contiguous memory locations to maximize bandwidth utilization.

### Occupancy Optimization
Balance registers, shared memory, and thread blocks per SM for maximum parallelism.

### Asynchronous Operations
Use streams for concurrent execution of memory transfers and kernel launches.

## Error Handling and Debugging

### CUDA Error Checking
Always check return codes:

```cuda
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
}
```

### Profiling Tools
- NVIDIA Nsight Systems for timeline analysis
- NVIDIA Nsight Compute for kernel optimization
- CUDA profilers for performance metrics

## Best Practices

1. **Minimize Host-Device Transfers**: Keep data on GPU when possible
2. **Use Appropriate Data Types**: float32 vs float64 based on precision needs
3. **Optimize Memory Access Patterns**: Coalesced access for global memory
4. **Leverage Shared Memory**: For data reuse within thread blocks
5. **Profile Regularly**: Use tools to identify bottlenecks

## Integration with Data Science Workflows

CUDA acceleration integrates seamlessly with popular frameworks:
- **PyTorch**: Native CUDA tensor operations
- **TensorFlow**: GPU-accelerated computations
- **RAPIDS**: GPU DataFrame operations
- **CuPy**: NumPy-compatible GPU arrays

This foundation in CUDA fundamentals prepares you for advanced GPU programming techniques covered in subsequent chapters.

---

*Note: This chapter provides theoretical foundations. Practical implementations and code examples will be developed in later phases.*