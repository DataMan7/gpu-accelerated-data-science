#!/usr/bin/env python3
"""
GPU-Accelerated Matrix Multiplication using CuPy
Demonstrates the performance difference between CPU and GPU computation
"""

import time
import numpy as np
import cupy as cp

def cpu_matrix_multiply(A, B):
    """CPU-based matrix multiplication using NumPy"""
    return np.dot(A, B)

def gpu_matrix_multiply(A, B):
    """GPU-based matrix multiplication using CuPy"""
    # Transfer data to GPU
    A_gpu = cp.asarray(A)
    B_gpu = cp.asarray(B)

    # Perform multiplication on GPU
    C_gpu = cp.dot(A_gpu, B_gpu)

    # Transfer result back to CPU
    return cp.asnumpy(C_gpu)

def benchmark_matrix_multiplication(sizes):
    """Benchmark CPU vs GPU performance for different matrix sizes"""
    results = []

    for size in sizes:
        print(f"\nBenchmarking {size}x{size} matrices...")

        # Generate random matrices
        A = np.random.rand(size, size).astype(np.float32)
        B = np.random.rand(size, size).astype(np.float32)

        # CPU benchmark
        start_time = time.time()
        C_cpu = cpu_matrix_multiply(A, B)
        cpu_time = time.time() - start_time

        # GPU benchmark
        start_time = time.time()
        C_gpu = gpu_matrix_multiply(A, B)
        gpu_time = time.time() - start_time

        # Verify results
        max_error = np.max(np.abs(C_cpu - C_gpu))
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')

        results.append({
            'size': size,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup,
            'max_error': max_error
        })

        print(".4f")
        print(".4f")
        print(".2f")
        print(".2e")

    return results

def main():
    print("GPU-Accelerated Matrix Multiplication Benchmark")
    print("=" * 50)

    # Check CuPy installation and GPU availability
    print(f"CuPy version: {cp.__version__}")
    print(f"CUDA available: {cp.cuda.is_available()}")
    if cp.cuda.is_available():
        print(f"GPU device: {cp.cuda.Device(0).name}")
        print(f"GPU memory: {cp.cuda.Device(0).mem_info[0] / 1024**3:.1f} GB total")

    # Define matrix sizes to test
    sizes = [256, 512, 1024, 2048]

    # Run benchmarks
    results = benchmark_matrix_multiply(sizes)

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print("<8")
    print("-" * 50)

    for result in results:
        print("<8")

    print("\nNote: GPU performance includes data transfer overhead.")
    print("For larger matrices, the GPU speedup becomes more significant.")

if __name__ == "__main__":
    main()