#!/usr/bin/env python3
"""
Performance Benchmarking Suite for GPU-Accelerated Data Science Operations
Compares CPU vs GPU performance across various data science workloads
"""

import time
import numpy as np
import pandas as pd
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available. GPU benchmarks will be skipped.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Some benchmarks will be skipped.")

class PerformanceBenchmark:
    def __init__(self):
        self.results = []

    def time_function(self, func, *args, **kwargs):
        """Time a function execution"""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time

    def benchmark_operation(self, name, cpu_func, gpu_func=None, *args, **kwargs):
        """Benchmark CPU vs GPU performance for an operation"""
        print(f"\nBenchmarking: {name}")

        # CPU benchmark
        try:
            cpu_result, cpu_time = self.time_function(cpu_func, *args, **kwargs)
            print(".4f")
        except Exception as e:
            print(f"CPU benchmark failed: {e}")
            return

        # GPU benchmark
        if gpu_func and CUPY_AVAILABLE:
            try:
                gpu_result, gpu_time = self.time_function(gpu_func, *args, **kwargs)
                speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')

                # Verify results match (for numerical operations)
                if hasattr(cpu_result, 'shape') and hasattr(gpu_result, 'shape'):
                    if cpu_result.shape == gpu_result.shape:
                        max_error = np.max(np.abs(cpu_result - gpu_result))
                        print(".4f")
                        print(".2f")
                        print(".2e")
                    else:
                        print(".4f")
                        print("  Results shape mismatch")
                else:
                    print(".4f")
                    print(".2f")

            except Exception as e:
                print(f"GPU benchmark failed: {e}")
        elif gpu_func and not CUPY_AVAILABLE:
            print("  GPU benchmark skipped (CuPy not available)")
        else:
            print("  No GPU implementation provided")

        # Store results
        self.results.append({
            'operation': name,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time if 'gpu_time' in locals() else None,
            'speedup': speedup if 'speedup' in locals() else None
        })

def matrix_operations_benchmark(benchmark, sizes):
    """Benchmark matrix operations"""

    def cpu_matrix_multiply(A, B):
        return np.dot(A, B)

    def gpu_matrix_multiply(A, B):
        A_gpu = cp.asarray(A)
        B_gpu = cp.asarray(B)
        C_gpu = cp.dot(A_gpu, B_gpu)
        return cp.asnumpy(C_gpu)

    for size in sizes:
        A = np.random.rand(size, size).astype(np.float32)
        B = np.random.rand(size, size).astype(np.float32)

        benchmark.benchmark_operation(
            f"Matrix Multiply {size}x{size}",
            cpu_matrix_multiply, gpu_matrix_multiply, A, B
        )

def array_operations_benchmark(benchmark, sizes):
    """Benchmark element-wise array operations"""

    def cpu_array_ops(A, B, C):
        return np.sin(A) + np.cos(B) * np.exp(C)

    def gpu_array_ops(A, B, C):
        A_gpu = cp.asarray(A)
        B_gpu = cp.asarray(B)
        C_gpu = cp.asarray(C)
        result = cp.sin(A_gpu) + cp.cos(B_gpu) * cp.exp(C_gpu)
        return cp.asnumpy(result)

    for size in sizes:
        A = np.random.rand(size).astype(np.float32)
        B = np.random.rand(size).astype(np.float32)
        C = np.random.rand(size).astype(np.float32)

        benchmark.benchmark_operation(
            f"Array Operations (size {size})",
            cpu_array_ops, gpu_array_ops, A, B, C
        )

def data_processing_benchmark(benchmark, sizes):
    """Benchmark data processing operations"""

    def cpu_data_processing(df):
        # Simulate data processing pipeline
        result = df.copy()
        result['normalized'] = (result['value'] - result['value'].mean()) / result['value'].std()
        result['log_transform'] = np.log(result['value'] + 1)
        result['rolling_mean'] = result['value'].rolling(window=10).mean()
        return result

    # Note: GPU data processing would typically use RAPIDS cuDF
    # For this example, we'll focus on CPU-only operations

    for size in sizes:
        data = {
            'id': range(size),
            'value': np.random.rand(size) * 100,
            'category': np.random.choice(['A', 'B', 'C'], size)
        }
        df = pd.DataFrame(data)

        benchmark.benchmark_operation(
            f"Data Processing (size {size})",
            cpu_data_processing, None, df
        )

def main():
    print("GPU-Accelerated Data Science Performance Benchmark")
    print("=" * 60)

    # Check available libraries
    print(f"NumPy version: {np.__version__}")
    if CUPY_AVAILABLE:
        print(f"CuPy version: {cp.__version__}")
        print(f"CUDA available: {cp.cuda.is_available()}")
    if TORCH_AVAILABLE:
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")

    # Initialize benchmark
    benchmark = PerformanceBenchmark()

    # Define test sizes
    matrix_sizes = [512, 1024, 2048]
    array_sizes = [100000, 1000000, 10000000]
    data_sizes = [10000, 100000, 1000000]

    # Run benchmarks
    print("\n" + "-" * 60)
    print("MATRIX OPERATIONS")
    print("-" * 60)
    matrix_operations_benchmark(benchmark, matrix_sizes)

    print("\n" + "-" * 60)
    print("ARRAY OPERATIONS")
    print("-" * 60)
    array_operations_benchmark(benchmark, array_sizes)

    print("\n" + "-" * 60)
    print("DATA PROCESSING")
    print("-" * 60)
    data_processing_benchmark(benchmark, data_sizes)

    # Summary report
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)

    gpu_results = [r for r in benchmark.results if r['gpu_time'] is not None]
    if gpu_results:
        avg_speedup = np.mean([r['speedup'] for r in gpu_results if r['speedup'] != float('inf')])
        max_speedup = np.mean([r['speedup'] for r in gpu_results if r['speedup'] != float('inf')])

        print(".2f")
        print(".2f")

        print("\nTop GPU Speedups:")
        sorted_results = sorted(gpu_results, key=lambda x: x['speedup'] or 0, reverse=True)
        for result in sorted_results[:5]:
            if result['speedup'] and result['speedup'] != float('inf'):
                print("<30")

    print("\nNote: GPU performance includes data transfer overhead.")
    print("Real-world applications may see higher speedups with larger datasets")
    print("and optimized memory management.")

if __name__ == "__main__":
    main()