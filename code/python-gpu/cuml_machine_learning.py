#!/usr/bin/env python3
"""
GPU-Accelerated Machine Learning with cuML
Demonstrates zero-code-change acceleration of scikit-learn algorithms
"""

import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Enable cuML acceleration
try:
    import cuml
    cuml.set_global_output_type('numpy')  # Return numpy arrays for compatibility
    CUML_AVAILABLE = True
    print("cuML acceleration enabled!")
except ImportError:
    CUML_AVAILABLE = False
    print("cuML not available, using CPU-only scikit-learn")

def benchmark_algorithm(name, cpu_func, gpu_func, X_train, X_test, y_train, y_test, **kwargs):
    """Benchmark CPU vs GPU performance for ML algorithms"""

    print(f"\n=== {name} Benchmark ===")

    # CPU benchmark
    start_time = time.time()
    cpu_model = cpu_func(**kwargs)
    cpu_model.fit(X_train, y_train)
    cpu_pred = cpu_model.predict(X_test)
    cpu_time = time.time() - start_time

    # GPU benchmark (if available)
    if gpu_func and CUML_AVAILABLE:
        start_time = time.time()
        gpu_model = gpu_func(**kwargs)
        gpu_model.fit(X_train, y_train)
        gpu_pred = gpu_model.predict(X_test)
        gpu_time = time.time() - start_time

        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')

        # Verify results are similar
        if hasattr(cpu_pred, 'shape') and hasattr(gpu_pred, 'shape'):
            if len(cpu_pred) == len(gpu_pred):
                max_diff = np.max(np.abs(cpu_pred - gpu_pred))
                print(".4f")
                print(".4f")
                print(".2f")
                print(".2e")
            else:
                print(".4f")
                print("  GPU: N/A (cuML not available)")
        else:
            print(".4f")
            print(".2f")
    else:
        print(".4f")
        print("  GPU: N/A (cuML not available)")

    return cpu_model, gpu_model if 'gpu_model' in locals() else None

def classification_benchmarks():
    """Benchmark classification algorithms"""
    print("\n" + "="*60)
    print("CLASSIFICATION ALGORITHMS BENCHMARK")
    print("="*60)

    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=100000,
        n_features=20,
        n_informative=10,
        n_redundant=10,
        n_classes=2,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Import algorithms
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    if CUML_AVAILABLE:
        from cuml.ensemble import RandomForestClassifier as cuRandomForest
        from cuml.linear_model import LogisticRegression as cuLogisticRegression
        from cuml.svm import SVC as cuSVC

    # Benchmark algorithms
    algorithms = [
        ("Random Forest", RandomForestClassifier, cuRandomForest if CUML_AVAILABLE else None,
         {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}),
        ("Logistic Regression", LogisticRegression, cuLogisticRegression if CUML_AVAILABLE else None,
         {'max_iter': 1000, 'random_state': 42}),
        ("SVM (RBF)", SVC, cuSVC if CUML_AVAILABLE else None,
         {'kernel': 'rbf', 'C': 1.0, 'random_state': 42}),
    ]

    results = {}
    for name, cpu_algo, gpu_algo, params in algorithms:
        cpu_model, gpu_model = benchmark_algorithm(
            name, cpu_algo, gpu_algo, X_train, X_test, y_train, y_test, **params
        )
        results[name] = (cpu_model, gpu_model)

    return results

def regression_benchmarks():
    """Benchmark regression algorithms"""
    print("\n" + "="*60)
    print("REGRESSION ALGORITHMS BENCHMARK")
    print("="*60)

    # Generate synthetic regression data
    X, y = make_regression(
        n_samples=50000,
        n_features=10,
        noise=0.1,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Import algorithms
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge

    if CUML_AVAILABLE:
        from cuml.ensemble import RandomForestRegressor as cuRandomForestRegressor
        from cuml.linear_model import LinearRegression as cuLinearRegression
        from cuml.linear_model import Ridge as cuRidge

    # Benchmark algorithms
    algorithms = [
        ("Random Forest Regressor", RandomForestRegressor, cuRandomForestRegressor if CUML_AVAILABLE else None,
         {'n_estimators': 50, 'max_depth': 8, 'random_state': 42}),
        ("Linear Regression", LinearRegression, cuLinearRegression if CUML_AVAILABLE else None, {}),
        ("Ridge Regression", Ridge, cuRidge if CUML_AVAILABLE else None,
         {'alpha': 0.1}),
    ]

    results = {}
    for name, cpu_algo, gpu_algo, params in algorithms:
        cpu_model, gpu_model = benchmark_algorithm(
            name, cpu_algo, gpu_algo, X_train, X_test, y_train, y_test, **params
        )
        results[name] = (cpu_model, gpu_model)

    return results

def clustering_benchmarks():
    """Benchmark clustering algorithms"""
    print("\n" + "="*60)
    print("CLUSTERING ALGORITHMS BENCHMARK")
    print("="*60)

    # Generate synthetic clustering data
    np.random.seed(42)
    X = np.random.rand(20000, 10)

    # Import algorithms
    from sklearn.cluster import KMeans, DBSCAN

    if CUML_AVAILABLE:
        from cuml.cluster import KMeans as cuKMeans
        from cuml.cluster import DBSCAN as cuDBSCAN

    # Benchmark algorithms
    algorithms = [
        ("K-Means", KMeans, cuKMeans if CUML_AVAILABLE else None,
         {'n_clusters': 5, 'random_state': 42, 'n_init': 10}),
        ("DBSCAN", DBSCAN, cuDBSCAN if CUML_AVAILABLE else None,
         {'eps': 0.5, 'min_samples': 5}),
    ]

    results = {}
    for name, cpu_algo, gpu_algo, params in algorithms:
        print(f"\n=== {name} Benchmark ===")

        # CPU benchmark
        start_time = time.time()
        cpu_model = cpu_algo(**params)
        cpu_labels = cpu_model.fit_predict(X)
        cpu_time = time.time() - start_time

        print(".4f")

        # GPU benchmark
        if gpu_algo and CUML_AVAILABLE:
            start_time = time.time()
            gpu_model = gpu_algo(**params)
            gpu_labels = gpu_model.fit_predict(X)
            gpu_time = time.time() - start_time

            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            print(".4f")
            print(".2f")
        else:
            print("  GPU: N/A (cuML not available)")

        results[name] = (cpu_model, gpu_model if 'gpu_model' in locals() else None)

    return results

def dimensionality_reduction_benchmarks():
    """Benchmark dimensionality reduction algorithms"""
    print("\n" + "="*60)
    print("DIMENSIONALITY REDUCTION BENCHMARK")
    print("="*60)

    # Generate high-dimensional data
    np.random.seed(42)
    X = np.random.rand(10000, 50)

    # Import algorithms
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    if CUML_AVAILABLE:
        from cuml.decomposition import PCA as cuPCA
        from cuml.manifold import TSNE as cuTSNE

    # Benchmark algorithms
    algorithms = [
        ("PCA", PCA, cuPCA if CUML_AVAILABLE else None,
         {'n_components': 10, 'random_state': 42}),
        ("t-SNE", TSNE, cuTSNE if CUML_AVAILABLE else None,
         {'n_components': 2, 'random_state': 42, 'perplexity': 30.0}),
    ]

    results = {}
    for name, cpu_algo, gpu_algo, params in algorithms:
        print(f"\n=== {name} Benchmark ===")

        # CPU benchmark
        start_time = time.time()
        cpu_model = cpu_algo(**params)
        cpu_result = cpu_model.fit_transform(X)
        cpu_time = time.time() - start_time

        print(".4f")

        # GPU benchmark
        if gpu_algo and CUML_AVAILABLE:
            start_time = time.time()
            gpu_model = gpu_algo(**params)
            gpu_result = gpu_model.fit_transform(X)
            gpu_time = time.time() - start_time

            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            max_diff = np.max(np.abs(cpu_result - gpu_result))
            print(".4f")
            print(".2f")
            print(".2e")
        else:
            print("  GPU: N/A (cuML not available)")

        results[name] = (cpu_model, gpu_model if 'gpu_model' in locals() else None)

    return results

def main():
    """Main benchmarking function"""
    print("GPU-Accelerated Machine Learning with cuML")
    print("==========================================")

    if CUML_AVAILABLE:
        print(f"cuML version: {cuml.__version__}")
        print("GPU acceleration: ENABLED")
    else:
        print("GPU acceleration: DISABLED (cuML not available)")
        print("Install cuML for GPU acceleration: conda install -c rapidsai cuml")

    # Run all benchmarks
    classification_results = classification_benchmarks()
    regression_results = regression_benchmarks()
    clustering_results = clustering_benchmarks()
    dim_reduction_results = dimensionality_reduction_benchmarks()

    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print("cuML provides significant speedup for machine learning algorithms")
    print("while maintaining API compatibility with scikit-learn.")
    print("\nKey benefits:")
    print("- Zero-code-change acceleration")
    print("- Drop-in replacement for scikit-learn")
    print("- Multi-GPU support")
    print("- Memory efficiency")
    print("\nFor best performance:")
    print("- Use appropriate algorithms for your data size")
    print("- Consider data preprocessing on GPU with cuDF")
    print("- Leverage multi-GPU setups for large datasets")

if __name__ == "__main__":
    main()