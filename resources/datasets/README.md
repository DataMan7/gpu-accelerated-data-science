# Datasets Directory

This directory is designated for sample datasets, data samples, and benchmarking data used in GPU-accelerated data science examples and tutorials.

## Current Status: Placeholder for Future Content

This directory is currently empty but structured for future expansion. The project foundation is complete, and this directory will be populated with appropriate datasets in subsequent phases.

## Data Usage Guidelines

### Legal and Ethical Considerations
- **Public Domain Data**: Only include datasets that are legally shareable
- **Anonymized Data**: Ensure all personal/sensitive data is properly anonymized
- **Licensing**: Include clear licensing information for all datasets
- **Attribution**: Provide proper attribution to data sources

### Dataset Size Considerations
- **Reasonable Sizes**: Keep datasets small enough for easy download and testing
- **Compression**: Use efficient compression formats (Parquet, Zstandard)
- **Synthetic Data**: Generate synthetic datasets for examples when real data isn't available
- **Sampling**: Provide representative samples of larger datasets

## Planned Dataset Categories

### Benchmarking Datasets
- **Performance Testing**: Standard datasets for comparing GPU vs CPU performance
- **Scalability Testing**: Datasets of varying sizes for scalability analysis
- **Memory Testing**: Datasets designed to test memory limits and optimization

### Educational Examples
- **Simple Examples**: Small, clean datasets for learning basic concepts
- **Complex Examples**: More realistic datasets for advanced techniques
- **Domain-Specific**: Datasets relevant to different industries and use cases

### Research and Analysis
- **Real-World Data**: Anonymized samples from actual applications
- **Time Series**: Financial data, IoT sensor data, etc.
- **Graph Data**: Network data for graph analytics examples
- **Spatial Data**: Geographic data for spatial analytics

## Dataset Format Standards

### Preferred Formats
- **Parquet**: Columnar format, excellent for analytics
- **CSV**: Universal format, good for small datasets
- **HDF5**: Hierarchical format for complex data structures
- **NPZ**: NumPy compressed format for numerical data

### Metadata Requirements
Each dataset should include:
- **README.md**: Description, source, licensing, usage instructions
- **Data Dictionary**: Field descriptions and data types
- **Size Information**: File sizes and row counts
- **Generation Scripts**: Scripts to recreate synthetic datasets

## External Dataset Resources

While this directory is being populated, here are excellent sources for GPU-accelerated data science datasets:

### Public Benchmark Datasets
- **RAPIDS Datasets**: https://docs.rapids.ai/datasets/
- **UCI Machine Learning Repository**: https://archive.ics.uci.edu/
- **Kaggle Datasets**: https://www.kaggle.com/datasets
- **OpenML**: https://www.openml.org/

### GPU-Specific Datasets
- **cuDF Sample Data**: https://github.com/rapidsai/cudf/tree/main/python/cudf/cudf/tests/data
- **cuML Test Data**: https://github.com/rapidsai/cuml/tree/main/python/cuml/tests
- **NVIDIA NGC Datasets**: https://catalog.ngc.nvidia.com/datasets

### Cloud-Hosted Datasets
- **AWS Open Data**: https://registry.opendata.aws/
- **Google Cloud Public Datasets**: https://cloud.google.com/public-datasets
- **Azure Open Datasets**: https://azure.microsoft.com/en-us/services/open-datasets/

## Future Development

This directory will be expanded with:

### Phase 3: Core Datasets
- Basic benchmarking datasets
- Simple educational examples
- Synthetic data generation scripts

### Phase 4: Advanced Datasets
- Real-world application datasets
- Multi-modal data examples
- Large-scale data samples

### Phase 5: Specialized Collections
- Industry-specific datasets
- Research datasets
- Performance benchmarking suites

## Data Generation Scripts

For synthetic datasets, we'll provide generation scripts that create:
- **Configurable sizes**: Generate datasets of different scales
- **Realistic distributions**: Statistically representative data
- **Known characteristics**: Datasets with known properties for testing
- **Reproducible results**: Deterministic generation for consistent testing

## Contributing Datasets

When contributing datasets to this repository:

1. **Quality Check**: Ensure data quality and relevance
2. **Documentation**: Provide comprehensive metadata and usage instructions
3. **Size Limits**: Keep individual files under reasonable size limits
4. **Licensing**: Include appropriate licenses and attribution
5. **Testing**: Verify datasets work with the provided code examples

For now, the code examples in this project use synthetic data generation or reference external datasets to ensure the repository remains lightweight and focused on the core educational content.